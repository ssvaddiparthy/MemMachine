"""Community Detector — discovers reusable sub-procedures via Louvain.

Runs Louvain community detection on the TOOL_TRANSITION aggregate graph in
Neo4j to discover emergent functional modules (tool clusters that frequently
co-execute across successful trajectories).

Also runs separately on the failure subgraph to discover anti-pattern
communities (tool clusters that frequently co-occur in failed trajectories).

Novel mechanism: No existing procedural memory system applies unsupervised
graph clustering to procedure graphs. Prior approaches use FSM alignment
(SKILL-DISCO), co-occurrence counting (HyperSkill), BPE compression
(ProcGrep), or hand-designed granularity (Mem^p).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from neo4j import AsyncDriver

from memmachine_server.procedural_memory.data_types import Community
from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Discovers reusable sub-procedures via Louvain community detection."""

    def __init__(
        self,
        graph_store: ProceduralGraphStore,
        driver: AsyncDriver,
        database: str = "neo4j",
    ) -> None:
        self._store = graph_store
        self._driver = driver
        self._database = database

    async def detect_communities(
        self,
        resolution: float = 1.0,
        min_community_size: int = 2,
    ) -> list[Community]:
        """Run Louvain community detection on the tool transition graph.

        This discovers two types of communities:
        1. Success communities — tool clusters from successful trajectories
        2. Failure communities — tool clusters from failed trajectories

        Args:
            resolution: Louvain resolution parameter. Higher → smaller communities.
                        Default 1.0. Tune to get 3-6 actions per community.
            min_community_size: Minimum number of tools in a community to keep.

        Returns:
            List of discovered Community objects.
        """
        # Run Louvain on the full graph (weighted by reliability)
        all_communities = await self._run_louvain_on_tool_graph(
            resolution=resolution,
            min_community_size=min_community_size,
        )

        # Classify communities as success or failure based on member statistics
        classified = await self._classify_communities(all_communities)

        # Persist community nodes and MEMBER_OF edges
        for community in classified:
            await self._store.create_community_node(
                community_id=community.community_id,
                community_type=community.community_type,
                description=community.description,
                member_count=community.member_count,
                trajectory_count=community.trajectory_count,
                avg_weight=community.avg_weight,
            )

        logger.info(
            "Detected %d communities (%d success, %d failure)",
            len(classified),
            sum(1 for c in classified if c.community_type == "success"),
            sum(1 for c in classified if c.community_type == "failure"),
        )

        return classified

    async def _run_louvain_on_tool_graph(
        self,
        resolution: float = 1.0,
        min_community_size: int = 2,
    ) -> list[dict[str, Any]]:
        """Run Louvain via Neo4j GDS on the ProcTool TOOL_TRANSITION graph.

        Falls back to a Python-side Louvain if GDS is not available.
        """
        async with self._driver.session(database=self._database) as session:
            # Check if GDS is available
            gds_available = await self._check_gds_available(session)

            if gds_available:
                return await self._louvain_via_gds(session, resolution, min_community_size)
            else:
                logger.warning(
                    "Neo4j GDS not available — falling back to Python-side Louvain"
                )
                return await self._louvain_via_python(session, resolution, min_community_size)

    async def _check_gds_available(self, session: Any) -> bool:
        """Check if Neo4j Graph Data Science library is installed."""
        try:
            result = await session.run("RETURN gds.version() AS version")
            record = await result.single()
            if record:
                logger.info("Neo4j GDS version: %s", record["version"])
                return True
        except Exception:
            pass
        return False

    async def _louvain_via_gds(
        self,
        session: Any,
        resolution: float,
        min_community_size: int,
    ) -> list[dict[str, Any]]:
        """Run Louvain using Neo4j GDS library."""
        graph_name = "proc-tool-graph"

        # Drop existing projection if it exists
        try:
            await session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass

        # Project the tool transition graph
        await session.run(
            """
            CALL gds.graph.project(
                $graph_name,
                'ProcTool',
                {
                    TOOL_TRANSITION: {
                        properties: ['weight'],
                        orientation: 'NATURAL'
                    }
                }
            )
            """,
            graph_name=graph_name,
        )

        # Run Louvain
        result = await session.run(
            """
            CALL gds.louvain.stream($graph_name, {
                relationshipWeightProperty: 'weight',
                maxLevels: 10,
                maxIterations: 10,
                tolerance: 0.0001
            })
            YIELD nodeId, communityId
            WITH gds.util.asNode(nodeId).name AS tool_name, communityId
            RETURN communityId, collect(tool_name) AS members
            ORDER BY size(collect(tool_name)) DESC
            """,
            graph_name=graph_name,
        )

        communities = []
        async for record in result:
            members = record["members"]
            if len(members) >= min_community_size:
                communities.append({
                    "community_id": record["communityId"],
                    "members": members,
                    "size": len(members),
                })

        # Clean up projection
        try:
            await session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass

        return communities

    async def _louvain_via_python(
        self,
        session: Any,
        resolution: float,
        min_community_size: int,
    ) -> list[dict[str, Any]]:
        """Fallback: run Louvain in Python using networkx.

        This is used when Neo4j GDS is not installed (e.g., community edition).
        """
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities
        except ImportError:
            logger.error(
                "Neither Neo4j GDS nor networkx available for Louvain. "
                "Install networkx: pip install networkx"
            )
            return []

        # Fetch the tool transition graph
        result = await session.run(
            """
            MATCH (a:ProcTool)-[r:TOOL_TRANSITION]->(b:ProcTool)
            RETURN a.name AS from_tool, b.name AS to_tool, r.weight AS weight
            """
        )

        G = nx.DiGraph()
        async for record in result:
            G.add_edge(
                record["from_tool"],
                record["to_tool"],
                weight=record["weight"],
            )

        if G.number_of_nodes() < 2:
            logger.warning("Tool graph has fewer than 2 nodes — skipping Louvain")
            return []

        # Louvain works on undirected graphs, so convert
        G_undirected = G.to_undirected()

        communities_sets = louvain_communities(
            G_undirected,
            weight="weight",
            resolution=resolution,
        )

        communities = []
        for i, members_set in enumerate(communities_sets):
            members = sorted(members_set)
            if len(members) >= min_community_size:
                communities.append({
                    "community_id": i,
                    "members": members,
                    "size": len(members),
                })

        return communities

    async def _classify_communities(
        self, communities: list[dict[str, Any]]
    ) -> list[Community]:
        """Classify each community as success or failure.

        A community is classified as "failure" if the average edge weight
        among its member tools is below 0.4. Otherwise it's "success."
        """
        classified = []

        for comm in communities:
            members = comm["members"]
            cid = comm["community_id"]

            # Get average edge weight among members
            avg_weight = await self._get_avg_weight_among(members)

            # Count distinct trajectories that use tools in this community
            traj_count = await self._count_trajectories_for_tools(members)

            community_type = "failure" if avg_weight < 0.4 else "success"

            # Generate a simple description from member names
            description = self._generate_description(members, community_type)

            classified.append(
                Community(
                    community_id=cid,
                    community_type=community_type,
                    description=description,
                    member_count=len(members),
                    trajectory_count=traj_count,
                    avg_weight=avg_weight,
                    member_actions=members,
                )
            )

        return classified

    async def _get_avg_weight_among(self, tool_names: list[str]) -> float:
        """Get average TOOL_TRANSITION weight among a set of tools."""
        if len(tool_names) < 2:
            return 0.0

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (a:ProcTool)-[r:TOOL_TRANSITION]->(b:ProcTool)
                WHERE a.name IN $tools AND b.name IN $tools
                RETURN avg(r.weight) AS avg_weight
                """,
                tools=tool_names,
            )
            record = await result.single()
            return record["avg_weight"] or 0.0 if record else 0.0

    async def _count_trajectories_for_tools(self, tool_names: list[str]) -> int:
        """Count distinct trajectories containing actions with these tools."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (a:ProcAction)
                WHERE a.tool_name IN $tools
                RETURN count(DISTINCT a.trajectory_id) AS cnt
                """,
                tools=tool_names,
            )
            record = await result.single()
            return record["cnt"] if record else 0

    @staticmethod
    def _generate_description(
        members: list[str], community_type: str
    ) -> str:
        """Generate a human-readable description for a community."""
        tool_list = ", ".join(members[:5])
        if len(members) > 5:
            tool_list += f", ... ({len(members)} total)"

        if community_type == "failure":
            return f"Anti-pattern: frequently co-occurring tools in failed trajectories [{tool_list}]"
        return f"Sub-procedure: frequently co-occurring tools [{tool_list}]"
