"""Neo4j graph store for the procedural memory tier.

Manages the procedure graph schema: ProcTool, ProcAction, and ProcTrajectory
nodes with FOLLOWED_BY, TOOL_TRANSITION, USES_TOOL, and IN_TRAJECTORY edges.

Uses MemMachine's existing Neo4j AsyncDriver — no separate database connection.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

# Edge weight formula constants
EPSILON = 0.01

# Cypher queries as module constants for clarity and reuse
_SCHEMA_CONSTRAINTS = [
    "CREATE CONSTRAINT proc_tool_name_unique IF NOT EXISTS FOR (t:ProcTool) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT proc_action_id_unique IF NOT EXISTS FOR (a:ProcAction) REQUIRE a.action_id IS UNIQUE",
    "CREATE CONSTRAINT proc_trajectory_id_unique IF NOT EXISTS FOR (tr:ProcTrajectory) REQUIRE tr.trajectory_id IS UNIQUE",
]

_SCHEMA_INDEXES = [
    "CREATE INDEX proc_action_tool_name IF NOT EXISTS FOR (a:ProcAction) ON (a.tool_name)",
    "CREATE INDEX proc_action_trajectory IF NOT EXISTS FOR (a:ProcAction) ON (a.trajectory_id)",
    "CREATE INDEX proc_trajectory_success IF NOT EXISTS FOR (tr:ProcTrajectory) ON (tr.success)",
]


class ProceduralGraphStore:
    """Manages the procedural memory graph in Neo4j.

    All node labels are prefixed with 'Proc' (e.g., ProcTool, ProcAction) to
    avoid collisions with MemMachine's existing episodic graph nodes.
    """

    def __init__(self, driver: AsyncDriver, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    async def ensure_schema(self) -> None:
        """Create constraints and indexes if they don't exist.

        Idempotent — safe to call on every startup.
        """
        async with self._driver.session(database=self._database) as session:
            for stmt in _SCHEMA_CONSTRAINTS + _SCHEMA_INDEXES:
                try:
                    await session.run(stmt)
                except Exception as e:
                    # Constraints/indexes may already exist; log and continue
                    logger.debug("Schema statement skipped (may already exist): %s — %s", stmt[:60], e)
        logger.info("Procedural graph schema ensured (%d constraints, %d indexes)",
                     len(_SCHEMA_CONSTRAINTS), len(_SCHEMA_INDEXES))

    # ── Ingestion ────────────────────────────────────────────────────────

    async def ingest_trajectory(
        self,
        trajectory_id: str,
        task_description: str,
        success: bool,
        actions: list[dict[str, Any]],
        source: str = "unknown",
    ) -> int:
        """Ingest one trajectory into the procedure graph.

        Each action in the trajectory becomes a :ProcAction node. Consecutive
        actions are linked by :FOLLOWED_BY edges at the instance level, and
        :TOOL_TRANSITION edges are upserted at the aggregate Tool level with
        success/failure counts incremented.

        Args:
            trajectory_id: Unique ID for this trajectory.
            task_description: Natural-language task goal.
            success: Whether this trajectory completed the task.
            actions: List of dicts with keys: tool_name, parameters (dict),
                     step_id (int), state_description (str).
            source: Origin dataset identifier.

        Returns:
            Number of actions ingested.
        """
        if not actions:
            return 0

        async with self._driver.session(database=self._database) as session:
            # 1. Create/merge the Trajectory node
            await session.run(
                """
                MERGE (tr:ProcTrajectory {trajectory_id: $tid})
                ON CREATE SET
                    tr.task_description = $desc,
                    tr.success = $success,
                    tr.total_steps = $total_steps,
                    tr.source = $source,
                    tr.ingested_at = datetime()
                """,
                tid=trajectory_id,
                desc=task_description,
                success=success,
                total_steps=len(actions),
                source=source,
            )

            prev_action_id = None
            prev_tool_name = None

            for action in actions:
                tool_name = action["tool_name"]
                step_id = action.get("step_id", 0)
                action_id = f"{trajectory_id}_{step_id}"
                params = action.get("parameters", {})
                state_desc = action.get("state_description", "")

                # 2. MERGE the Tool node
                await session.run(
                    """
                    MERGE (t:ProcTool {name: $name})
                    ON CREATE SET t.total_invocations = 1
                    ON MATCH SET t.total_invocations = t.total_invocations + 1
                    """,
                    name=tool_name,
                )

                # 3. CREATE the Action node (unique per trajectory+step)
                await session.run(
                    """
                    MERGE (a:ProcAction {action_id: $aid})
                    ON CREATE SET
                        a.tool_name = $tool_name,
                        a.parameters = $params,
                        a.step_id = $step_id,
                        a.trajectory_id = $tid,
                        a.success = $success,
                        a.state_description = $state_desc
                    """,
                    aid=action_id,
                    tool_name=tool_name,
                    params=str(params),  # Neo4j stores as string
                    step_id=step_id,
                    tid=trajectory_id,
                    success=success,
                    state_desc=state_desc,
                )

                # 4. Link Action → Tool
                await session.run(
                    """
                    MATCH (a:ProcAction {action_id: $aid})
                    MATCH (t:ProcTool {name: $tool_name})
                    MERGE (a)-[:USES_TOOL]->(t)
                    """,
                    aid=action_id,
                    tool_name=tool_name,
                )

                # 5. Link Action → Trajectory
                await session.run(
                    """
                    MATCH (a:ProcAction {action_id: $aid})
                    MATCH (tr:ProcTrajectory {trajectory_id: $tid})
                    MERGE (a)-[:IN_TRAJECTORY {step_order: $step_id}]->(tr)
                    """,
                    aid=action_id,
                    tid=trajectory_id,
                    step_id=step_id,
                )

                # 6. Instance-level FOLLOWED_BY (Action → Action within trajectory)
                if prev_action_id is not None:
                    await session.run(
                        """
                        MATCH (prev:ProcAction {action_id: $prev_id})
                        MATCH (curr:ProcAction {action_id: $curr_id})
                        MERGE (prev)-[:FOLLOWED_BY {trajectory_id: $tid}]->(curr)
                        """,
                        prev_id=prev_action_id,
                        curr_id=action_id,
                        tid=trajectory_id,
                    )

                # 7. Aggregate TOOL_TRANSITION (Tool → Tool across trajectories)
                if prev_tool_name is not None:
                    if success:
                        await session.run(
                            """
                            MATCH (from_t:ProcTool {name: $from_name})
                            MATCH (to_t:ProcTool {name: $to_name})
                            MERGE (from_t)-[r:TOOL_TRANSITION]->(to_t)
                            ON CREATE SET
                                r.success_count = 1,
                                r.fail_count = 0,
                                r.weight = 1.0 / (1.0 + $epsilon),
                                r.cost = 1.0 - (1.0 / (1.0 + $epsilon))
                            ON MATCH SET
                                r.success_count = r.success_count + 1,
                                r.weight = (r.success_count) /
                                    (r.success_count + r.fail_count + $epsilon),
                                r.cost = 1.0 - r.weight
                            """,
                            from_name=prev_tool_name,
                            to_name=tool_name,
                            epsilon=EPSILON,
                        )
                    else:
                        await session.run(
                            """
                            MATCH (from_t:ProcTool {name: $from_name})
                            MATCH (to_t:ProcTool {name: $to_name})
                            MERGE (from_t)-[r:TOOL_TRANSITION]->(to_t)
                            ON CREATE SET
                                r.success_count = 0,
                                r.fail_count = 1,
                                r.weight = 0.0 / (0.0 + 1.0 + $epsilon),
                                r.cost = 1.0 - (0.0 / (0.0 + 1.0 + $epsilon))
                            ON MATCH SET
                                r.fail_count = r.fail_count + 1,
                                r.weight = r.success_count /
                                    (r.success_count + r.fail_count + $epsilon),
                                r.cost = 1.0 - r.weight
                            """,
                            from_name=prev_tool_name,
                            to_name=tool_name,
                            epsilon=EPSILON,
                        )

                prev_action_id = action_id
                prev_tool_name = tool_name

        logger.info(
            "Ingested trajectory %s (%d actions, success=%s)",
            trajectory_id, len(actions), success,
        )
        return len(actions)

    # ── Query helpers ────────────────────────────────────────────────────

    async def get_tool_transition_graph(self) -> list[dict[str, Any]]:
        """Return all TOOL_TRANSITION edges with weights.

        Returns a list of dicts: {from_tool, to_tool, weight, cost,
        success_count, fail_count}.
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (a:ProcTool)-[r:TOOL_TRANSITION]->(b:ProcTool)
                RETURN a.name AS from_tool, b.name AS to_tool,
                       r.weight AS weight, r.cost AS cost,
                       r.success_count AS success_count,
                       r.fail_count AS fail_count
                ORDER BY r.weight DESC
                """
            )
            return [dict(record) async for record in result]

    async def get_tool_names(self) -> list[str]:
        """Return all tool names in the procedure graph."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (t:ProcTool) RETURN t.name AS name ORDER BY name"
            )
            return [record["name"] async for record in result]

    async def get_trajectory_count(self) -> dict[str, int]:
        """Return counts of success/failure trajectories."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (tr:ProcTrajectory)
                RETURN tr.success AS success, count(tr) AS cnt
                """
            )
            counts = {"success": 0, "failure": 0, "total": 0}
            async for record in result:
                key = "success" if record["success"] else "failure"
                counts[key] = record["cnt"]
                counts["total"] += record["cnt"]
            return counts

    async def get_shortest_path_cost(
        self, from_tool: str, to_tool: str
    ) -> float | None:
        """Find shortest path cost between two tools using TOOL_TRANSITION edges.

        Returns the total cost (sum of edge costs along the path), or None if
        no path exists. Uses Neo4j's shortestPath with cost property.
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (a:ProcTool {name: $from_tool}),
                      (b:ProcTool {name: $to_tool}),
                      path = shortestPath((a)-[:TOOL_TRANSITION*]->(b))
                WITH path, reduce(cost = 0.0, r IN relationships(path) | cost + r.cost) AS total_cost
                RETURN total_cost
                LIMIT 1
                """,
                from_tool=from_tool,
                to_tool=to_tool,
            )
            record = await result.single()
            return record["total_cost"] if record else None

    async def get_actions_for_trajectory(
        self, trajectory_id: str
    ) -> list[dict[str, Any]]:
        """Return all actions for a trajectory, ordered by step."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (a:ProcAction {trajectory_id: $tid})
                RETURN a.action_id AS action_id,
                       a.tool_name AS tool_name,
                       a.step_id AS step_id,
                       a.parameters AS parameters,
                       a.state_description AS state_description,
                       a.success AS success
                ORDER BY a.step_id
                """,
                tid=trajectory_id,
            )
            return [dict(record) async for record in result]

    # ── Cleanup ──────────────────────────────────────────────────────────

    async def delete_all_procedural_data(self) -> dict[str, int]:
        """Remove all procedural memory nodes and relationships.

        Returns counts of deleted items by type.
        """
        counts: dict[str, int] = {}
        async with self._driver.session(database=self._database) as session:
            # Delete in dependency order
            for label in [
                "ProcAction", "ProcTool", "ProcTrajectory"
            ]:
                result = await session.run(
                    f"MATCH (n:{label}) DETACH DELETE n RETURN count(n) AS cnt"
                )
                record = await result.single()
                counts[label] = record["cnt"] if record else 0

        total = sum(counts.values())
        logger.info("Deleted %d procedural memory nodes: %s", total, counts)
        return counts
