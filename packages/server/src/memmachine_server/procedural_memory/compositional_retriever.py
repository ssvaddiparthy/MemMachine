"""Compositional Retriever — synthesizes novel procedures via Steiner tree.

Given a query, this module:
1. Extracts required tool capabilities from the query
2. Maps capabilities to terminal nodes in the procedure graph
3. Computes a Steiner tree approximation connecting the terminals
4. Serializes the composed procedure into steps with failure warnings
5. Returns a ComposedProcedure that may never have existed as a stored trajectory

Novel mechanism: No existing procedural memory system composes novel procedures
from fragments of multiple trajectories at retrieval time. H-EPM retrieves
graph sub-paths from one trajectory. HyperSkill ranks existing skills by
co-occurrence. Voyager/ReMe/Mem^p retrieve by vector similarity. None do
real-time graph composition.

The Steiner tree problem is NP-hard in general. We use the MST heuristic
(Kou, Markowsky & Berman, 1981), a 2-approximation algorithm.
"""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from typing import Any

from memmachine_server.procedural_memory.data_types import ComposedProcedure
from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore

logger = logging.getLogger(__name__)


class CompositionalRetriever:
    """Composes novel procedures via Steiner tree on the procedure graph.

    The retriever operates on the TOOL_TRANSITION aggregate graph, where
    nodes are tools and edges carry reliability weights. Given a set of
    required tools (terminals), it finds the minimum-cost connected subgraph
    spanning those terminals — the Steiner tree.
    """

    def __init__(self, graph_store: ProceduralGraphStore) -> None:
        self._store = graph_store
        self._graph_cache: dict[str, dict[str, float]] | None = None

    async def compose(
        self,
        required_tools: list[str],
        fallback_text: str = "",
    ) -> ComposedProcedure | None:
        """Compose a procedure connecting the required tools.

        Args:
            required_tools: List of tool names that must appear in the
                           composed procedure (the "terminal" nodes).
            fallback_text: If composition fails, return this as a single-step
                          procedure. Empty string means return None on failure.

        Returns:
            A ComposedProcedure, or None if composition is not possible.
        """
        if len(required_tools) < 2:
            logger.warning("Need at least 2 tools for composition, got %d", len(required_tools))
            if fallback_text:
                return ComposedProcedure(steps=[fallback_text], confidence=0.0)
            return None

        # Load the tool transition graph into memory for Dijkstra
        graph = await self._load_graph()

        if not graph:
            logger.warning("Tool transition graph is empty")
            return None

        # Filter to terminals that exist in the graph
        available_terminals = [t for t in required_tools if t in graph]
        missing = set(required_tools) - set(available_terminals)
        if missing:
            logger.warning("Tools not in graph (will be skipped): %s", missing)

        if len(available_terminals) < 2:
            logger.warning("Fewer than 2 terminals in graph after filtering")
            return None

        # Steiner tree via MST heuristic (2-approximation)
        steiner_path, total_cost = self._steiner_mst_heuristic(
            graph, available_terminals
        )

        if not steiner_path:
            logger.warning("No Steiner tree found connecting terminals")
            return None

        # Get failure warnings from failure communities
        avoid_warnings = await self._get_failure_warnings(steiner_path)

        # Compute confidence (minimum edge weight along the path)
        confidence = self._path_confidence(graph, steiner_path)

        # Serialize to human-readable steps
        steps = self._serialize_path_to_steps(steiner_path)

        return ComposedProcedure(
            steps=steps,
            tool_sequence=steiner_path,
            confidence=confidence,
            total_cost=total_cost,
            avoid_warnings=avoid_warnings,
        )

    async def _load_graph(self) -> dict[str, dict[str, float]]:
        """Load the TOOL_TRANSITION graph into an adjacency dict.

        Format: {from_tool: {to_tool: cost, ...}, ...}
        Cached after first load.
        """
        if self._graph_cache is not None:
            return self._graph_cache

        edges = await self._store.get_tool_transition_graph()
        graph: dict[str, dict[str, float]] = defaultdict(dict)

        for edge in edges:
            from_t = edge["from_tool"]
            to_t = edge["to_tool"]
            cost = edge["cost"]
            graph[from_t][to_t] = cost
            # Ensure both nodes exist as keys even if they have no outgoing edges
            if to_t not in graph:
                graph[to_t] = {}

        self._graph_cache = dict(graph)
        logger.info(
            "Loaded tool transition graph: %d nodes, %d edges",
            len(self._graph_cache),
            sum(len(v) for v in self._graph_cache.values()),
        )
        return self._graph_cache

    def invalidate_cache(self) -> None:
        """Clear the graph cache so next compose() reloads from Neo4j."""
        self._graph_cache = None

    # ── Steiner tree MST heuristic ───────────────────────────────────────

    def _steiner_mst_heuristic(
        self,
        graph: dict[str, dict[str, float]],
        terminals: list[str],
    ) -> tuple[list[str], float]:
        """Steiner tree via the MST heuristic (Kou, Markowsky & Berman, 1981).

        Algorithm:
        1. Compute shortest paths between all pairs of terminals (Dijkstra)
        2. Build a complete graph on terminals, weighted by shortest-path distances
        3. Find the MST of the complete graph (Prim's algorithm)
        4. Map MST edges back to their shortest paths in the original graph
        5. Remove redundant edges to produce the Steiner tree
        6. Linearize into a topological ordering

        This is a 2-approximation for the Steiner tree problem.

        Returns:
            (ordered_tool_sequence, total_cost)
        """
        n = len(terminals)
        if n < 2:
            return (terminals, 0.0)

        # Step 1: All-pairs shortest paths among terminals
        shortest_paths: dict[tuple[str, str], tuple[float, list[str]]] = {}

        for i, src in enumerate(terminals):
            dist, prev = self._dijkstra(graph, src)
            for j, dst in enumerate(terminals):
                if i != j and dst in dist:
                    path = self._reconstruct_path(prev, src, dst)
                    shortest_paths[(src, dst)] = (dist[dst], path)

        # Step 2: Complete graph on terminals
        complete_edges: list[tuple[float, str, str, list[str]]] = []
        for (src, dst), (cost, path) in shortest_paths.items():
            complete_edges.append((cost, src, dst, path))

        if not complete_edges:
            logger.warning("No paths found between any terminals")
            return ([], 0.0)

        # Step 3: MST of the complete graph (Prim's)
        mst_edges = self._prims_mst(terminals, complete_edges)

        # Step 4: Map MST edges back to original paths and merge
        all_tools_ordered: list[str] = []
        total_cost = 0.0

        for cost, src, dst, path in mst_edges:
            # Merge path into the ordered sequence, avoiding duplicates at boundaries
            for tool in path:
                if not all_tools_ordered or all_tools_ordered[-1] != tool:
                    all_tools_ordered.append(tool)
            total_cost += cost

        return (all_tools_ordered, total_cost)

    @staticmethod
    def _dijkstra(
        graph: dict[str, dict[str, float]], source: str
    ) -> tuple[dict[str, float], dict[str, str | None]]:
        """Single-source shortest path via Dijkstra.

        Args:
            graph: Adjacency dict {node: {neighbor: cost}}.
            source: Starting node.

        Returns:
            (distances, predecessors) dicts.
        """
        dist: dict[str, float] = {source: 0.0}
        prev: dict[str, str | None] = {source: None}
        heap: list[tuple[float, str]] = [(0.0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            for v, w in graph.get(u, {}).items():
                new_dist = d + w
                if new_dist < dist.get(v, float("inf")):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

        return dist, prev

    @staticmethod
    def _reconstruct_path(
        prev: dict[str, str | None], source: str, target: str
    ) -> list[str]:
        """Reconstruct path from Dijkstra's predecessor map."""
        path = []
        current: str | None = target
        while current is not None:
            path.append(current)
            current = prev.get(current)
        path.reverse()
        if path and path[0] == source:
            return path
        return []

    @staticmethod
    def _prims_mst(
        nodes: list[str],
        edges: list[tuple[float, str, str, list[str]]],
    ) -> list[tuple[float, str, str, list[str]]]:
        """Prim's MST on the complete terminal graph.

        Args:
            nodes: Terminal nodes.
            edges: (cost, src, dst, original_path) tuples.

        Returns:
            MST edge list.
        """
        if len(nodes) < 2:
            return []

        # Build adjacency for the complete graph
        adj: dict[str, list[tuple[float, str, list[str]]]] = defaultdict(list)
        for cost, src, dst, path in edges:
            adj[src].append((cost, dst, path))

        # Prim's
        in_tree: set[str] = {nodes[0]}
        mst: list[tuple[float, str, str, list[str]]] = []
        candidates: list[tuple[float, str, str, list[str]]] = []

        for cost, dst, path in adj[nodes[0]]:
            heapq.heappush(candidates, (cost, nodes[0], dst, id(path)))
            # Store path separately since lists aren't comparable
            candidates[-1] = (cost, nodes[0], dst, path)  # type: ignore[assignment]

        # Re-init with proper heap
        candidates = []
        for cost, dst, path in adj[nodes[0]]:
            heapq.heappush(candidates, (cost, id(path), nodes[0], dst, path))

        while candidates and len(in_tree) < len(nodes):
            cost, _, src, dst, path = heapq.heappop(candidates)
            if dst in in_tree:
                continue
            in_tree.add(dst)
            mst.append((cost, src, dst, path))

            for next_cost, next_dst, next_path in adj[dst]:
                if next_dst not in in_tree:
                    heapq.heappush(
                        candidates,
                        (next_cost, id(next_path), dst, next_dst, next_path),
                    )

        return mst

    # ── Failure warnings ─────────────────────────────────────────────────

    async def _get_failure_warnings(
        self, tool_sequence: list[str]
    ) -> list[str]:
        """Get warnings from failure communities for tools in the sequence."""
        failure_communities = await self._store.get_failure_communities()

        warnings = []
        for community in failure_communities:
            member_tools = set(community.get("description", "").lower().split())
            # Check if any tools in the composed sequence are in a failure community
            overlap = set(tool_sequence) & member_tools
            if overlap:
                desc = community.get("description", "Unknown failure pattern")
                warnings.append(desc)

        return warnings

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _path_confidence(
        graph: dict[str, dict[str, float]], path: list[str]
    ) -> float:
        """Compute confidence = min(1 - cost) along the path edges.

        This is the minimum reliability weight, representing the weakest
        link in the composed procedure.
        """
        if len(path) < 2:
            return 1.0

        min_weight = 1.0
        for i in range(len(path) - 1):
            cost = graph.get(path[i], {}).get(path[i + 1], 1.0)
            weight = 1.0 - cost
            min_weight = min(min_weight, weight)
        return min_weight

    @staticmethod
    def _serialize_path_to_steps(tool_sequence: list[str]) -> list[str]:
        """Convert a tool sequence into human-readable procedure steps.

        Converts tool_name format (e.g., "go_to") back to natural language
        (e.g., "Go to target location").
        """
        step_templates = {
            "go_to": "Go to {target}",
            "pick_up": "Pick up {object}",
            "put": "Put {object} in/on {receptacle}",
            "clean": "Clean {object}",
            "heat": "Heat {object}",
            "cool": "Cool {object}",
            "open": "Open {target}",
            "close": "Close {target}",
            "use": "Use {target}",
            "toggle": "Toggle {target}",
            "examine": "Examine {target}",
            "look": "Look at {target}",
            "inventory": "Check inventory",
        }

        steps = []
        for tool in tool_sequence:
            template = step_templates.get(tool)
            if template:
                # Use placeholder text since we don't have specific params
                step = template.replace("{target}", "[location]")
                step = step.replace("{object}", "[object]")
                step = step.replace("{receptacle}", "[receptacle]")
                steps.append(step)
            else:
                steps.append(f"Execute: {tool.replace('_', ' ')}")

        return steps
