"""Procedural retrieval agent — composes procedures via Steiner tree.

This agent is a drop-in alongside MemMachineAgent, ChainOfQueryAgent, and
SplitQueryAgent. It is selected by ToolSelectAgent when the query asks for
a procedural "how-to" answer (e.g., "How do I clean a soapbar and put it
in the cabinet?").

Instead of searching episodic memory, it runs compositional retrieval on the
procedural graph: Steiner tree over the tool transition graph, with failure
community avoidance.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any

from memmachine_server.common.episode_store import Episode, EpisodeType
from memmachine_server.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
)

logger = logging.getLogger(__name__)


class ProceduralAgent(AgentToolBase):
    """Agent that composes procedures from the procedural memory graph.

    Uses the CompositionalRetriever to build a Steiner tree across the
    tool transition graph, returning the composed procedure as Episode
    objects for compatibility with the existing retrieval pipeline.
    """

    def __init__(self, param: AgentToolBaseParam) -> None:
        """Initialize with optional procedural retriever reference."""
        super().__init__(param)
        # The compositional retriever is injected via extra_params
        # because it requires the Neo4j ProceduralGraphStore, which
        # is initialized at a different lifecycle point than the agents.
        self._extra = param.extra_params or {}

    @property
    def agent_name(self) -> str:
        return "ProceduralAgent"

    @property
    def agent_description(self) -> str:
        return (
            "This agent composes procedural knowledge from the tool execution "
            "graph. It synthesizes novel procedures from fragments of multiple "
            "trajectories using Steiner tree composition, with failure-path "
            "avoidance. Best for 'how to' queries about multi-step tool workflows."
        )

    @property
    def accuracy_score(self) -> int:
        return 7

    @property
    def token_cost(self) -> int:
        return 1  # Graph algorithms, no LLM calls

    @property
    def time_cost(self) -> int:
        return 2  # Sub-second Dijkstra + Steiner

    async def do_query(
        self,
        policy: QueryPolicy,
        query: QueryParam,
    ) -> tuple[list[Episode], dict[str, Any]]:
        """Compose a procedure for the query via Steiner tree retrieval.

        If the procedural graph is not available or composition fails,
        falls back to returning an empty result (the caller can then
        try episodic retrieval).
        """
        _ = policy
        logger.info("CALLING %s with query: %s", self.agent_name, query.query)

        perf_metrics: dict[str, Any] = {
            "memory_search_called": 0,
            "memory_retrieval_time": 0.0,
            "agent": self.agent_name,
            "procedural_composition": True,
        }

        start = time.time()

        try:
            # Import here to avoid circular dependency at module load time
            from memmachine_server.procedural_memory.compositional_retriever import (
                CompositionalRetriever,
            )
            from memmachine_server.procedural_memory.graph_store import (
                ProceduralGraphStore,
            )

            # Get the retriever from extra_params or create one
            retriever = self._extra.get("procedural_retriever")
            if retriever is None:
                neo4j_driver = self._extra.get("neo4j_driver")
                if neo4j_driver is None:
                    logger.warning(
                        "ProceduralAgent: no neo4j_driver in extra_params, "
                        "cannot compose procedures"
                    )
                    return [], perf_metrics

                store = ProceduralGraphStore(neo4j_driver)
                retriever = CompositionalRetriever(store)

            # Extract required tools from the query
            available_tools = await retriever._store.get_tool_names()
            required_tools = self._extract_tools_from_query(
                query.query, available_tools
            )

            if len(required_tools) < 2:
                logger.info(
                    "ProceduralAgent: fewer than 2 tools extracted from query, "
                    "skipping composition (tools=%s)",
                    required_tools,
                )
                return [], perf_metrics

            # Compose via Steiner tree
            composed = await retriever.compose(required_tools)

            perf_metrics["memory_search_called"] = 1
            perf_metrics["memory_retrieval_time"] = time.time() - start

            if composed is None or not composed.steps:
                logger.info("ProceduralAgent: composition returned no result")
                return [], perf_metrics

            # Convert the composed procedure into Episode objects
            # so it's compatible with the existing retrieval pipeline
            procedure_text = composed.to_text()
            tool_seq = " → ".join(composed.tool_sequence)

            episodes = [
                Episode(
                    uid=f"procedural_composed_{hash(procedure_text) % 10**8}",
                    content=f"[PROCEDURE] {procedure_text}",
                    session_key=query.memory.session_key if query.memory else "procedural",
                    created_at=datetime.datetime.now(tz=datetime.UTC),
                    producer_id="procedural_memory",
                    producer_role="system",
                    episode_type=EpisodeType.MESSAGE,
                    metadata={
                        "type": "composed_procedure",
                        "tool_sequence": tool_seq,
                        "confidence": composed.confidence,
                        "total_cost": composed.total_cost,
                        "source_trajectories": ",".join(
                            composed.source_trajectories[:5]
                        ),
                    },
                )
            ]

            # Add avoidance warnings as separate episodes
            for warning in composed.avoid_warnings:
                episodes.append(
                    Episode(
                        uid=f"procedural_warning_{hash(warning) % 10**8}",
                        content=f"[WARNING] Avoid: {warning}",
                        session_key=query.memory.session_key if query.memory else "procedural",
                        created_at=datetime.datetime.now(tz=datetime.UTC),
                        producer_id="procedural_memory",
                        producer_role="system",
                        episode_type=EpisodeType.MESSAGE,
                        metadata={"type": "failure_warning"},
                    )
                )

            perf_metrics["composed_steps"] = len(composed.steps)
            perf_metrics["composed_tools"] = len(composed.tool_sequence)
            perf_metrics["composition_confidence"] = composed.confidence

            logger.info(
                "ProceduralAgent composed %d-step procedure (tools: %s, "
                "confidence=%.3f)",
                len(composed.steps),
                tool_seq,
                composed.confidence,
            )

            return episodes, perf_metrics

        except Exception:
            logger.exception("ProceduralAgent: composition failed")
            perf_metrics["memory_retrieval_time"] = time.time() - start
            return [], perf_metrics

    @staticmethod
    def _extract_tools_from_query(
        query_text: str, available_tools: list[str]
    ) -> list[str]:
        """Extract required tools from query text via keyword matching."""
        query_lower = query_text.lower()
        tool_keywords = {
            "go_to": ["go to", "navigate", "move to", "find", "get to"],
            "pick_up": ["pick up", "take", "grab", "get"],
            "put": ["put", "place", "set down", "in the", "on the"],
            "clean": ["clean", "wash", "rinse"],
            "heat": ["heat", "warm", "microwave"],
            "cool": ["cool", "chill", "refrigerate", "fridge"],
            "open": ["open"],
            "close": ["close", "shut"],
            "use": ["use", "turn on", "activate"],
            "examine": ["examine", "inspect", "look at"],
        }

        required = []
        for tool, keywords in tool_keywords.items():
            if tool in available_tools:
                for kw in keywords:
                    if kw in query_lower:
                        required.append(tool)
                        break

        # Always include go_to for spatial tasks
        if required and "go_to" in available_tools and "go_to" not in required:
            required.insert(0, "go_to")

        return required
