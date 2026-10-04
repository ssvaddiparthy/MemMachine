"""Procedural retrieval agent -- gap-aware trajectory anchoring.

This agent is a drop-in alongside MemMachineAgent, ChainOfQueryAgent, and
SplitQueryAgent. It is selected by ToolSelectAgent when the query asks for a
procedural "how-to" answer (e.g., "How do I clean a soapbar and put it in the
cabinet?").

Instead of searching episodic memory, it runs the GroundedRetriever on the
procedural graph: decompose the query into ordered sub-goals, anchor them to
the best stored trajectory's REAL steps, and attach an explicit edit-script
diagnosis of which required sub-goals the trajectory COVERS and which are
MISSING. The diagnosis -- not the raw trajectory -- is the active ingredient
(see evaluation/procedural_memory/DETAILED_REPORT.md): it keeps the agent from
blindly imitating an incomplete example.
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
    """Agent that returns gap-aware grounded procedures from the procedure graph.

    Uses GroundedRetriever to anchor the query's ordered sub-goals to real
    trajectory steps and emit a COVERED/REBIND/GAP edit script, returned as
    Episode objects for compatibility with the existing retrieval pipeline.
    """

    def __init__(self, param: AgentToolBaseParam) -> None:
        super().__init__(param)
        # The retriever and LLM are injected via extra_params because they
        # require the Neo4j ProceduralGraphStore and LLM client, initialized at
        # a different lifecycle point than the agents.
        self._extra = param.extra_params or {}

    @property
    def agent_name(self) -> str:
        return "ProceduralAgent"

    @property
    def agent_description(self) -> str:
        return (
            "This agent answers 'how to' queries about multi-step tool "
            "workflows from the procedural memory graph. It decomposes the task "
            "into ordered sub-goals, anchors them to the most relevant stored "
            "trajectory's real steps, and diagnoses which required sub-goals "
            "that trajectory is missing so the agent supplies them itself. Best "
            "for multi-step procedural tasks."
        )

    @property
    def accuracy_score(self) -> int:
        return 7

    @property
    def token_cost(self) -> int:
        return 2  # one decomposition LLM call; alignment is deterministic

    @property
    def time_cost(self) -> int:
        return 3  # decomposition + in-memory alignment

    async def do_query(
        self,
        policy: QueryPolicy,
        query: QueryParam,
    ) -> tuple[list[Episode], dict[str, Any]]:
        """Retrieve a gap-aware grounded procedure for the query.

        Decomposes the query, anchors to the best trajectory, and attaches the
        edit-script diagnosis. Falls back to an empty result (triggering the
        caller's episodic fallback) when nothing in the graph matches.
        """
        _ = policy
        logger.info("CALLING %s with query: %s", self.agent_name, query.query)

        perf_metrics: dict[str, Any] = {
            "memory_search_called": 0,
            "memory_retrieval_time": 0.0,
            "agent": self.agent_name,
            "procedural_grounded": True,
        }

        start = time.time()

        try:
            from memmachine_server.procedural_memory.graph_store import (
                ProceduralGraphStore,
            )
            from memmachine_server.procedural_memory.grounded_retriever import (
                GroundedRetriever,
            )

            retriever = self._extra.get("procedural_retriever")
            if retriever is None:
                neo4j_driver = self._extra.get("neo4j_driver")
                llm = self._extra.get("procedural_llm")
                if neo4j_driver is None or llm is None:
                    logger.warning(
                        "ProceduralAgent: need neo4j_driver and procedural_llm "
                        "in extra_params (have driver=%s llm=%s)",
                        neo4j_driver is not None, llm is not None,
                    )
                    return [], perf_metrics
                store = ProceduralGraphStore(neo4j_driver)
                retriever = GroundedRetriever(store, llm)

            proc = await retriever.retrieve(query.query)

            perf_metrics["memory_search_called"] = 1
            perf_metrics["memory_retrieval_time"] = time.time() - start
            perf_metrics["retrieval_mode"] = proc.mode
            perf_metrics["num_gaps"] = len(proc.gaps)

            memory_text = proc.to_memory_text()
            if not memory_text:
                logger.info("ProceduralAgent: no grounded procedure for query")
                return [], perf_metrics

            session_key = query.memory.session_key if query.memory else "procedural"
            episodes = [
                Episode(
                    uid=f"procedural_grounded_{hash(memory_text) % 10**8}",
                    content=f"[PROCEDURE] {memory_text}",
                    session_key=session_key,
                    created_at=datetime.datetime.now(tz=datetime.UTC),
                    producer_id="procedural_memory",
                    producer_role="system",
                    episode_type=EpisodeType.MESSAGE,
                    metadata={
                        "type": "grounded_procedure",
                        "retrieval_mode": proc.mode,
                        "num_gaps": len(proc.gaps),
                        "source_trajectories": ",".join(
                            proc.ranked_trajectory_ids[:5]
                        ),
                    },
                )
            ]

            logger.info(
                "ProceduralAgent returned %s procedure (%d sub-goals, %d gaps, "
                "sources=%s)",
                proc.mode,
                len(proc.subgoals),
                len(proc.gaps),
                ",".join(proc.ranked_trajectory_ids[:3]),
            )
            return episodes, perf_metrics

        except Exception:
            logger.exception("ProceduralAgent: grounded retrieval failed")
            perf_metrics["memory_retrieval_time"] = time.time() - start
            return [], perf_metrics
