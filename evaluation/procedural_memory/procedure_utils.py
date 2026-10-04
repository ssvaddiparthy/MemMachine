"""Shared utilities for procedural memory benchmarks.

Trajectory logging, step counting, and IR metrics.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory logging
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryHop:
    """A single hop in a retrieval agent's trajectory."""

    agent_name: str
    query: str
    result_count: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryTrajectory:
    """Full trajectory of a single query through the retrieval agent.

    Tracks both retrieval quality (hops, latency) and LLM cost
    (token counts per stage, total LLM calls) so we can quantify
    whether procedural memory saves tokens/calls vs zero-shot ReAct.
    """

    query_id: str
    query_text: str
    hops: list[TrajectoryHop] = field(default_factory=list)
    total_latency_ms: float = 0.0
    selected_tool: str = ""

    # --- LLM cost tracking (tokens) ---
    # Retrieval agent tokens (tool selection + sub-query decomposition)
    tool_select_input_tokens: int = 0
    tool_select_output_tokens: int = 0
    retrieval_agent_input_tokens: int = 0
    retrieval_agent_output_tokens: int = 0

    # Answer generation tokens
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0

    # Embedding call count (each pgvector search = 1 embedding call)
    embedding_calls: int = 0
    # Reranker call count
    reranker_calls: int = 0

    @property
    def hop_count(self) -> int:
        return len(self.hops)

    @property
    def total_input_tokens(self) -> int:
        """All input tokens across all LLM calls (routing + retrieval + answer)."""
        return (
            self.tool_select_input_tokens
            + self.retrieval_agent_input_tokens
            + self.answer_input_tokens
        )

    @property
    def total_output_tokens(self) -> int:
        """All output tokens across all LLM calls (routing + retrieval + answer)."""
        return (
            self.tool_select_output_tokens
            + self.retrieval_agent_output_tokens
            + self.answer_output_tokens
        )

    @property
    def total_llm_calls(self) -> int:
        """Total LLM inference calls: tool_select(1) + retrieval_agent(N) + answer(1).

        ChainOfQuery does up to 3 LLM iterations; SplitQuery does 1 decomposition;
        MemMachineAgent does 0 LLM calls.
        """
        calls = 0
        # Tool select always makes 1 LLM call (unless MemMachineAgent directly)
        if self.tool_select_input_tokens > 0:
            calls += 1
        # Retrieval agent LLM calls (from perf_metrics)
        if self.retrieval_agent_input_tokens > 0:
            calls += max(1, self._retrieval_llm_calls)
        # Answer always makes 1 LLM call
        if self.answer_input_tokens > 0:
            calls += 1
        return calls

    # Internal: set by TrajectoryLogger from perf_metrics
    _retrieval_llm_calls: int = field(default=0, repr=False)

    def add_hop(
        self,
        agent_name: str,
        query: str,
        result_count: int,
        latency_ms: float,
        **metadata: Any,
    ) -> None:
        self.hops.append(
            TrajectoryHop(
                agent_name=agent_name,
                query=query,
                result_count=result_count,
                latency_ms=latency_ms,
                metadata=metadata,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "hop_count": self.hop_count,
            "total_latency_ms": self.total_latency_ms,
            "selected_tool": self.selected_tool,
            # Token cost breakdown
            "tool_select_input_tokens": self.tool_select_input_tokens,
            "tool_select_output_tokens": self.tool_select_output_tokens,
            "retrieval_agent_input_tokens": self.retrieval_agent_input_tokens,
            "retrieval_agent_output_tokens": self.retrieval_agent_output_tokens,
            "answer_input_tokens": self.answer_input_tokens,
            "answer_output_tokens": self.answer_output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_llm_calls": self.total_llm_calls,
            "embedding_calls": self.embedding_calls,
            "reranker_calls": self.reranker_calls,
            "hops": [
                {
                    "agent_name": h.agent_name,
                    "query": h.query,
                    "result_count": h.result_count,
                    "latency_ms": h.latency_ms,
                    "metadata": h.metadata,
                }
                for h in self.hops
            ],
        }


class TrajectoryLogger:
    """Wraps a retrieval agent query call to capture hop-level trajectory data.

    Captures both trajectory shape (hops, latency) and LLM cost (tokens per
    stage, embedding/reranker calls) so the benchmark can quantify efficiency
    savings from procedural memory.

    Usage:
        tlog = TrajectoryLogger()
        chunks, perf_metrics, trajectory = await tlog.query_with_logging(
            query_agent, query_param, query_id="q_001",
        )
    """

    async def query_with_logging(
        self,
        query_agent: Any,
        query_param: Any,
        query_policy: Any,
        query_id: str,
    ) -> tuple[list[Any], dict[str, Any], QueryTrajectory]:
        """Execute a query through the agent and capture trajectory metadata.

        Returns:
            (episodes, perf_metrics, trajectory)
        """
        trajectory = QueryTrajectory(
            query_id=query_id,
            query_text=query_param.query,
        )

        start = time.time()
        chunks, perf_metrics = await query_agent.do_query(query_policy, query_param)
        elapsed_ms = (time.time() - start) * 1000

        trajectory.total_latency_ms = elapsed_ms
        trajectory.selected_tool = perf_metrics.get("agent", "")

        # --- Populate LLM token cost from perf_metrics ---
        # ToolSelectAgent sets tool_select_input_token / tool_select_output_token
        trajectory.tool_select_input_tokens = perf_metrics.get(
            "tool_select_input_token", 0
        )
        trajectory.tool_select_output_tokens = perf_metrics.get(
            "tool_select_output_token", 0
        )

        # Child agent (CoQ/Split/MemMachine) sets input_token / output_token
        trajectory.retrieval_agent_input_tokens = perf_metrics.get("input_token", 0)
        trajectory.retrieval_agent_output_tokens = perf_metrics.get("output_token", 0)

        # memory_search_called tracks how many times pgvector was hit.
        # Each pgvector search = 1 embedding call + 1 reranker call.
        mem_searches = perf_metrics.get("memory_search_called", 1)
        trajectory.embedding_calls = mem_searches
        trajectory.reranker_calls = mem_searches

        # For ChainOfQuery, memory_search_called also reflects LLM iterations
        # (each hop = 1 LLM call to rewrite + 1 memory search).
        agent_name = perf_metrics.get("agent", "Unknown")
        if agent_name == "ChainOfQueryAgent":
            # CoQ does N iterations, each with an LLM call + memory search
            trajectory._retrieval_llm_calls = mem_searches
        elif agent_name == "SplitQueryAgent":
            # SplitQuery does 1 LLM decomposition call + N parallel searches
            trajectory._retrieval_llm_calls = 1
        else:
            # MemMachineAgent does 0 LLM calls (direct vector search)
            trajectory._retrieval_llm_calls = 0

        # Extract hop data from perf_metrics returned by MemMachine agents.
        mem_retrieval_time = perf_metrics.get("memory_retrieval_time", 0)
        llm_time = perf_metrics.get("llm_time", 0)

        # Record at least one hop per query (the top-level dispatch).
        trajectory.add_hop(
            agent_name=agent_name,
            query=query_param.query,
            result_count=len(chunks),
            latency_ms=mem_retrieval_time * 1000,
            memory_search_called=mem_searches,
            llm_time_ms=llm_time * 1000,
        )

        return chunks, perf_metrics, trajectory


# ---------------------------------------------------------------------------
# IR Metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Average precision for a single query."""
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            hits += 1
            sum_precision += hits / i
    return sum_precision / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@k using binary relevance (1 if relevant, 0 otherwise)."""
    if k <= 0 or not relevant_ids:
        return 0.0

    # DCG
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        rel = 1.0 if rid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant docs at the top
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


def compute_ir_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> dict[str, float]:
    """Compute all IR metrics for a single query."""
    return {
        "p_at_1": precision_at_k(retrieved_ids, relevant_ids, 1),
        "p_at_5": precision_at_k(retrieved_ids, relevant_ids, 5),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, relevant_ids, 10),
        "average_precision": average_precision(retrieved_ids, relevant_ids),
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


def aggregate_metrics(
    per_query_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Average IR metrics across queries."""
    if not per_query_metrics:
        return {
            "p_at_1": 0.0,
            "p_at_5": 0.0,
            "ndcg_at_10": 0.0,
            "map": 0.0,
        }

    n = len(per_query_metrics)
    return {
        "p_at_1": sum(m["p_at_1"] for m in per_query_metrics) / n,
        "p_at_5": sum(m["p_at_5"] for m in per_query_metrics) / n,
        "ndcg_at_10": sum(m["ndcg_at_10"] for m in per_query_metrics) / n,
        "map": sum(m["average_precision"] for m in per_query_metrics) / n,
    }


def aggregate_trajectory_stats(
    trajectories: list[QueryTrajectory],
) -> dict[str, float]:
    """Compute summary statistics over all query trajectories.

    Includes both efficiency (hops, latency) and cost (tokens, LLM calls,
    embedding calls, reranker calls) so the paper can show procedural memory
    reduces token spend alongside improving retrieval quality.
    """
    if not trajectories:
        return {
            "mean_hop_count": 0.0,
            "mean_latency_ms": 0.0,
            "total_queries": 0,
            # Token cost
            "mean_total_input_tokens": 0.0,
            "mean_total_output_tokens": 0.0,
            "mean_total_tokens": 0.0,
            "sum_total_input_tokens": 0,
            "sum_total_output_tokens": 0,
            # Per-stage token breakdown
            "mean_tool_select_input_tokens": 0.0,
            "mean_tool_select_output_tokens": 0.0,
            "mean_retrieval_agent_input_tokens": 0.0,
            "mean_retrieval_agent_output_tokens": 0.0,
            "mean_answer_input_tokens": 0.0,
            "mean_answer_output_tokens": 0.0,
            # LLM / infra call counts
            "mean_llm_calls": 0.0,
            "mean_embedding_calls": 0.0,
            "mean_reranker_calls": 0.0,
            "sum_llm_calls": 0,
            "sum_embedding_calls": 0,
            "sum_reranker_calls": 0,
        }

    n = len(trajectories)
    sum_input = sum(t.total_input_tokens for t in trajectories)
    sum_output = sum(t.total_output_tokens for t in trajectories)
    sum_llm = sum(t.total_llm_calls for t in trajectories)
    sum_embed = sum(t.embedding_calls for t in trajectories)
    sum_rerank = sum(t.reranker_calls for t in trajectories)

    return {
        "mean_hop_count": sum(t.hop_count for t in trajectories) / n,
        "mean_latency_ms": sum(t.total_latency_ms for t in trajectories) / n,
        "total_queries": n,
        # Token cost
        "mean_total_input_tokens": sum_input / n,
        "mean_total_output_tokens": sum_output / n,
        "mean_total_tokens": (sum_input + sum_output) / n,
        "sum_total_input_tokens": sum_input,
        "sum_total_output_tokens": sum_output,
        # Per-stage token breakdown
        "mean_tool_select_input_tokens": (
            sum(t.tool_select_input_tokens for t in trajectories) / n
        ),
        "mean_tool_select_output_tokens": (
            sum(t.tool_select_output_tokens for t in trajectories) / n
        ),
        "mean_retrieval_agent_input_tokens": (
            sum(t.retrieval_agent_input_tokens for t in trajectories) / n
        ),
        "mean_retrieval_agent_output_tokens": (
            sum(t.retrieval_agent_output_tokens for t in trajectories) / n
        ),
        "mean_answer_input_tokens": (
            sum(t.answer_input_tokens for t in trajectories) / n
        ),
        "mean_answer_output_tokens": (
            sum(t.answer_output_tokens for t in trajectories) / n
        ),
        # LLM / infra call counts
        "mean_llm_calls": sum_llm / n,
        "mean_embedding_calls": sum_embed / n,
        "mean_reranker_calls": sum_rerank / n,
        "sum_llm_calls": sum_llm,
        "sum_embedding_calls": sum_embed,
        "sum_reranker_calls": sum_rerank,
    }


# ---------------------------------------------------------------------------
# Results formatting (matches MemMachine eval pattern)
# ---------------------------------------------------------------------------


def format_results_json(
    per_query_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group results by tier for JSON output, matching the existing eval pattern.

    Input: list of dicts, each with at least a "tier" key.
    Output: dict keyed by tier, each value a list of result dicts.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in per_query_results:
        tier = str(result.get("tier", "unknown"))
        grouped.setdefault(tier, []).append(result)
    return grouped


def format_final_score(
    aggregate: dict[str, float],
    per_tier: dict[str, dict[str, float]],
    trajectory_stats: dict[str, float],
) -> str:
    """Format the final score report as a human-readable string.

    Includes retrieval quality (IR metrics), efficiency (latency, hops),
    and LLM cost (tokens, calls) so the paper can quantify all three
    dimensions of improvement from procedural memory.
    """
    lines = [
        "proced_mem_bench Baseline Results",
        "=" * 50,
        "",
        "Retrieval Quality:",
        f"  P@1:      {aggregate['p_at_1']:.4f}",
        f"  P@5:      {aggregate['p_at_5']:.4f}",
        f"  NDCG@10:  {aggregate['ndcg_at_10']:.4f}",
        f"  MAP:      {aggregate['map']:.4f}",
        "",
        "Efficiency:",
        f"  Mean Hop Count:    {trajectory_stats['mean_hop_count']:.2f}",
        f"  Mean Latency (ms): {trajectory_stats['mean_latency_ms']:.1f}",
        f"  Total Queries:     {trajectory_stats['total_queries']}",
        "",
        "LLM Token Cost (per query avg):",
        f"  Tool Select:      {trajectory_stats.get('mean_tool_select_input_tokens', 0):.0f} in / {trajectory_stats.get('mean_tool_select_output_tokens', 0):.0f} out",
        f"  Retrieval Agent:  {trajectory_stats.get('mean_retrieval_agent_input_tokens', 0):.0f} in / {trajectory_stats.get('mean_retrieval_agent_output_tokens', 0):.0f} out",
        f"  Answer Gen:       {trajectory_stats.get('mean_answer_input_tokens', 0):.0f} in / {trajectory_stats.get('mean_answer_output_tokens', 0):.0f} out",
        f"  Total:            {trajectory_stats.get('mean_total_input_tokens', 0):.0f} in / {trajectory_stats.get('mean_total_output_tokens', 0):.0f} out / {trajectory_stats.get('mean_total_tokens', 0):.0f} total",
        "",
        "Infra Calls (per query avg / total):",
        f"  LLM Calls:       {trajectory_stats.get('mean_llm_calls', 0):.2f} / {trajectory_stats.get('sum_llm_calls', 0)}",
        f"  Embedding Calls: {trajectory_stats.get('mean_embedding_calls', 0):.2f} / {trajectory_stats.get('sum_embedding_calls', 0)}",
        f"  Reranker Calls:  {trajectory_stats.get('mean_reranker_calls', 0):.2f} / {trajectory_stats.get('sum_reranker_calls', 0)}",
        "",
        "Cost Totals (all queries):",
        f"  Total Input Tokens:  {trajectory_stats.get('sum_total_input_tokens', 0):,}",
        f"  Total Output Tokens: {trajectory_stats.get('sum_total_output_tokens', 0):,}",
        f"  Total LLM Calls:     {trajectory_stats.get('sum_llm_calls', 0)}",
        "",
        "Per-Tier Breakdown:",
    ]

    for tier, metrics in sorted(per_tier.items()):
        lines.append(f"  {tier}:")
        lines.append(f"    P@1:     {metrics['p_at_1']:.4f}")
        lines.append(f"    P@5:     {metrics['p_at_5']:.4f}")
        lines.append(f"    NDCG@10: {metrics['ndcg_at_10']:.4f}")
        lines.append(f"    MAP:     {metrics['map']:.4f}")

    return "\n".join(lines)
