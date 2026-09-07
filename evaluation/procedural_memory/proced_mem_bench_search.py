"""Run procedural memory retrieval benchmark queries against MemMachine.

Baseline phase: queries go through MemMachine's existing retrieval agent
(episodic memory only, no procedure-specific tier). Measures retrieval
quality (P@k, NDCG, MAP) and efficiency (hop count, latency, tokens).

Usage:
    python proced_mem_bench_search.py \
        --data-path ../data/proced_mem_bench/queries.json \
        --eval-result-path result/proced_mem_bench_output.json \
        --test-target retrieval_agent \
        --config-path configuration.yml \
        --concurrency 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOTS = [
    REPO_ROOT,
    REPO_ROOT / "packages" / "common" / "src",
    REPO_ROOT / "packages" / "server" / "src",
    REPO_ROOT / "packages" / "client" / "src",
]
for package_root in PACKAGE_ROOTS:
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.append(package_root_str)

from evaluation.procedural_memory.procedure_utils import (  # noqa: E402
    QueryTrajectory,
    TrajectoryLogger,
    aggregate_metrics,
    aggregate_trajectory_stats,
    compute_ir_metrics,
    format_final_score,
    format_results_json,
)

BENCHMARK_SESSION_PREFIX = "proced_mem_bench"
DEFAULT_CONCURRENCY = 1
DEFAULT_SEARCH_LIMIT = 10


ANSWER_PROMPT = """
You are asked to find a procedure (a sequence of actions) that can solve the
given task. Based on the memories you have of past action sequences, identify
the most relevant procedure.

<memories>
{memories}
</memories>

Task: {question}
Describe the relevant procedure you found in your memories (or say "no relevant
procedure found" if none match):
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run proced_mem_bench queries against MemMachine retrieval agent",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to benchmark queries JSON file",
    )
    parser.add_argument(
        "--eval-result-path",
        required=True,
        help="Path to save evaluation results JSON",
    )
    parser.add_argument(
        "--test-target",
        required=True,
        choices=["memmachine", "retrieval_agent", "llm"],
        help="Retrieval mode: memmachine (direct), retrieval_agent, or llm (baseline)",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to configuration.yml",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent search requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Top-k results to retrieve per query (default: {DEFAULT_SEARCH_LIMIT})",
    )
    return parser


async def process_query(
    query: dict[str, Any],
    query_idx: int,
    memory: Any,
    query_agent: Any,
    answer_model: Any,
    trajectory_logger: TrajectoryLogger,
    search_limit: int,
    test_target: str,
    agent_model_id: str,
    answer_model_id: str,
) -> dict[str, Any]:
    """Process a single benchmark query and return results with metrics.

    Args:
        query: dict with "query_id", "query_text", "relevant_ids", "tier".
        query_idx: index for logging.
        memory: MemMachine EpisodicMemory instance.
        query_agent: retrieval agent (ToolSelectAgent / MemMachineAgent).
        answer_model: LLM for generating answers.
        trajectory_logger: TrajectoryLogger for capturing hop data.
        search_limit: top-k parameter.
        test_target: "memmachine", "retrieval_agent", or "llm".
        agent_model_id: model ID string for logging.
        answer_model_id: model ID string for logging.

    Returns:
        dict with query results, IR metrics, and trajectory data.
    """
    from memmachine_server.common.episode_store.episode_model import (
        episodes_to_string,
    )
    from memmachine_server.retrieval_agent.common.agent_api import (
        QueryParam,
        QueryPolicy,
    )

    query_id = query.get("query_id", f"q_{query_idx}")
    query_text = query.get("query_text", "")
    tier = query.get("tier", "unknown")

    # Extract relevant trajectory IDs from the benchmark's ground truth.
    # The benchmark uses "relevant_trajectories" with objects containing
    # trajectory_id + relevance_score + reasoning (scored by LLM judge,
    # threshold ≥6 = relevant). We extract just the IDs for IR metrics.
    raw_relevant = query.get("relevant_trajectories", [])
    if raw_relevant and isinstance(raw_relevant[0], dict):
        # Full format: [{"trajectory_id": "alfworld_22", "relevance_score": 10.0, ...}]
        relevant_ids = {
            r["trajectory_id"]
            for r in raw_relevant
            if r.get("relevance_score", 0) >= 6.0
        }
    elif raw_relevant and isinstance(raw_relevant[0], str):
        # Simple format: ["alfworld_22", "alfworld_28", ...]
        relevant_ids = set(raw_relevant)
    else:
        # Fallback: check for flat relevant_ids key
        relevant_ids = set(query.get("relevant_ids", []))

    print(f"  Query {query_id} [{tier}]: {query_text[:80]}...")

    # Run retrieval
    query_policy = QueryPolicy(
        token_cost=10,
        time_cost=10,
        accuracy_score=10,
        confidence_score=10,
        max_attempts=3,
        max_return_len=10000,
    )

    # Build the query param using the session prefix to search across
    # all ingested trajectories.
    query_param = QueryParam(
        query=query_text,
        limit=search_limit,
        memory=memory,
    )

    start_time = time.time()

    if test_target == "llm":
        # LLM baseline: no retrieval, LLM answers from scratch
        chunks: list[Any] = []
        perf_metrics: dict[str, Any] = {}
        trajectory = QueryTrajectory(query_id=query_id, query_text=query_text)
        formatted_context = "(no memories available -- LLM baseline)"
    else:
        chunks, perf_metrics, trajectory = await trajectory_logger.query_with_logging(
            query_agent=query_agent,
            query_param=query_param,
            query_policy=query_policy,
            query_id=query_id,
        )
        formatted_context = episodes_to_string(chunks)

    retrieval_time = time.time() - start_time

    # Generate answer using the LLM (with token tracking)
    prompt = ANSWER_PROMPT.format(memories=formatted_context, question=query_text)
    rsp_start = time.time()
    model_answer, _, answer_in_tokens, answer_out_tokens = (
        await answer_model.generate_response_with_token_usage(user_prompt=prompt)
    )
    answer_time = time.time() - rsp_start

    # Populate answer-generation tokens on the trajectory
    trajectory.answer_input_tokens = answer_in_tokens
    trajectory.answer_output_tokens = answer_out_tokens

    # Extract retrieved trajectory IDs from episode metadata
    retrieved_ids: list[str] = []
    for chunk in chunks:
        traj_id = ""
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            traj_id = chunk.metadata.get("trajectory_id", "")
        if traj_id and traj_id not in retrieved_ids:
            retrieved_ids.append(traj_id)

    # Compute IR metrics
    ir_metrics = compute_ir_metrics(retrieved_ids, relevant_ids)

    result = {
        "query_id": query_id,
        "query_text": query_text,
        "tier": tier,
        "relevant_ids": list(relevant_ids),
        "retrieved_ids": retrieved_ids,
        "model_answer": model_answer,
        "num_episodes_retrieved": len(chunks),
        # IR metrics
        "p_at_1": ir_metrics["p_at_1"],
        "p_at_5": ir_metrics["p_at_5"],
        "ndcg_at_10": ir_metrics["ndcg_at_10"],
        "average_precision": ir_metrics["average_precision"],
        # Efficiency metrics
        "hop_count": trajectory.hop_count,
        "retrieval_latency_ms": retrieval_time * 1000,
        "answer_latency_ms": answer_time * 1000,
        "total_latency_ms": (retrieval_time + answer_time) * 1000,
        # Agent metadata
        "selected_tool": perf_metrics.get("agent", "N/A"),
        "memory_search_called": perf_metrics.get("memory_search_called", 0),
        "memory_retrieval_time": perf_metrics.get("memory_retrieval_time", 0),
        "llm_time": perf_metrics.get("llm_time", 0),
        # --- LLM token cost (per-stage breakdown) ---
        "tool_select_input_tokens": trajectory.tool_select_input_tokens,
        "tool_select_output_tokens": trajectory.tool_select_output_tokens,
        "retrieval_agent_input_tokens": trajectory.retrieval_agent_input_tokens,
        "retrieval_agent_output_tokens": trajectory.retrieval_agent_output_tokens,
        "answer_input_tokens": trajectory.answer_input_tokens,
        "answer_output_tokens": trajectory.answer_output_tokens,
        "total_input_tokens": trajectory.total_input_tokens,
        "total_output_tokens": trajectory.total_output_tokens,
        "total_tokens": trajectory.total_input_tokens + trajectory.total_output_tokens,
        # --- LLM / infra call counts ---
        "total_llm_calls": trajectory.total_llm_calls,
        "embedding_calls": trajectory.embedding_calls,
        "reranker_calls": trajectory.reranker_calls,
        # Legacy fields (kept for backward compat with agent_utils)
        "input_token": perf_metrics.get("input_token", 0),
        "output_token": perf_metrics.get("output_token", 0),
        "tool_select_input_token": perf_metrics.get("tool_select_input_token", 0),
        "tool_select_output_token": perf_metrics.get("tool_select_output_token", 0),
        "agent_model_id": agent_model_id,
        "answer_model_id": answer_model_id,
        # Full trajectory for analysis
        "trajectory": trajectory.to_dict(),
    }

    print(
        f"    -> P@1={ir_metrics['p_at_1']:.2f} "
        f"NDCG@10={ir_metrics['ndcg_at_10']:.2f} "
        f"hops={trajectory.hop_count} "
        f"latency={retrieval_time * 1000:.0f}ms "
        f"tokens={trajectory.total_input_tokens}in/{trajectory.total_output_tokens}out "
        f"llm_calls={trajectory.total_llm_calls} "
        f"episodes={len(chunks)}"
    )

    return result


async def run_benchmark() -> tuple[str, dict[str, Any]]:
    """Run the full proced_mem_bench benchmark.

    Returns:
        (eval_result_path, results_dict)
    """
    from memmachine_server.common.utils import async_with

    from evaluation.utils import agent_utils

    args = build_parser().parse_args()

    print("Starting proced_mem_bench search...")
    print(f"Data path: {args.data_path}")
    print(f"Eval result path: {args.eval_result_path}")
    print(f"Test target: {args.test_target}")
    print(f"Search limit: {args.search_limit}")
    print(f"Concurrency: {args.concurrency}")

    with open(args.data_path, "r") as f:
        data = json.load(f)

    # Data format: dict with "queries" key (Proced_mem_bench format)
    # or a flat list of queries
    if isinstance(data, dict):
        queries = data.get("queries", [])
    elif isinstance(data, list):
        queries = data
    else:
        raise TypeError(f"Unexpected data format: {type(data)}")

    print(f"Loaded {len(queries)} queries")

    resource_manager = agent_utils.load_eval_config(args.config_path)

    # Initialize MemMachine with a session that can search across all
    # ingested trajectories. The session_id here scopes the search.
    agent_name = (
        "ToolSelectAgent"
        if args.test_target == "retrieval_agent"
        else "MemMachineAgent"
    )

    (
        memory,
        answer_model,
        query_agent,
        agent_model_id,
        answer_model_id,
    ) = await agent_utils.init_memmachine_params(
        resource_manager=resource_manager,
        session_id=BENCHMARK_SESSION_PREFIX,
        agent_name=agent_name,
    )

    trajectory_logger = TrajectoryLogger()

    # Process all queries
    all_results: list[dict[str, Any]] = []
    all_trajectories: list[QueryTrajectory] = []
    per_tier_metrics: dict[str, list[dict[str, float]]] = {}

    semaphore = asyncio.Semaphore(args.concurrency)

    async def wrapped_query(query: dict, idx: int) -> dict[str, Any]:
        return await process_query(
            query=query,
            query_idx=idx,
            memory=memory,
            query_agent=query_agent,
            answer_model=answer_model,
            trajectory_logger=trajectory_logger,
            search_limit=args.search_limit,
            test_target=args.test_target,
            agent_model_id=agent_model_id,
            answer_model_id=answer_model_id,
        )

    tasks = [async_with(semaphore, wrapped_query(q, i)) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks)

    for result in results:
        all_results.append(result)

        # Rebuild trajectory from result for stats (including token cost)
        traj = QueryTrajectory(
            query_id=result["query_id"],
            query_text=result["query_text"],
        )
        traj.total_latency_ms = result["retrieval_latency_ms"]
        traj.selected_tool = result["selected_tool"]
        # Token cost fields
        traj.tool_select_input_tokens = result.get("tool_select_input_tokens", 0)
        traj.tool_select_output_tokens = result.get("tool_select_output_tokens", 0)
        traj.retrieval_agent_input_tokens = result.get(
            "retrieval_agent_input_tokens", 0
        )
        traj.retrieval_agent_output_tokens = result.get(
            "retrieval_agent_output_tokens", 0
        )
        traj.answer_input_tokens = result.get("answer_input_tokens", 0)
        traj.answer_output_tokens = result.get("answer_output_tokens", 0)
        traj.embedding_calls = result.get("embedding_calls", 0)
        traj.reranker_calls = result.get("reranker_calls", 0)
        traj._retrieval_llm_calls = result.get("total_llm_calls", 0)
        # Add a synthetic hop from the recorded data
        traj.add_hop(
            agent_name=result["selected_tool"],
            query=result["query_text"],
            result_count=result["num_episodes_retrieved"],
            latency_ms=result["retrieval_latency_ms"],
        )
        all_trajectories.append(traj)

        # Group IR metrics by tier
        tier = result["tier"]
        ir = {
            "p_at_1": result["p_at_1"],
            "p_at_5": result["p_at_5"],
            "ndcg_at_10": result["ndcg_at_10"],
            "average_precision": result["average_precision"],
        }
        per_tier_metrics.setdefault(tier, []).append(ir)

    # Compute aggregates
    all_ir = [
        {
            "p_at_1": r["p_at_1"],
            "p_at_5": r["p_at_5"],
            "ndcg_at_10": r["ndcg_at_10"],
            "average_precision": r["average_precision"],
        }
        for r in all_results
    ]
    overall_aggregate = aggregate_metrics(all_ir)
    tier_aggregates = {
        tier: aggregate_metrics(metrics) for tier, metrics in per_tier_metrics.items()
    }
    trajectory_stats = aggregate_trajectory_stats(all_trajectories)

    # Print summary
    final_score = format_final_score(
        overall_aggregate, tier_aggregates, trajectory_stats
    )
    print("\n" + final_score)

    # Attach final score to first result for generate_scores.py compatibility
    if all_results:
        all_results[0]["proced_mem_bench_final_matrix"] = final_score

    # Format output
    grouped_results = format_results_json(all_results)

    return args.eval_result_path, grouped_results


async def main() -> None:
    eval_result_path, results = await run_benchmark()
    with open(eval_result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {eval_result_path}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
