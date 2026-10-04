"""Evaluate proced_mem_bench results and generate final scores.

Reads the search output JSON, computes per-tier and aggregate IR metrics,
and writes a human-readable score report. This replaces the LLM judge step
used by episodic benchmarks -- proced_mem_bench uses deterministic IR metrics
(P@k, NDCG, MAP) instead of LLM-based scoring.

Usage:
    python proced_mem_bench_evaluate.py \
        --data-path result/proced_mem_bench_output.json \
        --target-path result/proced_mem_bench_eval_metrics.json
"""

from __future__ import annotations

import argparse
import json

from evaluation.procedural_memory.procedure_utils import (
    QueryTrajectory,
    aggregate_metrics,
    aggregate_trajectory_stats,
    format_final_score,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate proced_mem_bench results",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to search output JSON",
    )
    parser.add_argument(
        "--target-path",
        required=True,
        help="Path to save evaluation metrics JSON",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    # data is grouped by tier: {"HARD": [...], "MEDIUM": [...], "EASY": [...]}
    all_results: list[dict] = []
    for items in data.values():
        all_results.extend(items)

    if not all_results:
        print("No results found in input file.")
        return

    # Compute per-tier and aggregate IR metrics
    per_tier_metrics: dict[str, list[dict[str, float]]] = {}
    all_ir: list[dict[str, float]] = []
    all_trajectories: list[QueryTrajectory] = []

    for result in all_results:
        ir = {
            "p_at_1": result.get("p_at_1", 0.0),
            "p_at_5": result.get("p_at_5", 0.0),
            "ndcg_at_10": result.get("ndcg_at_10", 0.0),
            "average_precision": result.get("average_precision", 0.0),
        }
        all_ir.append(ir)

        tier = str(result.get("tier", "unknown"))
        per_tier_metrics.setdefault(tier, []).append(ir)

        # Rebuild trajectory for stats (including token cost)
        traj = QueryTrajectory(
            query_id=result.get("query_id", ""),
            query_text=result.get("query_text", ""),
        )
        traj.total_latency_ms = result.get("retrieval_latency_ms", 0.0)
        traj.selected_tool = result.get("selected_tool", "")
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
        traj.add_hop(
            agent_name=result.get("selected_tool", ""),
            query=result.get("query_text", ""),
            result_count=result.get("num_episodes_retrieved", 0),
            latency_ms=result.get("retrieval_latency_ms", 0.0),
        )
        all_trajectories.append(traj)

    overall_aggregate = aggregate_metrics(all_ir)
    tier_aggregates = {
        tier: aggregate_metrics(metrics) for tier, metrics in per_tier_metrics.items()
    }
    trajectory_stats = aggregate_trajectory_stats(all_trajectories)

    # Add per-result eval scores back into the grouped format
    # (for compatibility with generate_scores.py pattern)
    eval_results: dict[str, list[dict]] = {}
    for result in all_results:
        tier = str(result.get("tier", "unknown"))
        eval_entry = {
            "query_id": result.get("query_id", ""),
            "query_text": result.get("query_text", ""),
            "tier": tier,
            "category": tier,  # generate_scores.py groups by "category"
            "p_at_1": result.get("p_at_1", 0.0),
            "p_at_5": result.get("p_at_5", 0.0),
            "ndcg_at_10": result.get("ndcg_at_10", 0.0),
            "average_precision": result.get("average_precision", 0.0),
            "hop_count": result.get("hop_count", 0),
            "retrieval_latency_ms": result.get("retrieval_latency_ms", 0.0),
            "answer_latency_ms": result.get("answer_latency_ms", 0.0),
            "total_latency_ms": result.get("total_latency_ms", 0.0),
            "selected_tool": result.get("selected_tool", ""),
            # Token cost
            "tool_select_input_tokens": result.get("tool_select_input_tokens", 0),
            "tool_select_output_tokens": result.get("tool_select_output_tokens", 0),
            "retrieval_agent_input_tokens": result.get(
                "retrieval_agent_input_tokens", 0
            ),
            "retrieval_agent_output_tokens": result.get(
                "retrieval_agent_output_tokens", 0
            ),
            "answer_input_tokens": result.get("answer_input_tokens", 0),
            "answer_output_tokens": result.get("answer_output_tokens", 0),
            "total_input_tokens": result.get("total_input_tokens", 0),
            "total_output_tokens": result.get("total_output_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            # Call counts
            "total_llm_calls": result.get("total_llm_calls", 0),
            "embedding_calls": result.get("embedding_calls", 0),
            "reranker_calls": result.get("reranker_calls", 0),
            "llm_score": result.get("p_at_1", 0.0),  # For generate_scores.py compat
        }
        eval_results.setdefault(tier, []).append(eval_entry)

    with open(args.target_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    # Print final score
    final_score = format_final_score(
        overall_aggregate, tier_aggregates, trajectory_stats
    )
    print(final_score)
    print(f"\nEvaluation metrics saved to {args.target_path}")


if __name__ == "__main__":
    main()
