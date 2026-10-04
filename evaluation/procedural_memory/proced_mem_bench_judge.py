"""LLM Judge for procedural memory evaluation.

Measures procedure *quality* -- whether a retrieved or composed procedure
actually solves the given task. Complements the IR metrics (P@k, NDCG, MAP)
in proced_mem_bench_search.py, which measure whether we retrieved the right
trajectories but NOT whether the result is actionable.

Produces a pre/post comparison:
  - PRE:  episodic retrieval only (baseline, no procedural memory)
  - POST: procedural retrieval (Steiner tree + LLM validation)

For each query the judge scores:
  - completeness  (1-5): are all steps needed to achieve the goal present?
  - correctness   (1-5): are the steps individually valid?
  - ordering      (1-5): are the steps in a logical sequence?
  - executability (1-5): could an agent follow this procedure?

Usage:
    # Run judge on baseline (episodic) results
    python proced_mem_bench_judge.py \\
        --search-output result/proced_mem_bench_output_baseline.json \\
        --mode episodic \\
        --config-path configuration.yml \\
        --output result/judge_baseline.json

    # Run judge on procedural results
    python proced_mem_bench_judge.py \\
        --search-output result/proced_mem_bench_output_procedural.json \\
        --mode procedural \\
        --config-path configuration.yml \\
        --output result/judge_procedural.json

    # Compare pre vs post
    python proced_mem_bench_judge.py \\
        --compare \\
        --baseline result/judge_baseline.json \\
        --procedural result/judge_procedural.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating whether a procedure retrieved from memory actually solves
a given task. Score the procedure on four dimensions.

Task: {task}

Retrieved Procedure:
{procedure}

Score each dimension from 1 to 5:
  completeness  — are all steps needed to achieve the goal present?
  correctness   — are the individual steps valid and sensible?
  ordering      — are the steps in a logical, executable sequence?
  executability — could an agent follow this procedure without ambiguity?

Also give an overall pass/fail: pass if the procedure would likely succeed,
fail if it is missing key steps, has critical errors, or is too vague.

Respond with JSON only:
{{
  "completeness": <1-5>,
  "correctness":  <1-5>,
  "ordering":     <1-5>,
  "executability":<1-5>,
  "pass": true | false,
  "reasoning":    "<one sentence>"
}}
"""

NO_PROCEDURE_SCORES = {
    "completeness": 1,
    "correctness": 1,
    "ordering": 1,
    "executability": 1,
    "pass": False,
    "reasoning": "No procedure retrieved.",
}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def judge_procedure(
    llm: Any,
    task: str,
    procedure_text: str,
) -> dict[str, Any]:
    """Call the LLM judge and return structured scores."""
    if not procedure_text.strip():
        return NO_PROCEDURE_SCORES

    prompt = JUDGE_PROMPT.format(task=task, procedure=procedure_text)

    from memmachine_server.procedural_memory.llm_utils import (
        llm_complete,
        parse_json_object,
    )

    try:
        raw = await llm_complete(llm, prompt)
        result = parse_json_object(raw or "{}")
        return {
            "completeness": int(result.get("completeness", 1)),
            "correctness": int(result.get("correctness", 1)),
            "ordering": int(result.get("ordering", 1)),
            "executability": int(result.get("executability", 1)),
            "pass": bool(result.get("pass", False)),
            "reasoning": str(result.get("reasoning", "")),
        }
    except TypeError:
        # Interface mismatch -- abort the run rather than emit fake floor scores.
        raise
    except Exception:
        logger.exception("Judge LLM call failed for task: %s", task[:60])
        return {**NO_PROCEDURE_SCORES, "reasoning": "JUDGE_ERROR", "judge_error": True}


def scores_to_mean(scores: dict[str, Any]) -> float:
    dims = ["completeness", "correctness", "ordering", "executability"]
    return sum(scores[d] for d in dims) / (len(dims) * 5)  # normalized 0-1


# ---------------------------------------------------------------------------
# Run judge on search output
# ---------------------------------------------------------------------------

async def run_judge(args) -> None:
    """Score each result in a search output file with the LLM judge."""
    from evaluation.utils import agent_utils

    resource_manager = agent_utils.load_eval_config(args.config_path)

    # Get LLM client — use the same model name as the search pipeline:
    # config.retrieval_agent.llm_model (e.g. "openai_model"), or --llm-name override.
    try:
        conf = resource_manager.config
        llm_name = (
            getattr(args, "llm_name", None)
            or getattr(getattr(conf, "retrieval_agent", None), "llm_model", None)
        )
        if llm_name is None:
            raise RuntimeError(
                "Cannot determine LLM name: retrieval_agent.llm_model not set in "
                "configuration.yml and --llm-name not passed"
            )
        llm = await resource_manager.get_language_model(llm_name, validate=True)
    except Exception:
        logger.exception("Could not load LLM -- check configuration.yml")
        sys.exit(1)

    with open(args.search_output) as f:
        data = json.load(f)

    # Flatten: data may be {tier: [results]} or [results]
    flat: list[dict] = []
    if isinstance(data, list):
        flat = data
    elif isinstance(data, dict):
        for items in data.values():
            if isinstance(items, list):
                flat.extend(items)

    print(f"Judging {len(flat)} results from {args.search_output} ...")

    judged = []
    pass_count = 0

    for i, result in enumerate(flat):
        task = result.get("query_text", result.get("task", ""))
        tier = result.get("tier", "unknown")

        # Extract procedure text from result
        procedure_text = _extract_procedure_text(result, args.mode)

        start = time.time()
        scores = await judge_procedure(llm, task, procedure_text)
        elapsed_ms = (time.time() - start) * 1000

        if scores["pass"]:
            pass_count += 1

        judged_result = {
            **result,
            "judge": scores,
            "judge_mean_score": scores_to_mean(scores),
            "judge_pass": scores["pass"],
            "judge_latency_ms": elapsed_ms,
            "mode": args.mode,
        }
        judged.append(judged_result)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "  %d/%d judged | pass_rate=%.1f%% | last: %s (%.0fms)",
                i + 1, len(flat),
                100 * pass_count / (i + 1),
                scores["reasoning"][:60],
                elapsed_ms,
            )

    # Aggregate per tier
    per_tier: dict[str, list[dict]] = {}
    for r in judged:
        per_tier.setdefault(r.get("tier", "unknown"), []).append(r)

    summary = _build_summary(judged, per_tier, args.mode)
    print_summary(summary)

    output = {
        "mode": args.mode,
        "source": args.search_output,
        "summary": summary,
        "results": judged,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJudge output saved to {out_path}")


def _extract_procedure_text(result: dict, mode: str) -> str:
    """Extract the procedure text from a search result dict."""
    # Procedural mode: composed steps list
    if mode == "procedural":
        steps = result.get("composed_steps", [])
        if steps:
            lines = []
            for i, s in enumerate(steps):
                if isinstance(s, dict):
                    tool = s.get("tool_name", s.get("tool", "?"))
                    params = s.get("parameters", s.get("params", {}))
                    lines.append(f"{i+1}. {tool}({params})")
                else:
                    lines.append(f"{i+1}. {s}")
            return "\n".join(lines)
        # Fall back to tool sequence if no steps
        seq = result.get("tool_sequence", [])
        if seq:
            return "Tool sequence: " + " → ".join(seq)

    # Episodic mode: the LLM's generated answer (model_answer is the field
    # used by proced_mem_bench_search.py; fall back to common aliases too)
    answer = result.get("model_answer") or result.get("answer") or result.get("answer_text", "")
    if answer:
        return answer

    # Fall back to retrieved episode content
    episodes = result.get("retrieved_episodes", result.get("memories", []))
    if episodes:
        return "\n".join(
            ep.get("content", ep) if isinstance(ep, dict) else str(ep)
            for ep in episodes[:10]
        )

    return ""


def _build_summary(judged: list[dict], per_tier: dict, mode: str) -> dict:
    def tier_stats(results: list[dict]) -> dict:
        if not results:
            return {}
        scores = [r["judge"] for r in results]
        return {
            "n": len(results),
            "pass_rate": sum(1 for r in results if r["judge_pass"]) / len(results),
            "mean_score": sum(r["judge_mean_score"] for r in results) / len(results),
            "completeness": sum(s["completeness"] for s in scores) / (len(scores) * 5),
            "correctness": sum(s["correctness"] for s in scores) / (len(scores) * 5),
            "ordering": sum(s["ordering"] for s in scores) / (len(scores) * 5),
            "executability": sum(s["executability"] for s in scores) / (len(scores) * 5),
        }

    return {
        "mode": mode,
        "overall": tier_stats(judged),
        "per_tier": {tier: tier_stats(results) for tier, results in per_tier.items()},
    }


def print_summary(summary: dict) -> None:
    mode = summary["mode"]
    ov = summary["overall"]
    print(f"\n{'='*60}")
    print(f"LLM Judge Summary — mode={mode}")
    print(f"{'='*60}")
    print(f"  N:             {ov.get('n', 0)}")
    print(f"  Pass rate:     {ov.get('pass_rate', 0):.1%}")
    print(f"  Mean score:    {ov.get('mean_score', 0):.3f}  (0=worst, 1=best)")
    print(f"  Completeness:  {ov.get('completeness', 0):.3f}")
    print(f"  Correctness:   {ov.get('correctness', 0):.3f}")
    print(f"  Ordering:      {ov.get('ordering', 0):.3f}")
    print(f"  Executability: {ov.get('executability', 0):.3f}")
    print()
    for tier, stats in summary.get("per_tier", {}).items():
        print(f"  [{tier}] n={stats.get('n',0)}  pass={stats.get('pass_rate',0):.1%}  "
              f"mean={stats.get('mean_score',0):.3f}")


# ---------------------------------------------------------------------------
# Compare pre vs post
# ---------------------------------------------------------------------------

def run_compare(args) -> None:
    """Print a side-by-side pre/post comparison from two judge output files."""
    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.procedural) as f:
        procedural = json.load(f)

    b = baseline["summary"]["overall"]
    p = procedural["summary"]["overall"]

    print(f"\n{'='*70}")
    print(f"{'PRE/POST COMPARISON':^70}")
    print(f"{'='*70}")
    print(f"{'Metric':<22} {'Baseline (episodic)':>20} {'Procedural':>15} {'Delta':>10}")
    print(f"{'-'*70}")

    metrics = [
        ("Pass rate",       "pass_rate",      ".1%"),
        ("Mean score",      "mean_score",      ".3f"),
        ("Completeness",    "completeness",    ".3f"),
        ("Correctness",     "correctness",     ".3f"),
        ("Ordering",        "ordering",        ".3f"),
        ("Executability",   "executability",   ".3f"),
    ]

    for label, key, fmt in metrics:
        bv = b.get(key, 0.0)
        pv = p.get(key, 0.0)
        delta = pv - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<20} {format(bv, fmt):>20} {format(pv, fmt):>15} "
              f"{sign}{format(delta, fmt):>9}")

    print(f"{'-'*70}")

    # Per-tier
    print(f"\nPer-tier pass rates:")
    all_tiers = sorted(
        set(baseline["summary"]["per_tier"]) | set(procedural["summary"]["per_tier"])
    )
    for tier in all_tiers:
        b_tier = baseline["summary"]["per_tier"].get(tier, {})
        p_tier = procedural["summary"]["per_tier"].get(tier, {})
        bv = b_tier.get("pass_rate", 0.0)
        pv = p_tier.get("pass_rate", 0.0)
        delta = pv - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {tier:<10} baseline={bv:.1%}  procedural={pv:.1%}  "
              f"delta={sign}{delta:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM judge for procedural memory pre/post evaluation",
    )
    sub = parser.add_subparsers(dest="command")

    # judge subcommand
    judge_p = sub.add_parser("judge", help="Score a search output with the LLM judge")
    judge_p.add_argument("--search-output", required=True,
                         help="Path to search output JSON (from proced_mem_bench_search.py)")
    judge_p.add_argument("--mode", choices=["episodic", "procedural"], required=True,
                         help="episodic = baseline, procedural = post")
    judge_p.add_argument("--config-path", required=True,
                         help="Path to configuration.yml")
    judge_p.add_argument("--output", required=True,
                         help="Path to save judge output JSON")
    judge_p.add_argument("--llm-name", default=None,
                         help="Language model name from configuration.yml (default: first configured)")

    # compare subcommand
    compare_p = sub.add_parser("compare", help="Compare baseline vs procedural judge outputs")
    compare_p.add_argument("--baseline", required=True,
                           help="Path to baseline judge output JSON")
    compare_p.add_argument("--procedural", required=True,
                           help="Path to procedural judge output JSON")

    return parser


async def async_main() -> None:
    args = build_parser().parse_args()
    if args.command == "judge":
        await run_judge(args)
    elif args.command == "compare":
        run_compare(args)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(async_main())
