"""Stage 1 kill-test runner.

Runs the deterministic retrieval arms from ``stage1_align`` over the 40
benchmark queries and scores each with the SAME IR metrics as the main harness.
Reports P@1/P@5/NDCG@10/MAP overall and per tier, and prints the align-vs-bag
delta (does order help?).

Decomposition source is selectable:
  --decompose keyword  (default) -- deterministic, offline, no LLM/DB.
  --decompose llm                -- LLM decomposer (needs config + OPENAI_API_KEY).

The LLM path is the Stage-1 RETEST: it changes ONLY the sub-goal decomposition,
so comparing align(llm) vs align(keyword) on MEDIUM answers whether the earlier
MEDIUM loss was a decomposition artifact or a real result. Scoring, vocab, and
the bag control are identical across both.

Usage (keyword, offline):
    PYTHONPATH=. python3 evaluation/procedural_memory/stage1_run.py \
        --queries evaluation/data/proced_mem_bench/queries.json \
        --trajectories evaluation/data/proced_mem_bench/trajectories.json \
        --scorer both

Usage (LLM retest -- needs your key + config):
    PYTHONPATH=. python3 evaluation/procedural_memory/stage1_run.py \
        --queries evaluation/data/proced_mem_bench/queries.json \
        --trajectories evaluation/data/proced_mem_bench/trajectories.json \
        --scorer both --decompose llm --config-path evaluation/procedural_memory/configuration.yml
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evaluation.procedural_memory.procedure_utils import compute_ir_metrics
from evaluation.procedural_memory.stage1_align import (
    TypedStep,
    build_vocab,
    load_queries,
    load_typed_trajectories,
    query_to_subgoals,
    rank,
    relevant_ids,
)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


async def build_subgoals(queries, decompose: str, config_path: str | None) -> dict:
    """Return {query_id: [TypedStep]} using keyword or LLM decomposition."""
    if decompose == "keyword":
        return {q.get("query_id"): query_to_subgoals(q) for q in queries}

    # llm path -- load the MemMachine model via the eval config
    import sys
    from pathlib import Path as _P
    repo_root = _P(__file__).resolve().parents[2]
    for p in (repo_root, repo_root / "packages" / "common" / "src",
              repo_root / "packages" / "server" / "src"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    from dotenv import load_dotenv
    load_dotenv()
    from evaluation.utils import agent_utils
    from evaluation.procedural_memory.stage1_llm_decompose import llm_subgoals

    if not config_path:
        raise SystemExit("--decompose llm requires --config-path")
    rm = agent_utils.load_eval_config(config_path)
    # Resolve the configured model the SAME way the judge/search pipeline does:
    # resource_manager.config.retrieval_agent.llm_model (e.g. "openai_model").
    conf = rm.config
    llm_name = getattr(getattr(conf, "retrieval_agent", None), "llm_model", None)
    if llm_name is None:
        raise SystemExit(
            "Cannot determine LLM name: retrieval_agent.llm_model not set in "
            "configuration.yml"
        )
    model = await rm.get_language_model(llm_name, validate=True)
    print(f"LLM decomposer using model '{llm_name}'")

    out: dict = {}
    kw_fallbacks = 0
    for q in queries:
        sg = await llm_subgoals(q, model)
        if sg == query_to_subgoals(q) and len(sg) > 1:
            kw_fallbacks += 1  # multi-subgoal match == likely a fallback
        out[q.get("query_id")] = sg
        print(f"  {q.get('query_id'):<9} {q.get('query_text','')[:48]:<48} "
              f"-> {[ (s.verb, s.obj, s.target) for s in sg ]}")
    if kw_fallbacks:
        print(f"NOTE: {kw_fallbacks} multi-subgoal queries matched the keyword "
              f"decomposition (possible LLM fallbacks).")
    return out


def run_scorer(queries, typed_trajs, scorer, top_k, subgoals_by_id) -> dict:
    per_query = []
    for q in queries:
        rel = relevant_ids(q)
        sg = subgoals_by_id.get(q.get("query_id"))
        res = rank(q, typed_trajs, scorer=scorer, top_k=top_k, subgoals=sg)
        ir = compute_ir_metrics(res.ranked_ids, rel)
        per_query.append({
            "query_id": q.get("query_id"),
            "tier": q.get("tier", "unknown"),
            "query_text": q.get("query_text", ""),
            "subgoals": [vars(s) for s in res.subgoals],
            "ranked_ids": res.ranked_ids,
            "relevant_ids": sorted(rel),
            **ir,
        })
    return {"scorer": scorer, "results": per_query}


def summarize(run: dict) -> str:
    rows = run["results"]
    def agg(subset):
        return {
            "n": len(subset),
            "p_at_1": _mean([r["p_at_1"] for r in subset]),
            "p_at_5": _mean([r["p_at_5"] for r in subset]),
            "ndcg_at_10": _mean([r["ndcg_at_10"] for r in subset]),
            "map": _mean([r["average_precision"] for r in subset]),
        }
    lines = [f"=== Stage 1 arm: {run['scorer']} ===",
             "slice            n   P@1    P@5    NDCG@10  MAP"]
    overall = agg(rows)
    lines.append("overall        {n:3d} {p_at_1:.3f}  {p_at_5:.3f}  {ndcg_at_10:.3f}    {map:.3f}".format(**overall))
    for tier in ("EASY", "MEDIUM", "HARD"):
        sub = [r for r in rows if r["tier"].upper() == tier]
        if sub:
            a = agg(sub)
            lines.append(f"{tier:<14} " + "{n:3d} {p_at_1:.3f}  {p_at_5:.3f}  {ndcg_at_10:.3f}    {map:.3f}".format(**a))
    return "\n".join(lines)


async def amain() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--scorer", choices=["align", "bag", "both"], default="both")
    p.add_argument("--decompose", choices=["keyword", "llm"], default="keyword")
    p.add_argument("--config-path", default=None,
                   help="required for --decompose llm")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out-dir", default="evaluation/procedural_memory/result/stage1")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.decompose  # keyword | llm, used in output filenames

    queries = load_queries(args.queries)
    typed_trajs = load_typed_trajectories(args.trajectories)
    print(f"Loaded {len(queries)} queries, {len(typed_trajs)} typed trajectories "
          f"| decompose={args.decompose}")

    vocab = build_vocab(typed_trajs)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2))
    print(f"Frozen vocab: {vocab['counts']}")

    subgoals_by_id = await build_subgoals(queries, args.decompose, args.config_path)
    (out_dir / f"subgoals_{tag}.json").write_text(json.dumps(
        {qid: [vars(s) for s in sg] for qid, sg in subgoals_by_id.items()}, indent=2))

    scorers = ["align", "bag"] if args.scorer == "both" else [args.scorer]
    runs = {}
    for scorer in scorers:
        run = run_scorer(queries, typed_trajs, scorer, args.top_k, subgoals_by_id)
        runs[scorer] = run
        (out_dir / f"stage1_{scorer}_{tag}.json").write_text(json.dumps(run, indent=2))
        print("\n" + summarize(run))
        print(f"saved -> {out_dir / f'stage1_{scorer}_{tag}.json'}")

    if args.scorer == "both":
        print(f"\n=== ALIGN - BAG delta (decompose={args.decompose}) ===")
        a = runs["align"]["results"]
        bmap = {r["query_id"]: r for r in runs["bag"]["results"]}
        for tier in ("EASY", "MEDIUM", "HARD", "ALL"):
            sel = [r for r in a if tier == "ALL" or r["tier"].upper() == tier]
            da = _mean([r["ndcg_at_10"] for r in sel])
            db = _mean([bmap[r["query_id"]]["ndcg_at_10"] for r in sel])
            print(f"  {tier:<7} NDCG@10  align={da:.3f}  bag={db:.3f}  delta={da-db:+.3f}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
