"""Stage 2 runner: synthetic gap-detection evaluation (offline, no LLM/DB).

Deletes one required sub-goal from a covering trajectory and measures whether
the edit-script aligner flags exactly that sub-goal as a GAP. The deletion is
the ground truth, so there is no hand-labelling. Also prints a few example edit
scripts so you can eyeball the diagnosis.

Usage:
    PYTHONPATH=. python3 evaluation/procedural_memory/stage2_run.py \
        --queries evaluation/data/proced_mem_bench/queries.json \
        --trajectories evaluation/data/proced_mem_bench/trajectories.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.procedural_memory.stage1_align import (
    load_queries,
    load_typed_trajectories,
    query_to_subgoals,
)
from evaluation.procedural_memory.stage2_gap import (
    _covers,
    _delete_subgoal,
    align_edit_script,
    gap_detection_eval,
    gaps,
)


def _fmt(step):
    return f"{step.verb}({step.obj or '-'}" + (f",{step.target}" if step.target else "") + ")"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--max-trajs-per-query", type=int, default=8,
                   help="cap covering trajectories per query (sample size knob)")
    p.add_argument("--out-dir", default="evaluation/procedural_memory/result/stage2")
    args = p.parse_args()

    queries = load_queries(args.queries)
    typed = load_typed_trajectories(args.trajectories)
    print(f"Loaded {len(queries)} queries, {len(typed)} typed trajectories")

    res = gap_detection_eval(queries, typed,
                             max_trajs_per_query=args.max_trajs_per_query)
    print("\n=== Stage 2: hardened gap-detection (deletion = ground truth) ===")
    print(f"  POSITIVES (a required step deleted, must detect it):")
    print(f"    pos trials: {res.pos_trials}   TP: {res.true_pos}   FN: {res.false_neg}")
    print(f"    recall:     {res.recall:.3f}")
    print(f"  NEGATIVES (fully covered, must report NO gap):")
    print(f"    neg trials: {res.neg_trials}   TN: {res.true_neg}   FP: {res.false_pos}")
    print(f"    specificity:{res.specificity:.3f}")
    print(f"  CONFUSERS (wrong-object lookalike left in place):")
    print(f"    conf trials:{res.conf_trials}   detected: {res.conf_detected}")
    print(f"  OVERALL:")
    print(f"    precision:  {res.precision:.3f}")
    print(f"    recall:     {res.recall:.3f}")
    print(f"    F1:         {res.f1:.3f}")
    print("  per deleted-verb (TP/trials):")
    for v, d in sorted(res.per_verb.items()):
        print(f"    {v:<8} {d['tp']}/{d['trials']}")

    print(f"\n=== {args.examples} example edit scripts (one required step deleted) ===")
    shown = 0
    for q in queries:
        if shown >= args.examples:
            break
        sg = query_to_subgoals(q)
        if len(sg) < 2:
            continue
        victim = next((s for s in sg if s.verb in ("clean", "heat", "cool")), sg[0])
        cover = next((s for s in typed.values() if _covers(sg, s)), None)
        if cover is None:
            continue
        ablated = _delete_subgoal(cover, victim)
        script = align_edit_script(sg, ablated)
        print(f"\n  {q.get('query_id')}: {q.get('query_text')}")
        print(f"    sub-goals: {[_fmt(s) for s in sg]}")
        print(f"    DELETED:   {_fmt(victim)}")
        print(f"    diagnosis: " + ", ".join(
            f"{op.op}:{_fmt(op.subgoal) if op.subgoal else _fmt(op.action)}"
            for op in script if op.op != "EXTRA"))
        print(f"    GAPs found: {[_fmt(g) for g in gaps(script)]}")
        shown += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gap_eval.json").write_text(json.dumps({
        "pos_trials": res.pos_trials, "true_pos": res.true_pos,
        "false_neg": res.false_neg,
        "neg_trials": res.neg_trials, "true_neg": res.true_neg,
        "false_pos": res.false_pos,
        "conf_trials": res.conf_trials, "conf_detected": res.conf_detected,
        "precision": res.precision, "recall": res.recall, "f1": res.f1,
        "specificity": res.specificity, "per_verb": res.per_verb,
    }, indent=2))
    print(f"\nsaved -> {out_dir / 'gap_eval.json'}")


if __name__ == "__main__":
    main()
