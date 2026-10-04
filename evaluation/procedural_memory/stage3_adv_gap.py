"""E3: adversarial gap-detection evaluation (offline, no LLM/DB).

The hardened Stage-2 eval saturated (recall/specificity 1.0) because
proced_mem_bench trajectories are clean. E3 SYNTHESIZES hard cases so gap
detection can actually fail, giving a precision/recall curve instead of a
foregone 1.0. Every trial still has deletion-based ground truth.

Difficulty injectors (applied to a covering trajectory after deleting one
required sub-goal = the true gap):

  * CONFUSER   -- insert a same-verb, WRONG-object action (e.g. query needs
                  clean(soapbar); we deleted it but inject clean(mug)). A naive
                  detector that only checks "is verb clean present?" is fooled.
  * PARTIAL    -- delete setup steps (take/go_to/open) around a kept sub-goal,
                  so coverage is ragged (does the aligner over-report GAPs?).
  * NEAR_DUP   -- duplicate a kept sub-goal's action, so the sequence has repeats
                  (does alignment mis-match against the wrong copy?).
  * SHUFFLE    -- randomly permute the trajectory actions (break temporal order
                  entirely) before deleting -- the adversarial order case.

Each injector has an intensity knob and the harness sweeps intensities so you
get a degradation curve. Deterministic given --seed.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from evaluation.procedural_memory.stage1_align import (
    TypedStep,
    load_queries,
    load_typed_trajectories,
    query_to_subgoals,
)
from evaluation.procedural_memory.stage2_gap import (
    _covers,
    _delete_subgoal,
    _REALIZES,
    align_edit_script,
    gaps,
)

# A small pool of distractor objects per verb, drawn from ALFWorld vocabulary,
# used to synthesize wrong-object confusers.
_DISTRACTORS = {
    "clean": ["mug", "cup", "plate", "knife"],
    "heat": ["bread", "egg", "cup", "tomato"],
    "cool": ["apple", "lettuce", "tomato", "potato"],
    "put": ["bowl", "cup", "box", "plate"],
    "take": ["bowl", "cup", "box", "plate"],
}


def _inject_confuser(actions, victim, rng):
    """Insert a same-verb wrong-object action for the deleted sub-goal's verb."""
    pool = [o for o in _DISTRACTORS.get(victim.verb, ["widget"]) if o != victim.obj]
    if not pool:
        return actions
    distractor = rng.choice(pool)
    verb = next(iter(_REALIZES.get(victim.verb, {victim.verb})))
    fake = TypedStep(verb, distractor, None)
    pos = rng.randint(0, len(actions))
    return actions[:pos] + [fake] + actions[pos:]


def _inject_partial(actions, rng, drop_frac):
    """Drop a fraction of setup steps (go_to/take/open) to make coverage ragged."""
    setup = {"go_to", "take", "open"}
    out = []
    for a in actions:
        if a.verb in setup and rng.random() < drop_frac:
            continue
        out.append(a)
    return out


def _inject_near_dup(actions, rng, dup_count):
    """Duplicate random kept actions to create near-duplicate steps."""
    out = list(actions)
    for _ in range(dup_count):
        if not out:
            break
        idx = rng.randint(0, len(out) - 1)
        out.insert(idx, out[idx])
    return out


def _inject_shuffle(actions, rng):
    out = list(actions)
    rng.shuffle(out)
    return out


@dataclass
class Cell:
    injector: str
    intensity: float
    pos_trials: int
    tp: int
    fn: int
    neg_trials: int
    tn: int
    fp: int
    recall: float
    specificity: float


def _hit(detected, victim):
    return any(g.verb == victim.verb and
              (g.obj is None or victim.obj is None or g.obj == victim.obj)
              for g in detected)


def run_cell(queries, typed, injector, intensity, seed, max_trajs):
    rng = random.Random(seed)
    tp = fn = fp = tn = 0
    pos = neg = 0
    for q in queries:
        sg = query_to_subgoals(q)
        if len(sg) < 2:
            continue
        covering = [s for s in typed.values() if _covers(sg, s)][:max_trajs]
        for cover in covering:
            # positive: delete each required sub-goal, then inject difficulty
            for victim in sg:
                ablated = _delete_subgoal(cover, victim)
                if len(ablated) == len(cover):
                    continue
                if injector == "confuser":
                    for _ in range(int(intensity)):
                        ablated = _inject_confuser(ablated, victim, rng)
                elif injector == "partial":
                    ablated = _inject_partial(ablated, rng, intensity)
                elif injector == "near_dup":
                    ablated = _inject_near_dup(ablated, rng, int(intensity))
                elif injector == "shuffle":
                    if intensity >= 1:
                        ablated = _inject_shuffle(ablated, rng)
                pos += 1
                detected = gaps(align_edit_script(sg, ablated))
                if _hit(detected, victim):
                    tp += 1
                else:
                    fn += 1
                for g in detected:
                    if not (g.verb == victim.verb and
                            (g.obj is None or victim.obj is None or g.obj == victim.obj)):
                        fp += 1
            # negative: covered trajectory + injection, expect no gap
            neg_traj = list(cover)
            if injector == "confuser":
                for _ in range(int(intensity)):
                    neg_traj = _inject_confuser(neg_traj, sg[0], rng)
            elif injector == "partial":
                neg_traj = _inject_partial(neg_traj, rng, intensity)
            elif injector == "near_dup":
                neg_traj = _inject_near_dup(neg_traj, rng, int(intensity))
            elif injector == "shuffle" and intensity >= 1:
                neg_traj = _inject_shuffle(neg_traj, rng)
            neg += 1
            ndet = gaps(align_edit_script(sg, neg_traj))
            if ndet:
                fp += len(ndet)
            else:
                tn += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return Cell(injector, intensity, pos, tp, fn, neg, tn, fp, recall, spec)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--max-trajs-per-query", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="evaluation/procedural_memory/result/stage3_adv")
    args = p.parse_args()

    queries = load_queries(args.queries)
    typed = load_typed_trajectories(args.trajectories)
    print(f"Loaded {len(queries)} queries, {len(typed)} typed trajectories (seed={args.seed})")

    sweeps = {
        "confuser": [0, 1, 2, 3],
        "partial":  [0.0, 0.25, 0.5, 1.0],
        "near_dup": [0, 1, 2, 4],
        "shuffle":  [0, 1],
    }
    all_cells = []
    for injector, intensities in sweeps.items():
        print(f"\n=== E3 injector: {injector} ===")
        print("intensity  pos   recall   neg   specificity")
        for inten in intensities:
            c = run_cell(queries, typed, injector, inten, args.seed,
                         args.max_trajs_per_query)
            all_cells.append(vars(c))
            print(f"  {str(inten):<8} {c.pos_trials:4d}  {c.recall:.3f}    "
                  f"{c.neg_trials:4d}  {c.specificity:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"adv_gap_seed{args.seed}.json").write_text(json.dumps(all_cells, indent=2))
    print(f"\nsaved -> {out_dir / f'adv_gap_seed{args.seed}.json'}")


if __name__ == "__main__":
    main()
