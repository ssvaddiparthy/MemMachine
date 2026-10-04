"""Stage 2: edit-script alignment + synthetic gap-detection evaluation.

Two pieces:

1. ``align_edit_script`` -- the same weighted-Levenshtein DP as Stage 1, but it
   BACKTRACES to emit a labelled edit script:
     COVERED  -- a query sub-goal matched a trajectory action (verb+obj+target ok)
     REBIND   -- right verb/object, wrong target instance (repairable binding)
     GAP      -- a required query sub-goal with no matching action (MISSING step)
     EXTRA    -- a trajectory action not required by the query (setup/noise)
   This is what the order-free bag CANNOT produce: the bag has no notion of a
   required-but-absent sub-goal.

2. ``gap_detection_eval`` -- SYNTHETIC GOLD by construction. Take a trajectory
   that COVERS a query, delete the steps realizing ONE required sub-goal, and
   check whether the aligner flags exactly that sub-goal as GAP. Because we did
   the deletion, we know the ground truth with no hand-labelling and no LLM
   marking its own homework. Scores precision/recall/F1 of GAP detection.

Deterministic, offline: no LLM, no DB. Reuses the Stage-1 cost model so the
diagnosis is consistent with the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.procedural_memory.stage1_align import (
    TypedStep,
    _EXTRA_COST,
    _GAP_COST,
    pair_cost,
    query_to_subgoals,
)

# A substitution is COVERED when verb+object match and target is compatible;
# REBIND when verb+object match but the target instance differs.
_COVERED_MAX = 0.0          # exact (pair_cost 0)
_REBIND_MAX = _GAP_COST - 0.01  # same verb+obj, wrong target < a full gap


@dataclass
class EditOp:
    op: str                       # COVERED | REBIND | GAP | EXTRA
    subgoal: TypedStep | None     # query side (None for EXTRA)
    action: TypedStep | None      # trajectory side (None for GAP)


def align_edit_script(subgoals: list[TypedStep],
                      actions: list[TypedStep]) -> list[EditOp]:
    """Backtrace the alignment DP into a labelled edit script."""
    m, n = len(subgoals), len(actions)
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + _EXTRA_COST
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + _GAP_COST
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = dp[i - 1][j - 1] + pair_cost(subgoals[i - 1], actions[j - 1])
            gap = dp[i - 1][j] + _GAP_COST
            extra = dp[i][j - 1] + _EXTRA_COST
            dp[i][j] = min(sub, gap, extra)

    # Backtrace from (m, n)
    ops: list[EditOp] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            pc = pair_cost(subgoals[i - 1], actions[j - 1])
            if abs(dp[i][j] - (dp[i - 1][j - 1] + pc)) < 1e-9:
                q, a = subgoals[i - 1], actions[j - 1]
                if q.verb == a.verb and (q.obj is None or a.obj is None or q.obj == a.obj):
                    # verb + object compatible -> COVERED or REBIND on target
                    if (q.target is not None and a.target is not None
                            and q.target != a.target):
                        op = "REBIND"
                    else:
                        op = "COVERED"
                else:
                    # substituted but verb/obj mismatch: treat as GAP for the
                    # sub-goal (its required effect was not really satisfied)
                    op = "GAP"
                ops.append(EditOp(op, q, a))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + _GAP_COST)) < 1e-9:
            ops.append(EditOp("GAP", subgoals[i - 1], None))
            i -= 1
            continue
        # else EXTRA
        ops.append(EditOp("EXTRA", None, actions[j - 1]))
        j -= 1
    ops.reverse()
    return ops


def gaps(script: list[EditOp]) -> list[TypedStep]:
    """The sub-goals the aligner diagnosed as missing."""
    return [op.subgoal for op in script if op.op == "GAP" and op.subgoal]


# ---------------------------------------------------------------------------
# Synthetic gap-detection evaluation (deletion = ground truth)
# ---------------------------------------------------------------------------

# which trajectory verbs realize a given sub-goal verb (for deletion)
_REALIZES = {
    "clean": {"clean"},
    "heat": {"heat"},
    "cool": {"cool"},
    "put": {"put"},
    "take": {"take"},
    "examine": {"examine"},
    "use": {"use"},
}


def _covers(subgoals: list[TypedStep], actions: list[TypedStep]) -> bool:
    """True if every sub-goal has a verb+object match somewhere in actions."""
    for q in subgoals:
        ok = any(a.verb == q.verb and (q.obj is None or a.obj is None or a.obj == q.obj)
                 for a in actions)
        if not ok:
            return False
    return True


def _delete_subgoal(actions: list[TypedStep], target: TypedStep) -> list[TypedStep]:
    """Remove the actions realizing one sub-goal (the synthetic gap)."""
    verbs = _REALIZES.get(target.verb, {target.verb})
    out = []
    for a in actions:
        if a.verb in verbs and (target.obj is None or a.obj is None or a.obj == target.obj):
            continue  # delete this step
        out.append(a)
    return out


@dataclass
class GapEvalResult:
    # positive trials: a required sub-goal was deleted, aligner must find it
    pos_trials: int
    true_pos: int
    false_neg: int
    # negative trials: nothing deleted (covered), aligner must report NO gap
    neg_trials: int
    true_neg: int
    false_pos: int
    # confuser trials: step deleted but a wrong-object lookalike left in place
    conf_trials: int
    conf_detected: int
    precision: float
    recall: float
    f1: float
    specificity: float   # TN / (TN + FP) on negatives
    per_verb: dict


def _has_confuser(actions: list[TypedStep], victim: TypedStep) -> bool:
    """Is there a same-verb action with a DIFFERENT object (a lookalike)?"""
    verbs = _REALIZES.get(victim.verb, {victim.verb})
    return any(a.verb in verbs and a.obj is not None and victim.obj is not None
               and a.obj != victim.obj for a in actions)


def gap_detection_eval(queries: list[dict],
                       typed_trajs: dict[str, list[TypedStep]],
                       subgoals_by_id: dict | None = None,
                       max_trajs_per_query: int = 8) -> GapEvalResult:
    """Hardened gap-detection eval. For every multi-subgoal query:

    * NEGATIVE: for each covering trajectory, run the aligner UNCHANGED and
      require zero GAPs (false positive if any). This is the control the toy
      eval lacked.
    * POSITIVE: for each covering trajectory and EACH required sub-goal, delete
      that sub-goal's steps and require the aligner to flag exactly it.
    * CONFUSER: a positive trial where a same-verb, wrong-object action remains
      after deletion (detection must not be fooled by the lookalike).

    ``max_trajs_per_query`` caps covering trajectories per query so the sample
    is bounded but much larger than the one-per-query toy.
    """
    tp = fn = fp = tn = 0
    pos_trials = neg_trials = 0
    conf_trials = conf_detected = 0
    per_verb: dict = {}

    for q in queries:
        sg = (subgoals_by_id or {}).get(q.get("query_id")) or query_to_subgoals(q)
        if len(sg) < 2:
            continue
        covering = [s for s in typed_trajs.values() if _covers(sg, s)]
        covering = covering[:max_trajs_per_query]
        for cover in covering:
            # --- negative: nothing deleted, expect no gaps ---
            neg_trials += 1
            neg_script = align_edit_script(sg, cover)
            if gaps(neg_script):
                fp += len(gaps(neg_script))
            else:
                tn += 1
            # --- positive: delete each required sub-goal in turn ---
            for victim in sg:
                ablated = _delete_subgoal(cover, victim)
                if len(ablated) == len(cover):
                    continue  # nothing actually removed (verb absent) -> skip
                pos_trials += 1
                is_conf = _has_confuser(ablated, victim)
                if is_conf:
                    conf_trials += 1
                script = align_edit_script(sg, ablated)
                detected = gaps(script)
                hit = any(g.verb == victim.verb and
                          (g.obj is None or victim.obj is None or g.obj == victim.obj)
                          for g in detected)
                pv = per_verb.setdefault(victim.verb, {"trials": 0, "tp": 0})
                pv["trials"] += 1
                if hit:
                    tp += 1
                    pv["tp"] += 1
                    if is_conf:
                        conf_detected += 1
                else:
                    fn += 1
                # extra GAPs beyond the deleted one = false positives
                for g in detected:
                    if not (g.verb == victim.verb and
                            (g.obj is None or victim.obj is None or g.obj == victim.obj)):
                        fp += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return GapEvalResult(pos_trials, tp, fn, neg_trials, tn, fp,
                         conf_trials, conf_detected,
                         precision, recall, f1, specificity, per_verb)
