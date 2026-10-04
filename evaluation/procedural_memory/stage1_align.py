"""Stage 1 deterministic retrieval arms for the TRACE-PM kill test.

NO LLM, NO database. Reads the on-disk benchmark files
(``queries.json`` + ``trajectories.json``), parses raw ALFWorld action text
into typed (verb, object_type, target_type) tuples, and ranks trajectories
against each query by one of two deterministic scorers:

    * ``align``  -- ORDERED weighted-Levenshtein alignment of the query's typed
                    sub-goals against each trajectory's typed action sequence.
                    This is the Stage-1 ordered-alignment arm.
    * ``bag``    -- ORDER-FREE typed bag-of-sub-goals overlap. Identical token
                    content to ``align`` but with order information removed.
                    This is the Stage-1 control that isolates "does ordering
                    actually matter?" from mere lexical normalization.

Both arms produce a ranked trajectory-ID list, which the Stage-1 runner scores
with the SAME IR metrics (P@1/P@5/NDCG@10/MAP) used by every other arm, so the
comparison is apples-to-apples. The frozen typed vocabulary is derived
mechanically from the trajectory corpus on disk (see ``build_vocab``), not from
Neo4j and not hand-written.

This module is import-safe and has zero MemMachine or network dependencies, so
it can be unit-tested offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Typed representation
# ---------------------------------------------------------------------------

# ALFWorld action verbs, mapped to a canonical verb label. The raw corpus uses
# a small fixed set of surface forms; this is the full observed set.
_VERB_CANON = {
    "go to": "go_to",
    "take": "take",
    "put": "put",
    "open": "open",
    "close": "close",
    "clean": "clean",
    "heat": "heat",
    "cool": "cool",
    "use": "use",
    "examine": "examine",
    "look": "examine",
    "inventory": "examine",
}

# Query keyword verbs (from extracted_keywords.verbs) mapped onto canonical
# action verbs so a query sub-goal can match a trajectory action.
_QUERY_VERB_CANON = {
    "place": "put",
    "put": "put",
    "move": "put",
    "store": "put",
    "take": "take",
    "pick": "take",
    "pickup": "take",
    "get": "take",
    "grab": "take",
    "clean": "clean",
    "wash": "clean",
    "rinse": "clean",
    "heat": "heat",
    "warm": "heat",
    "microwave": "heat",
    "cook": "heat",
    "cool": "cool",
    "chill": "cool",
    "examine": "examine",
    "look": "examine",
    "inspect": "examine",
    "find": "go_to",
    "open": "open",
    "close": "close",
    "use": "use",
}

# Trailing instance number is dropped so "soapbar 2" -> "soapbar".
_INSTANCE_RE = re.compile(r"\s+\d+$")


def normalize_object(raw: str | None) -> str | None:
    """Strip the instance number so an object becomes a bare type."""
    if not raw:
        return None
    raw = raw.strip().lower().replace(" ", "")
    # keep already-typed keywords ("soapbar") and strip "soapbar2" style too
    raw = re.sub(r"\d+$", "", raw)
    return raw or None


def _norm_instance_phrase(phrase: str) -> str | None:
    """'soapbar 2' / 'sinkbasin 1' -> 'soapbar' / 'sinkbasin'."""
    phrase = phrase.strip().lower()
    phrase = _INSTANCE_RE.sub("", phrase)
    return phrase.replace(" ", "") or None


@dataclass(frozen=True)
class TypedStep:
    """A typed (verb, object_type, target_type) tuple."""

    verb: str
    obj: str | None = None
    target: str | None = None


# ---------------------------------------------------------------------------
# Parsing raw ALFWorld action text -> TypedStep
# ---------------------------------------------------------------------------

# "take soapbar 1 from countertop 1"
_RE_TAKE = re.compile(r"^take\s+(.+?)\s+from\s+(.+)$")
# "put soapbar 1 in/on cabinet 2"
_RE_PUT = re.compile(r"^put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)$")
# "clean soapbar 1 with sinkbasin 1"  /  "heat apple 1 with microwave 1"
_RE_WITH = re.compile(r"^(clean|heat|cool)\s+(.+?)\s+with\s+(.+)$")
# "go to cabinet 2"  /  "open drawer 1" / "use desklamp 1" / "examine mug 1"
_RE_SIMPLE = re.compile(r"^(go to|open|close|use|examine)\s+(.+)$")


def parse_action(action: str) -> TypedStep | None:
    """Turn one raw ALFWorld action string into a TypedStep, or None."""
    a = action.strip().lower()
    if m := _RE_TAKE.match(a):
        return TypedStep("take", _norm_instance_phrase(m.group(1)),
                         _norm_instance_phrase(m.group(2)))
    if m := _RE_PUT.match(a):
        return TypedStep("put", _norm_instance_phrase(m.group(1)),
                         _norm_instance_phrase(m.group(2)))
    if m := _RE_WITH.match(a):
        return TypedStep(m.group(1), _norm_instance_phrase(m.group(2)),
                         _norm_instance_phrase(m.group(3)))
    if m := _RE_SIMPLE.match(a):
        verb = _VERB_CANON.get(m.group(1), m.group(1).replace(" ", "_"))
        return TypedStep(verb, _norm_instance_phrase(m.group(2)), None)
    return None


def trajectory_to_typed(traj: dict) -> list[TypedStep]:
    """Full typed action sequence for one trajectory."""
    steps: list[TypedStep] = []
    for pair in traj.get("state_action_pairs", []):
        ts = parse_action(pair.get("action", ""))
        if ts is not None:
            steps.append(ts)
    return steps


# ---------------------------------------------------------------------------
# Query -> ordered typed sub-goals (deterministic, from extracted_keywords)
# ---------------------------------------------------------------------------


def query_to_subgoals(query: dict) -> list[TypedStep]:
    """Ordered typed sub-goals for a query, from its extracted_keywords.

    Deterministic Stage-1 decomposition: no LLM. We pair each keyword verb with
    the keyword objects in text order. ALFWorld queries name the object before
    the receptacle, so (object, receptacle) ordering is taken as given. This is
    intentionally the SAME decomposition for both the align and bag arms, so the
    only difference between them is whether order is used downstream.
    """
    kw = query.get("extracted_keywords", {}) or {}
    objs = [normalize_object(o) for o in kw.get("objects", []) if o]
    objs = [o for o in objs if o]
    verbs_raw = kw.get("verbs", []) or []
    # canonicalize + dedupe verbs preserving order
    seen: set[str] = set()
    verbs: list[str] = []
    for v in verbs_raw:
        cv = _QUERY_VERB_CANON.get(v.strip().lower())
        if cv and cv not in seen:
            seen.add(cv)
            verbs.append(cv)
    if not verbs:
        verbs = ["put"]  # placement is the benchmark's dominant intent

    obj = objs[0] if objs else None
    target = objs[1] if len(objs) > 1 else None

    subgoals: list[TypedStep] = []
    for v in verbs:
        if v == "put":
            subgoals.append(TypedStep("put", obj, target))
        elif v in ("clean", "heat", "cool"):
            subgoals.append(TypedStep(v, obj, None))
        elif v == "take":
            subgoals.append(TypedStep("take", obj, None))
        else:
            subgoals.append(TypedStep(v, obj, None))
    return subgoals


# ---------------------------------------------------------------------------
# Pair cost + ordered alignment (weighted Levenshtein)
# ---------------------------------------------------------------------------

# Lexicographic-ish weights: effect/verb dominates, then object, then target.
_W_VERB = 3.0
_W_OBJ = 2.0
_W_TARGET = 1.0
_GAP_COST = 4.0    # a query sub-goal with no matching action (missing step)
_EXTRA_COST = 0.3  # a trajectory action not required by the query (cheap)


def pair_cost(q: TypedStep, a: TypedStep) -> float:
    """Substitution cost between a query sub-goal and a trajectory action."""
    cost = 0.0
    if q.verb != a.verb:
        cost += _W_VERB
    if q.obj is not None and a.obj is not None and q.obj != a.obj:
        cost += _W_OBJ
    if q.target is not None and a.target is not None and q.target != a.target:
        cost += _W_TARGET
    return cost


def ordered_align_cost(subgoals: list[TypedStep], actions: list[TypedStep]) -> float:
    """Weighted-Levenshtein alignment cost of sub-goals against actions.

    Only query->action GAPs and substitutions are penalized meaningfully;
    trajectory-only EXTRA actions are cheap (setup/navigation is expected).
    Normalized by the number of sub-goals so longer trajectories do not win
    automatically.
    """
    m, n = len(subgoals), len(actions)
    if m == 0:
        return 0.0
    # dp[i][j] = min cost aligning first i sub-goals with first j actions
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
            gap = dp[i - 1][j] + _GAP_COST          # sub-goal unmatched
            extra = dp[i][j - 1] + _EXTRA_COST       # action unneeded
            dp[i][j] = min(sub, gap, extra)
    return dp[m][n] / m


def bag_cost(subgoals: list[TypedStep], actions: list[TypedStep]) -> float:
    """ORDER-FREE control: best per-sub-goal match ignoring sequence.

    For each sub-goal, take the cheapest pair_cost against ANY action
    (multiset, each action usable once), add GAP for anything unmatchable.
    Same tokens as ordered_align_cost, order information removed.
    """
    m = len(subgoals)
    if m == 0:
        return 0.0
    remaining = list(actions)
    total = 0.0
    for q in subgoals:
        best_cost = _GAP_COST
        best_idx = -1
        for idx, a in enumerate(remaining):
            c = pair_cost(q, a)
            if c < best_cost:
                best_cost = c
                best_idx = idx
        total += best_cost
        if best_idx >= 0:
            remaining.pop(best_idx)
    return total / m


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


@dataclass
class RankedResult:
    ranked_ids: list[str]
    subgoals: list[TypedStep]
    scorer: str
    per_traj_cost: dict[str, float] = field(default_factory=dict)


def rank(
    query: dict,
    typed_trajs: dict[str, list[TypedStep]],
    scorer: str = "align",
    top_k: int = 10,
    subgoals: list[TypedStep] | None = None,
) -> RankedResult:
    """Rank all trajectories for one query by the chosen deterministic scorer.

    ``subgoals`` may be supplied (e.g. from an LLM decomposer) to override the
    default keyword decomposition; the scoring is otherwise identical, so the
    ONLY variable that changes between a keyword run and an LLM run is the
    sub-goal list.
    """
    if subgoals is None:
        subgoals = query_to_subgoals(query)
    fn = ordered_align_cost if scorer == "align" else bag_cost
    costs = {tid: fn(subgoals, steps) for tid, steps in typed_trajs.items()}
    ranked = sorted(costs, key=lambda t: (costs[t], t))[:top_k]
    return RankedResult(ranked, subgoals, scorer, {t: costs[t] for t in ranked})


# ---------------------------------------------------------------------------
# Vocabulary (frozen, derived mechanically from the corpus on disk)
# ---------------------------------------------------------------------------


def build_vocab(typed_trajs: dict[str, list[TypedStep]]) -> dict:
    """Frozen typed vocabulary: observed verbs, object types, target types.

    Derived purely from the on-disk trajectory corpus so it is reproducible and
    requires no Neo4j. Written to disk by the runner so a reviewer can audit
    exactly which vocabulary the alignment arm used.
    """
    verbs: set[str] = set()
    objs: set[str] = set()
    targets: set[str] = set()
    for steps in typed_trajs.values():
        for s in steps:
            verbs.add(s.verb)
            if s.obj:
                objs.add(s.obj)
            if s.target:
                targets.add(s.target)
    return {
        "verbs": sorted(verbs),
        "object_types": sorted(objs),
        "target_types": sorted(targets),
        "counts": {
            "verbs": len(verbs),
            "object_types": len(objs),
            "target_types": len(targets),
        },
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_queries(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["queries"] if isinstance(data, dict) else data


def load_typed_trajectories(path: str | Path) -> dict[str, list[TypedStep]]:
    data = json.loads(Path(path).read_text())
    trajs = data["trajectories"] if isinstance(data, dict) else data
    out: dict[str, list[TypedStep]] = {}
    for t in trajs:
        tid = t.get("task_instance_id") or t.get("trajectory_id")
        if tid:
            out[str(tid)] = trajectory_to_typed(t)
    return out


def relevant_ids(query: dict, threshold: float = 6.0) -> set[str]:
    raw = query.get("relevant_trajectories", [])
    if raw and isinstance(raw[0], dict):
        return {r["trajectory_id"] for r in raw
                if r.get("relevance_score", 0) >= threshold}
    return set(raw) if raw else set(query.get("relevant_ids", []))
