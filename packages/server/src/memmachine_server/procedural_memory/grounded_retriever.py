"""Grounded procedural retriever -- returns REAL trajectory steps for a query,
plus a gap-aware diagnosis of what the retrieved example is MISSING.

Algorithm (per query):
1. Decompose: LLM splits the query into ORDERED sub-goals, each
   {tool, object, target}. e.g. "clean a soap bar and put it in the cabinet"
   -> [clean(soapbar), put(soapbar, cabinet)].
2. Match: scan every stored trajectory (ProcAction nodes, in step order) for
   actions satisfying each sub-goal, in order.
3. Rank:
   a. Trajectories covering ALL sub-goals in order win (anchored case) --
      return that trajectory's real steps from the start of the object's
      handling to the last sub-goal. Preconditions and order come for free.
   b. Otherwise STITCH: per sub-goal, take the best real segment from any
      trajectory, concatenated in sub-goal order.
4. Serialize grounded steps + provenance (trajectory ids) for the answer LLM.

Matching is on normalized object TYPES ("soapbar 2" -> "soapbar",
"soap bar" -> "soapbar", "bottles" -> "bottle"), so it generalizes beyond
the exact instance ids in a trajectory.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore
from memmachine_server.procedural_memory.llm_utils import llm_complete, parse_json_object

logger = logging.getLogger(__name__)

# Actions that change world state (segment boundaries) / that express a goal
_MANIP = {"pick_up", "clean", "heat", "cool", "put", "slice"}
_GOAL_TOOLS = {"clean", "heat", "cool", "put", "use", "examine", "slice", "toggle"}

_DECOMPOSE_PROMPT = """\
Break the task into the ordered sub-goals an agent must achieve.
Use only these tool names: {tools}
For each sub-goal give the tool, the main object it acts on, and the
target/receptacle/appliance if any (else null). Repeat a sub-goal if the task
asks for multiple items (e.g. "two apples" -> two put sub-goals).
Only list goal-level actions (clean/heat/cool/put/use/examine/...), not
navigation or pick-up; those are filled in from memory.

Name objects and targets using the vocabulary seen in memory when the task
refers to the same thing (e.g. "counter" -> "countertop", "phone" ->
"cellphone"); keep the task's word if nothing matches.
Known objects: {objects}
Known targets: {targets}

Task: {task}

Respond with JSON only:
{{"subgoals": [{{"tool": "...", "object": "...", "target": "..." }}]}}
"""


def _norm(text: str | None) -> str:
    """Normalize an object mention to its type: letters only, singular."""
    if not text:
        return ""
    t = re.sub(r"[^a-z]", "", text.lower())
    return t[:-1] if t.endswith("s") and len(t) > 3 else t


# ── edit-script cost model (ported from the proven Stage-2 gap aligner) ─────
# A weighted-Levenshtein alignment of the query's ordered sub-goals against a
# trajectory's actions, backtraced into a labelled script. This is the active
# ingredient the plain whole-trajectory baseline structurally lacks: it names
# the required sub-goals the trajectory does NOT cover (GAP), so the agent is
# told what it must supply itself. See evaluation/procedural_memory § E2/E4.
_W_VERB = 3.0
_W_OBJ = 2.0
_W_TARGET = 1.0
_GAP_COST = 4.0     # a required sub-goal with no matching action (MISSING step)
_EXTRA_COST = 0.3   # a trajectory action not required by the query (setup/noise)


def _goal_fields(goal: dict[str, Any]) -> tuple[str, str, str]:
    """A decomposed sub-goal dict -> (verb, object_type, target_type)."""
    return (
        str(goal.get("tool") or ""),
        _norm(goal.get("object")),
        _norm(goal.get("target")),
    )


def _pair_cost(goal: dict[str, Any], step: "Step") -> float:
    """Substitution cost between a query sub-goal and a trajectory action."""
    gv, go, gt = _goal_fields(goal)
    cost = 0.0
    if gv != step.tool:
        cost += _W_VERB
    if go and step.object_type and go != step.object_type:
        cost += _W_OBJ
    if gt and step.target_type and gt != step.target_type:
        cost += _W_TARGET
    return cost


@dataclass
class EditOp:
    """One aligned edit between a required sub-goal and a trajectory action.

    op is one of:
      COVERED -- the sub-goal matched an action (verb+object ok, target ok)
      REBIND  -- right verb+object, wrong target instance (repairable binding)
      GAP     -- a required sub-goal with NO matching action (the MISSING step)
      EXTRA   -- a trajectory action not required by the query (setup/noise)
    """

    op: str
    subgoal: dict[str, Any] | None  # query side (None for EXTRA)
    action: "Step | None"           # trajectory side (None for GAP)

    def render(self) -> str:
        if self.op == "GAP" and self.subgoal:
            v, o, t = _goal_fields(self.subgoal)
            tail = f" {o}" + (f" -> {t}" if t else "")
            return f"MISSING: {v}{tail} (you must do this yourself)"
        if self.op == "REBIND" and self.subgoal and self.action:
            _, o, t = _goal_fields(self.subgoal)
            return f"REBIND: {o or self.action.object_type} -> {t} (example used a different target)"
        return ""


def build_edit_script(goals: list[dict[str, Any]], actions: list["Step"]) -> list[EditOp]:
    """Backtrace the alignment DP into a labelled COVERED/REBIND/GAP/EXTRA script."""
    m, n = len(goals), len(actions)
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + _EXTRA_COST
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + _GAP_COST
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = dp[i - 1][j - 1] + _pair_cost(goals[i - 1], actions[j - 1])
            gap = dp[i - 1][j] + _GAP_COST
            extra = dp[i][j - 1] + _EXTRA_COST
            dp[i][j] = min(sub, gap, extra)

    ops: list[EditOp] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            pc = _pair_cost(goals[i - 1], actions[j - 1])
            if abs(dp[i][j] - (dp[i - 1][j - 1] + pc)) < 1e-9:
                g, a = goals[i - 1], actions[j - 1]
                gv, go, gt = _goal_fields(g)
                if gv == a.tool and (not go or not a.object_type or go == a.object_type):
                    if gt and a.target_type and gt != a.target_type:
                        op = "REBIND"
                    else:
                        op = "COVERED"
                else:
                    op = "GAP"  # substituted but verb/obj mismatch: effect unmet
                ops.append(EditOp(op, g, a))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + _GAP_COST)) < 1e-9:
            ops.append(EditOp("GAP", goals[i - 1], None))
            i -= 1
            continue
        ops.append(EditOp("EXTRA", None, actions[j - 1]))
        j -= 1
    ops.reverse()
    return ops


def script_gaps(script: list[EditOp]) -> list[dict[str, Any]]:
    """The sub-goals the aligner diagnosed as MISSING."""
    return [op.subgoal for op in script if op.op == "GAP" and op.subgoal]


@dataclass
class Step:
    trajectory_id: str
    index: int
    tool: str
    params: dict[str, str]

    @property
    def object_type(self) -> str:
        return _norm(self.params.get("object"))

    @property
    def target_type(self) -> str:
        p = self.params
        return _norm(p.get("receptacle") or p.get("target") or p.get("with"))

    def render(self) -> str:
        p = self.params
        obj, tgt = p.get("object"), p.get("receptacle") or p.get("target")
        if self.tool == "go_to":
            return f"go to {tgt}"
        if self.tool == "pick_up":
            return f"take {obj} from {p.get('from', '?')}"
        if self.tool == "put":
            return f"put {obj} in/on {tgt}"
        if self.tool in {"clean", "heat", "cool"}:
            return f"{self.tool} {obj} with {p.get('with', '?')}"
        if obj and tgt:
            return f"{self.tool} {obj} {tgt}"
        return f"{self.tool.replace('_', ' ')} {obj or tgt or ''}".strip()


@dataclass
class GroundedProcedure:
    subgoals: list[dict[str, Any]]
    segments: list[tuple[str, list[Step]]] = field(default_factory=list)
    mode: str = "none"  # anchored | stitched | none
    match_level: int = -1  # 0 exact, 1 containment, 2 object-only
    precond_missing: int = 0  # 0 = every segment has its preconditions
    ranked_trajectory_ids: list[str] = field(default_factory=list)
    edit_script: list[EditOp] = field(default_factory=list)

    @property
    def gaps(self) -> list[dict[str, Any]]:
        """Required sub-goals the retrieved example does NOT cover."""
        return script_gaps(self.edit_script)

    def to_memory_text(self) -> str:
        if not self.segments:
            return ""
        lines = [f"Retrieval mode: {self.mode}"]
        if self.match_level == 2:
            lines.append("Note: closest match found acts on the same object but a "
                         "different target/receptacle -- adapt the final placement.")
        for i, (label, steps) in enumerate(self.segments, 1):
            src = steps[0].trajectory_id if steps else "?"
            lines.append(f"[{i}] {label} (from past trajectory {src}):")
            lines.extend(f"    {j}. {s.render()}" for j, s in enumerate(steps, 1))
        # Gap-aware diagnosis: the active ingredient. Tell the agent exactly
        # which required sub-goals this example does NOT cover, so it supplies
        # them itself instead of blindly imitating the trajectory.
        gaps = self.gaps
        rebinds = [op for op in self.edit_script if op.op == "REBIND"]
        if gaps or rebinds:
            lines.append("")
            lines.append("Gap diagnosis (this example is incomplete for your task):")
            for op in self.edit_script:
                if op.op in ("GAP", "REBIND"):
                    r = op.render()
                    if r:
                        lines.append(f"  - {r}")
        elif self.subgoals:
            lines.append("")
            lines.append("Gap diagnosis: the example covers all required sub-goals.")
        return "\n".join(lines)


class GroundedRetriever:
    """Trajectory-anchored retrieval over the procedural graph's real actions."""

    def __init__(self, graph_store: ProceduralGraphStore, llm: Any) -> None:
        self._store = graph_store
        self._llm = llm
        self._trajectories: dict[str, list[Step]] | None = None

    # ── data loading ────────────────────────────────────────────────────
    async def _load(self) -> dict[str, list[Step]]:
        if self._trajectories is not None:
            return self._trajectories
        async with self._store._driver.session(database=self._store._database) as s:
            result = await s.run(
                """
                MATCH (a:ProcAction)
                WHERE coalesce(a.success, true)
                RETURN a.trajectory_id AS tid, a.step_id AS step,
                       a.tool_name AS tool, a.parameters AS params
                """
            )
            rows = [r async for r in result]
        trajs: dict[str, list[Step]] = {}
        for r in rows:
            params = r["params"]
            if isinstance(params, str):
                try:
                    params = ast.literal_eval(params)
                except (ValueError, SyntaxError):
                    params = {}
            trajs.setdefault(r["tid"], []).append(
                Step(r["tid"], int(r["step"] or 0), r["tool"], params or {})
            )
        for steps in trajs.values():
            steps.sort(key=lambda s: s.index)
        self._trajectories = trajs
        logger.info("GroundedRetriever loaded %d trajectories", len(trajs))
        return trajs

    # ── step 1: decompose ───────────────────────────────────────────────
    @staticmethod
    def _vocab(trajs: dict[str, list[Step]], min_count: int = 3) -> tuple[list[str], list[str]]:
        """Object/target types seen in memory (rare ones are parse artifacts)."""
        from collections import Counter
        objs = Counter(s.object_type for st in trajs.values() for s in st if s.object_type)
        tgts = Counter(s.target_type for st in trajs.values() for s in st if s.target_type)
        keep = lambda c: sorted(k for k, n in c.items() if n >= min_count)  # noqa: E731
        return keep(objs), keep(tgts)

    async def decompose(self, query: str, trajs: dict[str, list[Step]]) -> list[dict[str, Any]]:
        tools = {s.tool for st in trajs.values() for s in st}
        objs, tgts = self._vocab(trajs)
        prompt = _DECOMPOSE_PROMPT.format(
            task=query, tools=", ".join(sorted(tools)),
            objects=", ".join(objs), targets=", ".join(tgts),
        )
        raw = await llm_complete(self._llm, prompt)
        goals = parse_json_object(raw).get("subgoals", [])
        return [g for g in goals if g.get("tool")]

    # ── step 2: matching ────────────────────────────────────────────────
    @staticmethod
    def _type_eq(want: str, have: str, level: int) -> bool:
        """level 0: exact type; level>=1: also containment ("bottle" ~ "soapbottle",
        "counter" ~ "countertop")."""
        if want == have:
            return True
        return level >= 1 and min(len(want), len(have)) >= 3 and (want in have or have in want)

    def _matches(self, step: Step, goal: dict[str, Any], level: int) -> bool:
        if step.tool != goal.get("tool"):
            return False
        obj, tgt = _norm(goal.get("object")), _norm(goal.get("target"))
        if obj and not self._type_eq(obj, step.object_type, level):
            return False
        if level < 2 and tgt and step.target_type and not self._type_eq(tgt, step.target_type, level):
            return False
        return True

    def _match_in_order(self, steps: list[Step], goals: list[dict], level: int) -> list[int | None]:
        """Step index matched per goal, in temporal order (greedy, earliest).
        A goal with no match is None so later goals still count."""
        idxs: list[int | None] = []
        start = 0
        for g in goals:
            hit = next((i for i in range(start, len(steps)) if self._matches(steps[i], g, level)), None)
            idxs.append(hit)
            if hit is not None:
                start = hit + 1
        return idxs

    @staticmethod
    def _continuation_start(steps: list[Step], anchor: int, floor: int) -> int:
        """Object already in hand: start at the navigation leading to the anchor,
        stopping at the previous manipulation."""
        start = anchor
        for i in range(anchor - 1, floor - 1, -1):
            if steps[i].tool in _MANIP:
                break
            start = i
        return start

    def _precond_score(self, seg: list[Step], goal: dict[str, Any], level: int) -> int:
        """0 = complete (navigates first, and picks up the object if any);
        higher = missing preconditions. Ranked before length."""
        missing = int(seg[0].tool != "go_to")
        obj = _norm(goal.get("object"))
        if obj and goal.get("tool") != "pick_up" and not any(
            s.tool == "pick_up" and self._type_eq(obj, s.object_type, level) for s in seg
        ):
            missing += 2
        return missing

    def _segment_start(self, steps: list[Step], anchor: int, obj: str, level: int) -> int:
        """Back up from the anchor to where handling of `obj` began:
        the go_to preceding the pick_up of that object (keeps preconditions)."""
        start = anchor
        for i in range(anchor - 1, -1, -1):
            if steps[i].tool == "pick_up" and obj and self._type_eq(obj, steps[i].object_type, level):
                start = i - 1 if i > 0 and steps[i - 1].tool == "go_to" else i
                break
        else:
            if anchor > 0 and steps[anchor - 1].tool == "go_to":
                start = anchor - 1
        return max(start, 0)

    # ── step 3: retrieve ────────────────────────────────────────────────
    async def retrieve(self, query: str, top_k: int = 10) -> GroundedProcedure:
        trajs = await self._load()
        goals = await self.decompose(query, trajs)
        proc = GroundedProcedure(subgoals=goals)
        if not goals:
            return proc
        # Strict first. Accept the first level whose result is complete
        # (preconditions present); otherwise fall back to the first level that
        # found anything at all.
        fallback = None
        for level in (0, 1, 2):
            cand = GroundedProcedure(subgoals=goals)
            self._retrieve_at(cand, query, goals, trajs, level, top_k)
            if cand.mode == "none":
                continue
            cand.match_level = level
            self._diagnose(cand, goals)
            if cand.precond_missing == 0:
                return cand
            fallback = fallback or cand
        return fallback or proc

    @staticmethod
    def _diagnose(proc: GroundedProcedure, goals: list[dict[str, Any]]) -> None:
        """Align the query sub-goals against the retrieved actions and attach
        the COVERED/REBIND/GAP/EXTRA edit script (the gap-aware diagnosis)."""
        actions: list[Step] = []
        for _label, steps in proc.segments:
            actions.extend(steps)
        if goals and actions:
            proc.edit_script = build_edit_script(goals, actions)

    def _retrieve_at(self, proc: GroundedProcedure, query: str, goals: list[dict],
                     trajs: dict[str, list[Step]], level: int, top_k: int) -> None:
        # Score: (#goals matched in order, -#extra goal-level actions in the
        # segment, -segment length). Prefers trajectories whose task SHAPE
        # matches (a plain put over a clean-and-put), then fewer detours.
        scored = []
        for tid, steps in trajs.items():
            idxs = self._match_in_order(steps, goals, level)
            hits = [i for i in idxs if i is not None]
            if not hits:
                continue
            g0 = goals[idxs.index(hits[0])]
            start = self._segment_start(steps, hits[0], _norm(g0.get("object")), level)
            seg = steps[start : hits[-1] + 1]
            extras = sum(1 for s in seg if s.tool in _GOAL_TOOLS) - len(hits)
            missing = self._precond_score(seg, g0, level)
            scored.append((len(hits), -missing, -extras, -len(seg), tid, start, hits[-1]))
        scored.sort(key=lambda x: x[:4], reverse=True)
        proc.ranked_trajectory_ids = [x[4] for x in scored[:top_k]]
        proc.segments = []
        if not scored:
            proc.mode = "none"
            return

        cov, neg_missing, _, _, tid, start, end = scored[0]
        if cov == len(goals):
            proc.mode = "anchored"
            proc.precond_missing = -neg_missing
            proc.segments = [(query, trajs[tid][start : end + 1])]
            return

        proc.mode = "stitched"
        prev_obj = None
        for g in goals:
            obj = _norm(g.get("object"))
            in_hand = bool(obj) and obj == prev_obj
            best, best_key = None, None
            for steps in trajs.values():
                hit = next((i for i, s in enumerate(steps) if self._matches(s, g, level)), None)
                if hit is None:
                    continue
                s0 = (self._continuation_start(steps, hit, 0) if in_hand
                      else self._segment_start(steps, hit, obj, level))
                seg = steps[s0 : hit + 1]
                extras = sum(1 for s in seg if s.tool in _GOAL_TOOLS) - 1
                missing = 0 if in_hand else self._precond_score(seg, g, level)
                key = (missing, seg[0].tool != "go_to", extras, len(seg))
                if best_key is None or key < best_key:
                    best, best_key = seg, key
            if best:
                proc.precond_missing += best_key[0]
                label = f"{g['tool']} {g.get('object') or ''} {g.get('target') or ''}".strip()
                proc.segments.append((label, best))
                src = best[0].trajectory_id
                if src not in proc.ranked_trajectory_ids:
                    proc.ranked_trajectory_ids.append(src)
            prev_obj = obj or prev_obj
        if not proc.segments:
            proc.mode = "none"
