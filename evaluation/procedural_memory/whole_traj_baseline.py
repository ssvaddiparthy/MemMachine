"""E4-lite: Memp-style whole-trajectory baseline retriever (offline, no DB).

The handoff mandates a whole-trajectory baseline (Memp): retrieve whole past
trajectories by similarity and hand their raw steps to the answer model, with
NO decomposition, NO gap analysis, NO stitching. This module provides that
baseline reading ``trajectories.json`` directly -- no Neo4j, no pgvector -- so it
can be compared head-to-head against the grounded gap-aware retriever through
the SAME answer prompt + judge the baseline episodic arm uses.

Similarity here is a cheap, dependency-free lexical overlap (token Jaccard over
the task description + object/verb keywords). It is intentionally simple: the
point of E4-lite is the DOWNSTREAM answer quality of whole-trajectory reuse vs
gap-aware retrieval, not embedding sophistication. A reviewer gets a clean,
reproducible baseline; swap in embeddings later if needed.

Usage: imported by proced_mem_bench_search.py as --test-target whole_traj.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOK.findall(text.lower()) if len(t) >= 3}


@dataclass
class WholeTrajResult:
    mode: str = "whole_traj"
    ranked_trajectory_ids: list[str] = field(default_factory=list)
    memory_text: str = ""

    def to_memory_text(self) -> str:
        return self.memory_text


class WholeTrajectoryRetriever:
    """Memp-style: top-k whole trajectories by lexical similarity, raw steps in."""

    def __init__(self, trajectories_path: str | Path):
        data = json.loads(Path(trajectories_path).read_text())
        trajs = data["trajectories"] if isinstance(data, dict) else data
        self._trajs = []
        for t in trajs:
            tid = t.get("task_instance_id") or t.get("trajectory_id")
            desc = t.get("task_description", "")
            actions = [p.get("action", "") for p in t.get("state_action_pairs", [])]
            toks = _tokens(desc + " " + " ".join(actions))
            self._trajs.append((str(tid), desc, actions, toks))

    def retrieve(self, query_text: str, top_k: int = 3) -> WholeTrajResult:
        q = _tokens(query_text)
        scored = []
        for tid, desc, actions, toks in self._trajs:
            if not toks:
                continue
            jac = len(q & toks) / len(q | toks) if (q | toks) else 0.0
            scored.append((jac, tid, desc, actions))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[:top_k]
        lines = []
        for jac, tid, desc, actions in top:
            lines.append(f"[trajectory {tid}] goal: {desc}")
            lines.append("  " + " -> ".join(actions))
        res = WholeTrajResult()
        res.ranked_trajectory_ids = [tid for _, tid, _, _ in top]
        res.memory_text = "\n".join(lines) or "(no relevant procedure found)"
        return res
