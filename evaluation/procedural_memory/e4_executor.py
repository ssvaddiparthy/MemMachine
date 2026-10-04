"""E4-full: ALFWorld execution harness.

Drives an executor agent (MemMachine answer LLM) through real ALFWorld TextWorld
games, with a memory payload prepended to the ReAct prompt, and records true
task-success / invalid-action / step metrics. This is the only true
task-success signal in the project (vs the retrieval/judge proxies in E1-E4-lite).

Verified against alfworld 0.4.2:
  env = AlfredTWEnv(config, train_eval='eval_in_distribution').init_env(batch_size=1)
  obs, info = env.reset()
  obs, scores, dones, infos = env.step([action])
  info['admissible_commands'][0] -> list of legal actions (used to score "invalid")
  dones[0] True + scores[0] >= 1 (won) marks success.

Arms (select with --arm):
  none       -- no memory
  whole_traj -- Memp-style whole-trajectory lexical retrieval (E4-lite retriever)
  (grounded / grounded_gap can be added once the Neo4j retriever is wired here)

MUST run under the conda env python that has alfworld + memmachine_server:
  /opt/anaconda3/envs/memmachine-eval-replicate/bin/python3
with ALFWORLD_DATA exported and OPENAI_API_KEY set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "packages" / "common" / "src",
          REPO_ROOT / "packages" / "server" / "src"):
    if str(p) not in sys.path:
        sys.path.append(str(p))


def build_config(num_eval_games: int) -> dict:
    """Complete AlfredTWEnv config (all keys init_env + collect_game_files read)."""
    data = os.path.expandvars("$ALFWORLD_DATA")
    return {
        "dataset": {
            "data_path": f"{data}/json_2.1.1/train",
            "eval_id_data_path": f"{data}/json_2.1.1/valid_seen",
            "eval_ood_data_path": f"{data}/json_2.1.1/valid_unseen",
            "num_train_games": 0,
            "num_eval_games": num_eval_games,
        },
        "env": {
            "task_types": [1, 2, 3, 4, 5, 6],
            "goal_desc_human_anns_prob": 0.0,
            "domain_randomization": False,
            "expert_type": "handcoded",
            "expert_timeout_steps": 150,
            "expert_timeout_cmds": 10,
            "regen_game_files": False,
        },
        "general": {"random_seed": 42, "use_cuda": False,
                    "training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
        "rl": {"training": {"max_nb_steps_per_episode": 50}},
        "logic": {"domain": f"{data}/logic/alfred.pddl",
                  "grammar": f"{data}/logic/alfred.twl2"},
    }


REACT_PROMPT = """You are an agent completing a household task in a TextWorld
kitchen/room. You act by choosing ONE admissible command each step. Think briefly,
then act. Format EXACTLY:
THINK: <one short sentence of reasoning>
ACT: <one command, copied verbatim from the admissible list>

Worked example (a different task):
Task: put a clean apple in the fridge.
THINK: I need to find the apple first; it is often on a countertop.
ACT: go to countertop 1
(then, after seeing the apple) ACT: take apple 1 from countertop 1
(then) ACT: go to sinkbasin 1
(then) ACT: clean apple 1 with sinkbasin 1
(then) ACT: go to fridge 1
(then) ACT: open fridge 1
(then) ACT: put apple 1 in/on fridge 1

Key rules: to clean -> go to a sinkbasin then 'clean X with sinkbasin'; to heat ->
microwave; to cool -> fridge. You must be AT a location and HOLDING an object before
using it. Open receptacles (fridge/cabinet/drawer) before putting things in them.
Do not repeat an action that did not change anything.

{memory_block}
Task: {task}

What you have done so far:
{history}

Current observation:
{obs}

Admissible commands:
{admissible}

Respond with THINK then ACT:"""


def _gap_aware_block(task, whole_retriever, typed_trajs, subgoals=None) -> str:
    """The NOVEL arm: same anchor trajectory as whole_traj, PLUS an explicit
    edit-script diagnosis — which required sub-goals the anchor COVERS and which
    are MISSING (GAP). This is the honest, provenance-tracked payload that plain
    whole-trajectory retrieval structurally cannot produce.

    ``subgoals`` should be the LLM-decomposed typed sub-goals for the task (the
    keyword decomposer is too weak on free-text ALFWorld goals and would leave
    the diagnosis empty, collapsing gap_aware into whole_traj). The caller
    computes them once per game and passes them in.
    """
    from evaluation.procedural_memory.stage1_align import query_to_subgoals
    from evaluation.procedural_memory.stage2_gap import align_edit_script, gaps

    res = whole_retriever.retrieve(task, top_k=1)
    anchor_text = res.to_memory_text()
    anchor_id = res.ranked_trajectory_ids[0] if res.ranked_trajectory_ids else None

    if subgoals is None:
        subgoals = query_to_subgoals({"query_text": task, "extracted_keywords": {}})
    anchor_steps = typed_trajs.get(anchor_id, []) if anchor_id else []
    script = align_edit_script(subgoals, anchor_steps) if (subgoals and anchor_steps) else []
    missing = gaps(script)

    def _fmt(s):
        return f"{s.verb}({s.obj or '-'}" + (f",{s.target}" if s.target else "") + ")"

    lines = ["<relevant past procedure>", anchor_text]
    if subgoals:
        covered = [f"{op.op}:{_fmt(op.subgoal)}" for op in script
                   if op.subgoal and op.op in ("COVERED", "REBIND")]
        lines.append("<coverage diagnosis>")
        lines.append("required sub-goals: " + ", ".join(_fmt(s) for s in subgoals))
        if covered:
            lines.append("covered by the example above: " + ", ".join(covered))
        if missing:
            lines.append("MISSING from the example (you must do these yourself, "
                         "they are NOT in memory): " + ", ".join(_fmt(s) for s in missing))
        else:
            lines.append("the example covers all required sub-goals.")
        lines.append("</coverage diagnosis>")
    lines.append("</relevant past procedure>")
    return "\n".join(lines) + "\n"


def _memory_block(arm, task, whole_retriever, typed_trajs=None, subgoals=None) -> str:
    if arm == "none":
        return ""
    if arm == "whole_traj" and whole_retriever is not None:
        res = whole_retriever.retrieve(task, top_k=1)
        return f"<relevant past procedure>\n{res.to_memory_text()}\n</relevant past procedure>\n"
    if arm == "gap_aware" and whole_retriever is not None:
        return _gap_aware_block(task, whole_retriever, typed_trajs or {}, subgoals)
    return ""


def _pick_command(raw: str, admissible: list[str]) -> tuple[str, bool]:
    """Map the LLM's THINK/ACT response to an admissible command. (cmd, was_valid)."""
    text = raw or ""
    # Prefer the ACT: line if present (ReAct format).
    m = re.search(r"ACT:\s*(.+)", text, re.IGNORECASE)
    s = (m.group(1) if m else text).strip().strip('"').strip().lower()
    # drop a trailing parenthetical / period
    s = s.split("\n")[0].strip().rstrip(".")
    for a in admissible:
        if a.lower() == s:
            return a, True
    # loose contains match
    for a in admissible:
        if a.lower() in s or s in a.lower():
            return a, True
    # fallback: first admissible, flagged invalid
    return (admissible[0] if admissible else "look"), False


async def run_arm(arm: str, num_games: int, max_steps: int, model_key: str | None = None) -> dict:
    from dotenv import load_dotenv
    load_dotenv()
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    from evaluation.utils import agent_utils

    # answer model (same resolution the judge/search use)
    rm = agent_utils.load_eval_config(
        str(REPO_ROOT / "evaluation" / "procedural_memory" / "configuration.yml"))
    conf = rm.config
    # --model-key overrides the config default (retrieval_agent.llm_model).
    llm_name = model_key or getattr(getattr(conf, "retrieval_agent", None), "llm_model", None)
    model = await rm.get_language_model(llm_name, validate=True)

    whole_retriever = None
    typed_trajs = {}
    if arm in ("whole_traj", "gap_aware"):
        from evaluation.procedural_memory.whole_traj_baseline import WholeTrajectoryRetriever
        tp = REPO_ROOT / "evaluation" / "data" / "proced_mem_bench" / "trajectories.json"
        whole_retriever = WholeTrajectoryRetriever(str(tp))
        if arm == "gap_aware":
            from evaluation.procedural_memory.stage1_align import load_typed_trajectories
            typed_trajs = load_typed_trajectories(str(tp))

    env = AlfredTWEnv(build_config(num_games), train_eval="eval_in_distribution")
    tw = env.init_env(batch_size=1)

    results = []
    for g in range(env.num_games):
        obs, info = tw.reset()
        obs0 = str(obs[0])
        # Goal is the "Your task is to: ..." line in the initial observation.
        m = re.search(r"your task is to:\s*(.+)", obs0, re.IGNORECASE)
        task = m.group(1).strip() if m else obs0
        done = False
        invalid = 0
        steps = 0
        success = False
        sg = None
        if arm == "gap_aware":
            from evaluation.procedural_memory.stage1_llm_decompose import llm_subgoals
            sg = await llm_subgoals({"query_text": task}, model)
        mem = _memory_block(arm, task, whole_retriever, typed_trajs, sg)
        history: list[str] = []
        for _ in range(max_steps):
            admissible = info.get("admissible_commands", [[]])[0]
            hist_text = "\n".join(history[-12:]) if history else "(nothing yet)"
            prompt = REACT_PROMPT.format(memory_block=mem, task=task,
                                         history=hist_text,
                                         obs=str(obs[0])[:600],
                                         admissible="\n".join(admissible))
            resp = await model.generate_response(user_prompt=prompt)
            text = resp[0] if isinstance(resp, tuple) else str(resp)
            cmd, valid = _pick_command(text, admissible)
            if not valid:
                invalid += 1
            obs, scores, dones, info = tw.step([cmd])
            steps += 1
            # record what happened for the agent's memory of its own actions
            history.append(f"ACT: {cmd} -> {str(obs[0])[:120]}")
            # Success signal on this alfworld build is info['won'] (a list), not score.
            won = info.get("won", [False])
            if (isinstance(won, (list, tuple)) and won and won[0]) or won is True:
                success = True
                done = True
                break
            if dones[0]:
                done = True
                break
        results.append({"game": g, "success": success, "invalid": invalid,
                        "steps": steps, "done": done})
        print(f"  [{arm}] game {g+1}/{env.num_games}: success={success} "
              f"invalid={invalid} steps={steps}")

    n = len(results) or 1
    summary = {
        "arm": arm,
        "model_key": llm_name,
        "n_games": len(results),
        "success_rate": sum(r["success"] for r in results) / n,
        "mean_invalid": sum(r["invalid"] for r in results) / n,
        "mean_steps": sum(r["steps"] for r in results) / n,
        "results": results,
    }
    return summary


def main() -> None:
    import asyncio
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["none", "whole_traj", "gap_aware"], default="whole_traj")
    p.add_argument("--num-games", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--model-key", default=None,
                   help="language_models config key to use as the acting model "
                        "(e.g. openai_model | openai_model_strong). "
                        "Default: retrieval_agent.llm_model from config.")
    p.add_argument("--out-dir", default="evaluation/procedural_memory/result/e4")
    args = p.parse_args()

    if not os.environ.get("ALFWORLD_DATA"):
        raise SystemExit("export ALFWORLD_DATA first (e.g. $HOME/.cache/alfworld)")

    summary = asyncio.run(run_arm(args.arm, args.num_games, args.max_steps, args.model_key))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tag by model so mini and strong runs of the same arm don't overwrite.
    tag = (summary.get("model_key") or "default").replace("/", "_")
    out_path = out_dir / f"e4_{args.arm}_{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== E4 arm={summary['arm']} model={summary.get('model_key')} ===")
    print(f"  games:        {summary['n_games']}")
    print(f"  success rate: {summary['success_rate']:.3f}")
    print(f"  mean invalid: {summary['mean_invalid']:.2f}")
    print(f"  mean steps:   {summary['mean_steps']:.2f}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
