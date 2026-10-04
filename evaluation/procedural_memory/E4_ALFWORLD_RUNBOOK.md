# E4 — ALFWorld execution harness (RUNBOOK, not yet built)

This is the one experiment that gives a true **task-success** number instead of
a retrieval or answer-quality proxy. It is deliberately NOT scaffolded with fake
code, because ALFWorld is not installed in this env (`import alfworld` fails,
`ALFWORLD_DATA` unset) and stubbing an API we cannot run would be worse than an
honest runbook.

## What E4 measures
For each ALFWorld task: give the executor agent the memory payload from an arm,
let it act in the real simulator, record **success / invalid-action count /
steps / tokens**. Arms (same executor model, prompt, action budget, seeds):

1. no memory
2. `whole_traj` — Memp-style (already built, offline retriever)
3. grounded gap-aware retriever (the `procedural` target, needs Neo4j)
4. grounded + gap-flag injected into the executor prompt

## Prerequisites (human/next-session, ~hours not minutes)
1. `pip install alfworld` (pulls TextWorld + Fast-Downward; heavy).
2. `alfworld-download` to fetch game files; set `ALFWORLD_DATA`.
3. Confirm the 40 benchmark queries map to runnable ALFWorld game files
   (the proced_mem_bench tasks are derived from AgentInstruct ALFWorld
   trajectories — verify the game ids still resolve).
4. An executor loop: observation -> (memory payload + ReAct prompt) -> action
   -> env.step, capped at N actions. MemMachine's answer model can drive it.

## Build plan (once prereqs exist)
- `e4_executor.py`: thin wrapper around `alfworld.agents.environment`
  `AlfredTWEnv`, single-game rollout, returns (success, n_invalid, n_steps,
  tokens).
- `e4_run.py`: for each query's game, run all arms x seeds, write per-game rows.
- Metrics: paired success delta (McNemar), invalid-action rate, FWT on the
  pick_and_place -> pick_clean_then_place transfer pair.

## Why it's deferred, not skipped
E1–E3 are retrieval/diagnosis-side and fully banked. E4 is the execution-side
headline; it needs a real environment install and is a multi-session build. The
whole_traj baseline (arm 2) is already done so E4 is not starting from zero.

## Honest expectation
Given E1 (order adds nothing) and E2/E3 (gap detection saturates on clean data),
E4 may well show the gap-aware arm ~ties Memp on this clean benchmark. That is a
publishable scoping result for a first paper: "structure helps diagnosis, not
execution, on causally-ordered benchmarks."
