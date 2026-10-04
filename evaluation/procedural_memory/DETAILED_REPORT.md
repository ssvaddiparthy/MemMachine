# TRACE-PM — Detailed Experiment Report

**Project:** CS 8903 — procedural-memory tier for MemMachine (TRACE-PM)
**Repo:** `MemMachine`  **Branch:** `feat/procedural-memory-tier`  **Status:** all local, **not pushed, no PR**
**Report date:** 2026-10-02
**Author of record:** checker (human) owns all novelty/claim decisions; this report is the evidence log.
**Benchmark:** proced_mem_bench (ALFWorld-derived) — 40 queries (15 EASY / 14 MEDIUM / 11 HARD), 336 successful trajectories, ground truth = relevant trajectory IDs (LLM-scored, relevance ≥ 6.0).

---

## 0. Reading guide

This is the exhaustive record. Every experiment has: hypothesis, method, exact commands, raw
numbers, interpretation, and honest caveats. Nothing here is a claim — the thesis is assembled
*backward* from these results (graduate first-paper strategy: breadth of clean experiments → a
hypothesis set that fits, presentable + honest over lofty).

Metric glossary:
- **P@1 / P@5** — precision at rank 1 / 5 (fraction of top-k retrieved IDs that are relevant).
- **NDCG@10** — rank-weighted relevance in top 10 (rewards relevant hits near the top).
- **MAP** — mean average precision over all relevant items.
- **recall / specificity (E2/E3)** — gap-detection: fraction of true gaps found / fraction of
  covered cases correctly reported gap-free.
- **pass rate / mean score (E4-lite)** — LLM-judge verdict on the generated procedure (1–5 scaled to 0–1).

All retrieval/diagnosis metrics (E1–E3) are **IR / diagnosis quality**, NOT task success. The LLM
judge (E4-lite) is NOISY: identical reruns moved the baseline pass rate 67.5% → 65%. Treat judge
deltas as directional. True task-success (ALFWorld execution) is **E4-full: harness built and run,
first result is a bare-agent floor (0/5), ReAct scaffold still needed for a meaningful comparison** (§6).

---

## 1. Prior context (before this session's experiments)

Earlier sessions established the setup and fixed three real bugs (all committed before this work):
- `89860be` — procedural LLM calls were using a non-existent client interface; fixed to MemMachine
  `LanguageModel.generate_response`. (This silently floored the judge at 0.2 across the board; the
  root cause of the "nothing improved" symptom seen earlier.)
- `adca5b9`, `74abbf9`, `9dd2f94` — judge LLM-name resolution + `_extract_procedure_text` using the
  `model_answer` field.
- `87dd495` — grounded trajectory-anchored retriever (`--test-target procedural`, needs Neo4j).
- `ab9cb9e` — vocab + tiered matching to fix 17/40 empty retrievals.

Baseline (episodic, `--test-target retrieval_agent`) judged pass rate: **~65–67.5%** (judge noise band).

Graph store at ingest time (Neo4j): 24 `ProcTool` verbs, 91 `TOOL_TRANSITION` edges, 4,542
`ProcAction` grounded steps, 336 `ProcTrajectory`. The ORIGINAL design (Louvain + query-time Steiner
over abstract verbs) was abandoned earlier: it produced placeholder procedures, dropped preconditions,
mis-ordered steps, and Louvain gave only 2 communities. This session does NOT use Louvain or Steiner.

---

## 2. Experiment E1 — Does action ORDER help retrieval?

**Hypothesis.** Ordered sub-goal alignment beats an order-free bag-of-sub-goals, most on multi-step
MEDIUM queries (where order "should" matter).

**Method.** Parse each ALFWorld action string into a typed `(verb, object_type, target_type)` step
(regex, instance numbers stripped: `soapbar 2` → `soapbar`). Decompose each query into ordered typed
sub-goals. Two deterministic scorers, identical tokens, differ only in whether order is used:
- **align** — weighted-Levenshtein DP over the sub-goal sequence vs the trajectory action sequence.
  Weights: verb mismatch 3.0, object 2.0, target 1.0; GAP (unmatched sub-goal) 4.0; EXTRA
  (unneeded trajectory action) 0.3; normalized by #sub-goals.
- **bag** — order-free: each sub-goal takes its cheapest match against any remaining action; same
  pair-cost; GAP for unmatched. Order information removed.

Frozen vocab (derived from the on-disk corpus, no DB): **10 verbs, 73 object types, 25 target types**.
Decomposition run two ways: deterministic keyword (from `extracted_keywords`) and LLM (`openai_model`).

**Files:** `stage1_align.py`, `stage1_run.py`, `stage1_llm_decompose.py`, `test_stage1_align.py`.
**Commits:** `3ab759c`, `bb39ee6`, `ad3209c`.
**Bug fixed mid-run:** `rm.conf` → `rm.config` (AttributeError on ResourceManagerImpl); copied the
judge's proven resolution `getattr(getattr(conf,"retrieval_agent",None),"llm_model",None)`.

**Commands.**
```bash
# keyword (offline)
PYTHONPATH=. python3 evaluation/procedural_memory/stage1_run.py \
  --queries evaluation/data/proced_mem_bench/queries.json \
  --trajectories evaluation/data/proced_mem_bench/trajectories.json --scorer both
# LLM (needs OPENAI_API_KEY)
PYTHONPATH=. python3 evaluation/procedural_memory/stage1_run.py \
  --queries ... --trajectories ... --scorer both --decompose llm \
  --config-path evaluation/procedural_memory/configuration.yml
```

**Raw results — keyword decomposition (n=40):**

| arm | P@1 | P@5 | NDCG@10 | MAP |
|---|---|---|---|---|
| align | 0.625 | 0.550 | 0.523 | 0.189 |
| bag | 0.575 | 0.550 | 0.509 | 0.203 |

Per-tier NDCG@10: EASY align 0.484 / bag 0.385; MEDIUM 0.637 / 0.706; HARD 0.432 / 0.427.

**Raw results — LLM decomposition (two runs, reproducible):**

| arm | P@1 | P@5 | NDCG@10 | MAP |
|---|---|---|---|---|
| align (run A / run B) | 0.775 / 0.750 | 0.685 / 0.660 | 0.653 / 0.633 | 0.250 / 0.241 |
| bag (run A / run B) | 0.725 / 0.700 | 0.680 / 0.650 | 0.659 / 0.635 | 0.280 / 0.265 |

**align − bag delta (NDCG@10):**

| Slice | keyword | LLM (A) | LLM (B) |
|---|---|---|---|
| EASY | +0.100 | +0.074 | +0.074 |
| MEDIUM | **−0.069** | **−0.094** | **−0.094** |
| HARD | +0.005 | −0.003 | +0.012 |
| ALL | +0.015 | −0.006 | −0.002 |

LLM-decomposition fallbacks: 4 (run A) / 3 (run B) multi-subgoal queries matched the keyword output
(possible silent fallbacks); does not change direction. The HARD ±0.01 wobble between A and B is
LLM-decomposition sampling noise.

**Interpretation.** Ordering adds nothing. With a *better* (LLM) decomposition the overall delta
flips slightly NEGATIVE, and on MEDIUM — where order should help most — the order-free bag clearly
WINS (0.864 vs 0.771). Both arms improve a lot with clean decomposition (align 0.523 → 0.653),
confirming **decomposition quality dominates; sequence does not.** The confound "bad decomposition
scrambled the order" is ruled out: the LLM sub-goals are correctly ordered (e.g. `clean(soapbar) →
put(soapbar,cabinet)`).

**Why (mechanistic).** ALFWorld sub-goals are causally forced — you cannot `put` a soap bar you
have not `clean`ed — so any trajectory containing both effects almost always has them in the only
workable order. Order carries ~no signal beyond "both effects present," which is exactly what the bag
measures.

**Verdict.** Order-aware alignment is NOT the paper's headline. Clean negative/ablation result;
doubles as the control that pre-empts "your gain is just ordering."

---

## 3. Experiment E2 — Can an edit script DETECT a missing step?

**Hypothesis.** The aligner's edit script (COVERED / REBIND / GAP / EXTRA) correctly flags a deleted
required sub-goal; the order-free bag structurally cannot (no "required-but-absent" concept).

**Method.** Backtrace the E1 alignment DP into a labelled edit script. **Synthetic gold by
construction:** take a trajectory that covers a query, delete the steps realizing one required
sub-goal, check the aligner flags exactly that sub-goal as GAP. Deletion = ground truth; no
hand-labelling, no LLM self-marking. Hardened with covered-trajectory NEGATIVES (nothing deleted →
must report zero gaps → specificity), every required sub-goal deleted across up to 8 covering
trajectories per query, and confuser detection.

**Files:** `stage2_gap.py`, `stage2_run.py`, `test_stage2_gap.py`.  **Commits:** `1e89907`, `bd1ac7a`.
**Tests:** 4 offline, all pass (covered→no-gap, deleted→gap, REBIND on wrong target, delete-only-targeted-verb).

**Command.**
```bash
PYTHONPATH=. python3 evaluation/procedural_memory/stage2_run.py \
  --queries evaluation/data/proced_mem_bench/queries.json \
  --trajectories evaluation/data/proced_mem_bench/trajectories.json
```

**Raw results (hardened).**
- POSITIVES: 110 trials, TP 110, FN 0 → **recall 1.000**
- NEGATIVES: 55 trials, TN 55, FP 0 → **specificity 1.000**
- CONFUSERS: 1 trial, 1 detected
- Per deleted-verb: clean 24/24, cool 7/7, heat 24/24, put 55/55

Example edit scripts (deleting one required step):
- `medium_1` "Clean a soap bar and put it in the cabinet" — delete `clean(soapbar)` →
  diagnosis `GAP:clean(soapbar), COVERED:put(soapbar,cabinet)`; GAP found ✓
- `easy_6` "Put a potato in the microwave" — keyword decomposition INVENTED a `heat(potato)` sub-goal;
  deleting it is detected, but this exposes that keyword decomposition can hallucinate steps (LLM
  decomposition does not — it returns just `put(potato,microwave)`).

**Interpretation + caveat.** Mechanism works and the bag cannot replicate it. BUT: only **1 natural
confuser trial** exists — proced_mem_bench trajectories are too clean to stress the hard case, so the
perfect score is informative only about the easy regime. "Mechanism proven; eval saturated."

---

## 4. Experiment E3 — WHEN does gap detection break? (adversarial)

**Hypothesis.** Synthetic difficulty injectors will degrade gap detection, yielding a precision/recall
curve instead of a flat 1.0.

**Method.** After deleting one required sub-goal (= true gap), inject difficulty and re-measure
recall + specificity. Injectors with intensity sweeps, deterministic given `--seed`:
- **confuser** (0–3): insert same-verb WRONG-object actions.
- **partial** (0.0–1.0): drop a fraction of setup steps (go_to/take/open).
- **near_dup** (0–4): duplicate random kept actions.
- **shuffle** (0/1): randomly permute all trajectory actions.

**File:** `stage3_adv_gap.py`.  **Commit:** `9615b9e`.

**Command.**
```bash
PYTHONPATH=. python3 evaluation/procedural_memory/stage3_adv_gap.py \
  --queries evaluation/data/proced_mem_bench/queries.json \
  --trajectories evaluation/data/proced_mem_bench/trajectories.json --seed 0
```

**Raw results (recall / specificity):**

| Injector | intensities | recall | specificity |
|---|---|---|---|
| confuser | 0,1,2,3 | 1.000 all | 1.000 all |
| partial | 0,0.25,0.5,1.0 | 1.000 all | 1.000 all |
| near_dup | 0,1,2,4 | 1.000 all | 1.000 all |
| **shuffle** | 0 → 1 | 1.000 | **1.000 → 0.418** |

(110 positive + 55 negative trials per cell.)

**Interpretation.** The aligner is robust to object confusers, partial coverage, and duplicate steps
— recall and specificity hold at 1.0. It is broken ONLY by **action-order shuffle**: specificity
collapses to 0.42, i.e. it hallucinates GAPs on trajectories that actually cover the query, because
ordered alignment cannot thread a correct path through scrambled actions.

**Why this matters — closes the loop with E1.** E1: exploiting sequence buys nothing on
causally-ordered data. E3: the ordered aligner is actively HARMED by disorder in a way the order-free
bag would not be. Together: **the order-free formulation is both sufficient and more robust.** This is
a clean, non-obvious "when does structure help vs hurt" finding.

---

## 5. Experiment E4-lite — Memp-style whole-trajectory baseline (downstream answer quality)

**Hypothesis.** Whole-trajectory reuse (Memp) through the answer LLM differs measurably from the
episodic baseline; test direction and tier pattern.

**Method.** Offline lexical (token-Jaccard) whole-trajectory retriever reads `trajectories.json`
directly (no DB), returns top-k whole trajectories' raw steps into the SAME `ANSWER_PROMPT` + answer
model as the episodic baseline. Judged by the LLM judge (mode=procedural), compared to the episodic
baseline (`judge_baseline.json`, mode=episodic).

**File:** `whole_traj_baseline.py`; wired as `--test-target whole_traj` in `proced_mem_bench_search.py`.
**Commit:** `42b4374`.

**Commands.**
```bash
PYTHONPATH=. python3 evaluation/procedural_memory/proced_mem_bench_search.py \
  --data-path .../queries.json --eval-result-path .../result/whole_traj_output.json \
  --test-target whole_traj --config-path .../configuration.yml --concurrency 1
PYTHONPATH=. python3 evaluation/procedural_memory/proced_mem_bench_judge.py judge \
  --search-output .../whole_traj_output.json --mode procedural \
  --config-path .../configuration.yml --output .../judge_whole_traj.json
PYTHONPATH=. python3 evaluation/procedural_memory/proced_mem_bench_judge.py compare \
  --baseline .../judge_baseline.json --procedural .../judge_whole_traj.json
```

**Raw retrieval-side (IR) for whole_traj:** P@1 0.550, P@5 0.550, NDCG@10 0.490, MAP 0.168.
Per-query cost ~964 in / 96 out tokens, 1 LLM call. (Retrieval latency ~2ms — lexical.)
## 6. E4-full — ALFWorld execution (BUILT and RUN; agent-floor result)

ALFWorld 0.4.2 was installed (`pip install alfworld`, `alfworld-download`;
`ALFWORLD_DATA=$HOME/.cache/alfworld`, 140 solvable games in valid_seen). A real
execution harness `e4_executor.py` was built against the verified API:
`AlfredTWEnv(config,'eval_in_distribution').init_env(1)`, reset/step loop, LLM
chooses one admissible command per step, success = `info['won'][0]`.
**Commits:** `faab07b` (harness), `5a4b36f` (two fixes below).

**Env note discovered:** the shell's default `python3` is Homebrew 3.14; alfworld +
memmachine live only in the conda env `memmachine-eval-replicate` (3.12). All E4
runs use `/opt/anaconda3/envs/memmachine-eval-replicate/bin/python3`.

**Command (per arm):**
```bash
export ALFWORLD_DATA="$HOME/.cache/alfworld"
PY=/opt/anaconda3/envs/memmachine-eval-replicate/bin/python3
PYTHONPATH=. $PY evaluation/procedural_memory/e4_executor.py --arm {none|whole_traj} --num-games 5 --max-steps 40
```

**Two bugs found and fixed (`5a4b36f`):**
1. Success was checked as `scores[0] >= 1` — but on this build the win signal is
   `info['won']` (a list). Real wins read as failures. **This alone made the first run 0/5.**
2. `task` was the entire observation blob; now the `Your task is to: ...` line is
   extracted. Also the full admissible-command list is sent (was truncated at 40,
   hiding the real action in cabinet-heavy games).

**Runs (5 games/arm, max 40 steps):**

| Run | arm | success | mean_invalid | mean_steps |
|---|---|---|---|---|
| pre-fix | none | 0/5 | 7.2 | 40 |
| pre-fix | whole_traj | 0/5 | 2.8 | 40 |
| post-fix | none | 0/5 | 0.2 | 40 |
| post-fix | whole_traj | 0/5 | 0.2 | 40 |

**Diagnosis — the 0/5 is REAL, not a harness bug.** A walkthrough-replay probe
confirmed success detection works: replaying a game's shipped 6-step
`walkthrough` (`go to countertop 3 → take ladle 1 → go to sinkbasin 1 → clean
ladle 1 with sinkbasin 1 → go to countertop 1 → move ladle 1 to countertop 1`)
flips `won=True` at the final step. So the env + metric are correct.

The post-fix invalid-action count dropping to ~0.2 shows the LLM now picks VALID
commands almost every step — yet still never wins in 40 steps. Conclusion: a bare
zero-shot ReAct agent (no think→act scaffold, no few-shot) cannot solve these
`pick_clean_then_place` games on the eval_in_distribution split. **This is a real
"bare-agent floor ≈ 0%" result**, consistent with the ALFWorld literature where
unaugmented agents are weak and scaffolding/memory is what lifts them.

**Why this is good for the thesis, and the blocker.** A near-zero floor means
there is headroom for a memory arm to show lift — but the current bottleneck is the
agent's REASONING, not the memory content, so the whole_traj payload cannot move a
0-vs-0 comparison. **A 0-vs-0 comparison is uninformative.** To get a usable E4
number the executor needs a proper ReAct `think→act` prompt + 1–2 few-shot
examples (standard ALFWorld practice) so the baseline can register non-zero
success; only then does the memory arm's lift become measurable. That scaffold is
NOT yet built — it is the next E4 step.

**E4 status:** infrastructure complete and verified end-to-end (env, games, LLM
loop, success detection, metrics, output). First numbers are a bare-agent floor
(0/5 both arms); a meaningful arm comparison awaits the ReAct scaffold.

---

## 6b. E4-lite — Memp whole-trajectory vs episodic (judged, retrieval-side)

**Judge comparison (whole_traj vs episodic baseline):**

| Metric | Baseline (episodic) | Whole-traj (Memp) | Δ |
|---|---|---|---|
| Pass rate | 65.0% | 55.0% | −10.0 |
| Mean score | 0.851 | 0.649 | −0.203 |
| Completeness | 0.745 | 0.595 | −0.150 |
| Correctness | 0.905 | 0.675 | −0.230 |
| Ordering | 0.915 | 0.685 | −0.230 |
| Executability | 0.840 | 0.640 | −0.200 |

Per-tier pass: EASY 60→40 (−20), HARD 90.9→72.7 (−18.2), **MEDIUM 50→57.1 (+7.1)**.
Whole-traj judge mean per tier: EASY 0.553, MEDIUM 0.643, HARD 0.786.

**Interpretation.** Whole-trajectory reuse UNDERPERFORMS the plain episodic baseline overall — dumping
a long raw trajectory adds noise that hurts EASY/HARD. BUT it HELPS on MEDIUM (+7.1): multi-sub-goal
tasks benefit from a whole worked example that carries the needed sequence. Pattern: **whole-trajectory
memory helps exactly on multi-step tasks and hurts on single-step ones.**

**Caveats (critical).**
1. **Retriever is cheap lexical Jaccard, not embeddings** — part of the loss is weak retrieval, not
   the whole-trajectory idea. A fair Memp needs embedding retrieval (follow-up E4-lite-v2).
2. **LLM judge is noisy** (±2–3 pts). The −10 is directional only.
3. Comparison is whole-traj (mode=procedural) vs episodic (mode=episodic); the grounded gap-aware arm
   (`--test-target procedural`, Neo4j) was NOT run head-to-head here — needs the DB up.

---

## 6c. E4-full status update (SUPERSEDED — now built)

The earlier plan deferred E4-full as "not built." That is NO LONGER TRUE: ALFWorld
was installed and `e4_executor.py` built + run — see §6 above for the harness, the
two bug fixes, the 5-game runs, and the bare-agent-floor diagnosis. The remaining
work is the ReAct `think→act` scaffold so the baseline scores non-zero and the
memory arm's lift becomes measurable; plus the grounded gap-aware arm (needs Neo4j)
and multi-seed scaling. `E4_ALFWORLD_RUNBOOK.md` holds the original prereqs/plan.

---

## 7. Cross-cutting finding (the emerging thesis, assembled backward)

All four banked experiments point one way:

> **On a clean, causally-ordered procedural benchmark (ALFWorld / proced_mem_bench), additional
> procedural STRUCTURE or CONTENT rarely helps retrieval and often hurts — with one narrow exception,
> multi-sub-goal (MEDIUM) tasks.** Specifically: (E1) sequence order gives no retrieval advantage and
> (E3) reduces robustness under disorder; (E2) edit-script gap detection is the one capability the
> order-free baseline cannot provide, but it (E2) saturates on this benchmark; and (E4-lite)
> whole-trajectory reuse underperforms episodic retrieval overall yet helps on MEDIUM.

This is honest, slightly contrarian, and defensible for a first paper — it stands on E1–E4-lite even
if E4-full is never built. The unique positive capability (gap detection / evidence-bounded
abstention) is real but needs a harder benchmark or execution evidence to *matter*.

---

## 8. Complete command reference

Offline (no key, no DB):
```bash
cd /Users/vaddipar/workplace/cs-8903-fall-2026/MemMachine
PYTHONPATH=. python3 evaluation/procedural_memory/test_stage1_align.py
PYTHONPATH=. python3 evaluation/procedural_memory/test_stage2_gap.py
PYTHONPATH=. python3 evaluation/procedural_memory/stage1_run.py --queries evaluation/data/proced_mem_bench/queries.json --trajectories evaluation/data/proced_mem_bench/trajectories.json --scorer both           # E1 keyword
PYTHONPATH=. python3 evaluation/procedural_memory/stage2_run.py --queries evaluation/data/proced_mem_bench/queries.json --trajectories evaluation/data/proced_mem_bench/trajectories.json                          # E2
PYTHONPATH=. python3 evaluation/procedural_memory/stage3_adv_gap.py --queries evaluation/data/proced_mem_bench/queries.json --trajectories evaluation/data/proced_mem_bench/trajectories.json --seed 0            # E3
```
Needs OPENAI_API_KEY:
```bash
PYTHONPATH=. python3 evaluation/procedural_memory/stage1_run.py --queries ... --trajectories ... --scorer both --decompose llm --config-path evaluation/procedural_memory/configuration.yml   # E1b
# E4-lite: search(whole_traj) -> judge -> compare (see §5)
```
Needs ALFWorld install: see `E4_ALFWORLD_RUNBOOK.md`.

---

## 9. Artifact inventory

New source (this session), all under `evaluation/procedural_memory/`:
`stage1_align.py`, `stage1_run.py`, `stage1_llm_decompose.py`, `test_stage1_align.py`,
`stage2_gap.py`, `stage2_run.py`, `test_stage2_gap.py`, `stage3_adv_gap.py`,
`whole_traj_baseline.py`, `E4_ALFWORLD_RUNBOOK.md`, this report.
Result JSONs under `result/stage1/`, `result/stage2/`, `result/stage3_adv/`, `result/whole_traj_output.json`,
`result/judge_whole_traj.json` — all gitignored (regenerable).

**Commit chain (feat/procedural-memory-tier, local, not pushed):**
`3ab759c` Stage-1 arms · `66b342b` gitignore results · `bb39ee6` LLM-decompose path ·
`ad3209c` fix .config · `1e89907` Stage-2 aligner · `bd1ac7a` hardened gap eval ·
`9615b9e` E3 adversarial · `42b4374` E4-lite whole-traj · `623882b` E4 runbook · (+ this report).

## 10. Open follow-ups (checker's call)
1. **E4-lite-v2:** embedding retriever for a fair Memp (removes caveat §5.1).
2. **E4-full:** ALFWorld execution — the real task-success headline.
3. **Grounded head-to-head:** run `--test-target procedural` (Neo4j up) vs whole_traj vs episodic.
4. **Harder benchmark:** gap detection needs non-saturated data to show a real P/R curve (E3 shows only
   shuffle breaks it on this corpus).
5. **Decide the paper's framing** from the §7 cross-cutting finding before building more.

---

## 6d. E4-full with ReAct scaffold — FIRST execution signal (n=5, smoke test)

After adding the ReAct `think→act` scaffold (commit `a79e4bb`: one-shot worked
example, domain rules, running action history), the floor broke and memory helped.

| arm | success rate | mean invalid | mean steps |
|---|---|---|---|
| none       | 0.200 (1/5) | 6.6 | 34.8 |
| whole_traj | 0.400 (2/5) | 1.8 | 26.4 |

All three metrics favor memory: success 20%→40%, invalid actions 6.6→1.8, steps
34.8→26.4 (whole_traj solved one game in 3 steps, another in 9). This is the
FIRST true task-success evidence (not a retrieval/judge proxy) that memory helps.

**CRITICAL caveat: n=5 is a smoke test, not a result.** 1/5 vs 2/5 is a one-game
difference — within coin-flip noise. Direction is encouraging and all three
metrics agree, but no quotable claim at this n. Next: scale to 20–30 paired games
per arm (same games across arms), ideally multi-seed, before any number is cited.
Arms already share the same game order (roughly paired). Grounded gap-aware arm
still pending (Neo4j).

---

## 6e. E4 three-arm comparison — the NOVEL claim (n=10, ReAct scaffold)

First 3-way run (same 10 games across arms, ReAct scaffold, max 40 steps):

| arm | success | mean invalid | mean steps |
|---|---|---|---|
| none       | 0.60 (6/10) | 3.3 | 28.9 |
| whole_traj | 0.40 (4/10) | 3.5 | 26.5 |
| gap_aware  | 0.60 (6/10) | **1.0** | **22.9** |

**Novel comparison (gap_aware vs whole_traj):** gap_aware wins — success 0.60 vs
0.40 (+20pp), invalid 1.0 vs 3.5, steps 22.9 vs 26.5. Adding the edit-script gap
diagnosis on top of the SAME anchor trajectory helped; the diagnosis, not the
trajectory content, is what moved the needle.

**Two honest qualifications:**
1. gap_aware only TIES none on success (0.60 = 0.60); its advantage over
   no-memory is EFFICIENCY (invalid 1.0 vs 3.3, steps 22.9 vs 28.9), not success.
2. whole_traj UNDERPERFORMS none (0.40 < 0.60) — plain whole-trajectory memory
   HURTS (consistent with E4-lite). gap_aware repairs that harm back to baseline.

**Emerging E4 headline (not 'context helps'):** plain whole-trajectory memory
hurts execution; explicit gap diagnosis repairs the harm and sharply improves
efficiency (invalid 3.5→1.0, steps 26.5→22.9). The DIAGNOSIS is the active
ingredient.

**CAVEAT:** n=10. 6-vs-4 success = two games (noisy). The efficiency deltas
(invalid, steps) are averaged over ~250 steps and are more trustworthy. Strong
enough to justify SCALING to 25–50 paired games + multi-seed; not yet quotable.
gap_aware costs one extra LLM call/game (the decomposition).

---

## 7b. Draft paper opener (DRAFT — claims pending the scaled run)

**One-sentence hypothesis:**
> Procedural memory helps LLM agents not by giving them more past experience to
> imitate, but by diagnosing what their retrieved experience is MISSING.

**Short abstract (draft):**
> Agent memory systems reuse past task trajectories to guide new tasks, but a
> trajectory that is *similar* to the current task is not necessarily *safe* to
> follow: it may skip a required step, point at objects that no longer exist, or
> impose the wrong order. We study procedural memory for LLM agents on ALFWorld
> and find, perhaps counterintuitively, that naively injecting a whole retrieved
> trajectory HURTS execution relative to no memory at all, because the agent
> follows irrelevant or missing-step guidance. We introduce a lightweight
> alternative: rather than hand the agent a raw trajectory, we ALIGN the task's
> ordered sub-goals against the retrieved trajectory and return an explicit edit
> script that flags which required steps the trajectory covers and which are
> MISSING — telling the agent, with provenance, exactly what it must supply
> itself. This diagnosis, not the trajectory content, is the active ingredient:
> it repairs the harm of naive reuse and sharply improves efficiency (fewer
> invalid actions, fewer steps). We further show, through controlled retrieval
> ablations, that sequence ORDER contributes little on causally-ordered tasks
> while gap DETECTION is the capability existing similarity- and
> whole-trajectory-based memories structurally lack.

**Honesty flags on this draft:**
- The "hurts / repairs / efficiency" claims rest on E4 at n=10. The abstract is
  written at the strength the SCALED run (25–50 paired games, multi-seed) must
  support; do not submit until that backs the direction.
- It deliberately leads with the NEGATIVE result (naive memory hurts) because
  that framing is what makes the gap-detection contribution land, and it is what
  the controlled ablations (E1 order-null, E3 shuffle-fragility) support.
- Provisional name/branding (TRACE-PM, "edit script") is parent/checker-owned;
  the implementation evidence does not by itself license a novelty claim.

---

## 7c. E4 cross-model — does gap-aware memory substitute for model tier? (n=40 paired)

**Question tested:** can gap-aware procedural memory let a weak/cheap model
(gpt-4o-mini) match a strong model (gpt-4o) that has no memory? I.e. is
alignment-memory worth a model-tier upgrade? Design: 2×2, same 40 ALFWorld games
across all four cells (paired), `none` and `gap_aware` arms × `gpt-4o-mini` and
`gpt-4o`. Harness: `--model-key openai_model` vs `openai_model_strong`.

| model | arm | success | mean invalid | mean steps |
|---|---|---|---|---|
| gpt-4o-mini | none       | 0.500 | 3.00 | 27.05 |
| gpt-4o-mini | gap_aware  | **0.550** | **1.75** | **22.65** |
| gpt-4o      | none       | 0.775 | 0.65 | 17.33 |
| gpt-4o      | gap_aware  | 0.800 | 0.95 | 16.23 |

Paired per-game success (win/loss/tie, of 40):
- mini: gap_aware **wins 5, loses 3, ties 32**
- 4o:   gap_aware **wins 4, loses 3, ties 33**

**The cross-model substitution hypothesis FAILS on success.** The target cell
(mini+gap_aware = 0.550) is far below the bar (4o+none = 0.775). Memory closed
only ~1/5 of the model-tier success gap (0.050 of 0.275). A model-tier upgrade
dominates alignment-memory for raw completion — do NOT write "cheap+memory =
expensive-no-memory".

**The result that DOES hold — an interaction effect (the stronger paper):**
gap-aware memory's value is concentrated where the agent is error-prone.
- Weak model: large efficiency gains — invalid 3.00→1.75 (−42%), steps
  27.05→22.65 (−16%).
- Strong model: ~flat — invalid 0.65→0.95 (slightly WORSE, within noise), steps
  essentially unchanged. The strong model is already clean (0.65 invalid/game),
  so there is little harm for the diagnosis to repair.

**Honest framing:**
> Alignment-memory does not substitute for model capability — a tier upgrade
> beats it on success. Its value is EFFICIENCY ON WEAK MODELS: it cuts invalid
> actions ~42% and steps ~16% for gpt-4o-mini while giving the already-clean
> gpt-4o essentially nothing. Procedural-memory diagnosis matters most exactly
> where models are weakest.

**CAVEATS:**
- Success deltas are tiny: mini 0.50→0.55 is 2 games, 4o 0.775→0.80 is 1 game;
  the paired win/loss (5–3, 4–3) confirms success is near-noise. The EFFICIENCY
  deltas (invalid, steps), averaged over ~1000+ steps/cell, are the trustworthy
  signal — same pattern as §7a.
- 4o invalid rising 0.65→0.95 under gap_aware is within noise but worth a
  footnote, not a claim (possible mild over-direction of an already-competent
  model).
- n=40 single-seed. Multi-seed still needed before quoting any success number.

**Implication for the opener (§7b):** keep the "diagnosis is the active
ingredient" spine; ADD the interaction effect (memory helps weak agents, not
strong ones) as a scaling/cost angle. Drop any "memory replaces a bigger model"
framing — the data kills it.
