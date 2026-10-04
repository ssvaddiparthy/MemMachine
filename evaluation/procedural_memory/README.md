# Procedural Memory Evaluation

Benchmarks for evaluating procedural memory in MemMachine. Part of the
cs-8803-fall-2026 research paper extending MemMachine with a procedural
memory tier.

## Benchmarks

| Benchmark | Type | Status |
|-----------|------|--------|
| **proced_mem_bench** | Pure retrieval (P@k, NDCG, MAP) | ✅ Scaffolded |
| **ALFWorld** | End-to-end task execution | 🔲 Planned |
| **τ-Bench** | Multi-turn stateful tool-use | 🔲 Planned |
| **ToolSandbox** | Stateful conversational scenarios | 🔲 Planned |

## proced_mem_bench

Based on the [Procedural Memory Retrieval Benchmark](https://arxiv.org/abs/2511.21730)
(Kohar & Krishnan, MemAgents Workshop @ ICLR 2026).
Uses 336 AgentInstruct trajectories from ALFWorld and 40 stratified
queries (HARD/MEDIUM/EASY).

### Data Setup

Download the benchmark data from
[qpiai/Proced_mem_bench](https://github.com/qpiai/Proced_mem_bench)
and place it in `evaluation/data/proced_mem_bench/`:

- **Trajectories**: [`procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json`](https://github.com/qpiai/Proced_mem_bench/blob/main/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json)
  → save as `evaluation/data/proced_mem_bench/trajectories.json`
- **Queries**: [`procedural_memory_benchmark/benchmark/data/query_bank.json`](https://github.com/qpiai/Proced_mem_bench/blob/main/procedural_memory_benchmark/benchmark/data/query_bank.json)
  → save as `evaluation/data/proced_mem_bench/queries.json`

```
evaluation/data/proced_mem_bench/
├── trajectories.json    # 336 AgentInstruct trajectories
└── queries.json         # 40 stratified benchmark queries
```

### Data Schema Reference

**`trajectories.json`** — 336 recorded ALFWorld task executions from
the AgentInstruct dataset. Each trajectory is one complete playthrough
of a household task (e.g. "find two laptops and put them in bed").

```json
{
  "metadata": {"corpus_name": "AgentInstruct ALFWorld Trajectories", ...},
  "trajectories": [
    {
      "task_instance_id": "alfworld_0",
      "task_description": "find two laptop and put them in bed.",
      "state_action_pairs": [
        {"step_id": 1, "state": "You are in the middle of a room...", "action": "go to diningtable 1"},
        {"step_id": 2, "state": "On the diningtable 1, you see...", "action": "take laptop 1 from diningtable 1"}
      ],
      "total_steps": 14,
      "source": "agentinstruct"
    }
  ]
}
```

| Key | Type | Description |
|-----|------|-------------|
| `task_instance_id` | string | Unique ID for this trajectory (e.g. `alfworld_0`) |
| `task_description` | string | Natural-language task goal |
| `state_action_pairs` | list | Ordered steps the agent took |
| `state_action_pairs[].step_id` | int | 1-indexed step number |
| `state_action_pairs[].state` | string | Agent's observation before acting |
| `state_action_pairs[].action` | string | Action the agent chose |
| `total_steps` | int | Number of steps in this trajectory |
| `source` | string | Dataset origin (always `"agentinstruct"`) |

**`queries.json`** — 40 benchmark queries with **ground truth relevance
annotations**. These are NOT results — they are human-annotated relevance
judgments used to evaluate retrieval quality. An LLM judge scored each
(query, trajectory) pair from 0–10; trajectories scoring ≥ 6 are
considered relevant.

```json
{
  "metadata": {"total_queries": 40, ...},
  "queries": [
    {
      "query_id": "easy_1",
      "tier": "EASY",
      "query_type": "placement",
      "query_text": "Put a soap bar in the cabinet",
      "relevant_trajectories": [
        {"trajectory_id": "alfworld_22", "relevance_score": 10.0, "reasoning": "..."},
        {"trajectory_id": "alfworld_28", "relevance_score": 6.0, "reasoning": "..."}
      ]
    }
  ]
}
```

| Key | Type | Description |
|-----|------|-------------|
| `query_id` | string | Unique query ID (e.g. `easy_1`, `hard_3`) |
| `tier` | string | Difficulty tier: `EASY`, `MEDIUM`, or `HARD` |
| `query_type` | string | Task category (e.g. `placement`, `heating`) |
| `query_text` | string | Natural-language procedural query |
| `relevant_trajectories` | list | Ground truth: which trajectories are relevant |
| `relevant_trajectories[].trajectory_id` | string | References a `task_instance_id` |
| `relevant_trajectories[].relevance_score` | float | 0–10 relevance score (≥ 6 = relevant) |
| `relevant_trajectories[].reasoning` | string | LLM judge's rationale for the score |

For full benchmark methodology and scoring details, see the
[benchmark paper](https://arxiv.org/abs/2511.21730).

### Running

Requires MemMachine server dependencies and a `configuration.yml` in
`evaluation/retrieval_agent/`.

```bash
cd evaluation/procedural_memory

# Step 1: Ingest trajectories into episodic memory
./run_proced_mem_bench.sh exp1 ingest retrieval_agent

# Step 2: Run queries and evaluate
./run_proced_mem_bench.sh exp1 search retrieval_agent

# Optional: clean up
./run_proced_mem_bench.sh exp1 delete retrieval_agent
```

### Metrics

**Retrieval quality** (per-tier and aggregate):
- P@1, P@5 — Precision at k
- NDCG@10 — Normalized Discounted Cumulative Gain
- MAP — Mean Average Precision

**Efficiency**:
- Agent hop count — sub-agent calls per query
- Latency — wall-clock time per query (retrieval + answer)
- Token usage — input/output tokens consumed

### Output Files

```
evaluation/procedural_memory/result/
├── proced_mem_bench_retrieval_agent_output_exp1.json         # Raw results
├── proced_mem_bench_retrieval_agent_eval_metrics_exp1.json   # Eval metrics
└── final_score/
    └── proced_mem_bench_retrieval_agent_exp1.result          # Summary
```

## Shared Utilities

`procedure_utils.py` contains:
- `TrajectoryLogger` — wraps retrieval agent to capture hop-level trajectory data
- `QueryTrajectory` / `TrajectoryHop` — trajectory data structures
- IR metric functions: `precision_at_k`, `ndcg_at_k`, `average_precision`
- `aggregate_metrics`, `aggregate_trajectory_stats` — summary statistics
- `format_results_json`, `format_final_score` — output formatting
