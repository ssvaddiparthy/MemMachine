#!/usr/bin/env bash
# Run proced_mem_bench search N times and average the IR metrics.
#
# Usage:
#   ./run_5x_average.sh TEST_TARGET [N]
#
# Examples:
#   ./run_5x_average.sh retrieval_agent       # 5 runs (default)
#   ./run_5x_average.sh retrieval_agent 3     # 3 runs
#
# Prerequisites:
#   - Data already ingested (run ingest step first)
#   - AWS SSO session active
#   - OPENAI_API_KEY set
#
# Output:
#   result/multi_run/run_1/ ... run_N/   -- per-run results
#   result/multi_run/averaged_metrics.json -- averaged IR metrics

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
TEST_TARGET="${1:?Usage: $0 TEST_TARGET [NUM_RUNS]}"
NUM_RUNS="${2:-5}"
RESULTS_DIR="/Users/vaddipar/workplace/cs-8903-fall-2026/CS8903-Research-Notes/procedural-memmachine/pre-bench-mark"

MULTI_RUN_DIR="${RESULTS_DIR}/result/multi_run"
mkdir -p "$MULTI_RUN_DIR"

echo "=== Running proced_mem_bench ${NUM_RUNS}x with target=${TEST_TARGET} ==="
echo

for i in $(seq 1 "$NUM_RUNS"); do
    RUN_DIR="${MULTI_RUN_DIR}/run_${i}"
    echo "--- Run ${i}/${NUM_RUNS} ---"
    "${SCRIPT_DIR}/run_proced_mem_bench.sh" "run${i}" search "$TEST_TARGET" \
        --result-dir "$RUN_DIR"
    echo
done

echo "=== Averaging metrics across ${NUM_RUNS} runs ==="

python3 - "$MULTI_RUN_DIR" "$TEST_TARGET" "$NUM_RUNS" <<'PYEOF'
import json, sys, os
from pathlib import Path

multi_run_dir = Path(sys.argv[1])
test_target = sys.argv[2]
num_runs = int(sys.argv[3])

# Metrics to average
METRICS = ["p_at_1", "p_at_5", "ndcg_at_10", "average_precision"]

# Collect per-run, per-tier metrics
tier_runs: dict[str, list[dict[str, float]]] = {}  # tier -> [run_aggregate, ...]
overall_runs: list[dict[str, float]] = []

for i in range(1, num_runs + 1):
    eval_file = multi_run_dir / f"run_{i}" / f"proced_mem_bench_{test_target}_eval_metrics_run{i}.json"
    if not eval_file.exists():
        print(f"WARNING: {eval_file} not found, skipping run {i}")
        continue

    with open(eval_file) as f:
        data = json.load(f)

    # data is grouped by tier: {"HARD": [...], "MEDIUM": [...], "EASY": [...]}
    run_all_metrics: list[dict[str, float]] = []
    for tier, results in data.items():
        tier_avg = {m: sum(r.get(m, 0) for r in results) / len(results) for m in METRICS}
        tier_runs.setdefault(tier, []).append(tier_avg)
        run_all_metrics.extend([{m: r.get(m, 0) for m in METRICS} for r in results])

    if run_all_metrics:
        overall_avg = {m: sum(r[m] for r in run_all_metrics) / len(run_all_metrics) for m in METRICS}
        overall_runs.append(overall_avg)

if not overall_runs:
    print("ERROR: No valid runs found.")
    sys.exit(1)

# Average across runs
def avg_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    return {m: sum(d[m] for d in dicts) / len(dicts) for m in METRICS}

def std_dicts(dicts: list[dict[str, float]], means: dict[str, float]) -> dict[str, float]:
    n = len(dicts)
    if n < 2:
        return {m: 0.0 for m in METRICS}
    return {m: (sum((d[m] - means[m])**2 for d in dicts) / (n - 1))**0.5 for m in METRICS}

overall_mean = avg_dicts(overall_runs)
overall_std = std_dicts(overall_runs, overall_mean)

result = {
    "num_runs": len(overall_runs),
    "overall": {
        "mean": {m: round(v, 4) for m, v in overall_mean.items()},
        "std":  {m: round(v, 4) for m, v in overall_std.items()},
    },
    "per_tier": {},
}

for tier in sorted(tier_runs.keys()):
    tier_mean = avg_dicts(tier_runs[tier])
    tier_std = std_dicts(tier_runs[tier], tier_mean)
    result["per_tier"][tier] = {
        "mean": {m: round(v, 4) for m, v in tier_mean.items()},
        "std":  {m: round(v, 4) for m, v in tier_std.items()},
    }

out_path = multi_run_dir / "averaged_metrics.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

# Print summary
print(f"\nResults averaged over {result['num_runs']} runs:")
print(f"{'':>12}  {'P@1':>12}  {'P@5':>12}  {'NDCG@10':>12}  {'MAP':>12}")
print("-" * 64)
for tier in sorted(result["per_tier"].keys()):
    m = result["per_tier"][tier]["mean"]
    s = result["per_tier"][tier]["std"]
    print(f"{tier:>12}  {m['p_at_1']:.3f}±{s['p_at_1']:.3f}  {m['p_at_5']:.3f}±{s['p_at_5']:.3f}  {m['ndcg_at_10']:.3f}±{s['ndcg_at_10']:.3f}  {m['average_precision']:.3f}±{s['average_precision']:.3f}")
print("-" * 64)
m = result["overall"]["mean"]
s = result["overall"]["std"]
print(f"{'OVERALL':>12}  {m['p_at_1']:.3f}±{s['p_at_1']:.3f}  {m['p_at_5']:.3f}±{s['p_at_5']:.3f}  {m['ndcg_at_10']:.3f}±{s['ndcg_at_10']:.3f}  {m['average_precision']:.3f}±{s['average_precision']:.3f}")
print(f"\nSaved to {out_path}")
PYEOF
