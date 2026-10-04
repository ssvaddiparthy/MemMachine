#!/usr/bin/env bash
# run_eval.sh — full procedural memory eval pipeline
#
# USAGE:
#   # Run everything (all queries)
#   ./run_eval.sh
#
#   # Run a single task by query_id (e.g. easy_1)
#   ./run_eval.sh --task easy_1
#
#   # Skip ingest if data is already in Neo4j
#   ./run_eval.sh --skip-ingest
#
#   # Full run, specific task
#   ./run_eval.sh --task medium_3 --skip-ingest
#
# WHAT IT DOES:
#   Stage 1: Ingest trajectories into episodic memory (Postgres/pgvector)
#   Stage 2: Ingest trajectories into Neo4j procedure graph
#   Stage 3: Run Louvain community detection
#   Stage 4a: Baseline search (episodic only, no procedure graph)
#   Stage 4b: Procedural search (Steiner tree composition)
#   Stage 5:  Print prompt comparison (what the LLM saw before vs after)
#   Stage 6:  Compute IR metrics and show comparison table
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_DIR="$SCRIPT_DIR"
DATA_DIR="$REPO_ROOT/evaluation/data/proced_mem_bench"
CONFIG="$SCRIPT_DIR/configuration.yml"
RESULT_DIR="$EVAL_DIR/result"

TRAJECTORIES="$DATA_DIR/trajectories.json"
QUERIES="$DATA_DIR/queries.json"

BASELINE_OUTPUT="$RESULT_DIR/proced_mem_bench_retrieval_agent_output_baseline.json"
PROCEDURAL_OUTPUT="$RESULT_DIR/proced_mem_bench_procedural_output.json"
BASELINE_METRICS="$RESULT_DIR/proced_mem_bench_retrieval_agent_eval_metrics_baseline.json"
PROCEDURAL_METRICS="$RESULT_DIR/proced_mem_bench_procedural_eval_metrics.json"

SKIP_INGEST=false
TASK_ID=""

# ── arg parsing ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-ingest) SKIP_INGEST=true; shift ;;
    --task)        TASK_ID="$2"; shift 2 ;;
    -h|--help)
      sed -n '/^# /,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$RESULT_DIR"

# All python -m calls must run from repo root so package imports resolve
cd "$REPO_ROOT"

# If a single task is requested, extract it into a temp queries file
if [[ -n "$TASK_ID" ]]; then
  echo "==> Single-task mode: $TASK_ID"
  TASK_QUERIES="$RESULT_DIR/task_${TASK_ID}_queries.json"
  python3 - <<PYEOF
import json, sys
with open("$QUERIES") as f:
    data = json.load(f)
if isinstance(data, dict):
    all_queries = [q for qs in data.values() for q in qs]
else:
    all_queries = data
match = [q for q in all_queries if q.get("query_id") == "$TASK_ID"]
if not match:
    print(f"ERROR: query_id '$TASK_ID' not found. Available: {[q.get('query_id') for q in all_queries[:20]]}")
    sys.exit(1)
with open("$TASK_QUERIES", "w") as f:
    json.dump(match, f, indent=2)
print(f"Extracted query: {match[0].get('query_text','')}")
PYEOF
  QUERIES_TO_RUN="$TASK_QUERIES"
  BASELINE_OUTPUT="$RESULT_DIR/task_${TASK_ID}_baseline.json"
  PROCEDURAL_OUTPUT="$RESULT_DIR/task_${TASK_ID}_procedural.json"
else
  QUERIES_TO_RUN="$QUERIES"
fi

# ── Stage 1: Episodic ingest ─────────────────────────────────────────────────
if [[ "$SKIP_INGEST" == "false" ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Stage 1: Episodic ingest → Postgres/pgvector"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python -m evaluation.procedural_memory.proced_mem_bench_ingest \
    --data-path "$TRAJECTORIES" \
    --config-path "$CONFIG" \
    --concurrency 5

  # ── Stage 2: Procedure graph ingest ────────────────────────────────────────
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Stage 2: Procedure graph ingest → Neo4j"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python -m evaluation.procedural_memory.proced_mem_bench_procedural_ingest \
    --data-path "$TRAJECTORIES" \
    --config-path "$CONFIG" \
    --mode ingest

  # ── Stage 3: Louvain community detection ───────────────────────────────────
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Stage 3: Louvain community detection"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python -m evaluation.procedural_memory.proced_mem_bench_procedural_ingest \
    --data-path "$TRAJECTORIES" \
    --config-path "$CONFIG" \
    --mode detect

  # ── Graph stats ────────────────────────────────────────────────────────────
  echo ""
  echo "  Graph stats:"
  python -m evaluation.procedural_memory.proced_mem_bench_procedural_ingest \
    --data-path "$TRAJECTORIES" \
    --config-path "$CONFIG" \
    --mode stats
else
  echo "==> Skipping ingest (--skip-ingest)"
  echo "  Graph stats:"
  python -m evaluation.procedural_memory.proced_mem_bench_procedural_ingest \
    --data-path "$TRAJECTORIES" \
    --config-path "$CONFIG" \
    --mode stats
fi

# ── Stage 4a: Baseline search (episodic only) ────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stage 4a: Baseline search — episodic only (no procedure graph)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m evaluation.procedural_memory.proced_mem_bench_search \
  --data-path "$QUERIES_TO_RUN" \
  --eval-result-path "$BASELINE_OUTPUT" \
  --test-target retrieval_agent \
  --config-path "$CONFIG" \
  --concurrency 1

# ── Stage 4b: Procedural search (Steiner tree) ───────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stage 4b: Procedural search — Steiner tree composition"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m evaluation.procedural_memory.proced_mem_bench_procedural_ingest \
  --data-path "$QUERIES_TO_RUN" \
  --config-path "$CONFIG" \
  --mode search \
  --result-path "$PROCEDURAL_OUTPUT"

# ── Stage 5: Prompt diff ─────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stage 5: Prompt comparison (what did the LLM see?)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
BASELINE_OUTPUT="$BASELINE_OUTPUT" PROCEDURAL_OUTPUT="$PROCEDURAL_OUTPUT" python3 - <<'PYEOF'
import json, textwrap, sys, os

baseline_path = os.environ.get("BASELINE_OUTPUT", "")
procedural_path = os.environ.get("PROCEDURAL_OUTPUT", "")

try:
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(procedural_path) as f:
        procedural = json.load(f)
except Exception as e:
    print(f"  (prompt diff skipped: {e})")
    sys.exit(0)

# Baseline: episodes passed as <memories> to the answer LLM
b_items = baseline if isinstance(baseline, list) else baseline.get("results", [])
p_items = procedural if isinstance(procedural, list) else procedural.get("results", [])

shown = 0
for b in b_items[:3]:  # show first 3 queries
    qid = b.get("query_id") or b.get("id", "?")
    qtxt = b.get("query_text") or b.get("query", "?")
    memories = b.get("memories") or b.get("retrieved_episodes", [])
    p_match = next((p for p in p_items if (p.get("query_id") or p.get("id")) == qid), None)

    print(f"\n  Query: {qtxt}")
    print(f"  {'─'*60}")
    print(f"  BEFORE (episodic memories sent to LLM):")
    if memories:
        for m in memories[:3]:
            content = m.get("content", m.get("text", str(m)))[:120]
            print(f"    • {content}")
        if len(memories) > 3:
            print(f"    ... ({len(memories)-3} more)")
    else:
        print("    (no memories captured in output)")

    if p_match:
        steps = p_match.get("composed_steps", [])
        seq = p_match.get("tool_sequence", [])
        print(f"\n  AFTER (Steiner-composed procedure sent to LLM):")
        if seq:
            print(f"    Tool sequence: {' → '.join(seq)}")
        if steps:
            for s in steps[:5]:
                tool = s.get("tool_name", "?")
                params = s.get("parameters", {})
                print(f"    • {tool}({params})")
            if len(steps) > 5:
                print(f"    ... ({len(steps)-5} more steps)")
        else:
            print("    (no composition found)")
    shown += 1

if shown == 0:
    print("  (no results to compare)")
PYEOF

# ── Stage 6: IR metrics comparison ───────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stage 6: IR metrics comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m evaluation.procedural_memory.proced_mem_bench_evaluate \
  --eval-result-path "$BASELINE_OUTPUT" \
  --output-path "$BASELINE_METRICS" \
  --label "Baseline (episodic)" 2>/dev/null || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DONE — results in $RESULT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Baseline output:    $BASELINE_OUTPUT"
echo "  Procedural output:  $PROCEDURAL_OUTPUT"
