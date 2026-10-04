#!/usr/bin/env bash
# run_final_eval.sh -- clean run of the pre/post procedural-memory comparison.
#
# Every run writes to a FRESH directory result/runs/<timestamp>/, so a stale
# file from an earlier run can never be picked up. result/runs/latest points
# at the newest run.
#
# USAGE (from anywhere):
#   ./run_final_eval.sh                 # all 40 queries, reuse ingested data
#   ./run_final_eval.sh --task easy_1   # one query
#   ./run_final_eval.sh --reingest      # wipe Postgres+Neo4j benchmark data, re-ingest first
#   ./run_final_eval.sh --clean         # delete ALL previous result/runs/* first
#
# STAGES
#   0  preflight: containers up, API key present, data present
#   1  (optional --reingest) delete + re-ingest episodic and procedural stores
#   2  PRE : baseline search   (episodic retrieval agent -> answer LLM)
#   3  POST: procedural search (grounded trajectory retrieval -> SAME answer LLM)
#   4  LLM judge on both
#   5  compare + report.md (per tier, retrieval modes, found-vs-none split, IR metrics)
set -Eeuo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/configuration.yml"
DATA_DIR="$REPO_ROOT/evaluation/data/proced_mem_bench"
TRAJECTORIES="$DATA_DIR/trajectories.json"
QUERIES="$DATA_DIR/queries.json"
RUNS_DIR="$SCRIPT_DIR/result/runs"

TASK_ID=""; REINGEST=false; CLEAN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)     TASK_ID="$2"; shift 2 ;;
    --reingest) REINGEST=true; shift ;;
    --clean)    CLEAN=true; shift ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1 (see --help)"; exit 1 ;;
  esac
done

banner() { printf '\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n  %s\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n' "$1"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# Python must resolve the repo packages regardless of where this is invoked
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/common/src:$REPO_ROOT/packages/server/src:$REPO_ROOT/packages/client/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR"   # scripts call load_dotenv(); configuration.yml is here

# ── 0. preflight ────────────────────────────────────────────────────────────
banner "0. Preflight"
[[ -f "$CONFIG" ]]       || die "missing $CONFIG"
[[ -f "$QUERIES" ]]      || die "missing $QUERIES"
[[ -f "$TRAJECTORIES" ]] || die "missing $TRAJECTORIES"
for c in memmachine-postgres memmachine-neo4j; do
  state=$(docker inspect -f '{{.State.Health.Status}}' "$c" 2>/dev/null || echo missing)
  [[ "$state" == "healthy" ]] || die "container $c is '$state' (start it with docker compose)"
  echo "  $c: healthy"
done
if [[ -z "${OPENAI_API_KEY:-}" ]] && ! grep -qs '^OPENAI_API_KEY=.\+' "$REPO_ROOT/.env" "$SCRIPT_DIR/.env"; then
  die "OPENAI_API_KEY not set (export it or put it in $REPO_ROOT/.env)"
fi
echo "  OPENAI_API_KEY: found"

if $CLEAN; then
  echo "  --clean: removing previous runs in $RUNS_DIR"
  rm -rf "${RUNS_DIR:?}"/*
fi

RUN_DIR="$RUNS_DIR/$(date +%Y%m%d-%H%M%S)${TASK_ID:+-$TASK_ID}"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$RUNS_DIR/latest"
echo "  run dir: $RUN_DIR"

# Query set (single task -> filtered file in the same {"queries": [...]} format)
if [[ -n "$TASK_ID" ]]; then
  RUN_QUERIES="$RUN_DIR/queries.json"
  python - "$QUERIES" "$TASK_ID" "$RUN_QUERIES" <<'PY'
import json, sys
src, tid, dst = sys.argv[1:]
qs = json.load(open(src))["queries"]
hit = [q for q in qs if q.get("query_id") == tid]
if not hit:
    sys.exit(f"query_id {tid!r} not found; e.g. {[q['query_id'] for q in qs[:8]]}")
json.dump({"queries": hit}, open(dst, "w"), indent=2)
print(f"  single task: {tid} -- {hit[0]['query_text']}")
PY
else
  RUN_QUERIES="$QUERIES"
fi

# ── 1. optional re-ingest ───────────────────────────────────────────────────
if $REINGEST; then
  banner "1. Re-ingest (wipes benchmark data in Postgres + Neo4j)"
  python -u proced_mem_bench_delete.py --data-path "$TRAJECTORIES" --config-path "$CONFIG"
  python -u proced_mem_bench_procedural_ingest.py --data-path "$TRAJECTORIES" --config-path "$CONFIG" --mode delete
  python -u proced_mem_bench_ingest.py --data-path "$TRAJECTORIES" --config-path "$CONFIG" --concurrency 5
  python -u proced_mem_bench_procedural_ingest.py --data-path "$TRAJECTORIES" --config-path "$CONFIG" --mode ingest
else
  banner "1. Re-ingest skipped (using data already in Postgres + Neo4j)"
fi
python -u proced_mem_bench_procedural_ingest.py --data-path "$TRAJECTORIES" --config-path "$CONFIG" --mode stats 2>/dev/null \
  | grep -E "^(Tools|Trajectories)" || true

# ── 2. PRE: baseline ────────────────────────────────────────────────────────
banner "2. PRE  -- baseline search (episodic retrieval agent)"
python -u proced_mem_bench_search.py --data-path "$RUN_QUERIES" \
  --eval-result-path "$RUN_DIR/pre_search.json" \
  --test-target retrieval_agent --config-path "$CONFIG" --concurrency 1

# ── 3. POST: procedural ─────────────────────────────────────────────────────
banner "3. POST -- procedural search (grounded trajectory retrieval)"
python -u proced_mem_bench_search.py --data-path "$RUN_QUERIES" \
  --eval-result-path "$RUN_DIR/post_search.json" \
  --test-target procedural --config-path "$CONFIG" --concurrency 1

# ── 4. judge ────────────────────────────────────────────────────────────────
banner "4. LLM judge"
python -u proced_mem_bench_judge.py judge --search-output "$RUN_DIR/pre_search.json" \
  --mode episodic --config-path "$CONFIG" --output "$RUN_DIR/pre_judge.json"
python -u proced_mem_bench_judge.py judge --search-output "$RUN_DIR/post_search.json" \
  --mode procedural --config-path "$CONFIG" --output "$RUN_DIR/post_judge.json"

# ── 5. compare + report ─────────────────────────────────────────────────────
banner "5. Compare"
python proced_mem_bench_judge.py compare \
  --baseline "$RUN_DIR/pre_judge.json" --procedural "$RUN_DIR/post_judge.json" \
  | tee "$RUN_DIR/compare.txt"

python - "$RUN_DIR" <<'PY'
import json, sys, collections
run = sys.argv[1]
pre  = {r["query_id"]: r for r in json.load(open(f"{run}/pre_judge.json"))["results"]}
post = {r["query_id"]: r for r in json.load(open(f"{run}/post_judge.json"))["results"]}
ids = [q for q in pre if q in post]

def mean(rs, k): return sum(r[k] for r in rs) / len(rs) if rs else float("nan")
def row(name, a, b):
    return (f"| {name} | {len(a)} | {mean(a,'judge_pass'):.1%} | {mean(b,'judge_pass'):.1%} "
            f"| {mean(a,'judge_mean_score'):.3f} | {mean(b,'judge_mean_score'):.3f} |")

L = ["# Pre/post procedural memory -- LLM judge report", "",
     f"Run dir: `{run}`", "",
     "| Slice | n | Pre pass | Post pass | Pre mean | Post mean |", "|---|---|---|---|---|---|",
     row("ALL", [pre[q] for q in ids], [post[q] for q in ids])]
for t in sorted({pre[q]["tier"] for q in ids}):
    s = [q for q in ids if pre[q]["tier"] == t]
    L.append(row(t, [pre[q] for q in s], [post[q] for q in s]))

modes = collections.Counter(post[q]["selected_tool"].split(":")[-1] for q in ids)
found = [q for q in ids if not post[q]["selected_tool"].endswith(":none")]
none_ = [q for q in ids if q not in found]
L += ["", "## Retrieval coverage (post)", "",
      f"Modes: {dict(modes)}", "",
      "| Slice | n | Pre pass | Post pass | Pre mean | Post mean |", "|---|---|---|---|---|---|",
      row("post retrieved a procedure", [pre[q] for q in found], [post[q] for q in found]),
      row("post retrieved nothing", [pre[q] for q in none_], [post[q] for q in none_])]

better = sum(post[q]["judge_mean_score"] > pre[q]["judge_mean_score"] for q in ids)
worse  = sum(post[q]["judge_mean_score"] < pre[q]["judge_mean_score"] for q in ids)
L += ["", f"Per query: post better={better}, worse={worse}, tie={len(ids)-better-worse}", "",
      "## IR metrics (source trajectories vs benchmark relevant ids)", "",
      "| Metric | Pre | Post |", "|---|---|---|"]
for k in ["p_at_1", "p_at_5", "ndcg_at_10", "average_precision"]:
    L.append(f"| {k} | {mean([pre[q] for q in ids],k):.3f} | {mean([post[q] for q in ids],k):.3f} |")
errs = sum(r["judge"].get("judge_error", False) for d in (pre, post) for r in d.values())
L += ["", f"Judge errors: {errs} (API failures scored as floor; should be 0)"]
if none_:
    L += ["", "## Queries where post retrieved nothing", ""]
    L += [f"- {q}: {post[q]['query_text']} -- subgoals={post[q].get('procedural_subgoals')}" for q in none_]
open(f"{run}/report.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
PY

banner "DONE"
echo "  report:  $RUN_DIR/report.md"
echo "  prompts: jq '.results[0].prompt_sent_to_llm' $RUN_DIR/post_judge.json"
