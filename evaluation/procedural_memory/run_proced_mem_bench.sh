#!/usr/bin/env bash
# Orchestrator for proced_mem_bench evaluation.
# Follows the same pattern as evaluation/retrieval_agent/run_test.sh.
#
# Usage:
#   ./run_proced_mem_bench.sh RESULT_POSTFIX RUN_TYPE TEST_TARGET
#
# Examples:
#   ./run_proced_mem_bench.sh exp1 ingest retrieval_agent
#   ./run_proced_mem_bench.sh exp1 search retrieval_agent
#   ./run_proced_mem_bench.sh exp1 delete retrieval_agent

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

usage() {
    echo "Usage: $0 RESULT_POSTFIX RUN_TYPE TEST_TARGET [options]"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files (e.g. exp1)"
    echo "  RUN_TYPE          ingest | search | delete"
    echo "  TEST_TARGET       memmachine | retrieval_agent | llm"
    echo
    echo "Options:"
    echo "  --result-dir DIR  Directory for output files (default: <script_dir>/result)"
    echo "  --concurrency N   Max concurrent tasks (default: 5 for ingest, 1 for search)"
    echo "  --search-limit N  Top-k results per query (default: 10, search only)"
    echo
    echo "Examples:"
    echo "  $0 exp1 ingest retrieval_agent"
    echo "  $0 exp1 search retrieval_agent --search-limit 20"
    echo "  $0 exp1 search retrieval_agent --result-dir /tmp/my_results"
    echo "  $0 exp1 delete retrieval_agent"
    exit 1
}

if [ "$#" -lt 3 ]; then
    usage
fi

RESULT_POSTFIX="$1"
RUN_TYPE="$2"
TEST_TARGET="$3"
shift 3

# Parse optional flags
CONCURRENCY=""
SEARCH_LIMIT=""
RESULT_DIR=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --result-dir)
            RESULT_DIR="$2"; shift 2 ;;
        --result-dir=*)
            RESULT_DIR="${1#*=}"; shift ;;
        --concurrency)
            CONCURRENCY="$2"; shift 2 ;;
        --concurrency=*)
            CONCURRENCY="${1#*=}"; shift ;;
        --search-limit)
            SEARCH_LIMIT="$2"; shift 2 ;;
        --search-limit=*)
            SEARCH_LIMIT="${1#*=}"; shift ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown option: $1"; usage ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../retrieval_agent/configuration.yml"
DATA_DIR="${SCRIPT_DIR}/../data/proced_mem_bench"
TRAJECTORIES_FILE="${DATA_DIR}/trajectories.json"
QUERIES_FILE="${DATA_DIR}/queries.json"

# Set PYTHONPATH to match run_test.sh
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/common/src:${REPO_ROOT}/packages/server/src:${REPO_ROOT}/packages/client/src${PYTHONPATH:+:${PYTHONPATH}}"

# Require configuration.yml
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: configuration.yml not found at '${CONFIG_FILE}'"
    echo "Copy from evaluation/retrieval_agent/configuration.yml"
    exit 1
fi

# Setup result directories -- use --result-dir if provided, else default to <script_dir>/result
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/result}"
mkdir -p "${RESULT_DIR}/final_score"

RESULT_FILE="${RESULT_DIR}/proced_mem_bench_${TEST_TARGET}_output_${RESULT_POSTFIX}.json"
EVAL_FILE="${RESULT_DIR}/proced_mem_bench_${TEST_TARGET}_eval_metrics_${RESULT_POSTFIX}.json"
FINAL_SCORE_FILE="${RESULT_DIR}/final_score/proced_mem_bench_${TEST_TARGET}_${RESULT_POSTFIX}.result"

case "$RUN_TYPE" in
    ingest)
        if [ ! -f "$TRAJECTORIES_FILE" ]; then
            echo "Error: Trajectory data not found at ${TRAJECTORIES_FILE}"
            echo "Download the AgentInstruct trajectories first."
            echo "See evaluation/procedural_memory/README.md for instructions."
            exit 1
        fi
        INGEST_CMD=(python -u "${SCRIPT_DIR}/proced_mem_bench_ingest.py"
            --data-path "$TRAJECTORIES_FILE"
            --config-path "$CONFIG_FILE"
        )
        if [ -n "$CONCURRENCY" ]; then
            INGEST_CMD+=(--concurrency "$CONCURRENCY")
        fi
        echo "=== Ingesting trajectories ==="
        "${INGEST_CMD[@]}"
        ;;

    search)
        if [ ! -f "$QUERIES_FILE" ]; then
            echo "Error: Query data not found at ${QUERIES_FILE}"
            echo "Download the benchmark queries first."
            echo "See evaluation/procedural_memory/README.md for instructions."
            exit 1
        fi
        rm -f "$RESULT_FILE" "$EVAL_FILE" "$FINAL_SCORE_FILE"

        SEARCH_CMD=(python -u "${SCRIPT_DIR}/proced_mem_bench_search.py"
            --data-path "$QUERIES_FILE"
            --eval-result-path "$RESULT_FILE"
            --test-target "$TEST_TARGET"
            --config-path "$CONFIG_FILE"
        )
        if [ -n "$CONCURRENCY" ]; then
            SEARCH_CMD+=(--concurrency "$CONCURRENCY")
        fi
        if [ -n "$SEARCH_LIMIT" ]; then
            SEARCH_CMD+=(--search-limit "$SEARCH_LIMIT")
        fi

        echo "=== Running search ==="
        "${SEARCH_CMD[@]}"

        echo "=== Running evaluation ==="
        python "${SCRIPT_DIR}/proced_mem_bench_evaluate.py" \
            --data-path "$RESULT_FILE" \
            --target-path "$EVAL_FILE"

        echo "=== Generating final scores ==="
        python "${REPO_ROOT}/evaluation/retrieval_agent/generate_scores.py" \
            --data-path "$EVAL_FILE" > "$FINAL_SCORE_FILE"
        cat "$FINAL_SCORE_FILE"
        ;;

    delete)
        if [ ! -f "$TRAJECTORIES_FILE" ]; then
            echo "Error: Trajectory data not found at ${TRAJECTORIES_FILE}"
            exit 1
        fi
        echo "=== Deleting benchmark data ==="
        python -u "${SCRIPT_DIR}/proced_mem_bench_delete.py" \
            --data-path "$TRAJECTORIES_FILE" \
            --config-path "$CONFIG_FILE"
        ;;

    *)
        echo "Unknown RUN_TYPE: $RUN_TYPE"
        usage
        ;;
esac
