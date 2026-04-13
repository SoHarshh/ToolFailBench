#!/usr/bin/env bash
# run_judge_eval.sh — Run LLM-as-judge on ToolFailBench results
#
# Usage:
#   ./scripts/run_judge_eval.sh validate          # 10-task sample on gemma4 (quick check)
#   ./scripts/run_judge_eval.sh single gemma4-31b  # Full 90 tasks for one model
#   ./scripts/run_judge_eval.sh all                # All 5 Tier 1 models (450 tasks, ~$8)

set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate

JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-5}"
RESULTS_DIR="results"
OUTPUT_DIR="results/judge"

# Kill any running vLLM to free GPU memory
pkill -f vllm 2>/dev/null && sleep 1 || true

# Check API key
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    source .env 2>/dev/null || true
    export ANTHROPIC_API_KEY
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ] || [ ${#ANTHROPIC_API_KEY} -lt 20 ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set or looks invalid. Update .env first."
    exit 1
fi

echo "Judge model: $JUDGE_MODEL"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# Map model IDs to their latest result files
latest_result() {
    local model="$1"
    ls -t "$RESULTS_DIR"/${model}_202*.json 2>/dev/null | head -1
}

run_judge() {
    local results_file="$1"
    local sample="${2:-}"
    local extra_args=""

    if [ -n "$sample" ]; then
        extra_args="--sample $sample"
    fi

    echo "============================================"
    echo "  Judging: $(basename "$results_file")"
    echo "============================================"

    python runners/run_judge.py \
        --results-file "$results_file" \
        --judge-model "$JUDGE_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --delay 0.3 \
        $extra_args
}

case "${1:-help}" in
    validate)
        echo "=== VALIDATION RUN: 10 tasks from gemma4-31b ==="
        RESULT_FILE=$(latest_result "gemma4-31b")
        if [ -z "$RESULT_FILE" ]; then
            echo "ERROR: No gemma4-31b results found in $RESULTS_DIR/"
            exit 1
        fi
        run_judge "$RESULT_FILE" 10
        ;;

    single)
        MODEL="${2:?Usage: $0 single <model-id>}"
        RESULT_FILE=$(latest_result "$MODEL")
        if [ -z "$RESULT_FILE" ]; then
            echo "ERROR: No results found for $MODEL in $RESULTS_DIR/"
            exit 1
        fi
        echo "=== FULL RUN: $MODEL (90 tasks) ==="
        run_judge "$RESULT_FILE"
        ;;

    all)
        echo "=== FULL RUN: All Tier 1 models ==="
        MODELS="qwen3.5-9b mistral-7b glm4-9b llama3.1-8b gemma4-31b"
        for m in $MODELS; do
            RESULT_FILE=$(latest_result "$m")
            if [ -z "$RESULT_FILE" ]; then
                echo "WARNING: No results for $m, skipping"
                continue
            fi
            run_judge "$RESULT_FILE"
            echo ""
        done
        echo "=== ALL DONE ==="
        echo "Judge results saved to $OUTPUT_DIR/"
        ;;

    help|*)
        echo "Usage:"
        echo "  $0 validate               # 10-task sample on gemma4 (~\$0.04)"
        echo "  $0 single <model-id>      # Full 90 tasks for one model (~\$1.70)"
        echo "  $0 all                    # All 5 Tier 1 models (~\$8)"
        echo ""
        echo "Models: qwen3.5-9b, mistral-7b, glm4-9b, llama3.1-8b, gemma4-31b"
        echo ""
        echo "Set JUDGE_MODEL env var to override (default: claude-sonnet-4-5)"
        ;;
esac
