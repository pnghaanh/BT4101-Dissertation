#!/usr/bin/env bash
set -euo pipefail

# Run from repo root regardless of caller cwd
cd "$(dirname "$0")/../.."

RESULT_JSON="research_final/results/results.json"
RESULT_CSV="research_final/results/summary_table.csv"
RESULT_DIR="research_final/results"
MODEL_NAME="${MODEL_NAME:-gpt2}"
DEVICE="${DEVICE:-cpu}"
N_PROMPTS="${N_PROMPTS:-100}"
MAX_TOKENS="${MAX_TOKENS:-400}"

PYTHON="${HOME}/miniconda3/envs/clean_env/bin/python"

"${PYTHON}" -m research_final.scripts.run_comparison \
  --model_name "${MODEL_NAME}" \
  --device "${DEVICE}" \
  --n_prompts "${N_PROMPTS}" \
  --max_tokens "${MAX_TOKENS}" \
  --output_json "${RESULT_JSON}"

"${PYTHON}" research_final/scripts/plot_results.py \
  --input_json "${RESULT_JSON}" \
  --output_csv "${RESULT_CSV}" \
  --output_dir "${RESULT_DIR}"

echo "Done. Outputs:"
echo "  ${RESULT_JSON}"
echo "  ${RESULT_CSV}"
echo "  ${RESULT_DIR}/*.png"
