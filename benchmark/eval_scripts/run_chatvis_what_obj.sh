#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR/.."
YAML_FILE="$BENCHMARK_DIR/eval_cases/paraview/what_obj_cases_anonymized.yaml"
CASES_DIR="$BENCHMARK_DIR/anonymized_datasets"
EVAL_MODEL="gpt-5"
CONFIG_FILE="$BENCHMARK_DIR/configs/chatvis/config_anthropic.json"
OUTPUT_DIR="$BENCHMARK_DIR/test_results/chatvis/what_obj"
RUNNER_SCRIPT="$BENCHMARK_DIR/yaml_runner_chatvis.py"

cd "$BENCHMARK_DIR"
python3 "$RUNNER_SCRIPT" \
    --yaml "$YAML_FILE" \
    --cases "$CASES_DIR" \
    --output "$OUTPUT_DIR" \
    --config "$CONFIG_FILE" \
    --eval-model "$EVAL_MODEL" \
    --api-key "$OPENAI_API_KEY"
