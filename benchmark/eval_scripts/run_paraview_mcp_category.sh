#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR/.."
YAML_FILE="$BENCHMARK_DIR/eval_cases/paraview/category_specific_cases.yaml"
CONFIG_FILE="$BENCHMARK_DIR/configs/paraview_mcp/config_openai.json"
CASES_DIR="$BENCHMARK_DIR/../SciVisAgentBench-tasks/sci_volume_data"
OUTPUT_DIR="$BENCHMARK_DIR/test_results/paraview_mcp/category"
EVAL_MODEL="gpt-4o"
RUNNER_SCRIPT="$BENCHMARK_DIR/yaml_runner_paraview_mcp.py"

cd "$BENCHMARK_DIR"
python3 "$RUNNER_SCRIPT" \
    --config "$CONFIG_FILE" \
    --yaml "$YAML_FILE" \
    --cases "$CASES_DIR" \
    --output "$OUTPUT_DIR" \
    --eval-model "$EVAL_MODEL" \
    --api-key "$OPENAI_API_KEY"
