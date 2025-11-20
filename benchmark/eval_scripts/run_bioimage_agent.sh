#!/bin/bash

# Common variables that applies to all bioimage agent evals
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR/.."
DATA_DIR="$BENCHMARK_DIR/../SciVisAgentBench-tasks/bioimage_data/data"
EVAL_MODEL="gpt-4o"
RUNNER_SCRIPT="$BENCHMARK_DIR/yaml_runner_napari_mcp.py"
# Use Anthropic config, can be changed if needed
CONFIG_FILE="$BENCHMARK_DIR/configs/napari_mcp/config_anthropic.json"


# Case 1: Basic Napari Functions
YAML_FILE="$BENCHMARK_DIR/eval_cases/napari/0_actions/eval_basic_napari_functions.yaml"
CASES_DIR="$BENCHMARK_DIR/../SciVisAgentBench-tasks/bioimage_data/0_actions/eval_basic_napari_functions"
OUTPUT_DIR="$BENCHMARK_DIR/test_results/napari_mcp/0_actions/eval_basic_napari_functions"

cd "$BENCHMARK_DIR"
python3 "$RUNNER_SCRIPT" \
    --config "$CONFIG_FILE" \
    --yaml "$YAML_FILE" \
    --cases "$CASES_DIR" \
    --data_dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --eval-model "$EVAL_MODEL" \
    --api-key "$OPENAI_API_KEY"


# Case 2: Workflows: Analysis Workflow
YAML_FILE="$BENCHMARK_DIR/eval_cases/napari/1_workflows/eval_analysis_workflows.yaml"
CASES_DIR="$BENCHMARK_DIR/../SciVisAgentBench-tasks/bioimage_data/1_workflows/eval_analysis_workflows"
OUTPUT_DIR="$BENCHMARK_DIR/test_results/napari_mcp/1_workflows/eval_analysis_workflows"

cd "$BENCHMARK_DIR"
python3 "$RUNNER_SCRIPT" \
    --config "$CONFIG_FILE" \
    --yaml "$YAML_FILE" \
    --cases "$CASES_DIR" \
    --data_dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --eval-model "$EVAL_MODEL" \
    --api-key "$OPENAI_API_KEY"


# Case 3: Workflows: Figure Recreation
YAML_FILE="$BENCHMARK_DIR/eval_cases/napari/1_workflows/eval_figure_recreation.yaml"
CASES_DIR="$BENCHMARK_DIR/../SciVisAgentBench-tasks/bioimage_data/1_workflows/eval_figure_recreation"
OUTPUT_DIR="$BENCHMARK_DIR/test_results/napari_mcp/1_workflows/eval_figure_recreation"

cd "$BENCHMARK_DIR"
python3 "$RUNNER_SCRIPT" \
    --config "$CONFIG_FILE" \
    --yaml "$YAML_FILE" \
    --cases "$CASES_DIR" \
    --data_dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --eval-model "$EVAL_MODEL" \
    --api-key "$OPENAI_API_KEY"


# Case 4: Workflows: Visualization Workflows
YAML_FILE="$BENCHMARK_DIR/eval_cases/napari/1_workflows/eval_visualization_workflows.yaml"
CASES_DIR="$BENCHMARK_DIR/../SciVisAgentBench-tasks/bioimage_data/1_workflows/eval_visualization_workflows"
OUTPUT_DIR="$BENCHMARK_DIR/test_results/napari_mcp/1_workflows/eval_visualization_workflows"

cd "$BENCHMARK_DIR"
python3 "$RUNNER_SCRIPT" \
    --config "$CONFIG_FILE" \
    --yaml "$YAML_FILE" \
    --cases "$CASES_DIR" \
    --data_dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --eval-model "$EVAL_MODEL" \
    --api-key "$OPENAI_API_KEY"