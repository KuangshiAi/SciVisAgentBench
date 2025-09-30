#!/bin/bash

# Run YAML-based test cases using the yaml_test_runner.py
# This script loads test cases from main_test_cases.yaml and runs them through MCP

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
YAML_FILE="$BENCHMARK_DIR/eval_cases/paraview/category_specific_cases.yaml"
CONFIG_FILE="$BENCHMARK_DIR/configs/paraview_mcp/config_openai.json"
CASES_DIR="$BENCHMARK_DIR/../SciVisAgentBench-tasks/sci_volume_data"
OUTPUT_DIR="$BENCHMARK_DIR/test_results/paraview_mcp/category"

# Default values
EVAL_MODEL="gpt-4o"
SPECIFIC_CASE=""
LIST_CASES=false
NO_EVAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --case)
            SPECIFIC_CASE="$2"
            shift 2
            ;;
        --list)
            LIST_CASES=true
            shift
            ;;
        --no-eval)
            NO_EVAL=true
            shift
            ;;
        --eval-model)
            EVAL_MODEL="$2"
            shift 2
            ;;
        --yaml)
            YAML_FILE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --cases)
            CASES_DIR="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run test cases from YAML configuration file using MCP."
            echo ""
            echo "Options:"
            echo "  --case CASE_NAME      Run a specific test case by name"
            echo "  --list                List available test cases and exit"
            echo "  --no-eval             Skip LLM-based evaluation"
            echo "  --eval-model MODEL    OpenAI model for evaluation (default: gpt-4o)"
            echo "  --yaml FILE           Path to YAML test cases file (default: main_test_cases.yaml)"
            echo "  --config FILE         Path to MCP config file (default: tiny_agent/config_openai.json)"
            echo "  --cases DIR           Path to cases directory (default: ../SciVisAgentBench-tasks/main)"
            echo "  --output, -o DIR      Output directory for results (default: test_results/yaml_mcp)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  OPENAI_API_KEY        OpenAI API key (required for evaluation)"
            echo ""
            echo "Examples:"
            echo "  $0                              # Run all test cases with evaluation"
            echo "  $0 --list                       # List available test cases"
            echo "  $0 --case bonsai                # Run only the bonsai test case"
            echo "  $0 --no-eval                    # Run all test cases without evaluation"
            echo "  $0 --eval-model gpt-4o-mini     # Use different model for evaluation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting YAML-based test runner..."
echo "YAML file: $YAML_FILE"
echo "Config file: $CONFIG_FILE"
echo "Cases directory: $CASES_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Evaluation model: $EVAL_MODEL"

# Check if required files exist
if [ ! -f "$YAML_FILE" ]; then
    echo "ERROR: YAML test cases file not found: $YAML_FILE"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: MCP config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$CASES_DIR" ]; then
    echo "ERROR: Cases directory not found: $CASES_DIR"
    exit 1
fi

# Check if yaml_runner_paraview_mcp.py exists
RUNNER_SCRIPT="$BENCHMARK_DIR/yaml_runner_paraview_mcp.py"
if [ ! -f "$RUNNER_SCRIPT" ]; then
    echo "ERROR: YAML test runner script not found: $RUNNER_SCRIPT"
    exit 1
fi

# Check for OpenAI API key if evaluation is enabled
if [ "$NO_EVAL" != true ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY environment variable is not set"
    echo "Evaluation will be skipped. Set the API key to enable LLM-based evaluation."
    NO_EVAL=true
fi

# Build command arguments
CMD_ARGS=(
    "--config" "$CONFIG_FILE"
    "--yaml" "$YAML_FILE"
    "--cases" "$CASES_DIR"
    "--output" "$OUTPUT_DIR"
    "--eval-model" "$EVAL_MODEL"
)

if [ "$LIST_CASES" = true ]; then
    CMD_ARGS+=("--list")
fi

if [ "$NO_EVAL" = true ]; then
    CMD_ARGS+=("--no-eval")
fi

if [ -n "$SPECIFIC_CASE" ]; then
    CMD_ARGS+=("--case" "$SPECIFIC_CASE")
fi

if [ -n "$OPENAI_API_KEY" ]; then
    CMD_ARGS+=("--api-key" "$OPENAI_API_KEY")
fi

echo ""
echo "Running command:"
echo "python3 $RUNNER_SCRIPT ${CMD_ARGS[*]}"
echo ""

# Run the YAML test runner
cd "$BENCHMARK_DIR"
python3 "$RUNNER_SCRIPT" "${CMD_ARGS[@]}"

# Check if the command completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "YAML TEST RUNNER COMPLETED SUCCESSFULLY"
    echo "=========================================="
    if [ "$LIST_CASES" != true ]; then
        echo "Test results are saved in the test_results/yaml_mcp directory"
        echo "Individual case results are also saved in each case's evaluation_results/mcp directory"
    fi
else
    echo ""
    echo "=========================================="
    echo "YAML TEST RUNNER FAILED"
    echo "=========================================="
    echo "Please check the error messages above."
    exit 1
fi