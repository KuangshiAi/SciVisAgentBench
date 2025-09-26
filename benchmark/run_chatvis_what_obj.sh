#!/bin/bash

# Run YAML-based test cases using the chatvis_yaml_test_runner.py
# This script loads test cases from main_test_cases.yaml and runs them through ChatVis (pvpython)

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
YAML_FILE="$BENCHMARK_DIR/eval_cases/paraview/sci_volume_test_cases_anonymized.yaml"
CASES_DIR="$BENCHMARK_DIR/anonymized_datasets"

# Default values
EVAL_MODEL="gpt-5"
MODEL="gpt-4.1"
SPECIFIC_CASE=""
LIST_CASES=false
NO_EVAL=false
OUTPUT_DIR="$BENCHMARK_DIR/test_results/chatvis/what_obj"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --yaml|-y)
            YAML_FILE="$2"
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
        --case)
            SPECIFIC_CASE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --eval-model)
            EVAL_MODEL="$2"
            shift 2
            ;;
        --no-eval)
            NO_EVAL=true
            shift
            ;;
        --list)
            LIST_CASES=true
            shift
            ;;
        --api-key)
            OPENAI_API_KEY="$2"
            export OPENAI_API_KEY
            shift 2
            ;;
        --help|-h)
            echo "ChatVis YAML Test Runner for SciVisAgentBench"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --yaml, -y FILE       YAML test cases file (default: main_test_cases.yaml)"
            echo "  --cases DIR           Cases directory (default: ../SciVisAgentBench-tasks/main)"
            echo "  --output, -o DIR      Output directory (default: test_results/chatvis_yaml)"
            echo "  --case NAME           Run specific test case by name"
            echo "  --model MODEL         OpenAI model for script generation (default: gpt-4o)"
            echo "  --eval-model MODEL    OpenAI model for evaluation (default: gpt-4o)"
            echo "  --no-eval             Skip LLM-based evaluation"
            echo "  --list                List available test cases and exit"
            echo "  --api-key KEY         OpenAI API key (can also use OPENAI_API_KEY env var)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  OPENAI_API_KEY        Required for both script generation and evaluation"
            echo ""
            echo "Requirements:"
            echo "  - ParaView with pvpython in PATH or standard locations"
            echo "  - OpenAI API key for script generation"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all test cases with default settings"
            echo "  $0 --case bonsai                     # Run only the bonsai test case"
            echo "  $0 --no-eval                         # Run all cases but skip evaluation"
            echo "  $0 --list                            # List available test cases"
            echo "  $0 --model gpt-4 --eval-model gpt-4o # Use different models for generation and evaluation"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if files exist
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: YAML file not found: $YAML_FILE"
    echo "Please ensure the test cases YAML file exists."
    exit 1
fi

if [ ! -d "$CASES_DIR" ]; then
    echo "Error: Cases directory not found: $CASES_DIR"
    echo "Please ensure the cases directory exists."
    exit 1
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ] && [ "$LIST_CASES" = false ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "ChatVis requires OpenAI API key for script generation."
    echo "Set it with: export OPENAI_API_KEY='your-api-key-here'"
    echo "Or use: $0 --api-key 'your-api-key-here'"
    exit 1
fi

# Check if pvpython is available
PVPYTHON_FOUND=false
PVPYTHON_PATHS=(
    "/Applications/ParaView-5.13.3.app/Contents/bin/pvpython"
    "/usr/local/bin/pvpython"
    "pvpython"
)

for pvpython_path in "${PVPYTHON_PATHS[@]}"; do
    if command -v "$pvpython_path" &> /dev/null; then
        PVPYTHON_FOUND=true
        echo "Found pvpython at: $pvpython_path"
        break
    fi
done

if [ "$PVPYTHON_FOUND" = false ] && [ "$LIST_CASES" = false ]; then
    echo "Warning: pvpython not found in standard locations."
    echo "Please ensure ParaView is installed and pvpython is accessible."
    echo "Common locations:"
    echo "  - macOS: /Applications/ParaView-X.X.X.app/Contents/bin/pvpython"
    echo "  - Linux: /usr/local/bin/pvpython or /opt/paraview/bin/pvpython"
    echo "  - Or add pvpython to your PATH"
    echo ""
    echo "Continuing anyway (pvpython will be checked during execution)..."
fi

echo "Starting ChatVis YAML test runner..."
echo "YAML file: $YAML_FILE"
echo "Cases directory: $CASES_DIR"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$SPECIFIC_CASE" ]; then
    echo "Specific case: $SPECIFIC_CASE"
fi
if [ "$NO_EVAL" = false ]; then
    echo "Generation model: $MODEL"
    echo "Evaluation model: $EVAL_MODEL"
else
    echo "Generation model: $MODEL"
    echo "Evaluation: DISABLED"
fi
echo ""

# Build the command
CMD="python3 yaml_runner_chatvis.py"
CMD="$CMD --yaml \"$YAML_FILE\""
CMD="$CMD --cases \"$CASES_DIR\""
CMD="$CMD --output \"$OUTPUT_DIR\""
CMD="$CMD --model \"$MODEL\""

if [ -n "$SPECIFIC_CASE" ]; then
    CMD="$CMD --case \"$SPECIFIC_CASE\""
fi

if [ "$NO_EVAL" = true ]; then
    CMD="$CMD --no-eval"
else
    CMD="$CMD --eval-model \"$EVAL_MODEL\""
fi

if [ "$LIST_CASES" = true ]; then
    CMD="$CMD --list"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    CMD="$CMD --api-key \"$OPENAI_API_KEY\""
fi

# Change to the benchmark directory and run the command
cd "$BENCHMARK_DIR"
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "ChatVis YAML test runner completed."