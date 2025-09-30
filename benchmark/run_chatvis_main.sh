#!/bin/bash

# Run YAML-based test cases using the chatvis_yaml_test_runner.py
# This script loads test cases from main_test_cases.yaml and runs them through ChatVis (pvpython)

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
YAML_FILE="$BENCHMARK_DIR/eval_cases/paraview/main_test_cases.yaml"
CASES_DIR="$BENCHMARK_DIR/../SciVisAgentBench-tasks/main"

# Default values
EVAL_MODEL="gpt-5"
CONFIG_FILE="$BENCHMARK_DIR/configs/chatvis/config_anthropic.json"
SPECIFIC_CASE=""
LIST_CASES=false
NO_EVAL=false
OUTPUT_DIR="$BENCHMARK_DIR/test_results/chatvis/main"

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
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --case)
            SPECIFIC_CASE="$2"
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
            echo "  --config, -c FILE     Configuration JSON file (supports OpenAI, Anthropic, HuggingFace)"
            echo "  --case NAME           Run specific test case by name"
            echo "  --model MODEL         Model for script generation (default: gpt-4o, overridden by config)"
            echo "  --eval-model MODEL    Model for evaluation (default: gpt-4o)"
            echo "  --no-eval             Skip LLM-based evaluation"
            echo "  --list                List available test cases and exit"
            echo "  --api-key KEY         API key (can also use environment variables)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  OPENAI_API_KEY        For OpenAI models"
            echo "  ANTHROPIC_API_KEY     For Anthropic Claude models"
            echo "  HF_TOKEN              For Hugging Face models"
            echo ""
            echo "Configuration Files:"
            echo "  configs/chatvis/config_openai.json     - OpenAI GPT models"
            echo "  configs/chatvis/config_anthropic.json  - Anthropic Claude models"
            echo "  configs/chatvis/config_hf.json         - Hugging Face models"
            echo ""
            echo "Requirements:"
            echo "  - ParaView with pvpython in PATH or standard locations"
            echo "  - API key for the chosen provider"
            echo ""
            echo "Examples:"
            echo "  $0                                       # Run all test cases with default OpenAI settings"
            echo "  $0 --config configs/chatvis/config_anthropic.json  # Use Anthropic Claude"
            echo "  $0 --config configs/chatvis/config_hf.json         # Use Hugging Face models"
            echo "  $0 --case bonsai                        # Run only the bonsai test case"
            echo "  $0 --no-eval                            # Run all cases but skip evaluation"
            echo "  $0 --list                               # List available test cases"
            echo "  $0 --model gpt-4 --eval-model gpt-4o    # Use different models (without config file)"
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

# Check config file if provided
if [ -n "$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Available config files:"
    if [ -d "$BENCHMARK_DIR/configs/chatvis" ]; then
        ls -1 "$BENCHMARK_DIR/configs/chatvis/"
    fi
    exit 1
fi

# Check API keys (if no config file is provided, default to OpenAI requirements)
if [ -z "$CONFIG_FILE" ] && [ -z "$OPENAI_API_KEY" ] && [ "$LIST_CASES" = false ]; then
    echo "Error: No configuration file provided and OPENAI_API_KEY environment variable is not set."
    echo "Either:"
    echo "  1. Use a config file: $0 --config configs/chatvis/config_openai.json"
    echo "  2. Set environment variable: export OPENAI_API_KEY='your-api-key-here'"
    echo "  3. Use command line: $0 --api-key 'your-api-key-here'"
    echo ""
    echo "Available config files:"
    if [ -d "$BENCHMARK_DIR/configs/chatvis" ]; then
        ls -1 "$BENCHMARK_DIR/configs/chatvis/"
    fi
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
if [ -n "$CONFIG_FILE" ]; then
    echo "Configuration file: $CONFIG_FILE"
fi
if [ -n "$SPECIFIC_CASE" ]; then
    echo "Specific case: $SPECIFIC_CASE"
fi
if [ "$NO_EVAL" = false ]; then
    if [ -n "$CONFIG_FILE" ]; then
        echo "Generation: Using config file settings"
    else
        echo "Generation model: $MODEL"
    fi
    echo "Evaluation model: $EVAL_MODEL"
else
    if [ -n "$CONFIG_FILE" ]; then
        echo "Generation: Using config file settings"
    else
        echo "Generation model: $MODEL"
    fi
    echo "Evaluation: DISABLED"
fi
echo ""

# Build the command
CMD="python3 yaml_runner_chatvis.py"
CMD="$CMD --yaml \"$YAML_FILE\""
CMD="$CMD --cases \"$CASES_DIR\""
CMD="$CMD --output \"$OUTPUT_DIR\""

# Add config file if provided (this will override model settings)
if [ -n "$CONFIG_FILE" ]; then
    CMD="$CMD --config \"$CONFIG_FILE\""
fi

# Add model only if no config file is provided (config file takes precedence)
if [ -z "$CONFIG_FILE" ]; then
    CMD="$CMD --model \"$MODEL\""
fi

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

# Add API key only if provided (mainly for backward compatibility)
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