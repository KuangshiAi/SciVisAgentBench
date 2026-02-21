# Claude Code Agent Configuration

This directory contains configuration files for testing Claude Code as a general-purpose coding agent on the SciVisAgentBench benchmark.

## Overview

The Claude Code agent uses the Claude CLI to complete scientific visualization tasks without specialized MCP servers or pre-built tools. It receives natural language task descriptions and figures out how to interact with visualization packages (ParaView, Napari, GMX-VMD, etc.).

## Prerequisites

1. **Claude Code CLI**: Install and authenticate the Claude CLI
   ```bash
   # Check if Claude Code is installed
   claude --version
   ```

2. **Python Environment**: Set up environment with visualization packages
   ```bash
   # Using conda (recommended)
   conda create -n paraview_mcp python=3.10
   conda activate paraview_mcp
   conda install -c conda-forge paraview numpy scipy matplotlib
   pip install napari
   ```

3. **Benchmark Data**: Download test cases from HuggingFace
   ```bash
   # From the SciVisAgentBench root directory
   huggingface-cli download SciVis/SciVisAgentBench-tasks --repo-type dataset --local-dir SciVisAgentBench-tasks
   ```

## Configuration Options

### config.json

```json
{
  "provider": "anthropic",              // LLM provider
  "model": "claude-sonnet-4-5",         // Claude model to use
  "eval_mode": "generic",                // Evaluation mode (generic, mcp, etc.)
  "agent_name": "claude_code",           // Agent identifier
  "experiment_number": "exp1",           // Experiment number for tracking
  "timeout_per_task": 600,               // Max time per task in seconds
  "claude_code_path": "claude",          // Path to Claude CLI (or just "claude" if in PATH)
  "preserve_workdir": false,             // Keep working directories for debugging
  "environment": {                       // Environment specification (optional)
    "type": "conda",
    "name": "paraview_mcp",
    "packages": ["paraview", "napari", "numpy", "scipy"]
  },
  "price": {                             // Pricing for cost calculation
    "input_per_1m_tokens": "$3.00",
    "output_per_1m_tokens": "$15.00"
  }
}
```

### Configuration Parameters

- **provider**: LLM provider name (anthropic, openai, etc.)
- **model**: Model identifier (claude-sonnet-4-5, claude-opus-4-6, etc.)
- **eval_mode**: Evaluation mode, set to "generic" for Claude Code
- **agent_name**: Identifier for this agent (used in result filenames)
- **experiment_number**: Experiment identifier for tracking multiple runs
- **timeout_per_task**: Maximum execution time per task (default: 600 seconds)
- **claude_code_path**: Path to Claude Code CLI executable
  - Use "claude" if it's in your PATH
  - Use absolute path like "/usr/local/bin/claude" if needed
- **preserve_workdir**: Whether to keep working directories after tests (useful for debugging)
- **environment**: Environment specification (informational only)
- **price**: Token pricing for cost calculation
  - Use current Anthropic pricing for your model
  - Sonnet 4.5: $3/1M input, $15/1M output
  - Opus 4.6: $15/1M input, $75/1M output

## Usage

### Running a Single Test Case

```bash
cd SciVisAgentBench

python -m benchmark.evaluation_framework.run_evaluation \
    --agent claude_code \
    --config benchmark/configs/claude_code/config.json \
    --yaml benchmark/eval_cases/paraview/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main \
    --case bonsai \
    --eval-model gpt-4o
```

### Running Multiple Test Cases

```bash
# Run all main ParaView cases
python -m benchmark.evaluation_framework.run_evaluation \
    --agent claude_code \
    --config benchmark/configs/claude_code/config.json \
    --yaml benchmark/eval_cases/paraview/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main \
    --eval-model gpt-4o
```

### Running Different Benchmarks

```bash
# Bioimage data benchmark (Napari)
python -m benchmark.evaluation_framework.run_evaluation \
    --agent claude_code \
    --config benchmark/configs/claude_code/config.json \
    --yaml benchmark/eval_cases/napari/bioimage_data_cases.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data \
    --eval-model gpt-4o

# Molecular visualization (GMX-VMD)
python -m benchmark.evaluation_framework.run_evaluation \
    --agent claude_code \
    --config benchmark/configs/claude_code/config.json \
    --yaml benchmark/eval_cases/gmx_vmd/molecular_vis_cases.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model gpt-4o
```

## Output Structure

Results are saved in the test case directory:

```
SciVisAgentBench-tasks/main/bonsai/
├── results/
│   └── claude_code_claude-sonnet-4-5_exp1/
│       ├── bonsai.pvsm           # Generated state file
│       └── bonsai.png             # Screenshot (if generated)
├── test_results/
│   └── generic/
│       └── test_result_*.json     # Execution metadata
└── evaluation_results/
    └── generic/
        └── evaluation_*.json      # Evaluation scores
```

### Result Files

**test_result_*.json**: Execution metadata
- `status`: "completed" or "failed"
- `duration`: Execution time in seconds
- `token_usage`: Input/output token counts
- `cost`: Monetary cost breakdown
- `response`: Claude Code's output
- `output_files`: Paths to generated files

**evaluation_*.json**: Evaluation scores
- `vision_score`: LLM judge score for visual quality (0-5)
- `efficiency_score`: Based on time and token usage
- `total_score`: Combined score
- `rubrics`: Individual rubric scores
- `feedback`: LLM judge feedback

## Troubleshooting

### Claude CLI Not Found

```bash
# Check if Claude is installed
which claude

# If not found, install it or set claude_code_path in config.json
{
  "claude_code_path": "/path/to/claude"
}
```

### Timeout Issues

For complex tasks, increase the timeout:

```json
{
  "timeout_per_task": 1200  // 20 minutes
}
```

### Missing Output Files

Enable debug mode to keep working directories:

```json
{
  "preserve_workdir": true
}
```

Then check the working directory for generated files.

### Token Usage Not Captured

Token usage is extracted from Claude Code's output. If not available, it will be estimated based on output length. The `token_source` field in results indicates whether tokens are from "claude_output" or "estimated".

## Evaluation Models

The `--eval-model` parameter specifies which LLM to use for evaluation (vision rubrics). Common options:

- `gpt-4o`: OpenAI GPT-4 with vision (recommended)
- `gpt-4-turbo`: OpenAI GPT-4 Turbo
- `claude-sonnet-4-5`: Anthropic Claude Sonnet
- `claude-opus-4-6`: Anthropic Claude Opus (more expensive but higher quality)

## Cost Estimation

Based on the pricing configuration:

**Sonnet 4.5**: ~$0.05-0.20 per task (depends on complexity)
- Input: $3/1M tokens
- Output: $15/1M tokens

**Opus 4.6**: ~$0.25-1.00 per task
- Input: $15/1M tokens
- Output: $75/1M tokens

For the full main benchmark (~50 tasks), expect:
- Sonnet: $2.50 - $10
- Opus: $12.50 - $50

## Advanced Usage

### Running with Different Models

Create additional config files for different models:

```bash
# config_opus.json
{
  "model": "claude-opus-4-6",
  "experiment_number": "opus_exp1",
  "price": {
    "input_per_1m_tokens": "$15.00",
    "output_per_1m_tokens": "$75.00"
  }
}

# Run with Opus
python -m benchmark.evaluation_framework.run_evaluation \
    --agent claude_code \
    --config benchmark/configs/claude_code/config_opus.json \
    ...
```

### Comparing Multiple Runs

Use different experiment numbers to track multiple runs:

```json
{
  "experiment_number": "exp1"  // First run
}

{
  "experiment_number": "exp2"  // Second run
}
```

Results will be saved separately:
- `results/claude_code_claude-sonnet-4-5_exp1/`
- `results/claude_code_claude-sonnet-4-5_exp2/`

## Known Limitations

1. **Token Tracking**: Token usage may not be precisely trackable if Claude Code doesn't output detailed usage statistics. In this case, tokens will be estimated.

2. **Interactive Tasks**: Claude Code cannot handle tasks requiring interactive input during execution.

3. **Environment Setup**: Claude Code needs to figure out how to use visualization tools from scratch. Pre-installed packages help but are not strictly required.

4. **Error Recovery**: Limited error recovery - if Claude Code fails, the task is marked as failed without retry.

## Support

For issues or questions:
- Claude Code: https://github.com/anthropics/claude-code
- SciVisAgentBench: https://github.com/[your-repo]/SciVisAgentBench
