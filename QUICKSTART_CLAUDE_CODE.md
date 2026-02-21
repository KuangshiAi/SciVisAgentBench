# Claude Code Agent - Quick Start Guide

Get started with testing Claude Code on SciVisAgentBench in 5 minutes.

## Prerequisites

1. **Claude Code CLI** installed and authenticated
   ```bash
   claude --version
   ```

2. **Python environment** with visualization packages
   ```bash
   conda create -n scivis_bench python=3.10
   conda activate scivis_bench
   conda install -c conda-forge paraview numpy scipy matplotlib
   pip install napari
   ```

3. **Test data** downloaded from HuggingFace
   ```bash
   cd SciVisAgentBench
   huggingface-cli download SciVisAgentBench/SciVisAgentBench-tasks \
       --repo-type dataset \
       --local-dir SciVisAgentBench-tasks
   ```

4. **OpenAI API key** for evaluation (optional but recommended)
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

## Quick Test

Test that everything is working:

```bash
cd SciVisAgentBench

# Run integration tests
python benchmark/test_claude_code_agent.py
# Expected: ✓ All tests passed!

# Run a single simple task
python benchmark/run_claude_code_eval.py \
    --config benchmark/configs/claude_code/config.json \
    --test_file benchmark/eval_cases/paraview/main_cases.yaml \
    --case_name engine
```

Expected output:
```
============================================================
Claude Code Agent Evaluation Runner
============================================================
...
✓ Task completed in 71.2s
✓ Found output files: state, image
```

## Run Full Benchmark

```bash
# Run all main ParaView cases
python benchmark/run_claude_code_eval.py \
    --config benchmark/configs/claude_code/config.json \
    --test_file benchmark/eval_cases/paraview/main_cases.yaml \
    --eval-model gpt-4o
```

This will:
- Run ~38 visualization tasks
- Generate state files and screenshots
- Evaluate with LLM judge (if --eval-model provided)
- Save results with metrics

Estimated time: 30-60 minutes
Estimated cost: $2-5 (with Sonnet 4.5)

## Check Results

Results are saved in each test case directory:

```bash
# View generated files
ls SciVisAgentBench-tasks/main/engine/results/claude_code_*/
# engine.pvsm (state file)
# engine.png (screenshot)

# View test result (execution metadata)
cat SciVisAgentBench-tasks/main/engine/test_results/generic/test_result_*.json

# View evaluation scores (if --eval-model was used)
cat SciVisAgentBench-tasks/main/engine/evaluation_results/generic/evaluation_result_*.json
```

### Result Structure

After running, each test case will have:

```
SciVisAgentBench-tasks/main/engine/
├── data/                               # Input datasets
├── results/
│   └── claude_code_claude-sonnet-4-5_exp_default/
│       ├── engine.pvsm                 # ParaView state file
│       └── engine.png                  # Screenshot (1920x1080)
├── test_results/
│   └── generic/
│       └── test_result_*.json          # Execution results (tokens, cost, duration)
└── evaluation_results/                 # Created if --eval-model is used
    └── generic/
        └── evaluation_result_*.json    # Vision scores, image metrics
```

## Common Commands

### Run specific case
```bash
python benchmark/run_claude_code_eval.py \
    --config benchmark/configs/claude_code/config.json \
    --test_file benchmark/eval_cases/paraview/main_cases.yaml \
    --case_name bonsai
```

### Skip execution, only evaluate existing results
```bash
python benchmark/run_claude_code_eval.py \
    --config benchmark/configs/claude_code/config.json \
    --test_file benchmark/eval_cases/paraview/main_cases.yaml \
    --eval-only \
    --eval-model gpt-4o
```

### Clear previous results
```bash
python benchmark/run_claude_code_eval.py \
    --config benchmark/configs/claude_code/config.json \
    --test_file benchmark/eval_cases/paraview/main_cases.yaml \
    --clear-results
```

## Configuration

### Basic Config (`benchmark/configs/claude_code/config.json`)

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-5",
  "eval_mode": "generic",
  "agent_name": "claude_code",
  "experiment_number": "exp_default",
  "timeout_per_task": 600,
  "claude_code_path": "claude",
  "preserve_workdir": false,
  "auto_approve": true,
  "custom_system_prompt": "",
  "environment": {
    "type": "conda",
    "name": "paraview_mcp",
    "packages": ["paraview", "napari", "numpy", "scipy", "matplotlib"]
  },
  "price": {
    "input_per_1m_tokens": "$3.00",
    "output_per_1m_tokens": "$15.00"
  }
}
```

### Key Configuration Options

- **`model`**: Claude model to use (`claude-sonnet-4-5`, `claude-opus-4-6`, etc.)
- **`timeout_per_task`**: Maximum seconds per task (default: 600)
- **`auto_approve`**: Enable `--dangerously-skip-permissions` for non-interactive execution (required for benchmarking)
- **`custom_system_prompt`**: Optional global instructions prepended to all tasks
- **`experiment_number`**: Track different runs (results saved to `results/claude_code_*_{experiment_number}/`)
- **`preserve_workdir`**: Keep temporary files for debugging (default: false)

### Create Custom Config

For different models or experiments:

```bash
# Copy base config
cp benchmark/configs/claude_code/config.json \
   benchmark/configs/claude_code/config_opus.json

# Edit for Opus
cat > benchmark/configs/claude_code/config_opus.json <<EOF
{
  "model": "claude-opus-4-6",
  "experiment_number": "opus_exp1",
  "timeout_per_task": 1200,
  "price": {
    "input_per_1m_tokens": "$15.00",
    "output_per_1m_tokens": "$75.00"
  }
}
EOF

# Run with new config
python benchmark/run_claude_code_eval.py \
    --config benchmark/configs/claude_code/config_opus.json \
    --test_file benchmark/eval_cases/paraview/main_cases.yaml \
    --case_name engine
```

## Security & Sandboxing

### Current Setup

The benchmark uses `--dangerously-skip-permissions` because:
- Benchmarks need non-interactive execution
- Tasks require Python/Bash execution and file I/O
- No user can approve operations during automated runs

### Security Mitigations

**Already in place:**
1. Network access disabled in `~/.claude/settings.json`:
   ```json
   {
     "permissions": {
       "deny": ["WebFetch", "WebSearch"]
     }
   }
   ```
2. Isolated conda environment
3. No sensitive data in working directories
4. Scientific datasets only

**Additional recommendations:**
- Run in separate user account for complete isolation
- Use Docker containers for production deployments
- Monitor file system changes during runs

### Risk Level: LOW-MEDIUM

- ✅ Low risk: Network disabled, isolated environment, controlled datasets
- ⚠️ Medium risk: Full Bash access with auto-approve enabled

This is **acceptable for benchmarking** given the security measures in place.

## Recent Fixes

### Screenshot Generation (Feb 2026)
**Problem**: Vision evaluations failed because screenshots weren't generated.

**Fix**: Added explicit instructions to task prompts:
```
CRITICAL SCREENSHOT REQUIREMENT:
- After setting up the visualization, you MUST generate a screenshot
- Use SaveScreenshot() in ParaView to save a PNG image
- Screenshot filename should match the case name
- Save screenshots to the same results directory as state files
```

**Result**: Screenshots now automatically generated for all tasks.

### Custom System Prompt Support
**Feature**: Added `custom_system_prompt` config option to inject global instructions into all tasks.

**Usage**:
```json
{
  "custom_system_prompt": "You are an expert in scientific visualization. Always generate high-quality screenshots at 1920x1080 resolution."
}
```

## Troubleshooting

### Claude Code not found
```bash
# Check if installed
which claude

# If not found, set explicit path in config
{
  "claude_code_path": "/path/to/claude"
}
```

### Task times out
Increase timeout in config:
```json
{
  "timeout_per_task": 1200
}
```

### No output files generated
Enable debug mode to inspect working directory:
```json
{
  "preserve_workdir": true
}
```

Then check: `SciVisAgentBench-tasks/main/{case_name}/`

### Agent not found error
Make sure you're using the Claude Code runner:
```bash
python benchmark/run_claude_code_eval.py  # ✓ Correct
# NOT: python -m benchmark.evaluation_framework.run_evaluation
```

### Import errors in tests
Make sure you're in the correct directory:
```bash
cd /path/to/SciVisAgentBench
python benchmark/test_claude_code_agent.py
```

## Architecture Overview

### How It Works

```
User → run_claude_code_eval.py
         ↓
      ClaudeCodeAgent (registered via @register_agent)
         ↓
      _invoke_claude_code() - Runs `claude` CLI as subprocess
         ↓
      _find_output_files() - Locates generated files
         ↓
      Returns AgentResult(success, response, output_files, metadata)
         ↓
      EvaluationManager - Evaluates with LLM judge (if --eval-model)
         ↓
      Saves results to test_results/ and evaluation_results/
```

### Key Design Decisions

1. **Tool-Agnostic**: Claude Code receives only natural language task descriptions, no MCP servers or tool schemas
2. **Subprocess Invocation**: Runs Claude CLI directly (not via API) to test real-world usage
3. **Generic Eval Mode**: Uses `eval_mode="generic"` for framework compatibility
4. **Flexible File Detection**: Searches for common file patterns to handle slight naming variations

### Comparison with MCP Agents

| Aspect | MCP Agents | Claude Code |
|--------|-----------|-------------|
| **Tools** | MCP server with specialized functions | None - figures out APIs from scratch |
| **Setup** | Complex (MCP servers, stdio transport) | Simple (just CLI invocation) |
| **Token Usage** | Higher (tool schemas in context) | Lower (no tool overhead) |
| **Eval Mode** | `"mcp"` | `"generic"` |

## Next Steps

1. Run full benchmark on all test cases
2. Compare results with MCP agents
3. Analyze success rates, costs, and quality scores
4. Review detailed integration docs: `CLAUDE_CODE_INTEGRATION.md` (for architecture details)

## Support

- Claude Code: https://github.com/anthropics/claude-code
- SciVisAgentBench: Check repository documentation
