# Codex CLI Agent Configuration

This directory contains configuration for the Codex CLI agent, which integrates OpenAI's Codex CLI as a general-purpose coding agent for scientific visualization tasks.

## Prerequisites

1. **Install Codex CLI**:
   ```bash
   # Install via npm (if not already installed)
   npm install -g codex-cli

   # Verify installation
   codex --version
   ```

2. **Authenticate**:
   ```bash
   # Interactive login
   codex login

   # Or non-interactive with API key
   printenv OPENAI_API_KEY | codex login --with-api-key

   # Check status
   codex login status
   ```

3. **Set up Python environment**:
   ```bash
   conda env create -f environment.yml
   conda activate scivis_bench
   ```

## Configuration Options

### `config.json`

```json
{
  "provider": "openai",              // Provider type
  "model": "gpt-4o-commercial",      // OpenAI model to use
  "eval_mode": "generic",            // Evaluation mode
  "agent_name": "codex_cli",         // Agent identifier
  "experiment_number": "exp1",       // Experiment tag
  "timeout_per_task": 600,           // Timeout in seconds (10 min)
  "codex_cli_path": "codex",         // Path to codex executable
  "preserve_workdir": false,         // Keep working directories for debugging
  "auto_approve": true,              // Auto-approve for benchmarking
  "verbose": false,                  // Enable real-time output streaming
  "custom_system_prompt": "",        // Optional additional instructions
  "price": {
    "input_per_1m_tokens": "$5.00",  // Cost per 1M input tokens
    "output_per_1m_tokens": "$15.00" // Cost per 1M output tokens
  }
}
```

### Key Parameters

- **model**: Specify which OpenAI model to use. Options include:
  - `gpt-4o-commercial`: GPT-4o for commercial use
  - `gpt-5.1-codex`: GPT-5.1 Codex variant
  - `o1`: O1 reasoning model

- **timeout_per_task**: Maximum time (in seconds) for each task execution. Default is 600 seconds (10 minutes).

- **auto_approve**: When `true`, uses `--dangerously-bypass-approvals-and-sandbox` flag to skip confirmations. **Only use in controlled environments.**

- **verbose**: When `true`, enables real-time streaming of Codex CLI output with JSON event parsing. Creates detailed log files.

- **custom_system_prompt**: Add custom instructions that will be prepended to all task prompts.

## Usage

### Run Single Test Case

```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent codex_cli \
    --config benchmark/configs/codex_cli/config.json \
    --yaml benchmark/eval_cases/paraview/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main \
    --case bonsai \
    --eval-model gpt-4o
```

### Run Full Benchmark

```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent codex_cli \
    --config benchmark/configs/codex_cli/config.json \
    --yaml benchmark/eval_cases/paraview/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main \
    --eval-model gpt-4o
```

### Enable Verbose Mode

Edit `config.json` and set `"verbose": true`, or create a verbose config:

```bash
cp config.json config_verbose.json
# Edit config_verbose.json to set "verbose": true

python -m benchmark.evaluation_framework.run_evaluation \
    --agent codex_cli \
    --config benchmark/configs/codex_cli/config_verbose.json \
    --yaml benchmark/eval_cases/paraview/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main
```

## How It Works

The Codex CLI agent:

1. **Receives** natural language task descriptions from the benchmark YAML files
2. **Prepares** task prompts with environment context and instructions
3. **Invokes** `codex exec` command with:
   - `--json`: Structured JSON output for parsing
   - `--ephemeral`: No persistent session files
   - `--dangerously-bypass-approvals-and-sandbox`: Auto-approve (when enabled)
   - `-C <working_dir>`: Set working directory
4. **Parses** JSON event stream to extract:
   - Agent messages and code
   - Token usage (input/output)
   - Errors and completion status
5. **Locates** generated output files (state files, screenshots, scripts)
6. **Returns** AgentResult with success status, token usage, and file paths

## Codex CLI Commands Used

The agent uses the following Codex CLI features:

- **`codex exec`**: Run agent non-interactively
- **`--json`**: Emit JSONL events to stdout
- **`--ephemeral`**: Don't persist session files
- **`--dangerously-bypass-approvals-and-sandbox`**: Skip all confirmations (auto-approve mode)
- **`-C <dir>`**: Change working directory
- **`-m <model>`**: Specify model (optional)

## Output Structure

Results are saved to:
```
case_name/
├── results/codex_cli/              # Agent outputs
│   ├── case_name.png               # Screenshot
│   ├── case_name_script.py         # Generated Python script
│   ├── case_name.pvsm              # State file (if applicable)
│   └── codex_cli_verbose_*.log     # Verbose log (if enabled)
├── test_results/codex_cli/         # Execution results
│   └── case_name_result.json       # Timing, tokens, success
└── evaluation_results/codex_cli/   # Evaluation scores
    └── case_name_eval.json         # LLM judge scores, metrics
```

## Troubleshooting

### Authentication Issues

If you see authentication errors:
```bash
# Re-authenticate
codex logout
printenv OPENAI_API_KEY | codex login --with-api-key
codex login status
```

### Model Access Issues

If you see "model not allowed" errors, check your API key has access to the specified model. Edit `config.json` to use a different model:
```json
{
  "model": "gpt-4o"  // Try a different model
}
```

### Timeout Issues

If tasks timeout frequently, increase the timeout:
```json
{
  "timeout_per_task": 1200  // 20 minutes
}
```

### Debugging

Enable verbose mode to see detailed logs:
```json
{
  "verbose": true
}
```

Log files will be saved to the working directory with names like `codex_cli_verbose_*.log`.

## Comparison with Claude Code Agent

| Feature | Claude Code | Codex CLI |
|---------|-------------|-----------|
| CLI Command | `claude --print` | `codex exec` |
| Auto-approve flag | `--dangerously-skip-permissions` | `--dangerously-bypass-approvals-and-sandbox` |
| JSON output | `--output-format stream-json` | `--json` |
| Working directory | `cwd=` parameter | `-C, --cd <DIR>` |
| Ephemeral mode | N/A | `--ephemeral` |
| Token reporting | Estimated from output | Parsed from JSON events |

## Security Considerations

The `auto_approve: true` setting uses `--dangerously-bypass-approvals-and-sandbox` which:
- ⚠️ **Disables all safety checks and confirmations**
- ⚠️ **Runs code without sandboxing**
- ⚠️ **Should only be used in controlled, isolated environments**

For production use:
1. Set `auto_approve: false` in config
2. Run in containerized/VM environment
3. Restrict network access via firewall
4. Monitor file system changes
5. Use dedicated conda environment

## License

© 2025 University of Notre Dame
