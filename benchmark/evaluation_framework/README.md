# SciVisAgentBench Evaluation Framework

A plugin-based framework for evaluating scientific visualization agents with minimal code (~20 lines vs 500+).

## Features

- **Simple Interface**: Implement one method (`run_task`) to integrate your agent
- **Automatic Evaluation**: LLM judge, image metrics (PSNR, SSIM, LPIPS), efficiency analysis
- **Plugin System**: Register agents with `@register_agent` decorator
- **Full Compatibility**: Preserves all existing evaluator features
- **Dual Evaluation Modes**:
  - **Score-based**: LLM rubric evaluation (main, chatvis_bench)
  - **Assertion-based**: Binary pass/fail validation (bioimage_data, molecular_vis)
- **Multi-Subset Support**: main, bioimage_data, sci_volume_data, molecular_vis, chatvis_bench

## Quick Start

### 1. Create Agent

```python
from benchmark.evaluation_framework import BaseAgent, AgentResult, register_agent

@register_agent("my_agent")
class MyAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your agent

    async def run_task(self, task_description, task_config):
        # Your agent logic
        return AgentResult(
            success=True,
            response="Task completed",
            metadata={"duration": 10.5}
        )
```

### 2. Create Config

```json
{
  "provider": "openai",
  "model": "gpt-4o",
  "eval_mode": "mcp",
  "agent_name": "my_agent"
}
```

### 3. Run

```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent my_agent \
    --config config.json \
    --yaml SciVisAgentBench-tasks/main/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main
```

## Command-Line Options

```bash
# List agents
--list-agents

# Run specific case
--case case_name

# Skip evaluation
--no-eval

# Use static screenshots
--static-screenshot

# Change evaluation model
--eval-model gpt-4-turbo

# Custom output directory
--output /path/to/results
```

## Agent Interface

### Required Method

```python
async def run_task(self, task_description: str, task_config: Dict) -> AgentResult
```

**Parameters:**
- `task_description`: Natural language task description
- `task_config`: `{case_name, case_dir, data_dir, working_dir, eval_mode}`

**Returns:**
```python
AgentResult(
    success=True/False,
    response="agent response text",
    error="error message" if failed,
    output_files={"state": "/path/to/file.pvsm"},
    metadata={
        "duration": 12.5,
        "token_usage": {"input_tokens": 100, "output_tokens": 500, "total_tokens": 600},
        "custom_metric": "value"
    }
)
```

### Optional Methods

```python
async def setup(self)              # Called once before any tasks
async def teardown(self)           # Called once after all tasks
async def prepare_task(task_config)  # Called before each task
async def cleanup_task(task_config)  # Called after each task
```

## Pre-built Agents

```python
from benchmark.evaluation_framework.agents import (
    ParaViewMCPAgent,   # ParaView with MCP
    NapariMCPAgent,     # Napari with MCP
    ChatVisAgent,       # ChatVis pvpython
    GmxVmdMcpAgent      # GMX-VMD molecular visualization
)
```

## Benchmarks

### Main (ParaView Volume Visualization)
```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent paraview_mcp \
    --config benchmark/configs/paraview_mcp/config_openai.json \
    --yaml SciVisAgentBench-tasks/main/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main
```

### Bioimage Data (Napari)
```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_openai.json \
    --yaml benchmark/eval_cases/napari/0_actions/eval_basic_napari_functions.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data/0_actions \
    --data-dir SciVisAgentBench-tasks/bioimage_data/data
```

### Molecular Visualization (GMX-VMD)
```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/actions/basic_actions.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis/actions \
    --data-dir SciVisAgentBench-tasks/molecular_vis/data
```

## Evaluation Types

### Score-Based (Vision/Text Rubrics)

Used by: main, chatvis_bench, sci_volume_data

**Scoring:**
- Visualization quality: 10 points per goal
- Output generation: 5 points
- Efficiency: 10 points

**Example:**
```yaml
assert:
  - type: llm-rubric
    subtype: vision
    value: |
      1. Volume should be visible
      2. Colors should be appropriate
```

### Assertion-Based (Binary Pass/Fail)

Used by: bioimage_data, molecular_vis

**Assertion Types:**
- `contains-all`: Check if response contains value(s)
- `not-contains`: Check if response does NOT contain value(s)
- `llm-rubric`: LLM evaluates against criteria

**Binary Pattern:**
```yaml
assert:
  - type: contains-all
    value: "<1>"      # Success
  - type: not-contains
    value: "<0>"      # Failure
```

**Scoring:**
- Response contains `<1>` → Pass (score = 1)
- Response contains `<0>` → Fail (score = 0)
- Other response → Invalid (score = 0)

## Directory Structure

```
test_case/
├── data/                       # Input data
├── GS/                         # Ground truth
├── results/mcp/               # Agent outputs
├── test_results/mcp/          # Execution results (timing, tokens)
└── evaluation_results/mcp/    # Evaluation scores

test_results/{benchmark_name}/{agent_name}/  # Centralized results
```

## Evaluation Metrics

### Vision Evaluation
- **Image Metrics**: PSNR, SSIM, LPIPS (3 viewpoints)
- **LLM Judge**: Visual quality assessment
- **Screenshot Comparison**: Generated vs ground truth

### Text Evaluation
- **LLM Judge**: Evaluates text answers
- **Rubric-based**: Each criterion worth 10 points

### Efficiency
- **Execution Time**: < 15s = 5pts, < 30s = 4pts, etc.
- **Token Usage**: < 500 = 5pts, < 1000 = 4pts, etc.

## Programmatic Usage

```python
import asyncio
from benchmark.evaluation_framework import get_agent, UnifiedTestRunner

async def main():
    AgentClass = get_agent("my_agent")
    agent = AgentClass.from_config_file("config.json")

    runner = UnifiedTestRunner(
        agent=agent,
        yaml_path="cases.yaml",
        cases_dir="cases",
        eval_model="gpt-4o"
    )

    runner.load_yaml_test_cases()
    summary = await runner.run_all_test_cases()
    print(f"Success rate: {summary['success_rate']:.1%}")

asyncio.run(main())
```

## Migrating Existing Agents

Before (500+ lines):
```python
class MyRunner:
    def __init__(self, config, yaml_path, cases_dir):
        # Load YAML, setup directories, initialize agent...
    def run_test_case(self, case):
        # Run agent, save results, call evaluator...
    def run_all_cases(self):
        # For each case...
```

After (~20 lines):
```python
@register_agent("my_agent")
class MyAgent(BaseAgent):
    async def run_task(self, task_description, task_config):
        result = self.my_agent.process(task_description)
        return AgentResult(success=True, response=result.text)
```

## Troubleshooting

**Agent Not Found**
```python
@register_agent("my_agent")  # Add decorator
from my_module import MyAgent  # Import registers it
```

**Missing Output Files**
```python
dirs = self.get_result_directories(task_config["case_dir"], task_config["case_name"])
output_file = dirs["results_dir"] / f"{task_config['case_name']}.pvsm"
dirs["results_dir"].mkdir(parents=True, exist_ok=True)
```

**Evaluation Skipped**
```bash
export OPENAI_API_KEY="your-key"
# Or pass via CLI
--openai-api-key your-key
```

## Examples

See `examples/` directory:
- `simple_agent.py` - Minimal implementation
- `mcp_agent.py` - MCP-based agent
- `custom_evaluator.py` - Custom evaluation

## Architecture

```
User Agent → BaseAgent.run_task()
    ↓
UnifiedTestRunner → Manages test execution
    ↓
EvaluationManager → Routes to appropriate evaluator
    ↓
├─ Score-based → MCPAutoEvaluator/PVPythonAutoEvaluator
└─ Assertion-based → AssertionEvaluator
    ↓
Existing Evaluators (LLMEvaluator, ImageMetricsHelper, etc.)
```

## Additional Documentation

- **[ASSERTION_EVALUATION.md](ASSERTION_EVALUATION.md)** - Assertion-based evaluation details
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migrating existing agents
- **Config examples**: `benchmark/configs/{agent_name}/`

## License

© 2025 University of Notre Dame
