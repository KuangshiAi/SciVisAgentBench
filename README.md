# SciVisAgentBench

SciVisAgentBench is a comprehensive benchmark for evaluating scientific visualization agents. The benchmark supports evaluation of three autonomous agents, ParaView-MCP, bioimage-agent, and ChatVis, enabling users to create and manipulate scientific visualizations using natural language instead of complex commands or GUI operations. The benchmark uses YAML files compatible with [promptfoo](https://www.promptfoo.dev/) to store test cases and evaluation metrics. This initial version focuses on outcome-based evaluation, using both LLM-as-a-judge and quantitative metrics.

## 🚀 NEW: Easy Evaluation Framework

We now provide a **high-level evaluation framework** that makes it dramatically easier to evaluate new agents!

**Before:** Write 500+ lines of test runner code

**After:** Implement one method and run a single command

### Quick Example

```python
from benchmark.evaluation_framework import BaseAgent, AgentResult, register_agent

@register_agent("my_agent")
class MyAgent(BaseAgent):
    async def run_task(self, task_description, task_config):
        # Your agent logic here
        return AgentResult(success=True, response="Done", output_files={...})
```

Run evaluation:

```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent my_agent \
    --config my_config.json \
    --yaml benchmark/eval_cases/paraview/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main
    --eval-model gpt-5.2
```

**See [benchmark/evaluation_framework/README.md](benchmark/evaluation_framework/README.md) for details.**

## Download Benchmark Tasks

Download the benchmark tasks from our secondary huggingface dataset repo [KuangshiAi/SciVisAgentBench-tasks](https://huggingface.co/datasets/KuangshiAi/SciVisAgentBench-tasks) and place them in your workspace.

Make sure the `SciVisAgentBench-tasks` directory is placed at the same level as the `benchmark` directory, including the `main`, the `bioimage_data`, the `sci_volume_data`, the `molecular_vis`, and the `chatvis_bench` folders, as shown in the project structure:

```
SciVisAgentBench/
├── benchmark/
│   ├── yaml_runner_paraview_mcp.py
│   └── ...
└── SciVisAgentBench-tasks/
    ├── main/
    ├── sci_volume_data/
    ├── bioimage_data/
    ├── chatvis_bench/
    └── molecular_vis/
```

## ParaView-MCP, ChatVis, bioimage-agent, and GMX-VMD-MCP Installation
We provide out-of-box evaluation for ParaView-MCP, ChatVis, bioimage-agent, and GMX-VMD-MCP in SciVisAgentBench. We suggest installing them in seperated conda virtual environments (ParaView-MCP and ChatVis can share the same environment).

### To install ParaView, ParaView-MCP, and SciVisAgentBench requirements (it also works for ChatVis):
```shell
conda create -n paraview_mcp python=3.10
conda activate paraview_mcp
conda install conda-forge::paraview
pip install -r requirements.txt
```

### To install napari, bioimage-agent, and SciVisAgentBench requirements:
```shell
conda create -y -n bioimage_agent -c conda-forge python=3.11
conda activate bioimage_agent
python -m pip install "napari[all]"
pip install -r requirements.txt

cd src/napari_socket
pip install -e .
```

### To install GMX-VMD-MCP and SciVisAgentBench requirements:

Prerequisites:

- GROMACS (installed and accessible in PATH)
- VMD (Visual Molecular Dynamics, installed and accessible in PATH)
- (Optional) Python VMD module for enhanced visualization capabilities

```shell
conda create -n gmx_vmd_mcp python=3.10
conda activate gmx_vmd_mcp

# First install SciVisAgentBench requirements
pip install -r requirements.txt

cd src/gmx_vmd_mcp
# Then install GMX-VMD-MCP requirements
pip install -r requirements.txt
pip install -e .
```

Config GROMACS and VMD paths:

The MCP server uses a configuration file (`src/gmx_vmd_mcp/config.json`) for visualization engine path, search paths, and other settings. If this file doesn't exist, create one with the following structure (check `src/gmx_vmd_mcp/config.json.example`):

```json
{
  "vmd": {
    "vmd_path": "/path/to/vmd/executable",
    "search_paths": ["/path/to/search"]
  },
  "gmx": {
    "gmx_path": "/path/to/gromacs/executable"
  }
}
```

For macOS users, the VMD path is typically:

```bash
/Applications/VMD.app/Contents/MacOS/startup.command
```

## MCP Logger and Tiny Agent

- `mcp_logger.py` - Enhanced MCP communication logger with structured JSON logging and automatic screenshot capture
- `tiny_agent/agent.py` - Tiny Agent implementation based on MCPClient from HuggingFace Hub
- `tiny_agent/mcp_client.py` - MCPClient copied from HuggingFace Hub, but modified to provide support for OpenAI API

### Example usage of ParaView-MCP with logger
```json
"ParaView": {
    "command": "/path/to/paraview_mcp/conda/env/python.exe",
    "args": [
    ".../src/mcp_logger.py",
    "/path/to/paraview_mcp/conda/env/python.exe",
    ".../src/napari_mcp/napari_mcp_server.py"
    ]
}
```

## Configs for Agents
Set the config files for the agents at `benchmark/configs`. As an optional choice, use `src/mcp_logger.py` to record the communication logs (MCP function calls and arguments) between MCP agents and MCP servers, together with screenshots after each MCP function call.

## Run ParaView-MCP Evaluation

### 1. Start paraview server

In a new terminal:
```shell
conda activate paraview_mcp
python pvserver --multi-clients
```

### 2. Connect to paraview server from paraview GUI (file -> connect)

Make sure the ParaView GUI be of the same version as the ParaView server.

### 3. Setup agent configs

Check `benchmark/configs/paraview_mcp` and setup both MCP server and API provider info.

### 4. Run evaluation of ParaView-MCP on the `main` benchmark

```shell
conda activate paraview_mcp
python -m benchmark.evaluation_framework.run_evaluation \
    --agent paraview_mcp \
    --config benchmark/configs/paraview_mcp/config_openai.json \
    --yaml benchmark/eval_cases/paraview/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main \
    --eval-model gpt-5.2
```

## Run ChatVis Evaluation

### 1. Setup agent configs

Check `benchmark/configs/chatvis` and setup API provider info.

### 2. Run evaluation of ChatVis on the `chatvis-bench` benchmark

```shell
conda activate paraview_mcp
python -m benchmark.evaluation_framework.run_evaluation \
    --agent chatvis \
    --config benchmark/configs/chatvis/config_openai.json \
    --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml\
    --cases SciVisAgentBench-tasks/chatvis_bench \
    --eval-model gpt-5.2
```

## Run bioimage-agent Evaluation

### 1. Start paraview server

In a new terminal:
```shell
conda activate bioimage_agent
napari
```

### 2. In napari, choose **Plugins → Socket Server → Start Server**. You’ll see something like:

```text
Listening on 127.0.0.1:64908
```

### 3. Setup agent configs

Check `benchmark/configs/napari_mcp` and setup both MCP server and API provider info.

### 4. Run evaluation of bioimage-agent on the `bioimage_data` benchmark with level-0 action tasks

```shell
conda activate bioimage_agent
python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_openai.json \
    --yaml benchmark/eval_cases/napari/0_actions/eval_basic_napari_functions.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data/data \
    --eval-model gpt-5.2
```

You may also run evaluation with level-1 workflow tasks (`benchmark/eval_cases/napari/1_workflows`).

## Run GMX-VMD-MCP Evaluation

### 1. Environment setup

Make sure both GROMACS and VMD (Visual Molecular Dynamics) are installed and accessible in PATH, and explicitly create and set `src/gmx_vmd_mcp/config.json` file.

### 2. Setup agent configs

Check `benchmark/configs/gmx_vmd_mcp` and setup both MCP server and API provider info.

### 3. Run evaluation of GMX-VMD-MCP on the `molecular_vis` benchmark

```shell
conda activate gmx_vmd_mcp
python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/actions/basic_actions.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis/data \
    --eval-model gpt-5.2
```

## Anonymize Datasets

We provide a tool to help anonymize volume datasets so that we can test whether the LLM agent (for ParaView) is able to load the dataset and tell what the object is. Store the volume datasets at `SciVisAgentBench-tasks/sci_volume_data`, then anonymize both the yaml file and the datasets with:

```shell
conda activate paraview_mcp
cd benchmark
python anonymize_dataset.py eval_cases/paraview/what_obj_cases.yaml

# Then you can run the "what is the object" test
cd ..
python -m benchmark.evaluation_framework.run_evaluation \
    --agent paraview_mcp \
    --config benchmark/configs/paraview_mcp/config_openai.json \
    --yamlbenchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml \
    --cases SciVisAgentBench-tasks/anonymized_datasets \
    --eval-model gpt-5.2
```

## Evaluate Your Own Agents

We provide a **high-level evaluation framework** that makes it dramatically easier to evaluate your own agents. See [benchmark/evaluation_framework/README.md](benchmark/evaluation_framework/README.md) for details.

## Quantitative Metrics

The benchmark includes comprehensive image quality evaluation using multiple metrics to assess the visual fidelity of generated visualizations:

### Supported Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level differences between ground truth and generated images. Higher values indicate better quality.
- **SSIM (Structural Similarity Index)**: Evaluates structural similarity considering luminance, contrast, and structure. Values range from -1 to 1, with higher values being better.
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Uses deep learning to measure perceptual similarity, better correlating with human perception. Lower values indicate better similarity.

### Multi-viewpoint Analysis

Image metrics are calculated across three different viewpoints (diagonal, front, and side views) and then averaged for each test case to provide comprehensive visual assessment.

### Scaled Metrics

To account for varying completion rates across different evaluation modes, the benchmark provides scaled metrics that weight the results by the fraction of successfully completed tasks:

- **PSNR_scaled** = (# of passed cases)/(# of all cases) × PSNR
- **SSIM_scaled** = (# of passed cases)/(# of all cases) × SSIM  
- **LPIPS_scaled** = 1.0 - (# of passed cases)/(# of all cases) × (1.0 - LPIPS)

This scaling ensures fair comparison between different approaches and prevents incomplete evaluations from artificially inflating scores.

## Promptfoo Compatibility

SciVisAgentBench is compatible with [promptfoo](https://www.promptfoo.dev/) evaluation frameworks by:

- **Standardized YAML Format**: Test cases are defined using YAML configuration files with `vars` (input variables) and `assert` (evaluation criteria) sections, following promptfoo conventions
- **Multi-rubric Support**: Supports both text-based and vision-based evaluation rubrics within the same test case:
  - `subtype: text` - Evaluates text-based answers using LLM judges
  - `subtype: vision` - Evaluates visual outputs using image comparison and LLM visual assessment
- **Flexible Evaluation**: Each test case can have multiple evaluation criteria with different scoring methods
- **Structured Output**: Results are saved in JSON format compatible with analysis tools and CI/CD pipelines

Example YAML test case structure:

```yaml
- vars:
    question: "Load dataset and create visualization..."
  assert:
    - type: llm-rubric
      subtype: vision
      value: "1. Shows correct data structure\n2. Uses appropriate colors"
    - type: llm-rubric
      subtype: text
      value: "1. Identifies key features correctly"
```

This compatibility allows researchers to:

- Leverage existing promptfoo tooling and workflows
- Define test cases in a standardized, version-controllable format
- Integrate SciVisAgentBench into automated evaluation pipelines
- Compare results across different visualization agent implementations

## Additional: Setup MCP Servers for External MCP Clients

To set up integration with claude desktop, add the following to claude_desktop_config.json

```json
    "mcpServers": {
      "paraview": {
        "command": "/path/to/paraview_mcp/conda/env/python",
        "args": [
        ".../src/paraview_mcp/paraview_mcp_server.py"
        ]
      },
      "napari": {
        "command": "/path/to/bioimage_agent/conda/env/python",
        "args": [                        
          ".../src/napari_mcp/napari_mcp_server.py"
          ]
      },
      "gmx_vmd": {
      "command": "/path/to/gmx_vmd_mcp/conda/env/python",
      "args": [
        ".../src/gmx-vmd-mcp/mcp_server.py"
        ],
      "env": {
        "PYTHONPATH": ".../src/gmx-vmd-mcp",
        "MCP_DEBUG": "1",
        "PYTHONUNBUFFERED": "1"
      }
      }
    }
```

## Acknowledgement

SciVisAgentBench was mainly created by Kuangshi Ai (kai@nd.edu), Shusen Liu (liu42@llnl.gov), and Haichao Miao (miao1@llnl.gov). Some of the test cases are provided by Kaiyuan Tang (ktang2@nd.edu). We sincerely thank the open-source community for their invaluable contributions. This project is made possible thanks to the following outstanding projects:

- [ParaView-MCP](https://github.com/LLNL/paraview_mcp)
- [Bioimage-agent](https://github.com/LLNL/bioimage-agent)
- [ChatVis](https://github.com/tpeterka/ChatVis)
- [GMX-VMD-MCP](https://github.com/egtai/gmx-vmd-mcp)

## License

© 2025 University of Notre Dame.  
Released under the [License](./LICENSE).