# SciVisAgentBench

SciVisAgentBench is a comprehensive benchmark for evaluating scientific visualization agents. The benchmark supports evaluation of three autonomous agents, ParaView-MCP, bioimage-agent, and ChatVis, enabling users to create and manipulate scientific visualizations using natural language instead of complex commands or GUI operations. The benchmark uses YAML files compatible with [promptfoo](https://www.promptfoo.dev/) to store test cases and evaluation metrics. This initial version focuses on outcome-based evaluation, using both LLM-as-a-judge and quantitative metrics.

## ParaView-MCP and bioimage-agent Installation
We suggest installing ParaView-MCP and bioimage-agent in two seperated conda virtual environments.

To install ParaView, ParaView-MCP and other requirements:
```shell
conda create -n paraview_mcp python=3.10
conda activate paraview_mcp
conda install conda-forge::paraview
pip install -r requirements.txt
```

To install napari, bioimage-agent and other requirements:
```shell
conda create -y -n bioimage_agent -c conda-forge python=3.11
conda activate bioimage_agent
python -m pip install "napari[all]"
pip install -r requirements.txt

cd src/napari_socket
pip install -e .
```

## Setup for External MCP Clients

To set up integration with claude desktop, add the following to claude_desktop_config.json

```json
    "mcpServers": {
      "ParaView": {
        "command": "/path/to/paraview_mcp/conda/env/python",
        "args": [
        ".../src/paraview_mcp/paraview_mcp_server.py"
        ]
      },
      "Napari": {
        "command": "/path/to/bioimage_agent/conda/env/python",
        "args": [                        
          ".../src/napari_mcp/napari_mcp_server.py"
        ]
    }
    }
```

## Download Benchmark Tasks

Download the benchmark tasks from our secondary huggingface dataset repo [KuangshiAi/SciVisAgentBench-tasks](https://huggingface.co/datasets/KuangshiAi/SciVisAgentBench-tasks) and place them in your workspace.

Make sure the `SciVisAgentBench-tasks` directory is placed at the same level as the `benchmark` directory, including the `main`, the `bioimage_data`, the `sci_volume_data`, and the `chatvis_bench` folders, as shown in the project structure:

```
SciVisAgentBench/
├── benchmark/
│   ├── yaml_runner_paraview_mcp.py
│   └── ...
└── SciVisAgentBench-tasks/
    ├── main/
    ├── sci_volume_data/
    ├── bioimage_data/
    └── download_and_organize.py
```

Follow the instructions and make sure you download the datasets locally:
```shell
cd SciVisAgentBench-tasks
python download_and_organize.py
```

If evaluating on the `chatvis_bench`, please turn on the ```--static_screenshot``` mode for maximum compatability.

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

## Configs for MCP Agents
Set the config files for MCP agents at `benchmark/configs/napari_mcp` and `benchmark/configs/paraview_mcp`. As an optional choice, use `src/mcp_logger.py` to record the communication logs (MCP function calls and arguments) between MCP agents and MCP servers, together with screenshots after each MCP function call.

## Run ParaView-MCP Evaluation

### 1. Start paraview server

In a new terminal:
```shell
conda activate paraview_mcp
python pvserver --multi-clients
```

### 2. Connect to paraview server from paraview GUI (file -> connect)

Make sure the ParaView GUI be of the same version as the ParaView server.

### 3. Setup configs and the bash script

Check `benchmark/configs/paraview_mcp` and `benchmark/eval_scripts/run_paraview_mcp_main.sh`.

### 4. Run evaluation with tiny_agent

```shell
conda activate paraview_mcp
cd benchmark/eval_scripts
bash run_paraview_mcp_main.sh
```

## Run ChatVis Evaluation for ParaView

### 1. Setup the bash script

Check `benchmark/eval_scripts/run_chatvis_main.sh`.

### 2. Run evaluation with ChatVis

```shell
conda activate paraview_mcp
cd benchmark/eval_scripts
bash run_chatvis_main.sh
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

### 3. Setup configs and the bash script

Check `benchmark/configs/napari_mcp` and `benchmark/eval_scripts/run_bioimage_agent.sh`.
The bash script runs evaluation on both level-0 action tasks and level-1 workflow tasks.

### 4. Run evaluation with tiny_agent

```shell
conda activate bioimage_agent
cd benchmark/eval_scripts
bash run_bioimage_agent.sh
```

## Anonymize Datasets

We provide a tool to help anonymize volume datasets so that we can test whether the LLM agent (for ParaView) is able to load the dataset and tell what the object is. Store the volume datasets at `SciVisAgentBench-tasks/sci_volume_data`, then anonymize both the yaml file and the datasets with:

```shell
conda activate paraview_mcp
cd benchmark
python anonymize_dataset.py eval_cases/paraview/what_obj_cases.yaml

# Then you can run the "what is the object" test
cd eval_scripts
bash run_paraview_mcp_what_obj.sh
bash run_chatvis_what_obj.sh
```

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

## Acknowledgement

SciVisAgentBench was mainly created by Kuangshi Ai (kai@nd.edu), Shusen Liu (liu42@llnl.gov), and Haichao Miao (miao1@llnl.gov). Some of the test cases are provided by Kaiyuan Tang (ktang2@nd.edu). We sincerely thank the open-source community for their invaluable contributions. This project is made possible thanks to the following outstanding projects:

- [ParaView-MCP](https://github.com/LLNL/paraview_mcp)
- [Bioimage-agent](https://github.com/LLNL/bioimage-agent)
- [ChatVis](https://github.com/tpeterka/ChatVis)

## License

© 2025 University of Notre Dame.  
Released under the [License](./LICENSE).