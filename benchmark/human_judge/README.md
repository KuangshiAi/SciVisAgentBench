# Human Evaluation UI for SciVisAgentBench

A web-based interface for human judges to evaluate vision-based test cases from SciVisAgentBench. This tool allows evaluators to rate agent-generated visualizations against ground truth images using standardized metrics.

## Supported Benchmarks

- `main` - Main visualization benchmark (ParaView tasks)
- `chatvis_bench` - ChatVis benchmark

## Requirements

```bash
pip install flask pyyaml
```

The screenshot generation feature also requires ParaView to be installed and accessible.

## Quick Start

### 1. Evaluate ChatVis on chatvis_bench

```bash
conda activate paraview_mcp
python -m benchmark.human_judge.run_human_eval \
    --agent chatvis \
    --config benchmark/configs/chatvis/config_openai.json \
    --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \
    --cases SciVisAgentBench-tasks/chatvis_bench \
    --port 8081
```

### 2. Evaluate ParaView-MCP on main benchmark

```bash
conda activate paraview_mcp
python -m benchmark.human_judge.run_human_eval \
    --agent paraview_mcp \
    --config benchmark/configs/paraview_mcp/config_openai.json \
    --yaml benchmark/eval_cases/paraview/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main \
    --port 8081
```

### 3. Open browser

Once the server starts, open your browser and navigate to:

```
http://127.0.0.1:8081
```

## Command-Line Options

```
--agent AGENT              Name of the agent being evaluated
                          (e.g., chatvis, paraview_mcp, napari_mcp)

--config CONFIG           Path to agent configuration file (JSON)

--yaml YAML              Path to YAML file containing test cases

--cases CASES            Path to directory containing test case data
                          (e.g., SciVisAgentBench-tasks/main)

--output-dir DIR         Directory to save evaluation results
                          (default: benchmark/human_judge/evaluations)

--host HOST              Host to run the server on
                          (default: 127.0.0.1)

--port PORT              Port to run the server on
                          (default: 5000)

--debug                  Run server in debug mode
```

## Evaluation Output

Evaluations are saved to JSON files with the following structure:

```json
{
  "metadata": {
    "agent_name": "chatvis",
    "agent_mode": "mcp",
    "benchmark_type": "chatvis_bench",
    "yaml_file": "benchmark/eval_cases/paraview/chatvis_bench_cases.yaml",
    "cases_dir": "SciVisAgentBench-tasks/chatvis_bench",
    "total_cases": 10
  },
  "timestamp": "20231215_143022",
  "evaluations": [
    {
      "case_index": 0,
      "case_name": "ml-iso",
      "ratings": [8, 7, 9],
      "notes": "Good overall visualization, minor color discrepancy",
      "timestamp": "2023-12-15T14:32:45.123456"
    }
  ]
}
```