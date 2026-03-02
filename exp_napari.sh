conda activate napari_mcp
# python -m benchmark.evaluation_framework.run_evaluation \
#     --agent napari_mcp \
#     --config benchmark/configs/napari_mcp/config_anthropic.json \
#     --yaml benchmark/eval_cases/napari/1_workflows/eval_analysis_workflows.yaml \
#     --cases SciVisAgentBench-tasks/bioimage_data \
#     --eval-model gpt-5.2

python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_figure_recreation.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data \
    --eval-model gpt-5.2

python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_iso_surface_determination.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data \
    --eval-model gpt-5.2

python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_visualization_workflows.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data \
    --eval-model gpt-5.2

python -m benchmark.evaluation_reporter.run_reporter \
    --agent bioimage_agent \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_visualization_workflows.yaml\
    --cases SciVisAgentBench-tasks/bioimage_data/eval_visualization_workflows \
    --test-results test_results/bioimage_data/napari_mcp/eval_visualization_workflows \
    --output eval_reports/bioimage-napari_mcp/eval_visualization_workflows \
    --port 8081


python -m benchmark.evaluation_reporter.run_reporter \
    --agent bioimage_agent \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_figure_recreation.yaml\
    --cases SciVisAgentBench-tasks/bioimage_data/eval_figure_recreation \
    --test-results test_results/bioimage_data/napari_mcp/eval_figure_recreation \
    --output eval_reports/bioimage-napari_mcp/eval_figure_recreation \
    --port 8081


python -m benchmark.evaluation_reporter.run_reporter \
    --agent bioimage_agent \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_iso_surface_determination.yaml\
    --cases SciVisAgentBench-tasks/bioimage_data/eval_iso_surface_determination \
    --test-results test_results/bioimage_data/napari_mcp/eval_iso_surface_determination \
    --output eval_reports/bioimage-napari_mcp/eval_iso_surface_determination \
    --port 8081

python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp_agent \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/workflows/eval_analysis_workflows.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis/workflows \
    --test-results test_results/molecular_vis/gmx_vmd_mcp/eval_analysis_workflows \
    --output /Users/kuangshiai/Documents/ND-VIS/Code/SciVisAgentBench/eval_reports/molecular_vis-gmx_vmd_mcp \
    --agent-mode gmx_vmd_mcp_claude-sonnet-4-5_exp1 \
    --port 8081