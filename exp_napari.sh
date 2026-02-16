conda activate napari_mcp
python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_openai.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_analysis_workflows.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data \
    --eval-model gpt-5.2 \
    --eval-only

python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_openai.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_figure_recreation.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data \
    --eval-model gpt-5.2

python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_openai.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_iso_surface_determination.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data \
    --eval-model gpt-5.2 \
    --clear-result

python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_openai.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_visualization_workflows.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data \
    --eval-model gpt-5.2 \
    --eval-only