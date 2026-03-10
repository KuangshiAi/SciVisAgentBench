python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_openai.json \
    --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data \
    --eval-model gpt-5.2 \
    --experiment-number exp3

python -m benchmark.evaluation_reporter.run_reporter \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_openai.json \
    --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data/eval_visualization_tasks \
    --test-results test_results/bioimage_data/napari_mcp_gpt-5.2_exp3 \
    --output eval_reports/bioimage/napari_mcp_gpt-5.2_exp3 \
    --agent-mode napari_mcp_gpt-5.2_exp3 \
    --port 8081