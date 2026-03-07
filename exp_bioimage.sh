python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data \
    --eval-model gpt-5.2 \
    --experiment-number exp1 \
    --start-from case_11 \
    --eval-only

python -m benchmark.evaluation_reporter.run_reporter \
    --agent napari_mcp \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml \
    --cases SciVisAgentBench-tasks/bioimage_data/eval_visualization_tasks \
    --test-results test_results/bioimage_data/napari_mcp_claude-sonnet-4-5_exp1/eval_visualization_tasks \
    --output eval_reports/bioimage/napari_mcp_claude-sonnet-4-5_exp1 \
    --agent-mode napari_mcp_claude-sonnet-4-5_exp1 \
    --port 8081