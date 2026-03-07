# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml 
python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model gpt-5.2 \
    --experiment-number exp1

python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp_agent \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --test-results test_results/molecular_vis/gmx_vmd_mcp_claude-sonnet-4-5_exp1/eval_analysis_tasks \
    --output eval_reports/molecular_vis/tasks/gmx_vmd_mcp_claude-sonnet-4-5_exp1 \
    --agent-mode gmx_vmd_mcp_claude-sonnet-4-5_exp1 \
    --port 8081

# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml
python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model gpt-5.2 \
    --experiment-number exp1

python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp_agent \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --test-results test_results/molecular_vis/gmx_vmd_mcp_claude-sonnet-4-5_exp1/eval_analysis_workflows \
    --output eval_reports/molecular_vis/workflows/gmx_vmd_mcp_claude-sonnet-4-5_exp1 \
    --agent-mode gmx_vmd_mcp_claude-sonnet-4-5_exp1 \
    --port 8081

