# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml 
python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_openai.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model gpt-5.2 \
    --experiment-number exp3

# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml
python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_openai.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model gpt-5.2 \
    --experiment-number exp3 \
    --exe-only

python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_openai.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model gpt-5.2 \
    --experiment-number exp3 \
    --eval-only

# Generate report for all cases (which includes both eval_analysis_tasks.yaml and eval_analysis_workflows.yaml)
python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp_agent \
    --config benchmark/configs/gmx_vmd_mcp/config_openai.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --test-results test_results/molecular_vis/gmx_vmd_mcp_gpt-5.2_exp3 \
    --output eval_reports/molecular_vis/gmx_vmd_mcp_gpt-5.2_exp3 \
    --agent-mode gmx_vmd_mcp_gpt-5.2_exp3 \
    --port 8081