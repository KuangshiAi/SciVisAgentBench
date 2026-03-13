# GPT-5.2 backbone
python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_openai.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model claude-opus-4-6 \
    --experiment-number exp1

python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp_agent \
    --config benchmark/configs/gmx_vmd_mcp/config_openai.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --test-results test_results/molecular_vis/gmx_vmd_mcp_gpt-5.2_exp1 \
    --output eval_reports_claude-opus-4-6/molecular_vis/gmx_vmd_mcp_gpt-5.2_exp1 \
    --agent-mode gmx_vmd_mcp_gpt-5.2_exp1

python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_openai.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model claude-opus-4-6 \
    --experiment-number exp2

python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp_agent \
    --config benchmark/configs/gmx_vmd_mcp/config_openai.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --test-results test_results/molecular_vis/gmx_vmd_mcp_gpt-5.2_exp2 \
    --output eval_reports_claude-opus-4-6/molecular_vis/gmx_vmd_mcp_gpt-5.2_exp2 \
    --agent-mode gmx_vmd_mcp_gpt-5.2_exp2

python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_openai.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model claude-opus-4-6 \
    --experiment-number exp3

python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp_agent \
    --config benchmark/configs/gmx_vmd_mcp/config_openai.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --test-results test_results/molecular_vis/gmx_vmd_mcp_gpt-5.2_exp3 \
    --output eval_reports_claude-opus-4-6/molecular_vis/gmx_vmd_mcp_gpt-5.2_exp3 \
    --agent-mode gmx_vmd_mcp_gpt-5.2_exp3

# Sonnet as backbone
python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model claude-opus-4-6 \
    --experiment-number exp1

python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp_agent \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --test-results test_results/molecular_vis/gmx_vmd_mcp_claude-sonnet-4-5_exp1 \
    --output eval_reports_claude-opus-4-6/molecular_vis/gmx_vmd_mcp_claude-sonnet-4-5_exp1 \
    --agent-mode gmx_vmd_mcp_claude-sonnet-4-5_exp1

python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model claude-opus-4-6 \
    --experiment-number exp2

python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp_agent \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --test-results test_results/molecular_vis/gmx_vmd_mcp_claude-sonnet-4-5_exp2 \
    --output eval_reports_claude-opus-4-6/molecular_vis/gmx_vmd_mcp_claude-sonnet-4-5_exp2 \
    --agent-mode gmx_vmd_mcp_claude-sonnet-4-5_exp2

python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model claude-opus-4-6 \
    --experiment-number exp3

python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp_agent \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --test-results test_results/molecular_vis/gmx_vmd_mcp_claude-sonnet-4-5_exp3 \
    --output eval_reports_claude-opus-4-6/molecular_vis/gmx_vmd_mcp_claude-sonnet-4-5_exp3 \
    --agent-mode gmx_vmd_mcp_claude-sonnet-4-5_exp3