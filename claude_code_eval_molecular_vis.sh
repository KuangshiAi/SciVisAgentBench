# molecular_VIS (VMD) exp1
# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1

python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp1 --openai-base-url https://livai-api.llnl.gov

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --test-results test_results/molecular_vis/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/molecular_vis/tasks/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081 --no-browser --static-only

# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1

python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp1 --openai-base-url https://livai-api.llnl.gov

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --test-results test_results/molecular_vis/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/molecular_vis/workflows/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081 --no-browser --static-only


# molecular_VIS (VMD) exp2
# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks-exp2/molecular_vis --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp2

python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks-exp2/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp2

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks-exp2/molecular_vis --test-results test_results/molecular_vis/claude_code_claude-sonnet-4-5_exp2/ --output eval_reports/molecular_vis/tasks/claude_code_claude-sonnet-4-5_exp2 --agent-mode claude_code_claude-sonnet-4-5_exp2 --port 8081 --no-browser --static-only

# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks-exp2/molecular_vis --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp2

python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks-exp2/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp2 --openai-base-url https://livai-api.llnl.gov

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks-exp2/molecular_vis --test-results test_results/molecular_vis/claude_code_claude-sonnet-4-5_exp2/ --output eval_reports/molecular_vis/workflows/claude_code_claude-sonnet-4-5_exp2 --agent-mode claude_code_claude-sonnet-4-5_exp2 --port 8081 --no-browser --static-only


# molecular_VIS (VMD) exp3
# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks-exp3/molecular_vis --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp3

python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks-exp3/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp3 --openai-base-url https://livai-api.llnl.gov

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks-exp3/molecular_vis --test-results test_results/molecular_vis/claude_code_claude-sonnet-4-5_exp3/ --output eval_reports/molecular_vis/tasks/claude_code_claude-sonnet-4-5_exp3 --agent-mode claude_code_claude-sonnet-4-5_exp3 --port 8081 --no-browser --static-only

# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks-exp3/molecular_vis --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp3

python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks-exp3/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp3 --openai-base-url https://livai-api.llnl.gov

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks-exp3/molecular_vis --test-results test_results/molecular_vis/claude_code_claude-sonnet-4-5_exp3/ --output eval_reports/molecular_vis/workflows/claude_code_claude-sonnet-4-5_exp3 --agent-mode claude_code_claude-sonnet-4-5_exp3 --port 8081 --no-browser --static-only











