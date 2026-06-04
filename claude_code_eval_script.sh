# SciVisAgentBench — Claude Code evaluation (Docker execution + host evaluation)
#
# Each suite has three steps, run in order:
#   (1) EXECUTE the agent INSIDE the container via ./docker/run_eval_in_docker.sh
#       — sandboxed away from your local filesystem and the ground truth. It
#       stages a GS-free copy, runs the agent, and collects outputs back.
#   (2) EVALUATE on the HOST (needs the ground truth GS/ + the metric stack).
#   (3) REPORT on the HOST.
#
# Prerequisites (one time):
#   ./docker/build.sh && ./docker/build_claude.sh   # base + Claude Code image
#   export ANTHROPIC_API_KEY=sk-ant-...             # forwarded into the container
# See docker/RUNNING_AGENTS.md for details.


######### paraview ###########
### paraview cases claude code
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview --experiment-number exp1

# (2) evaluate on host
python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview --eval-model gpt-5.2 --eval-only --experiment-number exp1

# (3) report on host
python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview --test-results test_results/paraview/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/paraview/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081 --no-browser --static-only


######### anonymized #########
### paraview cases claude code
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets --experiment-number exp1

# (2) evaluate on host
python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets/ --eval-model gpt-5.2 --eval-only --experiment-number exp1

# (3) report on host
python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets/ --test-results test_results/anonymized_datasets/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/sci_volume_data/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081 --no-browser --static-only


#### Napari #####
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data --experiment-number exp1

# (2) evaluate on host
python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data/ --eval-model gpt-5.2 --eval-only --experiment-number exp1

# (3) report on host
python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data/eval_visualization_tasks/ --test-results test_results/bioimage_data/claude_code_claude-sonnet-4-5_exp1 --output eval_reports/napari/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081 --no-browser --static-only


# molecular_VIS (VMD)
# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --experiment-number exp1

# (2) evaluate on host
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp1

# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --experiment-number exp1

# (2) evaluate on host
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp1

# (3) report on host
python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --test-results test_results/molecular_vis/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/molecular_vis/workflows/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081 --no-browser --static-only


# ######## topology #########
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --experiment-number exp1

# (2) evaluate on host
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --eval-model gpt-5.2 --eval-only --experiment-number exp1

# (3) report on host
python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --test-results test_results/topology/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/topology/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081 --no-browser --static-only
