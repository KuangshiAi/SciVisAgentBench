# SciVisAgentBench — Codex CLI evaluation (Docker execution + host evaluation)
#
# Each suite has three steps, run in order:
#   (1) EXECUTE the agent INSIDE the container via ./docker/run_eval_in_docker.sh
#       — sandboxed away from your local filesystem and the ground truth. It
#       stages a GS-free copy, runs the agent, and collects outputs back.
#   (2) EVALUATE on the HOST (needs the ground truth GS/ + the metric stack).
#   (3) REPORT on the HOST.
#
# Prerequisites (one time):
#   ./docker/build.sh && ./docker/build_codex.sh    # base + Codex image
#   codex login                                     # host auth; ~/.codex is mounted in
# See docker/RUNNING_AGENTS.md for details.


######### paraview ###########
### paraview cases codex cli
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview --experiment-number exp1

# (2) evaluate on host
python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview --eval-model gpt-5.2 --eval-only --experiment-number exp1

# (3) report on host
python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview --test-results test_results/paraview/codex_cli_gpt-5.2_exp1/ --output eval_reports/paraview/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8082 --no-browser --static-only


######### anonymized #########
### paraview cases codex cli
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets --experiment-number exp1

# (2) evaluate on host
python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets --eval-model gpt-5.2 --eval-only --experiment-number exp1

# (3) report on host
python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets --test-results test_results/anonymized_datasets/codex_cli_gpt-5.2_exp1/ --output eval_reports/anonymized_datasets/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8082 --no-browser --static-only


#### Napari #####
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data --experiment-number exp1

# (2) evaluate on host
python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data --eval-model gpt-5.2 --eval-only --experiment-number exp1

# (3) report on host
python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data/eval_visualization_tasks --test-results test_results/bioimage_data/codex_cli_gpt-5.2_exp1 --output eval_reports/bioimage_data/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8082 --no-browser --static-only


# molecular_VIS (VMD)
# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --experiment-number exp1

# (2) evaluate on host
python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp1

# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --experiment-number exp1

# (2) evaluate on host
python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp1

# (3) report on host
python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml --cases SciVisAgentBench-tasks/molecular_vis --test-results test_results/molecular_vis/codex_cli_gpt-5.2_exp1/ --output eval_reports/molecular_vis/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8082 --no-browser --static-only


# # ######## topology #########
# (1) execute in Docker (sandboxed)
./docker/run_eval_in_docker.sh --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --experiment-number exp1

# (2) evaluate on host
python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --eval-model gpt-5.2 --eval-only --experiment-number exp1

# (3) report on host
python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --test-results test_results/topology/codex_cli_gpt-5.2_exp1/ --output eval_reports/topology/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8082 --no-browser --static-only
