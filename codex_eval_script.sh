export PV_FORCE_OFFSCREEN_RENDERING=1
export VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1

######### paraview ###########
## single eval test
# python benchmark/run_codex_cli_eval.py --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview/ --eval-model gpt-5.2 --case bonsai --verbose --exe-only

### paraview cases codex cli
# python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview --eval-model gpt-5.2 --verbose --experiment-number exp1 --verbose --exe-only

# python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview --eval-model gpt-5.2 --eval-only --experiment-number exp1 --openai-base-url https://livai-api.llnl.gov

# python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview --test-results test_results/paraview/codex_cli_gpt-5.2_exp1/ --output eval_reports/paraview/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8082 --no-browser --static-only


######### anonymized #########
### paraview cases codex cli
# python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1

# python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets --eval-model gpt-5.2 --eval-only --experiment-number exp1 --openai-base-url https://livai-api.llnl.gov

# python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets --test-results test_results/anonymized_datasets/codex_cli_gpt-5.2_exp1/ --output eval_reports/anonymized_datasets/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8082 --no-browser --static-only


#### Napari #####
# ### test
# python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/napari/eval_visualization_workflows.yaml --cases SciVisAgentBench-tasks/bioimage_data --eval-model gpt-5.2 --case case_1 --verbose

# python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1 --exe-only

# python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data --eval-model gpt-5.2 --eval-only --experiment-number exp1 --openai-base-url https://livai-api.llnl.gov

# python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data/eval_visualization_tasks --test-results test_results/bioimage_data/codex_cli_gpt-5.2_exp1 --output eval_reports/bioimage_data/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8082 --no-browser --static-only

# python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks-exp2/bioimage_data/eval_visualization_tasks --test-results test_results/bioimage_data/codex_cli_gpt-5.2_exp1/ --output eval_reports/bioimage_data/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8081 --no-browser --static-only

# molecular_VIS (VMD)
# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml
python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1 --start-from case_8

python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp1 --openai-base-url https://livai-api.llnl.gov

# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml
python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1

python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp1 --openai-base-url https://livai-api.llnl.gov

python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml --cases SciVisAgentBench-tasks/molecular_vis --test-results test_results/molecular_vis/codex_cli_gpt-5.2_exp1/ --output eval_reports/molecular_vis/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8082 --no-browser --static-only


# # ######## topology #########
# python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1

# python -m benchmark.run_codex_cli_eval --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --eval-model gpt-5.2 --eval-only --experiment-number exp1 --openai-base-url https://livai-api.llnl.gov

# python -m benchmark.evaluation_reporter.run_reporter --agent codex_cli --config benchmark/configs/codex_cli/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --test-results test_results/topology/codex_cli_gpt-5.2_exp1/ --output eval_reports/topology/codex_cli_gpt-5.2_exp1 --agent-mode codex_cli_gpt-5.2_exp1 --port 8082 --no-browser --static-only
