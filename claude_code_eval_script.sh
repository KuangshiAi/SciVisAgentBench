export PV_FORCE_OFFSCREEN_RENDERING=1
export VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1

######### paraview ###########
## single eval test
# python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview/ --eval-model gpt-5.2 --case bonsai --verbose

### paraview cases claude code
python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview/ --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1


python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview/ --eval-model gpt-5.2 --eval-only --experiment-number exp1


python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/paraview_cases.yaml --cases SciVisAgentBench-tasks/paraview/ --test-results test_results/paraview/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/paraview/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081


######### anonymized #########
### paraview cases claude code
python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets/ --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1

python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets/ --eval-model gpt-5.2 --eval-only --experiment-number exp1 --openai-base-url https://livai-api.llnl.gov

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml --cases SciVisAgentBench-tasks/anonymized_datasets/ --test-results test_results/anonymized_datasets/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/sci_volume_data/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081


#### Napari #####
### test
# python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/napari/eval_visualization_workflows.yaml --cases SciVisAgentBench-tasks/bioimage_data/ --eval-model gpt-5.2 --case case_1 --verbose

python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data/ --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1

python benchmark/run_claude_code_eval.py --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data/ --eval-model gpt-5.2 --eval-only --experiment-number exp1

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/napari/eval_visualization_tasks.yaml --cases SciVisAgentBench-tasks/bioimage_data/eval_visualization_tasks/ --test-results /Users/liu42/gitRepo/LC/2026_AgentBenchmark/SciVisAgentBench/test_results/bioimage_data/claude_code_claude-sonnet-4-5_exp1 --output eval_reports/napari/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081


# molecular_VIS (VMD)
# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1

python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp1

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml --cases SciVisAgentBench-tasks/molecular_vis --test-results test_results/molecular_vis/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/molecular_vis/tasks/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081

# Run cases in benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1

python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --eval-model gpt-5.2 --eval-only --experiment-number exp1

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/molecular_vis/eval_analysis_workflows.yaml --cases SciVisAgentBench-tasks/molecular_vis --test-results test_results/molecular_vis/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/molecular_vis/workflows/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081


######## topology #########
python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --eval-model gpt-5.2 --exe-only --verbose --experiment-number exp1

python -m benchmark.run_claude_code_eval --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --eval-model gpt-5.2 --eval-only --experiment-number exp1

python -m benchmark.evaluation_reporter.run_reporter --agent claude_code --config benchmark/configs/claude_code/config.json --yaml benchmark/eval_cases/topology/topology_cases.yaml --cases SciVisAgentBench-tasks/topology --test-results test_results/topology/claude_code_claude-sonnet-4-5_exp1/ --output eval_reports/topology/claude_code_claude-sonnet-4-5_exp1 --agent-mode claude_code_claude-sonnet-4-5_exp1 --port 8081









