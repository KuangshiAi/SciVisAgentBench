conda activate paraview_mcp
python -m benchmark.evaluation_framework.run_evaluation `
  --agent paraview_mcp `
  --config benchmark/configs/paraview_mcp/config_anthropic.json `
  --yaml benchmark/eval_cases/paraview/main_cases.yaml `
  --cases SciVisAgentBench-tasks/main `
  --eval-model gpt-5.2 `
  --eval-only

python -m benchmark.evaluation_framework.run_evaluation `
  --agent chatvis `
  --config benchmark/configs/chatvis/config_anthropic.json `
  --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml `
  --cases SciVisAgentBench-tasks/chatvis_bench `
  --eval-model gpt-5.2 `
  --clear-result

python -m benchmark.evaluation_framework.run_evaluation `
  --agent paraview_mcp `
  --config benchmark/configs/paraview_mcp/config_anthropic.json `
  --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml `
  --cases SciVisAgentBench-tasks/anonymized_datasets `
  --eval-model gpt-5.2 `
  --clear-result

python -m benchmark.evaluation_reporter.run_reporter `
    --agent chatvis `
    --config benchmark/configs/chatvis/config_anthropic.json `
    --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml `
    --cases SciVisAgentBench-tasks/chatvis_bench `
    --test-results test_results/chatvis_bench/chatvis `
    --output eval_reports/chatvis_bench-chatvis `
    --port 8081

python -m benchmark.evaluation_reporter.run_reporter `
    --agent paraview_mcp `
    --config benchmark/configs/chatvis/config_anthropic.json `
    --yaml benchmark/eval_cases/paraview/main_cases.yaml `
    --cases SciVisAgentBench-tasks/main `
    --test-results test_results/main/paraview_mcp `
    --output eval_reports/main-paraview_mcp `
    --port 8081

python -m benchmark.evaluation_reporter.run_reporter `
    --agent paraview_mcp `
    --config benchmark/configs/chatvis/config_anthropic.json `
    --yaml benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml `
    --cases SciVisAgentBench-tasks/anonymized_datasets `
    --test-results test_results/anonymized_datasets/paraview_mcp `
    --output eval_reports/anonymized-paraview_mcp `
    --port 8081


python -m benchmark.human_judge.run_human_eval `
    --agent chatvis `
    --config benchmark/configs/chatvis/config_openai.json `
    --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml `
    --cases SciVisAgentBench-tasks/chatvis_bench `
    --port 8081