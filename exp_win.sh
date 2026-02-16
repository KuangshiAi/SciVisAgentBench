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
    --config benchmark/configs/paraview_mcp/config_anthropic.json `
    --yaml benchmark/eval_cases/paraview/main_cases.yaml `
    --cases SciVisAgentBench-tasks/main `
    --test-results test_results/main/paraview_mcp `
    --output eval_reports/main-paraview_mcp `
    --port 8081

python -m benchmark.evaluation_reporter.run_reporter `
    --agent paraview_mcp `
    --config benchmark/configs/paraview_mcp/config_anthropic.json `
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


python -m benchmark.evaluation_reporter.run_reporter \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/workflows/eval_analysis_workflows.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis/workflows \
    --test-results test_results/molecular_vis/gmx_vmd_mcp/eval_analysis_workflows \
    --output eval_reports/molecular_vis-gmx_vmd_mcp \
    --port 8081

python -m benchmark.evaluation_framework.run_evaluation \
    --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/workflows/eval_analysis_workflows.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis/workflows \
    --eval-model gpt-5.2 \
    --eval-only


python -m benchmark.evaluation_reporter.run_reporter \
    --agent bioimage_agent \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_visualization_workflows.yaml\
    --cases SciVisAgentBench-tasks/bioimage_data/eval_visualization_workflows \
    --test-results /Users/kuangshiai/Documents/ND-VIS/Code/SciVisAgentBench/test_results/bioimage_data/napari_mcp/eval_visualization_workflows \
    --output eval_reports/bioimage-napari_mcp/eval_visualization_workflows \
    --port 8081

python -m benchmark.evaluation_reporter.run_reporter \
    --agent bioimage_agent \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_iso_surface_determination.yaml\
    --cases SciVisAgentBench-tasks/bioimage_data/eval_iso_surface_determination \
    --test-results /Users/kuangshiai/Documents/ND-VIS/Code/SciVisAgentBench/test_results/bioimage_data/napari_mcp/eval_iso_surface_determination \
    --output eval_reports/bioimage-napari_mcp/eval_iso_surface_determination \
    --port 8081

python -m benchmark.evaluation_reporter.run_reporter \
    --agent bioimage_agent \
    --config benchmark/configs/napari_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/napari/1_workflows/eval_figure_recreation.yaml\
    --cases SciVisAgentBench-tasks/bioimage_data/eval_figure_recreation \
    --test-results /Users/kuangshiai/Documents/ND-VIS/Code/SciVisAgentBench/test_results/bioimage_data/napari_mcp/eval_figure_recreation \
    --output eval_reports/bioimage-napari_mcp/eval_figure_recreation \
    --port 8081