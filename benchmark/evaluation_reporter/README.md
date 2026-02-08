# Evaluation Reporter

An interactive HTML report generator for SciVisAgentBench evaluation results.

## Overview

The Evaluation Reporter generates beautiful, interactive HTML reports from test and evaluation results. It provides a comprehensive view of agent performance, including:

- **Overall Performance Summary**: Aggregate statistics across all test cases
- **Individual Case Details**: Per-case breakdowns with visualizations
- **LLM Rubric Scores**: Detailed goal-by-goal evaluations
- **Image Comparisons**: Side-by-side ground truth vs. result visualizations
- **Code Similarity Metrics**: Code quality comparisons (when available)
- **Image Quality Metrics**: PSNR, SSIM, and LPIPS scores
- **Token Usage Statistics**: Input/output token consumption
- **Efficiency Metrics**: Execution time and performance scores

## Features

✨ **Modern, Responsive Design**: Beautiful gradient-based UI that works on all devices
🔍 **Interactive Navigation**: Quick jump-to links for all test cases
📊 **Comprehensive Metrics**: All evaluation data in one place
🎨 **Visual Comparisons**: Side-by-side image comparisons
📈 **Expandable Sections**: Collapsible rubric details to reduce clutter
🚀 **Live Server Mode**: Built-in HTTP server for instant viewing
💾 **Static Export**: Generate standalone HTML files

## Installation

No additional dependencies required! The reporter uses only Python standard library modules.

## Usage

### Basic Usage

Generate and serve a report:

```bash
python -m benchmark.evaluation_reporter.run_reporter \
    --agent chatvis \
    --config benchmark/configs/chatvis/config_anthropic.json \
    --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \
    --cases SciVisAgentBench-tasks/chatvis_bench \
    --test-results test_results/chatvis_bench/chatvis \
    --port 8081
```

This will:
1. Load all test results from the specified directory
2. Generate an interactive HTML report
3. Start a local HTTP server
4. Automatically open the report in your browser

### Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--agent` | Yes | Agent name (e.g., `chatvis`, `topopilot`) |
| `--config` | Yes | Path to agent configuration JSON file |
| `--yaml` | Yes | Path to YAML test cases file |
| `--cases` | Yes | Path to test cases directory |
| `--test-results` | Yes | Path to test results directory (contains JSON files) |
| `--output` | No | Output directory for report (default: `test_results/<agent>_report`) |
| `--port` | No | Port for HTTP server (default: `8080`) |
| `--no-browser` | No | Don't automatically open browser |
| `--static-only` | No | Generate static HTML only, don't start server |

### Examples

#### 1. Generate Report with Custom Port

```bash
python -m benchmark.evaluation_reporter.run_reporter \
    --agent chatvis \
    --config benchmark/configs/chatvis/config_anthropic.json \
    --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \
    --cases SciVisAgentBench-tasks/chatvis_bench \
    --test-results test_results/chatvis_bench/chatvis \
    --port 9000
```

#### 2. Generate Static HTML Only

```bash
python -m benchmark.evaluation_reporter.run_reporter \
    --agent chatvis \
    --config benchmark/configs/chatvis/config_openai.json \
    --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \
    --cases SciVisAgentBench-tasks/chatvis_bench \
    --test-results test_results/chatvis_bench/chatvis \
    --output reports/chatvis_report \
    --static-only
```

The generated report will be saved to `reports/chatvis_report/report.html`.

#### 3. Generate Report Without Auto-Opening Browser

```bash
python -m benchmark.evaluation_reporter.run_reporter \
    --agent chatvis \
    --config benchmark/configs/chatvis/config_anthropic.json \
    --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \
    --cases SciVisAgentBench-tasks/chatvis_bench \
    --test-results test_results/chatvis_bench/chatvis \
    --no-browser
```

## Report Structure

### 1. Overall Performance Summary

The top of the report shows aggregate statistics:

- **Overall Score**: Total score across all test cases
- **Test Cases**: Number of completed vs. total cases
- **Average Vision Score**: Mean visualization quality percentage
- **Average Code Similarity**: Mean code quality match percentage
- **Average PSNR/SSIM/LPIPS**: Image quality metrics
- **Total Tokens**: Cumulative token usage
- **Configuration**: Agent model, provider, and pricing info

### 2. Case Navigation

A sticky navigation bar allows quick jumping to any test case.

### 3. Individual Case Sections

Each test case includes:

#### Task Description
The natural language prompt given to the agent.

#### Visualization Comparison
Side-by-side comparison of:
- **Ground Truth**: Expected visualization
- **Agent Result**: Agent-generated visualization

#### Vision Evaluation Rubrics
Detailed breakdown of each evaluation goal:
- Goal-by-goal scores (0-10 points each)
- LLM explanations for each goal
- Overall assessment

#### Detailed Metrics
Grid of all metrics:
- Visualization Quality score
- Output Generation score
- Efficiency score
- PSNR, SSIM, LPIPS (if available)
- Input/Output token counts

#### Code Similarity (if applicable)
- Code similarity score (0-10 points)
- Raw similarity score (0-1 scale)
- File paths for ground truth and result code

## File Structure

```
benchmark/evaluation_reporter/
├── __init__.py                      # Package initialization
├── run_reporter.py                  # Main CLI entry point
├── reporter.py                      # Core report generation logic
├── templates/
│   ├── __init__.py
│   └── report_template.py           # HTML/CSS/JS templates
└── README.md                        # This file
```

## Output Structure

Generated reports are saved to the output directory with this structure:

```
<output_dir>/
├── report.html          # Main HTML report
└── images/              # Copied visualization images
    ├── case1_gt.png     # Ground truth images
    ├── case1_result.png # Result images
    ├── case2_gt.png
    ├── case2_result.png
    └── ...
```

## Data Sources

The reporter loads data from:

1. **Test Results**: JSON files in `--test-results` directory
   - Each file contains test execution and evaluation data
   - Latest result per case is used if multiple exist

2. **Test Case Definitions**: YAML file specified by `--yaml`
   - Provides task descriptions and rubrics

3. **Visualization Images**: PNG files from cases directory
   - Ground truth: `{case-name}/GS/{case-name}_gs.png`
   - Results: `{case-name}/results/pvpython/{case-name}.png`

4. **Agent Configuration**: JSON file specified by `--config`
   - Model name, provider, pricing information

## Customization

### Modifying the HTML Template

Edit `templates/report_template.py` to customize:
- CSS styles (in `get_css_styles()`)
- HTML structure (in `generate_html_template()`)
- JavaScript behavior (in `get_javascript()`)

### Changing Colors/Styling

The report uses a purple gradient theme by default. To change:

1. Find the CSS in `get_css_styles()` function
2. Modify the color variables:
   ```css
   /* Primary gradient */
   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

   /* Accent colors */
   color: #667eea;
   border-left: 4px solid #667eea;
   ```

### Adding New Metrics

To add new metrics to the report:

1. Update `compute_summary_stats()` in `reporter.py` to calculate the metric
2. Add a new card in `generate_summary_section()` in `report_template.py`
3. Add to individual case sections in `generate_metrics_section()`

## Troubleshooting

### Issue: No images showing in report

**Cause**: Images not found or incorrect paths

**Solution**:
- Ensure images exist in the expected locations:
  - Ground truth: `{cases_dir}/{case-name}/GS/{case-name}_gs.png`
  - Results: `{cases_dir}/{case-name}/results/pvpython/{case-name}.png`
- Check that `--cases` path is correct

### Issue: Missing test cases in report

**Cause**: No JSON result files found

**Solution**:
- Verify `--test-results` path points to directory with `*_result_*.json` files
- Check that evaluation has been run for the test cases

### Issue: Port already in use

**Cause**: Another process using the specified port

**Solution**:
- Use a different port: `--port 8081`
- Or kill the process using the port

### Issue: Browser doesn't open automatically

**Cause**: System configuration or headless environment

**Solution**:
- Manually navigate to the URL shown in terminal
- Or use `--no-browser` flag and open manually

## Integration with Evaluation Framework

### Typical Workflow

1. **Run Tests**:
   ```bash
   python -m benchmark.evaluation_framework.run_evaluation \
       --agent chatvis \
       --config benchmark/configs/chatvis/config_anthropic.json \
       --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \
       --cases SciVisAgentBench-tasks/chatvis_bench \
       --eval-model gpt-5.2
   ```

2. **Generate Report**:
   ```bash
   python -m benchmark.evaluation_reporter.run_reporter \
       --agent chatvis \
       --config benchmark/configs/chatvis/config_anthropic.json \
       --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \
       --cases SciVisAgentBench-tasks/chatvis_bench \
       --test-results test_results/chatvis_bench/chatvis
   ```

3. **View Report**: Browser opens automatically at `http://localhost:8080/report.html`

### Continuous Integration

For CI/CD pipelines, generate static reports:

```bash
# Run evaluation
python -m benchmark.evaluation_framework.run_evaluation ...

# Generate static report (no server)
python -m benchmark.evaluation_reporter.run_reporter \
    --agent chatvis \
    --config ... \
    --yaml ... \
    --cases ... \
    --test-results ... \
    --output ci_reports/chatvis \
    --static-only

# Publish report (example)
aws s3 cp ci_reports/chatvis s3://my-bucket/reports/chatvis/ --recursive
```

## Performance

The reporter is optimized for:
- **Fast Generation**: Typically <5 seconds for 20 test cases
- **Minimal Dependencies**: Uses only Python standard library
- **Small Output**: Compressed CSS/JS, optimized images
- **Responsive Design**: Works on mobile, tablet, and desktop

## Future Enhancements

Potential improvements:
- [ ] Export to PDF
- [ ] Comparison between multiple agents
- [ ] Time-series tracking across evaluation runs
- [ ] Custom theme support via config file
- [ ] Export individual case reports
- [ ] Search/filter functionality

## License

Same as SciVisAgentBench project.

## Support

For issues or questions:
1. Check this README
2. Review example commands
3. Open an issue on GitHub

---

**Generated by SciVisAgentBench Evaluation Reporter v1.0.0**
