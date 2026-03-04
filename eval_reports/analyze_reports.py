#!/usr/bin/env python3
"""
Analyze evaluation reports and generate comprehensive statistics.

Compares different agents and task sets across multiple trials.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics

def extract_metrics_from_html(html_path: Path) -> Dict:
    """Extract metrics from an HTML report file."""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        metrics = {}

        # Extract overall score (percentage)
        match = re.search(r'Overall Score</h3>\s*<div class="value[^"]*">\s*([0-9.]+)%', html_content)
        if match:
            metrics['overall_score'] = float(match.group(1))

        # Extract points from sub-value
        match = re.search(r'<div class="sub-value">\s*([0-9]+)/([0-9]+) Points', html_content)
        if match:
            metrics['points_earned'] = int(match.group(1))
            metrics['points_total'] = int(match.group(2))

        # Extract completed test cases
        match = re.search(r'Test Cases</h3>\s*<div class="value">\s*([0-9]+)/([0-9]+)', html_content)
        if match:
            metrics['completed_cases'] = int(match.group(1))
            metrics['total_cases'] = int(match.group(2))
            metrics['completion_rate'] = (int(match.group(1)) / int(match.group(2)) * 100) if int(match.group(2)) > 0 else 0

        # Extract vision score
        match = re.search(r'Avg Vision Score</h3>\s*<div class="value">\s*([0-9.]+)%', html_content)
        if match:
            metrics['avg_vision_score'] = float(match.group(1))

        # Extract visualization quality
        match = re.search(r'Visualization Quality</h3>\s*<div class="value">\s*([0-9]+)/([0-9]+)', html_content)
        if match:
            metrics['viz_quality_earned'] = int(match.group(1))
            metrics['viz_quality_total'] = int(match.group(2))

        # Extract PSNR
        match = re.search(r'PSNR \(Scaled\)</h3>\s*<div class="value">\s*([0-9.]+)\s*dB', html_content)
        if match:
            metrics['psnr'] = float(match.group(1))

        # Extract valid PSNR count
        match = re.search(r'Peak SNR \(([0-9]+)/([0-9]+) valid\)', html_content)
        if match:
            metrics['psnr_valid_count'] = int(match.group(1))
            metrics['psnr_total_count'] = int(match.group(2))

        # Extract SSIM
        match = re.search(r'SSIM \(Scaled\)</h3>\s*<div class="value">\s*([0-9.]+)', html_content)
        if match:
            metrics['ssim'] = float(match.group(1))

        # Extract LPIPS
        match = re.search(r'LPIPS \(Scaled\)</h3>\s*<div class="value">\s*([0-9.]+)', html_content)
        if match:
            metrics['lpips'] = float(match.group(1))

        # Extract aggregated vision quality points from individual cases
        viz_quality_pattern = r'<div class="metric-label">Visualization Quality</div>\s*<div class="metric-value">([0-9]+)/([0-9]+)</div>'
        viz_matches = re.findall(viz_quality_pattern, html_content)
        if viz_matches:
            total_earned = sum(int(m[0]) for m in viz_matches)
            total_possible = sum(int(m[1]) for m in viz_matches)
            metrics['viz_quality_earned'] = total_earned
            metrics['viz_quality_total'] = total_possible

        # Extract per-case results for pass@k calculation
        # Look for case titles and their scores
        case_results = []
        # Match pattern: <h2 class="case-title">📝 case-name</h2>...optional status badge...<div class="case-score">X/Y (Z%)</div>
        # Extract case name and percentage score
        case_block_pattern = r'<h2 class="case-title">.*?\s+([a-zA-Z0-9_-]+)</h2>.*?<div class="case-score">.*?\(([0-9.]+)%\)</div>'
        for match in re.finditer(case_block_pattern, html_content, re.DOTALL):
            case_name = match.group(1).strip()
            percentage = float(match.group(2))
            # A case passes if it scores >= 50%
            passed = percentage >= 50.0
            case_results.append({
                'name': case_name,
                'percentage': percentage,
                'passed': passed
            })

        metrics['case_results'] = case_results

        return metrics

    except Exception as e:
        print(f"[ERROR] Error parsing {html_path}: {e}")
        return {}

def calculate_pass_at_k(trials: List[List[Dict]], k: int) -> Dict[str, float]:
    """
    Calculate pass@k metric: probability that at least one of k trials passes.

    Args:
        trials: List of trials, where each trial is a list of case results
        k: Number of trials to sample

    Returns:
        Dict with pass@k for each case
    """
    if not trials or k > len(trials):
        return {}

    # Group results by case name
    case_trials = defaultdict(list)
    for trial in trials:
        for case in trial:
            case_trials[case['name']].append(case['passed'])

    # Calculate pass@k for each case
    pass_at_k = {}
    for case_name, passes in case_trials.items():
        if len(passes) >= k:
            # pass@k = 1 - (number of ways to choose k failures) / (number of ways to choose k trials)
            n_trials = len(passes)
            n_passes = sum(passes)
            n_fails = n_trials - n_passes

            # If all pass or all fail, it's deterministic
            if n_fails == 0:
                pass_at_k[case_name] = 1.0
            elif n_passes == 0:
                pass_at_k[case_name] = 0.0
            else:
                # Calculate using binomial coefficient
                from math import comb
                if k <= n_fails:
                    pass_at_k[case_name] = 1 - (comb(n_fails, k) / comb(n_trials, k))
                else:
                    pass_at_k[case_name] = 1.0

    return pass_at_k

def calculate_pass_k(trials: List[List[Dict]], k: int) -> Dict[str, float]:
    """
    Calculate pass^k metric: probability that all k trials pass.

    Args:
        trials: List of trials, where each trial is a list of case results
        k: Number of trials

    Returns:
        Dict with pass^k for each case
    """
    if not trials or k > len(trials):
        return {}

    # Group results by case name
    case_trials = defaultdict(list)
    for trial in trials:
        for case in trial:
            case_trials[case['name']].append(case['passed'])

    # Calculate pass^k for each case
    pass_k = {}
    for case_name, passes in case_trials.items():
        if len(passes) >= k:
            # pass^k = (number of ways to choose k passes) / (number of ways to choose k trials)
            n_trials = len(passes)
            n_passes = sum(passes)

            # If all pass or all fail, it's deterministic
            if n_passes == n_trials:
                pass_k[case_name] = 1.0
            elif n_passes < k:
                pass_k[case_name] = 0.0
            else:
                # Calculate using binomial coefficient
                from math import comb
                pass_k[case_name] = comb(n_passes, k) / comb(n_trials, k)

    return pass_k

def analyze_reports(output_file: str = 'evaluation_statistics.md'):
    """Analyze all reports and generate statistics.

    Args:
        output_file: Path to save the markdown report
    """

    # Define report groups
    report_groups = {
        'ParaView MCP (ParaView Tasks)': {
            'pattern': 'eval_reports/paraview/paraview_mcp_*',
            'description': 'ParaView MCP agent on standard ParaView visualization tasks'
        },
        'ChatVis (ParaView Tasks)': {
            'pattern': 'eval_reports/paraview/chatvis_*',
            'description': 'ChatVis agent on standard ParaView visualization tasks'
        },
        'ParaView MCP (Anonymized Tasks)': {
            'pattern': 'eval_reports/anonymized/paraview_mcp_*',
            'description': 'ParaView MCP agent on "what obj" anonymized tasks'
        }
    }

    # Collect data for each group
    results = {}

    for group_name, group_config in report_groups.items():
        print(f"\nAnalyzing: {group_name}")
        print(f"  Pattern: {group_config['pattern']}")

        # Find all matching reports
        base_dir = Path('.')
        pattern = group_config['pattern']
        report_dirs = list(base_dir.glob(pattern))

        print(f"  Found {len(report_dirs)} reports")

        # Extract metrics from each report
        trials_data = []
        for report_dir in sorted(report_dirs):
            report_file = report_dir / 'report.html'
            if report_file.exists():
                print(f"    - {report_dir.name}")
                metrics = extract_metrics_from_html(report_file)
                if metrics:
                    trials_data.append(metrics)

        results[group_name] = trials_data

    # Generate statistics report (both console and markdown)
    print("\n" + "="*80)
    print("EVALUATION REPORT STATISTICS")
    print("="*80)

    # Start markdown content
    md_lines = []
    md_lines.append("# Evaluation Report Statistics\n")
    md_lines.append(f"*Generated: {Path(__file__).name}*\n")

    for group_name, trials in results.items():
        if not trials:
            print(f"\n{group_name}: No data available")
            md_lines.append(f"\n## {group_name}\n")
            md_lines.append("No data available\n")
            continue

        print(f"\n{group_name}")
        print(f"  Number of trials: {len(trials)}")
        print(f"  {report_groups[group_name]['description']}")
        print(f"  {'-'*76}")

        md_lines.append(f"\n## {group_name}\n")
        md_lines.append(f"**Number of trials:** {len(trials)}\n\n")
        md_lines.append(f"*{report_groups[group_name]['description']}*\n")

        # Calculate statistics for each metric
        metrics_to_analyze = [
            ('overall_score', 'Overall Score', '%', None),
            ('points_earned', 'Overall Score (Points)', '', 'points_total'),
            ('completion_rate', 'Completion Rate', '%', None),
            ('avg_vision_score', 'Avg Vision Score', '%', None),
            ('viz_quality_earned', 'Vision Quality (Points)', '', 'viz_quality_total'),
            ('psnr', 'PSNR', 'dB', None),
            ('ssim', 'SSIM', '', None),
            ('lpips', 'LPIPS', '', None),
        ]

        # Start markdown table
        md_lines.append("\n### Metrics Summary\n")
        md_lines.append("| Metric | Mean | Std Dev | Variance | Min | Max |\n")
        md_lines.append("|--------|------|---------|----------|-----|-----|\n")

        for item in metrics_to_analyze:
            if len(item) == 4:
                metric_key, metric_name, unit, total_key = item
            else:
                metric_key, metric_name, unit = item
                total_key = None

            values = [t[metric_key] for t in trials if metric_key in t]

            if not values:
                continue

            mean_val = statistics.mean(values)
            if len(values) > 1:
                variance = statistics.variance(values)
                std_dev = statistics.stdev(values)
            else:
                variance = 0.0
                std_dev = 0.0

            min_val = min(values)
            max_val = max(values)

            # For points metrics, also show the total denominator
            if total_key:
                total_values = [t[total_key] for t in trials if total_key in t]
                if total_values:
                    mean_total = statistics.mean(total_values)
                    print(f"  {metric_name:30} Mean: {mean_val:7.2f}/{mean_total:.0f}  Std: {std_dev:6.2f}  Var: {variance:7.2f}  "
                          f"Min: {min_val:7.2f}  Max: {max_val:7.2f}")
                    md_lines.append(f"| {metric_name} | {mean_val:.2f}/{mean_total:.0f} | {std_dev:.2f} | {variance:.2f} | {min_val:.2f} | {max_val:.2f} |\n")
            else:
                print(f"  {metric_name:30} Mean: {mean_val:7.2f}{unit:3}  Std: {std_dev:6.2f}  Var: {variance:7.2f}  "
                      f"Min: {min_val:7.2f}  Max: {max_val:7.2f}")
                md_lines.append(f"| {metric_name} | {mean_val:.2f}{unit} | {std_dev:.2f} | {variance:.2f} | {min_val:.2f}{unit} | {max_val:.2f}{unit} |\n")

        # Calculate pass@k and pass^k for different k values
        if len(trials) >= 2:
            print(f"\n  Pass@k and Pass^k Metrics:")
            print(f"  {'-'*76}")

            md_lines.append("\n### Pass@k and Pass^k Metrics\n")
            md_lines.append("| k | pass@k | pass^k |\n")
            md_lines.append("|---|--------|--------|\n")

            # Extract case results from all trials
            all_case_results = [t.get('case_results', []) for t in trials]

            # Calculate for k=1, 2, 3, etc.
            for k in range(1, min(len(trials) + 1, 6)):
                pass_at_k = calculate_pass_at_k(all_case_results, k)
                pass_k = calculate_pass_k(all_case_results, k)

                if pass_at_k:
                    avg_pass_at_k = statistics.mean(pass_at_k.values())
                    avg_pass_k = statistics.mean(pass_k.values())

                    print(f"  k={k}:  pass@{k} = {avg_pass_at_k:.3f}  pass^{k} = {avg_pass_k:.3f}")
                    md_lines.append(f"| {k} | {avg_pass_at_k:.3f} | {avg_pass_k:.3f} |\n")

        # Print detailed case-by-case pass rates
        if trials and 'case_results' in trials[0]:
            print(f"\n  Per-Case Success Rates:")
            print(f"  {'-'*76}")

            # Collect all case names
            all_cases = set()
            for trial in trials:
                for case in trial.get('case_results', []):
                    all_cases.add(case['name'])

            # Calculate success rate for each case
            case_success_rates = {}
            for case_name in sorted(all_cases):
                successes = 0
                total = 0
                for trial in trials:
                    for case in trial.get('case_results', []):
                        if case['name'] == case_name:
                            total += 1
                            if case['passed']:
                                successes += 1

                if total > 0:
                    case_success_rates[case_name] = (successes, total, successes / total)

            # Print top failures and successes
            sorted_cases = sorted(case_success_rates.items(), key=lambda x: x[1][2])

            print(f"  Top 5 Failures:")
            md_lines.append("\n### Top 5 Failures\n")
            md_lines.append("| Case | Success Rate |\n")
            md_lines.append("|------|-------------|\n")
            for case_name, (succ, total, rate) in sorted_cases[:5]:
                print(f"    {case_name:30} {succ}/{total} ({rate*100:5.1f}%)")
                md_lines.append(f"| {case_name} | {succ}/{total} ({rate*100:.1f}%) |\n")

            print(f"\n  Top 5 Successes:")
            md_lines.append("\n### Top 5 Successes\n")
            md_lines.append("| Case | Success Rate |\n")
            md_lines.append("|------|-------------|\n")
            for case_name, (succ, total, rate) in sorted_cases[-5:]:
                print(f"    {case_name:30} {succ}/{total} ({rate*100:5.1f}%)")
                md_lines.append(f"| {case_name} | {succ}/{total} ({rate*100:.1f}%) |\n")

    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80)

    # Save markdown report
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)

    print(f"\nMarkdown report saved to: {output_path.absolute()}")

if __name__ == "__main__":
    analyze_reports()
