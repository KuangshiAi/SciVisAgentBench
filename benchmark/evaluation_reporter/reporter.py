"""
Core reporter logic for generating evaluation reports.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil
import base64


class EvaluationReporter:
    """Generate HTML reports from evaluation results."""

    def __init__(
        self,
        agent_name: str,
        test_results_dir: Path,
        cases_dir: Path,
        yaml_path: Path,
        config: Dict[str, Any],
        output_dir: Path,
        agent_mode: Optional[str] = None
    ):
        """
        Initialize the reporter.

        Args:
            agent_name: Name of the agent being evaluated
            test_results_dir: Directory containing JSON result files
            cases_dir: Directory containing test cases
            yaml_path: Path to YAML file with test definitions
            config: Agent configuration dictionary
            output_dir: Directory to output the report
            agent_mode: Full agent mode string for locating result images (e.g., 'chatvis_claude-sonnet-4-5_exp1')
        """
        self.agent_name = agent_name
        self.agent_mode = agent_mode  # Use provided agent_mode if available
        self.test_results_dir = test_results_dir
        self.cases_dir = cases_dir
        self.yaml_path = yaml_path
        self.config = config
        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_test_results(self) -> List[Dict[str, Any]]:
        """Load all JSON test result files."""
        results = []
        json_files = sorted(self.test_results_dir.glob("*_result_*.json"))

        print(f"   Found {len(json_files)} result files")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_result_file'] = json_file.name
                    # Extract timestamp from filename (e.g., line-plot_result_1770505853.json)
                    filename_parts = json_file.stem.split('_')
                    data['_timestamp'] = int(filename_parts[-1]) if filename_parts and filename_parts[-1].isdigit() else 0
                    results.append(data)
            except Exception as e:
                print(f"   ⚠️  Error loading {json_file.name}: {e}")

        # Group by case_name and keep only the latest result for each case
        case_results = {}
        for result in results:
            case_name = result.get('case_name', 'unknown')
            timestamp = result.get('_timestamp', 0)

            if case_name not in case_results or timestamp > case_results[case_name].get('_timestamp', 0):
                case_results[case_name] = result

        # Sort by case name for consistent ordering
        sorted_cases = sorted(case_results.values(), key=lambda x: x.get('case_name', ''))

        # Print which files were selected
        print(f"   Selected {len(sorted_cases)} latest results (one per case)")
        for case in sorted_cases:
            print(f"      {case.get('case_name')}: {case.get('_result_file')} (timestamp: {case.get('_timestamp')})")

        return sorted_cases

    def load_yaml_cases(self) -> Dict[str, Dict[str, Any]]:
        """Load test case definitions from YAML."""
        try:
            if not self.yaml_path.exists():
                print(f"   ⚠️  YAML file not found: {self.yaml_path}")
                return {}

            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)

            if not yaml_data:
                print(f"   ⚠️  YAML file is empty or could not be parsed: {self.yaml_path}")
                return {}

            cases = {}
            import re

            for idx, item in enumerate(yaml_data):
                if 'vars' in item:
                    # Extract case name from the question
                    question = item['vars'].get('question', '')
                    case_name = None

                    # Try to extract case name from file paths in the question
                    # Pattern matches directory name before /data/ or /results/ (includes underscores)
                    match = re.search(r'([a-z0-9_-]+)/(?:data|results)/', question)
                    if match:
                        case_name = match.group(1)
                    else:
                        # Fallback: use operation_{index+1} naming pattern
                        # This works for napari workflows and similar cases
                        case_name = f"operation_{idx + 1}"

                    if case_name:
                        cases[case_name] = item

            return cases
        except yaml.YAMLError as e:
            print(f"   ⚠️  YAML parsing error in {self.yaml_path}: {e}")
            print(f"   ⚠️  Report will be generated without YAML case definitions")
            return {}
        except KeyboardInterrupt:
            print(f"   ⚠️  YAML parsing interrupted (file might be malformed)")
            print(f"   ⚠️  Report will be generated without YAML case definitions")
            return {}
        except Exception as e:
            print(f"   ⚠️  Error loading YAML: {e}")
            print(f"   ⚠️  Report will be generated without YAML case definitions")
            return {}

    def compute_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics from all results."""
        import math

        total_cases = len(results)
        completed_cases = len([r for r in results if r.get('evaluation', {}).get('status') == 'completed'])

        total_score = 0
        max_score = 0
        vision_scores = []
        code_scores = []
        psnr_values = []
        ssim_values = []
        lpips_values = []

        for result in results:
            eval_data = result.get('evaluation', {})
            scores = eval_data.get('scores', {})

            # Include ALL cases in max_score calculation (both completed and failed)
            # This gives a true overall percentage across all test cases
            case_max_score = scores.get('max_possible_score', 0) if scores else 0
            max_score += case_max_score

            # For completed cases, add their earned scores and collect metrics
            if eval_data.get('status') == 'completed':
                total_score += scores.get('total_score', 0) if scores else 0
            # For failed cases, total_score should already be 0 (or we don't add anything)
            # This ensures failed cases contribute 0/{max_possible_score} to the overall score

                # Vision scores
                subtype_results = eval_data.get('subtype_results', {})
                if 'vision' in subtype_results:
                    vision_data = subtype_results['vision']
                    vision_scores.append(vision_data.get('scores', {}).get('percentage', 0))

                    # Image metrics - exclude infinite PSNR values
                    image_metrics = vision_data.get('image_metrics', {}).get('averaged_metrics', {})
                    if image_metrics.get('psnr') is not None:
                        psnr = image_metrics['psnr']
                        # Only include finite PSNR values
                        if not math.isinf(psnr):
                            psnr_values.append(psnr)
                    if image_metrics.get('ssim') is not None:
                        ssim_values.append(image_metrics['ssim'])
                    if image_metrics.get('lpips') is not None:
                        lpips_values.append(image_metrics['lpips'])

                # Code scores
                if 'code' in subtype_results:
                    code_data = subtype_results['code']
                    code_scores.append(code_data.get('scores', {}).get('percentage', 0))

        # Compute average metrics (non-scaled)
        avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else None
        avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else None
        avg_lpips = sum(lpips_values) / len(lpips_values) if lpips_values else None

        # Compute scaled metrics based on completion rate
        completion_rate = completed_cases / total_cases if total_cases > 0 else 0

        psnr_scaled = (completion_rate * avg_psnr) if avg_psnr is not None else None
        ssim_scaled = (completion_rate * avg_ssim) if avg_ssim is not None else None
        # For LPIPS, lower is better, so scale: 1.0 - completion_rate * (1.0 - avg_lpips)
        lpips_scaled = (1.0 - completion_rate * (1.0 - avg_lpips)) if avg_lpips is not None else None

        return {
            'total_cases': total_cases,
            'completed_cases': completed_cases,
            'total_score': total_score,
            'max_score': max_score,
            'overall_percentage': (total_score / max_score * 100) if max_score > 0 else 0,
            'avg_vision_score': sum(vision_scores) / len(vision_scores) if vision_scores else 0,
            'avg_code_score': sum(code_scores) / len(code_scores) if code_scores else 0,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'avg_lpips': avg_lpips,
            'psnr_scaled': psnr_scaled,
            'ssim_scaled': ssim_scaled,
            'lpips_scaled': lpips_scaled,
            'completion_rate': completion_rate,
            'psnr_count': len(psnr_values),  # Number of valid (finite) PSNR values
        }

    def load_token_usage_from_test_results(self, results: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """
        Load token usage from test result files in each case directory.

        Returns a dict mapping case_name to token_usage dict.
        """
        token_usage_map = {}

        for result in results:
            case_name = result.get('case_name', 'unknown')
            case_dir = self.cases_dir / case_name

            if not case_dir.exists():
                continue

            # Get agent_mode from result, fallback to "pvpython" for legacy results
            agent_mode = result.get('agent_mode', 'pvpython')

            # Look for test result JSON files in test_results/{agent_mode}/
            test_results_dir = case_dir / "test_results" / agent_mode
            if not test_results_dir.exists():
                continue

            # Find all test result JSON files
            test_result_files = list(test_results_dir.glob("test_result_*.json"))
            if not test_result_files:
                continue

            # Sort by timestamp (extracted from filename) and get the latest
            def get_timestamp(filepath):
                stem = filepath.stem  # e.g., "test_result_1770506357"
                parts = stem.split('_')
                try:
                    return int(parts[-1])
                except (ValueError, IndexError):
                    return 0

            latest_file = max(test_result_files, key=get_timestamp)

            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                    token_usage = test_data.get('token_usage', {})
                    if token_usage:
                        token_usage_map[case_name] = token_usage
            except Exception as e:
                print(f"   ⚠️  Error loading token usage from {latest_file}: {e}")

        return token_usage_map

    def copy_images(self, results: List[Dict[str, Any]]):
        """Copy result images and ground truth images to output directory. Mark cases as failures if images not found."""
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for result in results:
            case_name = result.get('case_name', 'unknown')
            case_dir = self.cases_dir / case_name

            if not case_dir.exists():
                continue

            # Use provided agent_mode if available, otherwise detect from result data
            if self.agent_mode:
                agent_mode = self.agent_mode
            else:
                agent_mode = self._detect_agent_mode(result)

            # Try to find and copy result images
            result_img_found = False

            # Pattern 1: results/{agent_mode}/{case_name}.png (use agent_mode first)
            result_img = case_dir / "results" / agent_mode / f"{case_name}.png"
            if result_img.exists():
                shutil.copy(result_img, images_dir / f"{case_name}_result.png")
                result_img_found = True

            # Pattern 2: Fallback to other modes if agent_mode not specified
            if not result_img_found and not self.agent_mode:
                for mode in ['pvpython', 'mcp', 'generic']:
                    result_img = case_dir / "results" / mode / f"{case_name}.png"
                    if result_img.exists():
                        shutil.copy(result_img, images_dir / f"{case_name}_result.png")
                        result_img_found = True
                        break

            # Pattern 3: evaluation_results/{agent_mode}/screenshots/result_*.png
            if not result_img_found:
                modes_to_try = [agent_mode] if self.agent_mode else [agent_mode, 'pvpython', 'mcp', 'generic']
                for mode in modes_to_try:
                    eval_results_dir = case_dir / "evaluation_results" / mode / "screenshots"
                    if eval_results_dir.exists():
                        # Look for any result_*.png files
                        result_imgs = list(eval_results_dir.glob("result_*.png"))
                        if result_imgs:
                            # Use the first one found, or prefer diagonal view
                            img_to_copy = result_imgs[0]
                            for img in result_imgs:
                                if 'diagonal' in img.name:
                                    img_to_copy = img
                                    break
                            shutil.copy(img_to_copy, images_dir / f"{case_name}_result.png")
                            result_img_found = True
                            break

            # Mark case as failure if result image not found
            if not result_img_found:
                print(f"   ⚠️  Result image not found for {case_name}, marking as failure")
                result['image_missing'] = True
                # Set scores to 0 if not already failed
                if result.get('status') == 'completed':
                    result['status'] = 'failed'
                    result['error'] = f'Result image not found at: results/{agent_mode}/{case_name}.png'

                    # Set scores to 0 in evaluation data
                    if 'evaluation' in result and 'scores' in result['evaluation'] and result['evaluation']['scores']:
                        result['evaluation']['scores']['total_score'] = 0
                        result['evaluation']['scores']['percentage'] = 0.0
                    # Also mark evaluation status as failed
                    if 'evaluation' in result:
                        result['evaluation']['status'] = 'failed'

            # Try to find and copy ground truth images
            gt_img_found = False

            # Pattern 1: GS/{case_name}_gs.png
            gt_img = case_dir / "GS" / f"{case_name}_gs.png"
            if gt_img.exists():
                shutil.copy(gt_img, images_dir / f"{case_name}_gt.png")
                gt_img_found = True

            # Pattern 2: GS/gs_*.png (try diagonal view first)
            if not gt_img_found:
                gs_dir = case_dir / "GS"
                if gs_dir.exists():
                    gs_imgs = list(gs_dir.glob("gs_*.png"))
                    if gs_imgs:
                        # Prefer diagonal view
                        img_to_copy = gs_imgs[0]
                        for img in gs_imgs:
                            if 'diagonal' in img.name:
                                img_to_copy = img
                                break
                        shutil.copy(img_to_copy, images_dir / f"{case_name}_gt.png")
                        gt_img_found = True

    def _detect_agent_mode(self, result: Dict[str, Any]) -> str:
        """Detect the agent mode from result data."""
        # Try to extract from task_description
        task_desc = result.get('task_description', '')
        if 'agent_mode' in task_desc:
            # Extract agent_mode value from text like 'Your agent_mode is "claude_code_claude-sonnet-4-5_exp_default"'
            # Use [^"]+ to match everything inside quotes (including hyphens, underscores, etc.)
            import re
            match = re.search(r'agent_mode["\s]*is["\s]+"([^"]+)"', task_desc)
            if match:
                return match.group(1)

        # Fallback to agent name
        return self.agent_name

    def mark_no_visualization_as_failure(self, results: List[Dict[str, Any]], yaml_cases: Dict[str, Any]):
        """Mark cases with no visualization output (missing image/text files) as failures."""
        import os

        for result in results:
            case_name = result.get('case_name', 'unknown')
            eval_data = result.get('evaluation', {})

            # Skip if already failed (check both top-level and evaluation status)
            if result.get('status') == 'failed' or eval_data.get('status') == 'failed':
                continue

            # Determine the results directory path
            # Format: {cases_dir}/{case_name}/results/{agent_mode}/
            if not self.agent_mode:
                # Skip checking if agent_mode is not set
                continue

            results_dir = self.cases_dir / case_name / "results" / self.agent_mode

            # Check if visualization image exists
            # Format: {cases_dir}/{case_name}/results/{agent_mode}/{case_name}.png
            viz_image_path = results_dir / f"{case_name}.png"
            has_visualization = viz_image_path.exists()

            # Mark as failure if visualization image is missing
            if not has_visualization:
                print(f"   ⚠️  {case_name}: no visualization image at {viz_image_path}, marking as failure")
                result['no_visualization'] = True

                # Update both top-level and evaluation-level status for consistency
                result['status'] = 'failed'

                # Ensure 'evaluation' key exists
                if 'evaluation' not in result:
                    result['evaluation'] = {}

                result['evaluation']['status'] = 'failed'

                # Set error message
                error_msg = f"No visualization image found at {viz_image_path}"
                if not result.get('error'):
                    result['error'] = error_msg
                else:
                    result['error'] = f"{result['error']}; {error_msg}"

                # Set total score and percentage to 0 (ensure scores structure exists)
                if 'evaluation' in result and 'scores' in result['evaluation'] and result['evaluation']['scores']:
                    result['evaluation']['scores']['total_score'] = 0
                    result['evaluation']['scores']['percentage'] = 0.0

    def generate_report(self) -> Path:
        """Generate the HTML report."""
        # Load all data
        print("   Loading test results...")
        results = self.load_test_results()
        print(f"   Loaded {len(results)} test cases")

        print("   Loading YAML definitions...")
        yaml_cases = self.load_yaml_cases()
        print(f"   Loaded {len(yaml_cases)} case definitions")

        print("   Computing summary statistics...")
        summary = self.compute_summary_stats(results)

        print("   Loading token usage from test results...")
        token_usage_map = self.load_token_usage_from_test_results(results)
        print(f"   Loaded token usage for {len(token_usage_map)} cases")

        print("   Copying images...")
        self.copy_images(results)

        print("   Checking for cases with no visualization output...")
        self.mark_no_visualization_as_failure(results, yaml_cases)

        # Recompute summary statistics after marking failures
        print("   Recomputing summary statistics with updated scores...")
        summary = self.compute_summary_stats(results)

        # Sort results by case name
        results = sorted(results, key=lambda x: x.get('case_name', ''))

        # Generate HTML
        print("   Generating HTML...")
        html = self.generate_html(results, yaml_cases, summary, token_usage_map)

        # Write to file
        report_path = self.output_dir / "report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return report_path

    def generate_html(
        self,
        results: List[Dict[str, Any]],
        yaml_cases: Dict[str, Dict[str, Any]],
        summary: Dict[str, Any],
        token_usage_map: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate the HTML report content."""
        from evaluation_reporter.templates.report_template import generate_html_template

        return generate_html_template(
            agent_name=self.agent_name,
            config=self.config,
            results=results,
            yaml_cases=yaml_cases,
            summary=summary,
            token_usage_map=token_usage_map,
            generated_time=datetime.now().isoformat()
        )
