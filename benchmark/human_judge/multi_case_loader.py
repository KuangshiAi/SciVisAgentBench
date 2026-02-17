"""
Multi-case evaluation loader that loads cases from a comprehensive YAML file.
Each case can come from different directories and have different evaluation criteria.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.evaluation_helpers.screenshot_helper import take_screenshots_from_state


class MultiCaseLoader:
    """Load evaluation data from a comprehensive cases YAML file."""

    def __init__(self, cases_yaml_path: str):
        """
        Initialize multi-case loader.

        Args:
            cases_yaml_path: Path to YAML file with complete case definitions
        """
        self.cases_yaml_path = Path(cases_yaml_path).resolve()

        # Find workspace root by looking for SciVisAgentBench-tasks directory
        current = self.cases_yaml_path.parent
        while current != current.parent:
            if (current / "SciVisAgentBench-tasks").exists():
                self.workspace_root = current
                break
            current = current.parent
        else:
            # Fallback: assume YAML is in benchmark/eval_cases/*, go up 3 levels
            self.workspace_root = self.cases_yaml_path.parent.parent.parent

        # Load cases from YAML
        self.cases_config = self._load_cases_config()

        # Load evaluation criteria for each unique YAML file
        self.yaml_test_cases = self._load_yaml_test_cases()

    def _load_cases_config(self) -> List[Dict[str, str]]:
        """Load case configurations from YAML file."""
        with open(self.cases_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or 'cases' not in data:
            raise ValueError("Cases YAML must contain a 'cases' key with a list of case definitions")

        cases = data['cases']
        if not isinstance(cases, list):
            raise ValueError("'cases' must be a list")

        # Validate each case has required fields
        for i, case in enumerate(cases):
            if not isinstance(case, dict):
                raise ValueError(f"Case {i} must be a dictionary")
            required_fields = ['name', 'path', 'yaml']
            for field in required_fields:
                if field not in case:
                    raise ValueError(f"Case {i} ({case.get('name', 'unknown')}) missing required field: {field}")

        return cases

    def _load_yaml_test_cases(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load test cases from each unique YAML file."""
        yaml_files = {}

        # Find all unique YAML files
        unique_yamls = set(case['yaml'] for case in self.cases_config)

        for yaml_path in unique_yamls:
            full_path = self.workspace_root / yaml_path
            if not full_path.exists():
                print(f"Warning: YAML file not found: {yaml_path}")
                continue

            with open(full_path, 'r') as f:
                test_cases = yaml.safe_load(f)

            if not isinstance(test_cases, list):
                raise ValueError(f"YAML file {yaml_path} must contain a list of test cases")

            yaml_files[yaml_path] = test_cases

        return yaml_files

    def _get_vision_metrics_for_case(self, case_name: str, yaml_path: str) -> List[Dict[str, str]]:
        """Get vision metrics for a specific case from its YAML file."""
        if yaml_path not in self.yaml_test_cases:
            return []

        test_cases = self.yaml_test_cases[yaml_path]

        # Find the test case matching this case name
        for test_case in test_cases:
            # Extract case name from test case
            question = test_case.get("vars", {}).get("question", "")

            # Try to match case name from save path in question
            if case_name in question:
                # Extract vision metrics
                return self._get_vision_metrics(test_case)

        return []

    def _get_vision_metrics(self, test_case: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract vision-based metrics from a test case."""
        import re
        metrics = []

        if "assert" not in test_case:
            return metrics

        for assertion in test_case["assert"]:
            if assertion.get("type") == "llm-rubric" and assertion.get("subtype") == "vision":
                # Parse the multi-line criteria
                criteria_text = assertion.get("value", "")

                # Split by numbered criteria (support both "1." and "1)" formats)
                lines = criteria_text.strip().split('\n')
                current_criterion = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check if line starts with a numbered item: "1." or "1)" or "1. " or "1) "
                    # Pattern: digit(s) followed by either . or ) within first few characters
                    is_numbered = False
                    criterion_text = line

                    if line and line[0].isdigit():
                        # Look for "1." or "1)" pattern
                        match = re.match(r'^(\d+)[\.\)](\s*)(.*)', line)
                        if match:
                            is_numbered = True
                            criterion_text = match.group(3).strip()  # Get text after number

                    if is_numbered:
                        # Save previous criterion if exists
                        if current_criterion:
                            metrics.append({
                                "criterion": ' '.join(current_criterion).strip()
                            })
                            current_criterion = []

                        # Start new criterion
                        if criterion_text:
                            current_criterion.append(criterion_text)
                    else:
                        # Continuation of current criterion
                        current_criterion.append(line)

                # Don't forget the last criterion
                if current_criterion:
                    metrics.append({
                        "criterion": ' '.join(current_criterion).strip()
                    })

        return metrics

    def _get_task_description_for_case(self, case_name: str, yaml_path: str) -> str:
        """Get task description for a specific case from its YAML file."""
        if yaml_path not in self.yaml_test_cases:
            return ""

        test_cases = self.yaml_test_cases[yaml_path]

        # Find the test case matching this case name
        for test_case in test_cases:
            question = test_case.get("vars", {}).get("question", "")
            if case_name in question:
                return question

        return ""

    def _detect_benchmark_type(self, case_path: str) -> str:
        """Detect benchmark type from case path."""
        if "/chatvis_bench/" in case_path:
            return "chatvis_bench"
        else:
            return "main"

    def _get_ground_truth_images(self, case_dir: Path, case_name: str, benchmark_type: str) -> List[Path]:
        """Get ground truth images/videos for a case."""
        gs_dir = case_dir / "GS"

        if not gs_dir.exists():
            return []

        if benchmark_type == "chatvis_bench":
            # ChatVis: prefer MP4 video for temporal cases
            # Check for MP4 first (web-friendly video)
            gs_video_mp4 = gs_dir / f"{case_name}_gs.mp4"
            if gs_video_mp4.exists():
                return [gs_video_mp4]

            # Fall back to PNG snapshot
            gs_images = list(gs_dir.glob("*_gs.png"))
            if gs_images:
                return gs_images[:1]

            # Last resort: AVI (likely won't play in browser)
            gs_video_avi = gs_dir / f"{case_name}_gs.avi"
            if gs_video_avi.exists():
                return [gs_video_avi]

            return []

        # Main benchmark: check for 3-view format first
        gs_patterns = ["gs_front_view.png", "gs_side_view.png", "gs_diagonal_view.png"]
        gs_images = []

        for pattern in gs_patterns:
            gs_path = gs_dir / pattern
            if gs_path.exists():
                gs_images.append(gs_path)

        # If we have all 3 views, return them
        if len(gs_images) == 3:
            return gs_images

        # Otherwise, check for single image format: {case_name}_gs.png
        single_gs_path = gs_dir / f"{case_name}_gs.png"
        if single_gs_path.exists():
            return [single_gs_path]

        return gs_images

    def _get_result_images(self, case_dir: Path, case_name: str, benchmark_type: str, gs_images: List[Path]) -> List[Path]:
        """Get result images/videos for a case."""
        if benchmark_type == "chatvis_bench":
            # ChatVis: prefer MP4 video for temporal cases in results/pvpython/
            result_dir = case_dir / "results" / "pvpython"
            if result_dir.exists():
                # Check for MP4 first (web-friendly video)
                mp4_file = result_dir / f"{case_name}.mp4"
                if mp4_file.exists():
                    return [mp4_file]

                # Fall back to PNG snapshot
                png_files = list(result_dir.glob("*.png"))
                if png_files:
                    return [png_files[0]]

                # Last resort: AVI (likely won't play in browser)
                avi_file = result_dir / f"{case_name}.avi"
                if avi_file.exists():
                    return [avi_file]

            return []

        # Determine expected format based on ground truth images
        if gs_images:
            gs_names = [gs.name for gs in gs_images]

            # Check if GS has single image format
            if any(f"{case_name}_gs.png" in name for name in gs_names):
                # Single image format: check both results/mcp/ and results/pvpython/
                result_images = []
                for agent_mode in ["mcp", "pvpython"]:
                    result_path = case_dir / "results" / agent_mode / f"{case_name}.png"
                    if result_path.exists():
                        result_images.append(result_path)
                return result_images

            # Check if GS has 3-view format
            if any("gs_front_view.png" in name for name in gs_names):
                # 3-view format: check both mcp and pvpython in evaluation_results
                result_images = []
                for agent_mode in ["mcp", "pvpython"]:
                    screenshots_dir = case_dir / "evaluation_results" / agent_mode / "screenshots"
                    if not screenshots_dir.exists():
                        continue

                    view_patterns = ["result_front_view.png", "result_side_view.png", "result_diagonal_view.png"]
                    agent_screenshots = []
                    for pattern in view_patterns:
                        result_path = screenshots_dir / pattern
                        if result_path.exists():
                            agent_screenshots.append(result_path)

                    # Only add if we have all 3 views for this agent
                    if len(agent_screenshots) == 3:
                        result_images.extend(agent_screenshots)

                return result_images

        # Fallback: try to find any result images
        result_images = []

        # Try 3-view format in evaluation_results
        for agent_mode in ["mcp", "pvpython"]:
            screenshots_dir = case_dir / "evaluation_results" / agent_mode / "screenshots"
            if screenshots_dir.exists():
                view_patterns = ["result_front_view.png", "result_side_view.png", "result_diagonal_view.png"]
                agent_screenshots = []
                for pattern in view_patterns:
                    result_path = screenshots_dir / pattern
                    if result_path.exists():
                        agent_screenshots.append(result_path)

                if len(agent_screenshots) == 3:
                    result_images.extend(agent_screenshots)

        if result_images:
            return result_images

        # Try single image format
        for agent_mode in ["mcp", "pvpython"]:
            result_path = case_dir / "results" / agent_mode / f"{case_name}.png"
            if result_path.exists():
                result_images.append(result_path)

        return result_images

    def get_evaluation_data(self) -> List[Dict[str, Any]]:
        """Get all evaluation data for human judging."""
        evaluation_data = []

        for i, case_config in enumerate(self.cases_config):
            case_name = case_config['name']
            case_path = case_config['path']
            yaml_path = case_config['yaml']
            description = case_config.get('description', '')

            # Get case directory
            case_dir = self.workspace_root / case_path

            if not case_dir.exists():
                print(f"Warning: Case directory not found: {case_path}")
                continue

            # Detect benchmark type
            benchmark_type = self._detect_benchmark_type(case_path)

            # Get vision metrics from the YAML
            metrics = self._get_vision_metrics_for_case(case_name, yaml_path)

            # Skip cases without vision metrics
            if not metrics:
                print(f"Warning: No vision metrics found for case: {case_name}")
                continue

            # Get task description
            task_description = self._get_task_description_for_case(case_name, yaml_path)

            # Get ground truth images
            gt_images = self._get_ground_truth_images(case_dir, case_name, benchmark_type)

            # Get result images
            result_images = self._get_result_images(case_dir, case_name, benchmark_type, gt_images)

            evaluation_data.append({
                "case_index": i,
                "case_name": case_name,
                "case_path": case_path,
                "task_description": task_description,
                "description": description,
                "benchmark_type": benchmark_type,
                "metrics": metrics,
                "ground_truth_images": [str(img.relative_to(self.workspace_root)) for img in gt_images],
                "result_images": [str(img.relative_to(self.workspace_root)) for img in result_images],
                "has_results": len(result_images) > 0
            })

        return evaluation_data

    def get_metadata(self) -> Dict[str, Any]:
        """Get evaluation session metadata."""
        return {
            "mode": "multi_case",
            "cases_yaml": str(self.cases_yaml_path),
            "total_cases": len(self.cases_config),
            "unique_yamls": list(self.yaml_test_cases.keys())
        }
