"""
Evaluation data loader for human judge UI.
Loads test cases, results, and ground truth images for human evaluation.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.evaluation_helpers.screenshot_helper import take_screenshots_from_state


class EvaluationLoader:
    """Load evaluation data for human judging."""

    def __init__(self, yaml_path: str, cases_dir: str, agent_name: str, config_path: str, filter_cases_path: str = None):
        """
        Initialize evaluation loader.

        Args:
            yaml_path: Path to YAML file with test cases
            cases_dir: Directory containing test cases (e.g., SciVisAgentBench-tasks/main)
            agent_name: Name of the agent being evaluated
            config_path: Path to agent config file
            filter_cases_path: Path to YAML file with case names to filter (optional)
        """
        self.yaml_path = Path(yaml_path).resolve()
        self.cases_dir = Path(cases_dir).resolve()
        self.agent_name = agent_name
        self.config_path = Path(config_path).resolve()
        self.filter_cases_path = Path(filter_cases_path).resolve() if filter_cases_path else None

        # Workspace root is two levels up from cases_dir
        self.workspace_root = self.cases_dir.parent.parent

        # Determine benchmark type from cases_dir
        self.benchmark_type = self._detect_benchmark_type()

        # Determine agent_mode from config
        self.agent_mode = self._detect_agent_mode()

        # Load filter case names if provided
        self.filter_case_names = self._load_filter_cases() if self.filter_cases_path else None

        # Load test cases
        self.test_cases = self._load_test_cases()

    def _detect_benchmark_type(self) -> str:
        """Detect benchmark type from cases_dir path."""
        cases_dir_name = self.cases_dir.name
        if cases_dir_name == "chatvis_bench":
            return "chatvis_bench"
        elif cases_dir_name == "main":
            return "main"
        else:
            raise ValueError(f"Unsupported benchmark type: {cases_dir_name}. Only 'main' and 'chatvis_bench' are supported.")

    def _detect_agent_mode(self) -> str:
        """Detect agent mode from config file and agent name."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Check for eval_mode in config
            if "eval_mode" in config:
                return config["eval_mode"]

            # Infer from agent name
            # ChatVis uses pvpython mode
            if self.agent_name == "chatvis":
                return "pvpython"

            # Default to "mcp" for MCP-based agents (paraview_mcp, napari_mcp, etc.)
            return "mcp"
        except Exception as e:
            print(f"Warning: Could not detect agent_mode from config: {e}")
            # Fallback based on agent name
            if self.agent_name == "chatvis":
                return "pvpython"
            return "mcp"

    def _load_filter_cases(self) -> List[str]:
        """Load filter case names from YAML file."""
        with open(self.filter_cases_path, 'r') as f:
            filter_data = yaml.safe_load(f)

        if isinstance(filter_data, dict) and 'cases' in filter_data:
            # YAML format: cases: [case1, case2, ...]
            case_names = filter_data['cases']
        elif isinstance(filter_data, list):
            # YAML format: direct list [case1, case2, ...]
            case_names = filter_data
        else:
            raise ValueError("Filter cases YAML must contain a 'cases' key with a list or be a direct list")

        if not isinstance(case_names, list):
            raise ValueError("Filter cases must be a list of case names")

        return case_names

    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from YAML file."""
        with open(self.yaml_path, 'r') as f:
            cases = yaml.safe_load(f)

        if not isinstance(cases, list):
            raise ValueError("YAML file must contain a list of test cases")

        return cases

    def get_vision_metrics(self, case: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract vision-based metrics from a test case.

        Args:
            case: Test case dictionary

        Returns:
            List of vision metrics, each with 'criterion' text
        """
        metrics = []

        if "assert" not in case:
            return metrics

        for assertion in case["assert"]:
            if assertion.get("type") == "llm-rubric" and assertion.get("subtype") == "vision":
                # Parse the multi-line criteria
                criteria_text = assertion.get("value", "")

                # Split by numbered criteria
                lines = criteria_text.strip().split('\n')
                current_criterion = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check if line starts with a number (new criterion)
                    if line and line[0].isdigit() and '.' in line[:3]:
                        # Save previous criterion if exists
                        if current_criterion:
                            metrics.append({
                                "criterion": ' '.join(current_criterion).strip()
                            })
                            current_criterion = []

                        # Start new criterion (remove number prefix)
                        criterion_text = line.split('.', 1)[1].strip() if '.' in line else line
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

    def _get_case_name(self, case_index: int) -> str:
        """Get case name from YAML by extracting dataset name from save path."""
        case = self.test_cases[case_index]
        question = case.get("vars", {}).get("question", "")

        # Try to extract dataset name from save path in question
        # Look for patterns like "bonsai/results/{agent_mode}/bonsai.pvsm"
        # or "ml-dvr/results/mcp/ml-dvr.png"
        for line in question.split('\n'):
            line = line.strip()
            if ('save' in line.lower() or 'Save' in line) and '/' in line:
                # Extract path from quotes if present
                import re
                # Match quoted paths
                quoted_match = re.search(r'["\']([^"\']+/results/[^"\']+)["\']', line)
                if quoted_match:
                    path = quoted_match.group(1)
                else:
                    # Try to find path pattern without quotes
                    # Look for pattern: word/results/
                    match = re.search(r'(\w[\w-]*)/results/', line)
                    if match:
                        return match.group(1)
                    continue

                # Extract dataset name (first part before /results/)
                parts = path.split('/')
                for i, part in enumerate(parts):
                    if part == 'results' and i > 0:
                        return parts[i-1]

        # Fallback to index-based naming
        return f"case_{case_index + 1}"

    def _ensure_result_screenshots(self, case_name: str, case_dir: Path, data_dir: Path, gs_images: List[Path]) -> List[Path]:
        """
        Load result screenshots from appropriate directories.

        Args:
            case_name: Name of the case
            case_dir: Case directory path
            data_dir: Directory containing input data
            gs_images: List of ground truth images to determine expected result format

        Returns:
            List of screenshot paths
        """
        if self.benchmark_type == "chatvis_bench":
            # ChatVis: single PNG result image in results/pvpython/
            result_dir = case_dir / "results" / self.agent_mode
            if result_dir.exists():
                png_files = list(result_dir.glob("*.png"))
                return [png_files[0]] if png_files else []
            return []

        # Determine expected format based on ground truth images
        if gs_images:
            gs_names = [gs.name for gs in gs_images]

            # Check if GS has single image format (e.g., case_name_gs.png)
            if any(f"{case_name}_gs.png" in name for name in gs_names):
                # Single image format: check both results/mcp/ and results/pvpython/
                result_images = []
                for agent_mode in ["mcp", "pvpython"]:
                    result_path = case_dir / "results" / agent_mode / f"{case_name}.png"
                    if result_path.exists():
                        result_images.append(result_path)
                return result_images

            # Check if GS has 3-view format (gs_front_view.png, etc.)
            if any("gs_front_view.png" in name for name in gs_names):
                # 3-view format: look in evaluation_results/{agent_mode}/screenshots/
                screenshots_dir = case_dir / "evaluation_results" / self.agent_mode / "screenshots"

                if not screenshots_dir.exists():
                    return []

                view_patterns = ["result_front_view.png", "result_side_view.png", "result_diagonal_view.png"]
                result_screenshots = []
                for pattern in view_patterns:
                    result_path = screenshots_dir / pattern
                    if result_path.exists():
                        result_screenshots.append(result_path)
                return result_screenshots

        # Fallback: try to find any result images
        # First try 3-view format in evaluation_results
        screenshots_dir = case_dir / "evaluation_results" / self.agent_mode / "screenshots"
        if screenshots_dir.exists():
            view_patterns = ["result_front_view.png", "result_side_view.png", "result_diagonal_view.png"]
            result_screenshots = []
            for pattern in view_patterns:
                result_path = screenshots_dir / pattern
                if result_path.exists():
                    result_screenshots.append(result_path)

            if result_screenshots:
                return result_screenshots

        # Try single image format in both results/mcp/ and results/pvpython/
        result_images = []
        for agent_mode in ["mcp", "pvpython"]:
            result_path = case_dir / "results" / agent_mode / f"{case_name}.png"
            if result_path.exists():
                result_images.append(result_path)

        return result_images

    def _get_ground_truth_images(self, case_name: str) -> List[Path]:
        """
        Get ground truth images for a case.

        Args:
            case_name: Name of the test case

        Returns:
            List of ground truth image paths
        """
        gs_dir = self.cases_dir / case_name / "GS"

        if not gs_dir.exists():
            return []

        if self.benchmark_type == "chatvis_bench":
            # ChatVis: single ground truth PNG image
            gs_images = list(gs_dir.glob("*_gs.png"))
            return gs_images[:1] if gs_images else []

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

        # If no images found, try to generate from .pvsm file
        pvsm_files = list(gs_dir.glob("*_gs.pvsm"))
        if pvsm_files:
            print(f"Generating ground truth screenshots from {pvsm_files[0].name}")
            try:
                data_dir = self.cases_dir / case_name / "data"

                screenshots = take_screenshots_from_state(
                    str(pvsm_files[0]),
                    str(gs_dir),
                    prefix="gs_",
                    data_directory=str(data_dir) if data_dir.exists() else None
                )
                return [Path(s) for s in screenshots]
            except Exception as e:
                print(f"Error generating ground truth screenshots: {e}")
                return []

        return gs_images

    def get_evaluation_data(self) -> List[Dict[str, Any]]:
        """
        Get all evaluation data for human judging.

        Returns:
            List of evaluation cases with all necessary data
        """
        evaluation_data = []

        for i, case in enumerate(self.test_cases):
            # Get vision metrics
            metrics = self.get_vision_metrics(case)

            # Skip cases without vision metrics
            if not metrics:
                continue

            # Get case name
            case_name = self._get_case_name(i)

            # Filter by case name if filter is provided
            if self.filter_case_names is not None:
                if case_name not in self.filter_case_names:
                    continue

            # Get task description
            task_description = case.get("vars", {}).get("question", "")

            # Get paths
            case_dir = self.cases_dir / case_name
            data_dir = case_dir / "data"

            # Get ground truth images
            gt_images = self._get_ground_truth_images(case_name)

            # Get result screenshots (pass gs_images to determine expected format)
            result_images = self._ensure_result_screenshots(case_name, case_dir, data_dir, gt_images)

            evaluation_data.append({
                "case_index": i,
                "case_name": case_name,
                "task_description": task_description,
                "metrics": metrics,
                "ground_truth_images": [str(img.relative_to(self.workspace_root)) for img in gt_images],
                "result_images": [str(img.relative_to(self.workspace_root)) for img in result_images],
                "has_results": len(result_images) > 0
            })

        return evaluation_data

    def _get_agent_config_info(self) -> Dict[str, Any]:
        """Extract relevant configuration information from agent config."""
        config_info = {}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Extract model information
            if "model" in config:
                config_info["model"] = config["model"]

            # Extract provider information
            if "provider" in config:
                config_info["provider"] = config["provider"]

            # Extract base_url if present (for custom endpoints)
            if "base_url" in config:
                config_info["base_url"] = config["base_url"]

            # Extract pricing information if present
            if "price" in config:
                config_info["price"] = config["price"]

        except Exception as e:
            print(f"Warning: Could not extract config info: {e}")

        return config_info

    def get_metadata(self) -> Dict[str, Any]:
        """Get evaluation session metadata including agent configuration."""
        agent_config = self._get_agent_config_info()

        metadata = {
            "agent_name": self.agent_name,
            "agent_mode": self.agent_mode,
            "benchmark_type": self.benchmark_type,
            "yaml_file": str(self.yaml_path),
            "cases_dir": str(self.cases_dir),
            "config_file": str(self.config_path),
            "total_cases": len(self.test_cases)
        }

        # Add agent configuration info
        if agent_config:
            metadata["agent_config"] = agent_config

        return metadata
