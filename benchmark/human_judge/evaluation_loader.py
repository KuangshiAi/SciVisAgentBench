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

    def __init__(self, yaml_path: str, cases_dir: str, agent_name: str, config_path: str):
        """
        Initialize evaluation loader.

        Args:
            yaml_path: Path to YAML file with test cases
            cases_dir: Directory containing test cases (e.g., SciVisAgentBench-tasks/main)
            agent_name: Name of the agent being evaluated
            config_path: Path to agent config file
        """
        self.yaml_path = Path(yaml_path).resolve()
        self.cases_dir = Path(cases_dir).resolve()
        self.agent_name = agent_name
        self.config_path = Path(config_path).resolve()

        # Workspace root is two levels up from cases_dir
        self.workspace_root = self.cases_dir.parent.parent

        # Determine benchmark type from cases_dir
        self.benchmark_type = self._detect_benchmark_type()

        # Determine agent_mode from config
        self.agent_mode = self._detect_agent_mode()

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

    def _ensure_result_screenshots(self, result_dir: Path, data_dir: Path) -> List[Path]:
        """
        Ensure result screenshots exist, generate if needed for main benchmark.

        Args:
            result_dir: Directory containing agent results
            data_dir: Directory containing input data

        Returns:
            List of screenshot paths
        """
        if self.benchmark_type == "chatvis_bench":
            # ChatVis: single PNG result image
            png_files = list(result_dir.glob("*.png"))
            # Return first PNG found, or empty list (no warning)
            return [png_files[0]] if png_files else []

        # Main benchmark: check for existing screenshots (3 views)
        screenshot_patterns = ["result_front_view.png", "result_side_view.png", "result_diagonal_view.png"]
        existing_screenshots = []

        for pattern in screenshot_patterns:
            screenshot_path = result_dir / pattern
            if screenshot_path.exists():
                existing_screenshots.append(screenshot_path)

        # If we have all 3 screenshots, return them
        if len(existing_screenshots) == 3:
            return existing_screenshots

        # Otherwise, try to generate from .pvsm file
        pvsm_files = list(result_dir.glob("*.pvsm"))
        if pvsm_files:
            print(f"Generating result screenshots from {pvsm_files[0].name}")
            try:
                screenshots = take_screenshots_from_state(
                    str(pvsm_files[0]),
                    str(result_dir),
                    prefix="result_",
                    data_directory=str(data_dir) if data_dir.exists() else None
                )
                return [Path(s) for s in screenshots]
            except Exception as e:
                print(f"Error generating result screenshots: {e}")
                return []

        # Return whatever screenshots we have (could be 0, 1, or 2)
        return existing_screenshots

    def _get_ground_truth_images(self, case_name: str) -> List[Path]:
        """
        Get ground truth images for a case, generating if needed for main benchmark.

        Args:
            case_name: Name of the test case

        Returns:
            List of ground truth image paths
        """
        gs_dir = self.cases_dir / case_name / "GS"

        if not gs_dir.exists():
            # Silently return empty list - no warning needed
            return []

        if self.benchmark_type == "chatvis_bench":
            # ChatVis: single ground truth PNG image
            gs_images = list(gs_dir.glob("*_gs.png"))
            return gs_images[:1] if gs_images else []

        # Main benchmark: three views (front, side, diagonal)
        gs_patterns = ["gs_front_view.png", "gs_side_view.png", "gs_diagonal_view.png"]
        gs_images = []

        for pattern in gs_patterns:
            gs_path = gs_dir / pattern
            if gs_path.exists():
                gs_images.append(gs_path)

        # If we have all 3 screenshots, return them
        if len(gs_images) == 3:
            return gs_images

        # Otherwise, try to generate from ground truth .pvsm file
        pvsm_files = list(gs_dir.glob("*_gs.pvsm"))
        if pvsm_files:
            print(f"Generating ground truth screenshots from {pvsm_files[0].name}")
            try:
                # Get data directory for this case
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

            # Get task description
            task_description = case.get("vars", {}).get("question", "")

            # Get paths
            case_dir = self.cases_dir / case_name
            result_dir = case_dir / "results" / self.agent_mode
            data_dir = case_dir / "data"

            # Get ground truth images
            gt_images = self._get_ground_truth_images(case_name)

            # Get or generate result screenshots
            result_images = []
            if result_dir.exists():
                result_images = self._ensure_result_screenshots(result_dir, data_dir)

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
