"""
Unified Test Runner

A high-level test runner that works with any agent implementing BaseAgent.
This preserves all functionality from the existing yaml_runner files while
providing a simpler interface.
"""

import asyncio
import json
import os
import sys
import time
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_agent import BaseAgent, AgentResult
from .evaluation_manager import EvaluationManager
from .rate_limiter import create_rate_limiter_from_config, RateLimiter


class YAMLTestCase:
    """
    Represents a single test case loaded from YAML.

    This is compatible with the existing YAML format used throughout the benchmark.
    """

    def __init__(self, yaml_data: Dict, case_name: str, cases_dir: str, data_dir: Optional[str] = None, yaml_filename: Optional[str] = None):
        self.yaml_data = yaml_data
        self.case_name = case_name
        self.cases_dir = Path(cases_dir)
        self.data_dir = Path(data_dir) if data_dir else self.cases_dir
        self.yaml_filename = yaml_filename  # YAML filename without extension

        # Extract question and rubric from YAML
        self.task_description = yaml_data.get('vars', {}).get('question', '').strip()

        # Extract all LLM rubrics from assert section
        assert_list = yaml_data.get('assert', [])
        self.rubrics = {}  # Dictionary mapping subtype to rubric value
        self.evaluation_subtypes = []  # List of all subtypes
        self.assertions = []  # Store all assertions for assertion-based evaluation
        self.is_assertion_based = False  # Flag to indicate if this is assertion-based
        self.rule_based_assertions = []  # Store rule_based assertions
        self.is_rule_based = False  # Flag for rule_based evaluation
        self.file_configs = {}  # Dictionary mapping subtype to file paths (gs_file, rs_file)

        # Check if this is assertion-based evaluation
        # (contains-all, not-contains, etc. instead of just llm-rubric)
        has_non_llm_assertions = False
        for assert_item in assert_list:
            assert_type = assert_item.get('type', '')
            if assert_type in ['contains-all', 'not-contains']:
                has_non_llm_assertions = True
                self.assertions.append(assert_item)
            elif assert_type == 'llm-rubric':
                self.assertions.append(assert_item)
                subtype = assert_item.get('subtype', 'vision')
                rubric_value = assert_item.get('value', '').strip()
                if rubric_value:
                    self.rubrics[subtype] = rubric_value
                    if subtype not in self.evaluation_subtypes:
                        self.evaluation_subtypes.append(subtype)

                    # Extract file paths for custom evaluation
                    file_config = {}
                    if 'gs_file' in assert_item or 'gs-file' in assert_item:
                        file_config['gs_file'] = assert_item.get('gs_file') or assert_item.get('gs-file')
                    if 'rs_file' in assert_item or 'rs-file' in assert_item:
                        file_config['rs_file'] = assert_item.get('rs_file') or assert_item.get('rs-file')
                    if file_config:
                        self.file_configs[subtype] = file_config
            elif assert_type == 'code-similarity':
                self.assertions.append(assert_item)
                subtype = assert_item.get('subtype', 'code')
                # Store the entire config (gs_file, rs_file) as the rubric
                code_config = {
                    'gs_file': assert_item.get('gs_file', []),
                    'rs_file': assert_item.get('rs_file', [])
                }
                self.rubrics[subtype] = code_config
                if subtype not in self.evaluation_subtypes:
                    self.evaluation_subtypes.append(subtype)
            elif assert_type == 'rule_based':
                self.rule_based_assertions.append(assert_item)

        # If we have non-LLM assertions, mark as assertion-based
        if has_non_llm_assertions:
            self.is_assertion_based = True

        # If we have rule_based assertions, mark as rule-based
        if self.rule_based_assertions:
            self.is_rule_based = True

        # Backward compatibility
        self.llm_rubric = self.rubrics.get('vision', '')
        self.evaluation_subtype = 'vision' if 'vision' in self.rubrics else (
            self.evaluation_subtypes[0] if self.evaluation_subtypes else 'vision'
        )

        # Set up paths
        # For assertion-based cases with yaml_filename, use: cases_dir / yaml_filename / case_name
        # For bioimage_data (napari) cases, also use yaml_filename subdirectory structure
        # For other cases, use: cases_dir / case_name
        cases_path_str = str(self.cases_dir)
        use_yaml_subdir = (self.is_assertion_based and self.yaml_filename) or \
                         ('bioimage_data' in cases_path_str and self.yaml_filename)

        if use_yaml_subdir:
            self.case_path = self.cases_dir / self.yaml_filename / case_name
        else:
            self.case_path = self.cases_dir / case_name

    def is_valid(self) -> bool:
        """Check if this is a valid test case."""
        has_task = bool(self.task_description)

        # Valid if it has either rubrics, assertions, or rule_based assertions
        if self.is_rule_based:
            return has_task and bool(self.rule_based_assertions)
        elif self.is_assertion_based:
            has_assertions = bool(self.assertions)
            return has_task and has_assertions
        else:
            has_rubrics = bool(self.rubrics)
            return has_task and has_rubrics

    def get_task_description(self, agent_mode: str) -> str:
        """Get the task description with working directory information."""
        if not self.task_description:
            raise ValueError(f"No task description found for case {self.case_name}")

        # Convert to absolute path
        working_dir = Path(self.data_dir).resolve()

        # Prepend working directory information with absolute path
        working_dir_info = f'Your agent_mode is "{agent_mode}", use it when saving results. Your working directory is "{working_dir}", and you should have access to it. In the following prompts, we will use relative path with respect to your working path. But remember, when you load or save any file, always stick to absolute path.'

        return f"{working_dir_info}\n\n{self.task_description}"

    def get_task_config(self, agent_mode: str) -> Dict[str, Any]:
        """Get task configuration for the agent."""
        return {
            "case_name": self.case_name,
            "case_dir": str(self.case_path),
            "data_dir": str(self.data_dir),
            "working_dir": str(self.data_dir),
            "eval_mode": agent_mode,
        }


class UnifiedTestRunner:
    """
    Unified test runner that works with any agent implementing BaseAgent.

    This runner preserves all features from the existing yaml_runner files:
    - YAML test case loading
    - Token counting and cost calculation
    - Comprehensive evaluation with LLM judge
    - Image metrics (PSNR, SSIM, LPIPS)
    - Support for multiple evaluation subtypes (vision, text)
    - Result saving in standard format
    """

    def __init__(
        self,
        agent: BaseAgent,
        yaml_path: str,
        cases_dir: str,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        eval_model: str = "gpt-4o",
        static_screenshot: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the unified test runner.

        Args:
            agent: Agent instance implementing BaseAgent
            yaml_path: Path to YAML test cases file
            cases_dir: Directory containing test case folders
            data_dir: Optional separate data directory (defaults to cases_dir)
            output_dir: Optional output directory for centralized results
            openai_api_key: OpenAI API key for evaluation
            eval_model: Model to use for LLM evaluation
            static_screenshot: If True, use pre-generated screenshots for evaluation
            config: Optional config dictionary for rate limiting (passed from agent)
        """
        self.agent = agent
        self.yaml_path = Path(yaml_path)
        self.cases_dir = Path(cases_dir)
        self.data_dir = Path(data_dir) if data_dir else self.cases_dir

        # Save centralized results to test_results/<benchmark_name>/<agent_name>/
        # e.g., test_results/main/paraview_mcp/, test_results/bioimage_data/napari_mcp/, etc.
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Get benchmark name from cases_dir
            # For paths like "SciVisAgentBench-tasks/bioimage_data/0_actions", extract "bioimage_data"
            # For paths like "SciVisAgentBench-tasks/main", extract "main"
            cases_dir_parts = self.cases_dir.parts
            if 'SciVisAgentBench-tasks' in cases_dir_parts:
                # Find index of SciVisAgentBench-tasks and get the next part
                idx = cases_dir_parts.index('SciVisAgentBench-tasks')
                if idx + 1 < len(cases_dir_parts):
                    benchmark_name = cases_dir_parts[idx + 1]
                else:
                    benchmark_name = self.cases_dir.name
            else:
                benchmark_name = self.cases_dir.name

            # Get agent name from agent config
            agent_name = agent.agent_name
            # Save to repository root / test_results / benchmark_name / agent_name
            repo_root = Path.cwd()  # Assumes running from repo root
            self.output_dir = repo_root / "test_results" / benchmark_name / agent_name

        # Store YAML filename for later use (without extension)
        self.yaml_filename = self.yaml_path.stem

        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.eval_model = eval_model

        # Initialize rate limiter from config if available
        self.rate_limiter: Optional[RateLimiter] = None
        if config:
            self.rate_limiter = create_rate_limiter_from_config(config)
            if self.rate_limiter:
                print("✓ Rate limiting enabled based on config")
        self.static_screenshot = static_screenshot
        self.test_cases: List[YAMLTestCase] = []

        # Initialize evaluation manager
        self.evaluation_manager = EvaluationManager(
            eval_mode=agent.eval_mode,
            openai_api_key=self.openai_api_key,
            eval_model=eval_model,
            static_screenshot=static_screenshot
        )

        # Initialize token counter
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is None:
            return len(text.split())
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text.split())

    def load_yaml_test_cases(self) -> List[YAMLTestCase]:
        """Load test cases from YAML file."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")

        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        test_cases = []

        # Extract YAML filename without extension for assertion-based cases
        yaml_filename = self.yaml_path.stem  # Get filename without extension

        if isinstance(yaml_data, list):
            for i, case_data in enumerate(yaml_data):
                case_name = self._extract_case_name(case_data, i)
                if case_name:
                    test_case = YAMLTestCase(
                        case_data,
                        case_name,
                        str(self.cases_dir),
                        str(self.data_dir),
                        yaml_filename=yaml_filename  # Pass YAML filename
                    )
                    if test_case.is_valid():
                        test_cases.append(test_case)
                        print(f"✓ Loaded test case: {case_name}")
                    else:
                        print(f"⚠️  Skipping invalid test case: {case_name}")
        else:
            raise ValueError("YAML file should contain a list of test cases")

        self.test_cases = test_cases
        return test_cases

    def _extract_case_name(self, case_data: Dict, index: int) -> Optional[str]:
        """Extract case name from YAML test case data."""
        # Check if this is molecular_vis or bioimage_data benchmark
        # For these benchmarks, use simple case numbering
        cases_path_str = str(self.cases_dir)
        if ('molecular_vis' in cases_path_str and 'workflows' not in cases_path_str) or 'bioimage_data' in cases_path_str:
            return f"operation_{index + 1}"

        task_description = case_data.get('vars', {}).get('question', '')

        import re
        # Case 1: anonymized dataset_X
        dataset_match = re.search(r'dataset_(\d+)', task_description)
        if dataset_match:
            return f"dataset_{dataset_match.group(1)}"

        # Case 2: path like aneurism/data/aneurism_256x256x256_uint8.raw
        path_pattern_extended = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)/data/\1_[^"]*', task_description)
        if path_pattern_extended:
            return path_pattern_extended.group(1)

        # Case 3: path like line-plot/data/line-plot.ex2 (allow dash in name)
        path_with_dash = re.search(r'([a-zA-Z0-9_-]+)/data/\1[^"]*', task_description)
        if path_with_dash:
            return path_with_dash.group(1)

        # Case 4: path like points-surf-clip/results (allow dash in name)
        results_path = re.search(r'([a-zA-Z0-9_-]+)/results', task_description)
        if results_path:
            return results_path.group(1)

        # Fallback
        return f"case_{index + 1}"

    async def run_single_test_case(self, test_case: YAMLTestCase, save_result: bool = True) -> Dict:
        """Run a single test case and return results."""
        print(f"\n{'='*60}")
        print(f"Running test case: {test_case.case_name}")
        print(f"{'='*60}")

        start_time = datetime.now()

        # Get task description and config
        task_description = test_case.get_task_description(self.agent.eval_mode)
        task_config = test_case.get_task_config(self.agent.eval_mode)

        result = {
            "case_name": test_case.case_name,
            "status": "running",
            "start_time": start_time.isoformat(),
            "task_description": task_description,
            "llm_rubric": test_case.llm_rubric,
            "evaluation_subtype": test_case.evaluation_subtype,
            "response": "",
            "error": None,
            "token_usage": {
                "input_tokens": 0,  # Will be updated after task runs
                "output_tokens": 0,
                "total_tokens": 0
            }
        }

        try:
            # Prepare for task
            await self.agent.prepare_task(task_config)

            # Run the task
            print(f"Starting execution for case: {test_case.case_name}")
            print(f"Task preview: {task_description[:200]}...")

            agent_result: AgentResult = await self.agent.run_task(task_description, task_config)

            # Get token usage from agent's _token_info if available (from API or comprehensive estimation)
            # Otherwise fall back to simple counting
            if "_token_info" in agent_result.metadata:
                token_info = agent_result.metadata["_token_info"]
                input_tokens = token_info.get("input_tokens", 0)
                output_tokens = token_info.get("output_tokens", 0)
                token_source = token_info.get("source", "unknown")
            else:
                # Fallback: simple token counting (legacy behavior)
                input_tokens = self.count_tokens(task_description)
                output_tokens = self.count_tokens(agent_result.response)
                token_source = "simple_estimate"

            # Update result
            result.update({
                "status": "completed" if agent_result.success else "failed",
                "response": agent_result.response,
                "error": agent_result.error,
                "output_files": agent_result.output_files,
                "metadata": agent_result.metadata,
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            })

            # Save test result using agent's method
            self.agent.save_test_result(
                agent_result,
                str(test_case.case_path),
                test_case.case_name,
                task_description
            )

            # Cleanup task
            await self.agent.cleanup_task(task_config)

            print(f"\n✅ Test case {test_case.case_name} completed successfully")

        except Exception as e:
            result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ Test case {test_case.case_name} failed: {str(e)}")

        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration_seconds"] = (end_time - start_time).total_seconds()

        # Save centralized result if requested
        if save_result:
            await self.save_centralized_result(test_case, result)

        return result

    def load_latest_result(self, test_case: YAMLTestCase) -> Optional[Dict]:
        """
        Load the latest test result for a given case.

        Returns None if no result file is found.
        """
        if not self.output_dir.exists():
            return None

        # Find all result files for this case
        result_files = list(self.output_dir.glob(f"{test_case.case_name}_result_*.json"))

        if not result_files:
            return None

        # Sort by timestamp (extracted from filename) and get the latest
        def get_timestamp(filepath):
            stem = filepath.stem  # e.g., "chart-opacity_result_1770506163"
            parts = stem.split('_')
            try:
                return int(parts[-1])
            except (ValueError, IndexError):
                return 0

        latest_file = max(result_files, key=get_timestamp)

        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading previous result from {latest_file}: {e}")
            return None

    async def save_centralized_result(self, test_case: YAMLTestCase, result: Dict):
        """Save test result to centralized output directory."""
        # For napari_mcp agent, add yaml filename subdirectory
        if self.agent.agent_name == "napari_mcp" or self.agent.agent_name == "gmx_vmd_mcp":
            output_dir = self.output_dir / self.yaml_filename
        else:
            output_dir = self.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        centralized_file = output_dir / f"{test_case.case_name}_result_{int(time.time())}.json"

        # Add model metadata from agent config
        if hasattr(self.agent, 'config') and self.agent.config:
            model_metadata = {
                "provider": self.agent.config.get("provider"),
                "model": self.agent.config.get("model"),
                "base_url": self.agent.config.get("base_url"),
                "price": self.agent.config.get("price")
            }
            # Remove None values
            model_metadata = {k: v for k, v in model_metadata.items() if v is not None}
            if model_metadata:
                result["model_metadata"] = model_metadata

        with open(centralized_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"💾 Centralized result saved to: {centralized_file}")

    async def run_evaluation(self, test_case: YAMLTestCase) -> Dict:
        """Run evaluation for a test case."""
        # Check if this is rule-based evaluation (topology-style eval scripts)
        if test_case.is_rule_based:
            from .topology_evaluator import evaluate_rule_based_assertions

            result = evaluate_rule_based_assertions(
                assertions=test_case.rule_based_assertions,
                data_dir=str(self.data_dir),
                agent_mode=self.agent.eval_mode,
            )
            return result

        # Check if this is assertion-based evaluation
        elif test_case.is_assertion_based:
            # Use AssertionEvaluator
            sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation_helpers"))
            from assertion_evaluator import AssertionEvaluator

            evaluator = AssertionEvaluator(
                case_dir=str(test_case.case_path),
                case_name=test_case.case_name,
                eval_mode=self.agent.eval_mode,
                openai_api_key=self.openai_api_key,
                eval_model=self.eval_model
            )

            return await evaluator.evaluate_assertions(test_case.assertions)
        else:
            # Use standard rubric-based evaluation
            return await self.evaluation_manager.run_evaluation(
                case_dir=str(test_case.case_path),
                case_name=test_case.case_name,
                evaluation_subtypes=test_case.evaluation_subtypes,
                rubrics=test_case.rubrics,
                file_configs=test_case.file_configs,
                data_dir=str(test_case.data_dir)
            )

    async def run_all_test_cases(self, run_evaluation: bool = True) -> Dict:
        """Run all test cases and optionally evaluate them."""
        if not self.test_cases:
            self.load_yaml_test_cases()

        if not self.test_cases:
            return {"error": "No valid test cases found"}

        print(f"\n🚀 Running {len(self.test_cases)} test cases")
        print(f"Agent: {self.agent}")
        print(f"YAML file: {self.yaml_path}")
        print(f"Cases directory: {self.cases_dir}")
        print(f"Output directory: {self.output_dir}")

        # Setup agent
        await self.agent.setup()

        start_time = datetime.now()
        results = []

        try:
            for test_case in self.test_cases:
                # Wait if needed to comply with rate limits (before starting the case)
                if self.rate_limiter:
                    await self.rate_limiter.wait_if_needed(
                        estimated_input_tokens=2000,  # Conservative estimate
                        estimated_output_tokens=20000
                    )

                # Run test case without saving (we'll save after evaluation)
                result = await self.run_single_test_case(test_case, save_result=False)
                results.append(result)

                # Record the actual token usage after the request completes
                if self.rate_limiter and result.get("token_usage"):
                    token_usage = result["token_usage"]
                    await self.rate_limiter.record_request(
                        input_tokens=token_usage.get("input_tokens", 0),
                        output_tokens=token_usage.get("output_tokens", 0)
                    )

                # Run evaluation if requested and test completed successfully
                if run_evaluation and result.get("status") == "completed":
                    eval_result = await self.run_evaluation(test_case)
                    result["evaluation"] = eval_result

                # Save centralized result with evaluation data
                await self.save_centralized_result(test_case, result)

        finally:
            # Teardown agent
            await self.agent.teardown()

        end_time = datetime.now()

        # Calculate summary statistics
        total_cases = len(results)
        successful_cases = len([r for r in results if r.get("status") == "completed"])
        failed_cases = total_cases - successful_cases

        total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in results)

        # Calculate overall evaluation scores
        total_eval_score = 0
        total_eval_max_score = 0
        evaluated_cases = 0
        assertion_passed_cases = 0  # For assertion-based evaluations

        for result in results:
            if "evaluation" in result and result["evaluation"].get("status") in ("completed", "partial_error"):
                eval_data = result["evaluation"]

                # Check if this is rule-based evaluation (topology eval scripts)
                if eval_data.get("eval_type") == "rule_based":
                    score = eval_data.get("score", 0)
                    max_score = eval_data.get("max_score", 10)
                    total_eval_score += score
                    total_eval_max_score += max_score
                    evaluated_cases += 1
                # Check if this is assertion-based evaluation
                elif "assertion_results" in eval_data:
                    # Assertion-based: use the top-level score field (1 for pass, 0 for fail)
                    score = eval_data.get("score", 0)
                    total_eval_score += score
                    total_eval_max_score += 1  # Each assertion-based case is worth 1 point
                    evaluated_cases += 1
                    if score == 1:
                        assertion_passed_cases += 1
                else:
                    # Score-based: use scores.total_score and max_possible_score
                    eval_scores = eval_data.get("scores", {})
                    total_eval_score += eval_scores.get("total_score", 0)
                    total_eval_max_score += eval_scores.get("max_possible_score", 0)
                    evaluated_cases += 1

        summary = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "agent": str(self.agent),
            "eval_mode": self.agent.eval_mode,
            "total_cases": total_cases,
            "successful_cases": successful_cases,
            "failed_cases": failed_cases,
            "success_rate": successful_cases / total_cases if total_cases > 0 else 0,
            "total_tokens": total_tokens,
            "evaluation_model": self.eval_model,
            # Overall evaluation scores
            "overall_evaluation": {
                "evaluated_cases": evaluated_cases,
                "total_score": total_eval_score,
                "max_possible_score": total_eval_max_score,
                "percentage": (total_eval_score / total_eval_max_score * 100) if total_eval_max_score > 0 else 0,
                "average_score_per_case": total_eval_score / evaluated_cases if evaluated_cases > 0 else 0,
                "average_max_per_case": total_eval_max_score / evaluated_cases if evaluated_cases > 0 else 0
            },
            "results": results
        }

        # Add model metadata from agent config
        if hasattr(self.agent, 'config') and self.agent.config:
            model_metadata = {
                "provider": self.agent.config.get("provider"),
                "model": self.agent.config.get("model"),
                "base_url": self.agent.config.get("base_url"),
                "price": self.agent.config.get("price")
            }
            # Remove None values
            model_metadata = {k: v for k, v in model_metadata.items() if v is not None}
            if model_metadata:
                summary["model_metadata"] = model_metadata

        # Save summary
        summary_file = self.output_dir / f"test_summary_{int(time.time())}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"🎯 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Agent: {self.agent}")
        print(f"Total cases: {total_cases}")
        print(f"Successful: {successful_cases}")
        print(f"Failed: {failed_cases}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")

        # Print overall evaluation scores
        if evaluated_cases > 0:
            overall_eval = summary['overall_evaluation']
            print(f"\n📊 OVERALL EVALUATION SCORES")
            print(f"Evaluated cases: {evaluated_cases}/{total_cases}")
            print(f"Total score: {overall_eval['total_score']:.1f}/{overall_eval['max_possible_score']:.1f} ({overall_eval['percentage']:.1f}%)")
            print(f"Average per case: {overall_eval['average_score_per_case']:.1f}/{overall_eval['average_max_per_case']:.1f}")

        print(f"\nResults saved: {summary_file}")

        return summary
