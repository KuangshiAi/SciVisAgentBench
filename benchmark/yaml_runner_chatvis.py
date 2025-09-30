#!/usr/bin/env python3
"""
ChatVis YAML Test Runner for SciVisAgentBench

This script loads test cases from a YAML configuration file and runs them through 
ChatVis (pvpython-based approach), then evaluates the results using LLM-based evaluation.
Combines functionality from chatvis_test_runner.py and pvpython_auto_evaluator.py.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tiktoken
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from openai import OpenAI

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


class TokenCounter:
    """Utility class for counting tokens using GPT-4o tokenizer."""
    
    def __init__(self):
        """Initialize the tokenizer for GPT-4o."""
        try:
            # GPT-4o uses the same tokenizer as GPT-4
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception as e:
            print(f"Warning: Could not initialize tokenizer: {e}")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        if self.tokenizer is None:
            return len(text.split())  # Rough approximation
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"Warning: Could not count tokens: {e}")
            return len(text.split())


class YAMLTestCase:
    """Represents a single test case loaded from YAML."""
    
    def __init__(self, yaml_data: Dict, case_name: str, cases_dir: str):
        self.yaml_data = yaml_data
        self.case_name = case_name
        self.cases_dir = Path(cases_dir)
        
        # Extract question and rubric from YAML
        self.task_description = yaml_data.get('vars', {}).get('question', '').strip()
        
        # Extract all LLM rubrics from assert section
        assert_list = yaml_data.get('assert', [])
        self.rubrics = {}  # Dictionary mapping subtype to rubric value
        self.evaluation_subtypes = []  # List of all subtypes
        
        for assert_item in assert_list:
            if assert_item.get('type') in ['llm-rubric']:
                subtype = assert_item.get('subtype', 'vision')
                rubric_value = assert_item.get('value', '').strip()
                if rubric_value:
                    self.rubrics[subtype] = rubric_value
                    if subtype not in self.evaluation_subtypes:
                        self.evaluation_subtypes.append(subtype)
        
        # Backward compatibility: maintain old properties for vision evaluation
        self.llm_rubric = self.rubrics.get('vision', '')
        self.evaluation_subtype = 'vision' if 'vision' in self.rubrics else (
            self.evaluation_subtypes[0] if self.evaluation_subtypes else 'vision'
        )
        
        # Set up paths
        self.case_path = self.cases_dir / case_name
        self.results_dir = self.case_path / "results" / "pvpython"
        self.evaluation_dir = self.case_path / "evaluation_results" / "pvpython"
        
    def is_valid(self) -> bool:
        """Check if this is a valid test case."""
        has_rubrics = bool(self.rubrics)  # Check if we have any rubrics
        has_task = bool(self.task_description)
        # Note: We don't require case_path.exists() for anonymized datasets
        return has_task and has_rubrics
    
    def get_task_description(self) -> str:
        """Get the task description with working directory information."""
        if not self.task_description:
            raise ValueError(f"No task description found for case {self.case_name}")
        
        # Use current working directory as default
        working_dir = self.cases_dir
        
        # Prepend working directory information
        working_dir_info = f'Your agent_mode is "pvpython", use it when saving results. Your working directory is "{working_dir}", and you should have access to it. In the following prompts, we will use relative path with respect to your working path. But remember, when you load or save any file, always stick to absolute path.'
        
        return f"{working_dir_info}\n\n{self.task_description}"
    
    def get_llm_rubric(self) -> str:
        """Get the LLM evaluation rubric (for backward compatibility - returns vision rubric)."""
        return self.llm_rubric
    
    def get_evaluation_subtype(self) -> str:
        """Get the evaluation subtype (vision/text) - returns primary subtype."""
        return self.evaluation_subtype
    
    def get_rubric_for_subtype(self, subtype: str) -> str:
        """Get the rubric for a specific subtype."""
        return self.rubrics.get(subtype, '')
    
    def get_all_evaluation_subtypes(self) -> List[str]:
        """Get all available evaluation subtypes."""
        return self.evaluation_subtypes.copy()
    
    def has_subtype(self, subtype: str) -> bool:
        """Check if the test case has a specific evaluation subtype."""
        return subtype in self.rubrics
    
    def is_vision_evaluation(self) -> bool:
        """Check if this test case requires vision-based evaluation."""
        return self.evaluation_subtype.lower() == 'vision'
    
    def ensure_directories(self):
        """Ensure results and evaluation directories exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)


class ChatVisYAMLTestRunner:
    """Runs test cases loaded from YAML configuration using ChatVis approach."""
    
    def __init__(self, yaml_path: str, cases_dir: str, output_dir: Optional[str] = None, 
                 config_path: Optional[str] = None, model: str = "gpt-4o", 
                 eval_model: str = "gpt-4o"):
        self.yaml_path = Path(yaml_path)
        self.cases_dir = Path(cases_dir)
        self.output_dir = Path(output_dir) if output_dir else self.cases_dir.parent / "test_results" / "chatvis_yaml"
        self.config_path = config_path
        self.model = model
        self.eval_model = eval_model
        self.test_cases: List[YAMLTestCase] = []
        self.token_counter = TokenCounter()
        
        # Initialize multi-provider client
        sys.path.insert(0, str(current_dir / "chatvis"))
        from multi_provider_client import MultiProviderClient
        
        if config_path and os.path.exists(config_path):
            # Load config first to check provider
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create client with explicit API key from environment
            if config.get('provider') == 'openai':
                self.client = MultiProviderClient(
                    provider="openai",
                    model=config.get('model', model),
                    api_key=os.getenv('OPENAI_API_KEY')
                )
            elif config.get('provider') == 'anthropic':
                self.client = MultiProviderClient(
                    provider="anthropic", 
                    model=config.get('model', model),
                    api_key=os.getenv('ANTHROPIC_API_KEY')
                )
            else:
                # Use config file method for other providers
                self.client = MultiProviderClient.from_config_file(config_path)
                
            self.pricing_info = config.get('price', {
                "input_per_1m_tokens": "$2.50",
                "output_per_1m_tokens": "$10.00"
            })
        else:
            # Fallback to OpenAI
            self.client = MultiProviderClient(
                provider="openai",
                model=model,
                api_key=os.getenv('OPENAI_API_KEY')
            )
            # Default pricing for GPT-4o
            self.pricing_info = {
                "input_per_1m_tokens": "$2.50",
                "output_per_1m_tokens": "$10.00"
            }
        
        # Initialize evaluation client (always OpenAI for now)
        self.eval_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) if os.getenv('OPENAI_API_KEY') else None
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> Optional[Dict]:
        """Calculate cost based on token usage and pricing info."""
        try:
            input_price_str = self.pricing_info["input_per_1m_tokens"].replace("$", "")
            output_price_str = self.pricing_info["output_per_1m_tokens"].replace("$", "")
            
            input_price_per_1m = float(input_price_str)
            output_price_per_1m = float(output_price_str)
            
            # Calculate costs
            input_cost = (input_tokens / 1_000_000) * input_price_per_1m
            output_cost = (output_tokens / 1_000_000) * output_price_per_1m
            total_cost = input_cost + output_cost
            
            return {
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6),
                "currency": "USD",
                "pricing_model": {
                    "input_per_1m_tokens": self.pricing_info["input_per_1m_tokens"],
                    "output_per_1m_tokens": self.pricing_info["output_per_1m_tokens"]
                }
            }
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not calculate cost: {e}")
            return None
    
    def load_yaml_test_cases(self) -> List[YAMLTestCase]:
        """Load test cases from YAML file."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")
        
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        test_cases = []
        
        # Parse YAML data - expect a list of test case dictionaries
        if isinstance(yaml_data, list):
            for i, case_data in enumerate(yaml_data):
                # Try to infer case name from the task description or use index
                case_name = self._extract_case_name(case_data, i)
                if case_name:
                    test_case = YAMLTestCase(case_data, case_name, str(self.cases_dir))
                    if test_case.is_valid():
                        test_cases.append(test_case)
                        print(f"Loaded test case: {case_name}")
                    else:
                        print(f"Skipping invalid test case: {case_name}")
                else:
                    print(f"Skipping test case {i}: could not determine case name")
        else:
            raise ValueError("YAML file should contain a list of test cases")
        
        self.test_cases = test_cases
        return test_cases
    
    def _extract_case_name(self, case_data: Dict, index: int) -> Optional[str]:
        """Extract case name from YAML test case data."""
        # Look for dataset name in the task description
        task_description = case_data.get('vars', {}).get('question', '')
        
        # For anonymized datasets, look for dataset_XXX pattern
        import re
        dataset_match = re.search(r'dataset_(\d+)', task_description)
        if dataset_match:
            return f"dataset_{dataset_match.group(1)}"
        
        # Alternative pattern for cases where the file has additional info after dataset name
        # Like "aneurism/data/aneurism_256x256x256_uint8.raw"
        path_pattern_extended = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)/data/\1_[^"]*', task_description)
        if path_pattern_extended:
            return path_pattern_extended.group(1)
        
        # Fallback to case index
        return f"case_{index + 1}"
    
    def execute_paraview_code(self, code: str) -> str:
        """Execute ParaView Python code directly using pvpython"""
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_script = f.name
            
            # Try to find pvpython in common locations
            pvpython_paths = [
                '/Applications/ParaView-5.13.3.app/Contents/bin/pvpython',  # macOS
                '/usr/local/bin/pvpython',  # Linux/macOS homebrew
                'pvpython',  # In PATH
            ]
            
            pvpython_cmd = None
            for path in pvpython_paths:
                try:
                    result = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        pvpython_cmd = path
                        break
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            if not pvpython_cmd:
                return "Error: pvpython not found. Please install ParaView or add pvpython to your PATH."
            
            # Execute the script
            result = subprocess.run([pvpython_cmd, temp_script], 
                                  capture_output=True, text=True, timeout=120)
            
            # Clean up temp file
            Path(temp_script).unlink()
            
            if result.returncode == 0:
                return result.stdout if result.stdout else "Execution completed successfully"
            else:
                return f"Error executing ParaView script:\n{result.stderr}"
                
        except Exception as e:
            return f"Exception during ParaView execution: {str(e)}"
    
    def extract_python(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""
        m = re.search(r"```python(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()
    
    def extract_errors(self, output: str) -> list:
        """Extract error messages from output."""
        return [ln for ln in output.splitlines() if "Error" in ln or "Traceback" in ln]
    
    def save_script(self, code: str, case: str, case_dir: Path):
        """Save generated script to results directory."""
        generated_dir = case_dir / "results" / "pvpython"
        generated_dir.mkdir(parents=True, exist_ok=True)
        fn = generated_dir / f"{case}.py"
        fn.write_text(code, encoding="utf-8")
        print(f"    • script saved → {fn.name}")
    
    async def run_single_test_case(self, test_case: YAMLTestCase) -> Dict:
        """Run a single test case and return results."""
        print(f"\n{'='*60}")
        print(f"Running ChatVis test case: {test_case.case_name}")
        print(f"{'='*60}")
        
        # Ensure directories exist
        test_case.ensure_directories()
        
        if not self.client:
            return {
                "case_name": test_case.case_name,
                "status": "failed",
                "error": "No client configured",
                "task_description": "",
                "response": "",
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            }
        
        # Get task description
        try:
            task_description = test_case.get_task_description()
        except Exception as e:
            return {
                "case_name": test_case.case_name,
                "status": "failed",
                "error": f"Failed to get task description: {str(e)}",
                "task_description": "",
                "response": "",
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            }
        
        # Create result structure
        start_time = datetime.now()
        
        # Initialize token counting
        input_tokens = self.token_counter.count_tokens(task_description)
        
        result = {
            "case_name": test_case.case_name,
            "status": "running",
            "start_time": start_time.isoformat(),
            "task_description": task_description,
            "llm_rubric": test_case.get_llm_rubric(),
            "evaluation_subtype": test_case.get_evaluation_subtype(),
            "is_vision_evaluation": test_case.is_vision_evaluation(),
            "response": "",
            "error": None,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "total_tokens": input_tokens
            }
        }
        
        try:
            print(f"Generating ParaView script for case: {test_case.case_name}")
            print(f"Task preview: {task_description[:200]}...")
            
            # Generate script using multi-provider client
            sr = self.client.create_completion(
                messages=[
                    {"role": "system", "content": (
                        "You are a code assistant. Output only a self-contained Python script for ParaView:\n"
                        "- Call ResetSession() once\n"
                        "- Import from paraview.simple only needed symbols\n"
                        "- Read raw via ImageReader(...) and configure DataScalarType, DataByteOrder, DataExtent, DataSpacing\n"
                        "- UpdatePipeline(), Show(...), Representation='Volume'\n"
                        "- Configure GetColorTransferFunction and GetOpacityTransferFunction\n"
                        "- SaveState(state_path) at end"
                    )},
                    {"role": "user", "content": task_description}
                ]
            )
            
            # Count output tokens and update token usage
            response_content = self.client.get_response_content(sr)
            token_usage = self.client.get_token_usage(sr)
            result["token_usage"]["output_tokens"] += token_usage["output_tokens"]
            result["token_usage"]["total_tokens"] = result["token_usage"]["input_tokens"] + result["token_usage"]["output_tokens"]
            
            # Calculate cost
            cost_info = self._calculate_cost(result["token_usage"]["input_tokens"], result["token_usage"]["output_tokens"])
            if cost_info:
                result["cost_info"] = cost_info
            
            code = self.extract_python(response_content)
            code = "from paraview.simple import ResetSession\nResetSession()\n" + code
            
            # Store the response
            result["response"] = response_content
            
            # Syntax check
            try:
                compile(code, '<string>', 'exec')
                print("    ✓ syntax check passed")
            except SyntaxError as e:
                print(f"    ✘ syntax error: {e}")
                result["status"] = "failed"
                result["error"] = f"Syntax error in generated code: {e}"
                return result
            
            # Save initial script
            self.save_script(code, test_case.case_name, test_case.case_path)
            
            # Execute and iterative fix
            max_attempts = 5
            out = self.execute_paraview_code(code)
            errors = self.extract_errors(out)
            attempt = 1
            
            while errors and attempt < max_attempts:
                print(f"    ⚠ attempt {attempt}: fixing {len(errors)} errors...")
                error_msg = "\\n".join(errors[:3])  # Limit to first 3 errors
                
                fix_sr = self.client.create_completion(
                    messages=[
                        {"role": "system", "content": "Fix the ParaView Python script based on the error message. Return only the corrected complete script."},
                        {"role": "user", "content": f"Original script:\n```python\n{code}\n```\n\nError:\n{error_msg}\n\nProvide the fixed script:"}
                    ]
                )
                
                # Count additional tokens
                fix_response = self.client.get_response_content(fix_sr)
                fix_token_usage = self.client.get_token_usage(fix_sr)
                result["token_usage"]["output_tokens"] += fix_token_usage["output_tokens"]
                result["token_usage"]["total_tokens"] += fix_token_usage["output_tokens"]
                
                code = self.extract_python(fix_response)
                code = "from paraview.simple import ResetSession\nResetSession()\n" + code
                
                # Try execution again
                out = self.execute_paraview_code(code)
                errors = self.extract_errors(out)
                attempt += 1
            
            if errors:
                print(f"    ✘ still has errors after {max_attempts} attempts")
                result["status"] = "failed"
                result["error"] = f"Script execution failed after {max_attempts} attempts. Last errors: {errors[:3]}"
            else:
                print("    ✓ script executed successfully")
                
                # Verify state file
                state_dir = test_case.case_path / "results" / "pvpython"
                state_dir.mkdir(parents=True, exist_ok=True)
                pvsm = state_dir / f"{test_case.case_name}.pvsm"
                
                if pvsm.exists():
                    print(f"    ✓ state file created: {pvsm.name}")
                    result["status"] = "completed"
                else:
                    print(f"    ⚠ no state file found at {pvsm}")
                    result["status"] = "completed_no_state"
                
                # Save final script
                self.save_script(code, test_case.case_name, test_case.case_path)
            
            # Update cost info with final token count
            if cost_info:
                final_cost_info = self._calculate_cost(result["token_usage"]["input_tokens"], result["token_usage"]["output_tokens"])
                if final_cost_info:
                    result["cost_info"] = final_cost_info
            
            print(f"✅ Test case {test_case.case_name} completed")
            
        except Exception as e:
            result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ Test case {test_case.case_name} failed: {str(e)}")
        
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration_seconds"] = (end_time - start_time).total_seconds()
        
        # Save test result
        await self.save_test_result(test_case, result)
        
        return result
    
    async def save_test_result(self, test_case: YAMLTestCase, result: Dict):
        """Save test result to file."""
        # Save to test_results directory (required for efficiency evaluation)
        test_results_dir = test_case.case_path / "test_results" / "pvpython"
        test_results_dir.mkdir(parents=True, exist_ok=True)
        result_file = test_results_dir / f"test_result_{int(time.time())}.json"
        
        # Also save to centralized output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        centralized_file = self.output_dir / f"{test_case.case_name}_result_{int(time.time())}.json"
        
        # Format result data for efficiency evaluation compatibility
        # SciVisEvaluator expects duration and token_usage at top level
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "case_name": result.get("case_name", test_case.case_name),
            "status": result.get("status", "unknown"),
            "duration": result.get("duration_seconds", 0),
            "token_usage": result.get("token_usage", {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }),
            "response": result.get("response", ""),
            "task_description": result.get("task_description", ""),
            "full_result": result  # Keep the original result for reference
        }
        
        for file_path in [result_file, centralized_file]:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Test result saved to: {result_file}")
    
    async def run_evaluation(self, test_case: YAMLTestCase) -> Dict:
        """Run comprehensive evaluation for a test case with support for multiple subtypes."""
        if not self.eval_client:
            print("⚠️  No evaluation client available, skipping evaluation")
            return {"status": "skipped", "reason": "No evaluation client available"}
        
        print(f"🔍 Evaluating test case: {test_case.case_name}")
        print(f"Available evaluation subtypes: {test_case.get_all_evaluation_subtypes()}")
        
        evaluation_results = {}
        total_score = 0
        max_possible_score = 0
        
        # Evaluate each subtype
        for subtype in test_case.get_all_evaluation_subtypes():
            print(f"Running {subtype} evaluation...")
            
            if subtype == 'vision':
                result = await self._run_vision_evaluation(test_case)
            elif subtype == 'text':
                result = await self._run_text_evaluation(test_case)
            else:
                print(f"⚠️  Unknown evaluation subtype: {subtype}")
                continue
            
            if result and result.get('status') == 'completed':
                evaluation_results[subtype] = result
                total_score += result.get('scores', {}).get('total_score', 0)
                max_possible_score += result.get('scores', {}).get('max_possible_score', 0)
        
        # Create comprehensive evaluation result
        final_result = {
            "status": "completed",
            "case_name": test_case.case_name,
            "model": self.eval_model,
            "evaluation_subtypes": test_case.get_all_evaluation_subtypes(),
            "subtype_results": evaluation_results,
            "scores": {
                "total_score": total_score,
                "max_possible_score": max_possible_score,
                "percentage": (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save evaluation result
        eval_file = test_case.evaluation_dir / f"evaluation_result_{int(time.time())}.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Evaluation completed for {test_case.case_name}")
        print(f"Total Score: {total_score}/{max_possible_score} ({final_result['scores']['percentage']:.1f}%)")
        
        return final_result

    async def _run_vision_evaluation(self, test_case: YAMLTestCase) -> Dict:
        """Run vision-based evaluation using PVPythonAutoEvaluator."""
        # Import evaluation helpers
        sys.path.append(str(Path(__file__).parent / "evaluation_helpers"))
        try:
            from evaluation_helpers.pvpython_auto_evaluator import PVPythonAutoEvaluator
        except ImportError as e:
            print(f"⚠️  Could not import evaluation helpers: {e}")
            return {"status": "failed", "reason": f"Import error: {e}"}
        
        try:
            # Create evaluator instance
            case_dir = str(test_case.case_path)
            evaluator = PVPythonAutoEvaluator(case_dir, test_case.case_name, os.getenv('OPENAI_API_KEY'), self.eval_model)
            
            # Check if required files exist for evaluation
            gs_state_path = test_case.case_path / "GS" / f"{test_case.case_name}_gs.pvsm"
            result_state_path = test_case.case_path / "results" / "pvpython" / f"{test_case.case_name}.pvsm"
            
            if not gs_state_path.exists():
                return {"status": "failed", "reason": f"Ground truth state not found: {gs_state_path}"}
            
            if not result_state_path.exists():
                return {"status": "failed", "reason": f"Result state not found: {result_state_path}"}
            
            # Write YAML rubric to visualization_goals.txt for the evaluator
            vision_rubric = test_case.get_rubric_for_subtype('vision')
            goals_file = test_case.case_path / "visualization_goals.txt"
            with open(goals_file, 'w', encoding='utf-8') as f:
                f.write(vision_rubric)
            
            # Get goals count for dynamic max score calculation
            goals_count = evaluator.get_goals_count()
            
            # Run all evaluation components
            viz_score = evaluator.evaluate_visualization_quality()
            code_score = evaluator.evaluate_code_similarity()
            output_score = evaluator.evaluate_output_generation()
            efficiency_score = evaluator.evaluate_efficiency()
            
            # Calculate image metrics
            image_metrics = evaluator.evaluate_image_metrics()
            
            # Calculate total score
            total_score = viz_score + code_score + output_score + efficiency_score
            max_possible_score = (goals_count * 10) + 20 + 10 + 10  # viz + code + output + efficiency
            
            # Create vision evaluation result
            result = {
                "status": "completed",
                "subtype": "vision",
                "rubric": vision_rubric,
                "goals_count": goals_count,
                "scores": {
                    "visualization_quality": viz_score,
                    "code_similarity": code_score,
                    "output_generation": output_score,
                    "efficiency": efficiency_score,
                    "total_score": total_score,
                    "max_possible_score": max_possible_score,
                    "percentage": (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
                },
                "image_metrics": image_metrics,
                "detailed_scores": evaluator.evaluation_results.get("scores", {}),
                "evaluator_metadata": {
                    "evaluator_type": "chatvis_yaml_auto_vision",
                    "evaluator_version": "1.0.0",
                    "goals_count": goals_count,
                    "scoring_scheme": {
                        "visualization_quality": f"{goals_count * 10} points (10 per goal)",
                        "code_similarity": "20 points (BERT-based similarity)",
                        "output_generation": "10 points (file existence)",
                        "efficiency": "10 points (execution success)",
                        "total_possible": f"{max_possible_score} points"
                    }
                }
            }
            
            # Print evaluation summary
            print(f"Vision Evaluation - Visualization Quality: {viz_score}/{goals_count * 10}")
            print(f"   Code Similarity: {code_score}/20")
            print(f"   Output Generation: {output_score}/10")
            print(f"   Efficiency: {efficiency_score}/10")
            
            # Print image metrics if available
            if image_metrics and "averaged_metrics" in image_metrics:
                metrics = image_metrics["averaged_metrics"]
                print(f"   Image Metrics - PSNR: {metrics.get('psnr', 'N/A')}, SSIM: {metrics.get('ssim', 'N/A')}, LPIPS: {metrics.get('lpips', 'N/A')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Vision evaluation failed for {test_case.case_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "subtype": "vision", "reason": str(e)}

    async def _run_text_evaluation(self, test_case: YAMLTestCase) -> Dict:
        """Run text-based evaluation using answers.txt file."""
        try:
            # Load the answers.txt file
            answers_file = test_case.case_path / "results" / "pvpython" / "answers.txt"
            if not answers_file.exists():
                print(f"⚠️  answers.txt not found: {answers_file}")
                return {"status": "failed", "subtype": "text", "reason": "answers.txt not found"}
            
            with open(answers_file, 'r', encoding='utf-8') as f:
                answers_content = f.read().strip()
            
            if not answers_content:
                print(f"⚠️  answers.txt is empty")
                return {"status": "failed", "subtype": "text", "reason": "answers.txt is empty"}
            
            # Get the text rubric
            text_rubric = test_case.get_rubric_for_subtype('text')
            if not text_rubric:
                print(f"⚠️  No text rubric found")
                return {"status": "failed", "subtype": "text", "reason": "No text rubric found"}
            
            # Count goals for scoring (each goal worth 10 points)
            goals_count = len([line for line in text_rubric.split('\n') if line.strip()])
            max_score = goals_count * 10
            
            # Create evaluation prompt
            prompt = f"""You are an expert evaluator for scientific visualization tasks. You need to evaluate the provided answers against specific criteria.

TASK ANSWERS:
{answers_content}

EVALUATION CRITERIA:
{text_rubric}

Please evaluate the answers based on how well they meet the evaluation criteria. Each criterion should be scored out of 10 points.

Respond with a JSON object in the following format:
{{
    "score": <total_score_out_of_{max_score}>,
    "max_score": {max_score},
    "explanation": "<detailed explanation of the evaluation>"
}}

Be specific about what aspects of the answers meet or don't meet the criteria."""
            
            # Use evaluation client
            client = self.eval_client
            
            response = client.chat.completions.create(
                model=self.eval_model,
                messages=[
                    {"role": "system", "content": "You are an expert scientific visualization evaluator. Provide accurate and fair assessments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            try:
                # Extract JSON from response
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = response_content[json_start:json_end]
                    evaluation_result = json.loads(json_content)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️  Failed to parse LLM response as JSON: {e}")
                print(f"Response was: {response_content}")
                return {"status": "failed", "subtype": "text", "reason": f"Failed to parse evaluation: {e}"}
            
            score = evaluation_result.get('score', 0)
            explanation = evaluation_result.get('explanation', 'No explanation provided')
            
            result = {
                "status": "completed",
                "subtype": "text",
                "rubric": text_rubric,
                "answers_content": answers_content,
                "scores": {
                    "total_score": score,
                    "max_possible_score": max_score,
                    "percentage": (score / max_score * 100) if max_score > 0 else 0
                },
                "explanation": explanation,
                "evaluation_response": response_content
            }
            
            print(f"Text Evaluation Score: {score}/{max_score} ({result['scores']['percentage']:.1f}%)")
            
            return result
            
        except Exception as e:
            print(f"❌ Text evaluation failed for {test_case.case_name}: {str(e)}")
            return {"status": "failed", "subtype": "text", "reason": str(e)}
    
    async def run_all_test_cases(self, run_evaluation: bool = True) -> Dict:
        """Run all test cases and optionally evaluate them."""
        if not self.test_cases:
            self.load_yaml_test_cases()
        
        if not self.test_cases:
            return {"error": "No valid test cases found"}
        
        print(f"\n🚀 Running {len(self.test_cases)} ChatVis test cases from YAML configuration")
        print(f"YAML file: {self.yaml_path}")
        print(f"Cases directory: {self.cases_dir}")
        print(f"Output directory: {self.output_dir}")
        
        start_time = datetime.now()
        results = []
        
        for test_case in self.test_cases:
            # Run test case
            result = await self.run_single_test_case(test_case)
            results.append(result)
            
            # Run evaluation if requested and test case completed successfully
            if run_evaluation and result.get("status") in ["completed", "completed_no_state"]:
                eval_result = await self.run_evaluation(test_case)
                result["evaluation"] = eval_result
        
        end_time = datetime.now()
        
        # Calculate summary statistics
        total_cases = len(results)
        successful_cases = len([r for r in results if r.get("status") in ["completed", "completed_no_state"]])
        failed_cases = total_cases - successful_cases
        
        total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in results)
        total_cost = sum(r.get("cost_info", {}).get("total_cost", 0) for r in results)
        
        summary = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "total_cases": total_cases,
            "successful_cases": successful_cases,
            "failed_cases": failed_cases,
            "success_rate": successful_cases / total_cases if total_cases > 0 else 0,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "currency": "USD",
            "agent_model": self.model,
            "evaluation_model": self.eval_model,
            "runner_type": "yaml_chatvis",
            "results": results
        }
        
        # Save summary
        summary_file = self.output_dir / f"chatvis_yaml_test_summary_{int(time.time())}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"🎯 CHATVIS YAML TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total cases: {total_cases}")
        print(f"Successful: {successful_cases}")
        print(f"Failed: {failed_cases}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total tokens: {total_tokens:,}")
        if total_cost > 0:
            print(f"Total cost: ${total_cost:.6f}")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Results saved: {summary_file}")
        
        return summary


async def main():
    parser = argparse.ArgumentParser(description="Run ChatVis test cases from YAML configuration")
    parser.add_argument("--yaml", "-y", required=True,
                       help="Path to the YAML test cases file")
    parser.add_argument("--cases", required=True,
                       help="Path to the cases directory")
    parser.add_argument("--output", "-o", 
                       help="Output directory for results (default: cases_parent/test_results/chatvis_yaml)")
    parser.add_argument("--case", 
                       help="Run a specific test case by name")
    parser.add_argument("--list", action="store_true",
                       help="List available test cases and exit")
    parser.add_argument("--no-eval", action="store_true",
                       help="Skip LLM-based evaluation")
    parser.add_argument("--config", "-c",
                       help="Path to the configuration JSON file (specifies model and provider)")
    parser.add_argument("--eval-model", default="gpt-4o",
                       help="Model for evaluation (default: gpt-4o)")
    parser.add_argument("--api-key",
                       help="API key (can also use environment variables)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.yaml):
        print(f"Error: YAML file not found: {args.yaml}")
        sys.exit(1)
    
    if not os.path.exists(args.cases):
        print(f"Error: Cases directory not found: {args.cases}")
        sys.exit(1)
    
    # Config file is now required
    if not args.config:
        print("Error: Configuration file is required. Use --config to specify a config file.")
        print("Available config files:")
        config_dir = Path(__file__).parent / "configs" / "chatvis"
        if config_dir.exists():
            for config_file in config_dir.glob("*.json"):
                print(f"  {config_file}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Get model from config file
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        model_from_config = config.get('model', '')
        print(f"Using model from config: {model_from_config}")
    except Exception as e:
        print(f"Error: Could not read config file: {e}")
        sys.exit(1)
    
    runner = ChatVisYAMLTestRunner(
        yaml_path=args.yaml,
        cases_dir=args.cases,
        output_dir=args.output,
        config_path=args.config,
        model=model_from_config,
        eval_model=args.eval_model
    )
    
    # Load test cases
    test_cases = runner.load_yaml_test_cases()
    
    if args.list:
        print("Available test cases:")
        for case in test_cases:
            print(f"  - {case.case_name}")
        return
    
    if not test_cases:
        print("Error: No valid test cases found in YAML file")
        sys.exit(1)
    
    # Run specific case or all cases
    if args.case:
        # Find specific case
        target_case = None
        for case in test_cases:
            if case.case_name == args.case:
                target_case = case
                break
        
        if not target_case:
            print(f"Error: Test case '{args.case}' not found")
            print("Available test cases:")
            for case in test_cases:
                print(f"  - {case.case_name}")
            sys.exit(1)
        
        # Run single test case
        result = await runner.run_single_test_case(target_case)
        
        # Run evaluation if requested and test case completed successfully
        if not args.no_eval and result.get("status") in ["completed", "completed_no_state"]:
            eval_result = await runner.run_evaluation(target_case)
            result["evaluation"] = eval_result
        
        # Print final result
        if result.get("status") in ["completed", "completed_no_state"]:
            print(f"✅ Test case {args.case} completed successfully")
        else:
            print(f"❌ Test case {args.case} failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    else:
        # Run all cases
        summary = await runner.run_all_test_cases(run_evaluation=not args.no_eval)
        
        if summary.get("failed_cases", 0) > 0:
            print(f"❌ {summary['failed_cases']} test case(s) failed")
            sys.exit(1)
        else:
            print(f"✅ All {summary['successful_cases']} test case(s) completed successfully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)