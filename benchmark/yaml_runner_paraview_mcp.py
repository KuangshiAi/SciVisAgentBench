#!/usr/bin/env python3
"""
YAML Test Runner for SciVisAgentBench

This script loads test cases from a YAML configuration file and runs them through 
the TinyAgent using MCP, then evaluates the results using LLM-based evaluation.
Combines functionality from mcp_test_runner.py and mcp_auto_evaluator.py.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import tiktoken
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from tiny_agent.agent import TinyAgent


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
    
    def __init__(self, yaml_data: Dict, case_name: str, cases_dir: str, config_path: Optional[str] = None):
        self.yaml_data = yaml_data
        self.case_name = case_name
        self.cases_dir = Path(cases_dir)
        self.config_path = config_path
        
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
        self.results_dir = self.case_path / "results" / "mcp"
        self.evaluation_dir = self.case_path / "evaluation_results" / "mcp"
        
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
        
        working_dir = self.cases_dir
        
        # Prepend working directory information
        working_dir_info = f'Your agent_mode is "mcp", use it when saving results. Your working directory is "{working_dir}", and you should have access to it. In the following prompts, we will use relative path with respect to your working path. But remember, when you load or save any file, always stick to absolute path.'
        
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


class YAMLTestRunner:
    """Runs test cases loaded from YAML configuration."""
    
    def __init__(self, config_path: str, yaml_path: str, cases_dir: str, 
                 output_dir: Optional[str] = None, openai_api_key: Optional[str] = None,
                 eval_model: str = "gpt-4o"):
        # Convert config_path to absolute path
        if not os.path.isabs(config_path):
            self.config_path = os.path.abspath(config_path)
        else:
            self.config_path = config_path
            
        self.yaml_path = Path(yaml_path)
        self.cases_dir = Path(cases_dir)
        self.output_dir = Path(output_dir) if output_dir else self.cases_dir.parent / "test_results" / "yaml_mcp"
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.eval_model = eval_model
        self.test_cases: List[YAMLTestCase] = []
        self.token_counter = TokenCounter()
        
        # Load agent model from config
        self.agent_model = self._load_agent_model()
        
        # Load pricing information from config
        self.pricing_info = self._load_pricing_info()
    
    def _load_agent_model(self) -> str:
        """Load agent model from the config file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("model", "unknown")
        except Exception as e:
            print(f"Warning: Could not load agent model from config: {e}")
            return "unknown"
    
    def _load_pricing_info(self) -> Optional[Dict]:
        """Load pricing information from the config file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("pricing")
        except Exception as e:
            print(f"Warning: Could not load pricing info: {e}")
            return None
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> Optional[Dict]:
        """Calculate cost based on token usage and pricing info."""
        if not self.pricing_info:
            return None
        
        try:
            input_cost_per_1k = float(self.pricing_info.get("input_cost_per_1k_tokens", 0))
            output_cost_per_1k = float(self.pricing_info.get("output_cost_per_1k_tokens", 0))
            
            input_cost = (input_tokens / 1000) * input_cost_per_1k
            output_cost = (output_tokens / 1000) * output_cost_per_1k
            total_cost = input_cost + output_cost
            
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6),
                "currency": self.pricing_info.get("currency", "USD")
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
                    test_case = YAMLTestCase(case_data, case_name, str(self.cases_dir), self.config_path)
                    print(f"Debug: Case {case_name} - Task: {bool(test_case.task_description)} - Rubrics: {test_case.rubrics} - Valid: {test_case.is_valid()}")
                    if test_case.is_valid():
                        test_cases.append(test_case)
                        print(f"Loaded test case: {case_name}")
                    else:
                        print(f"Skipping invalid test case: {case_name}")
                        print(f"  Task description present: {bool(test_case.task_description)}")
                        print(f"  Rubrics available: {list(test_case.rubrics.keys())}")
                        print(f"  Case path: {test_case.case_path}")
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
    
    def _create_test_case_config(self, test_case: YAMLTestCase) -> str:
        """Create a unique config file for each test case."""
        # Load the original config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        working_dir = self.cases_dir
        
        # Create a temporary config file for this test case
        temp_config_dir = Path(tempfile.gettempdir()) / "scivisagent_bench_configs"
        temp_config_dir.mkdir(exist_ok=True)
        
        case_config_file = temp_config_dir / f"config_{test_case.case_name}_{int(time.time())}.json"
        
        # Modify the config for this specific test case
        config_modified = False
        for server in config.get("servers", []):
            if server.get("command") == "python" and "paraview_mcp_server.py" in str(server.get("args", [])):
                # Update the server args to point to the correct working directory
                args = server.get("args", [])
                for i, arg in enumerate(args):
                    if arg == "--cases-dir" and i + 1 < len(args):
                        args[i + 1] = str(self.cases_dir)
                        config_modified = True
        
        # Add test case name as environment variable for unique session naming
        for server in config.get("servers", []):
            if "env" not in server:
                server["env"] = {}
            server["env"]["SCIVISBENCH_CASE_NAME"] = test_case.case_name
        
        # Save the modified config to the temporary file
        with open(case_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"Created test case config: {case_config_file}")
        return str(case_config_file)
    
    async def run_single_test_case(self, test_case: YAMLTestCase) -> Dict:
        """Run a single test case and return results."""
        print(f"\n{'='*60}")
        print(f"Running test case: {test_case.case_name}")
        print(f"{'='*60}")
        
        # Ensure directories exist
        test_case.ensure_directories()
        
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
        
        # Create agent from config
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
            # Create test case specific config
            case_config_path = self._create_test_case_config(test_case)
            
            try:
                # Create agent using from_config_file method like mcp_test_runner
                agent = TinyAgent.from_config_file(case_config_path)
                
                async with agent:
                    await agent.load_tools()
                    print(f"Agent loaded with {len(agent.available_tools)} tools")
                    
                    # Clear ParaView state at the beginning of each test case
                    print("Clearing ParaView state for fresh start...")
                    await self._clear_paraview_state(agent)
                    
                    # Run the task
                    print(f"Starting execution for case: {test_case.case_name}")
                    print(f"Task preview: {task_description[:200]}...")
                    
                    response_parts = []
                    
                    async for chunk in agent.run(task_description):
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                content = delta.content
                                response_parts.append(content)
                                print(content, end="", flush=True)
                        elif hasattr(chunk, 'role') and chunk.role == "tool":
                            tool_message = f"\n[Tool: {chunk.name}] {chunk.content}"
                            response_parts.append(tool_message)
                            print(tool_message)
                    
                    full_response = "".join(response_parts)
                    
                    # Count output tokens
                    output_tokens = self.token_counter.count_tokens(full_response)
                    
                    result.update({
                        "status": "completed",
                        "response": full_response,
                        "token_usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens
                        }
                    })
                    
                    # Calculate cost if pricing info is available
                    cost_info = self._calculate_cost(input_tokens, output_tokens)
                    if cost_info:
                        result["cost"] = cost_info
                    
                    print(f"\n✅ Test case {test_case.case_name} completed successfully")
                
            finally:
                # Clean up temporary config file
                try:
                    os.unlink(case_config_path)
                except Exception as e:
                    print(f"Warning: Could not clean up config file: {e}")
                
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
        test_results_dir = test_case.case_path / "test_results" / "mcp"
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
        if not self.openai_api_key:
            print("⚠️  No OpenAI API key provided, skipping evaluation")
            return {"status": "skipped", "reason": "No OpenAI API key"}
        
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
        """Run vision-based evaluation using MCPAutoEvaluator."""
        # Import evaluation helpers
        sys.path.append(str(Path(__file__).parent / "evaluation_helpers"))
        try:
            from evaluation_helpers.mcp_auto_evaluator import MCPAutoEvaluator
        except ImportError as e:
            print(f"⚠️  Could not import evaluation helpers: {e}")
            return {"status": "failed", "reason": f"Import error: {e}"}
        
        try:
            # Create visualization_goals.txt file for the evaluator
            vision_rubric = test_case.get_rubric_for_subtype('vision')
            goals_file = test_case.case_path / "visualization_goals.txt"
            with open(goals_file, 'w', encoding='utf-8') as f:
                f.write(vision_rubric)
            
            # Initialize the MCP auto evaluator
            evaluator = MCPAutoEvaluator(
                case_dir=str(test_case.case_path),
                case_name=test_case.case_name,
                openai_api_key=self.openai_api_key,
                model=self.eval_model
            )
            
            # Run the complete evaluation
            evaluation_results = evaluator.run_evaluation()
            
            # Extract key metrics
            total_score = evaluation_results.get("total_score", 0)
            max_possible_score = evaluation_results.get("max_possible_score", 0)
            scores = evaluation_results.get("scores", {})
            
            # Create vision evaluation result
            result = {
                "status": "completed",
                "subtype": "vision",
                "rubric": vision_rubric,
                "scores": {
                    "visualization_quality": scores.get("visualization_quality", {}).get("score", 0),
                    "output_generation": scores.get("output_generation", {}).get("score", 0),
                    "efficiency": scores.get("efficiency", {}).get("score", 0),
                    "total_score": total_score,
                    "max_possible_score": max_possible_score,
                    "percentage": (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
                },
                "image_metrics": evaluation_results.get("image_metrics", {}),
                "detailed_scores": scores,
                "evaluator_metadata": evaluation_results.get("evaluator_metadata", {}),
            }
            
            # Print image metrics if available
            if result.get("image_metrics"):
                metrics = result["image_metrics"]
                if "summary" in metrics:
                    summary = metrics["summary"]
                    print(f"Vision Evaluation - PSNR: {summary.get('avg_psnr', 'N/A'):.2f}, "
                          f"SSIM: {summary.get('avg_ssim', 'N/A'):.3f}, "
                          f"LPIPS: {summary.get('avg_lpips', 'N/A'):.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Vision evaluation failed for {test_case.case_name}: {str(e)}")
            return {"status": "failed", "subtype": "vision", "reason": str(e)}

    async def _run_text_evaluation(self, test_case: YAMLTestCase) -> Dict:
        """Run text-based evaluation using answers.txt file."""
        try:
            # Load the answers.txt file
            answers_file = test_case.case_path / "results" / "mcp" / "answers.txt"
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
            
            # Use OpenAI to evaluate
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
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
        
        print(f"\n🚀 Running {len(self.test_cases)} test cases from YAML configuration")
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
            if run_evaluation and result.get("status") == "completed":
                eval_result = await self.run_evaluation(test_case)
                result["evaluation"] = eval_result
        
        end_time = datetime.now()
        
        # Calculate summary statistics
        total_cases = len(results)
        successful_cases = len([r for r in results if r.get("status") == "completed"])
        failed_cases = total_cases - successful_cases
        
        total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in results)
        total_cost = sum(r.get("cost", {}).get("total_cost", 0) for r in results)
        
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
            "agent_model": self.agent_model,
            "evaluation_model": self.eval_model,
            "runner_type": "yaml_mcp",
            "results": results
        }
        
        # Save summary
        summary_file = self.output_dir / f"yaml_test_summary_{int(time.time())}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"🎯 YAML TEST SUMMARY")
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
    
    async def _clear_paraview_state(self, agent) -> None:
        """Clear ParaView pipeline state before starting."""
        try:
            print("Clearing ParaView pipeline state through MCP...")
            
            # Send a simple clear_pipeline command to the agent
            clear_message = "Call the clear_pipeline_and_reset tool to clear all sources from the ParaView pipeline."
            
            response_parts = []
            async for chunk in agent.run(clear_message):
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content = delta.content
                        response_parts.append(content)
                elif hasattr(chunk, 'role') and chunk.role == "tool":
                    tool_message = f"[Tool: {chunk.name}] {chunk.content}"
                    response_parts.append(tool_message)
                    if chunk.name == "clear_pipeline_and_reset":
                        print(f"Pipeline clearing result: {chunk.content}")
            
            if response_parts:
                full_response = "".join(response_parts)
                print("Pipeline clearing completed")
            else:
                print("No response received for pipeline clearing")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not clear ParaView state: {e}")
            print("Test case will continue, but may have residual state from previous runs")


async def main():
    parser = argparse.ArgumentParser(description="Run test cases from YAML configuration")
    parser.add_argument("--config", "-c", required=True, 
                       help="Path to the MCP configuration JSON file")
    parser.add_argument("--yaml", "-y", required=True,
                       help="Path to the YAML test cases file")
    parser.add_argument("--cases", required=True,
                       help="Path to the cases directory")
    parser.add_argument("--output", "-o", 
                       help="Output directory for results (default: cases_parent/test_results/yaml_mcp)")
    parser.add_argument("--case", 
                       help="Run a specific test case by name")
    parser.add_argument("--list", action="store_true",
                       help="List available test cases and exit")
    parser.add_argument("--no-eval", action="store_true",
                       help="Skip LLM-based evaluation")
    parser.add_argument("--eval-model", default="gpt-4o",
                       help="OpenAI model for evaluation (default: gpt-4o)")
    parser.add_argument("--api-key",
                       help="OpenAI API key for evaluation (can also use OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.yaml):
        print(f"Error: YAML file not found: {args.yaml}")
        sys.exit(1)
    
    if not os.path.exists(args.cases):
        print(f"Error: Cases directory not found: {args.cases}")
        sys.exit(1)
    
    # Get API key for evaluation
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not args.no_eval and not api_key:
        print("Warning: No OpenAI API key provided. Evaluation will be skipped.")
        print("Set OPENAI_API_KEY environment variable or use --api-key to enable evaluation.")
    
    runner = YAMLTestRunner(
        config_path=args.config,
        yaml_path=args.yaml,
        cases_dir=args.cases,
        output_dir=args.output,
        openai_api_key=api_key,
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
            print("Available cases:", [case.case_name for case in test_cases])
            sys.exit(1)
        
        # Run single case
        print(f"Running single test case: {args.case}")
        result = await runner.run_single_test_case(target_case)
        
        # Run evaluation if requested
        if not args.no_eval and result.get("status") == "completed" and api_key:
            eval_result = await runner.run_evaluation(target_case)
            result["evaluation"] = eval_result
        
        print(f"Case result: {result.get('status')}")
    else:
        # Run all cases
        summary = await runner.run_all_test_cases(run_evaluation=not args.no_eval)
        print(f"Overall success rate: {summary.get('success_rate', 0):.1%}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)