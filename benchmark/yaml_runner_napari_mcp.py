#!/usr/bin/env python3
"""
YAML Test Runner for SciVisAgentBench - Napari MCP

This script loads test cases from a YAML configuration file and runs them through 
the TinyAgent using MCP, then evaluates the results based on assertions in the YAML.
Supports contains-all, not-contains, and llm-rubric assertion types.
"""

import asyncio
import argparse
import json
import os
import re
import sys
import tempfile
import time
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
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        if self.tokenizer is None:
            return len(text.split()) * 2  # Rough approximation
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"Warning: Token counting failed: {e}")
            return len(text.split()) * 2


class YAMLTestCase:
    """Represents a single test case loaded from YAML."""
    
    def __init__(self, yaml_data: Dict, case_name: str, cases_dir: str, config_path: Optional[str] = None):
        self.yaml_data = yaml_data
        self.case_name = case_name
        self.cases_dir = Path(cases_dir)
        self.config_path = config_path
        
        # Extract question and rubric from YAML
        self.task_description = yaml_data.get('vars', {}).get('question', '').strip()
        
        # Extract assertions from assert section
        assert_list = yaml_data.get('assert', [])
        self.assertions = assert_list  # Store all assertions for evaluation
        
        # Extract LLM rubric if present (single rubric only)
        self.llm_rubric = ''
        for assert_item in assert_list:
            assert_type = assert_item.get('type', '')
            if assert_type == 'llm-rubric':
                rubric_value = assert_item.get('value', '')
                if isinstance(rubric_value, list):
                    rubric_value = '\n'.join(str(item) for item in rubric_value)
                self.llm_rubric = rubric_value
                break  # Only take the first llm-rubric
        
        # Set up paths
        self.case_path = self.cases_dir / case_name
        self.results_dir = self.case_path / "results" / "mcp"
        self.evaluation_dir = self.case_path / "evaluation_results" / "mcp"
        
    def is_valid(self) -> bool:
        """Check if this is a valid test case."""
        has_assertions = bool(self.assertions)  # Check if we have any assertions
        has_task = bool(self.task_description)
        return has_task and has_assertions
    
    def get_task_description(self) -> str:
        """Get the task description with working directory information."""
        if not self.task_description:
            return "No task description provided"
        
        working_dir = self.cases_dir
        
        # Prepend working directory information
        working_dir_info = f'Your agent_mode is "mcp", use it when saving results. Your working directory is "{working_dir}", and you should have access to it. In the following prompts, we will use relative path with respect to your working path. But remember, when you load or save any file, always stick to absolute path.'
        
        return f"{working_dir_info}\n\n{self.task_description}"
    
    def get_assertions(self) -> List[Dict]:
        """Get all assertions for this test case."""
        return self.assertions.copy()
    
    def has_llm_rubric(self) -> bool:
        """Check if this test case has an LLM rubric for evaluation."""
        return bool(self.llm_rubric)
    
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
            return config.get('agent_model', 'gpt-4o')
        except Exception as e:
            print(f"Warning: Could not load agent model from config: {e}")
            return 'gpt-4o'
    
    def _load_pricing_info(self) -> Optional[Dict]:
        """Load pricing information from the config file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('pricing', None)
        except Exception as e:
            print(f"Warning: Could not load pricing info from config: {e}")
            return None
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> Optional[Dict]:
        """Calculate cost based on token usage and pricing info."""
        if not self.pricing_info:
            return None
        
        try:
            model_pricing = self.pricing_info.get(self.agent_model, {})
            input_cost_per_1k = float(model_pricing.get('input_cost_per_1k_tokens', 0))
            output_cost_per_1k = float(model_pricing.get('output_cost_per_1k_tokens', 0))
            
            input_cost = (input_tokens / 1000) * input_cost_per_1k
            output_cost = (output_tokens / 1000) * output_cost_per_1k
            total_cost = input_cost + output_cost
            
            return {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "currency": "USD",
                "input_cost_per_1k": input_cost_per_1k,
                "output_cost_per_1k": output_cost_per_1k
            }
        except (ValueError, KeyError) as e:
            print(f"Warning: Cost calculation failed: {e}")
            return None
    
    def load_yaml_test_cases(self) -> List[YAMLTestCase]:
        """Load test cases from YAML file."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")
        
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = __import__('yaml').safe_load(f)
        
        test_cases = []
        
        # Parse YAML data - expect a list of test case dictionaries
        if isinstance(yaml_data, list):
            for index, case_data in enumerate(yaml_data):
                if isinstance(case_data, dict):
                    case_name = self._extract_case_name(case_data, index)
                    if case_name:
                        test_case = YAMLTestCase(
                            yaml_data=case_data,
                            case_name=case_name,
                            cases_dir=str(self.cases_dir),
                            config_path=self.config_path
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
        # Look for dataset name in the task description
        task_description = case_data.get('vars', {}).get('question', '')
        
        # For anonymized datasets, look for dataset_XXX pattern
        import re
        dataset_match = re.search(r'dataset_(\d+)', task_description)
        if dataset_match:
            return f"dataset_{dataset_match.group(1).zfill(3)}"
        
        # Common patterns to extract dataset names (for non-anonymized data)
        dataset_patterns = [
            'bonsai', 'carp', 'chameleon', 'engine', 'solar-plume', 
            'supernova', 'tangaroa', 'tornado', 'vortex'
        ]
        
        for pattern in dataset_patterns:
            if pattern.lower() in task_description.lower():
                return pattern
        
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
            if server.get("command") == "python" and "napari_mcp" in str(server.get("args", [])):
                server["cwd"] = str(working_dir)
                config_modified = True
        
        # Add test case name as environment variable for unique session naming
        for server in config.get("servers", []):
            if "env" not in server:
                server["env"] = {}
            server["env"]["TEST_CASE_NAME"] = test_case.case_name
        
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
            "full_result": result  # Include the complete result for reference
        }
        
        for file_path in [result_file, centralized_file]:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Test result saved to: {result_file}")
    
    async def run_evaluation(self, test_case: YAMLTestCase) -> Dict:
        """Run evaluation for a test case based on assertions."""
        print(f"🔍 Evaluating test case: {test_case.case_name}")
        
        # Get the agent's response from the most recent test result
        test_results_dir = test_case.case_path / "test_results" / "mcp"
        if not test_results_dir.exists():
            return {"status": "failed", "reason": "No test results found"}
        
        # Find the most recent test result file
        result_files = list(test_results_dir.glob("test_result_*.json"))
        if not result_files:
            return {"status": "failed", "reason": "No test result files found"}
        
        latest_result_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        # Load the test result to get the agent's response
        try:
            with open(latest_result_file, 'r', encoding='utf-8') as f:
                test_result = json.load(f)
            agent_response = test_result.get('response', '')
        except Exception as e:
            return {"status": "failed", "reason": f"Failed to load test result: {e}"}
        
        if not agent_response:
            return {"status": "failed", "reason": "No agent response found"}
        
        # Evaluate all assertions
        assertions = test_case.get_assertions()
        evaluation_results = []
        total_passed = 0
        
        for i, assertion in enumerate(assertions):
            assert_type = assertion.get('type', '')
            assert_value = assertion.get('value', '')
            
            print(f"  Evaluating assertion {i+1}: {assert_type}")
            
            result = await self._evaluate_assertion(agent_response, assert_type, assert_value, test_case)
            evaluation_results.append({
                "assertion_index": i,
                "type": assert_type,
                "value": assert_value,
                "passed": result["passed"],
                "details": result.get("details", "")
            })
            
            if result["passed"]:
                total_passed += 1
        
        # Create evaluation result
        final_result = {
            "status": "completed",
            "case_name": test_case.case_name,
            "model": self.eval_model if self.eval_model else "rule-based",
            "agent_response": agent_response,
            "assertion_results": evaluation_results,
            "scores": {
                "total_passed": total_passed,
                "total_assertions": len(assertions),
                "pass_rate": total_passed / len(assertions) if assertions else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save evaluation result
        eval_file = test_case.evaluation_dir / f"evaluation_result_{int(time.time())}.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Evaluation completed for {test_case.case_name}")
        print(f"Passed: {total_passed}/{len(assertions)} ({final_result['scores']['pass_rate']:.1%})")
        
        return final_result

    async def _evaluate_assertion(self, agent_response: str, assert_type: str, assert_value: str, test_case: YAMLTestCase) -> Dict:
        """Evaluate a single assertion against the agent response."""
        
        if assert_type == "contains-all":
            # Check if response contains the specified value
            passed = assert_value in agent_response
            details = f"Checking if response contains '{assert_value}': {'✓' if passed else '✗'}"
            
        elif assert_type == "not-contains":
            # Check if response does NOT contain the specified value
            passed = assert_value not in agent_response
            details = f"Checking if response does NOT contain '{assert_value}': {'✓' if passed else '✗'}"
            
        elif assert_type == "llm-rubric":
            # Use LLM to evaluate based on rubric
            if not self.openai_api_key:
                return {"passed": False, "details": "No OpenAI API key for LLM evaluation"}
                
            passed, details = await self._llm_evaluation(agent_response, assert_value)
            
        else:
            return {"passed": False, "details": f"Unknown assertion type: {assert_type}"}
        
        return {"passed": passed, "details": details}

    async def _llm_evaluation(self, agent_response: str, rubric: str) -> tuple[bool, str]:
        """Use LLM to evaluate the agent response against a rubric."""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.openai_api_key)
            
            evaluation_prompt = f"""
You are evaluating an AI agent's response against specific criteria.

AGENT RESPONSE:
{agent_response}

EVALUATION CRITERIA:
{rubric}

Please evaluate whether the agent's response meets the criteria. Respond with exactly "PASS" if it meets the criteria, or "FAIL" if it does not, followed by a brief explanation on the next line.
"""

            response = await client.chat.completions.create(
                model=self.eval_model,
                messages=[
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_tokens=200,
                temperature=0
            )
            
            evaluation_result = response.choices[0].message.content.strip()
            
            # Parse the result
            lines = evaluation_result.split('\n', 1)
            decision = lines[0].strip().upper()
            explanation = lines[1] if len(lines) > 1 else ""
            
            passed = decision == "PASS"
            details = f"LLM evaluation: {decision}. {explanation}"
            
            return passed, details
            
        except Exception as e:
            return False, f"LLM evaluation failed: {str(e)}"

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
            # Run the test case
            result = await self.run_single_test_case(test_case)
            results.append(result)
            
            # Run evaluation if requested and test completed successfully
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
            print(f"Total cost: ${total_cost:.4f}")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Results saved: {summary_file}")
        
        return summary


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
            print(f"  - {case.case_name}: {case.task_description[:100]}...")
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
            print("Available cases:", [c.case_name for c in test_cases])
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