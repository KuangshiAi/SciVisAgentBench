#!/usr/bin/env python3
"""
Test Runner for TinyAgent Cases

This script loads test cases from a directory structure and runs them through 
the TinyAgent using a specified configuration file.
"""

import asyncio
import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import json
import time
from datetime import datetime
import tiktoken

from agent import TinyAgent


class TokenCounter:
    """Utility class for counting tokens using GPT-4o tokenizer."""
    
    def __init__(self):
        """Initialize the tokenizer for GPT-4o."""
        try:
            # GPT-4o uses the same tokenizer as GPT-4
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            print(f"Warning: Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        if self.tokenizer is None:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"Warning: Failed to count tokens: {e}")
            return 0


class TestCase:
    """Represents a single test case."""
    
    def __init__(self, case_path: str, config_path: Optional[str] = None):
        self.case_path = Path(case_path)
        self.name = self.case_path.name
        self.task_description_file = self.case_path / "task_description.txt"
        self.results_dir = self.case_path / "results" / "mcp"
        self.evaluation_dir = self.case_path / "test_results" / "mcp"
        self.config_path = config_path
        
    def is_valid(self) -> bool:
        """Check if this is a valid test case (has task_description.txt)."""
        return self.task_description_file.exists()
    
    def get_task_description(self) -> str:
        """Read and return the task description."""
        if not self.is_valid():
            raise FileNotFoundError(f"Task description not found: {self.task_description_file}")
        
        working_dir = self.case_path.parent
        
        # Read original task description
        with open(self.task_description_file, 'r', encoding='utf-8') as f:
            original_task = f.read().strip()
        
        # Prepend working directory information
        working_dir_info = f'Your agent_mode is "mcp", use it when saving results. Your working directory is "{working_dir}", and you should have access to it. In the following prompts, we will use relative path with respect to your working path. But remember, when you load or save any file, always stick to absolute path.'
        
        return f"{working_dir_info}\n\n{original_task}"
    
    def ensure_directories(self):
        """Ensure results and evaluation directories exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)


class TestRunner:
    """Runs test cases through TinyAgent."""
    
    def __init__(self, config_path: str, cases_path: str, output_dir: Optional[str] = None):
        # Convert config_path to absolute path to avoid issues with relative paths
        if not os.path.isabs(config_path):
            self.config_path = os.path.abspath(config_path)
        else:
            self.config_path = config_path
            
        self.cases_path = Path(cases_path)
        self.output_dir = Path(output_dir) if output_dir else self.cases_path.parent / "test_results" / "mcp"
        self.test_cases: List[TestCase] = []
        self.token_counter = TokenCounter()
        
        # Load pricing information from config
        self.pricing_info = self._load_pricing_info()
    
    def _load_pricing_info(self) -> Optional[Dict]:
        """Load pricing information from the config file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get("price")
        except Exception as e:
            print(f"Warning: Could not load pricing info from config: {e}")
            return None
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> Optional[Dict]:
        """Calculate cost based on token usage and pricing info."""
        if not self.pricing_info:
            return None
        
        try:
            # Parse pricing strings (e.g., "$2.50" -> 2.50)
            input_price_str = self.pricing_info.get("input_per_1m_tokens", "").replace("$", "")
            output_price_str = self.pricing_info.get("output_per_1m_tokens", "").replace("$", "")
            
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
        
        # Load pricing information from config
        self.pricing_info = self._load_pricing_info()
        
    def discover_test_cases(self) -> List[TestCase]:
        """Discover all test cases in the cases directory."""
        test_cases = []
        
        if not self.cases_path.exists():
            raise FileNotFoundError(f"Cases directory not found: {self.cases_path}")
        
        for item in self.cases_path.iterdir():
            if item.is_dir():
                test_case = TestCase(item, self.config_path)
                if test_case.is_valid():
                    test_cases.append(test_case)
                    print(f"Found test case: {test_case.name}")
                else:
                    print(f"Skipping invalid test case: {test_case.name} (no task_description.txt)")
        
        self.test_cases = test_cases
        return test_cases
    
    def _create_test_case_config(self, test_case: TestCase) -> str:
        """Create a unique config file for each test case with case-specific session."""
        # Load the original config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        working_dir = config.get("working_dir")
        if not working_dir:
            raise ValueError("No 'working_dir' specified in config file")
        
        # Create a temporary config file for this test case
        temp_config_dir = Path(tempfile.gettempdir()) / "scivisagent_bench_configs"
        temp_config_dir.mkdir(exist_ok=True)
        
        case_config_file = temp_config_dir / f"config_{test_case.name}_{int(time.time())}.json"
        
        # Modify the config for this specific test case
        config_modified = False
        for server in config.get("servers", []):
            if (server.get("type") == "stdio" and 
                server.get("config", {}).get("command") == "npx" and
                any("@modelcontextprotocol/server-filesystem" in arg for arg in server.get("config", {}).get("args", []))):
                
                # Add working directory to args if not already present
                if working_dir not in server["config"]["args"]:
                    server["config"]["args"].append(working_dir)
                    config_modified = True
        
        # Add test case name as environment variable to mcp_logger for unique session naming
        for server in config.get("servers", []):
            if (server.get("type") == "stdio" and 
                "mcp_logger.py" in str(server.get("config", {}).get("args", []))):
                
                # Add environment variable to distinguish test cases
                if "env" not in server["config"]:
                    server["config"]["env"] = {}
                server["config"]["env"]["SCIVISAGENT_TEST_CASE"] = test_case.name
                config_modified = True
        
        # Save the modified config to the temporary file
        with open(case_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"Created test case config: {case_config_file}")
        return str(case_config_file)
    
    async def run_single_test_case(self, test_case: TestCase) -> Dict:
        """Run a single test case and return results."""
        print(f"\n{'='*60}")
        print(f"Running test case: {test_case.name}")
        print(f"{'='*60}")
        
        # Ensure directories exist
        test_case.ensure_directories()
        
        # Get task description
        try:
            task_description = test_case.get_task_description()
            print(f"Task description:\n{task_description}\n")
        except Exception as e:
            return {
                "case_name": test_case.name,
                "status": "error",
                "error": f"Failed to read task description: {e}",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": 0,
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            }
        
        # Create agent from config
        start_time = datetime.now()
        
        # Initialize token counting
        input_tokens = self.token_counter.count_tokens(task_description)
        
        result = {
            "case_name": test_case.name,
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
            # Create a unique config for this test case
            test_case_config_path = self._create_test_case_config(test_case)
            agent = TinyAgent.from_config_file(test_case_config_path)
            
            async with agent:
                await agent.load_tools()
                print(f"Agent loaded with {len(agent.available_tools)} tools")
                
                # Clear ParaView state at the beginning of each test case
                print("Clearing ParaView state for fresh start...")
                await self._clear_paraview_state(agent)
                
                print(f"Starting execution...\n")
                
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
                result["response"] = full_response
                
                # Count output tokens
                output_tokens = self.token_counter.count_tokens(full_response)
                result["token_usage"]["output_tokens"] = output_tokens
                result["token_usage"]["total_tokens"] = result["token_usage"]["input_tokens"] + output_tokens
                
                # Calculate cost if pricing info is available
                cost_info = self._calculate_cost(result["token_usage"]["input_tokens"], output_tokens)
                if cost_info:
                    result["cost_info"] = cost_info
                
                result["status"] = "completed"
            
            # Clean up the temporary config file
            try:
                os.unlink(test_case_config_path)
            except OSError:
                pass  # Ignore cleanup errors
                
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"\nError running test case {test_case.name}: {e}")
            import traceback
            traceback.print_exc()
        
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration"] = (end_time - start_time).total_seconds()
        
        # Save individual result
        await self.save_test_result(test_case, result)
        
        # Print completion summary including token usage
        print(f"\nTest case {test_case.name} completed in {result['duration']:.2f} seconds")
        if "token_usage" in result:
            token_usage = result["token_usage"]
            print(f"Token usage - Input: {token_usage['input_tokens']}, Output: {token_usage['output_tokens']}, Total: {token_usage['total_tokens']}")
        if "cost_info" in result:
            cost_info = result["cost_info"]
            print(f"Cost - Input: ${cost_info['input_cost']:.6f}, Output: ${cost_info['output_cost']:.6f}, Total: ${cost_info['total_cost']:.6f}")
        return result
    
    async def save_test_result(self, test_case: TestCase, result: Dict):
        """Save the result of a test case."""
        result_file = test_case.evaluation_dir / f"test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Result saved to: {result_file}")
            
            # Print token usage summary if available
            if "token_usage" in result:
                token_usage = result["token_usage"]
                print(f"Token usage saved - Input: {token_usage['input_tokens']}, Output: {token_usage['output_tokens']}, Total: {token_usage['total_tokens']}")
            
            # Print cost summary if available
            if "cost_info" in result:
                cost_info = result["cost_info"]
                print(f"Cost saved - Input: ${cost_info['input_cost']:.6f}, Output: ${cost_info['output_cost']:.6f}, Total: ${cost_info['total_cost']:.6f}")
                
        except Exception as e:
            print(f"Failed to save result: {e}")
    
    async def run_all_test_cases(self) -> Dict:
        """Run all discovered test cases."""
        if not self.test_cases:
            self.discover_test_cases()
        
        if not self.test_cases:
            print("No test cases found!")
            return {"results": [], "summary": {"total": 0, "completed": 0, "errors": 0}}
        
        print(f"Found {len(self.test_cases)} test cases to run")
        print(f"Using config: {self.config_path}")
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        overall_start_time = datetime.now()
        results = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n{'='*80}")
            print(f"STARTING TEST CASE {i}/{len(self.test_cases)}: {test_case.name}")
            print(f"{'='*80}")
            print(f"Each test case will get its own communication log and fresh ParaView state.")
            
            result = await self.run_single_test_case(test_case)
            results.append(result)
            
            print(f"\n{'='*80}")
            print(f"COMPLETED TEST CASE {i}/{len(self.test_cases)}: {test_case.name}")
            print(f"Status: {result['status']}")
            if result['status'] == 'error':
                print(f"Error: {result.get('error', 'Unknown error')}")
            print(f"Duration: {result['duration']:.2f} seconds")
            print(f"{'='*80}\n")
            
            # Small delay between test cases to ensure clean separation
            if i < len(self.test_cases):
                print("Waiting 2 seconds before starting next test case...")
                await asyncio.sleep(2)
        
        overall_end_time = datetime.now()
        overall_duration = (overall_end_time - overall_start_time).total_seconds()
        
        # Summary
        completed = sum(1 for r in results if r["status"] == "completed")
        errors = sum(1 for r in results if r["status"] == "error")
        
        # Calculate total token usage
        total_input_tokens = sum(r.get("token_usage", {}).get("input_tokens", 0) for r in results)
        total_output_tokens = sum(r.get("token_usage", {}).get("output_tokens", 0) for r in results)
        total_tokens = total_input_tokens + total_output_tokens
        
        # Calculate total cost
        total_cost_info = self._calculate_cost(total_input_tokens, total_output_tokens)
        
        summary = {
            "total": len(results),
            "completed": completed,
            "errors": errors,
            "start_time": overall_start_time.isoformat(),
            "end_time": overall_end_time.isoformat(),
            "duration": overall_duration,
            "token_usage": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens
            }
        }
        
        # Add cost summary if available
        if total_cost_info:
            summary["cost_summary"] = total_cost_info
        
        overall_result = {
            "results": results,
            "summary": summary
        }
        
        # Save overall results
        overall_result_file = self.output_dir / f"test_run_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(overall_result_file, 'w', encoding='utf-8') as f:
            json.dump(overall_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("TEST RUN SUMMARY")
        print(f"{'='*60}")
        print(f"Total test cases: {summary['total']}")
        print(f"Completed: {summary['completed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Total duration: {summary['duration']:.2f} seconds")
        print(f"Token usage - Input: {summary['token_usage']['total_input_tokens']}, Output: {summary['token_usage']['total_output_tokens']}, Total: {summary['token_usage']['total_tokens']}")
        if "cost_summary" in summary:
            cost_summary = summary["cost_summary"]
            print(f"Total cost - Input: ${cost_summary['input_cost']:.6f}, Output: ${cost_summary['output_cost']:.6f}, Total: ${cost_summary['total_cost']:.6f}")
        print(f"Overall results saved to: {overall_result_file}")
        
        return overall_result
    
    async def _clear_paraview_state(self, agent) -> None:
        """Clear ParaView state to ensure a fresh start for each test case."""
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
            print(f"Warning: Could not clear ParaView state through MCP: {e}")
            print("Test case will continue, but may have residual state from previous runs")


async def main():
    parser = argparse.ArgumentParser(description="Run test cases through TinyAgent")
    parser.add_argument("--config", "-c", required=True, 
                       help="Path to the configuration JSON file")
    parser.add_argument("--cases", required=True,
                       help="Path to the cases directory")
    parser.add_argument("--output", "-o", 
                       help="Output directory for results (default: cases_parent/test_results)")
    parser.add_argument("--case", 
                       help="Run a specific test case by name")
    parser.add_argument("--list", action="store_true",
                       help="List available test cases and exit")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    if not os.path.exists(args.cases):
        print(f"Error: Cases directory not found: {args.cases}")
        return 1
    
    runner = TestRunner(args.config, args.cases, args.output)
    
    # Discover test cases
    test_cases = runner.discover_test_cases()
    
    if args.list:
        print(f"\nAvailable test cases in {args.cases}:")
        for test_case in test_cases:
            print(f"  - {test_case.name}")
        return 0
    
    if not test_cases:
        print("No valid test cases found!")
        return 1
    
    # Run specific case or all cases
    if args.case:
        # Find and run specific case
        target_case = None
        for test_case in test_cases:
            if test_case.name == args.case:
                target_case = test_case
                break
        
        if not target_case:
            print(f"Error: Test case '{args.case}' not found")
            print("Available test cases:")
            for test_case in test_cases:
                print(f"  - {test_case.name}")
            return 1
        
        result = await runner.run_single_test_case(target_case)
        return 0 if result["status"] == "completed" else 1
    else:
        # Run all cases
        results = await runner.run_all_test_cases()
        return 0 if results["summary"]["errors"] == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
