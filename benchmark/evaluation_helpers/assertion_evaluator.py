"""
Assertion-Based Evaluator

Evaluates agent responses against assertion-based test cases.
Supports assertion types:
- contains-all: Check if response contains specified value(s)
- not-contains: Check if response does NOT contain specified value(s)
- llm-rubric: Use LLM to evaluate response against a rubric

This evaluator is designed to work with both MCP and PVPython eval modes.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class AssertionEvaluator:
    """Evaluates agent responses using assertion-based validation."""

    def __init__(
        self,
        case_dir: str,
        case_name: str,
        eval_mode: str = "mcp",
        openai_api_key: Optional[str] = None,
        eval_model: str = "gpt-4o"
    ):
        """
        Initialize the assertion evaluator.

        Args:
            case_dir: Test case directory
            case_name: Test case name
            eval_mode: Evaluation mode - "mcp" or "pvpython"
            openai_api_key: OpenAI API key for LLM evaluation
            eval_model: Model to use for LLM evaluation
        """
        self.case_dir = Path(case_dir)
        self.case_name = case_name
        self.eval_mode = eval_mode
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.eval_model = eval_model

        # Paths
        self.test_results_dir = self.case_dir / "test_results" / self.eval_mode
        self.evaluation_dir = self.case_dir / "evaluation_results" / self.eval_mode

    def get_agent_response(self) -> Optional[str]:
        """
        Load the agent's response from the most recent test result.

        Returns:
            Agent's response text, or None if not found
        """
        if not self.test_results_dir.exists():
            print(f"Test results directory not found: {self.test_results_dir}")
            return None

        # Find the most recent test result file
        result_files = list(self.test_results_dir.glob("test_result_*.json"))
        if not result_files:
            print(f"No test result files found in {self.test_results_dir}")
            return None

        latest_result_file = max(result_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_result_file, 'r', encoding='utf-8') as f:
                test_result = json.load(f)

            # Try to get assistant_response (without tool logs) first
            full_result_data = test_result.get('full_result', {})
            agent_response = full_result_data.get('assistant_response')

            # Fall back to full response
            if not agent_response:
                agent_response = full_result_data.get('response', '')

            # Try top level for backward compatibility
            if not agent_response:
                agent_response = test_result.get('assistant_response') or test_result.get('response', '')

            return agent_response

        except Exception as e:
            print(f"Failed to load test result: {e}")
            return None

    async def evaluate_assertion(
        self,
        agent_response: str,
        assertion_type: str,
        assertion_value: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a single assertion against the agent response.

        Args:
            agent_response: The agent's response text
            assertion_type: Type of assertion (contains-all, not-contains, llm-rubric)
            assertion_value: Value to check against

        Returns:
            Dictionary with 'passed', 'score', and 'details' keys
        """
        if assertion_type == "contains-all":
            # Check if response contains the specified value(s)
            if isinstance(assertion_value, list):
                # Check if ALL values are present in the response
                passed = all(str(val) in agent_response for val in assertion_value)
                details = f"Checking if response contains all of {assertion_value}: {'✓' if passed else '✗'}"
            else:
                # Single value check
                passed = str(assertion_value) in agent_response
                details = f"Checking if response contains '{assertion_value}': {'✓' if passed else '✗'}"

        elif assertion_type == "not-contains":
            # Check if response does NOT contain the specified value(s)
            if isinstance(assertion_value, list):
                # Check if NONE of the values are present in the response
                passed = all(str(val) not in agent_response for val in assertion_value)
                details = f"Checking if response contains none of {assertion_value}: {'✓' if passed else '✗'}"
            else:
                # Single value check
                passed = str(assertion_value) not in agent_response
                details = f"Checking if response does NOT contain '{assertion_value}': {'✓' if passed else '✗'}"

        elif assertion_type == "llm-rubric":
            # Use LLM to evaluate based on rubric
            if not self.openai_api_key:
                return {
                    "passed": False,
                    "score": 0,
                    "details": "No OpenAI API key for LLM evaluation"
                }

            passed, details = await self._llm_evaluation(agent_response, assertion_value)

        else:
            return {
                "passed": False,
                "score": 0,
                "details": f"Unknown assertion type: {assertion_type}"
            }

        return {
            "passed": passed,
            "score": 1 if passed else 0,
            "details": details
        }

    async def _llm_evaluation(self, agent_response: str, rubric: str) -> tuple:
        """
        Use LLM to evaluate the agent response against a rubric.

        Args:
            agent_response: The agent's response
            rubric: Evaluation rubric

        Returns:
            Tuple of (passed: bool, details: str)
        """
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.openai_api_key)

            evaluation_prompt = f"""You are evaluating an AI agent's response against specific criteria.

AGENT RESPONSE:
{agent_response}

EVALUATION CRITERIA:
{rubric}

Please evaluate whether the agent's response meets the criteria. Respond with exactly "PASS" if it meets the criteria, or "FAIL" if it does not, followed by a brief explanation on the next line."""

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

    async def evaluate_assertions(self, assertions: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate multiple assertions against the agent response.

        Args:
            assertions: List of assertion dictionaries with 'type' and 'value' keys

        Returns:
            Evaluation result dictionary
        """
        # Get agent response
        agent_response = self.get_agent_response()
        if not agent_response:
            return {
                "status": "failed",
                "reason": "No agent response found"
            }

        # Check if this is a <1>/<0> binary response pattern
        is_binary_response = False
        if len(assertions) == 2:
            types = [a.get('type', '') for a in assertions]
            values = [a.get('value', '') for a in assertions]
            if ('contains-all' in types and 'not-contains' in types and
                '<1>' in values and '<0>' in values):
                is_binary_response = True

        evaluation_results = []
        total_passed = 0
        total_score = 0

        if is_binary_response:
            # Handle <1>/<0> pattern: only check if response contains <1>
            print(f"  Detected binary <1>/<0> pattern - evaluating as single check")

            agent_response_stripped = agent_response.strip()

            if '<1>' in agent_response_stripped:
                passed = True
                status = "passed"
                details = "Response contains <1> (success)"
                score = 10  # Score 10 for pass (matching other test case scoring)
            elif '<0>' in agent_response_stripped:
                passed = False
                status = "failed"
                details = "Response contains <0> (failure)"
                score = 0
            else:
                passed = False
                status = "invalid"
                details = f"Response is neither <1> nor <0> (invalid response)"
                score = 0

            evaluation_results.append({
                "assertion_index": 0,
                "type": "binary_response",
                "value": "<1> for pass, <0> for fail",
                "passed": passed,
                "score": score,
                "details": details,
                "status": status
            })

            if passed:
                total_passed += 1
            total_score += score
        else:
            # Original logic for other assertion types
            for i, assertion in enumerate(assertions):
                assert_type = assertion.get('type', '')
                assert_value = assertion.get('value', '')

                print(f"  Evaluating assertion {i+1}: {assert_type}")

                result = await self.evaluate_assertion(agent_response, assert_type, assert_value)
                evaluation_results.append({
                    "assertion_index": i,
                    "type": assert_type,
                    "value": assert_value,
                    "passed": result["passed"],
                    "score": result.get("score", 1 if result["passed"] else 0),
                    "details": result.get("details", "")
                })

                if result["passed"]:
                    total_passed += 1
                total_score += result.get("score", 1 if result["passed"] else 0)

        # For binary responses, we treat it as 1 assertion (not 2)
        effective_assertions = 1 if is_binary_response else len(assertions)
        # Binary responses are worth 10 points (matching other test cases), non-binary worth 1 point each
        max_score_per_assertion = 10 if is_binary_response else 1
        max_possible_score = effective_assertions * max_score_per_assertion

        final_result = {
            "status": "completed",
            "case_name": self.case_name,
            "model": self.eval_model if self.eval_model else "rule-based",
            "agent_response": agent_response,
            "assertion_results": evaluation_results,
            "scores": {
                "total_score": total_score,
                "total_passed": total_passed,
                "total_assertions": effective_assertions,
                "max_possible_score": max_possible_score,
                "pass_rate": total_passed / effective_assertions if effective_assertions else 0,
                "average_score": total_score / effective_assertions if effective_assertions else 0
            },
            # Top-level score field for easy access (binary: 10 if passed, 0 if failed; non-binary: 1 if all passed, 0 otherwise)
            "score": total_score if is_binary_response else (1 if total_passed == effective_assertions else 0),
            "timestamp": datetime.now().isoformat()
        }

        # Save evaluation result
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        eval_file = self.evaluation_dir / f"evaluation_result_{int(datetime.now().timestamp())}.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

        print(f"✅ Evaluation completed for {self.case_name}")
        print(f"Passed: {total_passed}/{effective_assertions} ({final_result['scores']['pass_rate']:.1%})")
        print(f"Score: {final_result['score']} (Total: {total_score}/{max_possible_score})")

        return final_result
