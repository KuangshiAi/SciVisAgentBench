"""
Evaluation Manager

Orchestrates the evaluation process using the existing evaluator classes
(SciVisEvaluator, MCPAutoEvaluator, PVPythonAutoEvaluator, LLMEvaluator).
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Add evaluation_helpers to path
evaluation_helpers_path = Path(__file__).parent.parent / "evaluation_helpers"
if str(evaluation_helpers_path) not in sys.path:
    sys.path.insert(0, str(evaluation_helpers_path))

# Don't import evaluators at module level - they import paraview
# which may not be available. Import them lazily when needed.


class EvaluationManager:
    """
    Manages evaluation of test cases using the appropriate evaluator.

    This class serves as a bridge between the evaluation framework and the
    existing evaluator classes, preserving all their functionality.
    """

    def __init__(
        self,
        eval_mode: str = "mcp",
        openai_api_key: Optional[str] = None,
        eval_model: str = "gpt-4o",
        static_screenshot: bool = False
    ):
        """
        Initialize the evaluation manager.

        Args:
            eval_mode: Evaluation mode - "mcp", "pvpython", or "generic"
            openai_api_key: OpenAI API key for LLM evaluation
            eval_model: Model to use for LLM evaluation
            static_screenshot: If True, use pre-generated screenshots instead of generating from state files
        """
        self.eval_mode = eval_mode
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.eval_model = eval_model
        self.static_screenshot = static_screenshot

    def get_evaluator_for_case(
        self,
        case_dir: str,
        case_name: str,
        evaluation_subtypes: Optional[List[str]] = None
    ):
        """
        Get the appropriate evaluator for a test case.

        Args:
            case_dir: Test case directory
            case_name: Test case name
            evaluation_subtypes: List of evaluation subtypes (e.g., ["vision", "text"])

        Returns:
            Evaluator instance (MCPAutoEvaluator or PVPythonAutoEvaluator)
        """
        # Lazy import to avoid loading paraview at module import time
        if self.eval_mode == "mcp":
            from mcp_auto_evaluator import MCPAutoEvaluator
            return MCPAutoEvaluator(
                case_dir=case_dir,
                case_name=case_name,
                openai_api_key=self.openai_api_key,
                model=self.eval_model,
                static_screenshot=self.static_screenshot
            )
        elif self.eval_mode == "pvpython":
            from pvpython_auto_evaluator import PVPythonAutoEvaluator
            return PVPythonAutoEvaluator(
                case_dir=case_dir,
                case_name=case_name,
                openai_api_key=self.openai_api_key,
                model=self.eval_model,
                static_screenshot=self.static_screenshot
            )
        else:
            raise ValueError(f"Unsupported eval_mode: {self.eval_mode}")

    async def evaluate_vision(
        self,
        case_dir: str,
        case_name: str,
        vision_rubric: str
    ) -> Dict[str, Any]:
        """
        Run vision-based evaluation using the appropriate evaluator.

        This preserves all features from MCPAutoEvaluator/PVPythonAutoEvaluator including:
        - Screenshot comparison
        - Image metrics (PSNR, SSIM, LPIPS)
        - LLM-based visual quality assessment
        - Support for static screenshots or state file generation

        Args:
            case_dir: Test case directory
            case_name: Test case name
            vision_rubric: Vision evaluation rubric

        Returns:
            Evaluation result dictionary
        """
        try:
            # Write vision rubric to visualization_goals.txt
            goals_file = Path(case_dir) / "visualization_goals.txt"
            with open(goals_file, 'w', encoding='utf-8') as f:
                f.write(vision_rubric)

            # Get the appropriate evaluator
            evaluator = self.get_evaluator_for_case(case_dir, case_name)

            # Run individual evaluation components
            # (Don't use run_evaluation() as it looks for JSON rubric files)

            # Get goals count for scoring
            goals_count = evaluator.get_goals_count()
            print(f"DEBUG: Goals count from evaluator: {goals_count}")

            # Run visualization quality evaluation
            viz_score = evaluator.evaluate_visualization_quality()

            # Run output generation evaluation
            output_score = evaluator.evaluate_output_generation()

            # Run efficiency evaluation
            efficiency_score = evaluator.evaluate_efficiency()

            # Calculate image metrics
            image_metrics = evaluator.evaluate_image_metrics()

            # Calculate total scores
            total_score = viz_score + output_score + efficiency_score
            max_possible_score = (goals_count * 10) + 5 + 10  # viz(goals*10) + output(5) + efficiency(10)

            scores = evaluator.evaluation_results.get("scores", {})

            return {
                "status": "completed",
                "subtype": "vision",
                "rubric": vision_rubric,
                "goals_count": goals_count,
                "scores": {
                    "visualization_quality": viz_score,
                    "output_generation": output_score,
                    "efficiency": efficiency_score,
                    "total_score": total_score,
                    "max_possible_score": max_possible_score,
                    "percentage": (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
                },
                "image_metrics": image_metrics,
                "detailed_scores": scores,
                "evaluator_metadata": {
                    "evaluator_type": "framework_mcp_auto_vision",
                    "evaluator_version": "1.0.0",
                    "goals_count": goals_count,
                    "scoring_scheme": {
                        "visualization_quality": f"{goals_count * 10} points (10 per goal)",
                        "output_generation": "5 points (file existence)",
                        "efficiency": "10 points (execution metrics)",
                        "total_possible": f"{max_possible_score} points"
                    }
                }
            }

        except Exception as e:
            print(f"❌ Vision evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "failed",
                "subtype": "vision",
                "reason": str(e),
                "traceback": traceback.format_exc()
            }

    async def evaluate_text(
        self,
        case_dir: str,
        case_name: str,
        text_rubric: str
    ) -> Dict[str, Any]:
        """
        Run text-based evaluation using answers.txt file.

        Args:
            case_dir: Test case directory
            case_name: Test case name
            text_rubric: Text evaluation rubric

        Returns:
            Evaluation result dictionary
        """
        try:
            from openai import OpenAI

            # Load the answers.txt file
            answers_file = Path(case_dir) / "results" / self.eval_mode / "answers.txt"
            if not answers_file.exists():
                return {
                    "status": "failed",
                    "subtype": "text",
                    "reason": f"answers.txt not found: {answers_file}"
                }

            with open(answers_file, 'r', encoding='utf-8') as f:
                answers_content = f.read().strip()

            if not answers_content:
                return {
                    "status": "failed",
                    "subtype": "text",
                    "reason": "answers.txt is empty"
                }

            if not text_rubric:
                return {
                    "status": "failed",
                    "subtype": "text",
                    "reason": "No text rubric provided"
                }

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
            try:
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = response_content[json_start:json_end]
                    evaluation_result = json.loads(json_content)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                return {
                    "status": "failed",
                    "subtype": "text",
                    "reason": f"Failed to parse evaluation: {e}"
                }

            score = evaluation_result.get('score', 0)
            explanation = evaluation_result.get('explanation', 'No explanation provided')

            return {
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

        except Exception as e:
            return {
                "status": "failed",
                "subtype": "text",
                "reason": str(e)
            }

    async def run_evaluation(
        self,
        case_dir: str,
        case_name: str,
        evaluation_subtypes: List[str],
        rubrics: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation for a test case with support for multiple subtypes.

        This method preserves all functionality from the existing yaml_runner files,
        supporting both vision and text evaluation.

        Args:
            case_dir: Test case directory
            case_name: Test case name
            evaluation_subtypes: List of evaluation subtypes (e.g., ["vision", "text"])
            rubrics: Dictionary mapping subtype to rubric text

        Returns:
            Comprehensive evaluation result dictionary
        """
        if not self.openai_api_key:
            return {
                "status": "skipped",
                "reason": "No OpenAI API key provided"
            }

        print(f"🔍 Evaluating test case: {case_name}")
        print(f"Available evaluation subtypes: {evaluation_subtypes}")

        evaluation_results = {}
        total_score = 0
        max_possible_score = 0

        # Evaluate each subtype
        for subtype in evaluation_subtypes:
            print(f"Running {subtype} evaluation...")

            if subtype == 'vision':
                result = await self.evaluate_vision(
                    case_dir,
                    case_name,
                    rubrics.get('vision', '')
                )
            elif subtype == 'text':
                result = await self.evaluate_text(
                    case_dir,
                    case_name,
                    rubrics.get('text', '')
                )
            else:
                print(f"⚠️  Unknown evaluation subtype: {subtype}")
                continue

            if result and result.get('status') == 'completed':
                evaluation_results[subtype] = result
                subtype_score = result.get('scores', {}).get('total_score', 0)
                subtype_max = result.get('scores', {}).get('max_possible_score', 0)
                total_score += subtype_score
                max_possible_score += subtype_max

                # Print breakdown for vision evaluation
                if subtype == 'vision' and 'scores' in result:
                    viz_qual = result['scores'].get('visualization_quality', 0)
                    output_gen = result['scores'].get('output_generation', 0)
                    efficiency = result['scores'].get('efficiency', 0)
                    goals = result.get('goals_count', 0)
                    print(f"✅ {subtype} evaluation completed: {subtype_score}/{subtype_max}")
                    print(f"   - Visualization quality: {viz_qual}/{goals * 10}")
                    print(f"   - Output generation: {output_gen}/5")
                    print(f"   - Efficiency: {efficiency}/10")
                else:
                    print(f"✅ {subtype} evaluation completed: {subtype_score}/{subtype_max}")
            else:
                print(f"⚠️  {subtype} evaluation did not complete successfully")
                print(f"   Status: {result.get('status') if result else 'None'}")
                print(f"   Reason: {result.get('reason') if result else 'No result returned'}")
                if result and result.get('status') == 'failed':
                    evaluation_results[subtype] = result  # Still store failed result for debugging

        # Create comprehensive evaluation result
        from datetime import datetime
        import time

        final_result = {
            "status": "completed",
            "case_name": case_name,
            "model": self.eval_model,
            "evaluation_subtypes": evaluation_subtypes,
            "subtype_results": evaluation_results,
            "scores": {
                "total_score": total_score,
                "max_possible_score": max_possible_score,
                "percentage": (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }

        # Save evaluation result
        eval_dir = Path(case_dir) / "evaluation_results" / self.eval_mode
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_file = eval_dir / f"evaluation_result_{int(time.time())}.json"

        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

        print(f"✅ Evaluation completed for {case_name}")
        print(f"Total Score: {total_score}/{max_possible_score} ({final_result['scores']['percentage']:.1f}%)")

        return final_result
