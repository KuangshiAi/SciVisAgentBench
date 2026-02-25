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
        static_screenshot: bool = False,
        agent_mode: Optional[str] = None,
        openai_base_url: Optional[str] = None
    ):
        """
        Initialize the evaluation manager.

        Args:
            eval_mode: Evaluation mode - "mcp", "pvpython", or "generic" (for framework paths)
            openai_api_key: OpenAI API key for LLM evaluation
            eval_model: Model to use for LLM evaluation
            static_screenshot: If True, use pre-generated screenshots instead of generating from state files
            agent_mode: Full agent mode string (e.g., "paraview_mcp_gpt-4o_exp1") for finding result files. If None, uses eval_mode
            openai_base_url: Optional custom OpenAI-compatible API endpoint
        """
        self.eval_mode = eval_mode
        self.agent_mode = agent_mode if agent_mode else eval_mode  # Use agent_mode for results, default to eval_mode
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.eval_model = eval_model
        self.static_screenshot = static_screenshot
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")

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
        if self.eval_mode == "mcp" or self.eval_mode == "generic":
            # Use MCP evaluator for both MCP agents and generic agents (like Claude Code)
            # that generate ParaView state files and screenshots
            from mcp_auto_evaluator import MCPAutoEvaluator
            return MCPAutoEvaluator(
                case_dir=case_dir,
                case_name=case_name,
                openai_api_key=self.openai_api_key,
                model=self.eval_model,
                static_screenshot=self.static_screenshot,
                agent_mode=self.agent_mode,  # Pass agent_mode for finding result files
                openai_base_url=self.openai_base_url  # Pass custom API endpoint
            )
        elif self.eval_mode == "pvpython":
            from pvpython_auto_evaluator import PVPythonAutoEvaluator
            return PVPythonAutoEvaluator(
                case_dir=case_dir,
                case_name=case_name,
                openai_api_key=self.openai_api_key,
                model=self.eval_model,
                static_screenshot=self.static_screenshot,
                agent_mode=self.agent_mode,  # Pass agent_mode for finding result files
                openai_base_url=self.openai_base_url  # Pass custom API endpoint
            )
        else:
            raise ValueError(f"Unsupported eval_mode: {self.eval_mode}")

    async def evaluate_vision(
        self,
        case_dir: str,
        case_name: str,
        vision_rubric: str,
        gs_file: Optional[str] = None,
        rs_file: Optional[str] = None,
        data_dir: Optional[str] = None
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
            gs_file: Optional custom path to ground truth image (relative to data_dir)
            rs_file: Optional custom path to result image (relative to data_dir)
            data_dir: Optional data directory (working directory for the agent)

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

            # Handle custom file paths by copying them to expected locations
            import shutil

            # Determine base directory for resolving custom file paths
            # If data_dir is provided, use it; otherwise fall back to case_dir parent
            base_dir = Path(data_dir) if data_dir else Path(case_dir).parent

            # Override ground truth image path if gs_file is provided
            if gs_file:
                # Replace {agent_mode} placeholder
                gs_file_resolved = gs_file.replace('{agent_mode}', self.eval_mode)
                # Make path absolute (relative to base_dir)
                gs_source = base_dir / gs_file_resolved

                if gs_source.exists():
                    # Copy to expected location: case_dir/GS/{case_name}_gs.png
                    gs_dir = Path(case_dir) / "GS"
                    gs_dir.mkdir(parents=True, exist_ok=True)
                    gs_dest = gs_dir / f"{case_name}_gs.png"
                    shutil.copy2(gs_source, gs_dest)
                    print(f"Copied custom GS file: {gs_source} -> {gs_dest}")
                else:
                    print(f"⚠️  Warning: Custom gs_file not found: {gs_source}")

            # Override result image path if rs_file is provided
            if rs_file:
                # Replace {agent_mode} placeholder
                rs_file_resolved = rs_file.replace('{agent_mode}', self.eval_mode)
                # Make path absolute (relative to base_dir)
                rs_source = base_dir / rs_file_resolved

                if rs_source.exists():
                    # Copy to expected location: case_dir/results/{eval_mode}/{case_name}.png
                    results_dir = Path(case_dir) / "results" / self.eval_mode
                    results_dir.mkdir(parents=True, exist_ok=True)
                    rs_dest = results_dir / f"{case_name}.png"
                    shutil.copy2(rs_source, rs_dest)
                    print(f"Copied custom RS file: {rs_source} -> {rs_dest}")
                else:
                    print(f"⚠️  Warning: Custom rs_file not found: {rs_source}")

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
        text_rubric: str,
        rs_file: Optional[str] = None,
        data_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run text-based evaluation using answers.txt file.

        Args:
            case_dir: Test case directory
            case_name: Test case name
            text_rubric: Text evaluation rubric
            rs_file: Optional custom path to result text file (relative to data_dir)
            data_dir: Optional data directory (working directory for the agent)

        Returns:
            Evaluation result dictionary
        """
        try:
            from openai import OpenAI

            # Load the answers file
            if rs_file:
                # Determine base directory for resolving custom file paths
                base_dir = Path(data_dir) if data_dir else Path(case_dir).parent
                # Use custom path (relative to base_dir)
                rs_file_resolved = rs_file.replace('{agent_mode}', self.eval_mode)
                answers_file = base_dir / rs_file_resolved
            else:
                # Use default path
                answers_file = Path(case_dir) / "results" / self.eval_mode / "answers.txt"

            if not answers_file.exists():
                return {
                    "status": "failed",
                    "subtype": "text",
                    "reason": f"answers file not found: {answers_file}"
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
            client_kwargs = {"api_key": self.openai_api_key}
            if self.openai_base_url:
                client_kwargs["base_url"] = self.openai_base_url
            client = OpenAI(**client_kwargs)

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

    async def evaluate_code(
        self,
        case_dir: str,
        case_name: str,
        code_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run code similarity evaluation.

        Args:
            case_dir: Test case directory
            case_name: Test case name
            code_config: Code evaluation configuration with gs_file and rs_file paths

        Returns:
            Evaluation result dictionary
        """
        try:
            print("Evaluating code similarity...")

            # Get evaluator
            evaluator = self.get_evaluator_for_case(case_dir, case_name)

            # Parse file paths from config
            gs_files = code_config.get('gs_file', [])
            rs_files = code_config.get('rs_file', [])

            # Default paths if not specified
            if not gs_files:
                gs_files = [f"{case_name}/GS/{case_name}_gs.py"]
            if not rs_files:
                rs_files = [f"{case_name}/results/{self.eval_mode}/{case_name}.py"]

            # Get the first file (for now, we only compare one pair)
            gs_file = gs_files[0] if gs_files else None
            rs_file = rs_files[0] if rs_files else None

            if not gs_file or not rs_file:
                return {
                    "status": "failed",
                    "subtype": "code",
                    "reason": "Ground truth or result file path not specified"
                }

            # Replace {agent_mode} placeholder
            gs_file = gs_file.replace('{agent_mode}', self.eval_mode)
            rs_file = rs_file.replace('{agent_mode}', self.eval_mode)

            # Make paths absolute
            gs_path = Path(case_dir).parent / gs_file
            rs_path = Path(case_dir).parent / rs_file

            # Override evaluator's default paths
            evaluator.gs_code_path = str(gs_path)
            evaluator.generated_code_path = str(rs_path)

            # Run code similarity evaluation
            code_score = evaluator.evaluate_code_similarity()

            # Get the score details
            scores = evaluator.evaluation_results.get("scores", {})
            code_details = scores.get("code_similarity", {})

            # Scale from 20 to 10 points for consistency
            original_score = code_details.get("score", 0)
            original_max = code_details.get("max_score", 20)
            scaled_score = int((original_score / original_max) * 10) if original_max > 0 else 0
            scaled_max = 10

            similarity_raw = code_details.get("similarity_raw_score", 0)
            explanation = f"Code similarity: {similarity_raw:.3f} (scaled to {scaled_score}/10 points)"

            return {
                "status": "completed",
                "subtype": "code",
                "gs_file": str(gs_path),
                "rs_file": str(rs_path),
                "scores": {
                    "code_similarity": scaled_score,
                    "total_score": scaled_score,
                    "max_possible_score": scaled_max,
                    "percentage": (scaled_score / scaled_max * 100) if scaled_max > 0 else 0
                },
                "detailed_scores": {
                    "code_similarity": {
                        "score": scaled_score,
                        "max_score": scaled_max,
                        "explanation": explanation,
                        "similarity_raw_score": similarity_raw
                    }
                },
                "evaluator_metadata": {
                    "evaluator_type": f"{self.eval_mode}_code_similarity",
                    "evaluator_version": "1.0.0",
                    "scoring_scheme": {
                        "code_similarity": "10 points",
                        "total_possible": "10 points"
                    }
                }
            }

        except Exception as e:
            print(f"❌ Code evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "failed",
                "subtype": "code",
                "reason": str(e),
                "traceback": traceback.format_exc()
            }

    async def run_evaluation(
        self,
        case_dir: str,
        case_name: str,
        evaluation_subtypes: List[str],
        rubrics: Dict[str, str],
        file_configs: Optional[Dict[str, Dict[str, str]]] = None,
        data_dir: Optional[str] = None
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
            file_configs: Optional dictionary mapping subtype to file paths (gs_file, rs_file)
            data_dir: Optional data directory (working directory for the agent)

        Returns:
            Comprehensive evaluation result dictionary
        """
        if file_configs is None:
            file_configs = {}
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
                # Get file config for vision evaluation
                vision_config = file_configs.get('vision', {})
                gs_file = vision_config.get('gs_file')
                rs_file = vision_config.get('rs_file')
                result = await self.evaluate_vision(
                    case_dir,
                    case_name,
                    rubrics.get('vision', ''),
                    gs_file=gs_file,
                    rs_file=rs_file,
                    data_dir=data_dir
                )
            elif subtype == 'text':
                # Get file config for text evaluation
                text_config = file_configs.get('text', {})
                rs_file = text_config.get('rs_file')
                result = await self.evaluate_text(
                    case_dir,
                    case_name,
                    rubrics.get('text', ''),
                    rs_file=rs_file,
                    data_dir=data_dir
                )
            elif subtype == 'code':
                result = await self.evaluate_code(
                    case_dir,
                    case_name,
                    rubrics.get('code', {})
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
                elif subtype == 'code' and 'scores' in result:
                    code_sim = result['scores'].get('code_similarity', 0)
                    similarity_raw = result.get('detailed_scores', {}).get('code_similarity', {}).get('similarity_raw_score', 0)
                    print(f"✅ {subtype} evaluation completed: {subtype_score}/{subtype_max}")
                    print(f"   - Code similarity: {code_sim}/10 (raw: {similarity_raw:.3f})")
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
