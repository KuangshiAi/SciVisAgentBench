#!/usr/bin/env python3
"""
Automatic pvpython Evaluation Script

This script automatically evaluates all test cases in the benchmark for pvpython mode.
It inherits from SciVisEvaluator and uses both LLM judge and code similarity comparison.
"""
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add the evaluation_helpers directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sciviz_evaluator import SciVisEvaluator
from llm_evaluator import LLMEvaluator
from screenshot_helper import compare_states_screenshots
from image_metrics_helper import CaseImageMetrics


class PVPythonAutoEvaluator(SciVisEvaluator):
    """
    Automatic evaluator for pvpython test cases using LLM judge and code comparison
    """
    
    def __init__(self, case_dir: str, case_name: str, openai_api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize the pvpython evaluator
        
        Args:
            case_dir (str): Path to the test case directory
            case_name (str): Name of the test case
            openai_api_key (str): OpenAI API key for LLM evaluation
            model (str): OpenAI model to use for evaluation
        """
        super().__init__(case_dir, case_name, eval_mode="pvpython")
        
        # Initialize LLM evaluator
        self.llm_evaluator = LLMEvaluator(api_key=openai_api_key, model=model)
        
        # Initialize image metrics calculator
        self.image_metrics_calculator = CaseImageMetrics(case_dir, case_name, eval_mode="pvpython")
        
        # Set paths for pvpython evaluation
        self.gs_state_path = os.path.join(case_dir, "GS", f"{case_name}_gs.pvsm")
        self.gs_code_path = os.path.join(case_dir, "GS", f"{case_name}_gs.py")
        self.result_state_path = os.path.join(case_dir, "results", "pvpython", f"{case_name}.pvsm")
        self.generated_code_path = os.path.join(case_dir, "results", "pvpython", f"{case_name}.py")
        self.visualization_goals_path = os.path.join(case_dir, "visualization_goals.txt")
        self.screenshot_dir = os.path.join(case_dir, "evaluation_results", "pvpython", "screenshots")
        
        # Ensure screenshot directory exists
        os.makedirs(self.screenshot_dir, exist_ok=True)
    
    def load_visualization_goals(self) -> str:
        """
        Load visualization goals from the test case
        
        Returns:
            str: Visualization goals text
        """
        if not os.path.exists(self.visualization_goals_path):
            return "No visualization goals found for this test case."
        
        with open(self.visualization_goals_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def parse_visualization_goals(self) -> List[str]:
        """
        Parse visualization goals and return a list of individual goals
        
        Returns:
            List[str]: List of individual goals
        """
        goals_text = self.load_visualization_goals()
        if "No visualization goals found" in goals_text:
            return []
        
        # Split by lines and filter for numbered goals
        lines = goals_text.split('\n')
        goals = []
        
        for line in lines:
            line = line.strip()
            # Check if line starts with a number followed by a period or dot
            if line and (line[0].isdigit() or (len(line) > 1 and line[0].isdigit())):
                # Find the first occurrence of '. ' to separate number from goal text
                if '. ' in line:
                    goal_text = line.split('. ', 1)[1]
                    goals.append(goal_text)
        
        return goals
    
    def get_goals_count(self) -> int:
        """
        Get the number of visualization goals
        
        Returns:
            int: Number of goals
        """
        return len(self.parse_visualization_goals())
    
    def evaluate_visualization_quality(self) -> int:
        """
        Evaluate visualization quality using LLM judge with screenshots
        
        Returns:
            int: Score for visualization quality
        """
        print(f"Evaluating visualization quality for {self.case_name}...")
        
        # Get number of goals for dynamic scoring
        goals_count = self.get_goals_count()
        max_score = goals_count * 10  # 10 points per goal
        
        # Check if result state file exists (required)
        if not os.path.exists(self.result_state_path):
            explanation = f"Result state file not found: {self.result_state_path}"
            self.evaluation_results["scores"]["visualization_quality"] = {
                "score": 0,
                "max_score": max_score,
                "explanation": explanation
            }
            return 0
        
        # Check if ground truth state file exists
        has_ground_truth = os.path.exists(self.gs_state_path)
        
        try:
            # Load visualization goals
            visualization_goals = self.load_visualization_goals()
            
            if has_ground_truth:
                # Take screenshots from both states (original behavior)
                print("Taking screenshots from both ground truth and result states...")
                screenshots = compare_states_screenshots(
                    self.gs_state_path,
                    self.result_state_path,
                    self.screenshot_dir,
                    data_directory=self.data_dir
                )
                
                # Create evaluation prompt for comparison
                evaluation_prompt = self._create_evaluation_prompt(visualization_goals, goals_count)
                
                # Perform LLM evaluation with both ground truth and result
                llm_result = self.llm_evaluator.evaluate_visualization(
                    screenshots['ground_truth'],
                    screenshots['result'],
                    evaluation_prompt
                )
            else:
                # Take screenshots from result state only
                print("Ground truth state not found. Taking screenshots from result state only...")
                from screenshot_helper import take_screenshots_from_state
                
                result_screenshots = take_screenshots_from_state(
                    self.result_state_path,
                    self.screenshot_dir,
                    prefix="result_",
                    data_directory=self.data_dir
                )
                
                # Create evaluation prompt for result-only evaluation
                evaluation_prompt = self._create_result_only_evaluation_prompt(visualization_goals, goals_count)
                
                # Perform LLM evaluation with result screenshots only
                llm_result = self.llm_evaluator.evaluate_visualization_result_only(
                    result_screenshots,
                    evaluation_prompt
                )
            
            # Parse the LLM result
            score, explanation = self._parse_llm_result(llm_result, goals_count)
            
            # Add note about missing ground truth if applicable
            if not has_ground_truth:
                explanation = f"Note: Ground truth state file not found ({self.gs_state_path}). Evaluation based on result screenshots only. " + explanation
            
            self.evaluation_results["scores"]["visualization_quality"] = {
                "score": score,
                "max_score": max_score,
                "explanation": explanation,
                "llm_raw_response": llm_result
            }
            
            return score
            
        except Exception as e:
            explanation = f"Error during visualization evaluation: {str(e)}"
            self.evaluation_results["scores"]["visualization_quality"] = {
                "score": 0,
                "max_score": max_score,
                "explanation": explanation
            }
            return 0
    
    def _create_evaluation_prompt(self, visualization_goals: str, goals_count: int) -> str:
        """
        Create the evaluation prompt for LLM
        
        Args:
            visualization_goals (str): The visualization goals text
            goals_count (int): Number of goals to evaluate
            
        Returns:
            str: Complete evaluation prompt
        """
        # Create dynamic goal scoring format
        goal_format_lines = []
        for i in range(1, goals_count + 1):
            goal_format_lines.append(f'    "goal_{i}_score": <score_0_to_10>,')
            goal_format_lines.append(f'    "goal_{i}_explanation": "<detailed explanation>",')
        
        goal_format = '\n'.join(goal_format_lines)
        
        prompt = f"""
You are an expert scientific visualization evaluator. You will be shown two sets of images:

1. Ground Truth Images (first 3 images): These show the expected/correct visualization for the {self.case_name} test case
2. Result Images (next 3 images): These show the agent-generated visualization results

Please evaluate the Result Images against the Ground Truth Images based on these specific visualization goals:

{visualization_goals}

For each visualization goal, provide a score from 0-10 points based on how well the result matches the ground truth and meets the criteria.

Please respond with a JSON object in the following format:
{{
{goal_format}
    "total_score": <sum_of_all_scores>,
    "overall_explanation": "<summary of overall assessment>"
}}

Be specific about what you observe in the images and how well the results match the expected visualization goals.
"""
        return prompt
    
    def _create_result_only_evaluation_prompt(self, visualization_goals: str, goals_count: int) -> str:
        """
        Create the evaluation prompt for LLM when only result images are available
        
        Args:
            visualization_goals (str): The visualization goals text
            goals_count (int): Number of goals to evaluate
            
        Returns:
            str: Complete evaluation prompt for result-only evaluation
        """
        # Create dynamic goal scoring format
        goal_format_lines = []
        for i in range(1, goals_count + 1):
            goal_format_lines.append(f'    "goal_{i}_score": <score_0_to_10>,')
            goal_format_lines.append(f'    "goal_{i}_explanation": "<detailed explanation>",')
        
        goal_format = '\n'.join(goal_format_lines)
        
        prompt = f"""
You are an expert scientific visualization evaluator. You will be shown visualization result images for the {self.case_name} test case.

Note: Ground truth images are not available for this evaluation, so please evaluate based on how well the visualization achieves the stated goals and general visualization quality principles.

Please evaluate the visualization results based on these specific visualization goals:

{visualization_goals}

For each visualization goal, provide a score from 0-10 points based on how well the result meets the criteria and demonstrates good visualization practices.

Please respond with a JSON object in the following format:
{{
{goal_format}
    "total_score": <sum_of_all_scores>,
    "overall_explanation": "<summary of overall assessment, noting that evaluation was done without ground truth>"
}}

Be specific about what you observe in the images and how well the results meet the stated visualization goals.
"""
        return prompt
    
    def _parse_llm_result(self, llm_result: Dict[str, Any], goals_count: int) -> tuple:
        """
        Parse LLM evaluation result and extract score
        
        Args:
            llm_result (dict): Result from LLM evaluation
            goals_count (int): Number of goals to parse
            
        Returns:
            tuple: (score, explanation)
        """
        if "error" in llm_result:
            return 0, f"LLM evaluation error: {llm_result['error']}"
        
        try:
            # Extract total score from LLM response
            if "total_score" in llm_result:
                max_possible = goals_count * 10
                score = min(max_possible, max(0, int(llm_result["total_score"])))
            else:
                # Fallback: sum individual goal scores
                goal_scores = []
                for i in range(1, goals_count + 1):
                    goal_key = f"goal_{i}_score"
                    goal_scores.append(llm_result.get(goal_key, 0))
                
                max_possible = goals_count * 10
                score = min(max_possible, max(0, sum(goal_scores)))
            
            explanation = llm_result.get("overall_explanation", "LLM evaluation completed")
            return score, explanation
            
        except (ValueError, KeyError) as e:
            return 0, f"Error parsing LLM result: {str(e)}"
    
    def evaluate_code_similarity(self) -> int:
        """
        Evaluate code similarity between generated and ground truth code
        
        Returns:
            int: Score for code similarity
        """
        print(f"Evaluating code similarity for {self.case_name}...")
        
        # Check if files exist
        if not os.path.exists(self.gs_code_path):
            explanation = f"Ground truth code file not found: {self.gs_code_path}"
            self.evaluation_results["scores"]["code_similarity"] = {
                "score": 0,
                "max_score": 20,
                "explanation": explanation
            }
            return 0
        
        if not os.path.exists(self.generated_code_path):
            explanation = f"Generated code file not found: {self.generated_code_path}"
            self.evaluation_results["scores"]["code_similarity"] = {
                "score": 0,
                "max_score": 20,
                "explanation": explanation
            }
            return 0
        
        try:
            # Use the inherited code comparison method
            similarity_score = self.compare_code_with_reference(
                self.generated_code_path,
                self.gs_code_path
            )
            
            # Convert similarity score (0-1) to points (0-20)
            score = int(similarity_score * 20)
            
            explanation = f"Code similarity score: {similarity_score:.3f} (converted to {score}/20 points)"
            
            self.evaluation_results["scores"]["code_similarity"] = {
                "score": score,
                "max_score": 20,
                "explanation": explanation,
                "similarity_raw_score": similarity_score
            }
            
            return score
            
        except Exception as e:
            explanation = f"Error during code similarity evaluation: {str(e)}"
            self.evaluation_results["scores"]["code_similarity"] = {
                "score": 0,
                "max_score": 20,
                "explanation": explanation
            }
            return 0
    
    def evaluate_output_generation(self) -> int:
        """
        Evaluate if the required output files were generated
        
        Returns:
            int: Score for output generation
        """
        print("Evaluating output generation...")
        
        score = 0
        explanations = []
        
        # Check if state file exists (5 points)
        if os.path.exists(self.result_state_path):
            score += 5
            explanations.append("ParaView state file generated successfully")
        else:
            explanations.append(f"ParaView state file not found: {self.result_state_path}")
        
        # Check if Python code file exists (5 points)
        if os.path.exists(self.generated_code_path):
            score += 5
            explanations.append("Python code file generated successfully")
        else:
            explanations.append(f"Python code file not found: {self.generated_code_path}")
        
        self.evaluation_results["scores"]["output_generation"] = {
            "score": score,
            "max_score": 10,
            "explanation": "; ".join(explanations)
        }
        
        return score
    
    def evaluate_visualization_setup(self) -> int:
        """
        Evaluate visualization setup (required by base class)
        For pvpython mode, this is combined with output generation
        
        Returns:
            int: Score for visualization setup
        """
        return self.evaluate_output_generation()
    
    def evaluate_visual_quality(self) -> int:
        """
        Evaluate visual quality (required by base class)
        Maps to our visualization quality evaluation
        
        Returns:
            int: Score for visual quality
        """
        return self.evaluate_visualization_quality()
    
    def evaluate_image_metrics(self) -> Dict[str, Any]:
        """
        Evaluate image quality metrics (PSNR, SSIM, LPIPS)
        
        Returns:
            Dict[str, Any]: Image metrics results
        """
        print("Calculating image quality metrics...")
        
        try:
            # Calculate image metrics for this case
            metrics_result = self.image_metrics_calculator.calculate_case_metrics()
            
            # Store in evaluation results
            self.evaluation_results["image_metrics"] = metrics_result
            
            print(f"Image metrics calculated successfully:")
            averaged_metrics = metrics_result.get("averaged_metrics", {})
            for metric, value in averaged_metrics.items():
                if value is not None:
                    print(f"  Average {metric.upper()}: {value:.4f}")
                else:
                    print(f"  Average {metric.upper()}: N/A")
            
            return metrics_result
            
        except Exception as e:
            print(f"Warning: Failed to calculate image metrics: {e}")
            error_result = {
                "error": str(e),
                "averaged_metrics": {"psnr": None, "ssim": None, "lpips": None}
            }
            self.evaluation_results["image_metrics"] = error_result
            return error_result
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation for this test case
        
        Returns:
            Dict: Complete evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {self.case_name} (pvpython mode)")
        print(f"{'='*60}")
        
        # Get goals count for dynamic max score calculation
        goals_count = self.get_goals_count()
        
        # Run all evaluation components
        viz_score = self.evaluate_visualization_quality()
        code_score = self.evaluate_code_similarity()
        output_score = self.evaluate_output_generation()
        efficiency_score = self.evaluate_efficiency()
        
        # Calculate image metrics
        image_metrics = self.evaluate_image_metrics()
        
        # Calculate total score
        self.evaluation_results["total_score"] = viz_score + code_score + output_score + efficiency_score
        
        # Update max possible score for pvpython mode (dynamic viz score + 20 + 10 + 10)
        self.evaluation_results["max_possible_score"] = (goals_count * 10) + 20 + 10 + 10
        
        # Add evaluator metadata
        self.evaluation_results["evaluator_metadata"] = {
            "evaluator_type": "pvpython_auto",
            "evaluator_version": "2.0.0",
            "goals_count": goals_count,
            "llm_evaluator": self.llm_evaluator.get_evaluator_info(),
            "scoring_scheme": {
                "visualization_quality": f"{goals_count * 10} points (10 per goal)",
                "code_similarity": "20 points",
                "output_generation": "10 points",
                "efficiency": "10 points",
                "total_possible": f"{(goals_count * 10) + 40} points"
            }
        }
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return self.evaluation_results


class PVPythonBatchEvaluator:
    """
    Batch evaluator for all pvpython test cases
    """
    
    def __init__(self, cases_dir: str, openai_api_key: str = None, output_dir: str = None, model: str = "gpt-4o"):
        """
        Initialize the batch evaluator
        
        Args:
            cases_dir (str): Path to the cases directory
            openai_api_key (str): OpenAI API key for LLM evaluation
            output_dir (str): Output directory for batch results
            model (str): OpenAI model to use for evaluation
        """
        self.cases_dir = Path(cases_dir)
        self.openai_api_key = openai_api_key
        self.model = model
        self.output_dir = Path(output_dir) if output_dir else self.cases_dir.parent / "evaluation_results" / "pvpython_auto"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def discover_test_cases(self) -> List[str]:
        """
        Discover all test cases in the cases directory
        
        Returns:
            List[str]: List of test case names
        """
        test_cases = []
        
        if not self.cases_dir.exists():
            print(f"Cases directory not found: {self.cases_dir}")
            return test_cases
        
        for item in self.cases_dir.iterdir():
            if item.is_dir() and (item / "visualization_goals.txt").exists():
                test_cases.append(item.name)
        
        return sorted(test_cases)
    
    def run_all_evaluations(self) -> Dict[str, Any]:
        """
        Run evaluations for all discovered test cases
        
        Returns:
            Dict: Batch evaluation results
        """
        test_cases = self.discover_test_cases()
        
        if not test_cases:
            print("No test cases found!")
            return {"error": "No test cases found"}
        
        print(f"Found {len(test_cases)} test cases: {', '.join(test_cases)}")
        
        batch_results = {
            "evaluation_time": datetime.now().isoformat(),
            "batch_evaluator_metadata": {
                "evaluator_type": "pvpython_batch_auto",
                "evaluator_version": "2.0.0",
                "model": self.model,
                "cases_evaluated": len(test_cases)
            },
            "results": {},
            "summary": {
                "total_cases": len(test_cases),
                "completed": 0,
                "errors": 0,
                "total_score": 0,
                "max_possible_score": 0
            }
        }
        
        for case_name in test_cases:
            print(f"\nProcessing case: {case_name}")
            
            try:
                case_dir = str(self.cases_dir / case_name)
                evaluator = PVPythonAutoEvaluator(case_dir, case_name, self.openai_api_key, self.model)
                result = evaluator.run_evaluation()
                
                batch_results["results"][case_name] = result
                batch_results["summary"]["completed"] += 1
                batch_results["summary"]["total_score"] += result["total_score"]
                batch_results["summary"]["max_possible_score"] += result["max_possible_score"]
                
            except Exception as e:
                print(f"Error evaluating {case_name}: {str(e)}")
                batch_results["results"][case_name] = {
                    "error": str(e),
                    "case_name": case_name
                }
                batch_results["summary"]["errors"] += 1
        
        # Calculate batch image metrics
        print(f"\nCalculating batch image metrics...")
        try:
            from image_metrics_helper import BatchImageMetrics
            batch_image_calculator = BatchImageMetrics(str(self.cases_dir), eval_mode="pvpython", output_dir=str(self.output_dir))
            batch_image_results = batch_image_calculator.calculate_batch_metrics()
            batch_results["batch_image_metrics"] = batch_image_results
        except Exception as e:
            print(f"Warning: Failed to calculate batch image metrics: {e}")
            batch_results["batch_image_metrics"] = {"error": str(e)}
        
        # Save batch results
        batch_result_file = self.output_dir / f"pvpython_batch_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_result_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("BATCH EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total cases: {batch_results['summary']['total_cases']}")
        print(f"Completed: {batch_results['summary']['completed']}")
        print(f"Errors: {batch_results['summary']['errors']}")
        print(f"Overall score: {batch_results['summary']['total_score']}/{batch_results['summary']['max_possible_score']}")
        if batch_results['summary']['max_possible_score'] > 0:
            percentage = (batch_results['summary']['total_score'] / batch_results['summary']['max_possible_score']) * 100
            print(f"Overall percentage: {percentage:.1f}%")
        print(f"Results saved to: {batch_result_file}")
        
        return batch_results


def main():
    parser = argparse.ArgumentParser(description="Automatic pvpython Evaluation Script")
    parser.add_argument("--cases", required=True,
                       help="Path to the cases directory")
    parser.add_argument("--case", 
                       help="Evaluate a specific test case by name")
    parser.add_argument("--output", "-o",
                       help="Output directory for results")
    parser.add_argument("--api-key",
                       help="OpenAI API key (can also be set via OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o",
                       help="OpenAI model to use for evaluation (default: gpt-4o)")
    parser.add_argument("--list", action="store_true",
                       help="List available test cases and exit")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.cases):
        print(f"Error: Cases directory not found: {args.cases}")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    batch_evaluator = PVPythonBatchEvaluator(args.cases, api_key, args.output, args.model)
    
    # List cases if requested
    if args.list:
        test_cases = batch_evaluator.discover_test_cases()
        print(f"Available test cases ({len(test_cases)}):")
        for case in test_cases:
            print(f"  - {case}")
        return
    
    # Run specific case or all cases
    if args.case:
        case_dir = os.path.join(args.cases, args.case)
        if not os.path.exists(case_dir):
            print(f"Error: Test case not found: {args.case}")
            sys.exit(1)
        
        evaluator = PVPythonAutoEvaluator(case_dir, args.case, api_key, args.model)
        evaluator.run_evaluation()
    else:
        batch_evaluator.run_all_evaluations()


if __name__ == "__main__":
    main()