"""
Base class for scientific visualization test case evaluators

This provides common functionality that all test case evaluators can inherit from.
"""

import os
import json
import sys
import ast
import math
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from difflib import SequenceMatcher
from paraview.simple import *

# Optional imports for code comparison (will handle ImportError gracefully)
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    CODE_COMPARISON_AVAILABLE = True
except ImportError:
    CODE_COMPARISON_AVAILABLE = False

class SciVisEvaluator(ABC):
    """
    Abstract base class for scientific visualization test case evaluators
    """
    
    def __init__(self, case_dir: str, case_name: str, eval_mode: str = "mcp"):
        """
        Initialize the evaluator
        
        Args:
            case_dir (str): Path to the test case directory
            case_name (str): Name of the test case
            eval_mode (str): Evaluation mode - either "mcp" or "pvpython"
        """
        self.case_dir = case_dir
        self.case_name = case_name
        self.eval_mode = eval_mode
        
        # Set paths based on evaluation mode
        self.evaluation_dir = os.path.join(case_dir, "evaluation_results", eval_mode)
        self.result_state_dir = os.path.join(case_dir, "results", f"{eval_mode}_state")
        self.test_results_dir = os.path.join(case_dir, "test_results", eval_mode)
        self.rubric_path = os.path.join(case_dir, "evaluation_scripts", f"{case_name}_rubric.json")
        
        # Default data directory - can be overridden by subclasses
        self.data_dir = os.path.join(case_dir, "data")
        
        # Load rubric (optional)
        if os.path.exists(self.rubric_path):
            with open(self.rubric_path, 'r') as f:
                self.rubric = json.load(f)
            print(f"Loaded rubric from: {self.rubric_path}")
        else:
            print(f"No rubric file found at: {self.rubric_path} (using default values)")
            self.rubric = {
                "total_points": 100,  # Default total points
                "description": f"Default rubric for {case_name} test case"
            }
        
        # Initialize results
        self.evaluation_results = {
            "case_name": case_name,
            "evaluation_mode": eval_mode,
            "evaluation_time": datetime.now().isoformat(),
            "scores": {},
            "total_score": 0,
            "max_possible_score": self.rubric.get("total_points", 100)
        }
    
    def check_file_exists(self, file_path: str, description: str = "File") -> bool:
        """
        Check if a file exists and log appropriately
        
        Args:
            file_path (str): Path to check
            description (str): Description for logging
            
        Returns:
            bool: True if file exists, False otherwise
        """
        exists = os.path.exists(file_path)
        if not exists:
            print(f"WARNING: {description} not found: {file_path}")
        return exists
    
    def load_test_results(self) -> Dict[str, Any]:
        """
        Load the latest test result file for efficiency evaluation
        
        Returns:
            Dict containing test results or empty dict if not found
        """
        if not os.path.exists(self.test_results_dir):
            return {}
        
        test_result_files = [f for f in os.listdir(self.test_results_dir) 
                           if f.startswith("test_result_") and f.endswith(".json")]
        
        if not test_result_files:
            return {}
        
        # Get the latest test result
        latest_file = max(test_result_files)
        test_result_path = os.path.join(self.test_results_dir, latest_file)
        
        try:
            with open(test_result_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading test results: {e}")
            return {}
    
    def evaluate_efficiency(self, test_data: Dict[str, Any] = None) -> int:
        """
        Evaluate efficiency metrics (execution time, resource usage)
        
        Args:
            test_data: Test result data, if None will try to load automatically
            
        Returns:
            int: Total efficiency score
        """
        if test_data is None:
            test_data = self.load_test_results()
        
        efficiency_scores = {
            "execution_time": {"score": 0, "max_score": 5, "explanation": "No test result found"},
            "token_usage": {"score": 0, "max_score": 5, "explanation": "Token usage data not available"}
        }
        
        if test_data:
            # Evaluate execution time
            if "duration" in test_data:
                duration = test_data["duration"]
                if duration < 15:  # Less than 15 seconds
                    efficiency_scores["execution_time"]["score"] = 5
                    efficiency_scores["execution_time"]["explanation"] = f"Completed in {duration:.2f} seconds (excellent)"
                elif duration < 30:  # Less than 30 seconds
                    efficiency_scores["execution_time"]["score"] = 4
                    efficiency_scores["execution_time"]["explanation"] = f"Completed in {duration:.2f} seconds (very good)"
                elif duration < 45:  # Less than 45 seconds
                    efficiency_scores["execution_time"]["score"] = 3
                    efficiency_scores["execution_time"]["explanation"] = f"Completed in {duration:.2f} seconds (good)"
                elif duration < 60:  # Less than 60 seconds
                    efficiency_scores["execution_time"]["score"] = 2
                    efficiency_scores["execution_time"]["explanation"] = f"Completed in {duration:.2f} seconds (acceptable)"
                elif duration < 75:  # Less than 75 seconds
                    efficiency_scores["execution_time"]["score"] = 1
                    efficiency_scores["execution_time"]["explanation"] = f"Completed in {duration:.2f} seconds (slow)"
                else:
                    efficiency_scores["execution_time"]["score"] = 0
                    efficiency_scores["execution_time"]["explanation"] = f"Completed in {duration:.2f} seconds (very slow)"
            
            # Evaluate token usage (use real token data if available)
            if "token_usage" in test_data:
                token_usage = test_data["token_usage"]
                total_tokens = token_usage.get("total_tokens", 0)
                input_tokens = token_usage.get("input_tokens", 0)
                output_tokens = token_usage.get("output_tokens", 0)
                
                # Score based on total token usage
                if total_tokens < 500:
                    efficiency_scores["token_usage"]["score"] = 5
                    efficiency_scores["token_usage"]["explanation"] = f"Total {total_tokens} tokens (very efficient)"
                elif total_tokens < 1000:
                    efficiency_scores["token_usage"]["score"] = 4
                    efficiency_scores["token_usage"]["explanation"] = f"Total {total_tokens} tokens (efficient)"
                elif total_tokens < 2000:
                    efficiency_scores["token_usage"]["score"] = 3
                    efficiency_scores["token_usage"]["explanation"] = f"Total {total_tokens} tokens (moderate)"
                elif total_tokens < 3000:
                    efficiency_scores["token_usage"]["score"] = 2
                    efficiency_scores["token_usage"]["explanation"] = f"Total {total_tokens} tokens (high usage)"
                else:
                    efficiency_scores["token_usage"]["score"] = 1
                    efficiency_scores["token_usage"]["explanation"] = f"Total {total_tokens} tokens (very high usage)"
            elif "response" in test_data:
                # Fallback to old estimation method if token_usage not available
                response_length = len(test_data["response"])
                estimated_tokens = response_length // 4  # Rough estimation
                
                if estimated_tokens < 1000:
                    efficiency_scores["token_usage"]["score"] = 5
                    efficiency_scores["token_usage"]["explanation"] = f"Estimated ~{estimated_tokens} tokens (efficient)"
                elif estimated_tokens < 2000:
                    efficiency_scores["token_usage"]["score"] = 3
                    efficiency_scores["token_usage"]["explanation"] = f"Estimated ~{estimated_tokens} tokens (moderate)"
                else:
                    efficiency_scores["token_usage"]["score"] = 1
                    efficiency_scores["token_usage"]["explanation"] = f"Estimated ~{estimated_tokens} tokens (high usage)"
        
        self.evaluation_results["scores"]["efficiency"] = efficiency_scores
        
        return sum(score_data["score"] for score_data in efficiency_scores.values())
    
    def evaluate_state_saving(self, state_path: str) -> int:
        """
        Evaluate ParaView state file saving
        
        Args:
            state_path (str): Path to the state file to check
            
        Returns:
            int: Score for state saving
        """
        score = 0
        explanation = ""
        
        if os.path.exists(state_path):
            # Check if the file is a valid ParaView state file
            try:
                with open(state_path, 'r') as f:
                    content = f.read()
                    if "ServerManagerState" in content and "ParaView" in content:
                        score = 5
                        explanation = "ParaView state file saved correctly"
                    else:
                        explanation = "File exists but doesn't appear to be a valid ParaView state file"
            except Exception as e:
                explanation = f"Error reading state file: {str(e)}"
        else:
            explanation = "State file not found at expected location"
        
        self.evaluation_results["scores"]["output_saving"] = {
            "score": score,
            "max_score": 5,
            "explanation": explanation
        }
        
        return score
    
    def save_results(self, output_path: str = None) -> str:
        """
        Save evaluation results to JSON file
        
        Args:
            output_path (str): Custom output path, if None uses default
            
        Returns:
            str: Path to saved results file
        """
        if output_path is None:
            os.makedirs(self.evaluation_dir, exist_ok=True)
            output_path = os.path.join(self.evaluation_dir, "evaluation_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        return output_path
    
    def get_codebert_embedding(self, code: str,
                              tokenizer,
                              model,
                              device) -> torch.Tensor:
        """Return the [CLS] embedding for `code` using CodeBERT."""
        if not CODE_COMPARISON_AVAILABLE:
            raise ImportError("torch and transformers are required for code comparison")
            
        inputs = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # [CLS] token is at index 0
        return outputs.last_hidden_state[0, 0, :].cpu()

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two 1D tensors."""
        if not CODE_COMPARISON_AVAILABLE:
            raise ImportError("torch is required for code comparison")
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def diff_ratio(self, a: str, b: str) -> float:
        """Line-based diff ratio ∈ [0,1]."""
        a_lines = a.splitlines()
        b_lines = b.splitlines()
        return SequenceMatcher(None, a_lines, b_lines).ratio()

    def ast_ratio(self, a: str, b: str) -> float:
        """
        Rough AST-based ratio: 1 - |#nodes(a) - #nodes(b)| / max(#nodes).
        Returns 0 if parsing fails or if either is empty.
        """
        try:
            na = len(list(ast.walk(ast.parse(a))))
            nb = len(list(ast.walk(ast.parse(b))))
            if max(na, nb) == 0:
                return 0.0
            return 1 - abs(na - nb) / max(na, nb)
        except SyntaxError:
            return 0.0

    def compare_code_with_reference(self, ai_code_path: str, human_code_path: str,
                                   model_name: str = "microsoft/codebert-base",
                                   w_embed: float = 0.7, w_diff: float = 0.3, w_ast: float = 0.0) -> float:
        """
        Compare AI-generated code with human reference code using multiple metrics.
        
        Args:
            ai_code_path (str): Path to AI-generated code file
            human_code_path (str): Path to human reference code file
            model_name (str): HuggingFace model name for CodeBERT
            w_embed (float): Weight for embedding similarity
            w_diff (float): Weight for line-overlap ratio
            w_ast (float): Weight for AST-based ratio
            
        Returns:
            float: Composite similarity score between 0 and 1
        """
        if not CODE_COMPARISON_AVAILABLE:
            print("Warning: torch and transformers not available, code comparison disabled")
            return 0.0
            
        # Validate weights
        total_w = w_embed + w_diff + w_ast
        if abs(total_w - 1.0) > 1e-6:
            print(f"Warning: Weights don't sum to 1.0 (got {total_w:.2f}), normalizing...")
            w_embed /= total_w
            w_diff /= total_w
            w_ast /= total_w

        # Check if files exist
        if not os.path.exists(ai_code_path):
            print(f"AI code file not found: {ai_code_path}")
            return 0.0
        
        if not os.path.exists(human_code_path):
            print(f"Human reference code file not found: {human_code_path}")
            return 0.0

        try:
            # Read code files
            with open(human_code_path, "r", encoding="utf-8") as f:
                human_code = f.read()
            with open(ai_code_path, "r", encoding="utf-8") as f:
                ai_code = f.read()

            # Load CodeBERT
            print(f"Loading `{model_name}` for code comparison...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device).eval()

            # 1) Embedding similarity
            emb_human = self.get_codebert_embedding(human_code, tokenizer, model, device)
            emb_ai = self.get_codebert_embedding(ai_code, tokenizer, model, device)
            sim_embed = self.cosine_similarity(emb_human, emb_ai)

            # 2) Diff ratio
            sim_diff = self.diff_ratio(human_code, ai_code)

            # 3) AST ratio (if weighted)
            sim_ast = self.ast_ratio(human_code, ai_code) if w_ast > 0 else 0.0

            # 4) Composite score
            composite = (
                w_embed * sim_embed +
                w_diff * sim_diff +
                w_ast * sim_ast
            )

            print(f"Code comparison metrics:")
            print(f"  Embedding similarity: {sim_embed:.4f}")
            print(f"  Line-overlap ratio: {sim_diff:.4f}")
            if w_ast > 0:
                print(f"  AST-based ratio: {sim_ast:.4f}")
            print(f"  Composite score: {composite:.4f}")

            return composite

        except Exception as e:
            print(f"Error during code comparison: {str(e)}")
            return 0.0

    def evaluate_code_comparison(self) -> float:
        """
        Evaluate the similarity between the human-written (gold standard) code and the agent-generated code.
        Uses CodeBERT embedding cosine similarity, line-overlap ratio, and AST-based ratio to compute a
        composite score. The result is added to the evaluation rubric.
        
        This is a default implementation that subclasses can override.
        Subclasses should set self.gs_code_path and self.generated_code_path attributes.
        
        Returns:
            float: Composite similarity score between 0 and 1
        """
        if not CODE_COMPARISON_AVAILABLE:
            print("Warning: torch and transformers not available, code comparison disabled")
            explanation = "Code comparison requires torch and transformers packages"
            self.evaluation_results["scores"]["code_comparison"] = {
                "score": 0,
                "max_score": 1,
                "explanation": explanation
            }
            return 0.0

        # Check if paths are set by subclass
        if not hasattr(self, 'gs_code_path') or not hasattr(self, 'generated_code_path'):
            explanation = "Code comparison paths not set by subclass (gs_code_path, generated_code_path)"
            print(explanation)
            self.evaluation_results["scores"]["code_comparison"] = {
                "score": 0,
                "max_score": 1,
                "explanation": explanation
            }
            return 0.0

        # Read gold standard code
        try:
            with open(self.gs_code_path, "r", encoding="utf-8") as f:
                human_code = f.read()
        except Exception as e:
            human_code = ""
            print(f"Error reading gold standard code: {e}")

        # Read generated code
        try:
            with open(self.generated_code_path, "r", encoding="utf-8") as f:
                ai_code = f.read()
        except Exception as e:
            ai_code = ""
            print(f"Error reading generated code: {e}")

        if not human_code or not ai_code:
            msg = "Missing code files for comparison."
            print(msg)
            self.evaluation_results["scores"]["code_comparison"] = {
                "score": 0,
                "max_score": 1,
                "explanation": msg
            }
            return 0.0

        try:
            # Load CodeBERT model
            model_name = "microsoft/codebert-base"
            print(f"Loading `{model_name}` for code comparison...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device).eval()

            # Calculate similarity metrics
            emb_human = self.get_codebert_embedding(human_code, tokenizer, model, device)
            emb_ai = self.get_codebert_embedding(ai_code, tokenizer, model, device)
            sim_embed = self.cosine_similarity(emb_human, emb_ai)
            sim_diff = self.diff_ratio(human_code, ai_code)
            sim_ast = self.ast_ratio(human_code, ai_code)

            # Weight settings (default: embed 0.7, diff 0.3, ast 0.0)
            w_embed, w_diff, w_ast = 0.7, 0.3, 0.0
            composite = w_embed * sim_embed + w_diff * sim_diff + w_ast * sim_ast

            explanation = (
                f"Code similarity metrics:\n"
                f"  Embedding similarity: {sim_embed:.4f}\n"
                f"  Line-overlap diff ratio: {sim_diff:.4f}\n"
                f"  AST ratio: {sim_ast:.4f}\n"
                f"Composite score (weighted): {composite:.4f}"
            )
            
            self.evaluation_results["scores"]["code_comparison"] = {
                "score": composite,
                "max_score": 1,
                "explanation": explanation
            }
            
            print(f"Code comparison completed. Composite score: {composite:.4f}")
            return composite

        except Exception as e:
            error_msg = f"Error during code comparison evaluation: {str(e)}"
            print(error_msg)
            self.evaluation_results["scores"]["code_comparison"] = {
                "score": 0,
                "max_score": 1,
                "explanation": error_msg
            }
            return 0.0
    
    def print_summary(self):
        """Print a summary of the evaluation results"""
        print("\n" + "="*50)
        print(f"{self.case_name.upper()} EVALUATION RESULTS")
        print("="*50)
        print(f"Total Score: {self.evaluation_results['total_score']}/{self.evaluation_results['max_possible_score']}")
        print(f"Percentage: {(self.evaluation_results['total_score']/self.evaluation_results['max_possible_score']*100):.1f}%")
        print("\nDetailed Scores:")
        
        for category, scores in self.evaluation_results["scores"].items():
            print(f"\n{category.upper()}:")
            if isinstance(scores, dict) and "score" in scores:
                print(f"  Score: {scores['score']}/{scores['max_score']}")
                print(f"  Explanation: {scores['explanation']}")
            else:
                for criterion, score_data in scores.items():
                    if isinstance(score_data, dict) and "score" in score_data:
                        print(f"  {criterion}: {score_data['score']}/{score_data['max_score']}")
                        print(f"    {score_data['explanation']}")
    
    def evaluate_data_loading(self) -> int:
        """
        Evaluate data loading criteria by checking the result state file
        
        Returns:
            int: Score for data loading based on rubric
        """
        print("Evaluating data loading...")
        score = 0
        explanation = ""
        
        # Get max score from rubric
        max_score = 5  # Default fallback
        if hasattr(self, 'rubric') and 'data_loading' in self.rubric:
            data_loading_criteria = self.rubric['data_loading']
            if data_loading_criteria and len(data_loading_criteria) > 0:
                max_score = data_loading_criteria[0].get('points', 5)
        
        # Check for result state file
        result_state_path = os.path.join(self.result_state_dir, f"{self.case_name}.pvsm")
        
        if os.path.exists(result_state_path):
            try:
                # Load the state and check if data was loaded correctly
                LoadState(result_state_path, data_directory=self.data_dir)
                sources = GetSources()
                
                if sources:
                    # Check if there's a source that could be the data reader
                    data_loaded = False
                    for source_name, source_proxy in sources.items():
                        if "ImageReader" in str(source_name) or "Reader" in str(source_name):
                            data_loaded = True
                            break
                    
                    if data_loaded:
                        score = max_score
                        explanation = "Data appears to be loaded correctly in the state file"
                    else:
                        explanation = "No data reader found in the state file"
                else:
                    explanation = "No sources found in the state file"
                    
            except Exception as e:
                explanation = f"Error loading state file: {str(e)}"
        else:
            explanation = "Result state file not found"
        
        self.evaluation_results["scores"]["data_loading"] = {
            "score": score,
            "max_score": max_score,
            "explanation": explanation
        }
        
        return score
    
    @abstractmethod
    def evaluate_visualization_setup(self) -> int:
        """
        Evaluate visualization setup (implementation specific to each test case)
        
        Returns:
            int: Score for visualization setup
        """
        pass
    
    @abstractmethod
    def evaluate_visual_quality(self) -> int:
        """
        Evaluate visual quality (implementation specific to each test case)
        
        Returns:
            int: Score for visual quality
        """
        pass
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation process
        
        Returns:
            Dict containing evaluation results
        """
        print(f"Starting {self.case_name} case evaluation...")
        
        total_score = 0
        
        # Run all evaluation components
        total_score += self.evaluate_data_loading()
        total_score += self.evaluate_visualization_setup()
        total_score += self.evaluate_visual_quality()
        
        # Common evaluations
        total_score += self.evaluate_efficiency()
        
        # Case-specific state saving evaluation
        result_state_path = os.path.join(self.result_state_dir, f"{self.case_name}.pvsm")
        total_score += self.evaluate_state_saving(result_state_path)
        
        # --- Add Code Comparison Evaluation ---
        # Only run code comparison for pvpython mode
        if self.eval_mode == "pvpython":
            # Check if the subclass implements evaluate_code_comparison()
            if hasattr(self, "evaluate_code_comparison") and callable(self.evaluate_code_comparison):
                # Call the method (expected to return a composite between 0 and 1)
                comp = self.evaluate_code_comparison()
                # Scale it to a 5-point maximum and round to a whole number
                code_comp_score = round(comp * 5)
                total_score += code_comp_score
                
                # Store the result in the evaluation_results
                explanation = self.evaluation_results["scores"].get("code_comparison", {}).get("explanation", "")
                self.evaluation_results["scores"]["code_comparison"] = {
                    "score": code_comp_score,
                    "max_score": 5,
                    "explanation": explanation + f"\n(Scaled to 5 points: {comp:.4f} * 5 = {code_comp_score})"
                }
        # --- End Code Comparison Evaluation ---
        
        self.evaluation_results["total_score"] = total_score
        
        # Save and display results
        results_path = self.save_results()
        self.print_summary()
        print(f"\nResults saved to: {results_path}")
        
        return self.evaluation_results
