#!/usr/bin/env python3
"""
Image Metrics Helper for Scientific Visualization Evaluation

This module provides image quality metrics (PSNR, SSIM, LPIPS) for comparing
ground truth visualizations with agent-generated results across multiple viewpoints.
"""

import os
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Import image processing libraries with graceful fallback
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Please install: pip install Pillow")

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Please install: pip install torch torchvision")

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage import img_as_float
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Please install: pip install scikit-image")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Please install: pip install lpips")


class ImageMetricsCalculator:
    """
    Calculator for image quality metrics between ground truth and result images
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the image metrics calculator
        
        Args:
            device (str): Device to use for LPIPS calculation ("cpu", "cuda", or "auto")
        """
        self.device = self._get_device(device)
        self.lpips_net = None
        
        # Initialize LPIPS network if available
        if LPIPS_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.lpips_net = lpips.LPIPS(net='alex').to(self.device)
                print(f"LPIPS network initialized on {self.device}")
            except Exception as e:
                print(f"Warning: Failed to initialize LPIPS network: {e}")
                # Don't modify the global variable, just handle the error locally
    
    def _get_device(self, device: str) -> str:
        """Get the appropriate device for computation"""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _load_and_preprocess_image(self, image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load and preprocess an image
        
        Args:
            image_path (str): Path to the image file
            target_size (tuple): Target size (width, height) for resizing
            
        Returns:
            np.ndarray: Preprocessed image as numpy array
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for image loading")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize if target size specified
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to PyTorch tensor for LPIPS calculation
        
        Args:
            image (np.ndarray): Image array in range [0, 255]
            
        Returns:
            torch.Tensor: Image tensor in range [-1, 1] with shape (1, 3, H, W)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LPIPS calculation")
        
        # Normalize to [0, 1] then to [-1, 1]
        image_normalized = image.astype(np.float32) / 255.0
        image_normalized = (image_normalized - 0.5) * 2.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def calculate_psnr(self, gt_image: np.ndarray, result_image: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            gt_image (np.ndarray): Ground truth image
            result_image (np.ndarray): Result image
            
        Returns:
            float: PSNR value in dB
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image is required for PSNR calculation")
        
        # Ensure images have the same shape
        if gt_image.shape != result_image.shape:
            raise ValueError(f"Image shapes don't match: {gt_image.shape} vs {result_image.shape}")
        
        # Convert to float
        gt_float = img_as_float(gt_image)
        result_float = img_as_float(result_image)
        
        # Calculate PSNR
        psnr_value = psnr(gt_float, result_float, data_range=1.0)
        
        return float(psnr_value)
    
    def calculate_ssim(self, gt_image: np.ndarray, result_image: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM)
        
        Args:
            gt_image (np.ndarray): Ground truth image
            result_image (np.ndarray): Result image
            
        Returns:
            float: SSIM value between 0 and 1
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image is required for SSIM calculation")
        
        # Ensure images have the same shape
        if gt_image.shape != result_image.shape:
            raise ValueError(f"Image shapes don't match: {gt_image.shape} vs {result_image.shape}")
        
        # Convert to float
        gt_float = img_as_float(gt_image)
        result_float = img_as_float(result_image)
        
        # Calculate SSIM for multichannel (RGB) images
        ssim_value = ssim(gt_float, result_float, multichannel=True, channel_axis=2, data_range=1.0)
        
        return float(ssim_value)
    
    def calculate_lpips(self, gt_image: np.ndarray, result_image: np.ndarray) -> float:
        """
        Calculate Learned Perceptual Image Patch Similarity (LPIPS)
        
        Args:
            gt_image (np.ndarray): Ground truth image
            result_image (np.ndarray): Result image
            
        Returns:
            float: LPIPS value (lower is better, typically between 0 and 1)
        """
        if not LPIPS_AVAILABLE or self.lpips_net is None:
            raise ImportError("LPIPS library and network are required for LPIPS calculation")
        
        # Ensure images have the same shape
        if gt_image.shape != result_image.shape:
            raise ValueError(f"Image shapes don't match: {gt_image.shape} vs {result_image.shape}")
        
        # Convert to tensors
        gt_tensor = self._image_to_tensor(gt_image)
        result_tensor = self._image_to_tensor(result_image)
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_value = self.lpips_net(gt_tensor, result_tensor)
        
        return float(lpips_value.item())
    
    def calculate_all_metrics(self, gt_image_path: str, result_image_path: str, 
                            target_size: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
        """
        Calculate all available metrics for a pair of images
        
        Args:
            gt_image_path (str): Path to ground truth image
            result_image_path (str): Path to result image
            target_size (tuple): Target size for resizing (optional)
            
        Returns:
            Dict[str, float]: Dictionary containing all calculated metrics
        """
        metrics = {}
        
        try:
            # Load images
            gt_image = self._load_and_preprocess_image(gt_image_path, target_size)
            result_image = self._load_and_preprocess_image(result_image_path, target_size)
            
            # Calculate PSNR
            if SKIMAGE_AVAILABLE:
                try:
                    metrics['psnr'] = self.calculate_psnr(gt_image, result_image)
                except Exception as e:
                    print(f"Warning: PSNR calculation failed: {e}")
                    metrics['psnr'] = None
            else:
                metrics['psnr'] = None
            
            # Calculate SSIM
            if SKIMAGE_AVAILABLE:
                try:
                    metrics['ssim'] = self.calculate_ssim(gt_image, result_image)
                except Exception as e:
                    print(f"Warning: SSIM calculation failed: {e}")
                    metrics['ssim'] = None
            else:
                metrics['ssim'] = None
            
            # Calculate LPIPS
            if LPIPS_AVAILABLE and self.lpips_net is not None:
                try:
                    metrics['lpips'] = self.calculate_lpips(gt_image, result_image)
                except Exception as e:
                    print(f"Warning: LPIPS calculation failed: {e}")
                    metrics['lpips'] = None
            else:
                metrics['lpips'] = None
        
        except Exception as e:
            print(f"Error calculating metrics for {gt_image_path} vs {result_image_path}: {e}")
            metrics = {'psnr': None, 'ssim': None, 'lpips': None}
        
        return metrics


class CaseImageMetrics:
    """
    Calculate image metrics for a single test case across multiple viewpoints
    """
    
    def __init__(self, case_dir: str, case_name: str, eval_mode: str = "mcp"):
        """
        Initialize case metrics calculator

        Args:
            case_dir (str): Path to the test case directory
            case_name (str): Name of the test case
            eval_mode (str): Evaluation mode ("mcp" or "pvpython")
        """
        self.case_dir = case_dir
        self.case_name = case_name
        self.eval_mode = eval_mode

        # Set up paths
        self.gs_dir = os.path.join(case_dir, "GS")
        self.screenshot_dir = os.path.join(case_dir, "evaluation_results", eval_mode, "screenshots")
        self.results_dir = os.path.join(case_dir, "results", eval_mode)

        # Initialize metrics calculator
        self.calculator = ImageMetricsCalculator()

        # Define viewpoints
        self.viewpoints = ["diagonal", "front", "side"]
    
    def get_image_paths(self) -> Dict[str, Dict[str, str]]:
        """
        Get paths to ground truth and result images for all viewpoints

        Returns:
            Dict: Nested dictionary with viewpoint -> {"gt": path, "result": path}
        """
        image_paths = {}

        # First check if there's a single-image result (chatvis_bench style)
        # Format: {data_name}/GS/{data_name}_gs.png and {data_name}/results/{eval_mode}/{data_name}.png
        single_gt_path = os.path.join(self.gs_dir, f"{self.case_name}_gs.png")
        single_result_path = os.path.join(self.results_dir, f"{self.case_name}.png")

        if os.path.exists(single_gt_path) and os.path.exists(single_result_path):
            # Single image mode - use the same image for all viewpoints
            print(f"Using single-image mode for {self.case_name}")
            for viewpoint in self.viewpoints:
                image_paths[viewpoint] = {
                    "gt": single_gt_path,
                    "result": single_result_path
                }
        else:
            # Multi-viewpoint mode (main benchmark style)
            for viewpoint in self.viewpoints:
                # Check for pre-existing result image in results directory first
                preexisting_result = os.path.join(self.results_dir, f"{self.case_name}_{viewpoint}_view.png")

                gt_path = os.path.join(self.gs_dir, f"gs_{viewpoint}_view.png")

                # Use pre-existing result image if it exists, otherwise use screenshot path
                if os.path.exists(preexisting_result):
                    result_path = preexisting_result
                else:
                    result_path = os.path.join(self.screenshot_dir, f"result_{viewpoint}_view.png")

                image_paths[viewpoint] = {
                    "gt": gt_path,
                    "result": result_path
                }

        return image_paths
    
    def check_images_exist(self) -> Dict[str, bool]:
        """
        Check which viewpoint images exist
        
        Returns:
            Dict[str, bool]: Dictionary indicating which viewpoints have both images
        """
        image_paths = self.get_image_paths()
        existence = {}
        
        for viewpoint, paths in image_paths.items():
            gt_exists = os.path.exists(paths["gt"])
            result_exists = os.path.exists(paths["result"])
            existence[viewpoint] = gt_exists and result_exists
            
            if not gt_exists:
                print(f"Warning: Ground truth image missing for {self.case_name} {viewpoint} view: {paths['gt']}")
            if not result_exists:
                print(f"Warning: Result image missing for {self.case_name} {viewpoint} view: {paths['result']}")
        
        return existence
    
    def calculate_viewpoint_metrics(self, viewpoint: str) -> Dict[str, float]:
        """
        Calculate metrics for a specific viewpoint
        
        Args:
            viewpoint (str): Viewpoint name ("diagonal", "front", "side")
            
        Returns:
            Dict[str, float]: Metrics for this viewpoint
        """
        image_paths = self.get_image_paths()
        
        if viewpoint not in image_paths:
            raise ValueError(f"Unknown viewpoint: {viewpoint}")
        
        gt_path = image_paths[viewpoint]["gt"]
        result_path = image_paths[viewpoint]["result"]
        
        return self.calculator.calculate_all_metrics(gt_path, result_path)
    
    def calculate_case_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics for all viewpoints and compute averages

        Returns:
            Dict[str, Any]: Complete metrics for this case
        """
        print(f"Calculating image metrics for {self.case_name} ({self.eval_mode} mode)...")

        # Check if we're in single-image mode
        single_gt_path = os.path.join(self.gs_dir, f"{self.case_name}_gs.png")
        single_result_path = os.path.join(self.results_dir, f"{self.case_name}.png")
        is_single_image_mode = os.path.exists(single_gt_path) and os.path.exists(single_result_path)

        # Calculate metrics for each viewpoint
        viewpoint_metrics = {}
        valid_viewpoints = []

        if is_single_image_mode:
            # Single-image mode: calculate once and use for all viewpoints
            print(f"Single-image mode: using {single_result_path}")
            try:
                metrics = self.calculator.calculate_all_metrics(single_gt_path, single_result_path)
                # Use the same metrics for all viewpoints
                for viewpoint in self.viewpoints:
                    viewpoint_metrics[viewpoint] = metrics
                    valid_viewpoints.append(viewpoint)
                print(f"  PSNR: {metrics.get('psnr', 'N/A')}, SSIM: {metrics.get('ssim', 'N/A')}, LPIPS: {metrics.get('lpips', 'N/A')}")
            except Exception as e:
                print(f"  Error calculating single-image metrics: {e}")
                for viewpoint in self.viewpoints:
                    viewpoint_metrics[viewpoint] = {'psnr': None, 'ssim': None, 'lpips': None}
        else:
            # Multi-viewpoint mode: calculate for each viewpoint
            existence = self.check_images_exist()

            for viewpoint in self.viewpoints:
                if existence[viewpoint]:
                    try:
                        metrics = self.calculate_viewpoint_metrics(viewpoint)
                        viewpoint_metrics[viewpoint] = metrics
                        valid_viewpoints.append(viewpoint)
                        print(f"  {viewpoint} view - PSNR: {metrics.get('psnr', 'N/A')}, SSIM: {metrics.get('ssim', 'N/A')}, LPIPS: {metrics.get('lpips', 'N/A')}")
                    except Exception as e:
                        print(f"  Error calculating {viewpoint} view metrics: {e}")
                    viewpoint_metrics[viewpoint] = {'psnr': None, 'ssim': None, 'lpips': None}
            else:
                viewpoint_metrics[viewpoint] = {'psnr': None, 'ssim': None, 'lpips': None}
        
        # Calculate averages across valid viewpoints
        averages = self._calculate_averages(viewpoint_metrics, valid_viewpoints)
        
        # Prepare result
        result = {
            "case_name": self.case_name,
            "eval_mode": self.eval_mode,
            "calculation_time": datetime.now().isoformat(),
            "viewpoint_metrics": viewpoint_metrics,
            "averaged_metrics": averages,
            "valid_viewpoints": valid_viewpoints,
            "valid_viewpoint_count": len(valid_viewpoints),
            "total_viewpoints": len(self.viewpoints)
        }
        
        return result
    
    def _calculate_averages(self, viewpoint_metrics: Dict[str, Dict[str, float]], 
                          valid_viewpoints: List[str]) -> Dict[str, float]:
        """
        Calculate average metrics across valid viewpoints
        
        Args:
            viewpoint_metrics (dict): Metrics for each viewpoint
            valid_viewpoints (list): List of viewpoints with valid data
            
        Returns:
            Dict[str, float]: Averaged metrics
        """
        if not valid_viewpoints:
            return {'psnr': None, 'ssim': None, 'lpips': None}
        
        averages = {}
        
        for metric in ['psnr', 'ssim', 'lpips']:
            values = []
            for viewpoint in valid_viewpoints:
                value = viewpoint_metrics[viewpoint].get(metric)
                if value is not None:
                    values.append(value)
            
            if values:
                averages[metric] = sum(values) / len(values)
            else:
                averages[metric] = None
        
        return averages


class BatchImageMetrics:
    """
    Calculate image metrics across all test cases and provide batch statistics
    """
    
    def __init__(self, cases_dir: str, eval_mode: str = "mcp", output_dir: str = None):
        """
        Initialize batch metrics calculator
        
        Args:
            cases_dir (str): Path to the cases directory
            eval_mode (str): Evaluation mode ("mcp" or "pvpython")
            output_dir (str): Output directory for results
        """
        self.cases_dir = Path(cases_dir)
        self.eval_mode = eval_mode
        self.output_dir = Path(output_dir) if output_dir else self.cases_dir.parent / "evaluation_results" / f"{eval_mode}_image_metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def discover_test_cases(self) -> List[str]:
        """
        Discover all test cases in the cases directory
        
        Returns:
            List[str]: List of test case names
        """
        test_cases = []
        
        if not self.cases_dir.exists():
            print(f"Warning: Cases directory not found: {self.cases_dir}")
            return test_cases
        
        for item in self.cases_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                test_cases.append(item.name)
        
        return sorted(test_cases)
    
    def calculate_batch_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics for all test cases
        
        Returns:
            Dict[str, Any]: Batch metrics results
        """
        test_cases = self.discover_test_cases()
        
        if not test_cases:
            print("No test cases found")
            return {}
        
        print(f"Calculating image metrics for {len(test_cases)} test cases in {self.eval_mode} mode...")
        
        # Calculate metrics for each case
        case_results = {}
        passed_cases = []
        failed_cases = []
        
        for case_name in test_cases:
            case_dir = self.cases_dir / case_name
            
            try:
                case_calculator = CaseImageMetrics(str(case_dir), case_name, self.eval_mode)
                case_metrics = case_calculator.calculate_case_metrics()
                case_results[case_name] = case_metrics
                
                # Check if case has valid results
                if case_metrics["valid_viewpoint_count"] > 0:
                    passed_cases.append(case_name)
                else:
                    failed_cases.append(case_name)
                    
            except Exception as e:
                print(f"Error calculating metrics for {case_name}: {e}")
                failed_cases.append(case_name)
                case_results[case_name] = {
                    "case_name": case_name,
                    "error": str(e),
                    "valid_viewpoint_count": 0
                }
        
        # Calculate batch statistics
        batch_stats = self._calculate_batch_statistics(case_results, passed_cases, test_cases)
        
        # Prepare final results
        batch_results = {
            "evaluation_time": datetime.now().isoformat(),
            "eval_mode": self.eval_mode,
            "batch_statistics": batch_stats,
            "case_results": case_results,
            "summary": {
                "total_cases": len(test_cases),
                "passed_cases": len(passed_cases),
                "failed_cases": len(failed_cases),
                "success_rate": len(passed_cases) / len(test_cases) if test_cases else 0.0
            }
        }
        
        # Save results
        self._save_batch_results(batch_results)
        
        # Print summary
        self._print_batch_summary(batch_results)
        
        return batch_results
    
    def _calculate_batch_statistics(self, case_results: Dict[str, Any], 
                                  passed_cases: List[str], all_cases: List[str]) -> Dict[str, Any]:
        """
        Calculate statistics across all passed cases
        
        Args:
            case_results (dict): Results for all cases
            passed_cases (list): List of cases with valid results
            all_cases (list): List of all cases
            
        Returns:
            Dict[str, Any]: Batch statistics
        """
        if not passed_cases:
            return {
                "unscaled_averages": {"psnr": None, "ssim": None, "lpips": None},
                "scaled_averages": {"psnr": None, "ssim": None, "lpips": None},
                "completion_rate": 0.0
            }
        
        # Collect metrics from passed cases
        all_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
        
        for case_name in passed_cases:
            case_data = case_results[case_name]
            averaged_metrics = case_data.get("averaged_metrics", {})
            
            for metric in ['psnr', 'ssim', 'lpips']:
                value = averaged_metrics.get(metric)
                if value is not None:
                    all_metrics[metric].append(value)
        
        # Calculate unscaled averages
        unscaled_averages = {}
        for metric in ['psnr', 'ssim', 'lpips']:
            if all_metrics[metric]:
                unscaled_averages[metric] = sum(all_metrics[metric]) / len(all_metrics[metric])
            else:
                unscaled_averages[metric] = None
        
        # Calculate completion rate
        completion_rate = len(passed_cases) / len(all_cases)
        
        # Calculate scaled averages
        scaled_averages = {}
        for metric in ['psnr', 'ssim']:
            if unscaled_averages[metric] is not None:
                scaled_averages[metric] = completion_rate * unscaled_averages[metric]
            else:
                scaled_averages[metric] = None
        
        # Special handling for LPIPS (lower is better)
        if unscaled_averages['lpips'] is not None:
            scaled_averages['lpips'] = 1.0 - completion_rate * (1.0 - unscaled_averages['lpips'])
        else:
            scaled_averages['lpips'] = None
        
        return {
            "unscaled_averages": unscaled_averages,
            "scaled_averages": scaled_averages,
            "completion_rate": completion_rate,
            "metric_counts": {metric: len(values) for metric, values in all_metrics.items()}
        }
    
    def _save_batch_results(self, batch_results: Dict[str, Any]):
        """Save batch results to file"""
        filename = f"batch_image_metrics_{self.eval_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"Batch image metrics results saved to: {output_path}")
    
    def _print_batch_summary(self, batch_results: Dict[str, Any]):
        """Print a summary of batch results"""
        stats = batch_results["batch_statistics"]
        summary = batch_results["summary"]
        
        print(f"\n{'='*60}")
        print(f"BATCH IMAGE METRICS SUMMARY ({self.eval_mode.upper()} MODE)")
        print(f"{'='*60}")
        print(f"Total cases: {summary['total_cases']}")
        print(f"Passed cases: {summary['passed_cases']}")
        print(f"Failed cases: {summary['failed_cases']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Completion rate: {stats['completion_rate']:.2%}")
        
        print(f"\nUnscaled Average Metrics (across passed cases):")
        unscaled = stats["unscaled_averages"]
        for metric, value in unscaled.items():
            if value is not None:
                print(f"  {metric.upper()}: {value:.4f}")
            else:
                print(f"  {metric.upper()}: N/A")
        
        print(f"\nScaled Average Metrics (weighted by completion rate):")
        scaled = stats["scaled_averages"]
        for metric, value in scaled.items():
            if value is not None:
                print(f"  {metric.upper()}: {value:.4f}")
            else:
                print(f"  {metric.upper()}: N/A")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate image metrics for scientific visualization evaluation")
    parser.add_argument("--cases", required=True, help="Path to the cases directory")
    parser.add_argument("--mode", choices=["mcp", "pvpython"], default="mcp", 
                       help="Evaluation mode (default: mcp)")
    parser.add_argument("--case", help="Calculate metrics for a specific case only")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                       help="Device to use for LPIPS calculation (default: auto)")
    
    args = parser.parse_args()
    
    # Validate cases directory
    if not os.path.exists(args.cases):
        print(f"Error: Cases directory not found: {args.cases}")
        sys.exit(1)
    
    if args.case:
        # Calculate metrics for specific case
        case_dir = os.path.join(args.cases, args.case)
        if not os.path.exists(case_dir):
            print(f"Error: Case directory not found: {case_dir}")
            sys.exit(1)
        
        calculator = CaseImageMetrics(case_dir, args.case, args.mode)
        calculator.calculator = ImageMetricsCalculator(device=args.device)
        result = calculator.calculate_case_metrics()
        
        # Save individual case result
        output_dir = args.output or os.path.join(case_dir, "evaluation_results", args.mode)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"image_metrics_{args.case}_{args.mode}.json")
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to: {output_file}")
    
    else:
        # Calculate batch metrics
        batch_calculator = BatchImageMetrics(args.cases, args.mode, args.output)
        batch_calculator.calculate_batch_metrics()


if __name__ == "__main__":
    main()