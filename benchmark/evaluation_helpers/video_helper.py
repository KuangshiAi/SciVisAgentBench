#!/usr/bin/env python3
"""
Helper script for extracting frames from video files for evaluation
"""
import os
import cv2
from typing import List, Optional


def extract_frames_from_video(video_path: str, output_dir: str, prefix: str = "", num_frames: int = 10) -> List[str]:
    """
    Extract evenly spaced frames from a video file
    
    Args:
        video_path (str): Path to the .avi or other video file
        output_dir (str): Directory to save extracted frames
        prefix (str): Prefix for frame filenames
        num_frames (int): Number of frames to extract (default: 10)
    
    Returns:
        list: List of extracted frame file paths
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Calculate frame indices to extract (evenly spaced)
    if total_frames <= num_frames:
        # If video has fewer frames than requested, extract all
        frame_indices = list(range(total_frames))
    else:
        # Extract evenly spaced frames
        step = (total_frames - 1) / (num_frames - 1)
        frame_indices = [int(i * step) for i in range(num_frames)]
    
    extracted_paths = []
    
    # Extract frames
    for i, frame_idx in enumerate(frame_indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_idx} from {video_path}")
            continue
        
        # Save frame
        filename = f"{prefix}frame_{i:03d}.png" if prefix else f"frame_{i:03d}.png"
        frame_path = os.path.join(output_dir, filename)
        cv2.imwrite(frame_path, frame)
        extracted_paths.append(frame_path)
    
    cap.release()
    
    return extracted_paths


def load_static_screenshots_or_video(base_path: str, dataset_name: str, output_dir: str, 
                                     prefix: str = "", num_video_frames: int = 10) -> Optional[List[str]]:
    """
    Load static screenshots (.png) or extract frames from video (.avi)
    
    Args:
        base_path (str): Base directory path (e.g., "{dataset}/GS" or "{dataset}/results/{mode}")
        dataset_name (str): Name of the dataset
        output_dir (str): Directory to save extracted frames (for videos only)
        prefix (str): Prefix for extracted frame filenames
        num_video_frames (int): Number of frames to extract from video (default: 10)
    
    Returns:
        list: List of image file paths, or None if not found
    """
    # Try PNG first
    png_path = os.path.join(base_path, f"{dataset_name}.png")
    if os.path.exists(png_path):
        return [png_path]
    
    # Try _gs.png for GS directory
    gs_png_path = os.path.join(base_path, f"{dataset_name}_gs.png")
    if os.path.exists(gs_png_path):
        return [gs_png_path]
    
    # Try AVI video
    avi_path = os.path.join(base_path, f"{dataset_name}.avi")
    if os.path.exists(avi_path):
        print(f"Found video file: {avi_path}, extracting {num_video_frames} frames...")
        return extract_frames_from_video(avi_path, output_dir, prefix, num_video_frames)
    
    # Try _gs.avi for GS directory
    gs_avi_path = os.path.join(base_path, f"{dataset_name}_gs.avi")
    if os.path.exists(gs_avi_path):
        print(f"Found video file: {gs_avi_path}, extracting {num_video_frames} frames...")
        return extract_frames_from_video(gs_avi_path, output_dir, prefix, num_video_frames)
    
    return None
