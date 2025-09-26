"""
Helper script for taking screenshots from ParaView state files
"""
from paraview.simple import *
import os
import math

def take_screenshots_from_state(state_path, output_dir, prefix="", data_directory=None):
    """
    Load a ParaView state file and take 3 screenshots from different angles
    
    Args:
        state_path (str): Path to the .pvsm state file
        output_dir (str): Directory to save screenshots
        prefix (str): Prefix for screenshot filenames
        data_path (str): Directory of raw data file for state file
    
    Returns:
        list: List of screenshot file paths
    """
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"State file not found: {state_path}")
    
    # Load state file
    if data_directory:
        LoadState(state_path, data_directory=data_directory)
    else:
        LoadState(state_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the active view
    renderView = GetActiveViewOrCreate('RenderView')
    
    # Set background color to white using SetViewProperties
    from paraview.simple import SetViewProperties
    SetViewProperties(
        renderView,
        Background=[1.0, 1.0, 1.0],
        Background2=[1.0, 1.0, 1.0],  # Set same color to avoid gradient
        UseColorPaletteForBackground=0
    )
    
    # Reset camera to fit all data
    renderView.ResetCamera()
    
    # Get current camera position for reference
    camera = renderView.GetActiveCamera()
    original_position = camera.GetPosition()
    original_focal_point = camera.GetFocalPoint()
    
    # Calculate distance from focal point to position
    distance = math.sqrt(sum([(original_position[i] - original_focal_point[i])**2 for i in range(3)]))
    
    # Define three different camera angles
    angles = [
        {
            'name': 'front',
            'position': [original_focal_point[0], original_focal_point[1], original_focal_point[2] + distance],
            'up': [0, 1, 0]
        },
        {
            'name': 'side',
            'position': [original_focal_point[0] + distance, original_focal_point[1], original_focal_point[2]],
            'up': [0, 0, 1]
        },
        {
            'name': 'diagonal',
            'position': [original_focal_point[0] + distance*0.7, original_focal_point[1] + distance*0.7, original_focal_point[2] + distance*0.7],
            'up': [0, 0, 1]
        }
    ]
    
    screenshot_paths = []
    
    # Take screenshots from different angles
    for angle in angles:
        # Set camera position
        camera.SetPosition(angle['position'])
        camera.SetFocalPoint(original_focal_point)
        camera.SetViewUp(angle['up'])
        
        # Reset camera to ensure proper framing
        renderView.ResetCamera()
        
        # Render the view
        Render()
        
        # Save screenshot
        filename = f"{prefix}{angle['name']}_view.png" if prefix else f"{angle['name']}_view.png"
        screenshot_path = os.path.join(output_dir, filename)
        SaveScreenshot(screenshot_path, renderView, ImageResolution=[1920, 1080])
        screenshot_paths.append(screenshot_path)
    
    return screenshot_paths

def compare_states_screenshots(gs_state_path, result_state_path, output_dir, data_directory=None):
    """
    Take screenshots from both ground truth and result state files
    
    Args:
        gs_state_path (str): Path to ground truth state file
        result_state_path (str): Path to result state file
        output_dir (str): Directory to save screenshots
        data_directory (str): Directory containing data files
        
    Returns:
        dict: Dictionary with 'ground_truth' and 'result' screenshot paths
    """
    screenshots = {
        'ground_truth': [],
        'result': []
    }
    
    # Take screenshots from ground truth state
    print("Taking screenshots from ground truth state...")
    screenshots['ground_truth'] = take_screenshots_from_state(
        gs_state_path, output_dir, prefix="gs_", data_directory=data_directory
    )
    
    # Clear the current state
    #Delete(GetActiveSource())
    
    # Take screenshots from result state
    print("Taking screenshots from result state...")
    screenshots['result'] = take_screenshots_from_state(
        result_state_path, output_dir, prefix="result_", data_directory=data_directory
    )
    
    return screenshots
