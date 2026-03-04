"""
Helper script for taking screenshots from ParaView state files
"""
from paraview.simple import *
import os
import math
import sys
import subprocess
import json

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

def _take_screenshots_subprocess(state_path, output_dir, prefix="", data_directory=None):
    """
    Run screenshot generation in a subprocess to isolate ParaView crashes.

    This is a safer wrapper around take_screenshots_from_state that runs in
    a subprocess. If ParaView crashes (e.g., Bus error on macOS), it won't
    kill the parent evaluation process.
    """
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a temporary Python script to run in subprocess
    script_content = f'''
import sys
import os
import math

# Add script directory to path
sys.path.insert(0, {repr(script_dir)})

try:
    from paraview.simple import *

    state_path = {repr(state_path)}
    output_dir = {repr(output_dir)}
    prefix = {repr(prefix)}
    data_directory = {repr(data_directory)}

    # Inline implementation to avoid import issues
    def take_screenshots_impl(state_path, output_dir, prefix, data_directory):
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"State file not found: {{state_path}}")

        # Load state file
        if data_directory:
            LoadState(state_path, data_directory=data_directory)
        else:
            LoadState(state_path)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get the active view
        renderView = GetActiveViewOrCreate('RenderView')

        # Set background color to white
        from paraview.simple import SetViewProperties
        SetViewProperties(
            renderView,
            Background=[1.0, 1.0, 1.0],
            Background2=[1.0, 1.0, 1.0],
            UseColorPaletteForBackground=0
        )

        # Reset camera to fit all data
        renderView.ResetCamera()

        # Get current camera position
        camera = renderView.GetActiveCamera()
        original_position = camera.GetPosition()
        original_focal_point = camera.GetFocalPoint()

        # Calculate distance
        distance = math.sqrt(sum([(original_position[i] - original_focal_point[i])**2 for i in range(3)]))

        # Define angles
        angles = [
            {{'name': 'front', 'position': [original_focal_point[0], original_focal_point[1], original_focal_point[2] + distance], 'up': [0, 1, 0]}},
            {{'name': 'side', 'position': [original_focal_point[0] + distance, original_focal_point[1], original_focal_point[2]], 'up': [0, 0, 1]}},
            {{'name': 'diagonal', 'position': [original_focal_point[0] + distance*0.7, original_focal_point[1] + distance*0.7, original_focal_point[2] + distance*0.7], 'up': [0, 0, 1]}}
        ]

        screenshot_paths = []

        for angle in angles:
            camera.SetPosition(angle['position'])
            camera.SetFocalPoint(original_focal_point)
            camera.SetViewUp(angle['up'])
            renderView.ResetCamera()
            Render()

            filename = f"{{prefix}}{{angle['name']}}_view.png" if prefix else f"{{angle['name']}}_view.png"
            screenshot_path = os.path.join(output_dir, filename)
            SaveScreenshot(screenshot_path, renderView, ImageResolution=[1920, 1080])
            screenshot_paths.append(screenshot_path)

        return screenshot_paths

    # Try to set offscreen rendering (helps on some systems)
    try:
        import paraview
        paraview.simple._DisableFirstRenderCameraReset()
    except:
        pass

    screenshots = take_screenshots_impl(state_path, output_dir, prefix, data_directory)

    # Write result to stdout as JSON
    import json
    print(json.dumps({{"success": True, "screenshots": screenshots}}))

except Exception as e:
    import json
    import traceback
    print(json.dumps({{"success": False, "error": str(e), "traceback": traceback.format_exc()}}), file=sys.stderr)
    sys.exit(1)
'''

    # Write script to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=output_dir) as f:
        script_path = f.name
        f.write(script_content)

    try:
        # Run in subprocess with timeout
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.path.dirname(__file__)
        )

        # Parse output
        if result.returncode == 0:
            try:
                output_data = json.loads(result.stdout.strip().split('\n')[-1])
                if output_data.get('success'):
                    return output_data['screenshots']
                else:
                    raise RuntimeError(f"Screenshot generation failed: {output_data.get('error')}")
            except json.JSONDecodeError:
                # Fallback: return empty list and let caller handle
                raise RuntimeError(f"Failed to parse subprocess output. stdout: {result.stdout}, stderr: {result.stderr}")
        else:
            raise RuntimeError(f"Subprocess failed with code {result.returncode}. stderr: {result.stderr}")

    finally:
        # Clean up temporary script
        try:
            os.unlink(script_path)
        except:
            pass

def compare_states_screenshots_safe(gs_state_path, result_state_path, output_dir, data_directory=None):
    """
    Safe version of compare_states_screenshots that uses subprocesses to isolate crashes.

    This wrapper function runs screenshot generation in separate subprocesses to prevent
    ParaView crashes (like Bus errors on macOS) from killing the evaluation process.

    Args:
        gs_state_path (str): Path to ground truth state file
        result_state_path (str): Path to result state file
        output_dir (str): Directory to save screenshots
        data_directory (str): Directory containing data files

    Returns:
        dict: Dictionary with 'ground_truth' and 'result' screenshot paths

    Raises:
        RuntimeError: If screenshot generation fails in subprocess
    """
    screenshots = {
        'ground_truth': [],
        'result': []
    }

    # Take screenshots from ground truth state in subprocess
    print("Taking screenshots from ground truth state (in subprocess)...")
    try:
        screenshots['ground_truth'] = _take_screenshots_subprocess(
            gs_state_path, output_dir, prefix="gs_", data_directory=data_directory
        )
    except Exception as e:
        print(f"Warning: Failed to generate ground truth screenshots: {e}")
        raise

    # Take screenshots from result state in subprocess
    print("Taking screenshots from result state (in subprocess)...")
    try:
        screenshots['result'] = _take_screenshots_subprocess(
            result_state_path, output_dir, prefix="result_", data_directory=data_directory
        )
    except Exception as e:
        print(f"Warning: Failed to generate result screenshots: {e}")
        raise

    return screenshots
