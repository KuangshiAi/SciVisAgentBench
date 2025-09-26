from pathlib import Path
from napari.viewer import Viewer
import numpy as np
import collections.abc
import json
import tifffile

def open_file(
    path: str | Path,                 # ① first (supplied by you)
    viewer: Viewer,                   # ② injected by napari
    *,                                # keyword-only from here on
    plugin: str | None = None,
    layer_type: str | None = None,
):
    """Open *any* image file (TIFF, PNG, ND2…) with napari's readers."""
    layers = viewer.open(
        Path(path),
        plugin=plugin or "napari",     # built-in reader first
        layer_type=layer_type,
    )
    return layers[0] if layers else None

def remove_layer(name_or_index: str | int, viewer: Viewer):
    """Remove a layer by its name or positional index."""
    try:
        layer = _get_layer(viewer, name_or_index)
        viewer.layers.remove(layer)
        return f"Layer '{layer.name}' removed successfully."
    except (KeyError, IndexError) as e:
        # Get available layer names for better error message
        available_layers = [layer.name for layer in viewer.layers]
        return f"Layer '{name_or_index}' not found. Available layers: {available_layers}"

def toggle_ndisplay(viewer: Viewer):
    """Toggle napari between 2-D (ndisplay = 2) and 3-D (ndisplay = 3)."""
    viewer.dims.ndisplay = 3 if viewer.dims.ndisplay == 2 else 2
    return viewer.dims.ndisplay


# ----------------------------------------------------------------------
# iso-surface (contour) rendering
# ----------------------------------------------------------------------

def iso_contour(
    layer_name: str | int | None = None,
    threshold: float | None = None,
    viewer: Viewer | None = None,          # injected by napari
) -> int:
    """
    Switch layers to iso-surface rendering mode.
    
    Enables 3D iso-surface visualization for image/volume layers.
    Returns the number of layers that were modified.
    """
    # ensure a 3-D canvas so the surface is visible
    if viewer.dims.ndisplay != 3:
        viewer.dims.ndisplay = 3

    # resolve which layers to edit
    if layer_name is None:
        # Only apply to Image layers that support iso-surface rendering
        targets = [lyr for lyr in viewer.layers if hasattr(lyr, "rendering") and hasattr(lyr, "iso_threshold")]
    else:
        layer = viewer.layers[layer_name] if isinstance(layer_name, int) else viewer.layers[layer_name]
        # Check if the layer supports iso-surface rendering
        if hasattr(layer, "rendering") and hasattr(layer, "iso_threshold"):
            targets = [layer]
        else:
            return 0  # No layers modified

    # apply rendering mode / threshold
    for lyr in targets:
        lyr.rendering = "iso"
        if threshold is not None:
            lyr.iso_threshold = threshold

    return len(targets)


# ----------------------------------------------------------------------
# screenshot helper
# ----------------------------------------------------------------------

def screenshot(
    viewer: Viewer | None = None,          # injected by napari
    canvas_only: bool = True,
) -> str:
    """
    Take a screenshot of the current napari viewer.
    
    Saves a JPG file to the temp folder and returns the file path.
    Set canvas_only=False to capture the full UI instead of just the canvas.
    """
    import os
    import tempfile
    from PIL import Image

    screenshot_array = viewer.screenshot(canvas_only=canvas_only)
    img = Image.fromarray(screenshot_array)
    fd, tmp = tempfile.mkstemp(prefix="napari_scr_", suffix=".jpg")
    os.close(fd)
    img = img.convert("RGB")  # Ensure no alpha channel for JPG
    img.save(tmp, format="JPEG")
    return os.path.abspath(tmp)

# ----------------------------------------------------------------------
# layer introspection
# ----------------------------------------------------------------------

def to_serializable(obj):
    """Recursively convert an object to something JSON-serializable."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    # numpy types
    elif hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    # fallback: string representation
    return str(obj)

def list_layers(viewer: Viewer):
    """
    Return information about all loaded layers.

    Returns
    -------
    list[dict]
        One dict per layer with ``index``, ``name``, ``type``, and ``visible``.
    """
    return to_serializable([
        {
            "index": i,
            "name": layer.name,
            "type": layer.__class__.__name__,
            "visible": layer.visible,
        }
        for i, layer in enumerate(viewer.layers)
    ])

def get_layer_names(viewer: Viewer):
    """
    Return a simple list of available layer names.
    
    Returns
    -------
    list[str]
        List of layer names that can be used with other commands.
    """
    return [layer.name for layer in viewer.layers]

def set_colormap(
    layer_name: str | int,
    colormap: str,
    viewer: Viewer,
):
    """Change the colormap for a layer."""
    try:
        layer = _get_layer(viewer, layer_name)
        if hasattr(layer, 'colormap'):
            layer.colormap = colormap
            return f"Colormap for layer '{layer.name}' set to '{colormap}'."
        
        return f"Layer '{layer.name}' of type {type(layer).__name__} does not have a colormap attribute."
    except (KeyError, IndexError) as e:
        # Get available layer names for better error message
        available_layers = [layer.name for layer in viewer.layers]
        return f"Layer '{layer_name}' not found. Available layers: {available_layers}"

def _get_layer(viewer: Viewer, layer_name: str | int | None = None):
    """Get a layer by name/index or return the active layer."""
    if layer_name is not None:
        try:
            return viewer.layers[layer_name]
        except (KeyError, IndexError):
            # Re-raise with more context
            available_layers = [layer.name for layer in viewer.layers]
            raise KeyError(f"Layer '{layer_name}' not found. Available layers: {available_layers}")
    return viewer.layers.selection.active

def set_opacity(
    layer_name: str | int,
    opacity: float,
    viewer: Viewer,
    ):
    """Adjust layer transparency (0=transparent, 1=opaque)."""
    try:
        layer = _get_layer(viewer, layer_name)
        if hasattr(layer, 'opacity'):
            layer.opacity = opacity
            return f"Opacity for layer '{layer.name}' set to {opacity}."
        return f"Layer '{layer.name}' does not have an opacity attribute."
    except (KeyError, IndexError) as e:
        # Get available layer names for better error message
        available_layers = [layer.name for layer in viewer.layers]
        return f"Layer '{layer_name}' not found. Available layers: {available_layers}"

def set_blending(
        layer_name: str | int,
        blending: str,
        viewer: Viewer,
        ):
    """Set how the layer blends with layers below it."""
    layer = _get_layer(viewer, layer_name)
    if hasattr(layer, 'blending'):
        layer.blending = blending
        return f"Blending mode for layer '{layer.name}' set to '{blending}'."
    return f"Layer '{layer.name}' does not have a blending attribute."

def set_contrast_limits(
    layer_name: str | int, 
    contrast_min: float, 
    contrast_max: float, 
    viewer: Viewer,
    ):
    """Set the min/max values for contrast scaling."""
    layer = _get_layer(viewer, layer_name)
    if hasattr(layer, 'contrast_limits'):
        layer.contrast_limits = (contrast_min, contrast_max)
        return f"Contrast limits for layer '{layer.name}' set to ({contrast_min}, {contrast_max})."
    return f"Layer '{layer.name}' does not have a contrast_limits attribute."

def auto_contrast(
        layer_name: str | int,
        viewer: Viewer, 
    ):
    """Automatically adjust contrast to fit the data range."""
    layer = _get_layer(viewer, layer_name)
    if hasattr(layer, 'reset_contrast_limits'):
        layer.reset_contrast_limits()
        return f"Auto-contrasted layer '{layer.name}'. New limits: {layer.contrast_limits}."
    return f"Layer '{layer.name}' does not have auto-contrast capability."

def set_gamma(
        layer_name: str | int, 
        gamma: float, 
        viewer: Viewer, 
        ):
    """Adjust gamma correction for the layer."""
    layer = _get_layer(viewer, layer_name)
    if hasattr(layer, 'gamma'):
        layer.gamma = gamma
        return f"Gamma for layer '{layer.name}' set to {gamma}."
    return f"Layer '{layer.name}' does not have a gamma attribute."

def set_interpolation(
        layer_name: str | int,
        interpolation: str, 
        viewer: Viewer, 
        ):
    """Set the interpolation method for zooming."""
    layer = _get_layer(viewer, layer_name)
    if hasattr(layer, 'interpolation'):
        layer.interpolation = interpolation
        return f"Interpolation for layer '{layer.name}' set to '{interpolation}'."
    return f"Layer '{layer.name}' does not have an interpolation attribute."

def set_timestep(
    timestep: int,
    viewer: Viewer,
    ):
    """Set the timestep for the viewer."""
    current_step = list(viewer.dims.current_step)
    if not current_step:
        return "Viewer has no dimensions with steps."
    
    if timestep >= viewer.dims.nsteps[0]:
        return f"Timestep {timestep} is out of bounds (max: {viewer.dims.nsteps[0] - 1})."
        
    current_step[0] = timestep
    viewer.dims.current_step = tuple(current_step)
    return f"Timestep set to {timestep}."

def get_dims_info(viewer: Viewer):
    """Get information about the viewer's dimensions."""
    dims_info = {
        'ndim': viewer.dims.ndim,
        'nsteps': viewer.dims.nsteps,
        'current_step': viewer.dims.current_step,
        'axis_labels': list(viewer.dims.axis_labels),
    }
    return to_serializable(dims_info)

def set_camera(
    center=None,
    zoom=None,
    angle=None,
    viewer: Viewer = None,
):
    """Set the camera parameters: center (tuple), zoom (float), angle (tuple for 3D)."""
    if center is not None:
        viewer.camera.center = center
    if zoom is not None:
        viewer.camera.zoom = zoom
    if angle is not None and getattr(viewer.dims, 'ndisplay', 2) == 3:
        # Only set angles if it's a tuple/list of length 3 in 3D mode
        if isinstance(angle, (tuple, list)) and len(angle) == 3:
            viewer.camera.angles = angle
        # else: optionally log a warning or ignore
    return {
        'center': tuple(viewer.camera.center),
        'zoom': viewer.camera.zoom,
        'angles': tuple(viewer.camera.angles) if hasattr(viewer.camera, 'angles') else None,
    }

def get_camera(viewer: Viewer = None):
    """Get the current camera parameters."""
    return {
        'center': tuple(viewer.camera.center),
        'zoom': viewer.camera.zoom,
        'angles': tuple(viewer.camera.angles) if hasattr(viewer.camera, 'angles') else None,
    }

def reset_camera(viewer: Viewer = None):
    """Reset the camera to the default view."""
    viewer.camera.reset()
    return {
        'center': tuple(viewer.camera.center),
        'zoom': viewer.camera.zoom,
        'angles': tuple(viewer.camera.angles) if hasattr(viewer.camera, 'angles') else None,
    }

# ----------------------------------------------------------------------
# Layer Creation & Annotation Functions
# ----------------------------------------------------------------------

def add_points(
    coordinates: list | np.ndarray,
    properties: dict | None = None,
    name: str | None = None,
    viewer: Viewer = None,
):
    """Add point annotations to the viewer."""
    if isinstance(coordinates, list):
        coordinates = np.array(coordinates)
    
    layer = viewer.add_points(coordinates, properties=properties, name=name)
    return f"Added points layer '{layer.name}' with {len(coordinates)} points."

def add_shapes(
    shape_data: list | np.ndarray,
    shape_type: str = 'rectangle',
    name: str | None = None,
    viewer: Viewer = None,
):
    """Add shape annotations to the viewer."""
    if isinstance(shape_data, list):
        shape_data = np.array(shape_data)
    
    layer = viewer.add_shapes(shape_data, shape_type=shape_type, name=name)
    return f"Added shapes layer '{layer.name}' with {len(shape_data)} shapes."

def add_labels(
    label_image: np.ndarray | list,
    name: str | None = None,
    viewer: Viewer = None,
):
    """Add label image (segmentation mask) to the viewer."""
    # Convert list to numpy array if needed
    if isinstance(label_image, list):
        label_image = np.array(label_image)
    layer = viewer.add_labels(label_image, name=name)
    return f"Added labels layer '{layer.name}' with shape {label_image.shape}."

def add_surface(
    vertices: np.ndarray | list,
    faces: np.ndarray | list,
    name: str | None = None,
    viewer: Viewer = None,
):
    """Add 3D surface mesh to the viewer."""
    # Convert lists to numpy arrays if needed
    if isinstance(vertices, list):
        vertices = np.array(vertices)
    if isinstance(faces, list):
        faces = np.array(faces)
    layer = viewer.add_surface((vertices, faces), name=name)
    return f"Added surface layer '{layer.name}' with {len(vertices)} vertices and {len(faces)} faces."

def add_vectors(
    vectors: np.ndarray | list,
    name: str | None = None,
    viewer: Viewer = None,
):
    """Add vector field to the viewer."""
    # Convert list to numpy array if needed
    if isinstance(vectors, list):
        vectors = np.array(vectors)
    layer = viewer.add_vectors(vectors, name=name)
    return f"Added vectors layer '{layer.name}' with shape {vectors.shape}."

# ----------------------------------------------------------------------
# Data Export & Save Functions
# ----------------------------------------------------------------------

def save_layers(
    file_path: str | Path,
    layer_names: list | None = None,
    viewer: Viewer = None,
):
    """Save layers to file."""
    path = Path(file_path)
    
    if layer_names is None:
        # Save all layers
        layers_to_save = list(viewer.layers)
    else:
        # Save specific layers
        layers_to_save = [viewer.layers[name] for name in layer_names]
    
    # Save each layer individually based on file extension
    saved_count = 0
    for i, layer in enumerate(layers_to_save):
        if hasattr(layer, 'data') and layer.data is not None:
            # For image data, save as TIFF
            if path.suffix.lower() in ['.tif', '.tiff']:
                layer_data = layer.data
                if hasattr(layer_data, 'compute'):  # Handle dask arrays
                    layer_data = layer_data.compute()
                tifffile.imwrite(str(path), layer_data)
                saved_count += 1
                break  # Only save the first layer for TIFF
            else:
                # For other formats, try to save using layer's save method if available
                try:
                    if hasattr(layer, 'save'):
                        layer.save(str(path))
                        saved_count += 1
                    else:
                        return f"Layer '{layer.name}' does not support saving"
                except Exception as e:
                    return f"Error saving layer '{layer.name}': {e}"
    
    if saved_count == 0:
        return "No layers were saved"
    
    return f"Saved {saved_count} layer(s) to {path}"

def export_screenshot(
    file_path: str | Path,
    canvas_only: bool = True,
    viewer: Viewer = None,
):
    """Export screenshot to specific file path."""
    path = Path(file_path)
    screenshot_array = viewer.screenshot(canvas_only=canvas_only)
    
    from PIL import Image
    img = Image.fromarray(screenshot_array)
    img.save(path)
    
    return f"Screenshot saved to {path.absolute()}"

def get_layer_data(
    layer_name: str | int,
    viewer: Viewer = None,
):
    """Extract layer data as numpy array."""
    layer = _get_layer(viewer, layer_name)
    
    if hasattr(layer, 'data'):
        data = layer.data
        if hasattr(data, 'compute'):  # Handle dask arrays
            data = data.compute()
        return {
            'data': to_serializable(data),
            'shape': data.shape,
            'dtype': str(data.dtype),
            'layer_type': layer.__class__.__name__
        }
    else:
        return f"Layer '{layer.name}' does not have data attribute."

# ----------------------------------------------------------------------
# Advanced Visualization Controls
# ----------------------------------------------------------------------

def set_scale_bar(
    visible: bool = True,
    unit: str = 'um',
    viewer: Viewer = None,
):
    """Add or remove scale bar."""
    if hasattr(viewer, 'scale_bar'):
        viewer.scale_bar.visible = visible
        if visible:
            viewer.scale_bar.unit = unit
        return f"Scale bar {'visible' if visible else 'hidden'} with unit '{unit}'"
    return "Scale bar not available in this napari version"

def set_axis_labels(
    labels: list,
    viewer: Viewer = None,
):
    """Set axis labels."""
    if len(labels) != len(viewer.dims.axis_labels):
        return f"Expected {len(viewer.dims.axis_labels)} labels, got {len(labels)}"
    
    viewer.dims.axis_labels = labels
    return f"Axis labels set to {labels}"

def set_view_mode(
    mode: str,
    viewer: Viewer = None,
):
    """Set view mode (2D, 3D, etc.)."""
    if mode.lower() == '2d':
        viewer.dims.ndisplay = 2
    elif mode.lower() == '3d':
        viewer.dims.ndisplay = 3
    else:
        return f"Unknown view mode: {mode}"
    
    return f"View mode set to {mode.upper()}"

def set_layer_visibility(
    layer_name: str | int,
    visible: bool,
    viewer: Viewer = None,
):
    """Set layer visibility."""
    layer = _get_layer(viewer, layer_name)
    layer.visible = visible
    return f"Layer '{layer.name}' {'visible' if visible else 'hidden'}"

# ----------------------------------------------------------------------
# Measurement & Analysis Functions
# ----------------------------------------------------------------------

def measure_distance(
    point1: list | tuple,
    point2: list | tuple,
    viewer: Viewer = None,
):
    """Measure distance between two points."""
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    distance = np.linalg.norm(p2 - p1)
    return {
        'distance': float(distance),
        'point1': list(p1),
        'point2': list(p2)
    }

def get_layer_statistics(
    layer_name: str | int,
    viewer: Viewer = None,
):
    """Get statistics for a layer."""
    layer = _get_layer(viewer, layer_name)
    
    if hasattr(layer, 'data'):
        data = layer.data
        if hasattr(data, 'compute'):  # Handle dask arrays
            data = data.compute()
        
        return {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'shape': data.shape,
            'dtype': str(data.dtype)
        }
    else:
        return f"Layer '{layer.name}' does not have data attribute."

def crop_layer(
    layer_name: str | int,
    bounds: list | tuple,
    viewer: Viewer = None,
):
    """Crop layer to specific bounds."""
    layer = _get_layer(viewer, layer_name)
    
    if hasattr(layer, 'data'):
        # bounds should be [start_t, end_t, start_z, end_z, start_y, end_y, start_x, end_x]
        if len(bounds) != 8:
            return "Bounds must be a list of 8 values: [start_t, end_t, start_z, end_z, start_y, end_y, start_x, end_x]"
        
        # Create slice object
        slices = tuple(slice(bounds[i], bounds[i+1]) for i in range(0, 8, 2))
        
        # Crop the data
        cropped_data = layer.data[slices]
        
        # Add as new layer
        new_name = f"{layer.name}_cropped"
        viewer.add_image(cropped_data, name=new_name)
        
        return f"Cropped layer '{layer.name}' to shape {cropped_data.shape}, added as '{new_name}'"
    else:
        return f"Layer '{layer.name}' does not have data attribute."

# ----------------------------------------------------------------------
# Time Series & Multi-dimensional Data
# ----------------------------------------------------------------------

def set_channel(
    channel_index: int,
    viewer: Viewer = None,
):
    """Set current channel."""
    if channel_index >= viewer.dims.nsteps[1]:  # Assuming channel is second dimension
        return f"Channel index {channel_index} out of bounds (max: {viewer.dims.nsteps[1] - 1})"
    
    current_step = list(viewer.dims.current_step)
    current_step[1] = channel_index
    viewer.dims.current_step = tuple(current_step)
    
    return f"Channel set to {channel_index}"

def set_z_slice(
    z_index: int,
    viewer: Viewer = None,
):
    """Set current z-slice."""
    if z_index >= viewer.dims.nsteps[2]:  # Assuming z is third dimension
        return f"Z-slice index {z_index} out of bounds (max: {viewer.dims.nsteps[2] - 1})"
    
    current_step = list(viewer.dims.current_step)
    current_step[2] = z_index
    viewer.dims.current_step = tuple(current_step)
    
    return f"Z-slice set to {z_index}"

def play_animation(
    start_frame: int,
    end_frame: int,
    fps: int = 10,
    viewer: Viewer = None,
):
    """Play animation through time series."""
    if start_frame >= viewer.dims.nsteps[0] or end_frame >= viewer.dims.nsteps[0]:
        return f"Frame indices out of bounds (max: {viewer.dims.nsteps[0] - 1})"
    
    # Set the current frame to start_frame
    current_step = list(viewer.dims.current_step)
    current_step[0] = start_frame
    viewer.dims.current_step = tuple(current_step)
    
    # For animation, we need to set the range for the time dimension (first dimension)
    # Create a new range tuple for the time dimension
    from napari.utils.misc import RangeTuple
    
    # Get current ranges
    current_ranges = list(viewer.dims.range)
    
    # Update the time dimension range (first dimension)
    current_ranges[0] = RangeTuple(start=float(start_frame), stop=float(end_frame), step=1.0)
    
    # Set the new range
    viewer.dims.range = tuple(current_ranges)
    
    # Start animation briefly and then stop it to demonstrate functionality
    # without causing the function to hang
    viewer.dims.animation = True
    
    # Stop animation immediately to avoid hanging
    viewer.dims.animation = False
    
    return f"Animation range set from frame {start_frame} to {end_frame} at {fps} FPS (animation started and stopped)"