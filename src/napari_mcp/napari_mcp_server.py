# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# napari_mcp_server.py
"""
MCP server that connects to napari via TCP socket.

This script runs as an agent process launched by Claude Desktop or other MCP clients.
It forwards requests to a live napari GUI session over a socket connection.

Usage:
    python napari_mcp_server.py --help
    python napari_mcp_server.py  # starts server on stdin/stdout
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import argparse, json, logging, os

from mcp.server.fastmcp import FastMCP, Image

from napari_manager import NapariManager

_LOGGER = logging.getLogger("napari_mcp_socket")


###########################################################################
# CLI parsing / logging
###########################################################################

def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the napari MCP server.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - host: Napari socket host (default: "127.0.0.1")
            - port: Napari socket port (default: 64908)
            - timeout: TCP timeout in seconds (default: 5.0)
            - loglevel: Console log level (default: "INFO")
    """
    parser = argparse.ArgumentParser(description="Napari MCP server (socket backend)")
    parser.add_argument("--host", default="127.0.0.1", help="Napari‑socket host [default: %(default)s]")
    parser.add_argument("--port", type=int, default=64908, help="Napari‑socket port [default: %(default)s]")
    parser.add_argument("--timeout", type=float, default=5.0, help="TCP timeout seconds [default: %(default)s]")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log‑level",
    )
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    """Set up logging configuration for the napari MCP server.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR)
        
    Creates log files in ~/napari_logs/napari_mcp_socket.log and outputs to console.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)

    log_dir = Path.home() / "napari_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "napari_mcp_socket.log"

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


###########################################################################
# FastMCP definition
###########################################################################

def build_mcp(manager: NapariManager) -> FastMCP:
    """Build the FastMCP server with all napari tools.
    
    Args:
        manager: NapariManager instance for communicating with napari
        
    Returns:
        FastMCP: Configured MCP server with all napari tools
    """
    prompt = (
        "You control a remote napari GUI through a TCP socket. "
        "Use the screenshot tool to see the current viewport.\n\n"
        "Available tools:\n"
        "• open_file(path) - load image files (TIFF, PNG, ND2, NPZ, etc.)\n"
        "• remove_layer(name_or_index) - remove a layer\n"
        "• toggle_view() - switch between 2D and 3D view\n"
        "• iso_contour(layer_name=None, threshold=None) - enable iso-surface rendering\n"
        "• screenshot() - capture current view as JPG\n"
        "• list_layers() - get info about loaded layers\n"
        "• set_colormap(layer_name, colormap) - change layer colormap\n"
        "• set_opacity(layer_name, opacity) - adjust layer transparency\n"
        "• set_blending(layer_name, blending) - set layer blend mode\n"
        "• set_contrast_limits(layer_name, min, max) - adjust contrast\n"
        "• auto_contrast(layer_name) - auto-adjust contrast\n"
        "• set_gamma(layer_name, gamma) - adjust gamma correction\n"
        "• set_interpolation(layer_name, mode) - set interpolation\n"
        "• set_timestep(timestep) - set current time point\n"
        "• get_dims_info() - get dimension info\n"
        "• set_camera(center, zoom, angle) - adjust camera\n"
        "• get_camera() - get camera settings\n"
        "• reset_camera() - reset to default view\n"
        "• add_points(coords, properties, name) - add point annotations\n"
        "• add_shapes(data, shape_type, name) - add shape annotations\n"
        "• add_labels(image, name) - add segmentation masks\n"
        "• add_surface(vertices, faces, name) - add 3D meshes\n"
        "• add_vectors(vectors, name) - add vector fields\n"
        "• save_layers(file_path, layer_names) - save layers to file\n"
        "• export_screenshot(file_path, canvas_only) - save screenshot\n"
        "• get_layer_data(layer_name) - extract layer data\n"
        "• set_scale_bar(visible, unit) - show/hide scale bar\n"
        "• set_axis_labels(labels) - set axis labels\n"
        "• set_view_mode(mode) - change view mode\n"
        "• set_layer_visibility(layer_name, visible) - show/hide layer\n"
        "• measure_distance(point1, point2) - measure between points\n"
        "• get_layer_statistics(layer_name) - get layer stats\n"
        "• crop_layer(layer_name, bounds) - crop layer data\n"
        "• set_channel(index) - set current channel\n"
        "• set_z_slice(index) - set current z-slice\n"
        "• play_animation(start, end, fps) - play time series\n"
    )

    mcp = FastMCP("Napari‑Socket", system_prompt=prompt)

    @mcp.tool()
    def open_file(file_path: str) -> str:  
        """Load an image file into napari.
        
        Args:
            file_path: Path to the image file to load. Supports TIFF, PNG, ND2, NPZ, 
                      and other formats supported by napari's built-in readers.
                      
        Returns:
            str: Success message with layer name or error message prefixed with ❌
            
        Note:
            The file path should be absolute or relative to the current working directory.
            napari will automatically detect the file format and use the appropriate reader.
        """
        success, message = manager.open_file(file_path)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def remove_layer(name_or_index: str | int) -> str:  
        """Remove a layer by name or index.
        
        Args:
            name_or_index: Layer name (str) or positional index (int) of the layer to remove
            
        Returns:
            str: Success message or error message prefixed with ❌
            
        Note:
            Use list_layers() to see available layers and their indices.
            Layer indices are 0-based.
        """
        success, message = manager.remove_layer(name_or_index)
        return message if success else f"❌ {message}"


    @mcp.tool(name="toggle_view")
    def toggle_view() -> str:  
        """Switch between 2D and 3D view.
        
        Returns:
            str: Success message indicating the new view mode or error message prefixed with ❌
            
        Note:
            Toggles between 2D (ndisplay=2) and 3D (ndisplay=3) rendering modes.
            Some features like iso-surface rendering require 3D mode.
        """
        success, message = manager.toggle_ndisplay()
        return message if success else f"❌ {message}"

    @mcp.tool(name="iso_contour")
    def iso_contour(
        layer_name: str | int | None = None,
        threshold: float | None = None,
    ) -> str:  
        """Enable iso-surface rendering for layers.
        
        Args:
            layer_name: Layer name (str) or index (int) to apply iso-surface to. 
                       If None, applies to all compatible layers.
            threshold: Iso-surface threshold value. If None, uses layer's current threshold.
                      
        Returns:
            str: Success message with number of layers modified or error message prefixed with ❌
            
        Note:
            Only works with Image layers that support iso-surface rendering.
            Automatically switches to 3D view if needed.
            Use this for 3D volume visualization with surface rendering.
        """
        success, message = manager.iso_contour(layer_name, threshold)
        return message if success else f"❌ {message}"

    @mcp.tool(name="screenshot")
    def screenshot() -> str:  
        """Take a screenshot of the current view.
        
        Returns:
            Image: Screenshot image object or error message prefixed with ❌
            
        Note:
            Saves a JPG file to a temporary location and returns the image.
            Captures only the canvas area by default.
            The temporary file is automatically cleaned up by the system.
        """
        success, message = manager.screenshot()
        if success:
            return Image(path=message)  # message is the absolute path to the screenshot
        return f"\u274c {message}"
    
    @mcp.tool(name="list_layers")
    def list_layers() -> str:            
        """Get info about all loaded layers.
        
        Returns:
            str: JSON-formatted list of layer information including index, name, type, and visibility
                 or error message prefixed with ❌
                 
        Note:
            Each layer entry contains:
            - index: 0-based layer index
            - name: Layer name
            - type: Layer class name (Image, Labels, Points, etc.)
            - visible: Boolean indicating if layer is visible
        """
        success, message = manager.list_layers()
        return json.dumps(message, indent=2) if success else f"❌ {message}"

    @mcp.tool(name="set_colormap")
    def set_colormap(layer_name: str, colormap: str) -> str:
        """Change the colormap for a layer.
        
        Args:
            layer_name: Layer name (str) or index (int) to modify
            colormap: Colormap name (e.g., 'gray', 'viridis', 'plasma', 'hot', 'cool', etc.)
                     
        Returns:
            str: Success message or error message prefixed with ❌
            
        Note:
            Only works with layers that have a colormap attribute (typically Image layers).
            Common colormaps include: gray, viridis, plasma, hot, cool, rainbow, etc.
        """
        success, message = manager.set_colormap(layer_name, colormap)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def set_opacity(layer_name: str, opacity: float) -> str:
        """Adjust layer transparency.
        
        Args:
            layer_name: Layer name (str) or index (int) to modify
            opacity: Opacity value between 0.0 (transparent) and 1.0 (opaque)
                    
        Returns:
            str: Success message or error message prefixed with ❌
        """
        success, message = manager.set_opacity(layer_name, opacity)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def set_blending(layer_name: str, blending: str) -> str:
        """Set how the layer blends with layers below it.
        
        Args:
            layer_name: Layer name (str) or index (int) to modify
            blending: Blending mode ('opaque', 'translucent', 'additive', 'minimum')
                     
        Returns:
            str: Success message or error message prefixed with ❌
            
        Note:
            Common blending modes:
            - opaque: Normal blending (default)
            - translucent: Alpha blending
            - additive: Add pixel values
            - minimum: Take minimum of pixel values
        """
        success, message = manager.set_blending(layer_name, blending)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def set_contrast_limits(layer_name: str, contrast_min: float, contrast_max: float) -> str:
        """Set the min/max values for contrast scaling.
        
        Args:
            layer_name: Layer name (str) or index (int) to modify
            contrast_min: Minimum value for contrast scaling
            contrast_max: Maximum value for contrast scaling
                         
        Returns:
            str: Success message or error message prefixed with ❌
            
        Note:
            Only works with layers that have contrast_limits (typically Image layers).
            Values outside this range will be clipped to min/max.
            Use auto_contrast() to automatically set these values.
        """
        success, message = manager.set_contrast_limits(layer_name, contrast_min, contrast_max)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def auto_contrast(layer_name: str | None = None) -> str:
        """Automatically adjust contrast to fit the data range.
        
        Args:
            layer_name: Layer name (str) or index (int) to modify. If None, uses active layer.
                       
        Returns:
            str: Success message with new contrast limits or error message prefixed with ❌
        """
        success, message = manager.auto_contrast(layer_name)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def set_gamma(layer_name: str, gamma: float) -> str:
        """Adjust gamma correction for the layer.
        
        Args:
            layer_name: Layer name (str) or index (int) to modify
            gamma: Gamma correction value (typically 0.1 to 10.0)
                  
        Returns:
            str: Success message or error message prefixed with ❌
            
        Note:
            Gamma < 1.0 brightens dark regions, gamma > 1.0 darkens bright regions.
            Only works with layers that have a gamma attribute.
        """
        success, message = manager.set_gamma(layer_name, gamma)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def set_interpolation(layer_name: str, interpolation: str) -> str:
        """Set the interpolation method for zooming.
        
        Args:
            layer_name: Layer name (str) or index (int) to modify
            interpolation: Interpolation mode ('nearest', 'linear', 'cubic')
                          
        Returns:
            str: Success message or error message prefixed with ❌
            
        Note:
            Common interpolation modes:
            - nearest: No interpolation (pixelated)
            - linear: Linear interpolation (smooth)
            - cubic: Cubic interpolation (very smooth)
        """
        success, message = manager.set_interpolation(layer_name, interpolation)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def set_timestep(timestep: int) -> str:
        """Jump to a specific time point.
        
        Args:
            timestep: Time point index (0-based)
                     
        Returns:
            str: Success message or error message prefixed with ❌
            
        Note:
            Only works if the data has a time dimension.
            Use get_dims_info() to check available dimensions.
        """
        success, message = manager.set_timestep(timestep)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def get_dims_info() -> str:
        """Get info about the viewer's dimensions.
        
        Returns:
            str: JSON-formatted dimension information including:
                 - ndim: Number of dimensions
                 - nsteps: Number of steps in each dimension
                 - current_step: Current position in each dimension
                 - axis_labels: Labels for each dimension
                 or error message prefixed with ❌
        """
        success, message = manager.get_dims_info()
        return json.dumps(message, indent=2) if success else f"❌ {message}"

    @mcp.tool()
    def set_camera(center=None, zoom=None, angle=None) -> str:
        """Adjust camera position, zoom, and rotation.
        
        Args:
            center: Camera center point as list/tuple of coordinates [x, y, z]
            zoom: Zoom factor (float)
            angle: Camera angles as list/tuple [x_angle, y_angle, z_angle] (3D only)
                  
        Returns:
            str: JSON-formatted camera settings or error message prefixed with ❌
            
        Note:
            In 2D mode, only center and zoom are used.
            In 3D mode, all parameters are used.
            Use get_camera() to see current camera settings.
        """
        success, message = manager.set_camera(center, zoom, angle)
        return json.dumps(message) if success else f"❌ {message}"

    @mcp.tool()
    def get_camera() -> str:
        """Get current camera settings.
        
        Returns:
            str: JSON-formatted camera settings including center, zoom, and angles
                 or error message prefixed with ❌
        """
        success, message = manager.get_camera()
        return json.dumps(message) if success else f"❌ {message}"

    @mcp.tool()
    def reset_camera() -> str:
        """Reset camera to default view.
        
        Returns:
            str: JSON-formatted camera settings after reset or error message prefixed with ❌
        """
        success, message = manager.reset_camera()
        return json.dumps(message) if success else f"❌ {message}"

    # ------------------------------------------------------------------
    # Layer Creation & Annotation Functions
    # ------------------------------------------------------------------
    @mcp.tool()
    def add_points(coordinates: list, properties: dict | None = None, name: str | None = None) -> str:
        """Add point markers to the viewer.
        
        Args:
            coordinates: List of point coordinates, each as [x, y] or [x, y, z]
            properties: Optional dict of point properties (e.g., {'label': ['A', 'B', 'C']})
            name: Optional layer name
                     
        Returns:
            str: Success message with layer name and point count or error message prefixed with ❌
            
        Note:
            Coordinates should be a list of lists, where each inner list represents one point.
            For 2D: [[x1, y1], [x2, y2], ...]
            For 3D: [[x1, y1, z1], [x2, y2, z2], ...]
        """
        success, message = manager.add_points(coordinates, properties, name)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def add_shapes(shape_data: list, shape_type: str = 'rectangle', name: str | None = None) -> str:
        """Add shape overlays (rectangles, circles, etc.).
        
        Args:
            shape_data: List of shape coordinates. For rectangles: [[[x1, y1], [x2, y2], ...]]
            shape_type: Shape type ('rectangle', 'ellipse', 'line', 'polygon', 'path')
            name: Optional layer name
                       
        Returns:
            str: Success message with layer name and shape count or error message prefixed with ❌
            
        Note:
            Shape data format depends on shape type:
            - rectangle: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]] (4 corners)
            - ellipse: [[[center_x, center_y], [radius_x, radius_y]]]
            - line: [[[x1, y1], [x2, y2]]]
            - polygon: [[[x1, y1], [x2, y2], [x3, y3], ...]] (3+ points)
        """
        success, message = manager.add_shapes(shape_data, shape_type, name)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def add_labels(label_image: list, name: str | None = None) -> str:
        """Add segmentation masks or labeled regions.
        
        Args:
            label_image: 2D or 3D array of integer labels (0 = background, 1+ = regions)
            name: Optional layer name
                     
        Returns:
            str: Success message with layer name and shape or error message prefixed with ❌
            
        Note:
            Label image should be a 2D or 3D array where each unique integer represents
            a different region. 0 is typically used for background.
            Each region will be displayed with a different color.
        """
        success, message = manager.add_labels(label_image, name)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def add_surface(vertices: list, faces: list, name: str | None = None) -> str:
        """Add 3D mesh surface to the viewer.
        
        Args:
            vertices: List of vertex coordinates [[x1, y1, z1], [x2, y2, z2], ...]
            faces: List of face indices [[v1, v2, v3], [v4, v5, v6], ...] (triangles)
            name: Optional layer name
                     
        Returns:
            str: Success message with layer name and vertex/face counts or error message prefixed with ❌
            
        Note:
            Vertices define the 3D points of the mesh.
            Faces define triangles by referencing vertex indices (0-based).
            Each face should have exactly 3 vertex indices.
        """
        success, message = manager.add_surface(vertices, faces, name)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def add_vectors(vectors: list, name: str | None = None) -> str:
        """Add vector field arrows to the viewer.
        
        Args:
            vectors: 2D or 3D array of vectors [[[x1, y1], [dx1, dy1]], [[x2, y2], [dx2, dy2]], ...]
            name: Optional layer name
                     
        Returns:
            str: Success message with layer name and shape or error message prefixed with ❌
            
        Note:
            Vectors format: [[position, direction], ...]
            For 2D: [[[x, y], [dx, dy]], ...]
            For 3D: [[[x, y, z], [dx, dy, dz]], ...]
            Each vector shows direction and magnitude at a specific position.
        """
        success, message = manager.add_vectors(vectors, name)
        return message if success else f"❌ {message}"

    # ------------------------------------------------------------------
    # Data Export & Save Functions
    # ------------------------------------------------------------------
    @mcp.tool()
    def save_layers(file_path: str, layer_names: list | None = None) -> str:
        """Save one or more layers to disk.
        
        Args:
            file_path: Path where to save the file (supports .tif, .tiff extensions)
            layer_names: Optional list of layer names to save. If None, saves all layers.
                        
        Returns:
            str: Success message with save count or error message prefixed with ❌
            
        Note:
            Currently supports saving as TIFF format.
            For multiple layers, only the first layer is saved to the specified file.
            Layer data is saved as-is without any transformations.
        """
        success, message = manager.save_layers(file_path, layer_names)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def export_screenshot(file_path: str, canvas_only: bool = True) -> str:
        """Save a screenshot to a specific location.
        
        Args:
            file_path: Path where to save the screenshot
            canvas_only: If True, capture only the canvas area. If False, capture full UI.
                        
        Returns:
            str: Success message with file path or error message prefixed with ❌
        """
        success, message = manager.export_screenshot(file_path, canvas_only)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def get_layer_data(layer_name: str | int) -> str:
        """Extract the raw data from a layer.
        
        Args:
            layer_name: Layer name (str) or index (int) to extract data from
            
        Returns:
            str: JSON-formatted layer data including shape, dtype, and data array
                 or error message prefixed with ❌
                 
        Note:
            Returns the actual data array, shape, and data type.
            Large arrays may be truncated in the JSON output.
            Use this to inspect layer data for analysis.
        """
        success, message = manager.get_layer_data(layer_name)
        return json.dumps(message, indent=2) if success else f"❌ {message}"

    # ------------------------------------------------------------------
    # Advanced Visualization Controls
    # ------------------------------------------------------------------
    @mcp.tool()
    def set_scale_bar(visible: bool = True, unit: str = 'um') -> str:
        """Show or hide the scale bar.
        
        Args:
            visible: Whether to show the scale bar
            unit: Unit of measurement (e.g., 'um', 'nm', 'mm', 'm')
                 
        Returns:
            str: Success message or error message prefixed with ❌
        """
        success, message = manager.set_scale_bar(visible, unit)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def set_axis_labels(labels: list) -> str:
        """Set custom labels for the axes.
        
        Args:
            labels: List of axis labels (e.g., ['t', 'z', 'y', 'x'] for 4D data)
                   
        Returns:
            str: Success message or error message prefixed with ❌
            
        Note:
            Number of labels must match the number of dimensions in the data.
            Use get_dims_info() to see the current number of dimensions.
        """
        success, message = manager.set_axis_labels(labels)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def set_view_mode(mode: str) -> str:
        """Switch between different view modes.
        
        Args:
            mode: View mode ('2d' or '3d')
                 
        Returns:
            str: Success message or error message prefixed with ❌
        """
        success, message = manager.set_view_mode(mode)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def set_layer_visibility(layer_name: str | int, visible: bool) -> str:
        """Show or hide a specific layer.
        
        Args:
            layer_name: Layer name (str) or index (int) to modify
            visible: Whether the layer should be visible
                   
        Returns:
            str: Success message or error message prefixed with ❌
        """
        success, message = manager.set_layer_visibility(layer_name, visible)
        return message if success else f"❌ {message}"

    # ------------------------------------------------------------------
    # Measurement & Analysis Functions
    # ------------------------------------------------------------------
    @mcp.tool()
    def measure_distance(point1: list, point2: list) -> str:
        """Calculate distance between two points in the data.
        
        Args:
            point1: First point coordinates [x, y] or [x, y, z]
            point2: Second point coordinates [x, y] or [x, y, z]
                   
        Returns:
            str: JSON-formatted measurement result including distance and point coordinates
                 or error message prefixed with ❌
                 
        Note:
            Points should have the same dimensionality (both 2D or both 3D).
            Distance is calculated using Euclidean distance.
        """
        success, message = manager.measure_distance(point1, point2)
        return json.dumps(message, indent=2) if success else f"❌ {message}"

    @mcp.tool()
    def get_layer_statistics(layer_name: str | int) -> str:
        """Get basic stats (min, max, mean, std) for a layer.
        
        Args:
            layer_name: Layer name (str) or index (int) to analyze
            
        Returns:
            str: JSON-formatted statistics including min, max, mean, std, shape, and dtype
                 or error message prefixed with ❌
        """
        success, message = manager.get_layer_statistics(layer_name)
        return json.dumps(message, indent=2) if success else f"❌ {message}"

    @mcp.tool()
    def crop_layer(layer_name: str | int, bounds: list) -> str:
        """Crop a layer to a specific region.
        
        Args:
            layer_name: Layer name (str) or index (int) to crop
            bounds: Crop bounds as [start_t, end_t, start_z, end_z, start_y, end_y, start_x, end_x]
                   
        Returns:
            str: Success message with new layer name or error message prefixed with ❌
                
        Note:
            Bounds must be a list of 8 values defining the crop region for each dimension.
            Creates a new layer with the cropped data.
            Use get_dims_info() to understand the dimension order.
        """
        success, message = manager.crop_layer(layer_name, bounds)
        return message if success else f"❌ {message}"

    # ------------------------------------------------------------------
    # Time Series & Multi-dimensional Data
    # ------------------------------------------------------------------
    @mcp.tool()
    def set_channel(channel_index: int) -> str:
        """Switch to a specific channel in multi-channel data.
        
        Args:
            channel_index: Channel index (0-based)
                          
        Returns:
            str: Success message or error message prefixed with ❌
                
        Note:
            Only works if the data has a channel dimension.
            Use get_dims_info() to check available dimensions.
            Channel is typically the second dimension (index 1).
        """
        success, message = manager.set_channel(channel_index)
        return message if success else f"❌ {message}"

    @mcp.tool()
    def set_z_slice(z_index: int) -> str:
        """Jump to a specific z-slice in 3D data.
        
        Args:
            z_index: Z-slice index (0-based)
                    
        Returns:
            str: Success message or error message prefixed with ❌
                
        Note:
            Only works if the data has a Z dimension.
            Use get_dims_info() to check available dimensions.
            Z is typically the third dimension (index 2).
        """
        success, message = manager.set_z_slice(z_index)
        return message if success else f"❌ {message}"

    #TODO currently not working
    @mcp.tool()
    def play_animation(start_frame: int, end_frame: int, fps: int = 10) -> str:
        """Animate through a time series at specified FPS.
        
        Args:
            start_frame: Starting frame index (0-based)
            end_frame: Ending frame index (0-based)
            fps: Frames per second for animation (default: 10)
                       
        Returns:
            str: Success message or error message prefixed with ❌
                
        Note:
            Only works if the data has a time dimension.
            Use get_dims_info() to check available dimensions.
            Time is typically the first dimension (index 0).
            Currently limited functionality - sets animation range but doesn't play continuously.
        """
        success, message = manager.play_animation(start_frame, end_frame, fps)
        return message if success else f"❌ {message}"

    return mcp

    

###########################################################################
# entry‑point
###########################################################################

def main() -> None:  # pragma: no cover
    """Main entry point for the napari MCP server.
    
    Parses command line arguments, sets up logging, creates the NapariManager,
    builds the MCP server, and starts listening for requests.
    """
    args = _parse_args()
    _setup_logging(args.loglevel)

    mgr = NapariManager(host=args.host, port=args.port, timeout=args.timeout)
    mcp = build_mcp(mgr)

    _LOGGER.info("Napari MCP (socket backend) listening…")
    mcp.run()


if __name__ == "__main__":
    main()