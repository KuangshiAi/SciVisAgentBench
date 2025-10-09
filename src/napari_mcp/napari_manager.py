# napari_manager.py
"""
Socket‑based Napari Manager
---------------------------
Encapsulates communication with the *napari‑socket* plugin that runs inside
a live napari GUI session.  All interaction happens over a plain TCP socket
(the plugin listens on 127.0.0.1:64908 by default).

Currently we expose a single helper – ``open_file`` – as proof‑of‑concept.
More commands from the plugin's manifest (``napari.yaml``) can be added by
calling ``NapariManager.send_command`` with the appropriate command id.
"""
from __future__ import annotations

import json
import logging
import pathlib
import socket
from typing import Any, Sequence, Tuple
import numpy as np

_LOGGER = logging.getLogger(__name__)



def _convert_numpy_for_json(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_for_json(v) for v in obj]
    else:
        return obj


class NapariManager:  # pylint: disable=too-few-public-methods
    """Small helper that talks to the TCP server spawned by *napari‑socket*."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 64908,
        timeout: float = 20.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    # ---------------------------------------------------------------------
    # low‑level I/O helpers
    # ---------------------------------------------------------------------
    def _send(self, payload: dict[str, Any] | list[Any]) -> str:
        """Send *one* JSON payload and return the raw string reply.

        The *napari‑socket* plugin expects **exactly** one JSON line per
        connection and responds with a single line that starts with either
        ``"OK"`` or ``"ERR ..."``.
        """
        # Convert numpy arrays to lists for JSON serialization
        payload = _convert_numpy_for_json(payload)
        data = json.dumps(payload).encode() + b"\n"
        _LOGGER.debug("→ %s", data)

        with socket.create_connection((self.host, self.port), self.timeout) as sck:
            sck.sendall(data)
            reply = sck.recv(8192).decode().strip()

        _LOGGER.debug("← %s", reply)
        return reply

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def send_command(self, cmd_id: str, args: Sequence[Any] | None = None) -> Tuple[bool, Any]:
        """Invoke *cmd_id* inside napari and return *(success, message)*."""
        payload: list[Any] = [cmd_id, list(args or [])]
        reply = self._send(payload)
        if reply == "OK":                 # no payload
            return True, None
        if reply.startswith("OK "):       # payload present
            payload = reply[3:].strip()
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                pass                       # plain-text payload
            return True, payload
        return False, reply

    # ------------------------------------------------------------------
    # high‑level helpers
    # ------------------------------------------------------------------
    def open_file(self, file_path: str | pathlib.Path) -> Tuple[bool, str]:
        """Open *file_path* in napari using the plugin command.

        The command id is *napari‑socket.open_file* as declared in the plugin's
        manifest.
        """
        path = pathlib.Path(file_path).expanduser().resolve()
        if not path.exists():
            return False, f"File not found: {path}"

        return self.send_command("napari-socket.open_file", [str(path)])



    def remove_layer(self, name_or_index: str | int) -> Tuple[bool, str]:
        """Remove a layer by its name or positional index using the plugin command.

        The command id is *napari‑socket.remove_layer* as declared in the plugin's
        manifest.
        """
        return self.send_command("napari-socket.remove_layer", [name_or_index])


    # ------------------------------------------------------------------
    # view helpers
    # ------------------------------------------------------------------
    def toggle_ndisplay(self) -> Tuple[bool, str]:
        """Switch the remote viewer between 2-D and 3-D modes."""
        return self.send_command("napari-socket.toggle_ndisplay")
    
    # ------------------------------------------------------------------
    # iso-surface helper
    # ------------------------------------------------------------------
    def iso_contour(
        self,
        layer_name: str | int | None = None,
        threshold: float | None = None,
    ) -> Tuple[bool, str]:
        """Apply iso-surface (contour) rendering to one or more layers."""
        args: list[Any] = []
        if layer_name is not None:
            args.append(layer_name)
        if threshold is not None:
            args.append(float(threshold))
        return self.send_command("napari-socket.iso_contour", args)
    
    def iso_contour_all_layers(self, threshold: float) -> Tuple[bool, str]:
        """Apply iso-surface (contour) rendering to all layers with the given threshold."""
        return self.send_command("napari-socket.iso_contour", [None, threshold])
    

    # ------------------------------------------------------------------
    # screenshot helper
    # ------------------------------------------------------------------
    def screenshot(self, filename: str | None = None) -> tuple[bool, str]:
        """Ask the remote viewer to save a JPG screenshot and return the absolute path as a string."""
        args = []
        if filename is not None:
            args.append(filename)
        return self.send_command("napari-socket.screenshot", args)

    # ------------------------------------------------------------------
    # layer introspection helper
    # ------------------------------------------------------------------
    def list_layers(self) -> Tuple[bool, Any]:
        """Get info about all loaded layers."""
        return self.send_command("napari-socket.list_layers")

    def set_colormap(self, layer_name: str | int, colormap: str) -> Tuple[bool, Any]:
        """Change the colormap for a layer."""
        return self.send_command("napari-socket.set_colormap", [layer_name, colormap])

    def set_opacity(self, layer_name: str | int, opacity: float) -> Tuple[bool, Any]:
        """Adjust layer transparency (0=transparent, 1=opaque)."""
        args = [layer_name, opacity]
        return self.send_command("napari-socket.set_opacity", args)

    def set_blending(self, layer_name: str | int, blending: str) -> Tuple[bool, Any]:
        """Set how the layer blends with layers below it."""
        args = [layer_name, blending]
        return self.send_command("napari-socket.set_blending", args)

    def set_contrast_limits(self, layer_name: str | int, contrast_min: float, contrast_max: float) -> Tuple[bool, Any]:
        """Set the min/max values for contrast scaling."""
        args = [layer_name, contrast_min, contrast_max]
        return self.send_command("napari-socket.set_contrast_limits", args)

    def auto_contrast(self, layer_name: str | int) -> Tuple[bool, Any]:
        """Automatically adjust contrast to fit the data range."""
        args = [layer_name]
        return self.send_command("napari-socket.auto_contrast", args)

    def set_gamma(self, layer_name: str | int, gamma: float) -> Tuple[bool, Any]:
        """Adjust gamma correction for the layer."""
        args = [layer_name, gamma]
        return self.send_command("napari-socket.set_gamma", args)

    def set_interpolation(self, layer_name: str | int, interpolation: str) -> Tuple[bool, Any]:
        """Set the interpolation method for zooming."""
        args = [layer_name, interpolation]
        return self.send_command("napari-socket.set_interpolation", args)

    def set_timestep(self, timestep: int) -> Tuple[bool, Any]:
        """Jump to a specific time point."""
        return self.send_command("napari-socket.set_timestep", [timestep])

    def get_dims_info(self) -> Tuple[bool, Any]:
        """Get info about the viewer's dimensions."""
        return self.send_command("napari-socket.get_dims_info")

    def set_camera(self, center=None, zoom=None, angle=None) -> Tuple[bool, Any]:
        """Adjust camera position, zoom, and rotation."""
        args = []
        if center is not None:
            args.append(center)
        else:
            args.append(None)
        if zoom is not None:
            args.append(zoom)
        else:
            args.append(None)
        if angle is not None:
            args.append(angle)
        else:
            args.append(None)
        return self.send_command("napari-socket.set_camera", args)

    def get_camera(self) -> Tuple[bool, Any]:
        """Get current camera settings."""
        return self.send_command("napari-socket.get_camera")

    def reset_camera(self) -> Tuple[bool, Any]:
        """Reset camera to default view."""
        return self.send_command("napari-socket.reset_camera")

    # ------------------------------------------------------------------
    # Layer Creation & Annotation Functions
    # ------------------------------------------------------------------
    def add_points(self, coordinates: list | np.ndarray, properties: dict | None = None, name: str | None = None) -> Tuple[bool, Any]:
        """Add point markers to the viewer."""
        args = [coordinates]
        if properties is not None:
            args.append(properties)
        if name is not None:
            args.append(name)
        return self.send_command("napari-socket.add_points", args)

    def add_shapes(self, shape_data: list | np.ndarray, shape_type: str = 'rectangle', name: str | None = None) -> Tuple[bool, Any]:
        """Add shape overlays (rectangles, circles, etc.)."""
        args = [shape_data, shape_type]
        if name is not None:
            args.append(name)
        return self.send_command("napari-socket.add_shapes", args)

    def add_labels(self, label_image: np.ndarray, name: str | None = None) -> Tuple[bool, Any]:
        """Add segmentation masks or labeled regions."""
        args = [label_image]
        if name is not None:
            args.append(name)
        return self.send_command("napari-socket.add_labels", args)

    def add_surface(self, vertices: np.ndarray, faces: np.ndarray, name: str | None = None) -> Tuple[bool, Any]:
        """Add 3D mesh surface to the viewer."""
        args = [vertices, faces]
        if name is not None:
            args.append(name)
        return self.send_command("napari-socket.add_surface", args)

    def add_vectors(self, vectors: np.ndarray, name: str | None = None) -> Tuple[bool, Any]:
        """Add vector field arrows to the viewer."""
        args = [vectors]
        if name is not None:
            args.append(name)
        return self.send_command("napari-socket.add_vectors", args)

    # ------------------------------------------------------------------
    # Data Export & Save Functions
    # ------------------------------------------------------------------
    def save_layers(self, file_path: str, layer_names: list | None = None) -> Tuple[bool, Any]:
        """Save one or more layers to disk."""
        args = [file_path]
        if layer_names is not None:
            args.append(layer_names)
        return self.send_command("napari-socket.save_layers", args)

    def get_layer_data(self, layer_name: str | int) -> Tuple[bool, Any]:
        """Extract the raw data from a layer."""
        return self.send_command("napari-socket.get_layer_data", [layer_name])

    # ------------------------------------------------------------------
    # Advanced Visualization Controls
    # ------------------------------------------------------------------
    def set_scale_bar(self, visible: bool = True, unit: str = 'um') -> Tuple[bool, Any]:
        """Show or hide the scale bar."""
        return self.send_command("napari-socket.set_scale_bar", [visible, unit])

    def set_axis_labels(self, labels: list) -> Tuple[bool, Any]:
        """Set custom labels for the axes."""
        return self.send_command("napari-socket.set_axis_labels", [labels])

    def set_view_mode(self, mode: str) -> Tuple[bool, Any]:
        """Switch between different view modes."""
        return self.send_command("napari-socket.set_view_mode", [mode])

    def set_layer_visibility(self, layer_name: str | int, visible: bool) -> Tuple[bool, Any]:
        """Show or hide a specific layer."""
        return self.send_command("napari-socket.set_layer_visibility", [layer_name, visible])

    # ------------------------------------------------------------------
    # Measurement & Analysis Functions
    # ------------------------------------------------------------------
    def measure_distance(self, point1: list, point2: list) -> Tuple[bool, Any]:
        """Calculate distance between two points in the data."""
        return self.send_command("napari-socket.measure_distance", [point1, point2])

    def get_layer_statistics(self, layer_name: str | int) -> Tuple[bool, Any]:
        """Get basic stats (min, max, mean, std) for a layer."""
        return self.send_command("napari-socket.get_layer_statistics", [layer_name])

    def crop_layer(self, layer_name: str | int, bounds: list) -> Tuple[bool, Any]:
        """Crop a layer to a specific region."""
        return self.send_command("napari-socket.crop_layer", [layer_name, bounds])

    # ------------------------------------------------------------------
    # Time Series & Multi-dimensional Data
    # ------------------------------------------------------------------
    def set_channel(self, channel_index: int) -> Tuple[bool, Any]:
        """Switch to a specific channel in multi-channel data."""
        return self.send_command("napari-socket.set_channel", [channel_index])

    def set_z_slice(self, z_index: int) -> Tuple[bool, Any]:
        """Jump to a specific z-slice in 3D data."""
        return self.send_command("napari-socket.set_z_slice", [z_index])

    def play_animation(self, start_frame: int, end_frame: int, fps: int = 10) -> Tuple[bool, Any]:
        """Animate through a time series at specified FPS."""
        return self.send_command("napari-socket.play_animation", [start_frame, end_frame, fps])

    # ------------------------------------------------------------------
    # Enhanced Channel Management Functions
    # ------------------------------------------------------------------
    def get_channel_info(self, layer_name: str | int) -> Tuple[bool, Any]:
        """Get information about channels in a layer."""
        return self.send_command("napari-socket.get_channel_info", [layer_name])

    def split_channels(self, layer_name: str | int) -> Tuple[bool, Any]:
        """Split a multi-channel layer into separate single-channel layers."""
        return self.send_command("napari-socket.split_channels", [layer_name])

    def merge_channels(self, layer_names: list, output_name: str = None) -> Tuple[bool, Any]:
        """Merge multiple single-channel layers into one multi-channel layer."""
        args = [layer_names]
        if output_name is not None:
            args.append(output_name)
        return self.send_command("napari-socket.merge_channels", args)

# ---------------------------------------------------------------------------
# quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mgr = NapariManager()
    ok, msg = mgr.open_file("/path/to/image.tif")
    print(ok, msg)

