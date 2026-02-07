"""
Pre-built agent adapters for existing agents.

This module provides ready-to-use adapters for the existing agents
(ParaView MCP, Napari MCP, ChatVis, GMX-VMD MCP) that work with
the evaluation framework.
"""

from .paraview_mcp_agent import ParaViewMCPAgent
from .napari_mcp_agent import NapariMCPAgent
from .chatvis_agent import ChatVisAgent
from .gmx_vmd_mcp_agent import GmxVmdMcpAgent
from .topopilot_mcp_agent import TopoPilotMCPAgent

__all__ = [
    'ParaViewMCPAgent',
    'NapariMCPAgent',
    'ChatVisAgent',
    'GmxVmdMcpAgent',
    'TopoPilotMCPAgent',
]
