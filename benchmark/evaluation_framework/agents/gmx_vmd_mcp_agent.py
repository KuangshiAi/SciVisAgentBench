"""
GMX-VMD MCP Agent Adapter

Wraps the GMX-VMD MCP implementation to work with the evaluation framework.
"""

import os
import sys
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation_framework.base_agent import BaseAgent, AgentResult
from evaluation_framework.agent_registry import register_agent
from tiny_agent.agent import TinyAgent


@register_agent("gmx_vmd_mcp")
class GmxVmdMcpAgent(BaseAgent):
    """
    GMX-VMD MCP agent using TinyAgent and MCP servers.

    This agent uses the Model Context Protocol (MCP) to interact with GROMACS and VMD
    for molecular visualization and analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GMX-VMD MCP agent.

        Args:
            config: Configuration dictionary containing:
                   - model: LLM model to use
                   - provider: LLM provider
                   - servers: List of MCP server configurations
        """
        config["eval_mode"] = "mcp"
        config["agent_name"] = "gmx_vmd_mcp"
        super().__init__(config)

        self.tiny_agent = None
        self.current_config_path = None

        # Initialize token counter
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is None:
            return len(text.split()) * 2  # Rough approximation
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text.split()) * 2

    async def setup(self):
        """Setup the agent."""
        print(f"Setting up GMX-VMD MCP agent with model: {self.config.get('model', 'unknown')}")

    async def teardown(self):
        """Cleanup the agent."""
        print("Tearing down GMX-VMD MCP agent")

    def _create_task_config_file(self, task_config: Dict[str, Any]) -> str:
        """Create a temporary config file for this specific task."""
        config = self.config.copy()

        # Create a temporary config file
        temp_config_dir = Path(tempfile.gettempdir()) / "scivisagent_bench_configs"
        temp_config_dir.mkdir(exist_ok=True)

        case_name = task_config.get("case_name", "unknown")
        case_config_file = temp_config_dir / f"config_gmx_vmd_{case_name}_{int(time.time())}.json"

        # Modify config for this specific test case
        working_dir = task_config.get("data_dir", task_config.get("working_dir"))

        # Update server configuration with working directory
        for server in config.get("servers", []):
            # Check if this is the GMX-VMD MCP server
            if "gmx_vmd_mcp" in str(server.get("args", [])) or "mcp_server.py" in str(server.get("args", [])):
                server["cwd"] = str(working_dir)

                # Update environment variables
                if "env" not in server:
                    server["env"] = {}

                # Set test case name
                server["env"]["TEST_CASE_NAME"] = case_name

                # Set workflow base directory if not already set
                if "WORKFLOW_BASE_DIR" not in server["env"]:
                    # Create workspace directory for molecular visualization tasks
                    workspace_dir = Path(working_dir) / "workspace"
                    workspace_dir.mkdir(parents=True, exist_ok=True)
                    server["env"]["WORKFLOW_BASE_DIR"] = str(workspace_dir)

        # Save config
        with open(case_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        return str(case_config_file)

    async def prepare_task(self, task_config: Dict[str, Any]):
        """Prepare for a specific task."""
        self.current_config_path = self._create_task_config_file(task_config)
        print(f"Created task config: {self.current_config_path}")

    async def cleanup_task(self, task_config: Dict[str, Any]):
        """Cleanup after a specific task."""
        if self.current_config_path and os.path.exists(self.current_config_path):
            try:
                os.unlink(self.current_config_path)
            except Exception as e:
                print(f"Warning: Could not clean up config file: {e}")
        self.current_config_path = None

    async def run_task(self, task_description: str, task_config: Dict[str, Any]) -> AgentResult:
        """
        Run a single molecular visualization task using GMX-VMD MCP.

        Args:
            task_description: Natural language description of the task
            task_config: Task configuration

        Returns:
            AgentResult with success status, response, and metadata
        """
        start_time = time.time()

        try:
            # Create agent from config file
            agent = TinyAgent.from_config_file(self.current_config_path)

            # Separate assistant responses from tool calls
            assistant_response_parts = []  # Only assistant's text
            full_response_parts = []  # Complete log including tool calls

            async with agent:
                await agent.load_tools()
                print(f"Agent loaded with {len(agent.available_tools)} tools")

                # Run the task
                print(f"Starting execution...")

                async for chunk in agent.run(task_description):
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content = delta.content
                            assistant_response_parts.append(content)
                            full_response_parts.append(content)
                            print(content, end="", flush=True)
                    elif hasattr(chunk, 'role') and chunk.role == "tool":
                        # Truncate tool messages if they contain large data
                        tool_content = chunk.content
                        if len(tool_content) > 500:
                            tool_content = tool_content[:500] + "... [truncated]"
                        tool_message = f"\n[Tool: {chunk.name}] {tool_content}"
                        full_response_parts.append(tool_message)
                        print(tool_message)

            # For evaluation, use only the assistant's text (without tool logs)
            assistant_response = "".join(assistant_response_parts)
            full_response = "".join(full_response_parts)

            duration = time.time() - start_time

            # Count tokens
            input_tokens = self.count_tokens(task_description)
            output_tokens = self.count_tokens(full_response)

            # Get output file paths
            dirs = self.get_result_directories(task_config["case_dir"], task_config["case_name"])

            return AgentResult(
                success=True,
                response=full_response,
                metadata={
                    "duration": duration,
                    "assistant_response": assistant_response,  # For evaluation
                    "token_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            import traceback
            error_details = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"❌ Error during task execution: {error_details}")
            return AgentResult(
                success=False,
                error=error_details,
                metadata={"duration": duration}
            )
