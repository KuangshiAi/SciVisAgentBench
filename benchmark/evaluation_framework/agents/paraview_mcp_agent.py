"""
ParaView MCP Agent Adapter

Wraps the existing ParaView MCP implementation to work with the evaluation framework.
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


@register_agent("paraview_mcp")
class ParaViewMCPAgent(BaseAgent):
    """
    ParaView MCP agent using TinyAgent and MCP servers.

    This agent uses the Model Context Protocol (MCP) to interact with ParaView
    for creating scientific visualizations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ParaView MCP agent.

        Args:
            config: Configuration dictionary containing:
                   - model: LLM model to use (e.g., "gpt-4o")
                   - provider: LLM provider ("openai", "anthropic", etc.)
                   - servers: List of MCP server configurations
                   - pricing: Optional pricing information for cost calculation
        """
        config["eval_mode"] = "mcp"
        config["agent_name"] = "paraview_mcp"
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
            return len(text.split())
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text.split())

    async def setup(self):
        """Setup the agent (called once before running any tasks)."""
        print(f"Setting up ParaView MCP agent with model: {self.config.get('model', 'unknown')}")

    async def teardown(self):
        """Cleanup the agent (called once after all tasks complete)."""
        print("Tearing down ParaView MCP agent")

    def _create_task_config_file(self, task_config: Dict[str, Any]) -> str:
        """Create a temporary config file for this specific task."""
        # Load the base config
        config = self.config.copy()

        # Create a temporary config file for this task
        temp_config_dir = Path(tempfile.gettempdir()) / "scivisagent_bench_configs"
        temp_config_dir.mkdir(exist_ok=True)

        case_name = task_config.get("case_name", "unknown")
        case_config_file = temp_config_dir / f"config_{case_name}_{int(time.time())}.json"

        # Add test case name as environment variable for unique session naming
        for server in config.get("servers", []):
            if "env" not in server:
                server["env"] = {}
            server["env"]["SCIVISBENCH_CASE_NAME"] = case_name

        # Save the modified config to the temporary file
        with open(case_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        return str(case_config_file)

    async def _clear_paraview_state(self, agent: TinyAgent) -> None:
        """Clear ParaView pipeline state before starting."""
        try:
            print("Clearing ParaView pipeline state through MCP...")

            # Send a simple clear_pipeline command to the agent
            clear_message = "Call the clear_pipeline_and_reset tool to clear all sources from the ParaView pipeline."

            response_parts = []
            async for chunk in agent.run(clear_message):
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content = delta.content
                        response_parts.append(content)
                elif hasattr(chunk, 'role') and chunk.role == "tool":
                    tool_message = f"[Tool: {chunk.name}] {chunk.content}"
                    response_parts.append(tool_message)
                    if chunk.name == "clear_pipeline_and_reset":
                        print(f"Pipeline clearing result: {chunk.content}")

            if response_parts:
                print("Pipeline clearing completed")
            else:
                print("No response received for pipeline clearing")

        except Exception as e:
            print(f"⚠️  Warning: Could not clear ParaView state: {e}")

    async def prepare_task(self, task_config: Dict[str, Any]):
        """Prepare for a specific task."""
        # Create task-specific config
        self.current_config_path = self._create_task_config_file(task_config)
        print(f"Created task config: {self.current_config_path}")

    async def cleanup_task(self, task_config: Dict[str, Any]):
        """Cleanup after a specific task."""
        # Clean up temporary config file
        if self.current_config_path and os.path.exists(self.current_config_path):
            try:
                os.unlink(self.current_config_path)
                print(f"Cleaned up config file: {self.current_config_path}")
            except Exception as e:
                print(f"Warning: Could not clean up config file: {e}")
        self.current_config_path = None

    async def run_task(self, task_description: str, task_config: Dict[str, Any]) -> AgentResult:
        """
        Run a single visualization task using ParaView MCP.

        Args:
            task_description: Natural language description of the visualization task
            task_config: Task configuration including case_name, case_dir, data_dir

        Returns:
            AgentResult with success status, response, output files, and metadata
        """
        start_time = time.time()

        try:
            # Create agent from config file
            agent = TinyAgent.from_config_file(self.current_config_path)

            response_parts = []

            async with agent:
                await agent.load_tools()
                print(f"Agent loaded with {len(agent.available_tools)} tools")

                # Clear ParaView state at the beginning
                print("Clearing ParaView state for fresh start...")
                await self._clear_paraview_state(agent)

                # Run the task
                print(f"Starting execution for task...")
                print(f"Task preview: {task_description[:200]}...")

                async for chunk in agent.run(task_description):
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content = delta.content
                            response_parts.append(content)
                            # Don't print content to avoid flooding console with base64 images
                            # print(content, end="", flush=True)
                    elif hasattr(chunk, 'role') and chunk.role == "tool":
                        # Only show tool name and truncated content to avoid base64 spam
                        content_preview = chunk.content[:100] if len(chunk.content) > 100 else chunk.content
                        if "data:image" in chunk.content or "base64" in chunk.content:
                            tool_message = f"\n[Tool: {chunk.name}] <image data truncated>"
                        else:
                            tool_message = f"\n[Tool: {chunk.name}] {content_preview}"
                        response_parts.append(f"\n[Tool: {chunk.name}] {chunk.content}")  # Store full content
                        print(tool_message)  # Print truncated version

            full_response = "".join(response_parts)
            duration = time.time() - start_time

            # Get output file paths
            dirs = self.get_result_directories(task_config["case_dir"], task_config["case_name"])
            state_file = dirs["results_dir"] / f"{task_config['case_name']}.pvsm"

            # Count tokens
            input_tokens = self.count_tokens(task_description)
            output_tokens = self.count_tokens(full_response)

            return AgentResult(
                success=True,
                response=full_response,
                output_files={"state": str(state_file)},
                metadata={
                    "duration": duration,
                    "token_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"duration": duration}
            )
