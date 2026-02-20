"""
Napari MCP Agent Adapter

Wraps the existing Napari MCP implementation to work with the evaluation framework.
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


@register_agent("napari_mcp")
class NapariMCPAgent(BaseAgent):
    """
    Napari MCP agent using TinyAgent and MCP servers.

    This agent uses the Model Context Protocol (MCP) to interact with Napari
    for bioimage visualization and analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Napari MCP agent.

        Args:
            config: Configuration dictionary containing:
                   - model: LLM model to use
                   - provider: LLM provider
                   - servers: List of MCP server configurations
        """
        config["eval_mode"] = "mcp"
        config["agent_name"] = "napari_mcp"
        super().__init__(config)

        self.tiny_agent = None
        self.current_config_path = None

    async def setup(self):
        """Setup the agent."""
        print(f"Setting up Napari MCP agent with model: {self.config.get('model', 'unknown')}")

    async def teardown(self):
        """Cleanup the agent."""
        print("Tearing down Napari MCP agent")

    def _create_task_config_file(self, task_config: Dict[str, Any]) -> str:
        """Create a temporary config file for this specific task."""
        config = self.config.copy()

        # Create a temporary config file
        temp_config_dir = Path(tempfile.gettempdir()) / "scivisagent_bench_configs"
        temp_config_dir.mkdir(exist_ok=True)

        case_name = task_config.get("case_name", "unknown")
        case_config_file = temp_config_dir / f"config_{case_name}_{int(time.time())}.json"

        # Modify config for this specific test case
        working_dir = task_config.get("data_dir", task_config.get("working_dir"))
        for server in config.get("servers", []):
            if server.get("command") == "python" and "napari_mcp" in str(server.get("args", [])):
                server["cwd"] = str(working_dir)

        # Add test case name as environment variable
        for server in config.get("servers", []):
            if "env" not in server:
                server["env"] = {}
            server["env"]["TEST_CASE_NAME"] = case_name
            # DO NOT set NAPARI_WORKING_DIR - the agent should use absolute paths
            # instead of relying on os.chdir() which can cause path confusion

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
        Run a single bioimage visualization task using Napari MCP.

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

            # Track actual token usage from API
            # Use message ID deduplication as per Claude Agent SDK docs
            processed_message_ids = set()
            total_input_tokens = 0
            total_output_tokens = 0
            total_cache_creation_tokens = 0
            total_cache_read_tokens = 0

            async with agent:
                await agent.load_tools()
                print(f"Agent loaded with {len(agent.available_tools)} tools")

                # Run the task
                print(f"Starting execution...")

                async for chunk in agent.run(task_description):
                    # Try to extract usage from chunk (if available)
                    # Per Claude Agent SDK: Only count each message ID once
                    # Multiple messages with same ID have identical usage - charge only once per step
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        # Get message ID to deduplicate
                        message_id = getattr(chunk, 'id', None)
                        if message_id and message_id not in processed_message_ids:
                            processed_message_ids.add(message_id)
                            usage = chunk.usage

                            # Anthropic-style field names
                            if hasattr(usage, 'input_tokens') and usage.input_tokens:
                                total_input_tokens += usage.input_tokens
                            if hasattr(usage, 'output_tokens') and usage.output_tokens:
                                total_output_tokens += usage.output_tokens

                            # Anthropic cache tokens (CRITICAL for accurate counting!)
                            if hasattr(usage, 'cache_creation_input_tokens') and usage.cache_creation_input_tokens:
                                total_cache_creation_tokens += usage.cache_creation_input_tokens
                            if hasattr(usage, 'cache_read_input_tokens') and usage.cache_read_input_tokens:
                                total_cache_read_tokens += usage.cache_read_input_tokens

                            # OpenAI-style field names (fallback)
                            if not hasattr(usage, 'input_tokens') and hasattr(usage, 'prompt_tokens') and usage.prompt_tokens:
                                total_input_tokens += usage.prompt_tokens
                            if not hasattr(usage, 'output_tokens') and hasattr(usage, 'completion_tokens') and usage.completion_tokens:
                                total_output_tokens += usage.completion_tokens

                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content = delta.content
                            assistant_response_parts.append(content)
                            full_response_parts.append(content)
                            print(content, end="", flush=True)
                    elif hasattr(chunk, 'role') and chunk.role == "tool":
                        tool_message = f"\n[Tool: {chunk.name}] {chunk.content}"
                        full_response_parts.append(tool_message)
                        print(tool_message)

            # For evaluation, use only the assistant's text (without tool logs)
            assistant_response = "".join(assistant_response_parts)
            full_response = "".join(full_response_parts)

            duration = time.time() - start_time

            # Use actual API token usage if available, otherwise estimate
            if total_input_tokens > 0 or total_output_tokens > 0:
                # Use actual API-reported usage
                # Per Anthropic docs: total_input = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
                input_tokens = total_input_tokens + total_cache_creation_tokens + total_cache_read_tokens
                output_tokens = total_output_tokens
                token_source = "api_reported"
            else:
                # Fallback to comprehensive estimation
                input_tokens, output_tokens = self.estimate_comprehensive_tokens(
                    agent, task_description, full_response
                )
                token_source = "estimated"

            # Get output file paths
            dirs = self.get_result_directories(task_config["case_dir"], task_config["case_name"])

            return AgentResult(
                success=True,
                response=full_response,
                metadata={
                    "duration": duration,
                    "assistant_response": assistant_response,  # For evaluation
                    # Token usage is handled by unified_runner, not stored in metadata
                    "_token_info": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_creation_input_tokens": total_cache_creation_tokens,
                        "cache_read_input_tokens": total_cache_read_tokens,
                        "source": token_source
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
