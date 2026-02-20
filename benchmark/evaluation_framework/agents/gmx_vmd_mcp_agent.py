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
        for i, server in enumerate(config.get("servers", [])):
            # Handle both nested and flat server configurations
            server_config = server.get("config", server)

            # Check if this is the GMX-VMD MCP server
            args_str = str(server_config.get("args", []))
            if "gmx_vmd_mcp" in args_str or "mcp_server.py" in args_str:
                # For nested config, set cwd in the config section
                if "config" in server:
                    server["config"]["cwd"] = str(working_dir)
                else:
                    server["cwd"] = str(working_dir)

                # Update environment variables (always in config section)
                if "config" in server:
                    if "env" not in server["config"]:
                        server["config"]["env"] = {}
                    env = server["config"]["env"]
                else:
                    if "env" not in server:
                        server["env"] = {}
                    env = server["env"]

                # Set test case name
                env["TEST_CASE_NAME"] = case_name

                # Set workflow base directory if not already set
                if "WORKFLOW_BASE_DIR" not in env:
                    # Create workspace directory for molecular visualization tasks
                    workspace_dir = Path(working_dir) / "workspace"
                    workspace_dir.mkdir(parents=True, exist_ok=True)
                    env["WORKFLOW_BASE_DIR"] = str(workspace_dir)

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

        # Timeout in seconds (default: 10 minutes)
        timeout_seconds = task_config.get("timeout", 600)

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

                # Run the task with timeout
                try:
                    async def run_with_chunks():
                        """Wrapper to run agent and collect chunks."""
                        nonlocal total_input_tokens, total_output_tokens, total_cache_creation_tokens, total_cache_read_tokens, processed_message_ids

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

                                    input_added = 0
                                    output_added = 0
                                    cache_creation_added = 0
                                    cache_read_added = 0

                                    # Anthropic-style field names
                                    if hasattr(usage, 'input_tokens') and usage.input_tokens:
                                        total_input_tokens += usage.input_tokens
                                        input_added = usage.input_tokens
                                    if hasattr(usage, 'output_tokens') and usage.output_tokens:
                                        total_output_tokens += usage.output_tokens
                                        output_added = usage.output_tokens

                                    # Anthropic cache tokens (CRITICAL for accurate counting!)
                                    if hasattr(usage, 'cache_creation_input_tokens') and usage.cache_creation_input_tokens:
                                        total_cache_creation_tokens += usage.cache_creation_input_tokens
                                        cache_creation_added = usage.cache_creation_input_tokens
                                    if hasattr(usage, 'cache_read_input_tokens') and usage.cache_read_input_tokens:
                                        total_cache_read_tokens += usage.cache_read_input_tokens
                                        cache_read_added = usage.cache_read_input_tokens

                                    # OpenAI-style field names (fallback)
                                    if not input_added and hasattr(usage, 'prompt_tokens') and usage.prompt_tokens:
                                        total_input_tokens += usage.prompt_tokens
                                        input_added = usage.prompt_tokens
                                    if not output_added and hasattr(usage, 'completion_tokens') and usage.completion_tokens:
                                        total_output_tokens += usage.completion_tokens
                                        output_added = usage.completion_tokens

                                    # Calculate true total
                                    true_total_input = total_input_tokens + total_cache_creation_tokens + total_cache_read_tokens
                                    print(f"\n[TOKEN] Step {len(processed_message_ids)}:")
                                    print(f"  +{input_added} input, +{cache_creation_added} cache_write, +{cache_read_added} cache_read, +{output_added} output")
                                    print(f"  Total: {true_total_input:,} input, {total_output_tokens:,} output")

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

                    # Execute with timeout
                    import asyncio
                    await asyncio.wait_for(run_with_chunks(), timeout=timeout_seconds)

                except asyncio.TimeoutError:
                    duration = time.time() - start_time
                    error_msg = f"Task execution timed out after {timeout_seconds} seconds"
                    print(f"\n❌ {error_msg}")
                    return AgentResult(
                        success=False,
                        error=error_msg,
                        metadata={
                            "duration": duration,
                            "timeout": True,
                            "partial_response": "".join(full_response_parts) if full_response_parts else None
                        }
                    )

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
                print(f"\n✓ Final token count: {input_tokens:,} input, {output_tokens:,} output (source: {token_source})")
            else:
                # Fallback to comprehensive estimation
                input_tokens, output_tokens = self.estimate_comprehensive_tokens(
                    agent, task_description, full_response
                )
                token_source = "estimated"
                print(f"\n⚠️  Warning: Token usage is estimated (includes system prompt, tools, history). "
                      f"Actual usage may vary by ±20%.")

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
                        "source": token_source  # Track whether tokens are from API or estimated
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
