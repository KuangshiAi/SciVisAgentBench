"""
TopoPilot MCP Agent Adapter

Wraps TinyAgent with the TopoPilot MCP server for topology visualization tasks.
Connects via HTTP to the TopoPilot MCP server and executes topology analysis
tasks (critical points, degenerate points, merge trees, Morse-Smale segmentation, etc.).
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


@register_agent("topopilot_mcp")
class TopoPilotMCPAgent(BaseAgent):
    """
    TopoPilot MCP agent using TinyAgent and the TopoPilot HTTP MCP server.

    This agent uses the Model Context Protocol (MCP) to interact with TopoPilot
    for performing topological data analysis on scalar, vector, and tensor fields.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TopoPilot MCP agent.

        Args:
            config: Configuration dictionary containing:
                   - model: LLM model to use (e.g., "gpt-4o")
                   - provider: LLM provider ("openai", "anthropic", etc.)
                   - servers: List of MCP server configurations
                   - pricing: Optional pricing information for cost calculation
        """
        config["eval_mode"] = "mcp"
        config["agent_name"] = "topopilot_mcp"
        super().__init__(config)

        self.tiny_agent = None
        self.current_config_path = None

    async def setup(self):
        """Setup the agent (called once before running any tasks)."""
        print(f"Setting up TopoPilot MCP agent with model: {self.config.get('model', 'unknown')}")

    async def teardown(self):
        """Cleanup the agent (called once after all tasks complete)."""
        print("Tearing down TopoPilot MCP agent")

    def _create_task_config_file(self, task_config: Dict[str, Any]) -> str:
        """Create a temporary config file for this specific task."""
        config = self.config.copy()

        temp_config_dir = Path(tempfile.gettempdir()) / "scivisagent_bench_configs"
        temp_config_dir.mkdir(exist_ok=True)

        case_name = task_config.get("case_name", "unknown")
        case_config_file = temp_config_dir / f"topopilot_config_{case_name}_{int(time.time())}.json"

        with open(case_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        return str(case_config_file)

    async def prepare_task(self, task_config: Dict[str, Any]):
        """Prepare for a specific task."""
        self.current_config_path = self._create_task_config_file(task_config)
        print(f"Created task config: {self.current_config_path}")

        # Pre-create the results directory so export tools don't fail
        case_name = task_config.get("case_name", "")
        data_dir = task_config.get("data_dir", "")
        if case_name and data_dir:
            results_dir = Path(data_dir) / case_name / "results" / self.eval_mode
            results_dir.mkdir(parents=True, exist_ok=True)
            print(f"Ensured results directory exists: {results_dir}")

    async def cleanup_task(self, task_config: Dict[str, Any]):
        """Cleanup after a specific task."""
        if self.current_config_path and os.path.exists(self.current_config_path):
            try:
                os.unlink(self.current_config_path)
                print(f"Cleaned up config file: {self.current_config_path}")
            except Exception as e:
                print(f"Warning: Could not clean up config file: {e}")
        self.current_config_path = None

    async def run_task(self, task_description: str, task_config: Dict[str, Any]) -> AgentResult:
        """
        Run a single topology analysis task using TopoPilot MCP.

        Args:
            task_description: Natural language description of the topology task
            task_config: Task configuration including case_name, case_dir, data_dir

        Returns:
            AgentResult with success status, response, output files, and metadata
        """
        start_time = time.time()

        try:
            # Create agent from config file
            agent = TinyAgent.from_config_file(self.current_config_path)

            response_parts = []

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
                print(f"Starting execution for task...")
                print(f"Task preview: {task_description[:200]}...")

                async for chunk in agent.run(task_description):
                    # Try to extract usage from chunk
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
                            response_parts.append(content)
                    elif hasattr(chunk, 'role') and chunk.role == "tool":
                        content_preview = chunk.content[:200] if len(chunk.content) > 200 else chunk.content
                        tool_message = f"\n[Tool: {chunk.name}] {content_preview}"
                        response_parts.append(f"\n[Tool: {chunk.name}] {chunk.content}")
                        print(tool_message)

            full_response = "".join(response_parts)
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

            return AgentResult(
                success=True,
                response=full_response,
                output_files={},
                metadata={
                    "duration": duration,
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
            import traceback
            traceback.print_exc()
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"duration": duration}
            )
