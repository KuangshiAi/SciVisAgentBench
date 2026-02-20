"""
Base Agent Interface

This module defines the abstract base class that all agents must implement
to be compatible with the SciVisAgentBench evaluation framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import json


class AgentResult:
    """
    Standardized result object returned by agents.

    This ensures consistent result formatting across all agents.
    """

    def __init__(
        self,
        success: bool,
        response: str = "",
        error: Optional[str] = None,
        output_files: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agent result.

        Args:
            success: Whether the task completed successfully
            response: Text response from the agent
            error: Error message if task failed
            output_files: Dictionary mapping file types to paths (e.g., {"state": "path/to/state.pvsm"})
            metadata: Additional metadata (e.g., execution time, token usage)
        """
        self.success = success
        self.response = response
        self.error = error
        self.output_files = output_files or {}
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "success": self.success,
            "response": self.response,
            "error": self.error,
            "output_files": self.output_files,
            "metadata": self.metadata
        }


class BaseAgent(ABC):
    """
    Abstract base class for scientific visualization agents.

    All agents must inherit from this class and implement the required methods.
    This provides a consistent interface for the evaluation framework.

    Example:
        @register_agent("my_agent")
        class MyAgent(BaseAgent):
            def __init__(self, config: Dict[str, Any]):
                super().__init__(config)
                # Initialize your agent

            async def run_task(self, task_description: str, task_config: Dict[str, Any]) -> AgentResult:
                # Implement your agent's logic
                result = your_agent_logic(task_description)
                return AgentResult(
                    success=True,
                    response=result.text,
                    output_files={"state": result.state_file},
                    metadata={"duration": result.time}
                )
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent with configuration.

        Args:
            config: Configuration dictionary containing agent-specific settings.
                   Common keys include:
                   - model: LLM model to use
                   - api_key: API key for LLM provider
                   - provider: LLM provider (openai, anthropic, etc.)
                   - servers: MCP server configurations (for MCP agents)
                   - Any agent-specific configuration
        """
        self.config = config
        self.agent_name = config.get("agent_name", self.__class__.__name__)
        self.eval_mode = config.get("eval_mode", "generic")  # e.g., "mcp", "pvpython", "generic"

        # Initialize tokenizer for token counting
        self._tokenizer = None
        try:
            import tiktoken
            # Use cl100k_base which works for both GPT-4 and is a reasonable approximation for other models
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or fallback approximation.

        Args:
            text: Text to count tokens for

        Returns:
            int: Estimated token count
        """
        if self._tokenizer is None:
            # Fallback: rough approximation (1 token ≈ 0.75 words)
            return int(len(text.split()) * 1.33)
        try:
            return len(self._tokenizer.encode(text))
        except Exception:
            return int(len(text.split()) * 1.33)

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """
        Calculate monetary cost based on token usage and pricing configuration.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dict containing cost breakdown and total cost
        """
        pricing = self.config.get("price", {})

        # Parse pricing strings (e.g., "$3.00" -> 3.00)
        def parse_price(price_str):
            if isinstance(price_str, (int, float)):
                return float(price_str)
            if isinstance(price_str, str):
                # Remove $ and any other non-numeric characters except . and -
                price_str = price_str.replace('$', '').replace(',', '').strip()
                try:
                    return float(price_str)
                except ValueError:
                    return 0.0
            return 0.0

        input_price_per_1m = parse_price(pricing.get("input_per_1m_tokens", 0))
        output_price_per_1m = parse_price(pricing.get("output_per_1m_tokens", 0))

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * output_price_per_1m
        total_cost = input_cost + output_cost

        return {
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "pricing": {
                "input_per_1m_tokens": input_price_per_1m,
                "output_per_1m_tokens": output_price_per_1m,
                "currency": "USD"
            }
        }

    def estimate_comprehensive_tokens(self, agent, task_description: str, full_response: str,
                                     num_tools: int = 0) -> tuple:
        """
        Estimate comprehensive token usage including all overhead.

        This method accounts for:
        - System prompts
        - Tool schemas
        - Conversation history
        - Response text
        - Tool call overhead

        Args:
            agent: The agent object (e.g., TinyAgent) with messages and tools
            task_description: User's task description
            full_response: Complete response text
            num_tools: Number of tools available (if agent doesn't expose this)

        Returns:
            tuple: (input_tokens, output_tokens)
        """
        # === INPUT TOKENS ===

        # 1. System prompt
        system_prompt_tokens = 0
        if hasattr(agent, 'prompt') and agent.prompt:
            system_prompt_tokens = self.count_tokens(agent.prompt)
        elif hasattr(agent, 'system_prompt') and agent.system_prompt:
            system_prompt_tokens = self.count_tokens(agent.system_prompt)

        # 2. Tool definitions
        tool_schema_tokens = 0
        if hasattr(agent, 'available_tools') and agent.available_tools:
            num_tools = len(agent.available_tools)
        if num_tools > 0:
            # Each tool schema: ~100-200 tokens (name, description, parameters)
            tool_schema_tokens = num_tools * 150

        # 3. Task description
        task_tokens = self.count_tokens(task_description)

        # 4. Conversation history
        history_tokens = 0
        if hasattr(agent, 'messages') and agent.messages:
            for msg in agent.messages:
                # Count message content
                content = msg.get('content', '')
                if isinstance(content, str):
                    history_tokens += self.count_tokens(content)
                elif isinstance(content, list):
                    # Handle multi-modal content
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            history_tokens += self.count_tokens(item.get('text', ''))

                # Count tool calls in history (part of input context on subsequent turns)
                if msg.get('tool_calls'):
                    for tc in msg.get('tool_calls', []):
                        if isinstance(tc, dict):
                            # Function name + arguments
                            history_tokens += 20  # Base overhead
                            if tc.get('function', {}).get('arguments'):
                                args_str = tc['function']['arguments']
                                if isinstance(args_str, str):
                                    history_tokens += self.count_tokens(args_str)
                                else:
                                    history_tokens += self.count_tokens(json.dumps(args_str))

        # 5. API formatting overhead (approximately 10% for JSON structure, role labels, etc.)
        total_input = int((system_prompt_tokens + tool_schema_tokens + task_tokens + history_tokens) * 1.1)

        # === OUTPUT TOKENS ===

        # 1. Response text
        response_tokens = self.count_tokens(full_response)

        # 2. Tool calls overhead
        tool_call_tokens = 0
        if hasattr(agent, 'messages') and agent.messages:
            for msg in agent.messages:
                if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                    for tc in msg.get('tool_calls', []):
                        if isinstance(tc, dict):
                            # JSON structure + function name + arguments
                            tool_call_tokens += 30  # Base overhead
                            if tc.get('function', {}).get('arguments'):
                                args_str = tc['function']['arguments']
                                if isinstance(args_str, str):
                                    tool_call_tokens += self.count_tokens(args_str)
                                else:
                                    tool_call_tokens += self.count_tokens(json.dumps(args_str))

        # 3. Formatting overhead (15% for JSON structure, stop sequences, etc.)
        total_output = int((response_tokens + tool_call_tokens) * 1.15)

        return total_input, total_output

    @abstractmethod
    async def run_task(self, task_description: str, task_config: Dict[str, Any]) -> AgentResult:
        """
        Run a single task and return the result.

        This is the main method that agents must implement. The framework will call this
        method for each test case in the benchmark.

        Args:
            task_description: The natural language description of the visualization task
            task_config: Additional configuration for this specific task, including:
                        - case_name: Name of the test case
                        - case_dir: Directory containing test case data
                        - data_dir: Directory containing input data
                        - working_dir: Working directory for the agent
                        - Any task-specific parameters

        Returns:
            AgentResult: Standardized result object containing:
                        - success: Whether task completed successfully
                        - response: Agent's text response
                        - error: Error message if failed
                        - output_files: Dictionary of output file paths
                        - metadata: Additional information (timing, tokens, etc.)

        Example:
            async def run_task(self, task_description: str, task_config: Dict[str, Any]) -> AgentResult:
                try:
                    # Your agent logic here
                    result = await self.agent.process(task_description)

                    # Save outputs to expected locations
                    output_dir = Path(task_config["case_dir"]) / "results" / self.eval_mode
                    output_dir.mkdir(parents=True, exist_ok=True)
                    state_file = output_dir / f"{task_config['case_name']}.pvsm"

                    return AgentResult(
                        success=True,
                        response=result.text,
                        output_files={"state": str(state_file)},
                        metadata={"duration": result.duration, "tokens": result.tokens}
                    )
                except Exception as e:
                    return AgentResult(success=False, error=str(e))
        """
        pass

    async def setup(self):
        """
        Optional: Setup method called before running any tasks.

        Use this for one-time initialization (e.g., starting servers, loading models).
        """
        pass

    async def teardown(self):
        """
        Optional: Teardown method called after all tasks complete.

        Use this for cleanup (e.g., stopping servers, releasing resources).
        """
        pass

    async def prepare_task(self, task_config: Dict[str, Any]):
        """
        Optional: Prepare for a specific task.

        Called before run_task() for each test case.
        Use this for task-specific setup (e.g., clearing state, setting working directory).

        Args:
            task_config: Configuration for the upcoming task
        """
        pass

    async def cleanup_task(self, task_config: Dict[str, Any]):
        """
        Optional: Cleanup after a specific task.

        Called after run_task() for each test case.
        Use this for task-specific cleanup.

        Args:
            task_config: Configuration for the completed task
        """
        pass

    def get_result_directories(self, case_dir: str, case_name: str) -> Dict[str, Path]:
        """
        Get standard directory paths for saving results.

        This ensures consistent directory structure across all agents.

        Args:
            case_dir: Test case directory
            case_name: Test case name

        Returns:
            Dictionary with standard directory paths:
            - results_dir: Where to save agent outputs (state files, images, etc.)
            - test_results_dir: Where to save test result JSON
            - evaluation_dir: Where evaluation results will be saved
        """
        case_path = Path(case_dir)

        return {
            "results_dir": case_path / "results" / self.eval_mode,
            "test_results_dir": case_path / "test_results" / self.eval_mode,
            "evaluation_dir": case_path / "evaluation_results" / self.eval_mode,
        }

    def save_test_result(
        self,
        result: AgentResult,
        case_dir: str,
        case_name: str,
        task_description: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Save test result in the standard format expected by evaluators.

        Args:
            result: AgentResult from run_task()
            case_dir: Test case directory
            case_name: Test case name
            task_description: Original task description
            output_path: Optional custom output path

        Returns:
            Path to saved result file
        """
        import time
        from datetime import datetime

        dirs = self.get_result_directories(case_dir, case_name)
        dirs["test_results_dir"].mkdir(parents=True, exist_ok=True)

        if output_path is None:
            output_path = dirs["test_results_dir"] / f"test_result_{int(time.time())}.json"

        # Extract token usage info
        token_usage = result.metadata.get("token_usage", {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        })

        # Get token source from _token_info if available
        token_source = "unknown"
        if "_token_info" in result.metadata:
            token_source = result.metadata["_token_info"].get("source", "unknown")
        elif "token_source" in result.metadata:
            token_source = result.metadata["token_source"]

        # Calculate monetary cost
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)
        cost_info = self.calculate_cost(input_tokens, output_tokens)

        # Format in the expected structure for SciVisEvaluator
        test_result = {
            "timestamp": datetime.now().isoformat(),
            "case_name": case_name,
            "status": "completed" if result.success else "failed",
            "duration": result.metadata.get("duration", 0),
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": token_usage.get("total_tokens", input_tokens + output_tokens),
                "source": token_source
            },
            "cost": cost_info,
            "response": result.response,
            "task_description": task_description,
            "error": result.error,
            "output_files": result.output_files,
            "full_result": result.to_dict()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=2, ensure_ascii=False)

        return str(output_path)

    @classmethod
    def from_config_file(cls, config_path: str) -> 'BaseAgent':
        """
        Create agent instance from a configuration file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Initialized agent instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        return cls(config)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.eval_mode})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_name={self.agent_name}, eval_mode={self.eval_mode})"
