#!/usr/bin/env python3
"""
Standalone Agent - Copy of agent.py that can work independently

This is a direct copy of the huggingface_hub Agent with fixed imports.
"""

from __future__ import annotations
import sys
import asyncio
import json
import os
from pathlib import Path
from typing import AsyncGenerator, Dict, Iterable, List, Optional, Union

# Add the directory containing mcp_client.py to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Fixed imports - import directly from the modules instead of top-level huggingface_hub
from huggingface_hub.inference._generated.types import (
    ChatCompletionInputMessage, 
    ChatCompletionStreamOutput
)
from mcp_client import MCPClient
from constants import DEFAULT_SYSTEM_PROMPT, EXIT_LOOP_TOOLS, MAX_NUM_TURNS
from huggingface_hub.inference._providers import PROVIDER_OR_POLICY_T


class TinyAgent(MCPClient):
    """
    Standalone Implementation of Agent - copied from huggingface_hub.inference._mcp.Agent
    
    This is a simple while loop built right on top of MCPClient that you can use directly.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        servers: Iterable[Dict],
        provider: Optional[PROVIDER_OR_POLICY_T] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        # # Handle OpenAI special case where we need both base_url and model
        # if provider == "openai" and base_url and model:
        #     # For OpenAI, use base_url in MCPClient but keep model for payload
        #     super().__init__(model=None, provider=provider, base_url=base_url, api_key=api_key)
        #     # Override payload_model to use the specified model
        #     self.payload_model = model
        # else:
        #     super().__init__(model=model, provider=provider, base_url=base_url, api_key=api_key)
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        super().__init__(model=model, provider=provider, base_url=base_url, api_key=api_key)

        self._servers_cfg = list(servers)
        self.messages: List[Union[Dict, ChatCompletionInputMessage]] = [
            {"role": "system", "content": prompt or DEFAULT_SYSTEM_PROMPT}
        ]

    @classmethod
    def from_config_file(cls, config_path: str = "config.json"):
        """
        Create a TinyAgent instance from a configuration file.
        
        Args:
            config_path (str): Path to the configuration JSON file. 
                              Defaults to "config.json" in the current directory.
        
        Returns:
            TinyAgent: Configured agent instance
        """
        # If config_path is relative, look for it in the same directory as this script
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        
        # Extract configuration parameters
        model = config.get("model")
        provider = config.get("provider")
        servers = config.get("servers", [])
        base_url = config.get("base_url")
        api_key = config.get("api_key")
        prompt = config.get("prompt")
        
        return cls(
            model=model,
            provider=provider,
            servers=servers,
            base_url=base_url,
            api_key=api_key,
            prompt=prompt
        )

    async def load_tools(self) -> None:
        for cfg in self._servers_cfg:
            await self.add_mcp_server(cfg["type"], **cfg["config"])
        # Note: Regular TinyAgent does not include interactive tools like ASK_QUESTION_TOOL

    async def run(
        self,
        user_input: str,
        *,
        abort_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Union[ChatCompletionStreamOutput, ChatCompletionInputMessage], None]:
        """
        Run the agent with the given user input.

        Args:
            user_input (`str`):
                The user input to run the agent with.
            abort_event (`asyncio.Event`, *optional*):
                An event that can be used to abort the agent. If the event is set, the agent will stop running.
        """
        self.messages.append({"role": "user", "content": user_input})

        num_turns: int = 0
        next_turn_should_call_tools = True

        while True:
            if abort_event and abort_event.is_set():
                return

            async for item in self.process_single_turn_with_tools(
                self.messages,
                exit_loop_tools=EXIT_LOOP_TOOLS,
                exit_if_first_chunk_no_tool=(num_turns > 0 and next_turn_should_call_tools),
            ):
                yield item

            num_turns += 1
            last = self.messages[-1]

            if (last.get("role") == "tool" and 
                last.get("name") in {t.function.name for t in EXIT_LOOP_TOOLS}):
                return

            if last.get("role") != "tool" and num_turns > MAX_NUM_TURNS:
                return

            if last.get("role") != "tool" and next_turn_should_call_tools:
                return

            next_turn_should_call_tools = last.get("role") != "tool"

    @classmethod
    def from_config(cls, config_path: str) -> TinyAgent:
        """
        Create an instance of TinyAgent from a configuration file.

        Args:
            config_path (`str`):
                The path to the configuration file.

        Returns:
            TinyAgent: An instance of TinyAgent configured with the settings from the file.
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        model = config.get("model")
        provider = config.get("provider")
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        prompt = config.get("prompt", DEFAULT_SYSTEM_PROMPT)

        servers = config.get("servers", [])
        if isinstance(servers, dict):
            # Single server case - convert to list
            servers = [servers]

        return cls(
            model=model,
            servers=servers,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            prompt=prompt,
        )


# Example usage with default configuration (same as original tiny-agent)
async def demo_with_default_config():
    """Demo using the same default configuration as the original tiny-agent."""
    import sys
    from pathlib import Path
    
    # Default configuration from constants.py - same as tiny-agent CLI
    agent = TinyAgent(
        model="Qwen/Qwen2.5-72B-Instruct",
        provider="nebius",
        servers=[
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(Path.home() / ("Desktop" if sys.platform == "darwin" else "")),
                    ],
                },
            },
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                },
            },
        ]
        # Uses DEFAULT_SYSTEM_PROMPT from constants.py
    )
    
    try:
        async with agent:
            await agent.load_tools()
            print(f"Agent loaded with {len(agent.available_tools)} tools")
            
            # Ask a question
            query = "Hello, please introduce yourself and tell me what you can do."
            print(f"\nQuery: {query}")
            print("Response:")
            print("-" * 50)
            
            async for chunk in agent.run(query):
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        print(delta.content, end="", flush=True)
                elif hasattr(chunk, 'role') and chunk.role == "tool":
                    print(f"\n[Tool: {chunk.name}] {chunk.content}")
            print("\n" + "=" * 50)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def run_single_query(query: str):
    """Run a single query with the default agent configuration."""
    import sys
    from pathlib import Path
    
    agent = TinyAgent(
        model="Qwen/Qwen2.5-72B-Instruct",
        provider="nebius",
        servers=[
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(Path.home() / ("Desktop" if sys.platform == "darwin" else "")),
                    ],
                },
            },
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                },
            },
        ]
    )
    
    async with agent:
        await agent.load_tools()
        print(f"Query: {query}")
        print("Response:")
        print("-" * 50)
        
        async for chunk in agent.run(query):
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    print(delta.content, end="", flush=True)
            elif hasattr(chunk, 'role') and chunk.role == "tool":
                print(f"\n[Tool: {chunk.name}] {chunk.content}")
        print("\n" + "=" * 50)


async def interactive_mode():
    """Interactive chat mode - similar to tiny-agents run."""
    import sys
    from pathlib import Path
    
    agent = TinyAgent(
        model="Qwen/Qwen2.5-72B-Instruct",
        provider="nebius",
        servers=[
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(Path.home() / ("Desktop" if sys.platform == "darwin" else "")),
                    ],
                },
            },
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                },
            },
        ]
    )
    
    try:
        async with agent:
            await agent.load_tools()
            print(f"Tiny Agent ready with {len(agent.available_tools)} tools")
            print("Type 'quit' or 'exit' to end the conversation")
            print("=" * 50)
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break
                    
                    if not user_input:
                        continue
                    
                    print("\nAgent: ", end="", flush=True)
                    
                    async for chunk in agent.run(user_input):
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                print(delta.content, end="", flush=True)
                        elif hasattr(chunk, 'role') and chunk.role == "tool":
                            print(f"\n[Tool: {chunk.name}] {chunk.content}")
                    
                    print()  # New line after response
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
    
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()


async def run_with_config(config_path: str = "config.json"):
    """Run the agent using configuration from a JSON file."""
    try:
        agent = TinyAgent.from_config_file(config_path)
        
        async with agent:
            await agent.load_tools()
            print(f"Agent loaded with {len(agent.available_tools)} tools from config: {config_path}")
            print("Type 'quit' or 'exit' to end the conversation")
            print("=" * 50)
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break
                    
                    if not user_input:
                        continue
                    
                    print("\nAgent: ", end="", flush=True)
                    
                    async for chunk in agent.run(user_input):
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                print(delta.content, end="", flush=True)
                        elif hasattr(chunk, 'role') and chunk.role == "tool":
                            print(f"\n[Tool: {chunk.name}] {chunk.content}")
                    
                    print()  # New line after response
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
    
    except Exception as e:
        print(f"Failed to initialize agent from config: {e}")
        import traceback
        traceback.print_exc()


async def run_single_query_with_config(query: str, config_path: str = "config.json"):
    """Run a single query with configuration from a JSON file."""
    try:
        agent = TinyAgent.from_config_file(config_path)
        
        async with agent:
            await agent.load_tools()
            print(f"Query: {query}")
            print("Response:")
            print("-" * 50)
            
            async for chunk in agent.run(query):
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        print(delta.content, end="", flush=True)
                elif hasattr(chunk, 'role') and chunk.role == "tool":
                    print(f"\n[Tool: {chunk.name}] {chunk.content}")
            print("\n" + "=" * 50)
    
    except Exception as e:
        print(f"Failed to run query with config: {e}")
        import traceback
        traceback.print_exc()


# Example usage with default configuration (same as original tiny-agent)
async def demo_with_default_config():
    """Demo using the same default configuration as the original tiny-agent."""
    import sys
    from pathlib import Path
    
    # Default configuration from constants.py - same as tiny-agent CLI
    agent = TinyAgent(
        model="Qwen/Qwen2.5-72B-Instruct",
        provider="nebius",
        servers=[
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(Path.home() / ("Desktop" if sys.platform == "darwin" else "")),
                    ],
                },
            },
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                },
            },
        ]
        # Uses DEFAULT_SYSTEM_PROMPT from constants.py
    )
    
    try:
        async with agent:
            await agent.load_tools()
            print(f"Agent loaded with {len(agent.available_tools)} tools")
            
            # Ask a question
            query = "Hello, please introduce yourself and tell me what you can do."
            print(f"\nQuery: {query}")
            print("Response:")
            print("-" * 50)
            
            async for chunk in agent.run(query):
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        print(delta.content, end="", flush=True)
                elif hasattr(chunk, 'role') and chunk.role == "tool":
                    print(f"\n[Tool: {chunk.name}] {chunk.content}")
            print("\n" + "=" * 50)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def run_single_query(query: str):
    """Run a single query with the default agent configuration."""
    import sys
    from pathlib import Path
    
    agent = TinyAgent(
        model="Qwen/Qwen2.5-72B-Instruct",
        provider="nebius",
        servers=[
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(Path.home() / ("Desktop" if sys.platform == "darwin" else "")),
                    ],
                },
            },
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                },
            },
        ]
    )
    
    async with agent:
        await agent.load_tools()
        print(f"Query: {query}")
        print("Response:")
        print("-" * 50)
        
        async for chunk in agent.run(query):
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    print(delta.content, end="", flush=True)
            elif hasattr(chunk, 'role') and chunk.role == "tool":
                print(f"\n[Tool: {chunk.name}] {chunk.content}")
        print("\n" + "=" * 50)


async def interactive_mode():
    """Interactive chat mode - similar to tiny-agents run."""
    import sys
    from pathlib import Path
    
    agent = TinyAgent(
        model="Qwen/Qwen2.5-72B-Instruct",
        provider="nebius",
        servers=[
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(Path.home() / ("Desktop" if sys.platform == "darwin" else "")),
                    ],
                },
            },
            {
                "type": "stdio",
                "config": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                },
            },
        ]
    )
    
    try:
        async with agent:
            await agent.load_tools()
            print(f"Tiny Agent ready with {len(agent.available_tools)} tools")
            print("Type 'quit' or 'exit' to end the conversation")
            print("=" * 50)
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break
                    
                    if not user_input:
                        continue
                    
                    print("\nAgent: ", end="", flush=True)
                    
                    async for chunk in agent.run(user_input):
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                print(delta.content, end="", flush=True)
                        elif hasattr(chunk, 'role') and chunk.role == "tool":
                            print(f"\n[Tool: {chunk.name}] {chunk.content}")
                    
                    print()  # New line after response
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
    
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()


async def run_with_config(config_path: str = "config.json"):
    """Run the agent using configuration from a JSON file."""
    try:
        agent = TinyAgent.from_config_file(config_path)
        
        async with agent:
            await agent.load_tools()
            print(f"Agent loaded with {len(agent.available_tools)} tools from config: {config_path}")
            print("Type 'quit' or 'exit' to end the conversation")
            print("=" * 50)
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break
                    
                    if not user_input:
                        continue
                    
                    print("\nAgent: ", end="", flush=True)
                    
                    async for chunk in agent.run(user_input):
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                print(delta.content, end="", flush=True)
                        elif hasattr(chunk, 'role') and chunk.role == "tool":
                            print(f"\n[Tool: {chunk.name}] {chunk.content}")
                    
                    print()  # New line after response
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
    
    except Exception as e:
        print(f"Failed to initialize agent from config: {e}")
        import traceback
        traceback.print_exc()


async def run_single_query_with_config(query: str, config_path: str = "config.json"):
    """Run a single query with configuration from a JSON file."""
    try:
        agent = TinyAgent.from_config_file(config_path)
        
        async with agent:
            await agent.load_tools()
            print(f"Query: {query}")
            print("Response:")
            print("-" * 50)
            
            async for chunk in agent.run(query):
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        print(delta.content, end="", flush=True)
                elif hasattr(chunk, 'role') and chunk.role == "tool":
                    print(f"\n[Tool: {chunk.name}] {chunk.content}")
            print("\n" + "=" * 50)
    
    except Exception as e:
        print(f"Failed to run query with config: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    print("Standalone Agent - Copy of original agent.py")
    print("This loads configuration from config.json by default")
    print()
    
    # Check if config.json exists in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_openai.json")
    
    if os.path.exists(config_path):
        print(f"Using configuration from: {config_path}")
        if len(sys.argv) > 1:
            # Single query mode with config
            query = " ".join(sys.argv[1:])
            asyncio.run(run_single_query_with_config(query, config_path))
        else:
            # Interactive mode with config
            asyncio.run(run_with_config(config_path))
    else:
        print(f"Config file not found at {config_path}, using default configuration")
        if len(sys.argv) > 1:
            # Single query mode with defaults
            query = " ".join(sys.argv[1:])
            asyncio.run(run_single_query(query))
        else:
            # Interactive mode with defaults
            asyncio.run(interactive_mode())
