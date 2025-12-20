"""
Agent Registry System

Provides decorator-based registration and discovery of agents.
"""

from typing import Dict, Type, Optional, List
from .base_agent import BaseAgent


# Global registry of all registered agents
_AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {}


def register_agent(name: str):
    """
    Decorator to register an agent class.

    This makes the agent available to the evaluation framework.

    Args:
        name: Unique identifier for the agent (e.g., "paraview_mcp", "chatvis")

    Example:
        @register_agent("my_agent")
        class MyAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)
                # Your initialization

            async def run_task(self, task_description, task_config):
                # Your implementation
                pass

    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseAgent]) -> Type[BaseAgent]:
        if not issubclass(cls, BaseAgent):
            raise TypeError(f"{cls.__name__} must inherit from BaseAgent")

        if name in _AGENT_REGISTRY:
            raise ValueError(
                f"Agent '{name}' is already registered by {_AGENT_REGISTRY[name].__name__}"
            )

        _AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def get_agent(name: str) -> Type[BaseAgent]:
    """
    Get a registered agent class by name.

    Args:
        name: Agent identifier

    Returns:
        Agent class

    Raises:
        KeyError: If agent not found
    """
    if name not in _AGENT_REGISTRY:
        available = ", ".join(_AGENT_REGISTRY.keys())
        raise KeyError(
            f"Agent '{name}' not found. Available agents: {available}"
        )

    return _AGENT_REGISTRY[name]


def list_agents() -> List[str]:
    """
    Get list of all registered agent names.

    Returns:
        List of agent identifiers
    """
    return list(_AGENT_REGISTRY.keys())


def is_agent_registered(name: str) -> bool:
    """
    Check if an agent is registered.

    Args:
        name: Agent identifier

    Returns:
        True if registered, False otherwise
    """
    return name in _AGENT_REGISTRY


def get_agent_info(name: str) -> Dict[str, str]:
    """
    Get information about a registered agent.

    Args:
        name: Agent identifier

    Returns:
        Dictionary with agent information
    """
    if name not in _AGENT_REGISTRY:
        raise KeyError(f"Agent '{name}' not found")

    agent_class = _AGENT_REGISTRY[name]

    return {
        "name": name,
        "class_name": agent_class.__name__,
        "module": agent_class.__module__,
        "doc": agent_class.__doc__ or "No description available"
    }


def clear_registry():
    """
    Clear all registered agents.

    This is mainly useful for testing.
    """
    _AGENT_REGISTRY.clear()
