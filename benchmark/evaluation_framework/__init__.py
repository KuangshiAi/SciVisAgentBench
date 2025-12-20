"""
SciVisAgentBench Evaluation Framework

A high-level, plugin-based framework for evaluating scientific visualization agents.

This framework provides:
- Simple agent interface for integrating new agents
- Automatic evaluation with multiple metrics (LLM judge, image metrics, efficiency)
- Support for multiple benchmark subsets
- Consistent result formatting and reporting

Example usage:
    from benchmark.evaluation_framework import BaseAgent, register_agent

    @register_agent("my_agent")
    class MyAgent(BaseAgent):
        def run_task(self, prompt, task_config):
            # Your agent implementation
            return result

Then run evaluation:
    python -m benchmark.evaluation_framework.run_evaluation \\
        --agent my_agent \\
        --benchmarks main,bioimage_data \\
        --config my_config.json
"""

from .base_agent import BaseAgent
from .agent_registry import register_agent, get_agent, list_agents
from .evaluation_manager import EvaluationManager
from .unified_runner import UnifiedTestRunner

__all__ = [
    'BaseAgent',
    'register_agent',
    'get_agent',
    'list_agents',
    'EvaluationManager',
    'UnifiedTestRunner',
]

__version__ = '1.0.0'
