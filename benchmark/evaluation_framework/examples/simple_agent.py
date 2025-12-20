"""
Simple Agent Example

This example shows how to create a minimal agent that works with
the SciVisAgentBench evaluation framework.
"""

import asyncio
from pathlib import Path
from benchmark.evaluation_framework import BaseAgent, AgentResult, register_agent


@register_agent("simple_agent")
class SimpleAgent(BaseAgent):
    """
    A simple example agent that demonstrates the minimal implementation needed.

    This agent just echoes the task description and creates a dummy output file.
    In a real agent, you would replace this with your actual visualization logic.
    """

    def __init__(self, config):
        """
        Initialize the agent.

        Args:
            config: Configuration dictionary. At minimum should include:
                   - eval_mode: "mcp" or "pvpython" or custom
        """
        super().__init__(config)
        print(f"SimpleAgent initialized with config: {config}")

    async def run_task(self, task_description, task_config):
        """
        Run a single visualization task.

        Args:
            task_description: Natural language description of the task
            task_config: Task configuration with case_name, case_dir, etc.

        Returns:
            AgentResult
        """
        print(f"\n{'='*60}")
        print(f"Running task: {task_config['case_name']}")
        print(f"Description: {task_description[:100]}...")
        print(f"{'='*60}\n")

        try:
            # Get output directory
            dirs = self.get_result_directories(
                task_config["case_dir"],
                task_config["case_name"]
            )

            # Create output directory
            dirs["results_dir"].mkdir(parents=True, exist_ok=True)

            # Example: Create a dummy state file
            # In a real agent, you would generate actual visualization outputs here
            output_file = dirs["results_dir"] / f"{task_config['case_name']}.pvsm"

            with open(output_file, 'w') as f:
                f.write(f"# Simple agent output for {task_config['case_name']}\n")
                f.write(f"# Task: {task_description[:100]}\n")

            # Return success result
            return AgentResult(
                success=True,
                response=f"Successfully processed task: {task_config['case_name']}",
                output_files={"state": str(output_file)},
                metadata={
                    "duration": 1.0,
                    "token_usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150
                    }
                }
            )

        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e)
            )


# Example usage
async def main():
    """Example of how to use the SimpleAgent."""

    # Create agent with minimal config
    config = {
        "eval_mode": "mcp",
        "agent_name": "simple_agent"
    }

    agent = SimpleAgent(config)

    # Example task
    task_description = "Load the aneurism dataset and create a volume rendering."

    task_config = {
        "case_name": "aneurism",
        "case_dir": "/path/to/test/case",
        "data_dir": "/path/to/data",
        "working_dir": "/path/to/data"
    }

    # Run the task
    result = await agent.run_task(task_description, task_config)

    # Check result
    if result.success:
        print(f"\n✅ Task completed successfully!")
        print(f"Response: {result.response}")
        print(f"Output files: {result.output_files}")
    else:
        print(f"\n❌ Task failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
