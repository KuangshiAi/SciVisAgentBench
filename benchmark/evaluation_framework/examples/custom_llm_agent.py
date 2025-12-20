"""
Custom LLM Agent Example

This example shows how to create an agent that uses an LLM to generate
visualization code, similar to ChatVis but with custom logic.
"""

import asyncio
import os
from pathlib import Path
from benchmark.evaluation_framework import BaseAgent, AgentResult, register_agent


@register_agent("custom_llm_agent")
class CustomLLMAgent(BaseAgent):
    """
    Example agent that uses an LLM to generate visualization code.

    This demonstrates:
    - Using an LLM API
    - Generating code
    - Saving outputs
    - Token tracking
    """

    def __init__(self, config):
        """
        Initialize the agent with LLM configuration.

        Expected config:
            - provider: "openai", "anthropic", etc.
            - model: Model name (e.g., "gpt-4o")
            - api_key: API key (or use environment variable)
            - eval_mode: "pvpython" or "mcp"
        """
        super().__init__(config)

        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4o")
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")

        # Initialize LLM client
        if self.provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Provider {self.provider} not supported in this example")

        print(f"CustomLLMAgent initialized with {self.provider}/{self.model}")

    async def generate_code(self, task_description):
        """
        Generate visualization code using LLM.

        Args:
            task_description: Task description

        Returns:
            tuple: (generated_code, token_usage)
        """
        system_prompt = """You are a scientific visualization expert.
Generate Python code for ParaView that accomplishes the given task.
The code should:
- Use paraview.simple
- Load data as needed
- Create appropriate visualization
- Save the state to a .pvsm file

Return ONLY the Python code, no explanations."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_description}
            ],
            temperature=0.7
        )

        code = response.choices[0].message.content
        token_usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return code, token_usage

    async def run_task(self, task_description, task_config):
        """
        Generate and save visualization code.

        Args:
            task_description: Natural language task description
            task_config: Task configuration

        Returns:
            AgentResult
        """
        import time
        start_time = time.time()

        try:
            print(f"Generating code for: {task_config['case_name']}")

            # Generate code using LLM
            code, token_usage = await self.generate_code(task_description)

            # Get output directory
            dirs = self.get_result_directories(
                task_config["case_dir"],
                task_config["case_name"]
            )
            dirs["results_dir"].mkdir(parents=True, exist_ok=True)

            # Save generated code
            code_file = dirs["results_dir"] / f"{task_config['case_name']}.py"
            with open(code_file, 'w') as f:
                f.write(code)

            print(f"Code saved to: {code_file}")

            # In a real implementation, you would execute the code here
            # For this example, we just save it

            duration = time.time() - start_time

            return AgentResult(
                success=True,
                response=f"Generated visualization code for {task_config['case_name']}",
                output_files={
                    "script": str(code_file)
                },
                metadata={
                    "duration": duration,
                    "token_usage": token_usage,
                    "model": self.model
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"duration": duration}
            )


# Example usage with the evaluation framework
async def example_usage():
    """Show how to use this agent with the evaluation framework."""

    from benchmark.evaluation_framework import UnifiedTestRunner

    # Create agent
    config = {
        "provider": "openai",
        "model": "gpt-4o",
        "eval_mode": "pvpython",
        "agent_name": "custom_llm_agent"
    }

    agent = CustomLLMAgent(config)

    # Create runner
    runner = UnifiedTestRunner(
        agent=agent,
        yaml_path="path/to/test_cases.yaml",
        cases_dir="path/to/cases",
        eval_model="gpt-4o"
    )

    # Load test cases
    runner.load_yaml_test_cases()

    # Run all test cases
    summary = await runner.run_all_test_cases(run_evaluation=True)

    print(f"\nResults:")
    print(f"  Total cases: {summary['total_cases']}")
    print(f"  Successful: {summary['successful_cases']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(example_usage())
