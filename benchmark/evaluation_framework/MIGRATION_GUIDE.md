# Migration Guide: Adapting Existing Agents to the Framework

This guide shows how to adapt an existing agent to work with the evaluation framework.

## Quick Comparison

### Before: Manual Integration

```python
# Your agent code (agent.py)
class MyVisualizationAgent:
    def __init__(self, config):
        self.config = config

    def process_task(self, prompt):
        # Your agent logic
        return result

# Separate test runner (my_runner.py - 500+ lines)
class MyTestRunner:
    def __init__(self, config, yaml_path, cases_dir):
        # Load YAML
        # Setup directories
        # Initialize agent
        # ... lots of boilerplate ...

    def run_test_case(self, case):
        # Run agent
        # Save results
        # Call evaluator
        # ... more boilerplate ...

    def run_all_cases(self):
        # Load test cases
        # For each case...
        # ... even more code ...

# Run it
runner = MyTestRunner(config, yaml_path, cases_dir)
runner.run_all_cases()
```

### After: Using Framework

```python
from benchmark.evaluation_framework import BaseAgent, AgentResult, register_agent

@register_agent("my_agent")
class MyAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.my_agent = MyVisualizationAgent(config)

    async def run_task(self, task_description, task_config):
        result = self.my_agent.process_task(task_description)

        return AgentResult(
            success=True,
            response=result.text,
            output_files={"state": result.state_file},
            metadata={"duration": result.time}
        )

# Run it with one command
# python -m benchmark.evaluation_framework.run_evaluation --agent my_agent ...
```

**Lines of code**: Reduced from 500+ to ~20

## Step-by-Step Migration

### Step 1: Wrap Your Agent

Create a new file `my_agent_adapter.py`:

```python
from benchmark.evaluation_framework import BaseAgent, AgentResult, register_agent
from pathlib import Path

# Import your existing agent
from my_existing_code import MyVisualizationAgent

@register_agent("my_agent")
class MyAgentAdapter(BaseAgent):
    """Adapter for MyVisualizationAgent to work with the framework."""

    def __init__(self, config):
        # Call BaseAgent.__init__ with eval_mode
        config["eval_mode"] = "mcp"  # or "pvpython"
        super().__init__(config)

        # Initialize your existing agent
        self.agent = MyVisualizationAgent(config)

    async def run_task(self, task_description, task_config):
        """
        Adapt run_task to call your existing agent.

        Args:
            task_description: The task prompt
            task_config: Contains case_name, case_dir, data_dir, etc.

        Returns:
            AgentResult
        """
        try:
            # Call your existing agent's method
            # (adjust method name and parameters as needed)
            result = self.agent.process_task(
                prompt=task_description,
                working_dir=task_config["data_dir"]
            )

            # Get output directory using helper
            dirs = self.get_result_directories(
                task_config["case_dir"],
                task_config["case_name"]
            )

            # Save outputs to expected location
            # (adjust based on what your agent produces)
            output_file = dirs["results_dir"] / f"{task_config['case_name']}.pvsm"

            # If your agent doesn't save to the right place, move/copy it
            if hasattr(result, 'output_file'):
                import shutil
                shutil.copy(result.output_file, output_file)

            # Return AgentResult
            return AgentResult(
                success=True,
                response=result.text if hasattr(result, 'text') else str(result),
                output_files={"state": str(output_file)},
                metadata={
                    "duration": result.execution_time if hasattr(result, 'execution_time') else 0,
                    "token_usage": getattr(result, 'token_usage', {})
                }
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))
```

### Step 2: Create Configuration

Create `my_agent_config.json`:

```json
{
  "eval_mode": "mcp",
  "agent_name": "my_agent",
  "model": "gpt-4o",
  "provider": "openai",

  # Add any config your existing agent needs
  "server_url": "http://localhost:8000",
  "timeout": 120
}
```

### Step 3: Run Evaluation

```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent my_agent \
    --config my_agent_config.json \
    --yaml SciVisAgentBench-tasks/main/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main
```

## Common Migration Patterns

### Pattern 1: Your Agent Has Setup/Teardown

```python
@register_agent("my_agent")
class MyAgent(BaseAgent):
    async def setup(self):
        """Called once before any tasks."""
        self.server = start_my_server()

    async def teardown(self):
        """Called once after all tasks."""
        self.server.stop()

    async def run_task(self, task_description, task_config):
        # Use self.server here
        ...
```

### Pattern 2: Your Agent Needs Per-Task Cleanup

```python
@register_agent("my_agent")
class MyAgent(BaseAgent):
    async def prepare_task(self, task_config):
        """Called before each task."""
        self.agent.clear_state()

    async def cleanup_task(self, task_config):
        """Called after each task."""
        self.agent.cleanup()

    async def run_task(self, task_description, task_config):
        ...
```

### Pattern 3: Your Agent Uses Different File Types

```python
@register_agent("my_agent")
class MyAgent(BaseAgent):
    async def run_task(self, task_description, task_config):
        result = self.agent.process(task_description)

        # Save multiple output files
        dirs = self.get_result_directories(
            task_config["case_dir"],
            task_config["case_name"]
        )

        state_file = dirs["results_dir"] / f"{task_config['case_name']}.pvsm"
        image_file = dirs["results_dir"] / f"{task_config['case_name']}.png"
        script_file = dirs["results_dir"] / f"{task_config['case_name']}.py"

        # Save them
        result.save_state(state_file)
        result.save_image(image_file)
        result.save_script(script_file)

        return AgentResult(
            success=True,
            output_files={
                "state": str(state_file),
                "image": str(image_file),
                "script": str(script_file)
            }
        )
```

### Pattern 4: Your Agent Tracks Custom Metrics

```python
@register_agent("my_agent")
class MyAgent(BaseAgent):
    async def run_task(self, task_description, task_config):
        result = self.agent.process(task_description)

        return AgentResult(
            success=True,
            response=result.text,
            metadata={
                "duration": result.execution_time,
                "token_usage": result.tokens,
                # Add custom metrics
                "custom_metric_1": result.quality_score,
                "custom_metric_2": result.complexity,
                "tool_calls": len(result.tool_calls)
            }
        )
```

## Real Example: Adapting an MCP Agent

Let's say you have an existing MCP agent:

**Before** (`my_mcp_agent.py`):

```python
class MyMCPAgent:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = json.load(f)
        self.tiny_agent = TinyAgent.from_config_file(config_file)

    async def run_visualization(self, prompt, output_dir):
        async with self.tiny_agent:
            await self.tiny_agent.load_tools()

            response = []
            async for chunk in self.tiny_agent.run(prompt):
                # Process chunks
                response.append(chunk)

        return "".join(response)
```

**After** (using framework):

```python
from benchmark.evaluation_framework import BaseAgent, AgentResult, register_agent
from tiny_agent.agent import TinyAgent

@register_agent("my_mcp")
class MyMCPAdapter(BaseAgent):
    def __init__(self, config):
        config["eval_mode"] = "mcp"
        super().__init__(config)

    async def run_task(self, task_description, task_config):
        import time
        import tempfile
        import json

        start_time = time.time()

        try:
            # Create temporary config for this task
            temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(self.config, temp_config)
            temp_config.close()

            # Create TinyAgent
            agent = TinyAgent.from_config_file(temp_config.name)

            response_parts = []

            async with agent:
                await agent.load_tools()

                async for chunk in agent.run(task_description):
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            response_parts.append(delta.content)

            response = "".join(response_parts)
            duration = time.time() - start_time

            # Get output paths
            dirs = self.get_result_directories(
                task_config["case_dir"],
                task_config["case_name"]
            )
            state_file = dirs["results_dir"] / f"{task_config['case_name']}.pvsm"

            return AgentResult(
                success=True,
                response=response,
                output_files={"state": str(state_file)},
                metadata={"duration": duration}
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))

        finally:
            os.unlink(temp_config.name)
```

Now run:

```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent my_mcp \
    --config my_config.json \
    --yaml cases.yaml \
    --cases cases_dir
```

## Handling Edge Cases

### Your Agent Is Synchronous (Not Async)

If your agent doesn't use async:

```python
@register_agent("my_sync_agent")
class MySyncAgent(BaseAgent):
    async def run_task(self, task_description, task_config):
        # Wrap synchronous code
        result = await asyncio.to_thread(
            self.my_sync_agent.process,
            task_description
        )

        return AgentResult(success=True, response=result)
```

### Your Agent Uses Different Input Format

```python
@register_agent("my_agent")
class MyAgent(BaseAgent):
    async def run_task(self, task_description, task_config):
        # Convert framework format to your agent's format
        my_format = {
            "prompt": task_description,
            "case": task_config["case_name"],
            "data_path": task_config["data_dir"]
        }

        result = await self.agent.run(my_format)

        return AgentResult(success=True, response=result.output)
```

### Your Agent Produces Different Outputs

```python
@register_agent("my_agent")
class MyAgent(BaseAgent):
    def __init__(self, config):
        # Override eval_mode if needed
        config["eval_mode"] = "custom"  # Your custom mode
        super().__init__(config)

    async def run_task(self, task_description, task_config):
        result = self.agent.process(task_description)

        # Save to your custom location
        output_dir = Path(task_config["case_dir"]) / "results" / "custom"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save whatever your agent produces
        output_file = output_dir / f"{task_config['case_name']}.custom"

        return AgentResult(
            success=True,
            output_files={"custom": str(output_file)}
        )
```

## Testing Your Migration

1. **Test single case first:**

```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent my_agent \
    --config config.json \
    --yaml cases.yaml \
    --cases cases_dir \
    --case simple_test_case
```

2. **Check output files:**

```bash
ls cases_dir/simple_test_case/results/mcp/
# Should see your output files
```

3. **Run without evaluation:**

```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent my_agent \
    --config config.json \
    --yaml cases.yaml \
    --cases cases_dir \
    --no-eval
```

4. **Run full evaluation on small subset:**

```bash
# Create a small YAML with 2-3 test cases
python -m benchmark.evaluation_framework.run_evaluation \
    --agent my_agent \
    --config config.json \
    --yaml small_test.yaml \
    --cases cases_dir
```

## Getting Help

If you run into issues:

1. Look at [examples/](examples/) for working examples
2. Review existing adapters in [agents/](agents/)
3. Compare with your old test runner to see what's missing

## Summary

Migration steps:

1. ✅ Create adapter class extending `BaseAgent`
2. ✅ Implement `run_task()` to call your existing agent
3. ✅ Return `AgentResult` with success, response, files, metadata
4. ✅ Register with `@register_agent("name")`
5. ✅ Create config file
6. ✅ Run with CLI tool

Benefits:

- **90% less code** to maintain
- **Automatic evaluation** with all existing features
- **Standardized interface** across all agents
- **Easy to run** with single command
- **Better organization** with plugin system
