"""
ChatVis Agent Adapter

Wraps the existing ChatVis implementation to work with the evaluation framework.
"""

import os
import sys
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "chatvis"))

from evaluation_framework.base_agent import BaseAgent, AgentResult
from evaluation_framework.agent_registry import register_agent
from chatvis.multi_provider_client import MultiProviderClient


@register_agent("chatvis")
class ChatVisAgent(BaseAgent):
    """
    ChatVis agent for ParaView visualization using pvpython.

    This agent generates ParaView Python scripts using an LLM and executes
    them with pvpython.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ChatVis agent.

        Args:
            config: Configuration dictionary containing:
                   - model: LLM model to use
                   - provider: LLM provider ("openai", "anthropic", "hf")
                   - price: Pricing information
        """
        config["eval_mode"] = "pvpython"
        config["agent_name"] = "chatvis"
        super().__init__(config)

        # Initialize multi-provider client
        provider = config.get("provider", "openai")
        model = config.get("model", "gpt-4o")

        if provider == "openai":
            self.client = MultiProviderClient(
                provider="openai",
                model=model,
                api_key=os.getenv('OPENAI_API_KEY')
            )
        elif provider == "anthropic":
            self.client = MultiProviderClient(
                provider="anthropic",
                model=model,
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )
        else:
            self.client = MultiProviderClient.from_config_file(str(config.get("config_path", "")))

        self.pricing_info = config.get("price", {
            "input_per_1m_tokens": "$2.50",
            "output_per_1m_tokens": "$10.00"
        })

        # Initialize token counter
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is None:
            return len(text.split())
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text.split())

    def execute_paraview_code(self, code: str) -> str:
        """Execute ParaView Python code directly using pvpython."""
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_script = f.name

            # Try to find pvpython in common locations
            pvpython_paths = [
                '/Applications/ParaView-5.13.3.app/Contents/bin/pvpython',  # macOS
                '/usr/local/bin/pvpython',  # Linux/macOS homebrew
                'pvpython',  # In PATH
            ]

            pvpython_cmd = None
            for path in pvpython_paths:
                try:
                    result = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        pvpython_cmd = path
                        break
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue

            if not pvpython_cmd:
                return "Error: pvpython not found. Please install ParaView or add pvpython to your PATH."

            # Execute the script
            result = subprocess.run([pvpython_cmd, temp_script],
                                    capture_output=True, text=True, timeout=120)

            # Clean up temp file
            Path(temp_script).unlink()

            if result.returncode == 0:
                return result.stdout if result.stdout else "Execution completed successfully"
            else:
                return f"Error executing ParaView script:\n{result.stderr}"

        except Exception as e:
            return f"Exception during ParaView execution: {str(e)}"

    def extract_python(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""
        m = re.search(r"```python(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    def extract_errors(self, output: str) -> list:
        """Extract error messages from output."""
        return [ln for ln in output.splitlines() if "Error" in ln or "Traceback" in ln]

    def save_script(self, code: str, case_name: str, case_dir: Path):
        """Save generated script to results directory."""
        generated_dir = case_dir / "results" / "pvpython"
        generated_dir.mkdir(parents=True, exist_ok=True)
        fn = generated_dir / f"{case_name}.py"
        fn.write_text(code, encoding="utf-8")
        print(f"    • script saved → {fn.name}")

    async def setup(self):
        """Setup the agent."""
        print(f"Setting up ChatVis agent with model: {self.config.get('model', 'unknown')}")

    async def teardown(self):
        """Cleanup the agent."""
        print("Tearing down ChatVis agent")

    async def run_task(self, task_description: str, task_config: Dict[str, Any]) -> AgentResult:
        """
        Run a single visualization task using ChatVis.

        Args:
            task_description: Natural language description of the task
            task_config: Task configuration

        Returns:
            AgentResult with success status, response, output files, and metadata
        """
        start_time = time.time()
        case_name = task_config["case_name"]
        case_dir = Path(task_config["case_dir"])

        # Count input tokens
        input_tokens = self.count_tokens(task_description)
        output_tokens = 0

        try:
            print(f"Generating ParaView script for case: {case_name}")

            # Generate script using multi-provider client
            sr = self.client.create_completion(
                messages=[
                    {"role": "system", "content": (
                        "You are a code assistant. Output only a self-contained Python script for ParaView:\n"
                        "- Call ResetSession() once\n"
                        "- Import from paraview.simple only needed symbols\n"
                        "- Read raw via ImageReader(...) and configure DataScalarType, DataByteOrder, DataExtent, DataSpacing\n"
                        "- UpdatePipeline(), Show(...), Representation='Volume'\n"
                        "- Configure GetColorTransferFunction and GetOpacityTransferFunction\n"
                        "- SaveState(state_path) at end"
                    )},
                    {"role": "user", "content": task_description}
                ]
            )

            # Get response and token usage
            response_content = self.client.get_response_content(sr)
            token_usage = self.client.get_token_usage(sr)
            output_tokens += token_usage["output_tokens"]

            code = self.extract_python(response_content)
            code = "from paraview.simple import ResetSession\nResetSession()\n" + code

            # Syntax check
            try:
                compile(code, '<string>', 'exec')
                print("    ✓ syntax check passed")
            except SyntaxError as e:
                print(f"    ✘ syntax error: {e}")
                duration = time.time() - start_time
                return AgentResult(
                    success=False,
                    error=f"Syntax error in generated code: {e}",
                    metadata={"duration": duration}
                )

            # Save initial script
            self.save_script(code, case_name, case_dir)

            # Execute and iterative fix
            max_attempts = 5
            out = self.execute_paraview_code(code)
            errors = self.extract_errors(out)
            attempt = 1

            while errors and attempt < max_attempts:
                print(f"    ⚠ attempt {attempt}: fixing {len(errors)} errors...")
                error_msg = "\\n".join(errors[:3])  # Limit to first 3 errors

                fix_sr = self.client.create_completion(
                    messages=[
                        {"role": "system", "content": "Fix the ParaView Python script based on the error message. Return only the corrected complete script."},
                        {"role": "user", "content": f"Original script:\n```python\n{code}\n```\n\nError:\n{error_msg}\n\nProvide the fixed script:"}
                    ]
                )

                # Count additional tokens
                fix_response = self.client.get_response_content(fix_sr)
                fix_token_usage = self.client.get_token_usage(fix_sr)
                output_tokens += fix_token_usage["output_tokens"]

                code = self.extract_python(fix_response)
                code = "from paraview.simple import ResetSession\nResetSession()\n" + code

                # Try execution again
                out = self.execute_paraview_code(code)
                errors = self.extract_errors(out)
                attempt += 1

            duration = time.time() - start_time

            if errors:
                print(f"    ✘ still has errors after {max_attempts} attempts")
                return AgentResult(
                    success=False,
                    error=f"Script execution failed after {max_attempts} attempts. Last errors: {errors[:3]}",
                    metadata={
                        "duration": duration,
                        "token_usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens
                        }
                    }
                )
            else:
                print("    ✓ script executed successfully")

                # Save final script
                self.save_script(code, case_name, case_dir)

                # Verify state file
                state_dir = case_dir / "results" / "pvpython"
                pvsm = state_dir / f"{case_name}.pvsm"

                if pvsm.exists():
                    print(f"    ✓ state file created: {pvsm.name}")

                return AgentResult(
                    success=True,
                    response=response_content,
                    output_files={"state": str(pvsm), "script": str(state_dir / f"{case_name}.py")},
                    metadata={
                        "duration": duration,
                        "token_usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens
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
