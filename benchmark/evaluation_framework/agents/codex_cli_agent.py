"""
Codex CLI Agent

This agent integrates OpenAI Codex CLI as a general-purpose coding agent
for the SciVisAgentBench evaluation framework.

Codex CLI receives natural language task descriptions and figures out
how to use visualization tools (ParaView, Napari, GMX-VMD, etc.) without
specialized MCP servers or pre-built tools.
"""

import asyncio
import json
import subprocess
import tempfile
import time
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation_framework.base_agent import BaseAgent, AgentResult
from evaluation_framework.agent_registry import register_agent


@register_agent("codex_cli")
class CodexCLIAgent(BaseAgent):
    """
    Codex CLI agent that uses the OpenAI Codex CLI to complete visualization tasks.

    This agent is tool-agnostic - it receives task descriptions and figures out
    how to interact with visualization packages (paraview.simple, napari, etc.)
    that are available in the Python environment.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Codex CLI agent.

        Args:
            config: Configuration dictionary with keys:
                - model: OpenAI model to use (e.g., "gpt-4o-commercial")
                - codex_cli_path: Path to Codex CLI (default: "codex")
                - timeout_per_task: Timeout in seconds (default: 600)
                - preserve_workdir: Keep working directories for debugging
                - environment: Environment specification (optional)
                - price: Pricing information for cost calculation
                - custom_system_prompt: Optional additional instructions to prepend to all tasks
        """
        # Set defaults for BaseAgent
        config["eval_mode"] = config.get("eval_mode", "generic")
        config["agent_name"] = config.get("agent_name", "codex_cli")

        super().__init__(config)

        self.codex_path = config.get("codex_cli_path", "codex")
        self.timeout = config.get("timeout_per_task", 600)
        self.preserve_workdir = config.get("preserve_workdir", False)
        self.auto_approve = config.get("auto_approve", True)  # Default to auto-approve for benchmarking
        self.custom_system_prompt = config.get("custom_system_prompt", "")
        self.verbose = config.get("verbose", False)  # Enable real-time output streaming

        print(f"CodexCLIAgent initialized:")
        print(f"  - Model: {config.get('model', 'default')}")
        print(f"  - Agent mode: {self.agent_mode}")
        print(f"  - Timeout: {self.timeout}s")
        print(f"  - Codex path: {self.codex_path}")
        print(f"  - Auto-approve: {self.auto_approve}")
        print(f"  - Verbose: {self.verbose}")
        if self.custom_system_prompt:
            print(f"  - Custom prompt: {len(self.custom_system_prompt)} chars")

    async def setup(self):
        """Verify Codex CLI is available."""
        try:
            result = subprocess.run(
                [self.codex_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✓ Codex CLI found: {result.stdout.strip()}")
            else:
                print(f"⚠ Codex CLI check returned non-zero: {result.returncode}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Codex CLI not found at '{self.codex_path}'. "
                "Please install it or set 'codex_cli_path' in config."
            )
        except Exception as e:
            print(f"⚠ Warning: Could not verify Codex CLI: {e}")

    async def run_task(
        self,
        task_description: str,
        task_config: Dict[str, Any]
    ) -> AgentResult:
        """
        Execute task using Codex CLI.

        Args:
            task_description: Natural language task description
            task_config: Task configuration with case_name, case_dir, etc.

        Returns:
            AgentResult with success status, response, output files, and metadata
        """
        case_name = task_config["case_name"]
        case_dir = Path(task_config["case_dir"])
        working_dir = Path(task_config.get("working_dir", case_dir))

        print(f"\n{'='*60}")
        print(f"Running task: {case_name}")
        print(f"Working directory: {working_dir}")
        print(f"{'='*60}\n")

        try:
            # Prepare task prompt with environment context
            prompt = self._prepare_task_prompt(task_description, task_config)

            # Get result directory
            dirs = self.get_result_directories(str(case_dir), case_name)
            dirs["results_dir"].mkdir(parents=True, exist_ok=True)

            # Invoke Codex CLI
            success, output, duration = await self._invoke_codex_cli(
                prompt=prompt,
                working_dir=working_dir,
                timeout=self.timeout
            )

            # Find output files generated by Codex CLI
            output_files = self._find_output_files(dirs["results_dir"], case_name)

            # Extract token usage from output if available
            token_usage = self._extract_token_usage(output)

            # If token usage is still estimated, provide a better estimate based on context
            if token_usage["source"] in ["estimated", "unknown"]:
                try:
                    token_usage = self._estimate_token_usage_from_context(
                        prompt, output_files, dirs["results_dir"], case_name
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not estimate token usage from context: {e}")
                    import traceback
                    traceback.print_exc()
                    # Keep the simple estimate if context estimation fails
                    pass

            # Prepare response
            response = output if success else f"Task failed: {output}"

            # Create result
            return AgentResult(
                success=success,
                response=response,
                error=None if success else output,
                output_files=output_files,
                metadata={
                    "duration": duration,
                    "token_usage": token_usage,
                    "working_dir": str(working_dir),
                    "codex_path": self.codex_path
                }
            )

        except Exception as e:
            import traceback
            error_msg = f"Exception during task execution: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return AgentResult(
                success=False,
                error=error_msg,
                metadata={"duration": 0}
            )

    def _prepare_task_prompt(
        self,
        task_description: str,
        task_config: Dict[str, Any]
    ) -> str:
        """
        Prepare task description for Codex CLI.

        Adds environment context and resolves placeholder variables.

        Args:
            task_description: Original task description from YAML
            task_config: Task configuration

        Returns:
            Complete prompt for Codex CLI
        """
        # Replace {agent_mode} placeholders with actual agent mode
        task = task_description.replace("{agent_mode}", self.agent_mode)

        # Start with custom system prompt if provided
        custom_prefix = ""
        if self.custom_system_prompt:
            custom_prefix = f"{self.custom_system_prompt}\n\n"

        # Add environment context
        context = f"""{custom_prefix}You are a general-purpose coding agent with access to scientific visualization tools.

Environment:
- Python environment with packages: paraview.simple, napari, numpy, scipy, matplotlib, vmd-python, ttk (topology tool-kit), GROMACS - gmx (non python CLI tool)
- You can install additional packages if needed using pip
- Working directory: {task_config.get('working_dir', 'current directory')}
- Data directory: {task_config.get('data_dir', 'same as working directory')}

Task Requirements:
- Read the task description carefully
- Write Python code to accomplish the visualization task
- Save all required output files to the specified paths
- For assertion-based tasks, output <1> for success or <0> for failure

Output Requirements:
1. **Python Script**: Save your Python script to the same results directory as other outputs
   - Use the naming pattern: {{case_name}}_script.py or {{case_name}}_visualization.py
   - Example: For case "engine", save to "engine/results/{{agent_mode}}/engine_script.py"

2. **Screenshot**: After setting up the visualization, you MUST generate a screenshot
   - Screenshot filename should match the case name (e.g., "bonsai.png", "engine.png")
   - Save screenshots to the same results directory as state files
   - The screenshot will be used for visual evaluation of your work

Strategy:
- You should check the visualization you generate to confirm if the task is accomplished or further modification is needed. Try to be efficient about the iteration and use least amount of checking while achieve reasonable result. Use headless mode for all the software tools. Do not use more than 40 turns.

Task:
{task}

IMPORTANT: Don't read folder not specificed in the instruction, and you can never check anything mark as GS (ground truth) for aidding the task. Make sure to save all output files (state files, screenshots, text files) to the exact paths specified in the task description. Use the current shell and python environment, it should have all the necessary package
"""

        return context

    async def _invoke_codex_cli(
        self,
        prompt: str,
        working_dir: Path,
        timeout: int
    ) -> Tuple[bool, str, float]:
        """
        Run Codex CLI and return results.

        Args:
            prompt: Complete task prompt
            working_dir: Directory to run Codex CLI in
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (success, output, duration)
        """
        # Create temporary file for prompt
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            # Build command with codex exec
            # --json: Output JSONL events
            # --ephemeral: Don't persist session files
            # --dangerously-bypass-approvals-and-sandbox: Skip confirmations and run without sandbox
            # -C: Change to working directory

            # Convert working_dir to absolute path to avoid relative path issues
            working_dir_abs = working_dir.resolve()

            cmd = [
                self.codex_path, "exec",
                "--json",
                "--ephemeral",
                "-C", str(working_dir_abs)
            ]

            if self.auto_approve:
                cmd.append("--dangerously-bypass-approvals-and-sandbox")

            # Add prompt from stdin
            cmd.append("-")

            print(f"Invoking Codex CLI...")
            print(f"Command: {' '.join(cmd[:-1])}... < prompt")

            start_time = time.time()

            # Verbose mode: stream output in real-time with JSON event parsing
            if self.verbose:
                # Create verbose log file in working directory
                verbose_log_path = working_dir / f"codex_cli_verbose_{int(time.time())}.log"

                print(f"\n{'='*60}")
                print(f"CODEX CLI OUTPUT (streaming):")
                print(f"Verbose log: {verbose_log_path}")
                print(f"{'='*60}\n")

                output_lines = []
                parsed_output_lines = []  # Human-readable parsed output

                with open(prompt_file, 'r') as prompt_input:
                    process = subprocess.Popen(
                        cmd,
                        cwd=None,  # Run from current directory, let -C handle directory change
                        stdin=prompt_input,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=0,  # Unbuffered for real-time streaming
                        encoding='utf-8',
                        errors='replace'
                    )

                # Open log file for writing
                with open(verbose_log_path, 'w', encoding='utf-8') as log_file:
                    log_file.write(f"Codex CLI Verbose Output Log\n")
                    log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write(f"Working Directory: {working_dir}\n")
                    log_file.write(f"{'='*80}\n\n")

                    # Parse and display JSON events line by line
                    for line in process.stdout:
                        output_lines.append(line)

                        # Write raw JSON to log file
                        log_file.write(line)
                        log_file.flush()  # Ensure immediate write

                        try:
                            event = json.loads(line.strip())
                            event_type = event.get("type")

                            if event_type == "thread.started":
                                thread_id = event.get("thread_id", "unknown")
                                msg = f"[Thread started: {thread_id}]\n"
                                print(msg.strip(), flush=True)
                                parsed_output_lines.append(msg)

                            elif event_type == "turn.started":
                                msg = "[Turn started]\n"
                                print(msg.strip(), flush=True)
                                parsed_output_lines.append(msg)

                            elif event_type == "item.started":
                                item = event.get("item", {})
                                item_type = item.get("type")
                                item_id = item.get("id", "")

                                if item_type == "reasoning":
                                    text = item.get("text", "")
                                    if text:
                                        print(f"\n🤔 Reasoning: {text[:200]}{'...' if len(text) > 200 else ''}", flush=True)
                                        parsed_output_lines.append(f"[Reasoning] {text}\n")
                                elif item_type == "command_execution":
                                    cmd = item.get("command", "")
                                    print(f"\n⚙️  Executing: {cmd[:100]}{'...' if len(cmd) > 100 else ''}", flush=True)
                                    parsed_output_lines.append(f"[Command] {cmd}\n")
                                elif item_type == "todo_list":
                                    items = item.get("items", [])
                                    print(f"\n📋 Todo list ({len(items)} items)", flush=True)
                                    for todo_item in items:
                                        status = "✅" if todo_item.get("completed") else "⬜"
                                        print(f"  {status} {todo_item.get('text', '')}", flush=True)
                                    parsed_output_lines.append(f"[Todo list] {len(items)} items\n")

                            elif event_type == "item.completed":
                                item = event.get("item", {})
                                item_type = item.get("type")

                                if item_type == "agent_message":
                                    text = item.get("text", "")
                                    if text:
                                        print(f"\n💬 Agent: {text}", flush=True)
                                        parsed_output_lines.append(f"[Agent message] {text}\n")
                                elif item_type == "reasoning":
                                    text = item.get("text", "")
                                    if text:
                                        print(f"\n🤔 Reasoning: {text[:200]}{'...' if len(text) > 200 else ''}", flush=True)
                                        parsed_output_lines.append(f"[Reasoning] {text}\n")
                                elif item_type == "command_execution":
                                    cmd = item.get("command", "")[:80]
                                    exit_code = item.get("exit_code")
                                    status = item.get("status")
                                    output = item.get("aggregated_output", "")

                                    if status == "completed" and exit_code == 0:
                                        print(f"  ✅ Command completed", flush=True)
                                        if output and len(output.strip()) > 0:
                                            print(f"  Output: {output[:150]}{'...' if len(output) > 150 else ''}", flush=True)
                                    elif status == "failed":
                                        print(f"  ❌ Command failed (exit {exit_code})", flush=True)
                                        if output:
                                            print(f"  Error: {output[:150]}{'...' if len(output) > 150 else ''}", flush=True)

                                    parsed_output_lines.append(f"[Command result] {status} (exit {exit_code})\n")
                                elif item_type == "error":
                                    error_msg = item.get("message", "")
                                    # Skip non-critical warnings
                                    if "under-development features" in error_msg.lower():
                                        print(f"⚠  {error_msg[:100]}", flush=True)
                                    else:
                                        print(f"❌ Error: {error_msg}", flush=True)
                                    parsed_output_lines.append(f"[Error] {error_msg}\n")

                            elif event_type == "item.updated":
                                item = event.get("item", {})
                                item_type = item.get("type")

                                if item_type == "todo_list":
                                    items = item.get("items", [])
                                    print(f"\n📋 Todo updated:", flush=True)
                                    for todo_item in items:
                                        status = "✅" if todo_item.get("completed") else "⬜"
                                        print(f"  {status} {todo_item.get('text', '')}", flush=True)
                                    parsed_output_lines.append(f"[Todo updated] {len(items)} items\n")

                            elif event_type == "turn.completed":
                                usage = event.get("usage", {})
                                base_input = usage.get("input_tokens", 0)
                                cached_input = usage.get("cached_input_tokens", 0)
                                combined_input = base_input + cached_input
                                output_tokens = usage.get("output_tokens", 0)
                                total_tokens = combined_input + output_tokens
                                result_line = f"\n{'='*60}\n✅ Turn completed - Input: {combined_input} tokens (includes cache), Output: {output_tokens} tokens, Total: {total_tokens} tokens\n{'='*60}\n"
                                print(result_line.strip(), flush=True)
                                parsed_output_lines.append(result_line)

                            elif event_type == "turn.failed":
                                error = event.get("error", {})
                                error_msg = error.get("message", "Unknown error")
                                result_line = f"\n{'='*60}\n❌ Turn failed: {error_msg}\n{'='*60}\n"
                                print(result_line.strip(), flush=True)
                                parsed_output_lines.append(result_line)

                            elif event_type == "error":
                                error_msg = event.get("message", "Unknown error")
                                print(f"❌ Error: {error_msg}", flush=True)
                                parsed_output_lines.append(f"[Error: {error_msg}]\n")

                        except json.JSONDecodeError:
                            # Not valid JSON - could be stderr or error message
                            if line.strip():
                                # Skip auth/model refresh warnings (non-critical stderr)
                                line_lower = line.lower()
                                if any(skip_pattern in line_lower for skip_pattern in [
                                    'failed to refresh token',
                                    'failed to refresh available models',
                                    'authentication error',
                                    'litellm virtual key'
                                ]):
                                    # Write to log but don't display
                                    continue

                                raw_line = f"[raw] {line}"
                                print(raw_line, end='', flush=True)
                                parsed_output_lines.append(raw_line)
                        except Exception as e:
                            # Unexpected error parsing event
                            error_msg = f"[parse error: {e}]\n"
                            print(error_msg.strip(), flush=True)
                            parsed_output_lines.append(error_msg)

                    # Wait for process to complete
                    process.wait(timeout=timeout)

                    # Write parsed output summary to log file
                    log_file.write(f"\n\n{'='*80}\n")
                    log_file.write(f"PARSED OUTPUT (Human-Readable)\n")
                    log_file.write(f"{'='*80}\n\n")
                    log_file.writelines(parsed_output_lines)

                    # Write footer
                    log_file.write(f"\n\n{'='*80}\n")
                    log_file.write(f"Ended: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write(f"Return Code: {process.returncode}\n")

                duration = time.time() - start_time
                output = ''.join(output_lines)  # Keep full JSON output for debugging
                success = process.returncode == 0

                print(f"\n✓ Verbose log saved to: {verbose_log_path}")
                print(f"  Log contains both raw JSON events and parsed output")

            # Non-verbose mode: capture output silently
            else:
                with open(prompt_file, 'r') as prompt_input:
                    result = subprocess.run(
                        cmd,
                        cwd=None,  # Run from current directory, let -C handle directory change
                        stdin=prompt_input,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        encoding='utf-8',
                        errors='replace'
                    )

                duration = time.time() - start_time

                # Combine stdout and stderr for complete output
                output = result.stdout + "\n" + result.stderr
                success = result.returncode == 0

            if success:
                print(f"✓ Task completed in {duration:.2f}s")
            else:
                print(f"✗ Task failed with return code {result.returncode if not self.verbose else process.returncode}")
                if not self.verbose:
                    print(f"Output preview: {output[:200]}...")

            return success, output, duration

        except subprocess.TimeoutExpired:
            duration = timeout
            output = f"Task timed out after {timeout} seconds"
            print(f"✗ {output}")
            return False, output, duration

        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            output = f"Error invoking Codex CLI: {str(e)}"
            print(f"✗ {output}")
            return False, output, duration

        finally:
            # Clean up prompt file
            try:
                os.unlink(prompt_file)
            except Exception:
                pass

    def _extract_token_usage(self, output: str) -> Dict[str, Any]:
        """
        Parse token usage from Codex CLI output.

        Codex CLI outputs token usage in JSON events like:
        {"type":"turn.completed","usage":{"input_tokens":6466,"cached_input_tokens":0,"output_tokens":57}}

        Cache tokens are combined into input_tokens for simplified reporting.

        Args:
            output: Codex CLI stdout/stderr

        Returns:
            Dictionary with input_tokens, output_tokens, total_tokens
        """
        token_info = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "source": "unknown"
        }

        # Parse JSON lines looking for turn.completed events with usage
        for line in output.split('\n'):
            try:
                event = json.loads(line.strip())
                if event.get("type") == "turn.completed":
                    usage = event.get("usage", {})
                    base_input = usage.get("input_tokens", 0)
                    cached_input = usage.get("cached_input_tokens", 0)
                    combined_input = base_input + cached_input
                    output_tokens = usage.get("output_tokens", 0)

                    # Accumulate tokens from all turns
                    token_info["input_tokens"] += combined_input
                    token_info["output_tokens"] += output_tokens
                    token_info["source"] = "codex_output"
            except (json.JSONDecodeError, AttributeError):
                continue

        # Calculate total
        token_info["total_tokens"] = (
            token_info["input_tokens"] +
            token_info["output_tokens"]
        )

        # If no tokens found, estimate based on output length
        if token_info["total_tokens"] == 0:
            estimated = self.count_tokens(output)
            token_info["output_tokens"] = estimated
            token_info["total_tokens"] = estimated
            token_info["source"] = "estimated"

        return token_info

    def _estimate_token_usage_from_context(
        self,
        prompt: str,
        output_files: Dict[str, str],
        results_dir: Path,
        case_name: str
    ) -> Dict[str, Any]:
        """
        Provide better token usage estimate based on actual content.

        Estimates based on:
        1. Input prompt length (task + system context)
        2. Generated code files (Python scripts)
        3. Terminal output text
        4. System prompt overhead (~1000 tokens for Codex internals)

        Args:
            prompt: The task prompt sent to Codex CLI
            output_files: Dictionary of generated output files
            results_dir: Results directory path
            case_name: Name of the test case

        Returns:
            Dict with input_tokens, output_tokens, total_tokens, source
        """
        # Estimate input tokens from prompt
        input_tokens = self.count_tokens(prompt)

        # Add Codex system prompt overhead
        input_tokens += 1000

        # Estimate output tokens from generated code files
        output_tokens = 0

        # Priority 1: Check results directory for Python scripts
        script_found = False
        for pattern in [f"{case_name}_script.py", f"{case_name}_visualization.py", f"{case_name}.py", "*.py"]:
            py_files = list(results_dir.glob(pattern))
            if py_files:
                for py_file in py_files:
                    try:
                        with open(py_file, 'r') as f:
                            code_content = f.read()
                            code_tokens = self.count_tokens(code_content)
                            output_tokens += code_tokens
                            print(f"  Counted {code_tokens} tokens from {py_file.name} (in results dir)")
                            script_found = True
                    except Exception as e:
                        print(f"  Warning: Could not read {py_file}: {e}")
                if script_found:
                    break

        # Priority 2: Check working directory for Python scripts
        if not script_found:
            working_dir = results_dir.parent
            import glob
            python_patterns = [
                str(working_dir / f"*{case_name}*.py"),
                str(working_dir / f"{case_name}*.py"),
            ]

            for pattern in python_patterns:
                python_files = glob.glob(pattern)
                for py_file in python_files:
                    try:
                        # Only count files modified in the last 5 minutes
                        file_mtime = os.path.getmtime(py_file)
                        current_time = time.time()

                        if (current_time - file_mtime) < 300:  # 5 minutes
                            with open(py_file, 'r') as f:
                                code_content = f.read()
                                code_tokens = self.count_tokens(code_content)
                                output_tokens += code_tokens
                                print(f"  Counted {code_tokens} tokens from {os.path.basename(py_file)} (in working dir)")
                                script_found = True
                    except Exception as e:
                        print(f"  Warning: Could not read {py_file}: {e}")
                if script_found:
                    break

        # Add tokens for terminal output and explanations
        output_tokens += 800

        # If we found no code or very little, use a reasonable default
        if output_tokens < 1500:
            print(f"  No recent Python files found, using default estimate")
            output_tokens = 3000  # Conservative default for visualization tasks

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "source": "context_estimate"
        }

    def _find_output_files(
        self,
        results_dir: Path,
        case_name: str
    ) -> Dict[str, str]:
        """
        Locate generated output files in results directory.

        Looks for common file types:
        - State files: .pvsm, .state, .json
        - Images: .png, .jpg, .jpeg
        - Text files: .txt, .log
        - Data files: .csv, .npy

        Args:
            results_dir: Directory where outputs should be saved
            case_name: Name of the test case

        Returns:
            Dictionary mapping file types to paths
        """
        output_files = {}

        # Define file patterns to look for
        file_patterns = {
            "state": [f"{case_name}.pvsm", f"{case_name}.state", "*.pvsm", "*.state"],
            "image": [f"{case_name}.png", f"{case_name}.jpg", "*.png", "*.jpg"],
            "script": [f"{case_name}_script.py", f"{case_name}_visualization.py", f"{case_name}.py", "*.py"],
            "text": [f"{case_name}.txt", "output.txt", "*.txt"],
            "data": [f"{case_name}.csv", f"{case_name}.npy", "*.csv", "*.npy"],
        }

        # Search for files
        for file_type, patterns in file_patterns.items():
            for pattern in patterns:
                matches = list(results_dir.glob(pattern))
                if matches:
                    # Use first match
                    output_files[file_type] = str(matches[0])
                    break

        if output_files:
            print(f"Found output files: {list(output_files.keys())}")
        else:
            print("⚠ Warning: No output files found in results directory")

        return output_files

    async def prepare_task(self, task_config: Dict[str, Any]):
        """Prepare for a specific task (optional hook)."""
        pass

    async def cleanup_task(self, task_config: Dict[str, Any]):
        """Cleanup after a specific task (optional hook)."""
        if not self.preserve_workdir:
            # Clean up any temporary files created during task
            pass

    async def teardown(self):
        """Teardown after all tasks complete."""
        print("Codex CLI agent teardown complete")
