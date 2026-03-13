"""
Claude Code Agent

This agent integrates Claude Code CLI as a general-purpose coding agent
for the SciVisAgentBench evaluation framework.

Claude Code receives natural language task descriptions and figures out
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


@register_agent("claude_code")
class ClaudeCodeAgent(BaseAgent):
    """
    Claude Code agent that uses the Claude CLI to complete visualization tasks.

    This agent is tool-agnostic - it receives task descriptions and figures out
    how to interact with visualization packages (paraview.simple, napari, etc.)
    that are available in the Python environment.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Claude Code agent.

        Args:
            config: Configuration dictionary with keys:
                - model: Claude model to use (e.g., "claude-sonnet-4-5")
                - claude_code_path: Path to Claude CLI (default: "claude")
                - timeout_per_task: Timeout in seconds (default: 600)
                - preserve_workdir: Keep working directories for debugging
                - environment: Environment specification (optional)
                - price: Pricing information for cost calculation
                - custom_system_prompt: Optional additional instructions to prepend to all tasks
        """
        # Set defaults for BaseAgent
        config["eval_mode"] = config.get("eval_mode", "generic")
        config["agent_name"] = config.get("agent_name", "claude_code")

        super().__init__(config)

        self.claude_path = config.get("claude_code_path", "claude")
        self.timeout = config.get("timeout_per_task", 600)
        self.preserve_workdir = config.get("preserve_workdir", False)
        self.auto_approve = config.get("auto_approve", True)  # Default to auto-approve for benchmarking
        self.custom_system_prompt = config.get("custom_system_prompt", "")
        self.verbose = config.get("verbose", False)  # Enable real-time output streaming

        print(f"ClaudeCodeAgent initialized:")
        print(f"  - Model: {config.get('model', 'default')}")
        print(f"  - Agent mode: {self.agent_mode}")
        print(f"  - Timeout: {self.timeout}s")
        print(f"  - Claude path: {self.claude_path}")
        print(f"  - Auto-approve: {self.auto_approve}")
        print(f"  - Verbose: {self.verbose}")
        if self.custom_system_prompt:
            print(f"  - Custom prompt: {len(self.custom_system_prompt)} chars")

    async def setup(self):
        """Verify Claude Code CLI is available."""
        try:
            result = subprocess.run(
                [self.claude_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✓ Claude Code CLI found: {result.stdout.strip()}")
            else:
                print(f"⚠ Claude Code CLI check returned non-zero: {result.returncode}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Claude Code CLI not found at '{self.claude_path}'. "
                "Please install it or set 'claude_code_path' in config."
            )
        except Exception as e:
            print(f"⚠ Warning: Could not verify Claude Code CLI: {e}")

    async def run_task(
        self,
        task_description: str,
        task_config: Dict[str, Any]
    ) -> AgentResult:
        """
        Execute task using Claude Code CLI.

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

            # Invoke Claude Code
            success, output, duration = await self._invoke_claude_code(
                prompt=prompt,
                working_dir=working_dir,
                timeout=self.timeout
            )

            # Find output files generated by Claude Code
            output_files = self._find_output_files(dirs["results_dir"], case_name)

            # Extract token usage from output if available
            token_usage = self._extract_token_usage(output)

            # If token usage is still estimated (Claude Code doesn't output usage),
            # provide a better estimate based on prompt + generated code
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
                    "claude_path": self.claude_path
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
        Prepare task description for Claude Code.

        Adds environment context and resolves placeholder variables.

        Args:
            task_description: Original task description from YAML
            task_config: Task configuration

        Returns:
            Complete prompt for Claude Code
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
- Python environment with packages: paraview.simple, napari, numpy, scipy, matplotlib, vmd-python, mdanalysis, ttk (topology tool-kit), GROMACS - gmx (non python CLI tool)
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

Stratergy:
- You should check the visualization you generate to confirm if the task is accomplished or further modification is needed. Try to be efficient about the iteration and use least amount of checking while achieve reasonable result. Use headless mode for all the software tools. Take screenshot of the viewport or rendering not the software. Do not use more than 40 turns.

Task:
{task}

IMPORTANT: You should never check anything mark as GS (ground truth) for aidding the task. Make sure to save all output files (state files, screenshots, text files) to the exact paths specified in the task description.
"""

        return context

    async def _invoke_claude_code(
        self,
        prompt: str,
        working_dir: Path,
        timeout: int
    ) -> Tuple[bool, str, float]:
        """
        Run Claude Code CLI and return results.

        Args:
            prompt: Complete task prompt
            working_dir: Directory to run Claude Code in
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
            # Prepare command with optional auto-approve
            # Note: --dangerously-skip-permissions is required for non-interactive benchmarking
            # Security considerations:
            # 1. Only use in controlled environments (no sensitive data)
            # 2. Network access can be restricted via settings.json "deny": ["WebFetch", "WebSearch"]
            # 3. Run in isolated conda environment
            # 4. Monitor file system changes

            # Build command with --print flag to prevent interactive session
            # --print makes Claude Code "print response and exit" instead of staying in interactive mode
            # In verbose mode, use stream-json for real-time streaming output
            if self.verbose:
                if self.auto_approve:
                    cmd = [self.claude_path, "--print", "--verbose", "--output-format", "stream-json",
                           "--dangerously-skip-permissions", prompt]
                else:
                    cmd = [self.claude_path, "--print", "--verbose", "--output-format", "stream-json", prompt]
            else:
                if self.auto_approve:
                    cmd = [self.claude_path, "--print", "--dangerously-skip-permissions", prompt]
                else:
                    cmd = [self.claude_path, "--print", prompt]

            print(f"Invoking Claude Code...")
            print(f"Command: {' '.join(cmd[:3] if len(cmd) > 2 else cmd[:2])}...")  # Don't print full prompt

            start_time = time.time()

            # Verbose mode: stream output in real-time with JSON event parsing
            if self.verbose:
                # Create verbose log file in working directory
                verbose_log_path = working_dir / f"claude_code_verbose_{int(time.time())}.log"

                print(f"\n{'='*60}")
                print(f"CLAUDE CODE OUTPUT (streaming):")
                print(f"Verbose log: {verbose_log_path}")
                print(f"{'='*60}\n")

                output_lines = []
                parsed_output_lines = []  # Human-readable parsed output

                process = subprocess.Popen(
                    cmd,
                    cwd=str(working_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=0,  # Unbuffered for real-time streaming
                    encoding='utf-8',
                    errors='replace'
                )

                # Open log file for writing
                with open(verbose_log_path, 'w', encoding='utf-8') as log_file:
                    log_file.write(f"Claude Code Verbose Output Log\n")
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

                            if event_type == "system":
                                # Skip system init messages (not user-facing)
                                parsed_output_lines.append("[System initialization]\n")
                                continue

                            elif event_type == "assistant":
                                message = event.get("message", {})
                                content = message.get("content", [])

                                for item in content:
                                    if item.get("type") == "text":
                                        # Display text content
                                        text = item.get("text", "")
                                        if text:
                                            print(text, flush=True)
                                            parsed_output_lines.append(f"{text}\n")

                                    elif item.get("type") == "tool_use":
                                        # Display tool use
                                        tool_name = item.get("name", "unknown")
                                        tool_input = item.get("input", {})

                                        # Format tool display
                                        output_line = f"\n→ Using tool: {tool_name}\n"
                                        print(output_line.strip(), flush=True)
                                        parsed_output_lines.append(output_line)

                                        # Show key parameters (not all, to avoid clutter)
                                        if "command" in tool_input:
                                            param_line = f"  Command: {tool_input['command'][:80]}\n"
                                            print(param_line.strip(), flush=True)
                                            parsed_output_lines.append(param_line)
                                        elif "file_path" in tool_input:
                                            param_line = f"  File: {tool_input['file_path']}\n"
                                            print(param_line.strip(), flush=True)
                                            parsed_output_lines.append(param_line)
                                        elif "pattern" in tool_input:
                                            param_line = f"  Pattern: {tool_input['pattern']}\n"
                                            print(param_line.strip(), flush=True)
                                            parsed_output_lines.append(param_line)

                            elif event_type == "user":
                                # Tool result - show abbreviated output
                                tool_result = event.get("tool_use_result", {})
                                stdout = tool_result.get("stdout", "")
                                stderr = tool_result.get("stderr", "")

                                if stdout:
                                    preview = stdout[:200] + ("..." if len(stdout) > 200 else "")
                                    output_line = f"  Output: {preview}\n"
                                    print(output_line.strip(), flush=True)
                                    parsed_output_lines.append(output_line)
                                if stderr:
                                    preview = stderr[:200] + ("..." if len(stderr) > 200 else "")
                                    error_line = f"  Error: {preview}\n"
                                    print(error_line.strip(), flush=True)
                                    parsed_output_lines.append(error_line)

                            elif event_type == "result":
                                # Final result with token usage
                                subtype = event.get("subtype", "unknown")
                                duration_ms = event.get("duration_ms", 0)
                                num_turns = event.get("num_turns", 0)

                                # Extract token usage and combine cache tokens
                                usage = event.get("usage", {})
                                base_input = usage.get("input_tokens", 0)
                                cache_creation = usage.get("cache_creation_input_tokens", 0)
                                cache_read = usage.get("cache_read_input_tokens", 0)
                                combined_input = base_input + cache_creation + cache_read
                                output_tokens = usage.get("output_tokens", 0)
                                total_tokens = combined_input + output_tokens
                                cost_usd = event.get("total_cost_usd", 0.0)

                                result_line = f"\n{'='*60}\n✅ Completed: {subtype} in {duration_ms/1000:.2f}s ({num_turns} turns)\n"
                                result_line += f"📊 Tokens: {combined_input} input (includes cache) + {output_tokens} output = {total_tokens} total\n"
                                result_line += f"💰 Cost: ${cost_usd:.4f}\n"
                                result_line += f"{'='*60}\n"

                                print(f"\n{'='*60}", flush=True)
                                print(f"✅ Completed: {subtype} in {duration_ms/1000:.2f}s ({num_turns} turns)", flush=True)
                                print(f"📊 Tokens: {combined_input} input (includes cache) + {output_tokens} output = {total_tokens} total", flush=True)
                                print(f"💰 Cost: ${cost_usd:.4f}", flush=True)
                                print(f"{'='*60}\n", flush=True)
                                parsed_output_lines.append(result_line)

                        except json.JSONDecodeError:
                            # Not valid JSON - could be stderr or error message
                            # Print raw line
                            if line.strip():
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
                result = subprocess.run(
                    cmd,
                    cwd=str(working_dir),
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
            output = f"Error invoking Claude Code: {str(e)}"
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
        Parse token usage from Claude Code output.

        Claude Code with --output-format=stream-json provides a final result event:
        {"type":"result","usage":{"input_tokens":X,"output_tokens":Y,...},"total_cost_usd":Z}

        Cache tokens are combined into input_tokens for simplified reporting.

        Args:
            output: Claude Code stdout/stderr

        Returns:
            Dictionary with input_tokens, output_tokens, total_tokens, cost_usd
        """
        token_info = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "source": "unknown"
        }

        # Parse JSON lines looking for the final result event
        for line in output.split('\n'):
            try:
                event = json.loads(line.strip())
                if event.get("type") == "result":
                    usage = event.get("usage", {})

                    # Extract token counts and combine cache tokens into input
                    base_input = usage.get("input_tokens", 0)
                    cache_creation = usage.get("cache_creation_input_tokens", 0)
                    cache_read = usage.get("cache_read_input_tokens", 0)
                    combined_input = base_input + cache_creation + cache_read

                    output_tokens = usage.get("output_tokens", 0)

                    token_info["input_tokens"] = combined_input
                    token_info["output_tokens"] = output_tokens
                    token_info["total_tokens"] = combined_input + output_tokens
                    token_info["cost_usd"] = event.get("total_cost_usd", 0.0)
                    token_info["source"] = "claude_output"

                    # Successfully extracted from result event
                    return token_info
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue

        # Fallback: Try regex patterns for non-JSON output
        # Parse separate fields then combine cache tokens into input
        temp_tokens = {
            "base_input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_creation": 0,
        }

        patterns = {
            "base_input": r'input[_\s]tokens?[:\s=]+(\d+)',
            "output": r'output[_\s]tokens?[:\s=]+(\d+)',
            "cache_read": r'cache[_\s]read[_\s]input[_\s]tokens?[:\s=]+(\d+)',
            "cache_creation": r'cache[_\s]creation[_\s]input[_\s]tokens?[:\s=]+(\d+)',
        }

        for field, pattern in patterns.items():
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                # Prefer the last occurrence in case the output contains multiple summaries
                temp_tokens[field] = int(matches[-1])
                token_info["source"] = "claude_output_regex"

        # Combine cache tokens into input
        if token_info["source"] == "claude_output_regex":
            token_info["input_tokens"] = (
                temp_tokens["base_input"] +
                temp_tokens["cache_creation"] +
                temp_tokens["cache_read"]
            )
            token_info["output_tokens"] = temp_tokens["output"]
            token_info["total_tokens"] = token_info["input_tokens"] + token_info["output_tokens"]

        # Optional: cost, if present in plaintext logs
        cost_matches = re.findall(r'total[_\s]cost[_\s]usd[:\s=]+([0-9]*\\.?[0-9]+)', output, re.IGNORECASE)
        if cost_matches:
            try:
                token_info["cost_usd"] = float(cost_matches[-1])
                if token_info["source"] == "unknown":
                    token_info["source"] = "claude_output_regex"
            except ValueError:
                pass

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

        Claude Code CLI doesn't expose token usage, so we estimate based on:
        1. Input prompt length (task + system context)
        2. Generated code files (Python scripts)
        3. Terminal output text
        4. System prompt overhead (~1000 tokens for Claude Code internals)

        This provides much more realistic estimates than just terminal output.

        Args:
            prompt: The task prompt sent to Claude Code
            output_files: Dictionary of generated output files
            results_dir: Results directory path
            case_name: Name of the test case

        Returns:
            Dict with input_tokens, output_tokens, total_tokens, source
        """
        # Estimate input tokens from prompt
        # The prompt includes task description, environment context, and instructions
        input_tokens = self.count_tokens(prompt)

        # Add Claude Code system prompt overhead
        # Claude Code has built-in system instructions for tools, file operations, etc.
        # Estimated at ~1000 tokens based on typical Claude Code system context
        input_tokens += 1000

        # Estimate output tokens from generated code files
        output_tokens = 0

        # Priority 1: Check results directory for Python scripts
        # (if Claude Code followed our instructions)
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
        # (fallback if Claude Code saved in working dir)
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
                        # Get file modification time to avoid counting old files
                        import os
                        file_mtime = os.path.getmtime(py_file)
                        current_time = time.time()

                        # Only count files modified in the last 5 minutes
                        # This avoids counting old scripts from previous runs
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
        # Claude Code provides explanations, status updates, etc.
        # Typical: 500-1500 tokens depending on verbosity
        # We'll add a baseline of 800 tokens
        output_tokens += 800

        # If we found no code or very little, use a reasonable default
        # Typical ParaView visualization script: 2000-4000 tokens
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

        # Also check if files exist in working directory and copy them
        # (In case Claude Code saved them in the wrong location)

        if output_files:
            print(f"Found output files: {list(output_files.keys())}")
        else:
            print("⚠ Warning: No output files found in results directory")

        return output_files

    async def prepare_task(self, task_config: Dict[str, Any]):
        """Prepare for a specific task (optional hook)."""
        # Could be used to set up per-task environment if needed
        pass

    async def cleanup_task(self, task_config: Dict[str, Any]):
        """Cleanup after a specific task (optional hook)."""
        # Could be used to clean up temporary files if needed
        if not self.preserve_workdir:
            # Clean up any temporary files created during task
            pass

    async def teardown(self):
        """Teardown after all tasks complete."""
        print("Claude Code agent teardown complete")
