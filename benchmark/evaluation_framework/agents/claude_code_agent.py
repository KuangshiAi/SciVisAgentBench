"""
Claude Code Agent

This agent integrates Claude Code CLI as a general-purpose coding agent
for the SciVisAgentBench evaluation framework.

Claude Code receives natural language task descriptions and figures out
how to use visualization tools (ParaView, Napari, GMX-VMD, etc.) without
specialized MCP servers or pre-built tools.
"""

import asyncio
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

        print(f"ClaudeCodeAgent initialized:")
        print(f"  - Model: {config.get('model', 'default')}")
        print(f"  - Agent mode: {self.agent_mode}")
        print(f"  - Timeout: {self.timeout}s")
        print(f"  - Claude path: {self.claude_path}")
        print(f"  - Auto-approve: {self.auto_approve}")
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

            # Extract token usage from output if available
            token_usage = self._extract_token_usage(output)

            # Find output files generated by Claude Code
            output_files = self._find_output_files(dirs["results_dir"], case_name)

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
            return AgentResult(
                success=False,
                error=f"Exception during task execution: {str(e)}",
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
- Python environment with packages: paraview.simple, napari, numpy, scipy, matplotlib
- You can install additional packages if needed using pip
- Working directory: {task_config.get('working_dir', 'current directory')}
- Data directory: {task_config.get('data_dir', 'same as working directory')}

Task Requirements:
- Read the task description carefully
- Write Python code to accomplish the visualization task
- Save all required output files to the specified paths
- For assertion-based tasks, output <1> for success or <0> for failure

CRITICAL SCREENSHOT REQUIREMENT:
- After setting up the visualization, you MUST generate a screenshot
- Use SaveScreenshot() in ParaView to save a PNG image
- Screenshot filename should match the case name (e.g., "bonsai.png", "engine.png")
- Save screenshots to the same results directory as state files
- The screenshot will be used for visual evaluation of your work

Task:
{task}

IMPORTANT: Make sure to save all output files (state files, screenshots, text files) to the exact paths specified in the task description.
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
            if self.auto_approve:
                cmd = [self.claude_path, "--dangerously-skip-permissions", prompt]
            else:
                cmd = [self.claude_path, prompt]

            print(f"Invoking Claude Code...")
            print(f"Command: {' '.join(cmd[:3] if len(cmd) > 2 else cmd[:2])}...")  # Don't print full prompt

            start_time = time.time()

            # Run Claude Code
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
                print(f"✗ Task failed with return code {result.returncode}")
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

        Looks for patterns like:
        - "tokens: input=X, output=Y"
        - "Input tokens: X, Output tokens: Y"
        - JSON with token information

        Args:
            output: Claude Code stdout/stderr

        Returns:
            Dictionary with input_tokens, output_tokens, total_tokens
        """
        token_info = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "source": "unknown"
        }

        # Try various patterns to extract token usage
        patterns = [
            r'input[_\s]tokens?[:\s=]+(\d+)',
            r'output[_\s]tokens?[:\s=]+(\d+)',
            r'total[_\s]tokens?[:\s=]+(\d+)',
            r'tokens?[:\s]+input[=:]\s*(\d+)',
            r'tokens?[:\s]+output[=:]\s*(\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                # Extract numbers and assign to appropriate fields
                if 'input' in pattern.lower():
                    token_info["input_tokens"] = int(matches[0])
                    token_info["source"] = "claude_output"
                elif 'output' in pattern.lower():
                    token_info["output_tokens"] = int(matches[0])
                    token_info["source"] = "claude_output"
                elif 'total' in pattern.lower():
                    token_info["total_tokens"] = int(matches[0])
                    token_info["source"] = "claude_output"

        # Calculate total if not found
        if token_info["total_tokens"] == 0:
            token_info["total_tokens"] = (
                token_info["input_tokens"] + token_info["output_tokens"]
            )

        # If no tokens found, estimate based on output length
        if token_info["total_tokens"] == 0:
            estimated = self.count_tokens(output)
            token_info["output_tokens"] = estimated
            token_info["total_tokens"] = estimated
            token_info["source"] = "estimated"

        return token_info

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
