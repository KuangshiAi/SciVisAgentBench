#!/usr/bin/env python3
"""
Standalone runner for Claude Code agent evaluation.

This script imports the ClaudeCodeAgent (which registers it via decorator)
and then uses the standard evaluation framework. No modifications to existing
files needed!

Usage:
    python benchmark/run_claude_code_eval.py \
        --agent claude_code \
        --config benchmark/configs/claude_code/config.json \
        --yaml benchmark/eval_cases/paraview/main_cases.yaml \
        --cases SciVisAgentBench-tasks/main \
        --eval-model gpt-4o

    # Run single case for testing
    python benchmark/run_claude_code_eval.py \
        --agent claude_code \
        --config benchmark/configs/claude_code/config.json \
        --yaml benchmark/eval_cases/paraview/main_cases.yaml \
        --cases SciVisAgentBench-tasks/main \
        --case bonsai \
        --eval-model gpt-4o
"""

import sys
import asyncio
from pathlib import Path

# Add benchmark to path
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

# Import Claude Code agent - this registers it via @register_agent decorator
from evaluation_framework.agents.claude_code_agent import ClaudeCodeAgent

# Import and run standard evaluation
from evaluation_framework.run_evaluation import main as eval_main

if __name__ == "__main__":
    print("=" * 60)
    print("Claude Code Agent Evaluation Runner")
    print("=" * 60)
    print(f"Agent registered: claude_code")
    print(f"Agent class: {ClaudeCodeAgent.__name__}")
    print("=" * 60)
    print()

    # Run evaluation using standard framework
    try:
        sys.exit(asyncio.run(eval_main()))
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
        sys.exit(1)
