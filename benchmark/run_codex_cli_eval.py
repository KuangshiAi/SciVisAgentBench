#!/usr/bin/env python3
"""
Standalone runner for Codex CLI agent evaluation.

This script imports the CodexCLIAgent (which registers it via decorator)
and then uses the standard evaluation framework. No modifications to existing
files needed!

Usage:
    # Run all cases
    python benchmark/run_codex_cli_eval.py \
        --agent codex_cli \
        --config benchmark/configs/codex_cli/config.json \
        --yaml benchmark/eval_cases/paraview/main_cases.yaml \
        --cases SciVisAgentBench-tasks/main \
        --eval-model gpt-4o

    # Run single case for testing
    python benchmark/run_codex_cli_eval.py \
        --agent codex_cli \
        --config benchmark/configs/codex_cli/config.json \
        --yaml benchmark/eval_cases/paraview/main_cases.yaml \
        --cases SciVisAgentBench-tasks/main \
        --case bonsai \
        --eval-model gpt-4o

    # Resume from a specific case (e.g., after engine case failed)
    python benchmark/run_codex_cli_eval.py \
        --agent codex_cli \
        --config benchmark/configs/codex_cli/config.json \
        --yaml benchmark/eval_cases/paraview/main_cases.yaml \
        --cases SciVisAgentBench-tasks/main \
        --start-from engine \
        --eval-model gpt-4o
"""

import sys
import asyncio
from pathlib import Path

# Add benchmark to path
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

# Import Codex CLI agent - this registers it via @register_agent decorator
from evaluation_framework.agents.codex_cli_agent import CodexCLIAgent

# Import and run standard evaluation
from evaluation_framework.run_evaluation import main as eval_main

if __name__ == "__main__":
    print("=" * 60)
    print("Codex CLI Agent Evaluation Runner")
    print("=" * 60)
    print(f"Agent registered: codex_cli")
    print(f"Agent class: {CodexCLIAgent.__name__}")
    print("=" * 60)
    print()

    # Run evaluation using standard framework
    try:
        sys.exit(asyncio.run(eval_main()))
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
        sys.exit(1)
