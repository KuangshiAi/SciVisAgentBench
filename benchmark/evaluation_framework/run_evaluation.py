#!/usr/bin/env python3
"""
Run Evaluation CLI Tool

A command-line tool for easily evaluating agents on SciVisAgentBench.

Usage examples:

1. Evaluate ParaView MCP agent on main benchmark:
   python -m benchmark.evaluation_framework.run_evaluation \\
       --agent paraview_mcp \\
       --config benchmark/configs/paraview_mcp/config_openai.json \\
       --yaml SciVisAgentBench-tasks/main/main_cases.yaml \\
       --cases SciVisAgentBench-tasks/main

2. Clear previous results before running:
   python -m benchmark.evaluation_framework.run_evaluation \\
       --agent paraview_mcp \\
       --config benchmark/configs/paraview_mcp/config_openai.json \\
       --yaml SciVisAgentBench-tasks/main/main_cases.yaml \\
       --cases SciVisAgentBench-tasks/main \\
       --clear-results

3. Evaluate existing results without re-running agent:
   python -m benchmark.evaluation_framework.run_evaluation \\
       --agent paraview_mcp \\
       --config benchmark/configs/paraview_mcp/config_openai.json \\
       --yaml SciVisAgentBench-tasks/main/main_cases.yaml \\
       --cases SciVisAgentBench-tasks/main \\
       --eval-only

4. List available agents:
   python -m benchmark.evaluation_framework.run_evaluation --list-agents
"""

import argparse
import asyncio
import os
import sys
import shutil
from pathlib import Path

# Add benchmark directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import framework components
from evaluation_framework import get_agent, list_agents, UnifiedTestRunner

# Import all pre-built agents (this registers them)
from evaluation_framework.agents import (
    ParaViewMCPAgent,
    NapariMCPAgent,
    ChatVisAgent
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate scientific visualization agents on SciVisAgentBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Agent selection
    parser.add_argument(
        "--agent", "-a",
        help="Agent to evaluate (use --list-agents to see available agents)"
    )

    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List all registered agents and exit"
    )

    # Configuration
    parser.add_argument(
        "--config", "-c",
        help="Path to agent configuration JSON file"
    )

    # Test cases
    parser.add_argument(
        "--yaml", "-y",
        help="Path to YAML test cases file"
    )

    parser.add_argument(
        "--cases",
        help="Path to test cases directory"
    )

    parser.add_argument(
        "--data-dir",
        help="Optional separate data directory (defaults to cases directory)"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output directory for results (default: auto-generated based on agent)"
    )

    # Evaluation options
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip LLM-based evaluation (only run inference)"
    )

    parser.add_argument(
        "--eval-model",
        default="gpt-4o",
        help="Model to use for LLM evaluation (default: gpt-4o)"
    )

    parser.add_argument(
        "--static-screenshot",
        action="store_true",
        help="Use pre-generated screenshots instead of generating from state files"
    )

    parser.add_argument(
        "--clear-results",
        action="store_true",
        help="Clear all previous agent execution results before running (removes results/, evaluation_results/, test_results/ from each case)"
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip agent execution and only run evaluation on existing results"
    )

    # API keys
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (can also use OPENAI_API_KEY env var)"
    )

    # Specific case
    parser.add_argument(
        "--case",
        help="Run only a specific test case by name"
    )

    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available test cases and exit"
    )

    return parser.parse_args()


def clear_case_results(cases_dir: str, case_name: str = None):
    """
    Clear agent execution results from test case directories.

    Args:
        cases_dir: Path to the cases directory
        case_name: Optional specific case name. If None, clears all cases.
    """
    cases_path = Path(cases_dir)

    if not cases_path.exists():
        print(f"Warning: Cases directory not found: {cases_dir}")
        return

    # Directories to remove from each case
    dirs_to_remove = ["results", "evaluation_results", "test_results"]

    # Get list of cases to clear
    if case_name:
        cases_to_clear = [case_name]
    else:
        # Get all subdirectories (test cases)
        cases_to_clear = [d.name for d in cases_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    cleared_count = 0
    for case in cases_to_clear:
        case_dir = cases_path / case
        if not case_dir.exists():
            print(f"Warning: Case directory not found: {case_dir}")
            continue

        for dir_name in dirs_to_remove:
            dir_path = case_dir / dir_name
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    print(f"  ✓ Cleared {case}/{dir_name}/")
                    cleared_count += 1
                except Exception as e:
                    print(f"  ✗ Error clearing {case}/{dir_name}/: {e}")

    if cleared_count > 0:
        print(f"\n✓ Cleared {cleared_count} result directories")
    else:
        print("\nℹ No result directories found to clear")


def list_registered_agents():
    """List all registered agents with information."""
    agents = list_agents()

    if not agents:
        print("No agents registered.")
        print("\nTo register an agent, use:")
        print("  from benchmark.evaluation_framework import register_agent, BaseAgent")
        print("  ")
        print("  @register_agent('my_agent')")
        print("  class MyAgent(BaseAgent):")
        print("      ...")
        return

    print("Available agents:")
    print("=" * 60)

    for agent_name in sorted(agents):
        from evaluation_framework.agent_registry import get_agent_info
        info = get_agent_info(agent_name)

        print(f"\n{agent_name}")
        print(f"  Class: {info['class_name']}")
        print(f"  Module: {info['module']}")

        # Print first line of docstring
        doc = info['doc'].strip().split('\n')[0] if info['doc'] else "No description"
        print(f"  Description: {doc}")

    print("\n" + "=" * 60)


async def main():
    """Main entry point."""
    args = parse_args()

    # Handle --list-agents
    if args.list_agents:
        list_registered_agents()
        return 0

    # Validate required arguments
    if not args.agent:
        print("Error: --agent is required")
        print("Use --list-agents to see available agents")
        return 1

    if not args.config:
        print("Error: --config is required")
        return 1

    if not args.yaml:
        print("Error: --yaml is required")
        return 1

    if not args.cases:
        print("Error: --cases is required")
        return 1

    # Validate file paths
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    if not os.path.exists(args.yaml):
        print(f"Error: YAML file not found: {args.yaml}")
        return 1

    if not os.path.exists(args.cases):
        print(f"Error: Cases directory not found: {args.cases}")
        return 1

    # Get OpenAI API key
    openai_api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not args.no_eval and not openai_api_key:
        print("Warning: No OpenAI API key provided. Evaluation will be skipped.")
        print("Set OPENAI_API_KEY environment variable or use --openai-api-key to enable evaluation.")

    try:
        # Get agent class
        agent_class = get_agent(args.agent)

        # Load configuration
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)

        # Create agent instance
        print(f"\nCreating agent: {args.agent}")
        agent = agent_class(config)

        # Create test runner (pass config for rate limiting)
        runner = UnifiedTestRunner(
            agent=agent,
            yaml_path=args.yaml,
            cases_dir=args.cases,
            data_dir=args.data_dir,
            output_dir=args.output,
            openai_api_key=openai_api_key,
            eval_model=args.eval_model,
            static_screenshot=args.static_screenshot,
            config=config
        )

        # Load test cases
        test_cases = runner.load_yaml_test_cases()

        if args.list_cases:
            print("\nAvailable test cases:")
            for case in test_cases:
                print(f"  - {case.case_name}")
            return 0

        if not test_cases:
            print("Error: No valid test cases found in YAML file")
            return 1

        # Handle --clear-results flag
        if args.clear_results:
            print("\n🗑️  Clearing previous results...")
            clear_case_results(args.cases, args.case)
            print()

        # Handle --eval-only flag
        if args.eval_only:
            print("\n📊 Evaluation-only mode: Skipping agent execution\n")
            if not openai_api_key:
                print("Error: --eval-only requires an OpenAI API key for evaluation")
                print("Set OPENAI_API_KEY environment variable or use --openai-api-key")
                return 1

            # Run evaluation on all cases
            if args.case:
                # Find specific case
                target_case = None
                for case in test_cases:
                    if case.case_name == args.case:
                        target_case = case
                        break

                if not target_case:
                    print(f"Error: Test case '{args.case}' not found")
                    return 1

                print(f"Evaluating existing results for: {args.case}")
                eval_result = await runner.run_evaluation(target_case)

                # Create a dummy result with evaluation
                result = {
                    "status": "eval_only",
                    "case_name": args.case,
                    "evaluation": eval_result
                }
                await runner.save_centralized_result(target_case, result)

                print(f"\n✓ Evaluation complete for {args.case}")
                return 0
            else:
                # Evaluate all cases
                print(f"Evaluating existing results for {len(test_cases)} test cases...")
                for test_case in test_cases:
                    print(f"\n{'='*60}")
                    print(f"Evaluating: {test_case.case_name}")
                    print(f"{'='*60}")

                    try:
                        eval_result = await runner.run_evaluation(test_case)

                        # Create a dummy result with evaluation
                        result = {
                            "status": "eval_only",
                            "case_name": test_case.case_name,
                            "evaluation": eval_result
                        }
                        await runner.save_centralized_result(test_case, result)

                        print(f"✓ Evaluation complete")
                    except Exception as e:
                        print(f"✗ Evaluation failed: {e}")

                print(f"\n✓ All evaluations complete")
                return 0

        # Run specific case or all cases
        if args.case:
            # Find specific case
            target_case = None
            for case in test_cases:
                if case.case_name == args.case:
                    target_case = case
                    break

            if not target_case:
                print(f"Error: Test case '{args.case}' not found")
                print("Available cases:", [case.case_name for case in test_cases])
                return 1

            # Setup agent
            await agent.setup()

            try:
                # Run single case without saving (we'll save after evaluation)
                print(f"\nRunning single test case: {args.case}")
                result = await runner.run_single_test_case(target_case, save_result=False)

                # Run evaluation if requested
                if not args.no_eval and result.get("status") == "completed" and openai_api_key:
                    eval_result = await runner.run_evaluation(target_case)
                    result["evaluation"] = eval_result

                # Save result with evaluation data
                await runner.save_centralized_result(target_case, result)

                print(f"\nCase result: {result.get('status')}")
                return 0 if result.get('status') == 'completed' else 1

            finally:
                await agent.teardown()

        else:
            # Run all cases
            summary = await runner.run_all_test_cases(run_evaluation=not args.no_eval)

            print(f"\nOverall success rate: {summary.get('success_rate', 0):.1%}")

            return 0 if summary.get('failed_cases', 0) == 0 else 1

    except KeyError as e:
        print(f"Error: {e}")
        print("\nUse --list-agents to see available agents")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
        sys.exit(1)
