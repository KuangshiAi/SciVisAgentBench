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

4. Run agent execution only without evaluation:
   python -m benchmark.evaluation_framework.run_evaluation \\
       --agent paraview_mcp \\
       --config benchmark/configs/paraview_mcp/config_openai.json \\
       --yaml SciVisAgentBench-tasks/main/main_cases.yaml \\
       --cases SciVisAgentBench-tasks/main \\
       --exe-only

6. Resume evaluation from a specific case (useful after failures):
   python -m benchmark.evaluation_framework.run_evaluation \\
       --agent paraview_mcp \\
       --config benchmark/configs/paraview_mcp/config_openai.json \\
       --yaml SciVisAgentBench-tasks/main/main_cases.yaml \\
       --cases SciVisAgentBench-tasks/main \\
       --start-from engine

7. List available agents:
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
    ChatVisAgent,
    TopoPilotMCPAgent
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

    parser.add_argument(
        "--exe-only",
        action="store_true",
        help="Skip evaluation and only run agent execution (execution-only mode)"
    )

    # API keys
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (can also use OPENAI_API_KEY env var)"
    )

    parser.add_argument(
        "--anthropic-api-key",
        help="Anthropic API key (can also use ANTHROPIC_API_KEY env var)"
    )

    parser.add_argument(
        "--openai-base-url",
        help="Custom OpenAI-compatible API endpoint. Can also use OPENAI_BASE_URL env var."
    )

    # Specific case
    parser.add_argument(
        "--case",
        nargs='+',
        help="Run only specific test case(s) by name. Can specify multiple cases separated by spaces."
    )

    parser.add_argument(
        "--start-from",
        help="Start evaluation from this case and continue with all subsequent cases (cannot be used with --case)"
    )

    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available test cases and exit"
    )

    # Experiment tracking
    parser.add_argument(
        "--experiment-number",
        "--exp",
        default="exp_default",
        help="Experiment number for tracking multiple runs (e.g., exp1, exp2, trial_3). Default: exp_default"
    )

    parser.add_argument(
        "--agent-mode",
        help="For eval-only mode: specify the full agent_mode string (e.g., 'paraview_mcp_claude-sonnet-4-5_exp1'). If not provided, will be constructed from agent name, model, and experiment number."
    )

    # Verbose mode
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show agent output in real-time (for Claude Code agent)"
    )

    return parser.parse_args()


def clear_case_results(cases_dir: str, case_names = None):
    """
    Clear agent execution results from test case directories.

    Args:
        cases_dir: Path to the cases directory
        case_names: Optional specific case name(s). Can be a string, list of strings, or None for all cases.
    """
    cases_path = Path(cases_dir)

    if not cases_path.exists():
        print(f"Warning: Cases directory not found: {cases_dir}")
        return

    # Normalize case_names to a list
    if case_names is None:
        case_names_list = None
    elif isinstance(case_names, str):
        case_names_list = [case_names]
    else:
        case_names_list = case_names

    # Directories to remove from each case
    dirs_to_remove = ["results", "evaluation_results", "test_results"]

    # Get list of cases to clear
    # For bioimage_data, we need to handle nested structure: bioimage_data/eval_xxx/case_1
    cases_to_clear = []

    if case_names_list:
        # Specific cases - need to find them in nested structure
        for case_name in case_names_list:
            for subdir in cases_path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    case_path = subdir / case_name
                    if case_path.exists():
                        cases_to_clear.append((subdir.name, case_name))
    else:
        # All cases - find all nested cases
        for subdir in cases_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                # Check if this is a YAML-named directory with case_X subdirectories
                has_case_subdirs = any(
                    d.name.startswith('case_') for d in subdir.iterdir()
                    if d.is_dir()
                )
                if has_case_subdirs:
                    # This is a YAML-named directory with cases
                    for case_subdir in subdir.iterdir():
                        if case_subdir.is_dir() and not case_subdir.name.startswith('.'):
                            cases_to_clear.append((subdir.name, case_subdir.name))
                else:
                    # This is a direct case directory
                    cases_to_clear.append(('', subdir.name))

    cleared_count = 0
    cleared_files_count = 0

    for yaml_dir, case in cases_to_clear:
        if yaml_dir:
            # Nested structure: bioimage_data/eval_xxx/case_1
            case_dir = cases_path / yaml_dir / case
            parent_dir = cases_path / yaml_dir
        else:
            # Direct structure
            case_dir = cases_path / case
            parent_dir = None

        if not case_dir.exists():
            print(f"Warning: Case directory not found: {case_dir}")
            continue

        # Clear subdirectories within the case
        for dir_name in dirs_to_remove:
            dir_path = case_dir / dir_name
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    display_path = f"{yaml_dir}/{case}/{dir_name}" if yaml_dir else f"{case}/{dir_name}"
                    print(f"  ✓ Cleared {display_path}/")
                    cleared_count += 1
                except Exception as e:
                    print(f"  ✗ Error clearing {dir_path}: {e}")

        # Clear .png and .txt files from parent directory (for bioimage_data structure)
        if parent_dir and parent_dir.exists():
            for pattern in ['*.png', '*.txt']:
                for file_path in parent_dir.glob(pattern):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            print(f"  ✓ Cleared {yaml_dir}/{file_path.name}")
                            cleared_files_count += 1
                        except Exception as e:
                            print(f"  ✗ Error clearing {file_path}: {e}")

    if cleared_count > 0 or cleared_files_count > 0:
        print(f"\n✓ Cleared {cleared_count} directories and {cleared_files_count} files")
    else:
        print("\nℹ No result directories or files found to clear")


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

    # Validate that --exe-only and --eval-only are mutually exclusive
    if args.exe_only and args.eval_only:
        print("Error: --exe-only and --eval-only cannot be used together")
        print("  Use --exe-only to run agent execution without evaluation")
        print("  Use --eval-only to run evaluation without agent execution")
        return 1

    # Get API keys
    openai_api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    anthropic_api_key = args.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')

    # Check if evaluation model requires specific API key
    if not args.no_eval and not args.exe_only:
        eval_model_lower = args.eval_model.lower()
        if 'claude' in eval_model_lower or 'anthropic' in eval_model_lower:
            if not anthropic_api_key:
                print("Warning: No Anthropic API key provided. Evaluation will be skipped.")
                print("Set ANTHROPIC_API_KEY environment variable or use --anthropic-api-key to enable evaluation.")
        else:
            if not openai_api_key:
                print("Warning: No OpenAI API key provided. Evaluation will be skipped.")
                print("Set OPENAI_API_KEY environment variable or use --openai-api-key to enable evaluation.")

    try:
        # Get agent class
        agent_class = get_agent(args.agent)

        # Load configuration
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)

        # Add experiment_number to config
        config["experiment_number"] = args.experiment_number

        # Add verbose flag to config if provided
        if args.verbose:
            config["verbose"] = True

        # Create agent instance
        print(f"\nCreating agent: {args.agent}")
        print(f"Experiment number: {args.experiment_number}")
        agent = agent_class(config)
        print(f"Agent mode: {agent.agent_mode}")

        # Determine which API key to pass based on eval model
        eval_model_lower = args.eval_model.lower()
        eval_api_key = anthropic_api_key if ('claude' in eval_model_lower or 'anthropic' in eval_model_lower) else openai_api_key

        # Create test runner (pass config for rate limiting)
        runner = UnifiedTestRunner(
            agent=agent,
            yaml_path=args.yaml,
            cases_dir=args.cases,
            data_dir=args.data_dir,
            output_dir=args.output,
            openai_api_key=eval_api_key,  # Pass the appropriate API key for the eval model
            eval_model=args.eval_model,
            static_screenshot=args.static_screenshot,
            config=config,
            openai_base_url=args.openai_base_url
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

        # Validate that --case and --start-from are mutually exclusive
        if args.case and args.start_from:
            print("Error: --case and --start-from cannot be used together")
            print("  Use --case to run specific case(s)")
            print("  Use --start-from to resume from a case and run all subsequent cases")
            return 1

        # Handle --clear-results flag
        if args.clear_results:
            print("\n🗑️  Clearing previous results...")
            clear_case_results(args.cases, args.case)
            print()

        # Handle --exe-only flag
        if args.exe_only:
            print("\n⚙️  Execution-only mode: Skipping evaluation\n")

        # Handle --eval-only flag
        elif args.eval_only:
            print("\n📊 Evaluation-only mode: Skipping agent execution\n")
            if not eval_api_key:
                eval_model_lower = args.eval_model.lower()
                if 'claude' in eval_model_lower or 'anthropic' in eval_model_lower:
                    print("Error: --eval-only requires an Anthropic API key for evaluation")
                    print("Set ANTHROPIC_API_KEY environment variable or use --anthropic-api-key")
                else:
                    print("Error: --eval-only requires an OpenAI API key for evaluation")
                    print("Set OPENAI_API_KEY environment variable or use --openai-api-key")
                return 1

            # For eval-only mode, use agent_mode if provided, otherwise construct it
            if args.agent_mode:
                eval_agent_mode = args.agent_mode
                print(f"Using provided agent_mode: {eval_agent_mode}")
            else:
                eval_agent_mode = agent.agent_mode
                print(f"Using constructed agent_mode: {eval_agent_mode}")

            # Run evaluation on all cases
            if args.case:
                # Handle multiple cases
                case_names = args.case if isinstance(args.case, list) else [args.case]

                for case_name in case_names:
                    # Find specific case
                    target_case = None
                    for case in test_cases:
                        if case.case_name == case_name:
                            target_case = case
                            break

                    if not target_case:
                        print(f"Error: Test case '{case_name}' not found")
                        print("Available cases:", [case.case_name for case in test_cases])
                        continue

                    print(f"\nEvaluating existing results for: {case_name}")
                    eval_result = await runner.run_evaluation(target_case)

                    # Load previous test result if it exists
                    previous_result = runner.load_latest_result(target_case)

                    if previous_result:
                        # Merge evaluation with existing test data
                        result = previous_result.copy()
                        result["evaluation"] = eval_result
                        # Update status to indicate eval-only run
                        result["status"] = "eval_only"
                        # Update agent_mode if provided
                        if args.agent_mode:
                            result["agent_mode"] = eval_agent_mode
                    else:
                        # No previous result found, create minimal result
                        result = {
                            "status": "eval_only",
                            "case_name": case_name,
                            "agent_mode": eval_agent_mode,
                            "evaluation": eval_result
                        }

                    await runner.save_centralized_result(target_case, result)

                    print(f"✓ Evaluation complete for {case_name}")

                print(f"\n✓ All {len(case_names)} case(s) evaluated")
                return 0
            else:
                # Handle --start-from flag for eval-only mode
                cases_to_eval = test_cases
                if args.start_from:
                    # Find the index of the start case
                    start_index = None
                    for i, case in enumerate(test_cases):
                        if case.case_name == args.start_from:
                            start_index = i
                            break

                    if start_index is None:
                        print(f"Error: Start case '{args.start_from}' not found")
                        print("Available cases:", [case.case_name for case in test_cases])
                        return 1

                    # Slice the list to start from the specified case
                    cases_to_eval = test_cases[start_index:]
                    print(f"\n▶️  Starting evaluation from case: {args.start_from}")
                    print(f"   Evaluating {len(cases_to_eval)} of {len(test_cases)} total cases")
                    print(f"   Cases to evaluate: {[c.case_name for c in cases_to_eval]}")

                # Evaluate all cases (or subset if --start-from was used)
                print(f"Evaluating existing results for {len(cases_to_eval)} test cases...")
                for test_case in cases_to_eval:
                    print(f"\n{'='*60}")
                    print(f"Evaluating: {test_case.case_name}")
                    print(f"{'='*60}")

                    try:
                        eval_result = await runner.run_evaluation(test_case)

                        # Load previous test result if it exists
                        previous_result = runner.load_latest_result(test_case)

                        if previous_result:
                            # Merge evaluation with existing test data
                            result = previous_result.copy()
                            result["evaluation"] = eval_result
                            # Update status to indicate eval-only run
                            result["status"] = "eval_only"
                            # Update agent_mode if provided
                            if args.agent_mode:
                                result["agent_mode"] = eval_agent_mode
                        else:
                            # No previous result found, create minimal result
                            result = {
                                "status": "eval_only",
                                "case_name": test_case.case_name,
                                "agent_mode": eval_agent_mode,
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
            # Handle multiple cases
            case_names = args.case if isinstance(args.case, list) else [args.case]

            # Find all target cases
            target_cases = []
            for case_name in case_names:
                target_case = None
                for case in test_cases:
                    if case.case_name == case_name:
                        target_case = case
                        break

                if not target_case:
                    print(f"Error: Test case '{case_name}' not found")
                    print("Available cases:", [case.case_name for case in test_cases])
                else:
                    target_cases.append(target_case)

            if not target_cases:
                print("Error: No valid test cases found")
                return 1

            # Setup agent
            await agent.setup()

            try:
                all_success = True
                for target_case in target_cases:
                    # Run single case without saving (we'll save after evaluation)
                    print(f"\nRunning test case: {target_case.case_name}")
                    result = await runner.run_single_test_case(target_case, save_result=False)

                    # Run evaluation if requested (skip if --exe-only is set)
                    if not args.no_eval and not args.exe_only and result.get("status") == "completed" and eval_api_key:
                        eval_result = await runner.run_evaluation(target_case)
                        result["evaluation"] = eval_result

                    # Save result with evaluation data
                    await runner.save_centralized_result(target_case, result)

                    print(f"Case result: {result.get('status')}")
                    if result.get('status') != 'completed':
                        all_success = False

                print(f"\n✓ Completed {len(target_cases)} case(s)")
                return 0 if all_success else 1

            finally:
                await agent.teardown()

        else:
            # Handle --start-from flag
            cases_to_run = test_cases
            if args.start_from:
                # Find the index of the start case
                start_index = None
                for i, case in enumerate(test_cases):
                    if case.case_name == args.start_from:
                        start_index = i
                        break

                if start_index is None:
                    print(f"Error: Start case '{args.start_from}' not found")
                    print("Available cases:", [case.case_name for case in test_cases])
                    return 1

                # Slice the list to start from the specified case
                cases_to_run = test_cases[start_index:]
                print(f"\n▶️  Starting from case: {args.start_from}")
                print(f"   Running {len(cases_to_run)} of {len(test_cases)} total cases")
                print(f"   Cases to run: {[c.case_name for c in cases_to_run]}")

            # Update runner's test_cases to only include the cases we want to run
            runner.test_cases = cases_to_run

            # Run all cases (or subset if --start-from was used)
            # Skip evaluation if --exe-only is set
            summary = await runner.run_all_test_cases(run_evaluation=not args.no_eval and not args.exe_only)

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
