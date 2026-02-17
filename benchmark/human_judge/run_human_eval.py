#!/usr/bin/env python3
"""
CLI entry point for human evaluation of SciVisAgentBench results.

Usage:
    # New simplified mode - just specify cases YAML
    python -m benchmark.human_judge.run_human_eval \
        --cases-yaml benchmark/eval_cases/paraview/selected_15_cases.yaml \
        --port 8081

    # Legacy mode - specify all parameters
    python -m benchmark.human_judge.run_human_eval \
        --agent chatvis \
        --config benchmark/configs/chatvis/config_openai.json \
        --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \
        --cases SciVisAgentBench-tasks/chatvis_bench
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.human_judge.app import init_evaluation_multi, init_evaluation, run_server


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Human Evaluation UI for SciVisAgentBench',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplified mode - evaluate using comprehensive YAML
  python -m benchmark.human_judge.run_human_eval \\
      --cases-yaml benchmark/eval_cases/paraview/selected_15_cases.yaml \\
      --port 8081

  # Legacy mode - evaluate ChatVis on chatvis_bench
  python -m benchmark.human_judge.run_human_eval \\
      --agent chatvis \\
      --config benchmark/configs/chatvis/config_openai.json \\
      --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \\
      --cases SciVisAgentBench-tasks/chatvis_bench

  # Legacy mode - evaluate ParaView-MCP on main benchmark
  python -m benchmark.human_judge.run_human_eval \\
      --agent paraview_mcp \\
      --config benchmark/configs/paraview_mcp/config_openai.json \\
      --yaml benchmark/eval_cases/paraview/main_cases.yaml \\
      --cases SciVisAgentBench-tasks/main
        """
    )

    parser.add_argument(
        '--cases-yaml',
        type=str,
        default=None,
        help='Path to YAML file with complete case definitions (name, path, yaml, description). '
             'When using this mode, --agent, --config, --yaml, and --cases are not needed.'
    )

    parser.add_argument(
        '--agent',
        type=str,
        default=None,
        help='[Legacy mode] Name of the agent being evaluated (e.g., chatvis, paraview_mcp)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='[Legacy mode] Path to agent configuration file (JSON)'
    )

    parser.add_argument(
        '--yaml',
        type=str,
        default=None,
        help='[Legacy mode] Path to YAML file containing test cases'
    )

    parser.add_argument(
        '--cases',
        type=str,
        default=None,
        help='[Legacy mode] Path to directory containing test case data'
    )

    parser.add_argument(
        '--filter-cases',
        type=str,
        default=None,
        help='[Legacy mode] Path to YAML file containing list of case names to include'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save evaluation results (default: benchmark/human_judge/evaluations)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to run the server on (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run the server on (default: 5000)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run server in debug mode'
    )

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    errors = []

    # Check if using new mode or legacy mode
    if args.cases_yaml:
        # New mode: only need cases-yaml
        cases_yaml_path = Path(args.cases_yaml)
        if not cases_yaml_path.exists():
            errors.append(f"Cases YAML file not found: {args.cases_yaml}")
    else:
        # Legacy mode: need agent, config, yaml, and cases
        if not args.agent:
            errors.append("--agent is required in legacy mode")
        if not args.config:
            errors.append("--config is required in legacy mode")
        if not args.yaml:
            errors.append("--yaml is required in legacy mode")
        if not args.cases:
            errors.append("--cases is required in legacy mode")

        if args.yaml:
            yaml_path = Path(args.yaml)
            if not yaml_path.exists():
                errors.append(f"YAML file not found: {args.yaml}")

        if args.filter_cases:
            filter_path = Path(args.filter_cases)
            if not filter_path.exists():
                errors.append(f"Filter cases file not found: {args.filter_cases}")

        if args.cases:
            cases_dir = Path(args.cases)
            if not cases_dir.exists():
                errors.append(f"Cases directory not found: {args.cases}")
            elif not cases_dir.is_dir():
                errors.append(f"Cases path is not a directory: {args.cases}")

        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                errors.append(f"Config file not found: {args.config}")

    if errors:
        print("Validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("\nUsage: python -m benchmark.human_judge.run_human_eval --help", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    validate_args(args)

    print("Initializing human evaluation session...")

    # Initialize evaluation
    try:
        if args.cases_yaml:
            # New simplified mode
            print(f"  Mode: Comprehensive YAML")
            print(f"  Cases YAML: {args.cases_yaml}")
            print()

            init_evaluation_multi(
                cases_yaml_path=args.cases_yaml,
                output_dir=args.output_dir
            )
        else:
            # Legacy mode
            print(f"  Mode: Legacy")
            print(f"  Agent: {args.agent}")
            print(f"  Config: {args.config}")
            print(f"  YAML: {args.yaml}")
            print(f"  Cases: {args.cases}")
            print()

            init_evaluation(
                yaml_path=args.yaml,
                cases_dir=args.cases,
                agent_name=args.agent,
                config_path=args.config,
                output_dir=args.output_dir,
                filter_cases_path=args.filter_cases
            )
    except Exception as e:
        print(f"Error initializing evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run server
    try:
        run_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"Error running server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
