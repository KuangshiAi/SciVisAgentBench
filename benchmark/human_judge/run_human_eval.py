#!/usr/bin/env python3
"""
CLI entry point for human evaluation of SciVisAgentBench results.

Usage:
    python -m benchmark.human_judge.run_human_eval \
        --agent chatvis \
        --config benchmark/configs/chatvis/config_openai.json \
        --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \
        --cases SciVisAgentBench-tasks/chatvis_bench

    python -m benchmark.human_judge.run_human_eval \
        --agent paraview_mcp \
        --config benchmark/configs/paraview_mcp/config_openai.json \
        --yaml benchmark/eval_cases/paraview/main_cases.yaml \
        --cases SciVisAgentBench-tasks/main
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.human_judge.app import init_evaluation, run_server


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Human Evaluation UI for SciVisAgentBench',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate ChatVis on chatvis_bench
  python -m benchmark.human_judge.run_human_eval \\
      --agent chatvis \\
      --config benchmark/configs/chatvis/config_openai.json \\
      --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \\
      --cases SciVisAgentBench-tasks/chatvis_bench

  # Evaluate ParaView-MCP on main benchmark
  python -m benchmark.human_judge.run_human_eval \\
      --agent paraview_mcp \\
      --config benchmark/configs/paraview_mcp/config_openai.json \\
      --yaml benchmark/eval_cases/paraview/main_cases.yaml \\
      --cases SciVisAgentBench-tasks/main

  # Custom output directory
  python -m benchmark.human_judge.run_human_eval \\
      --agent chatvis \\
      --config benchmark/configs/chatvis/config_openai.json \\
      --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \\
      --cases SciVisAgentBench-tasks/chatvis_bench \\
      --output-dir my_evaluations
        """
    )

    parser.add_argument(
        '--agent',
        type=str,
        required=True,
        help='Name of the agent being evaluated (e.g., chatvis, paraview_mcp, napari_mcp)'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to agent configuration file (JSON)'
    )

    parser.add_argument(
        '--yaml',
        type=str,
        required=True,
        help='Path to YAML file containing test cases'
    )

    parser.add_argument(
        '--cases',
        type=str,
        required=True,
        help='Path to directory containing test case data (e.g., SciVisAgentBench-tasks/main)'
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

    # Check YAML file exists
    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        errors.append(f"YAML file not found: {args.yaml}")

    # Check cases directory exists
    cases_dir = Path(args.cases)
    if not cases_dir.exists():
        errors.append(f"Cases directory not found: {args.cases}")
    elif not cases_dir.is_dir():
        errors.append(f"Cases path is not a directory: {args.cases}")

    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        errors.append(f"Config file not found: {args.config}")

    # Check benchmark type
    benchmark_type = cases_dir.name
    if benchmark_type not in ['main', 'chatvis_bench']:
        errors.append(
            f"Unsupported benchmark type: {benchmark_type}. "
            f"Only 'main' and 'chatvis_bench' are supported for human evaluation."
        )

    if errors:
        print("Validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    validate_args(args)

    print("Initializing human evaluation session...")
    print(f"  Agent: {args.agent}")
    print(f"  Config: {args.config}")
    print(f"  YAML: {args.yaml}")
    print(f"  Cases: {args.cases}")
    print()

    # Initialize evaluation
    try:
        init_evaluation(
            yaml_path=args.yaml,
            cases_dir=args.cases,
            agent_name=args.agent,
            config_path=args.config,
            output_dir=args.output_dir
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
