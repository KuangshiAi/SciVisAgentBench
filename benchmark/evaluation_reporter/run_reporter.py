"""
Evaluation Reporter for SciVisAgentBench

Generates an interactive HTML report from test and evaluation results.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation_reporter.reporter import EvaluationReporter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and serve evaluation report"
    )

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent name (e.g., chatvis, topopilot)"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to agent configuration file"
    )

    parser.add_argument(
        "--yaml",
        type=str,
        required=True,
        help="Path to YAML test cases file"
    )

    parser.add_argument(
        "--cases",
        type=str,
        required=True,
        help="Path to test cases directory"
    )

    parser.add_argument(
        "--test-results",
        type=str,
        required=True,
        help="Path to test results directory (contains JSON result files)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for generated report (default: test_results/<agent>_report)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve the report on (default: 8080)"
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )

    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Generate static HTML only, don't start server"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Resolve paths
    test_results_dir = Path(args.test_results).resolve()
    cases_dir = Path(args.cases).resolve()
    yaml_path = Path(args.yaml).resolve()
    config_path = Path(args.config).resolve()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = Path("test_results") / f"{args.agent}_report"
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 SciVisAgentBench Evaluation Reporter")
    print(f"{'='*60}")
    print(f"Agent: {args.agent}")
    print(f"Test Results: {test_results_dir}")
    print(f"Cases Directory: {cases_dir}")
    print(f"YAML File: {yaml_path}")
    print(f"Config File: {config_path}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*60}\n")

    # Check if test results directory exists
    if not test_results_dir.exists():
        print(f"❌ Error: Test results directory not found: {test_results_dir}")
        return 1

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1

    # Create reporter
    reporter = EvaluationReporter(
        agent_name=args.agent,
        test_results_dir=test_results_dir,
        cases_dir=cases_dir,
        yaml_path=yaml_path,
        config=config,
        output_dir=output_dir
    )

    # Generate report
    print("🔄 Generating report...")
    try:
        report_path = reporter.generate_report()
        print(f"✅ Report generated: {report_path}")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # If static-only mode, just exit
    if args.static_only:
        print(f"\n💾 Static report saved to: {report_path}")
        print(f"   Open in browser: file://{report_path}")
        return 0

    # Start HTTP server
    os.chdir(output_dir)

    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("localhost", args.port), handler)

    # Start server in background thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{args.port}/report.html"
    print(f"\n🌐 Server started at: {url}")
    print(f"   Press Ctrl+C to stop the server")

    # Open browser
    if not args.no_browser:
        print(f"   Opening browser...")
        webbrowser.open(url)

    try:
        # Keep the server running
        server_thread.join()
    except KeyboardInterrupt:
        print(f"\n\n👋 Shutting down server...")
        httpd.shutdown()
        return 0


if __name__ == "__main__":
    sys.exit(main())
