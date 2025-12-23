"""
Flask web application for human evaluation of visualization results.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluation_loader import EvaluationLoader

app = Flask(__name__)

# Global variables to store evaluation session data
evaluation_data = []
metadata = {}
output_file = None
workspace_root = None


def init_evaluation(yaml_path: str, cases_dir: str, agent_name: str, config_path: str, output_dir: str = None):
    """
    Initialize evaluation session.

    Args:
        yaml_path: Path to YAML test cases
        cases_dir: Directory containing test cases
        agent_name: Name of agent being evaluated
        config_path: Path to agent config
        output_dir: Directory to save evaluation results (optional)
    """
    global evaluation_data, metadata, output_file, workspace_root

    # Load evaluation data
    loader = EvaluationLoader(yaml_path, cases_dir, agent_name, config_path)
    evaluation_data = loader.get_evaluation_data()
    metadata = loader.get_metadata()

    # Set workspace root for serving images (use absolute path)
    workspace_root = str(Path(cases_dir).resolve().parent.parent)

    # Setup output file
    if output_dir is None:
        output_dir = Path("benchmark/human_judge/evaluations")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_type = metadata.get("benchmark_type", "unknown")
    output_file = output_dir / f"human_eval_{agent_name}_{benchmark_type}_{timestamp}.json"

    # Initialize evaluation results file
    save_results({
        "metadata": metadata,
        "timestamp": timestamp,
        "evaluations": []
    })

    print(f"Evaluation session initialized with {len(evaluation_data)} cases")
    print(f"Results will be saved to: {output_file}")


def save_results(data: dict):
    """Save evaluation results to JSON file."""
    global output_file

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_results() -> dict:
    """Load current evaluation results from JSON file."""
    global output_file

    if output_file.exists():
        with open(output_file, 'r') as f:
            return json.load(f)
    else:
        return {
            "metadata": metadata,
            "evaluations": []
        }


@app.route('/')
def index():
    """Render main evaluation page."""
    return render_template('evaluate.html')


@app.route('/api/metadata')
def get_metadata():
    """Get evaluation session metadata."""
    return jsonify(metadata)


@app.route('/api/cases')
def get_cases():
    """Get all evaluation cases."""
    return jsonify({
        "cases": evaluation_data,
        "total": len(evaluation_data)
    })


@app.route('/api/case/<int:case_index>')
def get_case(case_index):
    """Get specific evaluation case."""
    if 0 <= case_index < len(evaluation_data):
        # Load existing evaluations
        results = load_results()
        existing_evals = {e["case_index"]: e for e in results.get("evaluations", [])}

        case_data = evaluation_data[case_index].copy()

        # Add existing evaluation if available
        if case_index in existing_evals:
            case_data["existing_evaluation"] = existing_evals[case_index]

        return jsonify(case_data)
    else:
        return jsonify({"error": "Case not found"}), 404


@app.route('/api/evaluate', methods=['POST'])
def submit_evaluation():
    """Submit human evaluation for a case."""
    data = request.json

    case_index = data.get("case_index")
    ratings = data.get("ratings")  # List of ratings for each metric
    notes = data.get("notes", "")

    if case_index is None or ratings is None:
        return jsonify({"error": "Missing required fields"}), 400

    # Load current results
    results = load_results()

    # Create evaluation record
    evaluation = {
        "case_index": case_index,
        "case_name": evaluation_data[case_index]["case_name"],
        "ratings": ratings,
        "notes": notes,
        "timestamp": datetime.now().isoformat()
    }

    # Update or append evaluation
    existing_evals = results.get("evaluations", [])
    updated = False

    for i, e in enumerate(existing_evals):
        if e["case_index"] == case_index:
            existing_evals[i] = evaluation
            updated = True
            break

    if not updated:
        existing_evals.append(evaluation)

    results["evaluations"] = existing_evals

    # Save results
    save_results(results)

    return jsonify({
        "success": True,
        "message": "Evaluation saved successfully"
    })


@app.route('/api/results')
def get_results():
    """Get all evaluation results."""
    results = load_results()
    return jsonify(results)


@app.route('/api/progress')
def get_progress():
    """Get evaluation progress."""
    results = load_results()
    evaluations = results.get("evaluations", [])

    return jsonify({
        "total_cases": len(evaluation_data),
        "evaluated_cases": len(evaluations),
        "progress_percentage": (len(evaluations) / len(evaluation_data) * 100) if evaluation_data else 0
    })


@app.route('/images/<path:filepath>')
def serve_image(filepath):
    """Serve images from workspace."""
    global workspace_root

    # Security: ensure we're only serving from workspace
    full_path = Path(workspace_root) / filepath

    if not full_path.exists():
        return "Image not found", 404

    return send_from_directory(workspace_root, filepath)


def run_server(host='127.0.0.1', port=5000, debug=False):
    """Run the Flask development server."""
    print(f"\n{'='*60}")
    print(f"Human Evaluation UI")
    print(f"{'='*60}")
    print(f"Agent: {metadata.get('agent_name', 'N/A')}")
    print(f"Benchmark: {metadata.get('benchmark_type', 'N/A')}")
    print(f"Total cases: {len(evaluation_data)}")
    print(f"{'='*60}")
    print(f"\nOpen your browser and navigate to: http://{host}:{port}")
    print(f"Results will be saved to: {output_file}")
    print(f"\nPress Ctrl+C to stop the server\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    # This should not be called directly, use run_human_eval.py instead
    print("Please use run_human_eval.py to start the evaluation server")
