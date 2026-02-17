#!/usr/bin/env python3
"""
Export evaluation cases to JSON format for Firebase deployment.

This script reads the selected cases YAML file and exports all case data
(including task descriptions, metrics, and image paths) to a JSON file
that can be loaded by the Firebase web app.

Usage:
    python export_cases_to_json.py \
        --cases-yaml benchmark/eval_cases/selected_15_cases.yaml \
        --output firebase_deploy/data/cases.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.human_judge.multi_case_loader import MultiCaseLoader


def export_cases_to_json(cases_yaml_path: str, output_path: str):
    """Export evaluation cases to JSON format."""

    print(f"Loading cases from: {cases_yaml_path}")

    # Load cases using MultiCaseLoader
    loader = MultiCaseLoader(cases_yaml_path)
    evaluation_data = loader.get_evaluation_data()
    metadata = loader.get_metadata()

    print(f"Loaded {len(evaluation_data)} cases")

    # Prepare JSON structure
    json_data = {
        "metadata": {
            "mode": metadata["mode"],
            "cases_yaml": str(metadata["cases_yaml"]),
            "total_cases": metadata["total_cases"],
            "unique_yamls": metadata["unique_yamls"],
            "exported_at": Path(__file__).parent.parent.parent.name
        },
        "cases": []
    }

    # Add each case
    for case_data in evaluation_data:
        json_case = {
            "index": case_data["case_index"],
            "name": case_data["case_name"],
            "description": case_data.get("description", ""),
            "task_description": case_data.get("task_description", ""),
            "benchmark_type": case_data.get("benchmark_type", "main"),
            "metrics": case_data.get("metrics", []),
            "ground_truth_images": case_data.get("ground_truth_images", []),
            "result_images": case_data.get("result_images", []),
            "has_results": case_data.get("has_results", False)
        }
        json_data["cases"].append(json_case)

    # Write JSON file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✓ Exported {len(json_data['cases'])} cases to: {output_path}")
    print(f"  - Total cases: {json_data['metadata']['total_cases']}")
    print(f"  - Cases with results: {sum(1 for c in json_data['cases'] if c['has_results'])}")
    print(f"  - Ground truth images: {sum(len(c['ground_truth_images']) for c in json_data['cases'])}")
    print(f"  - Result images: {sum(len(c['result_images']) for c in json_data['cases'])}")

    return json_data


def main():
    parser = argparse.ArgumentParser(
        description='Export evaluation cases to JSON for Firebase deployment'
    )

    parser.add_argument(
        '--cases-yaml',
        type=str,
        required=True,
        help='Path to YAML file with complete case definitions'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='firebase_deploy/data/cases.json',
        help='Output JSON file path (default: firebase_deploy/data/cases.json)'
    )

    args = parser.parse_args()

    try:
        export_cases_to_json(args.cases_yaml, args.output)
        print("\n✅ Export completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
