#!/usr/bin/env python3
"""
Generate static HTML site from Flask app for Firebase deployment.

This script runs the Flask app internally, generates static HTML with embedded data,
and outputs files ready for Firebase deployment.

Usage:
    python -m benchmark.human_judge.generate_static_site \
        --cases-yaml benchmark/eval_cases/selected_15_cases.yaml \
        --output-dir benchmark/human_judge/firebase_deploy
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.human_judge.multi_case_loader import MultiCaseLoader


def generate_static_site(cases_yaml_path: str, output_dir: str):
    """Generate static HTML site from evaluation data."""

    output_path = Path(output_dir)
    print(f"Generating static site in: {output_path}")

    # Load evaluation data
    print(f"\nLoading cases from: {cases_yaml_path}")
    loader = MultiCaseLoader(cases_yaml_path)
    evaluation_data = loader.get_evaluation_data()
    metadata = loader.get_metadata()
    workspace_root = loader.workspace_root

    print(f"✓ Loaded {len(evaluation_data)} cases")

    # Prepare output directory structure
    (output_path / 'data').mkdir(parents=True, exist_ok=True)
    (output_path / 'images').mkdir(parents=True, exist_ok=True)

    # Generate cases.json
    cases_json = {
        "metadata": {
            "mode": metadata["mode"],
            "cases_yaml": str(metadata["cases_yaml"]),
            "total_cases": metadata["total_cases"],
            "unique_yamls": metadata["unique_yamls"]
        },
        "cases": []
    }

    # Copy images and build cases data
    print(f"\nProcessing cases and copying images...")
    all_images = set()

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
        cases_json["cases"].append(json_case)

        # Collect all image paths
        for img_path in case_data.get("ground_truth_images", []):
            all_images.add(img_path)
        for img_path in case_data.get("result_images", []):
            all_images.add(img_path)

    # Copy images to output directory
    print(f"Copying {len(all_images)} images/videos...")
    copied_count = 0
    failed_count = 0

    for img_path in sorted(all_images):
        src_path = workspace_root / img_path
        dst_path = output_path / 'images' / img_path

        if not src_path.exists():
            print(f"  ⚠️  Not found: {img_path}")
            failed_count += 1
            continue

        # Create parent directories
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src_path, dst_path)
        copied_count += 1

        if copied_count % 10 == 0:
            print(f"  Copied {copied_count}/{len(all_images)}...")

    print(f"✓ Copied {copied_count} images ({failed_count} failed)")

    # Write cases.json
    cases_json_path = output_path / 'data' / 'cases.json'
    with open(cases_json_path, 'w') as f:
        json.dump(cases_json, f, indent=2)
    print(f"✓ Generated: data/cases.json")

    # Check if Firebase frontend files exist
    required_files = [
        'index.html',
        'css/styles.css',
        'js/firebase-config.js',
        'js/app.js',
        'js/evaluation.js'
    ]

    print(f"\nChecking Firebase frontend files...")
    missing_files = []
    for file_path in required_files:
        full_path = output_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"⚠️  Missing Firebase frontend files:")
        for file_path in missing_files:
            print(f"    - {file_path}")
        print(f"\nThese should already exist in {output_path}")
        print(f"If not, they were created earlier in firebase_deploy/")
    else:
        print(f"✓ All required files present")

    # Summary
    print(f"\n{'='*60}")
    print(f"Static Site Generation Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"  - Cases: {len(cases_json['cases'])}")
    print(f"  - Images: {copied_count}")
    print(f"  - Data file: data/cases.json")
    print(f"\nNext steps:")
    print(f"1. Update js/firebase-config.js with your Firebase credentials")
    print(f"2. Deploy: cd {output_path} && firebase deploy")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate static HTML site for Firebase deployment'
    )

    parser.add_argument(
        '--cases-yaml',
        type=str,
        required=True,
        help='Path to YAML file with complete case definitions'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark/human_judge/firebase_deploy',
        help='Output directory for static site (default: benchmark/human_judge/firebase_deploy)'
    )

    args = parser.parse_args()

    try:
        generate_static_site(args.cases_yaml, args.output_dir)
        print("✅ Static site generation completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
