#!/usr/bin/env python3
"""
Upload images to Firebase Storage for web deployment.

This script reads the exported cases.json file and uploads all referenced
images to Firebase Storage, maintaining the directory structure.

Prerequisites:
    pip install firebase-admin

Setup:
    1. Go to Firebase Console → Project Settings → Service Accounts
    2. Click "Generate new private key"
    3. Save the JSON file as "serviceAccountKey.json" in this directory

Usage:
    python upload_images_to_storage.py \
        --cases-json firebase_deploy/data/cases.json \
        --service-account serviceAccountKey.json \
        --workspace-root /path/to/SciVisAgentBench
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import firebase_admin
    from firebase_admin import credentials, storage
except ImportError:
    print("Error: firebase-admin not installed")
    print("Install with: pip install firebase-admin")
    sys.exit(1)


def upload_image_to_storage(local_path: Path, storage_path: str, bucket):
    """Upload a single image to Firebase Storage."""

    if not local_path.exists():
        print(f"  ⚠️  File not found: {local_path}")
        return False

    try:
        blob = bucket.blob(f"images/{storage_path}")
        blob.upload_from_filename(str(local_path))

        # Make publicly accessible
        blob.make_public()

        print(f"  ✓ Uploaded: {storage_path}")
        return True

    except Exception as e:
        print(f"  ❌ Failed to upload {storage_path}: {e}")
        return False


def upload_images(cases_json_path: str, service_account_path: str, workspace_root: str):
    """Upload all images from cases.json to Firebase Storage."""

    # Initialize Firebase Admin SDK
    print("Initializing Firebase Admin SDK...")
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': None  # Will be auto-detected from credentials
    })

    bucket = storage.bucket()
    print(f"Connected to Firebase Storage bucket: {bucket.name}\n")

    # Load cases.json
    with open(cases_json_path, 'r') as f:
        data = json.load(f)

    cases = data.get('cases', [])
    workspace = Path(workspace_root).resolve()

    # Collect all unique image paths
    all_images = set()

    for case in cases:
        for img_path in case.get('ground_truth_images', []):
            all_images.add(img_path)
        for img_path in case.get('result_images', []):
            all_images.add(img_path)

    print(f"Found {len(all_images)} unique images to upload\n")

    # Upload each image
    uploaded_count = 0
    failed_count = 0

    for img_path in sorted(all_images):
        local_path = workspace / img_path
        success = upload_image_to_storage(local_path, img_path, bucket)

        if success:
            uploaded_count += 1
        else:
            failed_count += 1

    print(f"\n{'='*60}")
    print(f"Upload Summary:")
    print(f"  ✓ Uploaded: {uploaded_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  Total: {len(all_images)}")
    print(f"{'='*60}")

    # Update cases.json with Firebase Storage URLs
    print("\nUpdating cases.json with Firebase Storage URLs...")
    update_cases_with_storage_urls(cases_json_path, bucket)

    print("\n✅ Image upload completed!")


def update_cases_with_storage_urls(cases_json_path: str, bucket):
    """Update cases.json to use Firebase Storage URLs instead of local paths."""

    with open(cases_json_path, 'r') as f:
        data = json.load(f)

    # For each case, convert paths to Firebase Storage URLs
    for case in data.get('cases', []):
        # Update ground truth images
        gt_images = case.get('ground_truth_images', [])
        case['ground_truth_images'] = [
            f"https://storage.googleapis.com/{bucket.name}/images/{img_path}"
            for img_path in gt_images
        ]

        # Update result images
        result_images = case.get('result_images', [])
        case['result_images'] = [
            f"https://storage.googleapis.com/{bucket.name}/images/{img_path}"
            for img_path in result_images
        ]

    # Save updated JSON
    output_path = str(cases_json_path).replace('.json', '_firebase.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  ✓ Updated JSON saved to: {output_path}")
    print(f"  → Replace cases.json with this file before deploying")


def main():
    parser = argparse.ArgumentParser(
        description='Upload images to Firebase Storage'
    )

    parser.add_argument(
        '--cases-json',
        type=str,
        required=True,
        help='Path to exported cases.json file'
    )

    parser.add_argument(
        '--service-account',
        type=str,
        required=True,
        help='Path to Firebase service account key JSON file'
    )

    parser.add_argument(
        '--workspace-root',
        type=str,
        default='.',
        help='Workspace root directory (default: current directory)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.cases_json).exists():
        print(f"Error: cases.json not found: {args.cases_json}")
        sys.exit(1)

    if not Path(args.service_account).exists():
        print(f"Error: Service account key not found: {args.service_account}")
        print("\nTo get a service account key:")
        print("1. Go to Firebase Console → Project Settings → Service Accounts")
        print("2. Click 'Generate new private key'")
        print("3. Save the JSON file and provide its path")
        sys.exit(1)

    try:
        upload_images(args.cases_json, args.service_account, args.workspace_root)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
