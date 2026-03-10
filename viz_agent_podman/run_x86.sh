#!/bin/bash
# Run x86_64 container using Podman on Mac M2

set -e

IMAGE_NAME=${1:-viz-agent:x86}

echo "Running x86_64 container: $IMAGE_NAME"
echo "Architecture inside container will be x86_64"

podman run \
    --platform linux/amd64 \
    -it \
    --rm \
    -v "$(pwd)":/workspace \
    "$IMAGE_NAME" \
    /bin/bash
