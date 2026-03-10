#!/bin/bash
# Build x86_64 image using Podman on Mac M2

set -e

echo "Building x86_64 image using Podman on ARM64 Mac..."
echo "This will use Podman's emulation via QEMU"

podman build \
    --platform linux/amd64 \
    -t viz-agent:x86 \
    -f Dockerfile \
    .

echo ""
echo "Build complete! To run the container:"
echo "  podman run --platform linux/amd64 -it --rm viz-agent:x86"
echo ""
echo "To verify the architecture inside the container:"
echo "  podman run --platform linux/amd64 --rm viz-agent:x86 uname -m"
