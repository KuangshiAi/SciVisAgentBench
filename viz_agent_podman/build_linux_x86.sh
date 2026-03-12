#!/bin/bash
# Build x86_64 image using Podman on Linux x86_64

set -e

echo "Building x86_64 image using Podman on Linux x86_64..."
echo "Using single user namespace mapping (no subuid/subgid required)..."

# Remove problematic --platform flag from Dockerfile first
sed 's/FROM --platform=linux\/amd64 /FROM /' Dockerfile.x86 > Dockerfile.x86.tmp

podman build \
    --format docker \
    --tls-verify=false \
    --ulimit nofile=1024:1024 \
    --memory=8g \
    --memory-swap=12g \
    -t viz-agent:x86 \
    -f Dockerfile.x86.tmp \
    .

# Clean up temp file
rm -f Dockerfile.x86.tmp

echo ""
echo "Build complete! To run the container:"
echo "  podman run -it --rm viz-agent:x86"
echo ""
echo "To verify the architecture inside the container:"
echo "  podman run --rm viz-agent:x86 uname -m"
echo ""
