#!/usr/bin/env bash
# Build the unified SciVisAgentBench task-runtime image.
#
#   ./docker/build.sh [IMAGE_NAME]
#
# Default image name: scivis-bench:latest
# Works with docker or podman (set CONTAINER_ENGINE=podman to force podman).
set -euo pipefail

IMAGE_NAME="${1:-scivis-bench:latest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${CONTAINER_ENGINE:-docker}"

echo "Building ${IMAGE_NAME} with ${ENGINE} (linux/amd64)..."
"${ENGINE}" build \
    --platform linux/amd64 \
    -t "${IMAGE_NAME}" \
    -f "${HERE}/Dockerfile" \
    "${HERE}"

echo
echo "Done. Quick smoke test:"
echo "  ${ENGINE} run --rm --platform linux/amd64 ${IMAGE_NAME} \\"
echo "    python -c \"import paraview.simple, napari, vtk, MDAnalysis, gudhi; print('env OK')\""
