#!/usr/bin/env bash
# Build the Codex CLI child image on top of scivis-bench:latest.
#
#   ./docker/build_codex.sh [IMAGE_NAME]
#
# Default image name: scivis-bench:codex
# Requires the base image (scivis-bench:latest) to exist first (./docker/build.sh).
set -euo pipefail

IMAGE_NAME="${1:-scivis-bench:codex}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${CONTAINER_ENGINE:-docker}"

if ! "${ENGINE}" image inspect scivis-bench:latest >/dev/null 2>&1; then
    echo "ERROR: base image scivis-bench:latest not found. Build it first:" >&2
    echo "       ./docker/build.sh" >&2
    exit 1
fi

echo "Building ${IMAGE_NAME} with ${ENGINE} (linux/amd64)..."
"${ENGINE}" build \
    --platform linux/amd64 \
    -t "${IMAGE_NAME}" \
    -f "${HERE}/Dockerfile.codex" \
    "${HERE}"

echo
echo "Done. Smoke test:"
echo "  ${ENGINE} run --rm --platform linux/amd64 ${IMAGE_NAME} codex --version"
