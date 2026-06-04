#!/usr/bin/env bash
# Container entrypoint for the SciVisAgentBench task-runtime image.
#   1. ensures a writable HOME (so arbitrary --user UIDs work),
#   2. starts a virtual X display for headless OpenGL / Qt rendering,
#   3. activates the `scivis_bench` conda env,
#   4. execs whatever command was passed (default: bash).
set -euo pipefail

# --- writable HOME (matters when run with `--user $(id -u):$(id -g)`) --------
if [ -z "${HOME:-}" ] || [ ! -w "${HOME:-/nonexistent}" ]; then
    export HOME=/tmp
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
mkdir -p "$XDG_CACHE_HOME" 2>/dev/null || true

# --- headless display ---------------------------------------------------------
export DISPLAY="${DISPLAY:-:99}"
if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    mkdir -p /tmp/.X11-unix 2>/dev/null || true
    Xvfb "$DISPLAY" -screen 0 1920x1080x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
    # Wait (up to ~10s) until the server advertises GLX, not just until it
    # accepts a connection — a bare connection can succeed a beat before GLX is
    # ready, which makes the very first OpenGL render fail ("bad X server
    # connection"). Gating on GLX removes that cold-start race.
    for _ in $(seq 1 100); do
        xdpyinfo -display "$DISPLAY" 2>/dev/null | grep -q GLX && break
        sleep 0.1
    done
fi

# --- conda env ----------------------------------------------------------------
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate scivis_bench

exec "$@"
