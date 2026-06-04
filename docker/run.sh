#!/usr/bin/env bash
# Stage a sanitized workspace and drop into the task-runtime container.
#
# Usage:
#   ./docker/run.sh --cases-dir SciVisAgentBench-tasks/paraview [options] [-- CMD...]
#
# Options:
#   --cases-dir DIR   Source cases dir to stage (required).
#   --sandbox DIR     Sandbox location (default: ./sandbox/<basename of cases-dir>).
#   --cases "a b c"   Optional subset of case names to stage.
#   --image NAME      Image to run (default: scivis-bench:latest).
#   --no-stage        Reuse an existing sandbox instead of re-staging.
#   --root            Run as root in the container (default: your host UID:GID,
#                     so files written to the sandbox stay owned by you).
#   -- CMD...         Command to run in the container (default: interactive bash).
#
# The sandbox is mounted read-write at /workspace and contains NO ground truth.
# Bring your own agent: drop into bash and run it, or pass it as CMD. API keys
# in your shell (ANTHROPIC_API_KEY, OPENAI_API_KEY) are forwarded automatically.
#
# After the run, copy outputs back and evaluate on the host:
#   python docker/collect_results.py --sandbox <sandbox> \
#       --cases-dir <cases-dir> --agent-mode <mode>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
ENGINE="${CONTAINER_ENGINE:-docker}"

CASES_DIR=""
SANDBOX=""
CASES=""
IMAGE="scivis-bench:latest"
DO_STAGE=1
USER_FLAG="--user $(id -u):$(id -g)"
CMD=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cases-dir) CASES_DIR="$2"; shift 2 ;;
        --sandbox)   SANDBOX="$2"; shift 2 ;;
        --cases)     CASES="$2"; shift 2 ;;
        --image)     IMAGE="$2"; shift 2 ;;
        --no-stage)  DO_STAGE=0; shift ;;
        --root)      USER_FLAG=""; shift ;;
        --)          shift; CMD=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${CASES_DIR}" ]]; then
    echo "ERROR: --cases-dir is required." >&2
    exit 2
fi
if [[ -z "${SANDBOX}" ]]; then
    SANDBOX="${REPO_ROOT}/sandbox/$(basename "${CASES_DIR}")"
fi

if [[ "${DO_STAGE}" -eq 1 ]]; then
    STAGE_ARGS=(--cases-dir "${CASES_DIR}" --out "${SANDBOX}" --force)
    if [[ -n "${CASES}" ]]; then
        # shellcheck disable=SC2206
        STAGE_ARGS+=(--cases ${CASES})
    fi
    echo "Staging sanitized workspace..."
    python3 "${HERE}/stage_workspace.py" "${STAGE_ARGS[@]}"
fi

SANDBOX_ABS="$(cd "${SANDBOX}" && pwd)"
if [[ ${#CMD[@]} -eq 0 ]]; then
    CMD=(bash)
    TTY_FLAGS="-it"
else
    TTY_FLAGS="$([[ -t 0 ]] && echo -it || echo -i)"
fi

echo "Launching ${IMAGE} (workspace: ${SANDBOX_ABS})..."
# shellcheck disable=SC2086
exec "${ENGINE}" run --rm ${TTY_FLAGS} --platform linux/amd64 ${USER_FLAG} \
    -v "${SANDBOX_ABS}:/workspace" \
    -e ANTHROPIC_API_KEY -e OPENAI_API_KEY -e OPENAI_BASE_URL \
    "${IMAGE}" "${CMD[@]}"
