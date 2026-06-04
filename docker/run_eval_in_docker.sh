#!/usr/bin/env bash
# Run an agent's execution flow INSIDE the container, against a sanitized
# (ground-truth-free) copy of the cases, then collect results back to the real
# tree for host-side evaluation. Works for any agent with a benchmark runner
# (claude_code -> scivis-bench:claude, codex_cli -> scivis-bench:codex).
#
# This is the Docker equivalent of running, on the host:
#   python benchmark/run_<agent>_eval.py --config ... --agent <agent> \
#       --yaml ... --cases SciVisAgentBench-tasks/<cat> --exe-only --experiment-number ...
#
# Usage:
#   ./docker/run_eval_in_docker.sh \
#       --config benchmark/configs/claude_code/config.json \
#       --agent claude_code \
#       --yaml benchmark/eval_cases/paraview/paraview_subset_15.yaml \
#       --cases SciVisAgentBench-tasks/paraview \
#       --experiment-number exp1
#
#   # Codex (uses scivis-bench:codex + the codex config automatically):
#   ./docker/run_eval_in_docker.sh --agent codex_cli \
#       --yaml benchmark/eval_cases/paraview/paraview_subset_15.yaml \
#       --cases SciVisAgentBench-tasks/paraview --experiment-number exp1
#
# Options (flag names mirror run_<agent>_eval.py):
#   --cases DIR             Real category dir to sanitize + run on (required).
#                           (--cases-dir is accepted as an alias.)
#   --yaml FILE             YAML test cases (required).
#   --agent NAME            Agent (default: claude_code). Picks the runner
#                           benchmark/run_<agent>_eval.py, and the default config
#                           + image below.
#   --config FILE           Agent config (default: benchmark/configs/<agent>/config.json).
#   --experiment-number STR Experiment id (default: exp1). (--exp is an alias.)
#   --only "a b c"          Subset of case names (stages + runs only these).
#   --image NAME            Image (default: scivis-bench:claude for claude_code,
#                           scivis-bench:codex for codex_cli).
#   --no-stage              Reuse an existing sandbox.
#   --no-collect      Don't copy results back at all (run only).
#   --collect-end     Collect once at the end (batch) instead of the default
#                     per-case collection.
#
# By default each case is collected into the real tree the moment it finishes
# (detected by its centralized test_results JSON appearing), so a mid-run crash
# still leaves every completed case collected. (--collect-each is accepted as a
# no-op alias for this default.)
#
# Notes:
#   * The real SciVisAgentBench-tasks tree is NEVER mounted, so the agent cannot
#     see any GS/ ground truth. Only the sanitized sandbox is mounted, at the
#     same relative path, so the in-container command matches the host command.
#   * Requires ANTHROPIC_API_KEY in your shell (forwarded to the container).
#   * Evaluation is intentionally NOT run here (this mirrors --exe-only). Run it
#     afterward on the host, where GS and the metric stack live.
#   * Per-suite --cases / --yaml (the suite dir name is not always the yaml
#     folder name):
#       paraview     SciVisAgentBench-tasks/paraview       benchmark/eval_cases/paraview/paraview_cases.yaml
#       topology     SciVisAgentBench-tasks/topology       benchmark/eval_cases/topology/topology_cases.yaml
#       molecular    SciVisAgentBench-tasks/molecular_vis  benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml
#       anonymized   SciVisAgentBench-tasks/anonymized_datasets  benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml
#       bioimage     SciVisAgentBench-tasks/bioimage_data  benchmark/eval_cases/napari/eval_visualization_tasks.yaml
#     Layout differs by suite: paraview/topology/molecular/anonymized are flat
#     (<case>/) so case_path = cases_dir/<case>; bioimage is nested
#     (eval_visualization_tasks/case_N) because the framework adds the yaml stem
#     as a subdir. collect_results.py mirrors whatever path back automatically.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
ENGINE="${CONTAINER_ENGINE:-docker}"

# These mirror the run_*_eval.py flags. --config/--image are derived from --agent
# when not given. --cases-dir/--exp are kept as aliases for --cases/--experiment-number.
CASES_DIR="" ; YAML="" ; CONFIG="" ; AGENT="claude_code" ; EXP="exp1"
ONLY="" ; IMAGE=""
DO_STAGE=1 ; DO_COLLECT=1 ; COLLECT_MODE=each

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cases|--cases-dir)        CASES_DIR="$2"; shift 2 ;;
        --yaml)                     YAML="$2"; shift 2 ;;
        --config)                   CONFIG="$2"; shift 2 ;;
        --agent)                    AGENT="$2"; shift 2 ;;
        --experiment-number|--exp)  EXP="$2"; shift 2 ;;
        --only)                     ONLY="$2"; shift 2 ;;
        --image)                    IMAGE="$2"; shift 2 ;;
        --no-stage)                 DO_STAGE=0; shift ;;
        --no-collect)               DO_COLLECT=0; shift ;;
        --collect-each)             COLLECT_MODE=each; shift ;;  # default; accepted for compatibility
        --collect-end)              COLLECT_MODE=end;  shift ;;  # one batch copy at the end
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -z "${CASES_DIR}" ]] && { echo "ERROR: --cases (cases dir) is required." >&2; exit 2; }
[[ -z "${YAML}" ]] && { echo "ERROR: --yaml is required." >&2; exit 2; }

# Derive per-agent defaults (overridable). Runner = benchmark/run_<agent>_eval.py,
# config = benchmark/configs/<agent>/config.json, image = scivis-bench:<short>.
RUNNER="run_${AGENT}_eval.py"
[[ ! -f "${REPO_ROOT}/benchmark/${RUNNER}" ]] && {
    echo "ERROR: no runner for agent '${AGENT}' (expected benchmark/${RUNNER})." >&2; exit 2; }
[[ -z "${CONFIG}" ]] && CONFIG="benchmark/configs/${AGENT}/config.json"
if [[ -z "${IMAGE}" ]]; then
    case "${AGENT}" in
        claude_code) IMAGE="scivis-bench:claude" ;;
        codex_cli)   IMAGE="scivis-bench:codex" ;;
        *)           IMAGE="scivis-bench:latest" ;;
    esac
fi

# Agent-specific runtime needs. `codex exec` authenticates from its config dir
# (~/.codex/auth.json), NOT the OPENAI_API_KEY env var — so without it the
# container 401s ("Missing bearer"). Mount the host's codex home in and point
# CODEX_HOME at it so the container uses the exact credentials that work locally.
AGENT_DOCKER_ARGS=()
if [[ "${AGENT}" == "codex_cli" ]]; then
    CODEX_HOME_HOST="${CODEX_HOME:-${HOME}/.codex}"
    if [[ -f "${CODEX_HOME_HOST}/auth.json" ]]; then
        AGENT_DOCKER_ARGS+=(-v "${CODEX_HOME_HOST}:/codex-home" -e CODEX_HOME=/codex-home)
    else
        echo "WARN: ${CODEX_HOME_HOST}/auth.json not found — codex will 401." >&2
        echo "      Run 'codex login' on the host first (or set CODEX_HOME)." >&2
    fi
fi

CAT="$(basename "${CASES_DIR}")"
SANDBOX="${REPO_ROOT}/sandbox/${CAT}"
# Absolute config path for host-side reads (e.g. deriving agent_mode).
if [[ "${CONFIG}" = /* ]]; then CONFIG_ABS="${CONFIG}"; else CONFIG_ABS="${REPO_ROOT}/${CONFIG}"; fi

# 1) Stage a sanitized copy (strips GS/ and labels).
if [[ "${DO_STAGE}" -eq 1 ]]; then
    STAGE_ARGS=(--cases-dir "${CASES_DIR}" --out "${SANDBOX}" --force)
    [[ -n "${ONLY}" ]] && STAGE_ARGS+=(--cases ${ONLY})
    python3 "${HERE}/stage_workspace.py" "${STAGE_ARGS[@]}"
fi
SANDBOX_ABS="$(cd "${SANDBOX}" && pwd)"

# 2) Run the orchestrator + agent inside the container. The sandbox is mounted
#    at the same relative path the host command uses, so the command is identical.
RUN_ARGS=(python "benchmark/${RUNNER}"
          --config "${CONFIG}" --agent "${AGENT}" --yaml "${YAML}"
          --cases "SciVisAgentBench-tasks/${CAT}" --exe-only --experiment-number "${EXP}")
[[ -n "${ONLY}" ]] && RUN_ARGS+=(--case ${ONLY})

# The framework also writes a *centralized* run-metadata copy to
#   <cwd>/test_results/<category>/<agent_mode>/   (Path.cwd() == /work in here).
# /work is root-owned, so a --user run can't create it there. Mount the host's
# test_results/ at that path: it makes the write succeed AND maps the centralized
# results straight back to the host, where the reporter reads them. (Pre-create
# it as the host user so docker doesn't auto-create it root-owned.)
mkdir -p "${REPO_ROOT}/test_results"

# Shared `docker run` arguments for both modes.
DOCKER_ARGS=(run --rm --platform linux/amd64 --user "$(id -u):$(id -g)"
    -v "${REPO_ROOT}/benchmark:/work/benchmark:ro"
    -v "${SANDBOX_ABS}:/work/SciVisAgentBench-tasks/${CAT}"
    -v "${REPO_ROOT}/test_results:/work/test_results"
    -e ANTHROPIC_API_KEY -e OPENAI_API_KEY -e OPENAI_BASE_URL
    "${AGENT_DOCKER_ARGS[@]}"
    -w /work)

collect_one() {  # $1 = case name
    python3 "${HERE}/collect_results.py" --sandbox "${SANDBOX_ABS}" \
        --cases-dir "${CASES_DIR}" --agent-mode "${MODE}" --case "$1" || true
}

run_foreground() {  # run the container attached, streaming output
    local tty; tty="$([[ -t 0 ]] && echo -it || echo -i)"
    # shellcheck disable=SC2086
    "${ENGINE}" "${DOCKER_ARGS[@]}" ${tty} "${IMAGE}" "${RUN_ARGS[@]}"
}

# The agent_mode the framework writes results under: config agent_name + config
# model + exp (e.g. claude_code_claude-sonnet-4-5_exp1).
AGENT_NAME="$(python3 -c "import json;print(json.load(open('${CONFIG_ABS}')).get('agent_name','${AGENT}'))")"
MODEL="$(python3 -c "import json;print(json.load(open('${CONFIG_ABS}'))['model'].replace('/','-'))")"
MODE="${AGENT_NAME}_${MODEL}_${EXP}"

if [[ "${DO_COLLECT}" -eq 0 ]]; then
    # --- run only, never copy back -------------------------------------------
    echo "Running execution in ${IMAGE} (no collect; GS NOT mounted)..."
    run_foreground

elif [[ "${COLLECT_MODE}" == "each" ]]; then
    # --- per-case collection (default) ---------------------------------------
    # The container runs the whole case loop once; we watch the (host-mounted)
    # centralized test_results dir and collect each case the instant its
    # <case>_result_<ts>.json appears — which the framework writes only AFTER
    # that case's outputs are fully flushed, so the copy is race-free.
    TR_DIR="${REPO_ROOT}/test_results/${CAT}/${MODE}"
    mkdir -p "${TR_DIR}"

    declare -A SEEN COLLECTED
    for f in "${TR_DIR}"/*_result_*.json; do
        [[ -e "$f" ]] && SEEN["$(basename "$f")"]=1   # ignore pre-existing results
    done

    scan_and_collect() {
        local f b case_name
        for f in "${TR_DIR}"/*_result_*.json; do
            [[ -e "$f" ]] || continue
            b="$(basename "$f")"
            [[ -n "${SEEN[$b]:-}" ]] && continue
            case_name="$(printf '%s' "$b" | sed -E 's/_result_[0-9]+\.json$//')"
            [[ -n "${COLLECTED[$case_name]:-}" ]] && continue
            echo "  ↳ case '${case_name}' finished — collecting to real tree..."
            collect_one "${case_name}"
            COLLECTED["$case_name"]=1
        done
    }

    echo "Running execution in ${IMAGE} (per-case collect; GS NOT mounted)..."
    "${ENGINE}" "${DOCKER_ARGS[@]}" "${IMAGE}" "${RUN_ARGS[@]}" &
    CID=$!
    while kill -0 "$CID" 2>/dev/null; do
        scan_and_collect
        sleep 4
    done
    RC=0; wait "$CID" || RC=$?
    scan_and_collect   # final sweep for the last case
    [[ ${RC} -ne 0 ]] && { echo "Container exited ${RC}." >&2; exit ${RC}; }

else
    # --- single batch collect at the end (--collect-end) ---------------------
    echo "Running execution in ${IMAGE} (batch collect at end; GS NOT mounted)..."
    run_foreground
    python3 "${HERE}/collect_results.py" --sandbox "${SANDBOX_ABS}" --cases-dir "${CASES_DIR}"
fi

echo
echo "Execution done. Evaluate on the host (has GS + metric stack), e.g.:"
echo "  python -m benchmark.evaluation_framework.run_evaluation \\"
echo "    --agent ${AGENT} --config ${CONFIG} --yaml ${YAML} \\"
echo "    --cases ${CASES_DIR} --eval-only --eval-model gpt-5.2 --experiment-number ${EXP}"
