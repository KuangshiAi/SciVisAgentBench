# Running Agents Inside Docker

How to run a coding agent (Claude Code, Codex, …) on the SciVisAgentBench tasks
**inside the unified container**, with the agent sandboxed away from the ground
truth. This complements [`README.md`](README.md) (which documents the image
itself) and focuses on the *agent run* workflow.

> TL;DR — execution runs in the container against a **sanitized** copy of the
> cases; **evaluation runs on the host** (which has the ground truth + metric
> stack). One wrapper, [`run_eval_in_docker.sh`](run_eval_in_docker.sh), does the
> staging, the container run, and the copy-back.

---

## Mental model

```
        host                          container (sandboxed)              host
 ┌────────────────────┐  mount RW   ┌──────────────────────┐  collect  ┌──────────────┐
 │ stage_workspace.py │ ──────────▶ │ run_<agent>_eval.py   │ ────────▶ │ --eval-only  │
 │  (strip GS/labels) │ /workspace  │  --exe-only, no GS     │  results  │ + reporter   │
 └────────────────────┘             └──────────────────────┘           └──────────────┘
```

- The real `SciVisAgentBench-tasks/<suite>` tree is **never** mounted. Only the
  GS-free sandbox is, at the same relative path, so the in-container command is
  identical to a host command.
- The agent writes outputs into the sandbox; the wrapper copies each case back
  into the real tree as soon as it finishes (default: per-case collection).
- You then score on the host with `--eval-only` and build a report.

Why split it: letting a coding agent see `GS/` (ground-truth state, images, eval
scripts) would let it copy the answer. Staging strips `GS/`,
`visualization_goals.txt`, eval scripts, and even files byte-identical to a GS
file. See [`stage_workspace.py`](stage_workspace.py).

---

## Prerequisites

### 1. Build the images (one time)

```bash
./docker/build.sh           # base, agent-free  -> scivis-bench:latest  (large; ~30–45 min)
./docker/build_claude.sh    # + Claude Code     -> scivis-bench:claude
./docker/build_codex.sh     # + Codex CLI        -> scivis-bench:codex
```

The base image bundles the whole viz/eval toolchain (ParaView, napari, VTK, TTK,
MDAnalysis, CPU torch + lpips, headless Mesa/Xvfb). The child images just add the
agent CLI on top.

### 2. Authentication

| Agent | What it needs | How the wrapper provides it |
|-------|---------------|-----------------------------|
| **Claude Code** | `ANTHROPIC_API_KEY` in your shell | forwarded with `-e ANTHROPIC_API_KEY` |
| **Codex** | a logged-in `~/.codex/auth.json` on the host (run `codex login` once) | host `~/.codex` is mounted at `/codex-home` and `CODEX_HOME` is set |

`OPENAI_API_KEY` / `OPENAI_BASE_URL` are also forwarded. Codex authenticates from
`~/.codex/auth.json`, **not** the env var — if that file is missing you'll get
`401 Unauthorized: Missing bearer`, and the wrapper warns you up front.

---

## Quick start

### Claude Code

```bash
export ANTHROPIC_API_KEY=sk-ant-...

./docker/run_eval_in_docker.sh \
    --config benchmark/configs/claude_code/config.json \
    --agent claude_code \
    --yaml benchmark/eval_cases/paraview/paraview_subset_15.yaml \
    --cases SciVisAgentBench-tasks/paraview \
    --experiment-number exp1
```

### Codex

```bash
codex login        # once on the host, if you haven't already

./docker/run_eval_in_docker.sh \
    --config benchmark/configs/codex_cli/config.json \
    --agent codex_cli \
    --yaml benchmark/eval_cases/paraview/paraview_subset_15.yaml \
    --cases SciVisAgentBench-tasks/paraview \
    --experiment-number exp1
```

`--config` and `--image` are optional — they auto-derive from `--agent`
(`benchmark/configs/<agent>/config.json` and `scivis-bench:claude`/`:codex`). So
the minimal form is:

```bash
./docker/run_eval_in_docker.sh --agent codex_cli \
    --yaml benchmark/eval_cases/paraview/paraview_subset_15.yaml \
    --cases SciVisAgentBench-tasks/paraview --experiment-number exp1
```

Iterate on a single case while testing (cheap):

```bash
./docker/run_eval_in_docker.sh --agent claude_code \
    --yaml benchmark/eval_cases/paraview/paraview_subset_15.yaml \
    --cases SciVisAgentBench-tasks/paraview \
    --only argon-bubble --experiment-number smoke
```

---

## Then evaluate on the host

Execution above only runs the agent (`--exe-only`). Scoring needs the ground
truth + metric stack, so it runs on the host:

```bash
# 2) evaluate (host)
python benchmark/run_claude_code_eval.py \
    --config benchmark/configs/claude_code/config.json --agent claude_code \
    --yaml benchmark/eval_cases/paraview/paraview_subset_15.yaml \
    --cases SciVisAgentBench-tasks/paraview \
    --eval-model claude-opus-4-6 --eval-only --experiment-number exp1

# 3) report (host)
python -m benchmark.evaluation_reporter.run_reporter \
    --agent claude_code --config benchmark/configs/claude_code/config.json \
    --yaml benchmark/eval_cases/paraview/paraview_subset_15.yaml \
    --cases SciVisAgentBench-tasks/paraview \
    --test-results test_results/paraview/claude_code_claude-sonnet-4-5_exp1 \
    --output eval_reports/paraview_subset_15/claude_code_claude-sonnet-4-5_exp1 \
    --agent-mode claude_code_claude-sonnet-4-5_exp1
```

For Codex, swap the runner/agent/agent-mode (`run_codex_cli_eval.py`,
`--agent codex_cli`, `codex_cli_gpt-5.2_exp1`).

> **agent_mode** is `{agent_name}_{model}_{experiment_number}` from the config,
> e.g. `claude_code_claude-sonnet-4-5_exp1`, `codex_cli_gpt-5.2_exp1`. Use the
> same string for `--test-results`, `--output`, and `--agent-mode`.

---

## Wrapper options

```
./docker/run_eval_in_docker.sh [options]

  --cases DIR              Suite dir to run on (required). Alias: --cases-dir
  --yaml FILE              YAML test cases (required)
  --agent NAME             Agent (default: claude_code). Picks
                           benchmark/run_<agent>_eval.py + the defaults below
  --config FILE            Agent config (default: benchmark/configs/<agent>/config.json)
  --experiment-number STR  Experiment id (default: exp1). Alias: --exp
  --only "a b c"           Stage + run only these case names
  --image NAME             Image (default: scivis-bench:claude | :codex by agent)
  --no-stage               Reuse an existing sandbox instead of re-staging
  --no-collect             Run only; don't copy results back
  --collect-end            One batch copy at the end instead of per-case
```

By default results are collected **per case** the moment each finishes (detected
by its centralized `test_results` JSON appearing), so an interrupted run still
leaves every completed case collected.

---

## Per-suite `--cases` / `--yaml`

The suite directory name is **not** always the yaml folder name (bioimage uses
the `napari` folder), and layouts differ:

| Suite | `--cases` | `--yaml` | Layout |
|-------|-----------|----------|--------|
| paraview | `SciVisAgentBench-tasks/paraview` | `benchmark/eval_cases/paraview/paraview_cases.yaml` | flat `<case>/` |
| topology | `SciVisAgentBench-tasks/topology` | `benchmark/eval_cases/topology/topology_cases.yaml` | flat |
| molecular_vis | `SciVisAgentBench-tasks/molecular_vis` | `benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml` | flat |
| anonymized | `SciVisAgentBench-tasks/anonymized_datasets` | `benchmark/eval_cases/paraview/what_obj_cases_anonymized.yaml` | flat `dataset_NNN/` |
| bioimage | `SciVisAgentBench-tasks/bioimage_data` | `benchmark/eval_cases/napari/eval_visualization_tasks.yaml` | nested `eval_visualization_tasks/case_N/` |

The framework derives the bioimage nesting from the yaml stem; staging and
`collect_results.py` handle either layout automatically.

---

## How results map back

| Stream | In the sandbox | After the run |
|--------|----------------|---------------|
| Agent outputs (`.png`/`.py`/state) | `<case>/results/<agent_mode>/` (or nested) | `collect_results.py` copies them into the real `SciVisAgentBench-tasks/.../results/<agent_mode>/` |
| Centralized run metadata (reporter input) | written to host `test_results/<suite>/<agent_mode>/` via a mount | already on the host |

`--eval-only` reads agent outputs from the real tree; the reporter reads
`test_results/<suite>/<agent_mode>/`.

---

## Bring your own agent

The wrapper supports any agent that has a `benchmark/run_<agent>_eval.py` runner
and a `benchmark/configs/<agent>/config.json`. To add one:

1. Create a child image that installs the agent CLI (copy
   [`Dockerfile.claude`](Dockerfile.claude) / [`Dockerfile.codex`](Dockerfile.codex)).
2. Add an image default in the wrapper's `case "${AGENT}"` block (or always pass
   `--image`).
3. If the agent authenticates from a host config dir (like Codex), mount it in
   the wrapper's `AGENT_DOCKER_ARGS` block.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `401 Unauthorized: Missing bearer` (Codex) | No `~/.codex/auth.json`. Run `codex login` on the host; the wrapper mounts it in. |
| Codex uses a different model than `agent_mode` says | The agent doesn't pass `--model`; Codex reads `~/.codex/config.toml`. The `agent_mode` label comes from `config.json`. Align them in `config.toml`/`config.json` if needed. |
| `Permission denied: '/work/test_results'` | Old wrapper without the `test_results` mount — pull the current `run_eval_in_docker.sh`. |
| `mounts denied: path ... is not shared` (Docker Desktop) | The sandbox / mounted paths must live under a shared path (your `$HOME`). The defaults already do. |
| Nothing collected after a crash | A `set -e` abort before collection. With per-case collection (default) completed cases are already copied; re-run to finish the rest. |
| Want to see GS never entered the container | Check `sandbox/<suite>/STAGING_MANIFEST.json` (`ground_truth_included: false`, plus any `content_duplicates_removed`). |

---

## Files

| File | Purpose |
|------|---------|
| [`run_eval_in_docker.sh`](run_eval_in_docker.sh) | Stage → run agent (`--exe-only`) in the container → collect back |
| [`stage_workspace.py`](stage_workspace.py) | Build the GS-free sandbox (suite-aware + content dedup) |
| [`collect_results.py`](collect_results.py) | Copy agent outputs back, preserving nested paths |
| [`Dockerfile.claude`](Dockerfile.claude) / [`build_claude.sh`](build_claude.sh) | Claude Code child image |
| [`Dockerfile.codex`](Dockerfile.codex) / [`build_codex.sh`](build_codex.sh) | Codex child image |
