# SciVisAgentBench — Unified Docker Environment

A single, cross-platform Docker image that bundles the **entire scientific
visualization toolchain** the benchmark tasks need — so you can test *any* agent
(Claude Code, Codex, a custom CLI, an API loop, …) on the same reproducible
runtime, with the agent's filesystem **sandboxed away from the ground truth**.

This supersedes the old `viz_agent_podman/` prototype (a libraries-only Podman
container with no benchmark integration and no data isolation).

> The per-agent **conda** setups documented in the top-level
> [`README.md`](../README.md) are still the way to run the MCP servers
> (ParaView-MCP, bioimage-agent, GMX-VMD-MCP, TopoPilot2). This Docker image is
> an additional, agent-agnostic path that is especially convenient for
> **coding agents** like Claude Code.

---

## What's in the image

Built from the validated local `scivis_bench` conda env (see
[`environment.yml`](environment.yml)):

| Area | Tools |
|------|-------|
| Volume / mesh / flow | `paraview` (5.13.3, incl. `pvpython`/`pvbatch`), `vtk` |
| Bio-imaging | `napari` (headless) |
| Topology | `topologytoolkit` (TTK), `gudhi` |
| Molecular | `vmd-python`, `MDAnalysis` |
| Scientific Python | `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-image`, `opencv`, `netcdf4`, `imageio`, … |
| Evaluation metrics | CPU **torch** + `lpips` (PSNR/SSIM/LPIPS), `POT`, `faiss-cpu` |
| Rendering | Mesa software OpenGL (`llvmpipe`) + `Xvfb` for headless screenshots |

Notes:
- **Linux/amd64** image. Runs on Apple Silicon / Windows via Docker Desktop's
  Linux VM (emulated). The image is GPU-free and CPU-only by design.
- **No GROMACS / VMD GUI binaries** (molecular tasks use `vmd-python` +
  `MDAnalysis`, matching the local `scivis_bench` env).
- **No agent is installed.** Bring your own (see [Running an agent](#running-an-agent)).

---

## The data-isolation model

Each benchmark case ships its answer alongside its inputs:

```
SciVisAgentBench-tasks/paraview/vortex/
├── task_description.txt        # ← the agent SHOULD see this
├── data/                       # ← and this (inputs)
├── visualization_goals.txt     # ✗ evaluation rubric (a label)
├── GS/                         # ✗ GROUND TRUTH (state, images, eval scripts)
├── results/  test_results/  evaluation_results/   # ✗ prior-run artifacts
```

Letting a coding agent see `GS/` would let it copy the answer. So the flow is:

```
          host                          container (sandboxed)            host
  ┌───────────────────┐   mount RW   ┌──────────────────────┐   copy   ┌──────────────┐
  │ stage_workspace.py │ ──────────▶ │  agent runs in        │ ───────▶ │ collect +     │
  │  (strip GS/labels) │  /workspace │  /workspace, no GS     │ results  │ evaluate (GS) │
  └───────────────────┘             └──────────────────────┘          └──────────────┘
```

1. **Stage** a sanitized copy of a cases directory ([`stage_workspace.py`](stage_workspace.py)).
   It strips `GS/`, `visualization_goals.txt`, prior `results/`/`test_results/`/
   `evaluation_results/`, agent transcripts, and any stray `*_gs.*` / `gs_*` /
   `*_eval.py` files — then **audits** the output and refuses to produce a
   sandbox that still contains ground truth.
2. **Run** the agent in the container with only the sandbox mounted at
   `/workspace`. The agent literally cannot read what isn't mounted.
3. **Collect** the agent's outputs back into the real tree
   ([`collect_results.py`](collect_results.py)) and **evaluate on the host**,
   which has both the ground truth and the metric stack.

---

## Quick start

```bash
# 0. (one time) build the image  — takes a while; it's a large scientific stack
./docker/build.sh                       # -> scivis-bench:latest

# 1. stage + drop into the sandboxed container for the paraview category
./docker/run.sh --cases-dir SciVisAgentBench-tasks/paraview

#    …inside the container, /workspace is the sanitized tree. Run your agent,
#    or sanity-check the env:
python -c "import paraview.simple, napari, vtk, MDAnalysis, gudhi; print('env OK')"
ls /workspace                            # vortex/  foot/  ...  (no GS/)
exit

# 2. copy the agent's outputs back and evaluate on the host
python docker/collect_results.py \
    --sandbox sandbox/paraview \
    --cases-dir SciVisAgentBench-tasks/paraview \
    --agent-mode docker_myagent_exp1

python -m benchmark.evaluation_framework.run_evaluation \
    --agent claude_code \
    --config benchmark/configs/claude_code/config.json \
    --yaml benchmark/eval_cases/paraview/paraview_cases.yaml \
    --cases SciVisAgentBench-tasks/paraview \
    --eval-only --agent-mode docker_myagent_exp1 \
    --eval-model gpt-5.2
```

Stage only a few cases while iterating:

```bash
./docker/run.sh --cases-dir SciVisAgentBench-tasks/paraview --cases "vortex foot"
```

---

## Running an agent

> For the end-to-end benchmark workflow (Claude Code / Codex child images, the
> `run_eval_in_docker.sh` wrapper, and host-side evaluation) see
> **[`RUNNING_AGENTS.md`](RUNNING_AGENTS.md)**. The options below are the lower-level
> primitives that doc builds on.

The image is intentionally agent-free. Three common options:

### A) Mount an agent CLI from the host
If your agent is a self-contained binary on the host, mount it in:
```bash
./docker/run.sh --cases-dir SciVisAgentBench-tasks/paraview -- \
    bash -lc 'your-agent --help'      # if it's already on PATH via a mount
```
or add `-v /path/to/agent:/opt/agent` by editing/extending `run.sh`.

### B) Layer the agent in a child image (recommended for repeatable runs)
Create `docker/Dockerfile.claude` (example for a Node-based CLI):
```dockerfile
FROM scivis-bench:latest
USER root
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*
```
```bash
docker build -t scivis-bench:claude -f docker/Dockerfile.claude docker/
./docker/run.sh --cases-dir SciVisAgentBench-tasks/paraview --image scivis-bench:claude -- \
    claude --print --dangerously-skip-permissions "Do the task in /workspace/vortex/task_description.txt"
```

### C) Run an API-based agent
`anthropic`, `openai`, and `mcp` Python SDKs are already in the image. Mount your
own driver script into `/workspace` and run it. `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, and `OPENAI_BASE_URL` from your shell are forwarded by
`run.sh`.

> **agent_mode / output paths.** Task prompts tell the agent to save to
> `<case>/results/{agent_mode}/…`. Pick an `agent_mode` (e.g.
> `docker_myagent_exp1`) and use the *same* string when you run
> `collect_results.py --agent-mode` and the evaluator `--agent-mode`.

---

## Docker Desktop note (WSL2 / macOS)

Docker Desktop only bind-mounts paths it has **file sharing** for — typically your
`$HOME`. The sandbox must therefore live under a shared path. `run.sh` already
defaults it to `<repo>/sandbox/...` (inside your home), so the default works; just
avoid pointing `--sandbox` at `/tmp` or other unshared locations, or you'll get
`Error response from daemon: mounts denied: path ... is not shared from the host`.
(With a native Docker Engine there's no such restriction.)

## Headless rendering

The entrypoint starts `Xvfb` on `:99` and the image forces Mesa software GL
(`LIBGL_ALWAYS_SOFTWARE=1`, `llvmpipe`). ParaView/VTK/napari therefore render
offscreen with no GPU. If a tool misbehaves, useful overrides:

```bash
-e QT_QPA_PLATFORM=offscreen          # for some napari/Qt setups
-e LIBGL_ALWAYS_SOFTWARE=1            # already the default
```

---

## File ownership

`run.sh` runs the container as your host `UID:GID` so files written to the
sandbox stay owned by you. Pass `--root` to run as root instead (e.g. if you
need to `apt-get install` inside an ephemeral container).

---

## Rebuilding the environment spec

`environment.yml` was produced from the local env and then made portable
(conda-forge + defaults; GPU/CUDA wheels removed; CPU torch + lpips installed in
the Dockerfile). To regenerate from a current env:

```bash
conda env export -n scivis_bench --no-builds > docker/environment.yml
# then re-strip torch/torchvision/nvidia-*/cuda-*/lpips/sentence-transformers
# (they are reinstalled CPU-only in the Dockerfile)
```

---

## Files

| File | Purpose |
|------|---------|
| [`Dockerfile`](Dockerfile) | The unified image definition |
| [`environment.yml`](environment.yml) | Pinned conda+pip spec (from `scivis_bench`) |
| [`entrypoint.sh`](entrypoint.sh) | Starts Xvfb, activates env, execs your command |
| [`stage_workspace.py`](stage_workspace.py) | Build a sanitized, GS-free sandbox |
| [`collect_results.py`](collect_results.py) | Copy agent outputs back for host-side eval |
| [`build.sh`](build.sh) | Build the image (docker or podman) |
| [`run.sh`](run.sh) | Stage + run the sandboxed container |
