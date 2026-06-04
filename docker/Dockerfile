# =============================================================================
# SciVisAgentBench — unified, agent-agnostic task-runtime image
# =============================================================================
# A single Linux/amd64 image that contains the full scientific-visualization
# toolchain used by the benchmark tasks (ParaView, napari, VTK, TopologyToolKit,
# MDAnalysis / vmd-python, gudhi, ...) plus CPU-only torch + lpips so the same
# image can also run the image-quality evaluation when ground truth is mounted.
#
# It deliberately ships NO agent (Claude Code, Codex, ...). Bring your own agent
# and run it inside the container against the sanitized /workspace mount. The
# host stages /workspace so the agent never sees ground truth (see docker/README.md).
#
# Build:  docker build -t scivis-bench:latest -f docker/Dockerfile docker/
# Run:    see docker/run.sh
# =============================================================================
# The platform is pinned to linux/amd64 on purpose: the conda env is solved
# against linux-64 package builds, so the image must be amd64 even when built on
# an Apple-Silicon host. BuildKit emits a cosmetic "FromPlatformFlagConstDisallowed"
# lint warning for this constant — it is expected and safe to ignore.
FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# -----------------------------------------------------------------------------
# System packages: a headless OpenGL stack (Mesa software rendering + Xvfb) so
# ParaView / VTK / napari can render offscreen with no GPU, plus base tooling.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget bzip2 git procps tini \
        xvfb x11-utils \
        libgl1 libglx0 libegl1 libopengl0 libglu1-mesa \
        libgl1-mesa-dri libglx-mesa0 libegl-mesa0 \
        libxrender1 libxext6 libsm6 libice6 libxi6 libxrandr2 libxfixes3 \
        libxcursor1 libxinerama1 libxdamage1 libxcomposite1 libxtst6 \
        libxcb1 libx11-6 libxkbcommon0 libdbus-1-3 libfontconfig1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Miniforge (conda-forge flavoured conda + mamba). Pinned-ish via "latest" tag;
# the *environment* is what is pinned, not the installer.
# -----------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
SHELL ["/bin/bash", "-c"]

RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
        -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniforge.sh \
    # The environment pins ~48 low-level packages from Anaconda's `defaults`
    # channel; accept its Terms of Service non-interactively so the solve does
    # not fail in an automated build. The `|| true` keeps older conda builds
    # (which lack the `tos` subcommand) working.
    && conda config --set channel_priority flexible \
    && conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main \
        --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# -----------------------------------------------------------------------------
# Create the `scivis_bench` environment from the exported spec.
# -----------------------------------------------------------------------------
COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml \
    && mamba clean -afy \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete

# -----------------------------------------------------------------------------
# CPU-only torch (the validated env had CUDA wheels we stripped), then the two
# pip packages that depend on torch — installed *after* torch so they reuse the
# CPU build instead of pulling the multi-GB CUDA stack from PyPI.
# -----------------------------------------------------------------------------
ENV ENV_BIN=${CONDA_DIR}/envs/scivis_bench/bin
RUN ${ENV_BIN}/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 torchvision==0.26.0 \
    && ${ENV_BIN}/pip install --no-cache-dir \
        lpips==0.1.4 sentence-transformers==5.4.0

# -----------------------------------------------------------------------------
# Make the env the default interpreter and configure headless rendering.
# -----------------------------------------------------------------------------
ENV CONDA_DEFAULT_ENV=scivis_bench \
    PATH=${CONDA_DIR}/envs/scivis_bench/bin:${PATH} \
    DISPLAY=:99 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
    MPLBACKEND=Agg \
    QT_QPA_PLATFORM=xcb \
    PYTHONUNBUFFERED=1

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    # Pre-create the X socket dir so a non-root (`--user`) Xvfb can use it
    # without the "_XSERVTransmkdir: euid != 0" warning.
    && mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

WORKDIR /workspace

# tini reaps the background Xvfb process cleanly; entrypoint starts Xvfb and
# activates the conda env, then execs the requested command (default: bash).
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
