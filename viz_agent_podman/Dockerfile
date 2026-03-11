FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic build and runtime dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    ca-certificates \
    bzip2 \
    libgl1 \
    libxrender1 \
    libxext6 \
    libsm6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# -------- Install Mambaforge (conda with mamba), architecture aware --------
ENV MAMBA_VERSION=23.11.0-0
ENV CONDA_DIR=/opt/conda

# Detect architecture and download appropriate installer
RUN arch=$(uname -m) && \
    echo "Detected arch: ${arch}" && \
    if [ "${arch}" = "x86_64" ]; then \
        installer="Mambaforge-${MAMBA_VERSION}-Linux-x86_64.sh"; \
    elif [ "${arch}" = "aarch64" ]; then \
        installer="Mambaforge-${MAMBA_VERSION}-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: ${arch}"; exit 1; \
    fi && \
    wget https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/${installer} -O /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/mambaforge.sh

ENV PATH=${CONDA_DIR}/bin:${PATH}
SHELL ["/bin/bash", "-c"]

# -------- Environment file --------
COPY environment.yml /tmp/environment.yml

RUN mamba env create -f /tmp/environment.yml && \
    mamba clean -a -y

# Activate this environment by default
ENV CONDA_DEFAULT_ENV=viz-agent
ENV PATH=${CONDA_DIR}/envs/viz-agent/bin:${PATH}

ENV PATH=${PARAVIEW_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${PARAVIEW_DIR}/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${PARAVIEW_DIR}/lib/python3.10/site-packages:${PYTHONPATH}

WORKDIR /workspace

# Only test ParaView import on x86_64
RUN arch=$(uname -m) && \
    if [ "${arch}" = "x86_64" ]; then \
        python -c "import paraview.simple as pvs; print('ParaView version:', pvs.GetParaViewVersion().GetVersion())"; \
    else \
        echo "Skipping ParaView test on arch ${arch}"; \
    fi

CMD ["/bin/bash"]