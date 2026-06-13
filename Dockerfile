# ==============================================================================
# SPACESHIELD ZERO-TRUST HARDENED DOCKERFILE
# ==============================================================================
# Execution & Air-Gapped Deployment Guide:
# 
# 1. Build the Hardened Image (On Networked Build Server):
#    $ docker build -t spaceshield-edge:latest .
#
# 2. Export the Air-Gapped Archive (Creates a portable tarball):
#    $ docker save spaceshield-edge:latest | gzip > spaceshield-edge-airgap.tar.gz
#
# 3. Sneakernet / Transfer the archive via secure USB to the isolated node.
#
# 4. Load the Image on the Isolated Deployment Target:
#    $ docker load < spaceshield-edge-airgap.tar.gz
#
# 5. Spin Up the Container with WORM Volume Mounts (Host directory retention):
#    $ docker run -d --name spaceshield_l1_agent \
#        --network host \
#        -v /secure/host/compliance:/app/compliance \
#        -v /secure/host/data:/app/data \
#        -p 8000:8000 \
#        spaceshield-edge:latest
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1: COMPILATION & DEPENDENCY EXTRACTION
# ------------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

# Enforce non-interactive environment for clean compilation
ENV DEBIAN_FRONTEND=noninteractive

# Install C/C++ build toolchains, cmake, and Python headers required for 
# low-level driver compilation (UHD / SoapySDR) and wheel generation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libuhd-dev \
    uhd-host \
    libsoapysdr-dev \
    soapysdr-module-all \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated compilation workspace
WORKDIR /build

# Initialize isolated Python virtual environment to prevent global pollution
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Compile and extract scientific computing and API wheels.
# Using --no-cache-dir ensures we don't carry over build artifacts.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    scipy \
    onnxruntime \
    fastapi \
    uvicorn \
    websockets \
    pydantic

# ------------------------------------------------------------------------------
# STAGE 2: FOOTPRINT MINIMIZATION & SECURE RUNTIME
# ------------------------------------------------------------------------------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# 1. Install ONLY the required native C/C++ shared object runtime libraries (.so).
# 2. Immediately strip the package manager (apt/dpkg) to reduce attack surface.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libuhd4.1.3 \
    libsoapysdr0.8 \
    soapysdr-module-all \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove apt dpkg \
    && rm -rf /usr/bin/apt* /usr/bin/dpkg* /var/cache/apt /var/lib/dpkg

# Copy over the compiled Python dependency wheels from Stage 1
COPY --from=builder /opt/venv /opt/venv

# ------------------------------------------------------------------------------
# SECURITY CONFIGURATION: NON-ROOT EXECUTION ISOLATION
# ------------------------------------------------------------------------------

# Create a dedicated unprivileged user to prevent container root breakout attacks
RUN useradd -ms /bin/bash spaceshield_operator
WORKDIR /app

# Inject source code payload with strict ownership boundaries
COPY --chown=spaceshield_operator:spaceshield_operator backend/ ./backend/
COPY --chown=spaceshield_operator:spaceshield_operator frontend/ ./frontend/

# Establish internal mount targets for external WORM volume bindings
RUN mkdir -p /app/compliance /app/data && \
    chown -R spaceshield_operator:spaceshield_operator /app/compliance /app/data

# Drop privileges by switching to the non-root user
USER spaceshield_operator

# Expose the API configuration loop and telemetry streaming socket
EXPOSE 8000

# Execute the background processing pipeline via the ASGI server
CMD ["uvicorn", "backend.src.dashboard_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
