# ==============================================================================
# SPACESHIELD DEVSECOPS PIPELINE
# AIR-GAPPED HARDENED COMPUTING INFRASTRUCTURE
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1: BUILDER COMPONENT
# ------------------------------------------------------------------------------
# Pinned Debian slim base to maximize pre-compiled wheel compatibility for SIMD Numba
FROM python:3.11-slim-bookworm AS builder

# Set strict non-interactive flags
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install static build-essential headers, C-extension toolchains, and RT dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    librtlsdr-dev \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/build

# Copy explicitly defined dependencies to maximize layer cache
COPY backend/requirements.txt .

# Pre-compile specialized C-extensions, SIMD Numba runtime dependencies, 
# and fixed-point CORDIC abstractions into isolated binary wheels
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /opt/build/wheels -r requirements.txt

# ------------------------------------------------------------------------------
# STAGE 2: PRODUCTION RUNTIME LAYER
# ------------------------------------------------------------------------------
# Minimal, hardened base snapshot
FROM python:3.11-slim-bookworm

# System Environment Lock
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Establish non-root system user for strict privilege separation
# Drops default administrative rights
RUN groupadd -g 10001 spaceshield_sec && \
    useradd -u 10001 -g spaceshield_sec -s /sbin/nologin -M spaceshield_rt

WORKDIR /opt/spaceshield

# Transfer compiled wheels from the Builder Component
# This strictly guarantees no build-essential headers or compilers exist in production
COPY --from=builder /opt/build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy complete backend/src pool and embed fixed-size ONNX tensor model weights
# Assign strict ownership to the non-root execution user
COPY --chown=spaceshield_rt:spaceshield_sec backend/src /opt/spaceshield/backend/src
COPY --chown=spaceshield_rt:spaceshield_sec models /opt/spaceshield/models

# Disable writable layer access across critical directories to prevent runtime tampering
# Enforces an absolute Read-Only (555) execution boundary
RUN chmod -R 555 /opt/spaceshield/backend/src /opt/spaceshield/models

# Expose exclusively port 8000 for the FastAPI WebSocket loop
EXPOSE 8000

# Drop to the hardened non-root system user
USER spaceshield_rt

# ------------------------------------------------------------------------------
# RUNTIME CAPABILITIES NOTE
# When executing this container in production, the orchestrator MUST enforce:
# docker run --cap-drop=ALL --cap-add=IPC_LOCK --cap-add=SYS_NICE 
# (IPC_LOCK and SYS_NICE are required for rt_thread_allocator.py mlockall and sched_setscheduler)
# ------------------------------------------------------------------------------

# Launch the FastAPI WebSocket Loop bound to the exposed interface
ENTRYPOINT ["python", "-m", "uvicorn", "backend.src.api_bridge:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "warning"]
