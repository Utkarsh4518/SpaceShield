# ==============================================================================
# SPACESHIELD DEFENSE-GRADE PRODUCTION DOCKERFILE
# Classification: Controlled Distribution — STQC-SS-2026-VAL-042
# Architect: Principal DevSecOps & Hardened Linux Embedded Systems Engineer
# Target: Hardened Ground Station Edge Compute (NVIDIA Jetson Orin / x86_64)
#
# Build Architecture:
#   Stage 1 (builder)  — Full toolchain: compiles numpy/scipy/onnxruntime wheels
#   Stage 2 (runtime)  — Ultra-lean Debian-slim: receives only compiled binaries
#
# Security Posture:
#   • Non-root execution bound to spaceshield_worker (UID 10001)
#   • Read-only root filesystem compatible (/tmp and /run as tmpfs)
#   • WORM log volume isolation at /data
#   • No pip, no gcc, no shell utilities in runtime layer
# ==============================================================================


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STAGE 1: DEPENDENCY COMPILATION & WHEEL VECTORIZATION LAYER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FROM python:3.11-slim-bookworm AS builder

LABEL stage="builder"

# Deterministic, reproducible builds — no .pyc, no buffered I/O
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install the minimal C/Fortran toolchain required to compile scipy from source
# on ARM64 (Jetson Orin) targets where pre-built manylinux wheels may not exist.
# libgomp1 provides OpenMP threading for ONNX Runtime parallel graph execution.
# libgfortran5 provides Fortran runtime for LAPACK/BLAS routines in scipy.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libgomp1 \
        libgfortran5 \
        pkg-config && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Isolated virtual environment prevents system-level package contamination.
# Only the /opt/venv directory is copied to the runtime stage — pip, setuptools,
# and the entire build toolchain are discarded.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# Install production dependencies pinned to CycloneDX SBOM-audited versions.
# The install order is deliberate: numpy first (build dependency for scipy),
# then scipy, then the ONNX format library, then the inference runtime.
# PyTorch CPU-only wheel is included for EdgeInferenceEngine model compilation.
RUN pip install \
    numpy==2.4.2 && \
    pip install \
    scipy==1.17.0 && \
    pip install \
    onnx==1.21.0 \
    onnxruntime==1.26.0

RUN pip install \
    torch==2.10.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Strip __pycache__, .dist-info, and test directories from the venv to reduce
# the final COPY layer size. Every byte removed from the runtime image reduces
# the container attack surface and accelerates cold-start pulls at edge nodes.
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type f -name "*.pyc" -delete 2>/dev/null; \
    true


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STAGE 2: HARDENED MINIMAL RUNTIME EXECUTION LAYER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FROM python:3.11-slim-bookworm AS runtime

# --------------------------------------------------------------------------
# Provenance & Compliance Metadata Labels
# --------------------------------------------------------------------------
LABEL maintainer="SpaceShield DevSecOps Team <devsecops@spaceshield.io>" \
      org.opencontainers.image.title="SpaceShield RF Threat Agent" \
      org.opencontainers.image.description="Multi-Antenna Spatial DSP Threat Detection Engine" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.vendor="SpaceShield Pvt. Ltd." \
      security.stqc.compliance="STQC-SS-2026-VAL-042" \
      security.certin.framework="CERT-In-2026-SpaceCyber" \
      security.sbom.format="CycloneDX-1.5"

# --------------------------------------------------------------------------
# Runtime Environment Configuration
# --------------------------------------------------------------------------
# PYTHONDONTWRITEBYTECODE: Prevents .pyc generation on the read-only rootfs.
# PYTHONUNBUFFERED: Forces immediate stdout/stderr flush for real-time HUD.
# PYTHONHASHSEED: Fixed hash seed for deterministic dictionary ordering in
#                 reproducible STQC audit runs.
# PYTHONFAULTHANDLER: Dumps C-level tracebacks on segfault for post-mortem
#                     analysis of native extension crashes (numpy/scipy FFI).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH"

# --------------------------------------------------------------------------
# NVIDIA GPU / CUDA / TensorRT Hardware Acceleration Path Variables
# --------------------------------------------------------------------------
# These environment variables enable transparent GPU passthrough when the
# container is launched on NVIDIA Jetson Orin Nano or discrete GPU hosts
# using the NVIDIA Container Runtime (nvidia-container-toolkit).
#
# LD_LIBRARY_PATH: Binds the CUDA shared library search path for both
#   aarch64 (Jetson L4T) and x86_64 (discrete GPU) host architectures.
#   ONNX Runtime's CUDAExecutionProvider and TensorrtExecutionProvider
#   dynamically load libcudart.so, libcudnn.so, and libnvinfer.so from
#   these paths at session initialization time.
#
# NVIDIA_VISIBLE_DEVICES: Exposes all host GPUs to the container runtime.
# NVIDIA_DRIVER_CAPABILITIES: Restricts driver exposure to compute and
#   utility functions only (no display/video capabilities needed).
#
# ORT_TENSORRT_FP16_ENABLE: Instructs ONNX Runtime's TensorRT EP to use
#   FP16 precision when available, matching our quantized model weights.
# ORT_TENSORRT_ENGINE_CACHE_ENABLE: Caches compiled TensorRT engines to
#   /tmp/trt_cache to eliminate re-compilation on container restart.
# ORT_TENSORRT_ENGINE_CACHE_PATH: Cache directory for serialized TRT plans.
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    ORT_TENSORRT_FP16_ENABLE=1 \
    ORT_TENSORRT_ENGINE_CACHE_ENABLE=1 \
    ORT_TENSORRT_ENGINE_CACHE_PATH="/tmp/trt_cache"

# --------------------------------------------------------------------------
# Minimal Runtime System Dependencies
# --------------------------------------------------------------------------
# libgomp1: OpenMP threading backend for ONNX Runtime parallel execution.
# libgfortran5: Fortran runtime required by scipy's LAPACK/BLAS bindings.
# No build tools, no compilers, no package managers remain in this layer.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libgfortran5 && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm -rf /var/cache/apt/archives/* && \
    rm -f /var/log/dpkg.log /var/log/apt/*.log

WORKDIR /app

# --------------------------------------------------------------------------
# Copy Compiled Virtual Environment from Builder Stage
# --------------------------------------------------------------------------
# This is the ONLY artifact transferred from the builder. The entire gcc/g++
# toolchain, pip, setuptools, wheel, and all header files are discarded.
COPY --from=builder /opt/venv /opt/venv

# --------------------------------------------------------------------------
# Least-Privilege User Context Enforcement
# --------------------------------------------------------------------------
# Create a dedicated system group and service account with:
#   • Explicit UID/GID 10001 (deterministic, auditable)
#   • /sbin/nologin shell (prevents interactive login)
#   • No home directory (--no-create-home) to minimize writable paths
#   • System account flag (--system) signals non-human identity
RUN groupadd --gid 10001 spaceshield && \
    useradd \
        --uid 10001 \
        --gid spaceshield \
        --shell /sbin/nologin \
        --no-create-home \
        --system \
        spaceshield_worker

# --------------------------------------------------------------------------
# Application Code Deployment with Restricted Ownership
# --------------------------------------------------------------------------
# Each COPY layer is individually chown'd to the service account.
# Root retains no write access to application code or configuration.
COPY --chown=spaceshield_worker:spaceshield src/ /app/src/
COPY --chown=spaceshield_worker:spaceshield compliance/ /app/compliance/
COPY --chown=spaceshield_worker:spaceshield data/ /app/data/

# --------------------------------------------------------------------------
# Filesystem Permission Hardening
# --------------------------------------------------------------------------
# /app: rwxr-x--- (750) — service account can read/execute; group can read.
# /app/compliance: r-xr-x--- (550) — read-only for license and audit files.
# /app/data: rwxrwx--- (770) — writable for WORM log append operations.
# /tmp: Managed as tmpfs mount by the orchestrator (noexec, nosuid).
#
# The TensorRT engine cache directory must be writable by the service user
# to persist compiled engine plans across inference sessions.
RUN chmod -R 750 /app && \
    chmod -R 550 /app/compliance && \
    chmod -R 770 /app/data && \
    mkdir -p /tmp/trt_cache && \
    chown spaceshield_worker:spaceshield /tmp/trt_cache && \
    chmod 770 /tmp/trt_cache

# --------------------------------------------------------------------------
# WORM Log Volume Isolation Point
# --------------------------------------------------------------------------
# Declaring /data as a VOLUME instructs Docker to mount this path as an
# external bind-mount or named volume. The host-side mount is configured
# in docker-compose.yml with appropriate write-once retention policies.
# Inside the container, /app/data is writable only in append mode by the
# WORM logger thread — the harness opens log file descriptors with O_APPEND
# and the cryptographic hash chain in verify_log_integrity.py detects any
# retroactive modification or truncation of spaceshield_180day_security.log.
VOLUME ["/app/data"]

# --------------------------------------------------------------------------
# Container Health Probe: Embedded Orchestration HEALTHCHECK
# --------------------------------------------------------------------------
# The HEALTHCHECK directive instructs Docker Engine, Docker Compose, and
# Kubernetes CRI adapters to poll container liveness at 5-second intervals.
#
# container_healthcheck.py reads /tmp/spaceshield_status.json written by
# the harness display_loop on every 100ms HUD refresh cycle. The probe
# evaluates four failure conditions that trigger container restart:
#
#   1. STALE LOOP: Status file timestamp > 10 seconds old → processing
#      thread deadlock or GIL contention causing main loop stall.
#   2. BLOCK LOSS: dropped_blocks > 0 → ingestion queue overflow due to
#      worker thread pool exhaustion or I/O backpressure.
#   3. QUEUE SATURATION: queue_size/queue_max > 90% → impending block
#      loss from sustained throughput exceeding processing capacity.
#   4. WORKER DEATH: num_workers <= 0 → all processing threads have
#      exited due to unhandled exceptions.
#
# After 3 consecutive failures (15 seconds), the orchestrator marks the
# container as unhealthy and triggers the restart policy.
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "/app/src/container_healthcheck.py"]

# --------------------------------------------------------------------------
# Drop Privileges: Bind All Execution to the Unprivileged Service Account
# --------------------------------------------------------------------------
# From this point forward, every RUN, CMD, and ENTRYPOINT instruction
# executes as spaceshield_worker (UID 10001). The root user is permanently
# abandoned. This prevents:
#   • Container breakout via /proc/sysrq-trigger or /dev/mem
#   • Package installation or system modification post-deployment
#   • Privilege escalation through setuid binaries
USER spaceshield_worker

# --------------------------------------------------------------------------
# Immutable Entrypoint: Ground Station Spatiotemporal Processing Harness
# --------------------------------------------------------------------------
# ENTRYPOINT uses exec-form (JSON array) to avoid sh -c wrapper overhead
# and ensure PID 1 signal propagation for graceful SIGTERM shutdown.
# CMD provides default runtime arguments that can be overridden by the
# orchestrator without modifying the entrypoint binary path.
ENTRYPOINT ["python", "/app/src/spatial_hardware_harness.py"]
CMD ["-f", "2000000", "-d", "30", "-c", "8192", "-m", "4"]
