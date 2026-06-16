#!/usr/bin/env bash
# ==============================================================================
# SPACESHIELD RELEASE AUTOMATION
# Air-Gapped Secure Artifact Bundler & Cryptographic Sealer
# ==============================================================================

# Enforce strict error handling to fail fast on command failure or unbound variables
set -euo pipefail
IFS=$'\n\t'

# ==============================================================================
# Initialization & Environment Mapping
# ==============================================================================
VERSION_TAG=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="$(pwd)"
STAGING_BASE="releases"
RELEASE_NAME="spaceshield_airgap_v${VERSION_TAG}"
RELEASE_DIR="${STAGING_BASE}/${RELEASE_NAME}"

ARCHIVE_NAME="${RELEASE_NAME}.tar.gz"
CHECKSUM_NAME="${RELEASE_NAME}.sha256"

echo "==============================================================================="
echo "SPACESHIELD MILITARY-GRADE RELEASE BUNDLER INITIATING..."
echo "Release ID: v${VERSION_TAG}"
echo "==============================================================================="

# ==============================================================================
# Graceful Cleanup Hook
# ==============================================================================
cleanup() {
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        echo ""
        echo "[ERROR] Release bundling encountered a fatal exception (Code ${exit_code})!"
        echo "[ERROR] Purging partial/compromised staging artifacts..."
        rm -rf "${RELEASE_DIR}"
        exit ${exit_code}
    fi
}
# Trap EXIT automatically triggers cleanup on abrupt failure
trap cleanup EXIT

# ==============================================================================
# 1. Structure Release Staging Directory
# ==============================================================================
echo "[INFO] Architecting structured release staging payload tree..."
mkdir -p "${RELEASE_DIR}/container"
mkdir -p "${RELEASE_DIR}/frontend"
mkdir -p "${RELEASE_DIR}/keys/public"
mkdir -p "${RELEASE_DIR}/manifests/baseline"

# ==============================================================================
# 2. Build & Serialize Container Runtime Layer
# ==============================================================================
echo "[INFO] Compiling DevSecOps Hardened Container Image (spaceshield-core:latest)..."
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker engine not found in the execution PATH. Aborting."
    exit 1
fi

docker build -t spaceshield-core:latest -f Dockerfile .

echo "[INFO] Serializing container layers into portable binary tarball (docker save)..."
docker save spaceshield-core:latest | gzip > "${RELEASE_DIR}/container/spaceshield-core-img.tar.gz"

# ==============================================================================
# 3. Synchronize Static Assets, Keys, and Compliance Ledgers
# ==============================================================================
echo "[INFO] Integrating compiled static frontend assets..."
if [ -d "frontend" ] && [ "$(ls -A frontend)" ]; then
    cp -r frontend/* "${RELEASE_DIR}/frontend/"
else
    echo "  -> [WARN] /frontend/ directory is empty or missing. Bypassing."
fi

echo "[INFO] Embedding Cryptographic Public Keys for transport authentication..."
if [ -d "keys" ] && [ "$(ls -A keys)" ]; then
    # We only ship public keys in the release payload to maintain air-gap integrity
    find keys -type f \( -name "*.pub" -o -name "*.pem" -o -name "*.crt" \) -exec cp {} "${RELEASE_DIR}/keys/public/" \;
else
    echo "  -> [WARN] /keys/ directory is empty. Injecting zero-trust placeholder."
    echo "PLACEHOLDER: AWAITING_PRODUCTION_KEY_INJECTION" > "${RELEASE_DIR}/keys/public/placeholder.txt"
fi

echo "[INFO] Securing Configuration Baseline Manifests & Compliance WORM Ledgers..."
if [ -d "compliance" ]; then
    cp -r compliance/* "${RELEASE_DIR}/manifests/baseline/"
fi
# Always bundle the exact Dockerfile logic that created the container for auditing
cp Dockerfile "${RELEASE_DIR}/manifests/baseline/Dockerfile.audit"

# ==============================================================================
# 4. Archive Consolidation
# ==============================================================================
echo "[INFO] Compressing release payload into zero-dependency physical transport archive..."
# Compress starting strictly from the releases directory to ensure clean extraction paths
tar -czf "${STAGING_BASE}/${ARCHIVE_NAME}" -C "${STAGING_BASE}" "${RELEASE_NAME}"

# ==============================================================================
# 5. Cryptographic SHA-256 Validation Manifest
# ==============================================================================
echo "[INFO] Calculating independent SHA-256 validation manifest..."
cd "${STAGING_BASE}"

if command -v sha256sum > /dev/null; then
    sha256sum "${ARCHIVE_NAME}" > "${CHECKSUM_NAME}"
elif command -v shasum > /dev/null; then
    shasum -a 256 "${ARCHIVE_NAME}" > "${CHECKSUM_NAME}"
else
    echo "[WARN] Cryptographic hashing utilities unavailable. Integrity manifest skipped."
fi

cd "${BASE_DIR}"

# ==============================================================================
# Final Cleanup & Verification
# ==============================================================================
echo "[INFO] Sweeping intermediate unencrypted staging directories..."
rm -rf "${RELEASE_DIR}"

# De-register failure trap on success
trap - EXIT

echo "==============================================================================="
echo "[SUCCESS] Air-Gapped Transport Artifact Compiled & Secured."
echo "Payload Path:       ${STAGING_BASE}/${ARCHIVE_NAME}"
echo "Integrity Manifest: ${STAGING_BASE}/${CHECKSUM_NAME}"
echo "Ready for physical data-diode extraction."
echo "==============================================================================="
