#!/usr/bin/env bash
# ==============================================================================
# SpaceShield: Linux cgroups v2 Core Hardening & Shielding Script
# Author: Senior Embedded Linux Kernel Engineer & System Security Architect
# Version: 1.0.0
#
# Isolates and shields the SpaceShield processing core by establishing a
# unified cgroup v2 slice hierarchy. Allocates 80% CPU capacity to real-time
# DSP and restricts memory limits on unprivileged dashboard/logging routines
# to protect the primary baseband receiver thread.
# ==============================================================================

set -euo pipefail

CGROUP_ROOT="/sys/fs/cgroup"
SLICE_NAME="spaceshield.slice"
SLICE_PATH="${CGROUP_ROOT}/${SLICE_NAME}"

# Sub-cgroup paths
DSP_PATH="${SLICE_PATH}/dsp"
DASHBOARD_PATH="${SLICE_PATH}/dashboard"
LOGGING_PATH="${SLICE_PATH}/logging"

# Logging Helpers
log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $*"
}

log_warn() {
    echo -e "\033[0;33m[WARN]\033[0m $*"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $*" >&2
}

# ------------------------------------------------------------------------------
# Graceful Error & Permission Checkers
# ------------------------------------------------------------------------------

check_cgroups_v2() {
    if [ ! -d "${CGROUP_ROOT}" ]; then
        log_error "cgroups root directory not found at ${CGROUP_ROOT}."
        exit 1
    fi
    
    # cgroups v2 is mounted if cgroup.controllers exists in the root
    if [ ! -f "${CGROUP_ROOT}/cgroup.controllers" ]; then
        log_error "cgroups v2 is not enabled on this system. cgroups v1 is not supported by this script."
        exit 1
    fi
}

check_write_permissions() {
    # Check if we can write to cgroup root files (needs root or delegated write permission)
    if [ ! -w "${CGROUP_ROOT}/cgroup.subtree_control" ]; then
        log_warn "Insufficient write permissions on ${CGROUP_ROOT}/cgroup.subtree_control."
        log_warn "cgroups shielding setup requires root privileges or delegated cgroup v2 access."
        log_warn "To configure shielding, please re-run as root: sudo $0 --setup"
        return 1
    fi
    return 0
}

# ------------------------------------------------------------------------------
# Shielding Setup
# ------------------------------------------------------------------------------

setup_cgroups() {
    log_info "Initializing SpaceShield cgroups v2 slice hierarchy..."
    
    # Determine CPU core count
    num_cores=$(nproc --all 2>/dev/null || echo 1)
    log_info "Detected ${num_cores} CPU core(s)."

    # 1. Enable cpu and memory controllers in the cgroups root
    if grep -q "cpu" "${CGROUP_ROOT}/cgroup.controllers" && ! grep -q "cpu" "${CGROUP_ROOT}/cgroup.subtree_control"; then
        echo "+cpu" > "${CGROUP_ROOT}/cgroup.subtree_control"
        log_info "Enabled CPU controller in root subtree control."
    fi
    if grep -q "memory" "${CGROUP_ROOT}/cgroup.controllers" && ! grep -q "memory" "${CGROUP_ROOT}/cgroup.subtree_control"; then
        echo "+memory" > "${CGROUP_ROOT}/cgroup.subtree_control"
        log_info "Enabled Memory controller in root subtree control."
    fi

    # 2. Create the unified slice cgroup
    mkdir -p "${SLICE_PATH}"
    log_info "Created unified slice: ${SLICE_PATH}"

    # 3. Enable controllers recursively within spaceshield.slice
    if grep -q "cpu" "${SLICE_PATH}/cgroup.controllers" && ! grep -q "cpu" "${SLICE_PATH}/cgroup.subtree_control"; then
        echo "+cpu" > "${SLICE_PATH}/cgroup.subtree_control"
    fi
    if grep -q "memory" "${SLICE_PATH}/cgroup.controllers" && ! grep -q "memory" "${SLICE_PATH}/cgroup.subtree_control"; then
        echo "+memory" > "${SLICE_PATH}/cgroup.subtree_control"
    fi
    log_info "Enabled CPU & Memory controllers inside spaceshield.slice."

    # 4. Construct sub-cgroup directories
    mkdir -p "${DSP_PATH}"
    mkdir -p "${DASHBOARD_PATH}"
    mkdir -p "${LOGGING_PATH}"
    log_info "Created sub-cgroups (dsp, dashboard, logging)."

    # 5. Configure Resource Limits
    
    # [DSP Group] - High CPU Priority, Unlimited Memory
    if [ -f "${DSP_PATH}/cpu.weight" ]; then
        echo "800" > "${DSP_PATH}/cpu.weight"  # Prioritize DSP scheduling
    fi
    if [ -f "${DSP_PATH}/cpu.max" ]; then
        echo "max" > "${DSP_PATH}/cpu.max"     # Unlimited execution ceiling
    fi
    log_info "Configured DSP: Priority Weight = 800, Execution Ceiling = unlimited."

    # [Dashboard Group] - Restricted to 10% CPU, 512M Memory Max
    if [ -f "${DASHBOARD_PATH}/cpu.max" ]; then
        # 10% of total system capacity (quota = 10000 * num_cores per 100000 period)
        echo "$((10000 * num_cores)) 100000" > "${DASHBOARD_PATH}/cpu.max"
    fi
    if [ -f "${DASHBOARD_PATH}/memory.max" ]; then
        echo "536870912" > "${DASHBOARD_PATH}/memory.max"  # 512MB hard ceiling
    fi
    if [ -f "${DASHBOARD_PATH}/memory.high" ]; then
        echo "503316480" > "${DASHBOARD_PATH}/memory.high" # 480MB soft limit
    fi
    log_info "Configured Dashboard: CPU Ceiling = 10%, Memory Max = 512MB."

    # [Logging Group] - Restricted to 10% CPU, 256M Memory Max
    if [ -f "${LOGGING_PATH}/cpu.max" ]; then
        # 10% of total system capacity
        echo "$((10000 * num_cores)) 100000" > "${LOGGING_PATH}/cpu.max"
    fi
    if [ -f "${LOGGING_PATH}/memory.max" ]; then
        echo "268435456" > "${LOGGING_PATH}/memory.max"  # 256MB hard ceiling
    fi
    if [ -f "${LOGGING_PATH}/memory.high" ]; then
        echo "251658240" > "${LOGGING_PATH}/memory.high" # 240MB soft limit
    fi
    log_info "Configured Logging: CPU Ceiling = 10%, Memory Max = 256MB."
    
    log_info "SpaceShield OS cgroups v2 shielding configured successfully."
}

# ------------------------------------------------------------------------------
# PID-Assignment
# ------------------------------------------------------------------------------

assign_pid() {
    local target_path="$1"
    local pid="$2"
    
    if [ ! -d "${target_path}" ]; then
        log_error "Target cgroup directory ${target_path} does not exist. Please run setup first: $0 --setup"
        exit 1
    fi
    
    if [ ! -d "/proc/${pid}" ]; then
        log_error "Process PID ${pid} does not exist."
        exit 1
    fi
    
    # Verify write permission to cgroup.procs
    if [ ! -w "${target_path}/cgroup.procs" ]; then
        log_error "Insufficient write permissions on ${target_path}/cgroup.procs."
        log_error "To assign PIDs, run this command with sudo: sudo $0 --assign-... ${pid}"
        exit 1
    fi
    
    echo "${pid}" > "${target_path}/cgroup.procs"
    log_info "Successfully assigned PID ${pid} to '$(basename "${target_path}")' group."
}

# ------------------------------------------------------------------------------
# Status Dashboard
# ------------------------------------------------------------------------------

show_status() {
    if [ ! -d "${SLICE_PATH}" ]; then
        log_warn "SpaceShield slice hierarchy is not configured at ${SLICE_PATH}."
        exit 0
    fi
    
    echo "=============================================================================="
    echo " SpaceShield cgroups v2 Shielding Status"
    echo "=============================================================================="
    echo "Slice Path: ${SLICE_PATH}"
    echo
    
    for cg in "dsp" "dashboard" "logging"; do
        local cg_path="${SLICE_PATH}/${cg}"
        if [ -d "${cg_path}" ]; then
            echo "------------------------------------------------------------"
            echo "Group: [${cg}]"
            echo "------------------------------------------------------------"
            if [ -f "${cg_path}/cpu.weight" ]; then
                echo "  CPU Weight: $(cat "${cg_path}/cpu.weight")"
            fi
            if [ -f "${cg_path}/cpu.max" ]; then
                echo "  CPU Max: $(cat "${cg_path}/cpu.max")"
            fi
            if [ -f "${cg_path}/memory.max" ]; then
                echo "  Memory Max: $(cat "${cg_path}/memory.max") bytes"
            fi
            if [ -f "${cg_path}/memory.high" ]; then
                echo "  Memory High: $(cat "${cg_path}/memory.high") bytes"
            fi
            if [ -f "${cg_path}/memory.current" ]; then
                echo "  Memory Current Usage: $(cat "${cg_path}/memory.current") bytes"
            fi
            echo "  Assigned PIDs:"
            if [ -f "${cg_path}/cgroup.procs" ]; then
                cat "${cg_path}/cgroup.procs" | sed 's/^/    - /'
            fi
            echo
        fi
    done
    echo "=============================================================================="
}

# ------------------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------------------

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --setup                   Configure spaceshield.slice and sub-cgroups"
    echo "  --assign-dsp <pid>        Assign a PID to the dsp group"
    echo "  --assign-dashboard <pid>  Assign a PID to the dashboard group"
    echo "  --assign-logging <pid>    Assign a PID to the logging group"
    echo "  --status                  Show cgroups v2 shielding configuration and statistics"
    echo "  --help                    Show this help message"
}

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Parse options
while [ $# -gt 0 ]; do
    case "$1" in
        --setup)
            check_cgroups_v2
            if ! check_write_permissions; then
                log_warn "Exiting setup gracefully due to lack of write permissions."
                exit 0
            fi
            setup_cgroups
            exit 0
            ;;
        --assign-dsp)
            if [ -z "${2:-}" ]; then
                log_error "PID is required for --assign-dsp"
                exit 1
            fi
            assign_pid "${DSP_PATH}" "$2"
            shift 2
            ;;
        --assign-dashboard)
            if [ -z "${2:-}" ]; then
                log_error "PID is required for --assign-dashboard"
                exit 1
            fi
            assign_pid "${DASHBOARD_PATH}" "$2"
            shift 2
            ;;
        --assign-logging)
            if [ -z "${2:-}" ]; then
                log_error "PID is required for --assign-logging"
                exit 1
            fi
            assign_pid "${LOGGING_PATH}" "$2"
            shift 2
            ;;
        --status)
            show_status
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done
