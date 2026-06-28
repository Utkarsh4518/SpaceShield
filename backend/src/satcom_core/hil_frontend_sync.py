"""
Task 61.3: Hardware-in-the-Loop (HIL) Frontend Synchronization Module
SpaceShield High-Velocity Receiver DSP Subsystem

Evaluates SDR hardware timing drift, PCIe/USB ingestion latency anomalies,
and frame continuity. Maintains zero-heap tracking arrays.
"""

import time
import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _evaluate_sync_jit(
    drift_history: np.ndarray,
    history_len: int,
    last_seq: int,
    curr_seq: int,
    sdr_ts: float,
    sys_ts: float,
    drift_warning_thresh: float,
    clock_slip_thresh: float
) -> tuple[float, float, int, int]:
    """
    Zero-Heap JIT Sync Evaluator:
    Tracks sequence continuity and drift variance without memory allocations.
    Returns: (drift_val, jitter_std, dropped_frames, state_verdict)
    State Verdict:
      0: NOMINAL (stable sample clock synchronization)
      1: DRIFT_WARNING (moderate timing jitter observed)
      2: CLOCK_SLIP (sudden clock jump detected)
      3: CRITICAL_DESYNC (substantial drift or massive packet loss)
    """
    # 1. Sequence checks
    dropped_frames = 0
    if last_seq > 0:
        gap = curr_seq - last_seq - 1
        if gap > 0:
            dropped_frames = gap
            
    # 2. Drift calculation (system timestamp - SDR hardware time)
    drift_val = sys_ts - sdr_ts
    
    # Update sliding window history in place (shift oldest out)
    if history_len > 0:
        for i in range(history_len - 1):
            drift_history[i] = drift_history[i + 1]
        drift_history[history_len - 1] = drift_val
        
    # Compute mean and standard deviation of drift
    sum_drift = 0.0
    for i in range(history_len):
        sum_drift += drift_history[i]
    mean_drift = sum_drift / history_len
    
    sum_sq_diff = 0.0
    for i in range(history_len):
        diff = drift_history[i] - mean_drift
        sum_sq_diff += diff * diff
    jitter_std = math.sqrt(sum_sq_diff / history_len)
    
    # 3. Determine state verdict
    verdict = 0
    
    if history_len > 1:
        last_drift = drift_history[history_len - 2]
        drift_jump = abs(drift_val - last_drift)
        
        # Check thresholds
        if drift_jump >= clock_slip_thresh:
            verdict = 2 # CLOCK_SLIP
        elif jitter_std >= drift_warning_thresh:
            verdict = 1 # DRIFT_WARNING
            
    if dropped_frames > 10 or (history_len > 1 and abs(drift_val - drift_history[0]) > 0.050):
        verdict = 3 # CRITICAL_DESYNC
        
    return drift_val, jitter_std, dropped_frames, verdict


class HILFrontendSyncTracker:
    """
    SpaceShield Hardware-in-the-Loop Frontend Timing Monitor.
    Tracks clock drift, latency bounds, and frame drops.
    """
    def __init__(
        self,
        history_window: int = 50,
        drift_warning_thresh: float = 0.002,  # 2 ms jitter
        clock_slip_thresh: float = 0.005      # 5 ms jump
    ):
        self.history_window = history_window
        self.drift_warning_thresh = drift_warning_thresh
        self.clock_slip_thresh = clock_slip_thresh
        
        # Pre-allocated drift history
        self.drift_history = np.zeros(self.history_window, dtype=np.float64)
        
        # State tracking registers
        self.last_sequence = 0
        self.total_dropped_frames = 0
        self.recovery_events_count = 0
        self.current_state = 0 # NOMINAL
        
        # Pre-warm JIT compiler
        self._warmup()

    def _warmup(self):
        """Forces compilation of Numba synchronization routines."""
        _evaluate_sync_jit(
            self.drift_history, 5, 0, 1, 0.0, 0.0,
            self.drift_warning_thresh, self.clock_slip_thresh
        )
        self.drift_history.fill(0.0)

    def evaluate_stride(
        self,
        sdr_timestamp: float,
        system_timestamp: float,
        sequence_num: int
    ) -> dict:
        """
        Ingests a frame's timing metrics.
        Performs non-blocking drift analysis and state evaluation.
        """
        # Initialize drift history on the very first frame to avoid startup transients
        if self.last_sequence == 0:
            self.drift_history.fill(system_timestamp - sdr_timestamp)
            
        # Run JIT logic
        drift, jitter, dropped, next_state = _evaluate_sync_jit(
            self.drift_history,
            self.history_window,
            self.last_sequence,
            sequence_num,
            sdr_timestamp,
            system_timestamp,
            self.drift_warning_thresh,
            self.clock_slip_thresh
        )
        
        # Detect state recovery
        if self.current_state in (1, 2, 3) and next_state == 0:
            self.recovery_events_count += 1
            
        self.current_state = next_state
        self.last_sequence = sequence_num
        self.total_dropped_frames += dropped
        
        # Bounded age calculation
        buffer_age = time.time() - system_timestamp

        return {
            "drift_seconds": drift,
            "jitter_seconds": jitter,
            "dropped_frames": dropped,
            "sync_state": next_state,
            "buffer_age_seconds": buffer_age,
            "total_drops": self.total_dropped_frames,
            "recovery_events": self.recovery_events_count
        }


# =========================================================================
# DETERMINISTIC SIMULATION AND REPLAY HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: HIL Frontend Sync Validation")
    print("==================================================================")
    
    tracker = HILFrontendSyncTracker(history_window=10)
    
    # Helper to print sync state label
    labels = {0: "NOMINAL", 1: "DRIFT_WARNING", 2: "CLOCK_SLIP", 3: "CRITICAL_DESYNC"}

    # 1. Nominal Stable Operation (Steps 1-15)
    print("[*] Scenario 1: Stable Nominal Sky Stream...")
    nominal_sdr_time = 1000.0
    nominal_sys_time = 1000.02  # 20ms fixed path delay
    
    for step in range(15):
        sdr_t = nominal_sdr_time + step * 0.010  # 10ms sampling interval
        sys_t = nominal_sys_time + step * 0.010
        metrics = tracker.evaluate_stride(sdr_t, sys_t, step + 1)
        
    print(f"    -> Sync State: {labels[tracker.current_state]} | Jitter: {metrics['jitter_seconds']*1e6:.2f} µs")
    assert tracker.current_state == 0, "Nominal loop failed to remain synchronized!"
    print("    -> Nominal stream checks: [PASSED]")

    # 2. Clock Slip Simulation (Step 16)
    print("\n[*] Scenario 2: Frontend Sample Clock Slip...")
    # SDR timestamp suddenly jumps forward by 8ms (clock slip)
    slip_sdr_time = nominal_sdr_time + 15 * 0.010 - 0.008  # 8ms lag
    sys_t = nominal_sys_time + 15 * 0.010
    metrics = tracker.evaluate_stride(slip_sdr_time, sys_t, 16)
    print(f"    -> Sync State: {labels[tracker.current_state]} | Drift Jump: {abs(metrics['drift_seconds'] - 0.02)*1000:.2f} ms")
    assert tracker.current_state == 2, "Failed to identify frontend clock slip!"
    print("    -> Clock slip checks: [PASSED]")

    # 3. Transient PCIe / USB Delay (Steps 17-20)
    print("\n[*] Scenario 3: Transient PCIe Connection Delay & Queue Recovery...")
    # Connection stalls for 4 strides (no data arrives), then bursts
    # We resume at sequence 21 (representing 4 dropped frames)
    metrics = tracker.evaluate_stride(
        nominal_sdr_time + 20 * 0.010,
        nominal_sys_time + 20 * 0.010,
        21
    )
    print(f"    -> Sync State: {labels[tracker.current_state]} | Sequence Gaps: {metrics['dropped_frames']}")
    assert metrics['dropped_frames'] == 4, "Failed to track dropped frame gaps!"
    
    # Check that we recovery back to nominal
    print("[*] Performing recovery stream strides...")
    for step in range(21, 35):
        sdr_t = nominal_sdr_time + step * 0.010
        sys_t = nominal_sys_time + step * 0.010
        metrics = tracker.evaluate_stride(sdr_t, sys_t, step + 1)
        
    print(f"    -> Restored Sync State: {labels[tracker.current_state]}")
    print(f"    -> Total drops:         {tracker.total_dropped_frames}")
    print(f"    -> Recovery count:       {tracker.recovery_events_count}")
    assert tracker.current_state == 0, "Failed to recover back to Nominal state!"
    assert tracker.recovery_events_count == 1, "Failed to register sync recovery event!"
    print("    -> Connection delay checks: [PASSED]")

    print("\n[+] HIL frontend synchronization verification successfully completed.")
