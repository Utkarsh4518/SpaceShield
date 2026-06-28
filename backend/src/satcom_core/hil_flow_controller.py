"""
Task 62.2: HIL Flow Controller Module
SpaceShield High-Velocity Receiver DSP Subsystem

Applies dynamic sample-rate corrections based on HIL sync telemetry.
Maintains timing alignment between SDR hardware frontends and compute planes.
"""

import time
import numpy as np
from numba import njit

from hil_frontend_sync import HILFrontendSyncTracker

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _evaluate_flow_control_jit(
    sync_state: int,
    drift: float,
    jitter: float,
    dropped: int,
    last_action: int,
    consecutive_nominal_count: int
) -> tuple[int, int]:
    """
    Zero-Heap Numba JIT Flow Control Evaluator:
    Returns (action_verdict, updated_consecutive_nominal)
    Action Verdicts:
      0: NOMINAL_CONTINUATION (keep sample rate at standard 100%)
      1: HOLD_SAMPLE_FLOW (temporarily throttle to clear buffer congestion)
      2: REDUCE_SAMPLE_RATE (scale down sample rate factor)
      3: RECOVERY_BURST (increase rate factor to catch up after a stall)
    """
    action = 0
    next_nominal = consecutive_nominal_count
    
    if sync_state == 3 or dropped > 5:
        action = 2  # REDUCE_SAMPLE_RATE
        next_nominal = 0
    elif sync_state == 2:
        action = 1  # HOLD_SAMPLE_FLOW
        next_nominal = 0
    elif sync_state == 1:
        action = 2  # REDUCE_SAMPLE_RATE
        next_nominal = 0
    else:  # sync_state == 0
        # If the last action was REDUCE or HOLD, trigger a RECOVERY_BURST to catch up
        if last_action in (1, 2):
            action = 3  # RECOVERY_BURST
            next_nominal = 0
        elif last_action == 3:
            # Maintain recovery burst for exactly 5 strides before returning to nominal
            if consecutive_nominal_count < 5:
                action = 3
                next_nominal = consecutive_nominal_count + 1
            else:
                action = 0
                next_nominal = 0
        else:
            action = 0
            next_nominal = 0
            
    return action, next_nominal


class HILFlowController:
    """
    Monitors HIL sync drift telemetry and computes corrective rate adjustments
    for the physical or simulated SDR frontend.
    """
    def __init__(self, history_window: int = 50):
        self.tracker = HILFrontendSyncTracker(history_window=history_window)
        self.last_action = 0  # NOMINAL_CONTINUATION
        self.consecutive_nominal_count = 0
        self.current_rate_factor = 1.0  # Multiplier scale (0.80 to 1.00)

    def process_stride(
        self,
        sdr_timestamp: float,
        system_timestamp: float,
        sequence_num: int
    ) -> dict:
        """
        Processes a single HIL stride:
        1. Evaluates drift and synchronization state.
        2. Computes the dynamic flow control action.
        3. Updates the target sample-rate multiplier.
        """
        metrics = self.tracker.evaluate_stride(sdr_timestamp, system_timestamp, sequence_num)
        
        # JIT-evaluate corrective action
        action, next_nominal = _evaluate_flow_control_jit(
            metrics["sync_state"],
            metrics["drift_seconds"],
            metrics["jitter_seconds"],
            metrics["dropped_frames"],
            self.last_action,
            self.consecutive_nominal_count
        )
        
        self.last_action = action
        self.consecutive_nominal_count = next_nominal
        
        # Update target rate factor (multiplier)
        if action == 2:  # REDUCE
            self.current_rate_factor = max(0.80, self.current_rate_factor - 0.05)
        elif action == 3:  # RECOVERY
            self.current_rate_factor = min(1.00, self.current_rate_factor + 0.05)
            
        return {
            "sync_metrics": metrics,
            "flow_action": action,
            "rate_factor": self.current_rate_factor
        }


# =========================================================================
# DETERMINISTIC REPLAYABLE VALIDATION HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: HIL Flow Controller Validation")
    print("==================================================================")
    
    controller = HILFlowController(history_window=10)
    
    action_labels = {
        0: "NOMINAL_CONTINUATION",
        1: "HOLD_SAMPLE_FLOW",
        2: "REDUCE_SAMPLE_RATE",
        3: "RECOVERY_BURST"
    }
    
    # 1. Stable Timing (Strides 1-10)
    print("[*] Scenario 1: Stable Clock Synchronization...")
    base_sdr = 2000.0
    base_sys = 2000.01
    
    for step in range(10):
        res = controller.process_stride(
            base_sdr + step * 0.010,
            base_sys + step * 0.010,
            step + 1
        )
        
    print(f"    -> Flow Action: {action_labels[res['flow_action']]} | Rate Factor: {res['rate_factor']:.2f}")
    assert res['flow_action'] == 0 and res['rate_factor'] == 1.0, "Stable timing check failed!"
    print("    -> Stable timing: [PASSED]")
    
    # 2. Gradual Drift Jitter (Strides 11-15)
    print("\n[*] Scenario 2: Gradual Drift and Jitter Growth...")
    for step in range(10, 15):
        # Gradual linear drift (1.5ms per step)
        noise = (step - 10) * 0.0015
        jitter_sdr = base_sdr + step * 0.010 - noise
        res = controller.process_stride(
            jitter_sdr,
            base_sys + step * 0.010,
            step + 1
        )
        
    print(f"    -> Flow Action: {action_labels[res['flow_action']]} | Rate Factor: {res['rate_factor']:.2f}")
    assert res['flow_action'] == 2 and res['rate_factor'] < 1.0, "Drift correction failed!"
    print("    -> Gradual drift correction: [PASSED]")
    
    # 3. Abrupt Drift Jump / Clock Slip (Stride 16)
    print("\n[*] Scenario 3: Clock Slip Timing Transition...")
    res = controller.process_stride(
        base_sdr + 15 * 0.010 - 0.016,  # 16ms slip (10ms jump from step 15)
        base_sys + 15 * 0.010,
        16
    )
    print(f"    -> Flow Action: {action_labels[res['flow_action']]} | Rate Factor: {res['rate_factor']:.2f}")
    assert res['flow_action'] == 1, "Clock slip flow throttling failed!"
    print("    -> Clock slip mitigation: [PASSED]")
    
    # 4. Connection Stall and Recovery (Strides 17-25)
    print("\n[*] Scenario 4: PCIe Connection Stall & Queue Catch-up...")
    # Resume after a 4-frame drop gap at sequence 21
    res = controller.process_stride(
        base_sdr + 20 * 0.010,
        base_sys + 20 * 0.010,
        21
    )
    print(f"    -> Stall Catch-up Flow Action: {action_labels[res['flow_action']]} | Rate Factor: {res['rate_factor']:.2f}")
    assert res['flow_action'] == 1 or res['flow_action'] == 2, "Stall warning state failed!"
    
    # Run recovery strides to restore nominal operation
    print("[*] Running recovery strides to restore NOMINAL state...")
    recovered = False
    for step in range(21, 36):
        res = controller.process_stride(
            base_sdr + step * 0.010,
            base_sys + step * 0.010,
            step + 1
        )
        if res['flow_action'] == 3:
            recovered = True
            
    print(f"    -> Post-Recovery Action: {action_labels[res['flow_action']]} | Rate Factor: {res['rate_factor']:.2f}")
    assert recovered, "Recovery burst failed to trigger!"
    assert res['flow_action'] == 0 and res['rate_factor'] == 1.0, "Failed to restore NOMINAL_CONTINUATION state!"
    print("    -> Connection stall recovery: [PASSED]")
    
    print("\n[+] HIL flow controller validation successfully completed.")
