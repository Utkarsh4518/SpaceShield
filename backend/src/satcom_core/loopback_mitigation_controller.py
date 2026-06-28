"""
Task 66.3: Closed-Loop Loopback Mitigation Controller Module
SpaceShield High-Velocity Receiver DSP Subsystem

Provides automated mitigation hooks for loopback instability, including sample-rate reduction,
handover postponement, tracker gain reduction, and operator escalation.
"""

import numpy as np
from numba import njit

# Bounded Control Action Codes
ACTION_NO_ACTION = 0
ACTION_REDUCE_SAMPLE_RATE = 1
ACTION_POSTPONE_HANDOVER = 2
ACTION_REDUCE_TRACKER_GAIN = 3
ACTION_ESCALATE_TO_OPERATOR = 4

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _evaluate_mitigation_decision_jit(
    sync_state: int,
    drift_s: float,
    event_tag: int,
    handover_active: int,
    consecutive_critical: int,
    consecutive_slips: int
) -> tuple[int, int, int]:
    """
    Zero-Heap JIT Mitigation Controller.
    Returns (action_type, updated_consecutive_critical, updated_consecutive_slips).
    """
    p_code = event_tag & 0x7
    
    # 1. Update stability counters
    new_critical = consecutive_critical
    new_slips = consecutive_slips
    
    if sync_state == 3:  # DESYNC
        new_critical += 1
    else:
        new_critical = max(0, new_critical - 1)
        
    if p_code == 2 or sync_state == 2:  # PURE_TRANSPORT_STALL or SLIP
        new_slips += 1
    else:
        new_slips = max(0, new_slips - 1)
        
    # 2. Decision Logic
    action = ACTION_NO_ACTION
    
    if new_critical >= 8:
        action = ACTION_ESCALATE_TO_OPERATOR
    elif new_slips >= 3:
        action = ACTION_REDUCE_SAMPLE_RATE
    elif handover_active == 1 and sync_state >= 2:
        action = ACTION_POSTPONE_HANDOVER
    elif p_code == 1 and abs(drift_s) > 0.003:
        action = ACTION_REDUCE_TRACKER_GAIN
    else:
        action = ACTION_NO_ACTION
        
    return action, new_critical, new_slips


class LoopbackMitigationController:
    """
    Triggers automated actions for SpaceShield hardware loops under stress.
    Monitors drift metrics and sequence tags on a zero-allocation JIT hot path.
    """
    def __init__(self, history_len: int = 100):
        self.history_len = history_len
        self.consecutive_critical = 0
        self.consecutive_slips = 0
        
        # Pre-allocated metrics arrays
        self.action_history = np.zeros(history_len, dtype=np.int32)
        self.history_idx = 0

    def process_telemetry_stride(
        self,
        sync_state: int,
        drift_s: float,
        event_tag: int,
        handover_active: int
    ) -> int:
        """
        Processes stride metrics and returns the selected mitigation action code.
        """
        action, new_crit, new_slips = _evaluate_mitigation_decision_jit(
            sync_state,
            drift_s,
            event_tag,
            handover_active,
            self.consecutive_critical,
            self.consecutive_slips
        )
        
        # Save counters
        self.consecutive_critical = new_crit
        self.consecutive_slips = new_slips
        
        # Record action history
        self.action_history[self.history_idx] = action
        self.history_idx = (self.history_idx + 1) % self.history_len
        
        return action


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Closed-Loop: Loopback Mitigation Controller Harness")
    print("==================================================================")
    
    controller = LoopbackMitigationController(history_len=20)
    
    # Scenario 1: Nominal Stable timing
    print("[*] Scenario 1: Stable Loopback Timing...")
    for step in range(5):
        act = controller.process_telemetry_stride(
            sync_state=0,
            drift_s=0.001,
            event_tag=0,  # CLEAN_OPERATION
            handover_active=0
        )
    print(f"    -> Action: {act}")
    assert act == ACTION_NO_ACTION, "Nominal loopback triggered false action!"
    print("    -> Nominal timing: [PASSED]")
    
    # Scenario 2: Repeated Slip Bursts
    print("\n[*] Scenario 2: Repeated Slip Bursts (Sample-Rate Mitigation)...")
    for step in range(3):
        act = controller.process_telemetry_stride(
            sync_state=2,  # SLIP
            drift_s=0.002,
            event_tag=2,   # PURE_TRANSPORT_STALL (slipping sequence)
            handover_active=0
        )
    print(f"    -> Action: {act}")
    assert act == ACTION_REDUCE_SAMPLE_RATE, "Failed to trigger rate reduction under sequence slips!"
    print("    -> Sample-rate mitigation: [PASSED]")
    
    # Scenario 3: Recover to Nominal and verify clearing
    print("\n[*] Scenario 3: Recovering to Nominal...")
    for step in range(5):
        act = controller.process_telemetry_stride(
            sync_state=0,
            drift_s=0.001,
            event_tag=0,
            handover_active=0
        )
    print(f"    -> Action: {act} | Slip counter: {controller.consecutive_slips}")
    assert act == ACTION_NO_ACTION, "Failed to clear actions upon recovery!"
    print("    -> Recovery clearing: [PASSED]")

    # Scenario 4: Timing Unstable During Handover (Postponement check)
    print("\n[*] Scenario 4: Handover Postponement...")
    act = controller.process_telemetry_stride(
        sync_state=2,  # SLIP
        drift_s=0.002,
        event_tag=0,
        handover_active=1  # Handover is currently running!
    )
    print(f"    -> Action: {act}")
    assert act == ACTION_POSTPONE_HANDOVER, "Failed to postpone handover during timing instabilities!"
    print("    -> Handover postponement: [PASSED]")

    # Scenario 5: Persistent Stall Conditions (Operator Escalation)
    print("\n[*] Scenario 5: Persistent Stall Escalation...")
    for step in range(8):
        act = controller.process_telemetry_stride(
            sync_state=3,  # DESYNC
            drift_s=0.016,
            event_tag=2,   # PURE_TRANSPORT_STALL
            handover_active=0
        )
    print(f"    -> Action: {act} | Critical count: {controller.consecutive_critical}")
    assert act == ACTION_ESCALATE_TO_OPERATOR, "Failed to escalate to operator under persistent stall!"
    print("    -> Operator escalation: [PASSED]")

    print("\n[+] Loopback mitigation controller validation complete.")
