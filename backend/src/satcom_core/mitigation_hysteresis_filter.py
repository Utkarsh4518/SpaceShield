"""
Task 67.1: Mitigation Hysteresis Filter Module
SpaceShield High-Velocity Receiver DSP Subsystem

Provides temporal debouncing to smooth mitigation action switches.
Prevents state machine oscillation and hunting under marginal timing conditions.
"""

import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _evaluate_hysteresis_jit(
    raw_action: int,
    action_history: np.ndarray,
    history_idx: int,
    history_len: int,
    current_filtered_action: int,
    action_duration: int,
    up_window: int,
    down_window: int
) -> tuple[int, int, int]:
    """
    Zero-Heap JIT Debounce/Hysteresis Filter:
    Returns (filtered_action, updated_idx, updated_duration).
    """
    # 1. Save raw action in history
    action_history[history_idx] = raw_action
    
    # 2. Count occurrences of raw_action in the history window
    new_action = current_filtered_action
    duration = action_duration
    
    if raw_action != current_filtered_action:
        # Check transition constraints
        if raw_action != 0:
            # Transitioning to an alert/mitigation action: requires 'up_window' consecutive requests
            consecutive = True
            idx = history_idx
            for _ in range(up_window):
                if action_history[idx] != raw_action:
                    consecutive = False
                    break
                idx = (idx - 1 + history_len) % history_len
                
            if consecutive:
                new_action = raw_action
                duration = 1
            else:
                duration = action_duration + 1
        else:
            # Clearing back to CLEAN (0): requires 'down_window' consecutive nominal actions
            consecutive = True
            idx = history_idx
            for _ in range(down_window):
                if action_history[idx] != 0:
                    consecutive = False
                    break
                idx = (idx - 1 + history_len) % history_len
                
            if consecutive:
                new_action = 0
                duration = 1
            else:
                duration = action_duration + 1
    else:
        duration = action_duration + 1
        
    return new_action, (history_idx + 1) % history_len, duration


class MitigationHysteresisFilter:
    """
    Filters raw action outputs from LoopbackMitigationController.
    Uses pre-allocated arrays and JIT-debounced decisions.
    """
    def __init__(self, history_len: int = 50, up_window: int = 3, down_window: int = 5):
        self.history_len = history_len
        self.up_window = up_window
        self.down_window = down_window
        
        # Pre-allocated variables
        self.action_history = np.zeros(history_len, dtype=np.int32)
        self.history_idx = 0
        self.current_filtered_action = 0
        self.action_duration = 0

    def filter_action(self, raw_action: int) -> int:
        """
        Filters a raw action stride.
        """
        new_act, next_idx, duration = _evaluate_hysteresis_jit(
            raw_action,
            self.action_history,
            self.history_idx,
            self.history_len,
            self.current_filtered_action,
            self.action_duration,
            self.up_window,
            self.down_window
        )
        
        self.current_filtered_action = new_act
        self.history_idx = next_idx
        self.action_duration = duration
        
        return new_act


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Closed-Loop: Mitigation Hysteresis Validation")
    print("==================================================================")
    
    # 3-step up debounce, 5-step down debounce
    filter_unit = MitigationHysteresisFilter(history_len=20, up_window=3, down_window=5)
    
    # 1. Stable nominal
    print("[*] Scenario 1: Stable Nominal inputs...")
    for _ in range(5):
        filtered = filter_unit.filter_action(0)
    print(f"    -> Filtered Action: {filtered}")
    assert filtered == 0, "Nominal input filtered incorrectly!"
    print("    -> Stable nominal: [PASSED]")
    
    # 2. Transient spike (single raw action of REDUCE_RATE = 1)
    print("\n[*] Scenario 2: Transient Trigger Suppression (Spike)...")
    # Single stride raw trigger (should be ignored by up_window=3)
    filtered = filter_unit.filter_action(1)
    print(f"    -> Filtered Action: {filtered} | Duration: {filter_unit.action_duration}")
    assert filtered == 0, "Failed to suppress transient raw action!"
    print("    -> Transient suppression: [PASSED]")
    
    # 3. Persistent slip burst (3 consecutive steps of raw action 1)
    print("\n[*] Scenario 3: Persistent Action Transition (Sustained)...")
    # Step 1 was already run in Scenario 2. Run two more:
    filter_unit.filter_action(1)
    filtered = filter_unit.filter_action(1)
    print(f"    -> Filtered Action: {filtered} | Duration: {filter_unit.action_duration}")
    assert filtered == 1, "Failed to switch to active mitigation after sustained window!"
    print("    -> Sustained action activation: [PASSED]")
    
    # 4. Alternating / Hunting condition (rapid recovery attempt)
    print("\n[*] Scenario 4: Borderline Hunting / Oscillation Suppression...")
    # Alternating raw actions: 0, 1, 0, 1 (should stay at 1 since down_window=5)
    for step in range(4):
        raw = 0 if (step % 2 == 0) else 1
        filtered = filter_unit.filter_action(raw)
        
    print(f"    -> Filtered Action: {filtered} | Duration: {filter_unit.action_duration}")
    assert filtered == 1, "Failed to prevent hunting under borderline oscillations!"
    print("    -> Hunting suppression: [PASSED]")

    # 5. Full recovery to nominal (5 consecutive nominals)
    print("\n[*] Scenario 5: Full Recovery to Nominal...")
    for _ in range(5):
        filtered = filter_unit.filter_action(0)
    print(f"    -> Filtered Action: {filtered} | Duration: {filter_unit.action_duration}")
    assert filtered == 0, "Failed to clear action after sustained recovery window!"
    print("    -> Recovery clearing: [PASSED]")

    print("\n[+] Mitigation hysteresis filter validation complete.")
