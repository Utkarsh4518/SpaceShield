"""
Task 71.1: Dynamic Hysteresis Tuner Module
SpaceShield High-Velocity Receiver DSP Subsystem

Dynamically adjusts the up and down debouncing windows of the hysteresis filter
based on current PLL relock metrics, interference profiles, and stability values.
"""

import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _tune_hysteresis_windows_jit(
    relock_time_ms: float,
    interference_persistence: float,
    handover_instability: float,
    fallback_up: int,
    fallback_down: int
) -> tuple[int, int]:
    """
    Zero-Heap JIT tuning loop for filter windows.
    Returns (up_window, down_window).
    """
    up = fallback_up
    down = fallback_down
    
    # 1. Scale down_window based on PLL relock speed to prevent hunting during recovery
    if relock_time_ms > 0.0:
        if relock_time_ms > 8.0:
            down = 10
        elif relock_time_ms > 4.0:
            down = 7
        else:
            down = 4
            
    # 2. Scale up_window based on interference persistence and stability
    if interference_persistence > 0.7 or handover_instability > 0.6:
        up = 6
    elif interference_persistence > 0.3:
        up = 4
    else:
        up = 2
        
    return up, down


class DynamicHysteresisTuner:
    """
    Dynamically adjusts debouncing filters to protect the transceiver PLL control loop.
    Enforces larger down-windows when recovery times are slow.
    """
    def __init__(self, fallback_up: int = 3, fallback_down: int = 5):
        self.fallback_up = fallback_up
        self.fallback_down = fallback_down
        
        self.active_up = fallback_up
        self.active_down = fallback_down

    def tune_windows(
        self,
        relock_time_ms: float,
        interference_persistence: float,
        handover_instability: float
    ) -> tuple[int, int]:
        """
        Calculates and returns the tuned up/down filter windows.
        """
        up, down = _tune_hysteresis_windows_jit(
            relock_time_ms,
            interference_persistence,
            handover_instability,
            self.fallback_up,
            self.fallback_down
        )
        
        self.active_up = up
        self.active_down = down
        return up, down


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Closed-Loop: Dynamic Hysteresis Tuner Validation")
    print("==================================================================")
    
    tuner = DynamicHysteresisTuner()
    
    # 1. Fast Relock & Low interference (Nominal stable state)
    print("[*] Scenario 1: Fast Relock Nominal state...")
    up, down = tuner.tune_windows(
        relock_time_ms=1.5,
        interference_persistence=0.1,
        handover_instability=0.1
    )
    print(f"    -> Up Window: {up} | Down Window: {down}")
    assert up == 2 and down == 4, "Incorrect windows loaded for fast relock!"
    print("    -> Fast relock config: [PASSED]")
    
    # 2. Slow Relock (Requires larger down window to clear mitigations safely)
    print("\n[*] Scenario 2: Slow Relock configuration...")
    up, down = tuner.tune_windows(
        relock_time_ms=9.5,  # 9.5 ms relock
        interference_persistence=0.1,
        handover_instability=0.1
    )
    print(f"    -> Up Window: {up} | Down Window: {down}")
    assert up == 2 and down == 10, "Failed to expand down window for slow relock!"
    print("    -> Slow relock config: [PASSED]")
    
    # 3. Oscillatory / Persistent Interference
    print("\n[*] Scenario 3: Oscillatory Jammer profile...")
    up, down = tuner.tune_windows(
        relock_time_ms=1.5,
        interference_persistence=0.85,  # high persistence
        handover_instability=0.1
    )
    print(f"    -> Up Window: {up} | Down Window: {down}")
    assert up == 6 and down == 4, "Failed to expand up window under persistent interference!"
    print("    -> Oscillatory config: [PASSED]")

    print("\n[+] Dynamic hysteresis tuner validation complete.")
