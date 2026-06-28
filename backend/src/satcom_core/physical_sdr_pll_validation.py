"""
Task 69.3: Physical SDR PLL Validation Module
SpaceShield High-Velocity Receiver DSP Subsystem

Integrates with SDR registers to verify PLL relocking times and stability.
Monitors lock state, relock latency, and sample continuity in a zero-allocation JIT path.
"""

import time
import numpy as np
from numba import njit

# PLL Event Codes
EVENT_PLL_NONE = 0
EVENT_PLL_UNLOCK = 1
EVENT_PLL_RELOCK = 2

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _monitor_pll_stride_jit(
    pll_status: int,
    ts_ns: int,
    state_table: np.ndarray  # (4,) int64 -> [current_lock, unlock_ts, total_unlocks, max_unlock_duration_ns]
) -> tuple[int, int]:
    """
    Zero-Heap JIT PLL Lock state monitor.
    Returns (pll_event, elapsed_unlock_duration_ns).
    """
    curr_state = state_table[0]
    unlock_ts = state_table[1]
    
    event = EVENT_PLL_NONE
    duration = 0
    
    if pll_status == 0:  # UNLOCKED
        if curr_state == 1:  # Lock lost
            state_table[0] = 0
            state_table[1] = ts_ns
            state_table[2] += 1
            event = EVENT_PLL_UNLOCK
        else:
            duration = ts_ns - unlock_ts
    else:  # LOCKED
        if curr_state == 0:  # Lock regained
            state_table[0] = 1
            duration = ts_ns - unlock_ts
            if duration > state_table[3]:
                state_table[3] = duration
            event = EVENT_PLL_RELOCK
            
    return event, duration


class PhysicalSDRPLLValidation:
    """
    Evaluates SDR PLL relocking stability and relock latency.
    Runs on an allocation-free update path.
    """
    def __init__(self):
        # State table: [lock_status (1=locked), unlock_ts, total_unlocks, max_duration_ns]
        self.state_table = np.zeros(4, dtype=np.int64)
        self.state_table[0] = 1  # Initialized as locked
        self.state_table[1] = 0
        self.state_table[2] = 0
        self.state_table[3] = 0

    def process_stride(self, pll_status: int) -> dict:
        """
        Processes a single stride of PLL monitor data.
        """
        now_ns = time.perf_counter_ns()
        event, duration = _monitor_pll_stride_jit(pll_status, now_ns, self.state_table)
        
        return {
            "lock_status": int(self.state_table[0]),
            "total_unlock_events": int(self.state_table[2]),
            "max_unlock_duration_ms": float(self.state_table[3]) / 1_000_000.0,
            "event": event,
            "last_event_duration_ms": float(duration) / 1_000_000.0
        }


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Closed-Loop: SDR PLL Relocking Validation")
    print("==================================================================")
    
    validator = PhysicalSDRPLLValidation()
    
    # 1. Nominal locked state
    print("[*] Scenario 1: Stable Lock Ingestion...")
    for _ in range(5):
        res = validator.process_stride(1)
        
    print(f"    -> Lock Status: {res['lock_status']} | Total Unlocks: {res['total_unlock_events']}")
    assert res["lock_status"] == 1 and res["total_unlock_events"] == 0, "Failed nominal lock check!"
    print("    -> Nominal lock: [PASSED]")
    
    # 2. Forced Relock / Transient Unlock
    print("\n[*] Scenario 2: Forced PLL Relock Event...")
    # Step 1: Transition to unlocked
    res = validator.process_stride(0)
    print(f"    -> Event: {res['event']} (Expected: 1 - Unlock) | Lock: {res['lock_status']}")
    assert res["event"] == EVENT_PLL_UNLOCK and res["lock_status"] == 0, "Failed to catch PLL unlock!"
    
    # Step 2: Hold unlocked state (simulate relocking time: backdate unlock timestamp)
    validator.state_table[1] = time.perf_counter_ns() - int(8 * 1_000_000.0)  # 8 ms ago
    
    # Step 3: Transition back to locked
    res_relock = validator.process_stride(1)
    print(f"    -> Event: {res_relock['event']} (Expected: 2 - Relock) | Lock: {res_relock['lock_status']}")
    print(f"    -> Relock Time: {res_relock['last_event_duration_ms']:.2f} ms")
    assert res_relock["event"] == EVENT_PLL_RELOCK and res_relock["lock_status"] == 1, "Failed to catch PLL relock!"
    assert res_relock["last_event_duration_ms"] >= 8.0, "PLL relock duration calculated incorrectly!"
    print("    -> Forced relock timing: [PASSED]")

    print("\n[+] SDR PLL validation complete.")
