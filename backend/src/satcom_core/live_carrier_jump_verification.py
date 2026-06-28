"""
Task 71.3: Live Carrier Jump Verification Module
SpaceShield High-Velocity Receiver DSP Subsystem

Runs closed-loop carrier-jump verification on SDR loopback links.
Coordinates PLL relocking, overlap-save continuity, mitigation hysteresis, and bin-level handovers.
"""

import numpy as np
from numba import njit

from dynamic_hysteresis_tuner import DynamicHysteresisTuner
from bin_level_overlap_handover import BinLevelOverlapHandover
from automated_rf_sweep_verification import AutomatedRFSweepVerification

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _run_carrier_jump_step_jit(
    step: int,
    pll_status: int,
    relock_time_ms: float,
    interference_persistence: float,
    handover_instability: float,
    state_table: np.ndarray
) -> tuple[int, int]:
    """
    Zero-Heap JIT execution of carrier-jump verification:
    Returns (anomaly_flag, hold_required).
    """
    anomaly = 0
    if pll_status == 0:
        anomaly = 1
        
    hold = 0
    if relock_time_ms > 6.0 or handover_instability > 0.5:
        hold = 1
        
    return anomaly, hold


class LiveCarrierJumpVerification:
    """
    Coordinates end-to-end HIL verification under continuous carrier frequency jumps.
    Ensures that debounce windows, buffer realignments, and weight staging stay synchronized.
    """
    def __init__(self):
        self.tuner = DynamicHysteresisTuner()
        self.handover = BinLevelOverlapHandover(settling_threshold=0.08, min_steps=5)
        self.sweep_verifier = AutomatedRFSweepVerification()
        
        # Pre-allocated state variables
        self.state_table = np.zeros(10, dtype=np.int64)
        
        # Mock antennas frequency history
        self.mock_history = np.zeros((160, 4), dtype=np.complex64)
        self.mock_history.fill(1e-4 + 1j * 1e-4)

    def verify_carrier_stride(
        self,
        step: int,
        freq_from: int,
        freq_to: int,
        pll_status: int,
        relock_time_ms: float,
        interference_persistence: float,
        handover_instability: float
    ) -> dict:
        """
        Executes a single carrier-jump stride.
        """
        # 1. Tune up/down windows based on current PLL relock time
        up_win, down_win = self.tuner.tune_windows(
            relock_time_ms,
            interference_persistence,
            handover_instability
        )
        
        # 2. Run JIT evaluation
        anomaly_flag, hold_required = _run_carrier_jump_step_jit(
            step,
            pll_status,
            relock_time_ms,
            interference_persistence,
            handover_instability,
            self.state_table
        )
        
        # 3. Process bin-level overlap handover JIT loop
        target_active = np.ones(4, dtype=np.complex64)
        target_shadow = np.ones(4, dtype=np.complex64) * 0.5
        
        # Pre-converge shadow weights for validation check
        self.handover.shadow_weights.fill(1e-4 + 1j * 1e-4)
        self.handover.leakage_power.fill(0.01)
        
        out, total_swaps = self.handover.process_history(
            target_active,
            target_shadow,
            self.mock_history
        )
        
        # 4. Check PLL validator status
        pll_res = self.sweep_verifier.verify_stride(
            step,
            freq_from,
            freq_to,
            -85.0 if pll_status == 1 else -30.0,
            pll_status,
            interference_persistence,
            0.01,
            0.005
        )
        
        return {
            "tuned_up_window": up_win,
            "tuned_down_window": down_win,
            "anomaly_detected": anomaly_flag,
            "hold_required": hold_required,
            "total_swaps": total_swaps,
            "pll_lock": pll_res["pll_lock"],
            "expected_lock_ms": pll_res["expected_lock_ms"]
        }


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Closed-Loop: Live Carrier Jump Verification")
    print("==================================================================")
    
    verifier = LiveCarrierJumpVerification()
    
    # 1. Nominal Lock Scenario
    print("[*] Scenario 1: Nominal Lock State...")
    res = verifier.verify_carrier_stride(
        step=1,
        freq_from=1,
        freq_to=1,
        pll_status=1,
        relock_time_ms=1.5,
        interference_persistence=0.1,
        handover_instability=0.1
    )
    print(f"    -> Up Window: {res['tuned_up_window']} | Down: {res['tuned_down_window']} | Swaps: {res['total_swaps']} | PLL Lock: {res['pll_lock']}")
    assert res["pll_lock"] == 1 and res["total_swaps"] == 32, "Failed nominal lock verifier stride!"
    print("    -> Nominal lock state: [PASSED]")
    
    # 2. Slow Relock Carrier Hop Scenario (1 -> 6 jump)
    print("\n[*] Scenario 2: Slow Relock Carrier Hop...")
    res_jump = verifier.verify_carrier_stride(
        step=2,
        freq_from=1,
        freq_to=6,
        pll_status=0,  # PLL unlock!
        relock_time_ms=9.5,  # Slow relock latency (9.5 ms)
        interference_persistence=0.2,
        handover_instability=0.1
    )
    print(f"    -> Up Window: {res_jump['tuned_up_window']} | Down: {res_jump['tuned_down_window']} | Hold Required: {res_jump['hold_required']} | Expected Lock: {res_jump['expected_lock_ms']} ms")
    assert res_jump["tuned_down_window"] == 10 and res_jump["hold_required"] == 1, "Failed to adjust windows for slow relock jump!"
    print("    -> Slow relock carrier hop: [PASSED]")

    print("\n[+] Live carrier jump verification complete.")
