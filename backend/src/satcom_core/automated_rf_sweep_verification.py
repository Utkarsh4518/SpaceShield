"""
Task 70.3: Automated RF Sweep Verification Module
SpaceShield High-Velocity Receiver DSP Subsystem

Coordinates closed-loop sweep validations combining Fast-Lock lookup,
Dynamic Overlap Buffer alignment, and Physical PLL monitoring.
"""

import numpy as np
from numba import njit

from fast_lock_calibration_table import FastLockCalibrationTable
from dynamic_overlap_window_manager import DynamicOverlapWindowManager
from adaptive_overlap_tuner import AdaptiveOverlapTuner
from physical_sdr_pll_validation import PhysicalSDRPLLValidation

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _run_sweep_step_jit(
    step: int,
    rf_power_dbm: float,
    pll_status: int,
    spectral_complexity: float,
    boundary_artifact: float,
    latency_margin_s: float,
    state_table: np.ndarray
) -> tuple[int, int]:
    """
    Zero-Heap JIT execution of verification stride:
    Returns (anomaly_flag, hold_required).
    """
    anomaly = 0
    if pll_status == 0 or rf_power_dbm > -40.0:
        anomaly = 1
        
    hold = 0
    if latency_margin_s < 0.0015:
        hold = 1
        
    return anomaly, hold


class AutomatedRFSweepVerification:
    """
    Coordinates end-to-end HIL validation sweeps across SpaceShield RF paths.
    Tracks PLL state, window alignments, and tuning latency in an allocation-free loop.
    """
    def __init__(self):
        self.calibration = FastLockCalibrationTable()
        self.overlap_manager = DynamicOverlapWindowManager()
        self.tuner = AdaptiveOverlapTuner()
        self.pll_validator = PhysicalSDRPLLValidation()
        
        # Pre-allocated state arrays
        self.state_table = np.zeros(10, dtype=np.int64)
        
        # Sliding buffer mock (64 x 4 elements)
        self.sliding_buffer = np.zeros((64, 4), dtype=np.complex64)

    def verify_stride(
        self,
        step: int,
        freq_from: int,
        freq_to: int,
        rf_power_dbm: float,
        pll_status: int,
        spectral_complexity: float,
        boundary_artifact: float,
        latency_margin_s: float
    ) -> dict:
        """
        Processes a single step of the automated HIL RF sweep.
        """
        # 1. Check calibration for hops
        cal_profile = self.calibration.lookup_profile(freq_from, freq_to)
        
        # 2. Run JIT evaluation
        anomaly_flag, hold_required = _run_sweep_step_jit(
            step,
            rf_power_dbm,
            pll_status,
            spectral_complexity,
            boundary_artifact,
            latency_margin_s,
            self.state_table
        )
        
        # 3. Process adaptive overlap tuner
        tune_res = self.tuner.tune_block_size(
            spectral_complexity,
            boundary_artifact,
            latency_margin_s
        )
        
        # 4. Transition block sizes if changed
        new_fft = tune_res["fft_size"]
        discard_len = self.overlap_manager.active_discard
        if new_fft != self.overlap_manager.active_fft_size:
            discard_len = self.overlap_manager.transition_block_size(self.sliding_buffer, new_fft)
            
        # 5. Process PLL monitor
        pll_res = self.pll_validator.process_stride(pll_status)
        
        return {
            "initial_mu": cal_profile["initial_mu"],
            "expected_lock_ms": cal_profile["expected_lock_ms"],
            "anomaly_detected": anomaly_flag,
            "hold_required": hold_required,
            "active_fft_size": new_fft,
            "discard_overlap": discard_len,
            "pll_lock": pll_res["lock_status"],
            "total_unlocks": pll_res["total_unlock_events"]
        }


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Automated RF Sweep Verification")
    print("==================================================================")
    
    verifier = AutomatedRFSweepVerification()
    
    # 1. Nominal Sweep
    print("[*] Scenario 1: Nominal Carrier Sweep...")
    # Transition channel 1 -> 2 under nominal levels
    res = verifier.verify_stride(
        step=1,
        freq_from=1,
        freq_to=2,
        rf_power_dbm=-85.0,
        pll_status=1,
        spectral_complexity=0.15,
        boundary_artifact=0.01,
        latency_margin_s=0.006
    )
    print(f"    -> FFT Size: {res['active_fft_size']} | Discard: {res['discard_overlap']} | PLL: {res['pll_lock']} | Mu: {res['initial_mu']}")
    assert res["pll_lock"] == 1 and res["active_fft_size"] == 16, "Failed nominal verification stride!"
    print("    -> Nominal sweep: [PASSED]")
    
    # 2. Injected Jamming & Carrier Hop Relock
    print("\n[*] Scenario 2: Injected Jamming & Carrier Hop relock...")
    # Jump channel 1 -> 6 (large jump!) under high power interference (-38 dBm)
    # The PLL loses lock temporarily during the jump (pll_status = 0)
    res_jump = verifier.verify_stride(
        step=2,
        freq_from=1,
        freq_to=6,
        rf_power_dbm=-38.0,
        pll_status=0,  # PLL unlock!
        spectral_complexity=0.85,
        boundary_artifact=0.15,
        latency_margin_s=0.004
    )
    print(f"    -> Anomaly: {res_jump['anomaly_detected']} | PLL: {res_jump['pll_lock']} | Expected lock: {res_jump['expected_lock_ms']} ms | Mu: {res_jump['initial_mu']}")
    assert res_jump["anomaly_detected"] == 1 and res_jump["pll_lock"] == 0, "Failed to capture relock anomaly!"
    assert res_jump["initial_mu"] == 0.03, "Failed to apply conservative fast-lock step size!"
    print("    -> Hop relock: [PASSED]")

    print("\n[+] Automated RF sweep verification complete.")
