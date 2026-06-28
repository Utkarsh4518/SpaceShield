"""
Task 69.1: Adaptive Overlap Tuner Module
SpaceShield High-Velocity Receiver DSP Subsystem

Monitors interference complexity, boundary artifacts, and latency margin to dynamically
adjust FFT block sizes. Keeps decision updates JIT-compiled and zero-allocation.
"""

import numpy as np
from numba import njit

# Pre-approved block configurations:
# idx 0: FFT=16, Step=8 (low complexity, low latency)
# idx 1: FFT=32, Step=16 (narrowband, medium latency)
# idx 2: FFT=64, Step=32 (multi-tone / chirp, high latency)
CONFIG_FFT_SIZES = np.array([16, 32, 64], dtype=np.int32)
CONFIG_STEPS = np.array([8, 16, 32], dtype=np.int32)

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _tuner_update_jit(
    spectral_complexity: float,
    boundary_artifact: float,
    latency_margin_s: float,
    current_size_idx: int,
    hold_counter: int,
    min_hold_steps: int
) -> tuple[int, int]:
    """
    Zero-Heap JIT Adaptive Tuning Decision:
    Returns (next_size_idx, updated_hold_counter).
    """
    # 1. Check hold timer to prevent rapid size oscillations
    if hold_counter > 0:
        return current_size_idx, hold_counter - 1
        
    # 2. Latency emergency override
    if latency_margin_s < 0.001:  # Less than 1ms margin
        if current_size_idx != 0:
            return 0, min_hold_steps
        return 0, 0
        
    # 3. Dynamic adjustment rules
    next_idx = current_size_idx
    
    # High complexity / artifact leap triggers shift up to resolve jammers
    if (spectral_complexity > 0.6 or boundary_artifact > 0.08) and latency_margin_s >= 0.002:
        next_idx = min(current_size_idx + 1, 2)
    # Low complexity / nominal operation shifts down to save CPU / latency
    elif (spectral_complexity < 0.2 and boundary_artifact < 0.02):
        next_idx = max(current_size_idx - 1, 0)
        
    if next_idx != current_size_idx:
        return next_idx, min_hold_steps
        
    return next_idx, 0


class AdaptiveOverlapTuner:
    """
    Adjusts overlap-save STAP block configurations dynamically.
    Optimizes frequency resolution vs timing latency trade-offs.
    """
    def __init__(self, min_hold_steps: int = 4):
        self.min_hold_steps = min_hold_steps
        self.current_size_idx = 1  # Start at nominal FFT=32
        self.hold_counter = 0

    def tune_block_size(
        self,
        spectral_complexity: float,
        boundary_artifact: float,
        latency_margin_s: float
    ) -> dict:
        """
        Executes a tuning stride and returns the target FFT and step sizes.
        """
        next_idx, new_hold = _tuner_update_jit(
            spectral_complexity,
            boundary_artifact,
            latency_margin_s,
            self.current_size_idx,
            self.hold_counter,
            self.min_hold_steps
        )
        
        self.current_size_idx = next_idx
        self.hold_counter = new_hold
        
        fft_size = CONFIG_FFT_SIZES[next_idx]
        step_size = CONFIG_STEPS[next_idx]
        
        return {
            "size_idx": next_idx,
            "fft_size": fft_size,
            "step_size": step_size,
            "hold_active": 1 if new_hold > 0 else 0
        }


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Closed-Loop: Adaptive Overlap Tuner Validation")
    print("==================================================================")
    
    tuner = AdaptiveOverlapTuner(min_hold_steps=3)
    
    # 1. Low complexity nominal operation (Should shift down to FFT=16)
    print("[*] Scenario 1: Low-Complexity Nominal (Downshift check)...")
    res = tuner.tune_block_size(
        spectral_complexity=0.1,
        boundary_artifact=0.01,
        latency_margin_s=0.005
    )
    print(f"    -> Next Size Index: {res['size_idx']} | FFT: {res['fft_size']} | Hold: {res['hold_active']}")
    assert res["fft_size"] == 16, "Failed to downshift under nominal conditions!"
    print("    -> Downshift: [PASSED]")
    
    # 2. Slow Hold Window Lock
    print("\n[*] Scenario 2: Tuning Hold Timer (Stability lock check)...")
    # Immediate high-complexity trigger (should be ignored due to hold lock)
    res2 = tuner.tune_block_size(
        spectral_complexity=0.8,
        boundary_artifact=0.12,
        latency_margin_s=0.004
    )
    print(f"    -> Size Index: {res2['size_idx']} | FFT: {res2['fft_size']} | Hold: {res2['hold_active']}")
    assert res2["fft_size"] == 16, "Tuner violated hold lock boundaries!"
    print("    -> Hold lock constraint: [PASSED]")

    # 3. High Complexity Multi-tone Jammer
    print("\n[*] Scenario 3: Complex Multi-tone Jammer (Upshift check)...")
    # Release hold by running empty steps
    tuner.hold_counter = 0
    res3 = tuner.tune_block_size(
        spectral_complexity=0.8,
        boundary_artifact=0.12,
        latency_margin_s=0.004
    )
    print(f"    -> Size Index: {res3['size_idx']} | FFT: {res3['fft_size']}")
    assert res3["fft_size"] == 32, "Failed to upshift to medium resolution!"
    
    # Upshift again to highest resolution
    tuner.hold_counter = 0
    res3_high = tuner.tune_block_size(
        spectral_complexity=0.8,
        boundary_artifact=0.12,
        latency_margin_s=0.004
    )
    print(f"    -> Size Index: {res3_high['size_idx']} | FFT: {res3_high['fft_size']}")
    assert res3_high["fft_size"] == 64, "Failed to upshift to highest resolution!"
    print("    -> Jamming upshift: [PASSED]")

    # 4. Latency emergency override
    print("\n[*] Scenario 4: Latency Emergency Override...")
    tuner.hold_counter = 0
    res4 = tuner.tune_block_size(
        spectral_complexity=0.8,
        boundary_artifact=0.12,
        latency_margin_s=0.0005  # extremely low latency margin (0.5ms)
    )
    print(f"    -> Size Index: {res4['size_idx']} | FFT: {res4['fft_size']}")
    assert res4["fft_size"] == 16, "Failed to trigger latency emergency downshift!"
    print("    -> Latency override: [PASSED]")

    print("\n[+] Adaptive overlap tuner validation complete.")
