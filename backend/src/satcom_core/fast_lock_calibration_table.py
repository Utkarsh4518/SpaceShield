"""
Task 70.1: Fast-Lock Calibration Table Module
SpaceShield High-Velocity Receiver DSP Subsystem

Provides lookup parameter mapping (mu, settle thresholds, lock envelopes) for PLL frequency hops.
Helps the spatial tracking and synchronization loops acquire locks quickly without overshoot.
"""

import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _lookup_calibration_jit(
    freq_from_idx: int,
    freq_to_idx: int,
    hop_profile_table: np.ndarray  # (8, 8, 3) float64 -> [mu, settle_threshold, expected_lock_ms]
) -> tuple[float, float, float]:
    """
    Zero-Heap JIT Lookup for Fast-Lock Calibration:
    Returns (mu, settle_threshold, expected_lock_ms).
    """
    f_from = min(max(freq_from_idx, 0), 7)
    f_to = min(max(freq_to_idx, 0), 7)
    
    mu = hop_profile_table[f_from, f_to, 0]
    settle = hop_profile_table[f_from, f_to, 1]
    lock_ms = hop_profile_table[f_from, f_to, 2]
    
    return mu, settle, lock_ms


class FastLockCalibrationTable:
    """
    Maintains pre-calibrated frequency-hop profile registers.
    Optimizes loop updates dynamically during carrier transitions.
    """
    def __init__(self):
        # 8 channels supported
        # Dimensions: (from_chan, to_chan, [mu, settle, lock_ms])
        self.hop_profile_table = np.zeros((8, 8, 3), dtype=np.float64)
        
        # Populate calibration profiles
        for f_from in range(8):
            for f_to in range(8):
                dist = abs(f_from - f_to)
                
                # Small hop vs large hop profiles
                if dist == 0:
                    # No hop / nominal tuning
                    mu = 0.05
                    settle = 0.04
                    lock_ms = 1.0
                elif dist <= 2:
                    # Small hop (stable profiles)
                    mu = 0.08
                    settle = 0.06
                    lock_ms = 3.5
                else:
                    # Large hop (accelerated settling, higher overshoot protection)
                    mu = 0.03
                    settle = 0.08
                    lock_ms = 8.0
                    
                self.hop_profile_table[f_from, f_to, 0] = mu
                self.hop_profile_table[f_from, f_to, 1] = settle
                self.hop_profile_table[f_from, f_to, 2] = lock_ms

    def lookup_profile(self, freq_from_idx: int, freq_to_idx: int) -> dict:
        """
        Retrieves parameters for a given transition.
        """
        mu, settle, lock_ms = _lookup_calibration_jit(
            freq_from_idx,
            freq_to_idx,
            self.hop_profile_table
        )
        return {
            "initial_mu": mu,
            "settle_threshold": settle,
            "expected_lock_ms": lock_ms
        }


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Closed-Loop: Fast-Lock Calibration Table Validation")
    print("==================================================================")
    
    table = FastLockCalibrationTable()
    
    # 1. Small Frequency Hop (channel 1 -> 2)
    print("[*] Scenario 1: Small Frequency Hop lookup...")
    p1 = table.lookup_profile(1, 2)
    print(f"    -> Mu: {p1['initial_mu']:.2f} | Settle: {p1['settle_threshold']:.2f} | Lock Time: {p1['expected_lock_ms']:.1f} ms")
    assert p1["initial_mu"] == 0.08 and p1["expected_lock_ms"] == 3.5, "Failed to load small hop profile!"
    print("    -> Small hop profile: [PASSED]")
    
    # 2. Large Frequency Jump (channel 1 -> 6)
    print("\n[*] Scenario 2: Large Frequency Jump lookup...")
    p2 = table.lookup_profile(1, 6)
    print(f"    -> Mu: {p2['initial_mu']:.2f} | Settle: {p2['settle_threshold']:.2f} | Lock Time: {p2['expected_lock_ms']:.1f} ms")
    assert p2["initial_mu"] == 0.03 and p2["expected_lock_ms"] == 8.0, "Failed to load large jump profile!"
    print("    -> Large hop profile: [PASSED]")
    
    # 3. Same Channel Nominal (channel 3 -> 3)
    print("\n[*] Scenario 3: Nominal Same-Channel lookup...")
    p3 = table.lookup_profile(3, 3)
    print(f"    -> Mu: {p3['initial_mu']:.2f} | Settle: {p3['settle_threshold']:.2f} | Lock Time: {p3['expected_lock_ms']:.1f} ms")
    assert p3["initial_mu"] == 0.05 and p3["expected_lock_ms"] == 1.0, "Failed to load nominal same-channel profile!"
    print("    -> Nominal lookup: [PASSED]")

    print("\n[+] Fast-lock calibration table validation complete.")
