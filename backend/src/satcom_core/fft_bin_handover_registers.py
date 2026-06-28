"""
Task 69.2: FFT Bin Handover Registers Module
SpaceShield High-Velocity Receiver DSP Subsystem

Stages standby weights inside individual FFT bin registers.
Swaps them independently bin-by-bin once the convergence criteria are satisfied.
"""

import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _process_bin_handover_jit(
    active_weights: np.ndarray,     # (16, 4) complex64
    shadow_weights: np.ndarray,     # (16, 4) complex64
    x_active: np.ndarray,           # (16, 4) complex64
    x_shadow: np.ndarray,           # (16, 4) complex64
    adaptation_steps: np.ndarray,   # (16,) int32
    leakage_power: np.ndarray,      # (16,) float64
    settling_threshold: float,
    min_steps: int
) -> tuple[int, int]:
    """
    Zero-Heap JIT evaluation of FFT bin-level weight handover.
    Returns (swapped_mask, total_swaps).
    """
    swapped_mask = 0
    total_swaps = 0
    
    for f in range(16):
        # 1. Increment adaptation steps for bin f
        adaptation_steps[f] += 1
        
        # 2. Evaluate leakage output power of shadow weights on shadow input
        w0 = shadow_weights[f, 0]
        w1 = shadow_weights[f, 1]
        w2 = shadow_weights[f, 2]
        w3 = shadow_weights[f, 3]
        
        w0_c = w0.real - 1j * w0.imag
        w1_c = w1.real - 1j * w1.imag
        w2_c = w2.real - 1j * w2.imag
        w3_c = w3.real - 1j * w3.imag
        
        xs0 = x_shadow[f, 0]
        xs1 = x_shadow[f, 1]
        xs2 = x_shadow[f, 2]
        xs3 = x_shadow[f, 3]
        
        y = w0_c * xs0 + w1_c * xs1 + w2_c * xs2 + w3_c * xs3
        p = y.real*y.real + y.imag*y.imag
        
        # Exponential smoothing of leakage power
        leakage_power[f] = 0.9 * leakage_power[f] + 0.1 * p
        
        # 3. Swap when convergence threshold is achieved
        if adaptation_steps[f] >= min_steps and leakage_power[f] < settling_threshold:
            for c in range(4):
                active_weights[f, c] = shadow_weights[f, c]
            
            # Reset tracking stats
            adaptation_steps[f] = 0
            leakage_power[f] = 1.0
            
            swapped_mask |= (1 << f)
            total_swaps += 1
            
    return swapped_mask, total_swaps


class FFTBinHandoverRegisters:
    """
    Stages and executes bin-level weight swaps during satellite handovers.
    Integrates with the STAP and mitigation components on a zero-allocation hot path.
    """
    def __init__(self, settling_threshold: float = 0.05, min_steps: int = 10):
        self.settling_threshold = settling_threshold
        self.min_steps = min_steps
        
        # Pre-allocated registers (16 bins x 4 channels)
        self.active_weights = np.zeros((16, 4), dtype=np.complex64)
        self.shadow_weights = np.zeros((16, 4), dtype=np.complex64)
        
        # Initialize default weights
        for f in range(16):
            self.active_weights[f, :] = 0.25
            self.shadow_weights[f, :] = 0.25
            
        # Pre-allocated metrics tracking
        self.adaptation_steps = np.zeros(16, dtype=np.int32)
        self.leakage_power = np.ones(16, dtype=np.float64)

    def process_handover_stride(
        self,
        x_active: np.ndarray,   # (16, 4) complex64
        x_shadow: np.ndarray    # (16, 4) complex64
    ) -> tuple[int, int]:
        """
        Runs JIT check on all 16 frequency bins.
        """
        return _process_bin_handover_jit(
            self.active_weights,
            self.shadow_weights,
            x_active.astype(np.complex64),
            x_shadow.astype(np.complex64),
            self.adaptation_steps,
            self.leakage_power,
            self.settling_threshold,
            self.min_steps
        )


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: FFT Bin Handover Registers")
    print("==================================================================")
    
    registers = FFTBinHandoverRegisters(settling_threshold=0.08, min_steps=5)
    
    # Mock data setup
    x_active = np.random.normal(0, 0.01, (16, 4)) + 1j * np.random.normal(0, 0.01, (16, 4))
    x_shadow = np.random.normal(0, 0.01, (16, 4)) + 1j * np.random.normal(0, 0.01, (16, 4))
    
    # Simulate shadow weights convergence
    # Set shadow weights to extremely small values so output power is tiny
    registers.shadow_weights.fill(1e-4 + 1j * 1e-4)
    registers.leakage_power.fill(0.01)
    
    # 1. Running steps to achieve min_steps constraint
    print("[*] Running 4 strides (less than min_steps=5)...")
    for step in range(4):
        mask, count = registers.process_handover_stride(x_active, x_shadow)
        
    print(f"    -> Swaps Triggered: {count} | Active weight (bin 0): {registers.active_weights[0, 0]}")
    assert count == 0 and registers.active_weights[0, 0] == 0.25, "Handover triggered before min_steps window!"
    print("    -> Handover lock: [PASSED]")
    
    # 2. Step 5 (should trigger all 16 swaps since leakage is tiny)
    print("\n[*] Step 5 (reaching min_steps constraint)...")
    mask, count = registers.process_handover_stride(x_active, x_shadow)
    print(f"    -> Swaps Triggered: {count} | Mask: {bin(mask)} | Active weight (bin 0): {registers.active_weights[0, 0]}")
    assert count == 16 and np.abs(registers.active_weights[0, 0]) < 1e-3, "Failed to swap weights upon convergence!"
    print("    -> Handover swap: [PASSED]")

    print("\n[+] FFT bin-level handover registers validation complete.")
