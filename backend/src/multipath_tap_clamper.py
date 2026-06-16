"""
Task 39.2: Multipath Tap Clamper
SpaceShield Sparsity-Enforcing Filter Optimizer

Enforces dynamic sparsity on the Least-Squares equalizer taps by clamping 
inactive multipath delay echoes below a computed energy threshold. Prevents 
noise amplification and stabilizes passband geometry.
"""

import numpy as np
from numba import njit

@njit(parallel=False, fastmath=True, boundscheck=False, cache=True)
def _clamp_taps(
    weights_pool: np.ndarray,
    channels: int,
    taps: int,
    threshold_ratio: float
):
    """
    Evaluates individual tap energies. Clamps to exact zero if 
    the tap energy falls beneath the noise-floor threshold relative 
    to the primary Line-of-Sight (LoS) peak.
    """
    for c in range(channels):
        # 1. Scan for primary LoS Peak Energy
        max_energy = np.float32(0.0)
        for i in range(taps):
            val = weights_pool[c, i]
            energy = np.float32(val.real * val.real + val.imag * val.imag)
            if energy > max_energy:
                max_energy = energy
                
        # 2. Compute dynamic noise floor clamp bound
        clamp_threshold = max_energy * np.float32(threshold_ratio)
        
        # 3. Apply in-place bit-wise zero clamping to enforce sparsity
        for i in range(taps):
            val = weights_pool[c, i]
            energy = np.float32(val.real * val.real + val.imag * val.imag)
            if energy < clamp_threshold:
                weights_pool[c, i] = 0.0 + 0j


class MultipathTapClamper:
    """
    SpaceShield Multipath Tap Clamping Utility.
    Couples seamlessly with the Cyclic LS Equalizer to suppress noise
    across inactive multipath delay taps in real time.
    """
    def __init__(self, channels: int = 4, taps: int = 5, isolation_db: float = 15.0):
        self.channels = channels
        self.taps = taps
        # Convert dB threshold to linear power ratio
        self.threshold_ratio = 10.0 ** (-isolation_db / 10.0)
        
        self._warmup()
        
    def _warmup(self):
        """Pre-compiles LLVM path natively."""
        dummy_weights = np.zeros((self.channels, self.taps), dtype=np.complex64)
        dummy_weights[0, 0] = 1.0 + 0j
        dummy_weights[0, 1] = 0.1 + 0j
        _clamp_taps(dummy_weights, self.channels, self.taps, self.threshold_ratio)
        
    def enforce_sparsity(self, weights_pool: np.ndarray) -> None:
        """
        Executes zero-allocation, in-place multipath noise suppression on the
        provided FIR filter tap pool. Operates optimally beneath a 6µs ceiling.
        """
        _clamp_taps(weights_pool, self.channels, self.taps, self.threshold_ratio)
