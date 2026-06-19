"""
Task 56.1: Sub-Sample Synchronization Module
SpaceShield High-Velocity Receiver DSP Subsystem

Zero-allocation, vectorized fractional-sample cross-correlation engine tracking 
group delay differences across 4 complex64 channel strides. Evaluates a 3-lag 
correlation window (-1, 0, +1 samples) and applies parabolic interpolation.
"""

import numpy as np
import time
from numba import njit, prange

@njit(fastmath=True, cache=True, boundscheck=False)
def _compute_fractional_delays(
    X_buffer: np.ndarray,          # (channels, stride_len) complex64
    fractional_delays: np.ndarray, # (channels,) float32
    ref_channel: int
):
    """
    Zero-Heap Numba JIT Kernel: Evaluates a 3-lag cross-correlation (-1, 0, +1) 
    against a reference channel and applies parabolic interpolation to extract 
    sub-sample fractional skew.
    """
    num_channels = X_buffer.shape[0]
    stride_len = X_buffer.shape[1]
    
    for ch in range(num_channels):
        if ch == ref_channel:
            fractional_delays[ch] = 0.0
            continue
            
        r_minus1_real = 0.0
        r_minus1_imag = 0.0
        r_0_real = 0.0
        r_0_imag = 0.0
        r_plus1_real = 0.0
        r_plus1_imag = 0.0
        
        # Compute correlation over interior samples to avoid bounds checking and zero-padding
        for n in range(1, stride_len - 1):
            ref_n = X_buffer[ref_channel, n]
            ref_n_minus1 = X_buffer[ref_channel, n - 1]
            ref_n_plus1 = X_buffer[ref_channel, n + 1]
            
            x_n = X_buffer[ch, n]
            
            # tau = 0: x(n) * conj(ref(n))
            r_0_real += x_n.real * ref_n.real + x_n.imag * ref_n.imag
            r_0_imag += x_n.imag * ref_n.real - x_n.real * ref_n.imag
            
            # tau = +1: x(n) * conj(ref(n - 1))
            r_plus1_real += x_n.real * ref_n_minus1.real + x_n.imag * ref_n_minus1.imag
            r_plus1_imag += x_n.imag * ref_n_minus1.real - x_n.real * ref_n_minus1.imag
            
            # tau = -1: x(n) * conj(ref(n + 1))
            r_minus1_real += x_n.real * ref_n_plus1.real + x_n.imag * ref_n_plus1.imag
            r_minus1_imag += x_n.imag * ref_n_plus1.real - x_n.real * ref_n_plus1.imag
            
        # Compute magnitudes of the complex correlation peaks
        mag_m1 = np.sqrt(r_minus1_real * r_minus1_real + r_minus1_imag * r_minus1_imag)
        mag_0  = np.sqrt(r_0_real * r_0_real + r_0_imag * r_0_imag)
        mag_p1 = np.sqrt(r_plus1_real * r_plus1_real + r_plus1_imag * r_plus1_imag)
        
        # Inline parabolic interpolation for sub-sample fractional peak
        # delta = (R[-1] - R[1]) / (2 * (R[-1] + R[1] - 2 * R[0]))
        denom = 2.0 * (mag_m1 + mag_p1 - 2.0 * mag_0)
        
        if abs(denom) > 1e-12:
            delta = (mag_m1 - mag_p1) / denom
        else:
            delta = 0.0
            
        # Clamp delta to valid fractional range [-1.0, 1.0] to prevent anomalies
        if delta > 1.0:
            delta = 1.0
        elif delta < -1.0:
            delta = -1.0
            
        fractional_delays[ch] = np.float32(delta)


class FractionalDelayTracker:
    """
    SpaceShield Fractional Delay Tracker Interface.
    Extracts high-resolution spatial group delays from complex baseband channel strides.
    """
    def __init__(
        self,
        channels: int = 4,
        stride_len: int = 4096,
        ref_channel: int = 0
    ):
        self.channels = channels
        self.stride_len = stride_len
        self.ref_channel = ref_channel
        
        # Pre-allocate static thread memory blocks for zero-allocation writes
        self.fractional_delays = np.zeros(self.channels, dtype=np.float32)
        
        # Warmup JIT compilation trace
        self._warmup()
        
    def _warmup(self):
        """Forces LLVM JIT compilation ahead of processing."""
        dummy_X = np.ones((self.channels, self.stride_len), dtype=np.complex64)
        _compute_fractional_delays(dummy_X, self.fractional_delays, self.ref_channel)

    def track_stride(self, X_buffer: np.ndarray) -> np.ndarray:
        """
        Processes an aligned input complex sample stride, evaluating fractional
        sample delay skews. Returns reference to internal delay array.
        """
        _compute_fractional_delays(X_buffer, self.fractional_delays, self.ref_channel)
        return self.fractional_delays


if __name__ == "__main__":
    print("[*] Instantiating FractionalDelayTracker and pre-warming LLVM compiler...")
    tracker = FractionalDelayTracker()
    
    # Generate mock target with a Gaussian pulse to test envelope correlation peak
    N = 4096
    t = np.arange(N) - N/2
    
    mock_X = np.zeros((4, N), dtype=np.complex64)
    # Target reference (Gaussian pulse at t=0, wide enough to be sampled smoothly)
    sigma = 2.0
    mock_X[0] = np.exp(-0.5 * (t / sigma)**2) + 1j * np.exp(-0.5 * (t / sigma)**2)
    
    # Sub-sample delayed variants
    def delay_gauss(delay):
        return np.exp(-0.5 * ((t - delay) / sigma)**2) + 1j * np.exp(-0.5 * ((t - delay) / sigma)**2)
        
    mock_X[1] = delay_gauss(0.25)
    mock_X[2] = delay_gauss(-0.35)
    mock_X[3] = delay_gauss(0.85)
    
    mock_X = mock_X.astype(np.complex64)
    
    delays = tracker.track_stride(mock_X)
    print("\n--- MEASURED FRACTIONAL DELAYS ---")
    for i, d in enumerate(delays):
        print(f"  Channel {i}: {d:+.3f} samples")
        
    print("\n[*] Running 1,000 benchmark strides...")
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = tracker.track_stride(mock_X)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)
        
    avg_us = np.mean(latencies)
    p99_us = np.percentile(latencies, 99.0)
    
    print("\n--- DELAY TRACKER PERFORMANCE HUD ---")
    print(f"  Average Stride Latency: {avg_us:.2f} µs")
    print(f"  P99 Stride Latency:      {p99_us:.2f} µs")
    
    if avg_us <= 15.0:
        print("[PASSED] Sub-sample synchronization operates well within the 15µs cap.")
    else:
        print("[FAIL] Operational ceiling breached! Check Numba compilation.")
