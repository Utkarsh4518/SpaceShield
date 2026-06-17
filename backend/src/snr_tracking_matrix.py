"""
Task 48.2: High-Speed Signal Quality Assessment Module
Zero-Allocation, Numba-Accelerated M2M4 C/N0 Estimation
"""

import time
import math
import numpy as np
import ctypes
from numba import njit

# ==============================================================================
# JIT-Compiled SNR Estimation Kernel
# ==============================================================================

@njit(fastmath=True, cache=True)
def _estimate_snr_kernel_f32(
    data: np.ndarray,
    sample_rate: np.float32,
    smoothing_factor: np.float32,
    rolling_cn0: np.ndarray
) -> None:
    """
    Computes a non-data-aided SNR estimate using the Second-Order and Fourth-Order
    Moments (M2M4) algorithm. Then converts it to a rolling C/N0 value.
    Operates strictly in-place with single-precision float32 math for maximum
    LLVM/SIMD compilation efficiency and zero runtime heap modifications.
    
    Args:
        data: (num_channels, N) complex64 data stride.
        sample_rate: Sampling frequency in Hz (float32).
        smoothing_factor: IIR filter weighting parameter (float32).
        rolling_cn0: (num_channels,) float32 rolling C/N0 output array.
    """
    num_channels = data.shape[0]
    N = data.shape[1]
    inv_N = np.float32(1.0 / N)
    
    for c in range(num_channels):
        sum_mag2 = np.float32(0.0)
        sum_mag4 = np.float32(0.0)
        
        # Calculate second and fourth power of magnitudes
        for k in range(N):
            val = data[c, k]
            v_r = val.real
            v_i = val.imag
            mag2 = v_r * v_r + v_i * v_i
            sum_mag2 += mag2
            sum_mag4 += mag2 * mag2
            
        m2 = sum_mag2 * inv_N
        m4 = sum_mag4 * inv_N
        
        # M2M4 algorithm for constant-envelope signal in complex AWGN
        val_diff = np.float32(2.0) * m2 * m2 - m4
        if val_diff > np.float32(0.0):
            S = np.float32(math.sqrt(val_diff))
            N_var = m2 - S
            if N_var > np.float32(1e-12):
                snr = S / N_var
            else:
                snr = np.float32(1e7) # Limit maximum SNR to 70 dB
        else:
            snr = np.float32(0.0)
            
        if snr > np.float32(0.0):
            # C/N0 = 10 * log10(SNR_linear * sample_rate)
            cn0_measured = np.float32(10.0) * np.float32(math.log10(snr * sample_rate))
        else:
            cn0_measured = np.float32(0.0)
            
        # Exponential moving average smoothing filter: C/N0_new = (1-b)*C/N0_old + b*C/N0_measured
        rolling_cn0[c] = (np.float32(1.0) - smoothing_factor) * rolling_cn0[c] + smoothing_factor * cn0_measured


# ==============================================================================
# SNR Tracking Matrix Class
# ==============================================================================

class SnrTrackingMatrix:
    """
    High-Speed Signal Quality Assessment Module.
    Calculates rolling real-time Carrier-to-Noise density ratio (C/N0) estimates
    for each of the 4 independent RF channels using the JIT-compiled M2M4 algorithm.
    Exposes these metrics thread-safely via lock-free ctypes arrays.
    """
    def __init__(self, sample_rate_hz: float = 2.0e6, smoothing_factor: float = 0.05):
        self.sample_rate = np.float32(sample_rate_hz)
        self.smoothing_factor = np.float32(smoothing_factor)
        
        # Pre-allocated rolling C/N0 buffer (numpy array for the JIT kernel)
        self.rolling_cn0 = np.zeros(4, dtype=np.float32)
        
        # Atomic/Thread-safe shared variable (ctypes array) for downstream tracking loops
        self.shared_cn0 = (ctypes.c_float * 4)()
        
        # Initialize default values to nominal NavIC L5/L1 acquisition value (45.0 dB-Hz)
        for c in range(4):
            self.rolling_cn0[c] = 45.0
            self.shared_cn0[c] = 45.0
            
        self._warmup()
        
    def _warmup(self):
        """Forces Numba JIT compilation trace."""
        dummy_data = np.ones((4, 1024), dtype=np.complex64)
        _estimate_snr_kernel_f32(
            dummy_data,
            self.sample_rate,
            self.smoothing_factor,
            self.rolling_cn0
        )
        # Reset state after warmup
        for c in range(4):
            self.rolling_cn0[c] = 45.0
            self.shared_cn0[c] = 45.0
            
    def update_metrics(self, data: np.ndarray) -> None:
        """
        Calculates and updates C/N0 estimates in-place using pre-allocated structures.
        Exposes results atomically to shared variables.
        
        Args:
            data: shape (4, N) complex64 numpy array.
        """
        if data.ndim != 2 or data.shape[0] != 4:
            raise ValueError("Input data must have shape (4, N).")
            
        # Run JIT-compiled estimation
        _estimate_snr_kernel_f32(
            data,
            self.sample_rate,
            self.smoothing_factor,
            self.rolling_cn0
        )
        
        # Write values to ctypes array. This is thread-safe on CPython since ctypes
        # writes do not yield the GIL and execute as atomic operations at the C level.
        self.shared_cn0[0] = self.rolling_cn0[0]
        self.shared_cn0[1] = self.rolling_cn0[1]
        self.shared_cn0[2] = self.rolling_cn0[2]
        self.shared_cn0[3] = self.rolling_cn0[3]


# ==============================================================================
# Standalone Diagnostic Verification Harness
# ==============================================================================

if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Signal Quality: SNR & C/N0 Tracking Matrix")
    print("==================================================================")
    
    # 1. Instantiate module
    sample_rate = 2.0e6
    tracker = SnrTrackingMatrix(sample_rate_hz=sample_rate, smoothing_factor=0.05)
    print("[PASS] SNR Tracking Module Initialized & JIT Compiled.")
    
    # 2. Run latency benchmark
    # Stride length = 2048 complex64 samples
    chunk_size = 2048
    num_channels = 4
    latencies = []
    
    # Pre-generate mock signals for benchmark
    rng = np.random.default_rng(0x7357)
    
    for _ in range(2000):
        # Generate random mock data stride to avoid cache-hitting optimization shortcuts
        data_bench = (rng.normal(0.0, 1.0, (num_channels, chunk_size)) + 
                      1j * rng.normal(0.0, 1.0, (num_channels, chunk_size))).astype(np.complex64)
                      
        t1 = time.perf_counter_ns()
        tracker.update_metrics(data_bench)
        t2 = time.perf_counter_ns()
        latencies.append((t2 - t1) / 1000.0)
        
    avg_us = sum(latencies) / len(latencies)
    max_us = np.percentile(latencies, 99.0)
    print(f"\n[EVAL] Stride Processing Latency (Chunk Size: {chunk_size} samples):")
    print(f"    -> Average Execution Latency:  {avg_us:.3f} microseconds")
    print(f"    -> P99 Max Execution Jitter:   {max_us:.3f} microseconds")
    
    # 3. Verify Estimation Accuracy (Using smoothing_factor = 1.0 for instantaneous measurement)
    accuracy_tracker = SnrTrackingMatrix(sample_rate_hz=sample_rate, smoothing_factor=1.0)
    print(f"\n[EVAL] M2M4 Estimator Accuracy Audit (Sample Rate: {sample_rate/1e6} MSPS):")
    
    # Test known SNRs
    test_snrs_db = [5.0, 15.0, 25.0]
    accuracy_ok = True
    
    for snr_target in test_snrs_db:
        snr_linear = 10.0 ** (snr_target / 10.0)
        # Analytical C/N0 target
        cn0_target = snr_target + 10.0 * math.log10(sample_rate)
        
        # Synthesize constant envelope signal + complex AWGN
        noise_var = 1.0 / snr_linear
        noise_std = math.sqrt(noise_var / 2.0)
        
        # Build 100k samples for high statistics accuracy check
        theta = rng.uniform(-np.pi, np.pi, (num_channels, 100000))
        sig = np.exp(1j * theta).astype(np.complex64)
        noise = (rng.normal(0, noise_std, (num_channels, 100000)) + 
                 1j * rng.normal(0, noise_std, (num_channels, 100000))).astype(np.complex64)
        data_synth = sig + noise
        
        accuracy_tracker.update_metrics(data_synth)
        
        measured_cn0 = accuracy_tracker.rolling_cn0[0]
        error = abs(measured_cn0 - cn0_target)
        
        print(f"    -> Target SNR: {snr_target:5.1f} dB | Expected C/N0: {cn0_target:6.2f} dB-Hz | Measured: {measured_cn0:6.2f} dB-Hz | Error: {error:5.3f} dB")
        if error > 0.2:
            accuracy_ok = False
            
    # Assert correctness
    latency_ok = avg_us < 6.0
    
    print("\n[VERIFY] Performance and DSP Model Verification:")
    if latency_ok:
        print("[PASS] Signal quality tracking runs well under the 6.0 microsecond limit.")
    else:
        print("[FAIL] Signal quality tracking breached the 6.0 microsecond real-time limit.")
        
    if accuracy_ok:
        print("[PASS] M2M4 C/N0 estimation matches target accuracy requirements.")
    else:
        print("[FAIL] M2M4 estimation error exceeded target bounds.")
        
    assert latency_ok, "Failing due to latency limit breach"
    assert accuracy_ok, "Failing due to estimation accuracy breach"
    print("==================================================================")
