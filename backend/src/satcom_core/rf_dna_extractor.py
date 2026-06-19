"""
Task 54.1: Intrinsic Radio Frequency DNA (RF-DNA) Feature Extraction Engine
SpaceShield High-Velocity Hardware Fingerprinting Subsystem

Executes zero-allocation, vectorized statistical feature extraction over localized 
carrier burst windows (512 samples) to isolate sub-segment hardware fingerprints.
Calculates variance, skewness, and kurtosis profiles of the instantaneous amplitude,
phase, and frequency deviations under strict latency bounds (< 18µs).
"""

import numpy as np
import time
import math
from numba import njit

@njit(fastmath=True, cache=True, boundscheck=False)
def _extract_rf_dna_kernel(
    Y_equalized: np.ndarray,      # (channels, stride_len) complex64
    amp_buf: np.ndarray,          # (channels, stride_len) float32
    phase_buf: np.ndarray,        # (channels, stride_len) float32
    freq_buf: np.ndarray,         # (channels, stride_len) float32
    feature_pool: np.ndarray,     # (channels, num_windows, num_subsegments, 9) float32
    num_channels: int,
    stride_len: int,
    window_len: int,
    num_subsegments: int,
    subseg_len: int
):
    """
    Zero-Heap Numba JIT Kernel: Processes IQ samples, calculates instantaneous features, 
    and maps statistical moments (variance, skewness, kurtosis) onto the pre-allocated pool.
    Fuses loops to minimize memory access overhead.
    """
    two_pi = np.float32(2.0 * np.pi)
    
    # 1. Vectorized phase, amplitude, and wrapped frequency difference estimation
    for c in range(num_channels):
        # Initial sample handling
        x0 = Y_equalized[c, 0]
        x0_r = x0.real
        x0_i = x0.imag
        amp_buf[c, 0] = math.sqrt(x0_r * x0_r + x0_i * x0_i)
        ph0 = math.atan2(x0_i, x0_r)
        phase_buf[c, 0] = ph0
        freq_buf[c, 0] = 0.0
        
        # Stride loop for raw parameters
        for n in range(1, stride_len):
            xn = Y_equalized[c, n]
            xn_r = xn.real
            xn_i = xn.imag
            
            a = math.sqrt(xn_r * xn_r + xn_i * xn_i)
            ph = math.atan2(xn_i, xn_r)
            
            amp_buf[c, n] = a
            phase_buf[c, n] = ph
            
            # Instantaneous frequency: unwrapped phase difference
            diff = ph - phase_buf[c, n - 1]
            wrapped_diff = diff - two_pi * math.floor(diff / two_pi + 0.5)
            freq_buf[c, n] = wrapped_diff
            
    # 2. Extract statistical metrics per sub-segment (fused for speed)
    num_windows = stride_len // window_len
    
    for c in range(num_channels):
        for w in range(num_windows):
            window_start = w * window_len
            for s in range(num_subsegments):
                subseg_start = window_start + s * subseg_len
                
                # Fused Pass 1: Compute means
                sum_a = 0.0
                sum_p = 0.0
                sum_f = 0.0
                for i in range(subseg_len):
                    idx = subseg_start + i
                    sum_a += amp_buf[c, idx]
                    sum_p += phase_buf[c, idx]
                    sum_f += freq_buf[c, idx]
                    
                mean_a = sum_a / subseg_len
                mean_p = sum_p / subseg_len
                mean_f = sum_f / subseg_len
                
                # Fused Pass 2: Compute moment accumulators
                v2_a = 0.0; v3_a = 0.0; v4_a = 0.0
                v2_p = 0.0; v3_p = 0.0; v4_p = 0.0
                v2_f = 0.0; v3_f = 0.0; v4_f = 0.0
                
                for i in range(subseg_len):
                    idx = subseg_start + i
                    
                    dev_a = amp_buf[c, idx] - mean_a
                    dev2_a = dev_a * dev_a
                    v2_a += dev2_a
                    v3_a += dev2_a * dev_a
                    v4_a += dev2_a * dev2_a
                    
                    dev_p = phase_buf[c, idx] - mean_p
                    dev2_p = dev_p * dev_p
                    v2_p += dev2_p
                    v3_p += dev2_p * dev_p
                    v4_p += dev2_p * dev2_p
                    
                    dev_f = freq_buf[c, idx] - mean_f
                    dev2_f = dev_f * dev_f
                    v2_f += dev2_f
                    v3_f += dev2_f * dev_f
                    v4_f += dev2_f * dev2_f
                    
                # Compute stats for amplitude
                var_a = v2_a / subseg_len
                if var_a > 1e-12:
                    std_a = math.sqrt(var_a)
                    skew_a = (v3_a / subseg_len) / (var_a * std_a)
                    kurt_a = (v4_a / subseg_len) / (var_a * var_a)
                else:
                    skew_a = 0.0
                    kurt_a = 3.0
                    
                # Compute stats for phase
                var_p = v2_p / subseg_len
                if var_p > 1e-12:
                    std_p = math.sqrt(var_p)
                    skew_p = (v3_p / subseg_len) / (var_p * std_p)
                    kurt_p = (v4_p / subseg_len) / (var_p * var_p)
                else:
                    skew_p = 0.0
                    kurt_p = 3.0
                    
                # Compute stats for frequency
                var_f = v2_f / subseg_len
                if var_f > 1e-12:
                    std_f = math.sqrt(var_f)
                    skew_f = (v3_f / subseg_len) / (var_f * std_f)
                    kurt_f = (v4_f / subseg_len) / (var_f * var_f)
                else:
                    skew_f = 0.0
                    kurt_f = 3.0
                
                # Write results directly into pre-allocated feature pool
                feature_pool[c, w, s, 0] = np.float32(var_a)
                feature_pool[c, w, s, 1] = np.float32(skew_a)
                feature_pool[c, w, s, 2] = np.float32(kurt_a)
                feature_pool[c, w, s, 3] = np.float32(var_p)
                feature_pool[c, w, s, 4] = np.float32(skew_p)
                feature_pool[c, w, s, 5] = np.float32(kurt_p)
                feature_pool[c, w, s, 6] = np.float32(var_f)
                feature_pool[c, w, s, 7] = np.float32(skew_f)
                feature_pool[c, w, s, 8] = np.float32(kurt_f)


class RFDnaExtractor:
    """
    Inline Hardware Fingerprint Extraction Engine.
    Operates over equalized complex signals using zero-heap-allocation JIT arrays 
    to extract variance, skewness, and kurtosis deviations of amplitude, phase, and frequency.
    """
    def __init__(
        self, 
        num_channels: int = 4, 
        stride_len: int = 4096, 
        window_len: int = 512, 
        num_subsegments: int = 4
    ):
        self.num_channels = num_channels
        self.stride_len = stride_len
        self.window_len = window_len
        self.num_subsegments = num_subsegments
        self.subseg_len = window_len // num_subsegments
        
        # Pre-allocate zero-heap operational buffers
        self.amp_buffer = np.zeros((self.num_channels, self.stride_len), dtype=np.float32)
        self.phase_buffer = np.zeros((self.num_channels, self.stride_len), dtype=np.float32)
        self.freq_buffer = np.zeros((self.num_channels, self.stride_len), dtype=np.float32)
        
        # Pre-allocate output feature pool: (channels, num_windows, num_subsegments, 9 features)
        self.num_windows = stride_len // window_len
        self._feature_pool = np.zeros(
            (self.num_channels, self.num_windows, self.num_subsegments, 9), 
            dtype=np.float32
        )
        
        # Pre-warm JIT compiler to ensure no latency spike on first real-time stride
        self._prewarm()

    def _prewarm(self):
        """Forces immediate JIT compilation by executing a mock stride."""
        mock_data = np.zeros((self.num_channels, self.stride_len), dtype=np.complex64)
        mock_data[:, :] = 1.0 + 0.5j
        _extract_rf_dna_kernel(
            mock_data,
            self.amp_buffer,
            self.phase_buffer,
            self.freq_buffer,
            self._feature_pool,
            self.num_channels,
            self.stride_len,
            self.window_len,
            self.num_subsegments,
            self.subseg_len
        )

    def extract_features(self, Y_equalized: np.ndarray) -> np.ndarray:
        """
        Executes feature extraction against the given stride of equalized IQ samples.
        Modifies and returns the internal pre-allocated feature matrix block.
        """
        _extract_rf_dna_kernel(
            Y_equalized,
            self.amp_buffer,
            self.phase_buffer,
            self.freq_buffer,
            self._feature_pool,
            self.num_channels,
            self.stride_len,
            self.window_len,
            self.num_subsegments,
            self.subseg_len
        )
        return self._feature_pool

    def get_flattened_features(self) -> np.ndarray:
        """
        Returns a reshaped zero-copy view of the feature pool matching ONNX tensor requirements.
        Output shape: (num_channels, num_windows * num_subsegments * 9)
        """
        return self._feature_pool.reshape((self.num_channels, -1))


if __name__ == "__main__":
    print("[*] Instantiating RFDnaExtractor and triggering LLVM JIT pre-warm...")
    extractor = RFDnaExtractor()
    
    # Generate mock equalized stride
    t = np.arange(4096)
    c_data = (np.sin(2 * np.pi * 0.05 * t) + 1j * np.cos(2 * np.pi * 0.05 * t)).astype(np.complex64)
    X_mock = np.vstack([c_data] * 4).copy()
    
    print("[*] Running 1,000 benchmark strides...")
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        feats = extractor.extract_features(X_mock)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)
        
    avg_us = np.mean(latencies)
    p99_us = np.percentile(latencies, 99.0)
    
    print(f"\n--- RF-DNA EXTRACTOR PERFORMANCE HUD ---")
    print(f"  Average Stride Latency: {avg_us:.2f} µs")
    print(f"  P99 Stride Latency:      {p99_us:.2f} µs")
    print(f"  Feature Matrix Shape:   {feats.shape}")
    print(f"  Flattened Shape:        {extractor.get_flattened_features().shape}")
    
    if avg_us <= 18.0:
        print("[PASSED] Engine operating well within the strict 18µs latency cap.")
    else:
        print("[FAIL] Performance SLA breached! Verify Numba configuration.")
