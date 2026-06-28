"""
Task 57.4: Sub-Chip Correlator Shape Dissector Block
SpaceShield High-Velocity Receiver DSP Subsystem

Zero-allocation, performance-optimized multi-channel correlation shape analyzer.
Calculates the absolute delta-symmetry of the correlation peak across fractional 
sub-chip intervals (Early, Prompt, Late, and fractional offsets) to detect asymmetric
distortion or multi-peak anomalies indicating coherent spoofing attacks.
"""

import time
import numpy as np
from numba import njit

# Index offsets for 7-tap correlator configurations:
# Tap 0: -0.75 chips
# Tap 1: -0.50 chips (Early)
# Tap 2: -0.25 chips
# Tap 3:  0.00 chips (Prompt)
# Tap 4:  0.25 chips
# Tap 5:  0.50 chips (Late)
# Tap 6:  0.75 chips

@njit(fastmath=True, cache=True, boundscheck=False)
def _dissect_channel_correlators(
    correlation_taps: np.ndarray,      # (num_channels, 7) complex64
    symmetry_scores: np.ndarray,       # (num_channels,) float64 (output)
    threat_flags: np.ndarray,          # (num_channels,) bool_ (output)
    asymmetry_threshold: float,        # trigger threshold for symmetry score
    multi_peak_threshold: float        # slope violation threshold
):
    """
    Zero-Heap Numba JIT Kernel:
    1. Extracts amplitude envelopes from complex correlator outputs.
    2. Calculates asymmetry scores across paired fractional sub-chip offsets.
    3. Checks for multi-peak distortion (violations of monotonic decay from prompt center).
    4. Raises threat flags for channels showing spoofing indicators.
    """
    num_channels = correlation_taps.shape[0]
    
    for m in range(num_channels):
        # 1. Compute absolute magnitudes for 7 taps
        a0 = np.abs(correlation_taps[m, 0])
        a1 = np.abs(correlation_taps[m, 1])
        a2 = np.abs(correlation_taps[m, 2])
        a3 = np.abs(correlation_taps[m, 3])  # Prompt
        a4 = np.abs(correlation_taps[m, 4])
        a5 = np.abs(correlation_taps[m, 5])
        a6 = np.abs(correlation_taps[m, 6])
        
        # Avoid division by zero by normalizing by prompt amplitude (if prompt is non-trivial)
        norm = a3 if a3 > 1e-9 else 1.0
        
        n0 = a0 / norm
        n1 = a1 / norm
        n2 = a2 / norm
        n3 = 1.0
        n4 = a4 / norm
        n5 = a5 / norm
        n6 = a6 / norm
        
        # 2. Compute absolute delta-symmetry scores at multiple sub-chip distances
        d_0_6 = np.abs(n0 - n6)
        d_1_5 = np.abs(n1 - n5)
        d_2_4 = np.abs(n2 - n4)
        
        # Total weighted delta-symmetry score
        total_sym = d_2_4 + d_1_5 + d_0_6
        symmetry_scores[m] = total_sym
        
        # 3. Detect Multi-Peak Distortion (Monotonicity Violation)
        # In a clean correlation triangle, amplitude must strictly decrease moving away from prompt (n3).
        # We look for positive slopes when moving left/right from prompt.
        left_violation = (n1 - n2 > multi_peak_threshold) or (n0 - n1 > multi_peak_threshold)
        right_violation = (n5 - n4 > multi_peak_threshold) or (n6 - n5 > multi_peak_threshold)
        
        # Also check if Prompt is not the peak
        prompt_not_peak = (n2 > 1.05) or (n4 > 1.05)
        
        is_spoofed = (total_sym > asymmetry_threshold) or left_violation or right_violation or prompt_not_peak
        threat_flags[m] = is_spoofed


class CorrelatorDissector:
    """
    SpaceShield Real-Time Correlator Shape Dissector.
    Ingests sub-chip correlator arrays and computes absolute distortion statistics 
    without dynamic memory allocations.
    """
    def __init__(
        self,
        num_channels: int = 4,
        asymmetry_threshold: float = 0.15,
        multi_peak_threshold: float = 0.05
    ):
        self.num_channels = num_channels
        self.asymmetry_threshold = asymmetry_threshold
        self.multi_peak_threshold = multi_peak_threshold
        
        # Pre-allocated zero-allocation NumPy states
        self.symmetry_scores = np.zeros(self.num_channels, dtype=np.float64)
        self.threat_flags = np.zeros(self.num_channels, dtype=np.bool_)
        
        # Pre-warm Numba compiler
        self._warmup()

    def _warmup(self):
        """Forces compilation of Numba kernels."""
        dummy_taps = np.zeros((self.num_channels, 7), dtype=np.complex64)
        # Populate dummy triangle correlation shapes
        for m in range(self.num_channels):
            dummy_taps[m, 0] = 0.25 + 0.0j
            dummy_taps[m, 1] = 0.50 + 0.0j
            dummy_taps[m, 2] = 0.75 + 0.0j
            dummy_taps[m, 3] = 1.00 + 0.0j
            dummy_taps[m, 4] = 0.75 + 0.0j
            dummy_taps[m, 5] = 0.50 + 0.0j
            dummy_taps[m, 6] = 0.25 + 0.0j
            
        _dissect_channel_correlators(
            dummy_taps,
            self.symmetry_scores,
            self.threat_flags,
            self.asymmetry_threshold,
            self.multi_peak_threshold
        )
        self.symmetry_scores.fill(0.0)
        self.threat_flags.fill(False)

    def process_stride(self, correlation_taps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes a single stride of multi-channel correlator data.
        Updates internal zero-heap buffers and returns view references.
        """
        _dissect_channel_correlators(
            correlation_taps,
            self.symmetry_scores,
            self.threat_flags,
            self.asymmetry_threshold,
            self.multi_peak_threshold
        )
        return self.symmetry_scores, self.threat_flags


if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Correlator Shape Dissector")
    print("==================================================================")
    
    dissector = CorrelatorDissector(num_channels=4, asymmetry_threshold=0.15, multi_peak_threshold=0.05)
    
    # Construct Mock Input Data
    # Channel 0: Clean symmetric correlation triangle
    # Channel 1: Asymmetric skew (coherent spoofing overlay)
    # Channel 2: Multi-peak distortion (severe spoofing/multipath multiplet)
    # Channel 3: Low-SNR clean symmetric signal
    mock_taps = np.zeros((4, 7), dtype=np.complex64)
    
    # Ch 0 (Nominal)
    mock_taps[0] = np.array([0.25, 0.50, 0.75, 1.00, 0.75, 0.50, 0.25], dtype=np.complex64)
    
    # Ch 1 (Asymmetric Spoofed)
    mock_taps[1] = np.array([0.20, 0.40, 0.85, 1.00, 0.65, 0.45, 0.20], dtype=np.complex64)
    
    # Ch 2 (Multi-peak Spoofed)
    mock_taps[2] = np.array([0.30, 0.60, 0.50, 1.00, 0.70, 0.80, 0.40], dtype=np.complex64)
    
    # Ch 3 (Nominal Low-SNR with noise phase)
    mock_taps[3] = np.array([
        (0.25 + np.random.normal(0, 0.01)) * np.exp(1j*0.2),
        (0.50 + np.random.normal(0, 0.01)) * np.exp(1j*0.2),
        (0.75 + np.random.normal(0, 0.01)) * np.exp(1j*0.2),
        (1.00 + np.random.normal(0, 0.01)) * np.exp(1j*0.2),
        (0.75 + np.random.normal(0, 0.01)) * np.exp(1j*0.2),
        (0.50 + np.random.normal(0, 0.01)) * np.exp(1j*0.2),
        (0.25 + np.random.normal(0, 0.01)) * np.exp(1j*0.2)
    ], dtype=np.complex64)

    # Process and benchmark
    print("[*] Performing shape dissection verification...")
    scores, threats = dissector.process_stride(mock_taps)
    
    for ch in range(4):
        print(f"  Channel {ch}: Score = {scores[ch]:.4f} | Threat Detected = {threats[ch]}")
        
    assert threats[0] == False, "Channel 0 nominal state misclassified!"
    assert threats[1] == True, "Channel 1 asymmetric spoofing missed!"
    assert threats[2] == True, "Channel 2 multi-peak spoofing missed!"
    assert threats[3] == False, "Channel 3 nominal noisy state misclassified!"
    
    print("\n--- SHAPE DISSECTOR BENCHMARK ---")
    print("[*] Simulating 10,000 continuous real-time verification cycles...")
    
    latencies = []
    for _ in range(10000):
        t0 = time.perf_counter()
        _, _ = dissector.process_stride(mock_taps)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)
        
    avg_us = np.median(latencies)
    p99_us = np.percentile(latencies, 99.0)
    
    print(f"  Median Execution Latency: {avg_us:.3f} µs")
    print(f"  P99 Execution Latency:    {p99_us:.3f} µs")
    
    if avg_us < 10.0:
        print("[PASSED] Correlator Shape Dissector executes beneath 10µs limit.")
    else:
        print("[FAILED] Correlator Shape Dissector overhead exceeded constraints.")
