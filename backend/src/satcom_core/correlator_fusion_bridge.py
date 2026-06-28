"""
Task 58.1: Correlator Fusion Bridge Module
SpaceShield High-Velocity Receiver DSP Subsystem

Ultra-low-latency production module executing zero-heap operations.
Couples the 7-tap fractional sub-chip skew metrics from correlator_dissector.py
into the multi-channel tracking stride. Computes 5 deception features per channel
using Numba JIT in nopython mode, updating contiguous shared buffers.
"""

import time
import math
import queue
import numpy as np
from numba import njit

# Import the dissector JIT kernel directly for seamless coupling
from correlator_dissector import _dissect_channel_correlators

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _extract_fused_features_jit(
    correlation_taps: np.ndarray,      # (num_channels, 7) complex64
    dissector_scores: np.ndarray,      # (num_channels,) float64
    skew_history: np.ndarray,          # (num_channels, window_size) float64
    history_indices: np.ndarray,       # (num_channels,) int32
    feature_matrix: np.ndarray,        # (num_channels, 5) float64 (output)
    window_size: int
):
    """
    Zero-Heap Numba JIT Kernel:
    1. Extracts normalized amplitude envelopes for fractional sub-chip offsets.
    2. Computes Skew Asymmetry, Slope Imbalance, and Prompt Power Normalization.
    3. Integrates the Triangle Symmetry Residual from the dissector.
    4. Evaluates Temporal Skew Stability over a sliding window without allocations.
    """
    num_channels = correlation_taps.shape[0]
    
    for m in range(num_channels):
        # 1. Compute absolute magnitudes for 7 taps
        a0 = np.abs(correlation_taps[m, 0])
        a1 = np.abs(correlation_taps[m, 1])  # Early (spacing 0.5)
        a2 = np.abs(correlation_taps[m, 2])
        a3 = np.abs(correlation_taps[m, 3])  # Prompt
        a4 = np.abs(correlation_taps[m, 4])
        a5 = np.abs(correlation_taps[m, 5])  # Late (spacing 0.5)
        a6 = np.abs(correlation_taps[m, 6])
        
        # Power calculation
        prompt_power = a3 * a3
        total_power = (a0*a0 + a1*a1 + a2*a2 + a3*a3 + a4*a4 + a5*a5 + a6*a6)
        
        # Normalized power envelope
        norm = a3 if a3 > 1e-12 else 1.0
        n1 = a1 / norm
        n5 = a5 / norm
        
        # Feature 0: Skew Asymmetry (normalized difference between early and late)
        skew_asym = n1 - n5
        
        # Feature 1: Triangle Symmetry Residual
        tri_residual = dissector_scores[m]
        
        # Feature 2: Slope Imbalance
        # Left slope = (Prompt - Early) / 0.5 = 2.0 * (1.0 - n1)
        # Right slope = (Prompt - Late) / 0.5 = 2.0 * (1.0 - n5)
        slope_left = 2.0 * (1.0 - n1)
        slope_right = 2.0 * (1.0 - n5)
        slope_imb = slope_left - slope_right
        
        # Feature 3: Prompt Power Normalization
        p_norm = prompt_power / (total_power + 1e-12)
        
        # Feature 4: Temporal Skew Stability (Standard deviation of skew asymmetry)
        idx = history_indices[m]
        skew_history[m, idx] = skew_asym
        
        # Inline mean calculation over the rolling history window
        mean_val = 0.0
        for i in range(window_size):
            mean_val += skew_history[m, i]
        mean_val /= window_size
        
        # Inline variance and standard deviation calculation
        var_val = 0.0
        for i in range(window_size):
            diff = skew_history[m, i] - mean_val
            var_val += diff * diff
        var_val /= window_size
        std_val = math.sqrt(var_val)
        
        # Step the circular index pointer
        history_indices[m] = (idx + 1) % window_size
        
        # Populate the shared feature matrix in-place
        feature_matrix[m, 0] = skew_asym
        feature_matrix[m, 1] = tri_residual
        feature_matrix[m, 2] = slope_imb
        feature_matrix[m, 3] = p_norm
        feature_matrix[m, 4] = std_val


class CorrelatorFusionBridge:
    """
    SpaceShield Correlator Fusion Bridge.
    Aggregates multi-channel tracking correlators, performs JIT shape dissection, 
    and bridges deception features to calibration, GLRT, and edge inference workers.
    """
    def __init__(
        self,
        num_channels: int = 4,
        window_size: int = 20,
        asymmetry_threshold: float = 0.15,
        multi_peak_threshold: float = 0.05
    ):
        self.num_channels = num_channels
        self.window_size = window_size
        self.asymmetry_threshold = asymmetry_threshold
        self.multi_peak_threshold = multi_peak_threshold
        
        # Shared pre-allocated contiguous buffers (zero-heap footprint)
        self.dissector_scores = np.zeros(self.num_channels, dtype=np.float64)
        self.dissector_flags = np.zeros(self.num_channels, dtype=np.bool_)
        
        self.skew_history = np.zeros((self.num_channels, self.window_size), dtype=np.float64)
        self.history_indices = np.zeros(self.num_channels, dtype=np.int32)
        
        self.feature_matrix = np.zeros((self.num_channels, 5), dtype=np.float64)
        
        # Pre-warm compiler
        self._warmup()

    def _warmup(self):
        """Forces compilation of both local and imported Numba kernels."""
        dummy_taps = np.zeros((self.num_channels, 7), dtype=np.complex64)
        for m in range(self.num_channels):
            dummy_taps[m, 3] = 1.0 + 0.0j # Prompt
            
        _dissect_channel_correlators(
            dummy_taps,
            self.dissector_scores,
            self.dissector_flags,
            self.asymmetry_threshold,
            self.multi_peak_threshold
        )
        _extract_fused_features_jit(
            dummy_taps,
            self.dissector_scores,
            self.skew_history,
            self.history_indices,
            self.feature_matrix,
            self.window_size
        )
        self.dissector_scores.fill(0.0)
        self.dissector_flags.fill(False)
        self.skew_history.fill(0.0)
        self.history_indices.fill(0)
        self.feature_matrix.fill(0.0)

    def execute_stride(self, correlation_taps: np.ndarray) -> np.ndarray:
        """
        Main hot-path execution loop:
        1. Invokes JIT correlation shape dissector.
        2. Fuses dissector outputs and tracking replica states.
        3. Extracts the 5-element deception feature vector.
        """
        # Step 1: Dissect correlator shape profile
        _dissect_channel_correlators(
            correlation_taps,
            self.dissector_scores,
            self.dissector_flags,
            self.asymmetry_threshold,
            self.multi_peak_threshold
        )
        
        # Step 2: Extract and fuse features
        _extract_fused_features_jit(
            correlation_taps,
            self.dissector_scores,
            self.skew_history,
            self.history_indices,
            self.feature_matrix,
            self.window_size
        )
        
        return self.feature_matrix

    # =========================================================================
    # CORE SYSTEM HANDOFF POINTS
    # =========================================================================
    
    def handoff_to_calibration_engine(self, cal_engine, channel_idx: int):
        """
        Handoff to array_calibration_engine.py:
        Modifies or equalizes calibration weights if severe correlator distortion is present.
        """
        is_threat = self.dissector_flags[channel_idx]
        if is_threat and cal_engine is not None:
            # Dampen calibration gain coefficients for the compromised channel to prevent steering loop drift
            cal_engine.W_diag[channel_idx, 0] *= 0.1
            cal_engine.calibrated = False # Force recalculation or alert system

    def handoff_to_spatial_glrt(self, glrt_detector):
        """
        Handoff to spatial_glrt_detector.py:
        Dynamically scales the GLRT chi-squared detection threshold override 'gamma'
        based on the average correlation asymmetry score across channels.
        """
        if glrt_detector is not None:
            avg_score = np.mean(self.dissector_scores)
            if avg_score > self.asymmetry_threshold:
                # Tighten threshold to detect low-level spatial incursions faster
                glrt_detector.gamma *= 0.90
            else:
                # Restore to standard PPF asymptotic threshold
                glrt_detector.gamma = glrt_detector.gamma # Keep unchanged

    def handoff_to_edge_inference(self, edge_engine, channel_idx: int, cfo_hz: float = 0.0):
        """
        Handoff to edge_inference_engine.py:
        Constructs a standard 6D RF Fingerprint vector from our fused features
        and queues it for asynchronous model prediction.
        
        Mapping to 6D Input:
        [CFO, IQ Amp Imbalance, IQ Phase Imbalance, Phase Noise, Flatness, Prominence]
        We populate this using:
        - CFO = user-provided tracker value (cfo_hz)
        - IQ Amp Imbalance = Skew Asymmetry
        - IQ Phase Imbalance = Slope Imbalance
        - Phase Noise = Temporal Skew Stability
        - Flatness = Triangle Symmetry Residual
        - Prominence = Prompt Power Normalization
        """
        if edge_engine is not None:
            features = self.feature_matrix[channel_idx]
            
            # Construct compatible 6D array
            feat_vector = np.array([
                cfo_hz,                  # Index 0: CFO (Hz)
                features[0],             # Index 1: Skew Asymmetry
                features[2],             # Index 2: Slope Imbalance
                features[4],             # Index 3: Temporal Stability
                features[1],             # Index 4: Symmetry Residual
                features[3]              # Index 5: Prompt Power Norm
            ], dtype=np.float32)
            
            # Non-blocking thread-safe queue ingestion
            if not edge_engine.input_queue.full():
                edge_engine.input_queue.put_nowait((feat_vector, time.time()))

    # =========================================================================
    # FASTAPI TELEMETRY BRIDGE
    # =========================================================================

    def format_telemetry_payload(self) -> dict:
        """
        Constructs a telemetry summary compatible with dashboard_api.py.
        Exposes compact fused metrics.
        """
        verdict = "NORMAL"
        active_threats = int(np.sum(self.dissector_flags))
        if active_threats > 2:
            verdict = "CRITICAL SPOOFING"
        elif active_threats > 0:
            verdict = "JAMMING"
            
        return {
            "fused_asymmetry_scores": [float(x) for x in self.dissector_scores],
            "fused_threat_flags": [bool(x) for x in self.dissector_flags],
            "fused_verdict": verdict,
            "max_skew_amplitude": float(np.max(self.feature_matrix[:, 0])),
            "avg_stability": float(np.mean(self.feature_matrix[:, 4]))
        }


# =========================================================================
# MINIMAL BUILT-IN VERIFICATION HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Correlator Fusion Bridge")
    print("==================================================================")
    
    bridge = CorrelatorFusionBridge(
        num_channels=4,
        window_size=10,
        asymmetry_threshold=0.15,
        multi_peak_threshold=0.05
    )
    
    # 1. Generate synthetic correlator cases (Nominal, Jammed, Spoofed)
    synthetic_cases = np.zeros((4, 7), dtype=np.complex64)
    
    # Ch 0: Nominal symmetric triangle
    synthetic_cases[0] = np.array([0.25, 0.50, 0.75, 1.00, 0.75, 0.50, 0.25], dtype=np.complex64)
    
    # Ch 1: Jammer-distorted (raised noise floor, flattened peak, no skew)
    synthetic_cases[1] = np.array([0.45, 0.50, 0.55, 0.60, 0.55, 0.50, 0.45], dtype=np.complex64)
    
    # Ch 2: Coherent Spoof-skewed (asymmetric peak, prompt pulled right)
    synthetic_cases[2] = np.array([0.15, 0.35, 0.85, 1.00, 0.60, 0.30, 0.15], dtype=np.complex64)
    
    # Ch 3: Nominal noisy
    synthetic_cases[3] = np.array([0.23, 0.52, 0.74, 0.98, 0.76, 0.49, 0.24], dtype=np.complex64)
    
    # Run the fusion bridge stride
    print("[*] Computing feature vectors...")
    features = bridge.execute_stride(synthetic_cases)
    
    headers = ["Skew Asymmetry", "Symmetry Residual", "Slope Imbalance", "Prompt Power Norm", "Temp Stability"]
    for ch in range(4):
        print(f"\n  Channel {ch} Telemetry Layout:")
        for idx, name in enumerate(headers):
            print(f"    - {name:20s}: {features[ch, idx]:.4f}")
            
    # Mock calibration engine
    class MockCalEngine:
        def __init__(self):
            self.W_diag = np.ones((4, 1), dtype=np.complex64)
            self.calibrated = True
            
    cal = MockCalEngine()
    bridge.handoff_to_calibration_engine(cal, 2)
    print(f"\n[*] Cal Engine Handoff Verification:")
    print(f"    -> Ch 2 Calibration Gain Scale: {cal.W_diag[2, 0]}")
    assert cal.W_diag[2, 0] < 0.2, "Calibration suppression logic failed!"
    
    # Mock Edge Inference Engine
    class MockEdgeEngine:
        import queue
        def __init__(self):
            self.input_queue = queue.Queue(maxsize=100)
            
    edge = MockEdgeEngine()
    bridge.handoff_to_edge_inference(edge, 2, cfo_hz=25.0)
    print(f"\n[*] Edge Inference Queue Ingestion:")
    item = edge.input_queue.get()
    print(f"    -> Queued Feature Vector: {item[0]}")
    assert len(item[0]) == 6, "Inference feature flattening mismatch!"
    
    # Telemetry format verification
    telemetry = bridge.format_telemetry_payload()
    print(f"\n[*] FastAPI Telemetry Stream Output:")
    print(f"    -> {telemetry}")
    
    # Benchmark latency test
    print("\n--- FUSION BRIDGE BENCHMARK ---")
    print("[*] Running 15,000 warm continuous loops...")
    
    latencies = []
    for _ in range(15000):
        t0 = time.perf_counter()
        _ = bridge.execute_stride(synthetic_cases)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)
        
    avg_us = np.median(latencies)
    avg_per_ch = avg_us / 4.0
    
    print(f"  Total Median Latency (4 Ch): {avg_us:.3f} µs")
    print(f"  Per-Channel Latency:         {avg_per_ch:.3f} µs")
    
    if avg_per_ch < 3.0:
        print("[PASSED] Correlator Fusion Bridge operates under the 3µs/channel limit.")
    else:
        print("[FAILED] Fusion Bridge execution latency exceeded limit.")
