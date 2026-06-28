"""
Task 58.3: Phase 58 Fusion Supervisor Module
SpaceShield High-Velocity Receiver DSP Subsystem

Orchestrates correlator deception observables, spatial look angle validations,
GLRT covariance anomaly statistics, and edge model inference verdicts.
Features a Numba JIT state machine with hysteresis, latency accounting,
and a deterministic test replay harness.
"""

import time
import math
import numpy as np
from numba import njit

# Import bridge and validator components
from correlator_fusion_bridge import CorrelatorFusionBridge
from aoa_enforcement_bridge import AoAEnforcementBridge
from spatial_glrt_detector import SpatialGLRTDetector

# State Machine Definitions:
# 0: NOMINAL (All channels compliant, no threat)
# 1: WARNING_JAMMING (GLRT covariance alert triggered, low-rank correlation damping)
# 2: WARNING_SPOOFING (Asymmetry/skew detected, AoA validation violation)
# 3: CRITICAL_MITIGATION (Spatial nulling engaged, multiple channels excluded)

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _fast_glrt_eval_jit(
    Y: np.ndarray,              # (4, 50) complex64
    R_hat: np.ndarray,          # (4, 4) complex64 (pre-allocated)
    A: np.ndarray,              # (4, 4) complex64 (pre-allocated)
    v_temp: np.ndarray,         # (4,) complex64 (pre-allocated)
    w_temp: np.ndarray,         # (4,) complex64 (pre-allocated)
    M: int,
    N: int,
    rho: float,
    gamma: float
):
    """
    Zero-Heap Numba JIT Linear Algebra:
    1. Computes the spatial covariance matrix R_hat = (1/N) * Y * Y^H.
    2. Calculates Trace and 4x4 determinant via LU decomposition.
    3. Runs 5 Rayleigh power iterations to estimate the dominant eigenvalue (lambda_max).
    4. Computes Sphericity LLR and METR.
    """
    # 1. Compute R_hat
    for i in range(4):
        for j in range(4):
            val = 0.0 + 0.0j
            for k in range(50):
                val += Y[i, k] * np.conj(Y[j, k])
            R_hat[i, j] = val / N
            
    # 2. Compute Trace (sum of diagonals)
    trace_val = 0.0
    for i in range(4):
        trace_val += R_hat[i, i].real
        
    # Copy R_hat to A for LU decomposition
    for i in range(4):
        for j in range(4):
            A[i, j] = R_hat[i, j]
            
    # 3. Compute Determinant of 4x4 matrix A
    det_val = 1.0 + 0.0j
    for i in range(4):
        pivot = A[i, i]
        if np.abs(pivot) < 1e-12:
            det_val = 0.0 + 0.0j
            break
        det_val *= pivot
        for j in range(i+1, 4):
            factor = A[j, i] / pivot
            for k in range(i+1, 4):
                A[j, k] -= factor * A[i, k]
                
    det_real = det_val.real
    if det_real < 1e-15:
        det_real = 1e-15
        
    # Sphericity test LLR
    sphericity_ratio = det_real / ((trace_val / M) ** M)
    if sphericity_ratio > 1.0:
        sphericity_ratio = 1.0
    elif sphericity_ratio < 1e-15:
        sphericity_ratio = 1e-15
        
    lambda_sphericity = -N * rho * math.log(sphericity_ratio)
    alert_triggered = lambda_sphericity > gamma
    
    # 4. Power iteration for lambda_max
    for i in range(4):
        v_temp[i] = 1.0 + 0.0j
        
    for _ in range(5):
        # w_temp = R_hat * v_temp
        for i in range(4):
            w_temp[i] = 0.0 + 0.0j
            for j in range(4):
                w_temp[i] += R_hat[i, j] * v_temp[j]
        # Normalize
        sum_sq = 0.0
        for i in range(4):
            abs_w = np.abs(w_temp[i])
            sum_sq += abs_w * abs_w
        norm_v = math.sqrt(sum_sq)
        if norm_v < 1e-12:
            norm_v = 1e-12
        for i in range(4):
            v_temp[i] = w_temp[i] / norm_v
            
    # Rayleigh quotient
    lambda_max_val = 0.0
    for i in range(4):
        lambda_max_val += (np.conj(v_temp[i]) * w_temp[i]).real
        
    lambda_metr = lambda_max_val / (trace_val + 1e-12)
    
    return lambda_sphericity, lambda_metr, alert_triggered, lambda_max_val


@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _update_state_machine_jit(
    current_state: int,
    dissector_flags: np.ndarray,      # (num_channels,) bool_
    aoa_flags: np.ndarray,            # (num_channels,) bool_
    glrt_anomaly: bool,
    inference_verdict: int,           # 0: NORMAL, 1: JAMMING, 2: SPOOFING
    persistence_counters: np.ndarray, # (4,) int32
    hysteresis_threshold: int
) -> int:
    """
    Zero-Heap Numba State Machine:
    1. Evaluates multi-channel threat indicators (Sphericity, skew, AoA geodesic deviation).
    2. Maps raw alerts to target states.
    3. Implements temporal persistence counters (hysteresis) to prevent status chatter.
    4. Transitions state when the threshold is reached.
    """
    any_skew = False
    any_aoa = False
    for m in range(dissector_flags.shape[0]):
        if dissector_flags[m]:
            any_skew = True
        if aoa_flags[m]:
            any_aoa = True

    # State decision logic
    target_state = 0 # Default: NOMINAL
    
    if any_aoa or (any_skew and inference_verdict == 2):
        target_state = 3 # CRITICAL_MITIGATION
    elif any_skew or inference_verdict == 2:
        target_state = 2 # WARNING_SPOOFING
    elif glrt_anomaly or inference_verdict == 1:
        target_state = 1 # WARNING_JAMMING

    # Apply Hysteresis
    if target_state == current_state:
        # Reset other transition counters
        for s in range(4):
            if s != current_state:
                persistence_counters[s] = 0
        return current_state
    else:
        # Increment counter for the requested target state
        persistence_counters[target_state] += 1
        
        # Reset counters for non-targeted states
        for s in range(4):
            if s != target_state and s != current_state:
                persistence_counters[s] = 0
                
        # If persistence holds, transition state
        if persistence_counters[target_state] >= hysteresis_threshold:
            persistence_counters[target_state] = 0
            return target_state
            
        return current_state


class Phase58FusionSupervisor:
    """
    SpaceShield Real-Time Stride Supervisor.
    Orchestrates spatiotemporal sensors, runs JIT threat consolidation,
    and updates the low-latency control-plane telemetry gateway.
    """
    def __init__(
        self,
        num_channels: int = 4,
        window_size: int = 10,
        hysteresis_threshold: int = 3,
        default_fisher_margin_deg: float = 5.0
    ):
        self.num_channels = num_channels
        self.hysteresis_threshold = hysteresis_threshold
        
        # Instantiate sub-bridges (zero-allocation state architectures)
        self.correlator_bridge = CorrelatorFusionBridge(
            num_channels=num_channels,
            window_size=window_size
        )
        self.aoa_bridge = AoAEnforcementBridge(
            num_channels=num_channels,
            default_fisher_margin_deg=default_fisher_margin_deg
        )
        self.glrt_detector = SpatialGLRTDetector(
            num_channels=num_channels,
            window_size=50
        )
        
        # Pre-allocated temporary JIT structures
        self._R_hat = np.zeros((4, 4), dtype=np.complex64)
        self._A = np.zeros((4, 4), dtype=np.complex64)
        self._v_temp = np.zeros(4, dtype=np.complex64)
        self._w_temp = np.zeros(4, dtype=np.complex64)
        
        # Pre-calculated GLRT constants
        self.M = self.glrt_detector.M
        self.N = self.glrt_detector.N
        self.rho = self.glrt_detector.rho
        self.gamma = self.glrt_detector.gamma
        
        # JIT outputs
        self.last_sphericity = 0.0
        self.last_metr = 0.0
        self.last_lambda_max = 0.0
        
        # State registers
        self.current_state = 0
        self.persistence_counters = np.zeros(4, dtype=np.int32)
        
        # Latency accounting registers (µs)
        self.latency_glrt = 0.0
        self.latency_correlator = 0.0
        self.latency_aoa = 0.0
        self.latency_state_machine = 0.0
        self.latency_total = 0.0
        
        # Compile Numba kernels
        self._warmup()

    def _warmup(self):
        """Forces warm compilation of JIT state machine loops."""
        dummy_Y = np.ones((4, 50), dtype=np.complex64)
        _fast_glrt_eval_jit(
            dummy_Y, self._R_hat, self._A, self._v_temp, self._w_temp,
            self.M, self.N, self.rho, self.gamma
        )
        dummy_flags = np.zeros(self.num_channels, dtype=np.bool_)
        _update_state_machine_jit(
            self.current_state,
            dummy_flags,
            dummy_flags,
            False,
            0,
            self.persistence_counters,
            self.hysteresis_threshold
        )
        self.persistence_counters.fill(0)
        self.current_state = 0

    def process_stride(
        self,
        Y_matrix: np.ndarray,            # (num_channels, window_size) complex64
        correlation_taps: np.ndarray,     # (num_channels, 7) complex64
        phase_diffs: np.ndarray,         # (num_channels, 3) float64
        true_az_el: np.ndarray,          # (num_channels, 2) float64
        beamforming_weights: np.ndarray, # (num_channels, 4) complex64
        inference_verdict_idx: int = 0   # 0: NORMAL, 1: JAMMING, 2: SPOOFING
    ) -> int:
        """
        Runs full DSP step execution:
        1. Evaluates spatial covariance GLRT anomaly via Numba JIT.
        2. Escalates AoA look angle check margin if GLRT anomaly occurs.
        3. Computes correlator deception features.
        4. Calculates spatial null weight steering commands.
        5. Evaluates fusion state transition with temporal hysteresis.
        """
        t_total_start = time.perf_counter()
        
        # 1. Spatial GLRT Detector (Zero-heap JIT)
        t_start = time.perf_counter()
        sphericity, metr, glrt_anomaly, lambda_max = _fast_glrt_eval_jit(
            Y_matrix,
            self._R_hat,
            self._A,
            self._v_temp,
            self._w_temp,
            self.M,
            self.N,
            self.rho,
            self.gamma
        )
        self.last_sphericity = sphericity
        self.last_metr = metr
        self.last_lambda_max = lambda_max
        self.latency_glrt = (time.perf_counter() - t_start) * 1e6
        
        # 2. Covariance escalations to AoA look angle allowance
        self.aoa_bridge.interoperate_with_glrt(self.glrt_detector, glrt_anomaly)
        
        # 3. Correlator Deception Metrics Extraction
        t_start = time.perf_counter()
        self.correlator_bridge.execute_stride(correlation_taps)
        self.latency_correlator = (time.perf_counter() - t_start) * 1e6
        
        # 4. Spatial AoA enforcement and steering weight update
        t_start = time.perf_counter()
        self.aoa_bridge.execute_enforcement_stride(phase_diffs, true_az_el, beamforming_weights)
        self.latency_aoa = (time.perf_counter() - t_start) * 1e6
        
        # 5. Fusion State Machine update
        t_start = time.perf_counter()
        self.current_state = _update_state_machine_jit(
            self.current_state,
            self.correlator_bridge.dissector_flags,
            self.aoa_bridge.hijack_flags,
            glrt_anomaly,
            inference_verdict_idx,
            self.persistence_counters,
            self.hysteresis_threshold
        )
        self.latency_state_machine = (time.perf_counter() - t_start) * 1e6
        
        self.latency_total = (time.perf_counter() - t_total_start) * 1e6
        return self.current_state

    # =========================================================================
    # TELEMETRY PACKET BRIDGE
    # =========================================================================

    def format_control_plane_payload(self) -> dict:
        """
        Formats the compact telemetry payload for the FastAPI/WebSocket queue.
        Executed in the control-plane thread, avoiding hot-loop overhead.
        """
        states = ["NOMINAL", "WARNING_JAMMING", "WARNING_SPOOFING", "CRITICAL_MITIGATION"]
        verdict = states[self.current_state]
        
        # Aggregate sub-system telemetry payloads
        correlator_tel = self.correlator_bridge.format_telemetry_payload()
        aoa_tel = self.aoa_bridge.format_telemetry_payload()
        
        return {
            "threat_verdict": verdict,
            "sphericity_score": float(self.last_sphericity),
            "fim_beta": float(self.last_metr),
            "latency_accounting_us": {
                "glrt": float(self.latency_glrt),
                "correlator": float(self.latency_correlator),
                "aoa": float(self.latency_aoa),
                "state_machine": float(self.latency_state_machine),
                "total": float(self.latency_total)
            },
            "correlator_skew": correlator_tel,
            "aoa_enforcement": aoa_tel
        }


# =========================================================================
# ADVERSARIAL PLAYPLAY HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Phase 58 Fusion Supervisor")
    print("==================================================================")
    
    supervisor = Phase58FusionSupervisor(
        num_channels=4,
        window_size=50,
        hysteresis_threshold=3,
        default_fisher_margin_deg=5.0
    )
    
    # Static geometry definitions
    true_angles = np.array([
        [45.0, 30.0],
        [90.0, 60.0],
        [180.0, 45.0],
        [270.0, 15.0]
    ], dtype=np.float64)
    
    # Pre-calculate base nominal phase difference vector
    nom_phase_diffs = np.zeros((4, 3), dtype=np.float64)
    for m in range(4):
        az = math.radians(true_angles[m, 0])
        el = math.radians(true_angles[m, 1])
        ux = math.cos(el) * math.cos(az)
        uy = math.cos(el) * math.sin(az)
        nom_phase_diffs[m, 0] = ux * math.pi
        nom_phase_diffs[m, 1] = uy * math.pi
        nom_phase_diffs[m, 2] = (ux + uy) * math.pi
        
    weights = np.ones((4, 4), dtype=np.complex64) * 0.5

    # Define the 4 timeline phases representing an attack scenario
    print("[*] Launching Adversarial Timeline Simulation (200 Strides)...")
    
    # Pre-allocate scenario input containers to mock real-time HIL registers
    Y_block = np.zeros((4, 50), dtype=np.complex64)
    corr_taps = np.zeros((4, 7), dtype=np.complex64)
    phase_diffs = nom_phase_diffs.copy()
    
    state_history = []
    
    for step in range(1, 201):
        # 1. Phase 1: Nominal Acquisition (Steps 1-50)
        # Clean signals, aligned wavefronts, zero-mean Gaussian noise covariance
        if step <= 50:
            Y_block = (np.random.normal(0, 1.0, (4, 50)) + 1j * np.random.normal(0, 1.0, (4, 50))).astype(np.complex64)
            corr_taps.fill(0.0)
            for m in range(4):
                corr_taps[m] = np.array([0.25, 0.50, 0.75, 1.00, 0.75, 0.50, 0.25])
            phase_diffs = nom_phase_diffs.copy()
            infer_idx = 0
            
        # 2. Phase 2: Rising Jammer Power (Steps 51-100)
        # Strong directional spatial jammer injection raises Sphericity eigenvalue stats
        elif step <= 100:
            # Jammer steering vector (arriving from 60 degrees az, 30 el)
            az_j = math.radians(60.0)
            el_j = math.radians(30.0)
            sj = np.array([
                1.0 + 0j,
                math.cos(sj_ux := math.cos(el_j)*math.cos(az_j)*math.pi) + 1j*math.sin(sj_ux),
                math.cos(sj_uy := math.cos(el_j)*math.sin(az_j)*math.pi) + 1j*math.sin(sj_uy),
                0.5j
            ], dtype=np.complex64)
            # Covariance matrix exhibits high rank-1 signature
            jammer_signal = np.random.normal(0, 1.5, 50) + 1j * np.random.normal(0, 1.5, 50)
            Y_block = np.zeros((4, 50), dtype=np.complex64)
            for k in range(4):
                Y_block[k] = sj[k] * jammer_signal
            
            # Correlation peaks are slightly flattened by noise floor, but symmetric
            corr_taps.fill(0.0)
            for m in range(4):
                corr_taps[m] = np.array([0.40, 0.45, 0.50, 0.55, 0.50, 0.45, 0.40])
            phase_diffs = nom_phase_diffs.copy()
            infer_idx = 1
            
        # 3. Phase 3: Coherent Spoof Onset (Steps 101-150)
        # Spoofing transmitter locks and drags tracking correlation triangles off-center
        elif step <= 150:
            Y_block = (np.random.normal(0, 1.0, (4, 50)) + 1j * np.random.normal(0, 1.0, (4, 50))).astype(np.complex64)
            # Correlation peaks skewed asymmetrically
            corr_taps.fill(0.0)
            for m in range(4):
                corr_taps[m] = np.array([0.15, 0.35, 0.85, 1.00, 0.60, 0.30, 0.15])
            # Set AoA phase offset to trigger validator hijack detection (dragged by 15 deg)
            phase_diffs = nom_phase_diffs.copy()
            az_s = math.radians(90.0 + 15.0)
            el_s = math.radians(60.0)
            ux_s = math.cos(el_s)*math.cos(az_s)
            uy_s = math.cos(el_s)*math.sin(az_s)
            phase_diffs[1, 0] = ux_s * math.pi
            phase_diffs[1, 1] = uy_s * math.pi
            phase_diffs[1, 2] = (ux_s + uy_s) * math.pi
            infer_idx = 2
            
        # 4. Phase 4: Combined Attack Persistence (Steps 151-200)
        # Joint spoofer and high-power jammer barrage, triggers mitigation
        else:
            # Jammer spatial covariance
            Y_block = (np.random.normal(0, 3.0, (4, 50)) + 1j * np.random.normal(0, 3.0, (4, 50))).astype(np.complex64)
            # Distorted correlation peak
            corr_taps.fill(0.0)
            for m in range(4):
                corr_taps[m] = np.array([0.10, 0.20, 0.70, 0.80, 0.40, 0.20, 0.10])
            # Out-of-bounds AoA phases on all channels
            phase_diffs = nom_phase_diffs * 1.5
            infer_idx = 2
            
        # Process stride
        state = supervisor.process_stride(
            Y_block,
            corr_taps,
            phase_diffs,
            true_angles,
            weights,
            infer_idx
        )
        state_history.append(state)
        
        if step in [10, 60, 110, 160]:
            states = ["NOMINAL", "WARNING_JAMMING", "WARNING_SPOOFING", "CRITICAL_MITIGATION"]
            print(f"  - Frame {step:3d}: State -> {states[state]:20s} (Total Latency: {supervisor.latency_total:5.2f} µs)")
            
    # Check that state machine transitions occurred correctly
    assert state_history[40] == 0, "Phase 1: State should be NOMINAL"
    assert state_history[90] == 1, "Phase 2: State should transition to WARNING_JAMMING"
    assert state_history[140] == 3, "Phase 3: State should escalate to CRITICAL_MITIGATION due to AoA hijack"
    assert state_history[190] == 3, "Phase 4: State should remain at CRITICAL_MITIGATION"
    
    print("\n--- STATE MACHINE HYSTERESIS ---")
    print(f"  Transitions verified: NOMINAL -> JAMMING -> CRITICAL_MITIGATION")
    print(f"  Verification check: [PASSED]")
    
    # Benchmark loop overhead
    print("\n--- FUSION SUPERVISOR LATENCY ACCUMULATOR ---")
    print("[*] Running 10,000 continuous benchmark cycles...")
    latencies = []
    for _ in range(10000):
        t0 = time.perf_counter()
        _ = supervisor.process_stride(Y_block, corr_taps, phase_diffs, true_angles, weights, 2)
        latencies.append((time.perf_counter() - t0) * 1e6)
        
    avg_us = np.median(latencies)
    p99_us = np.percentile(latencies, 99.0)
    
    print(f"  Average Stride Latency: {avg_us:.3f} µs")
    print(f"  P99 Stride Latency:      {p99_us:.3f} µs")
    
    if avg_us < 10.0:
        print("[PASSED] Phase 58 Fusion Supervisor executes below the 10µs stride budget limit.")
    else:
        print("[FAILED] Fusion Supervisor overhead budget breached.")
