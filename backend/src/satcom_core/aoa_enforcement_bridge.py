"""
Task 58.2: Spatial AoA Enforcement Bridge Module
SpaceShield High-Velocity Receiver DSP Subsystem

Standalone phased-array geolocation geolocation enforcement bridge.
Ingests unit vectors and Fisher Margin outputs from aoa_validator.py,
translates them into null-steering enforcement descriptors, and bridges
them to spatial_glrt_detector.py and the FastAPI telemetry stream.
"""

import time
import math
import numpy as np
from numba import njit

# Import JIT validator kernel directly for seamless processing
from aoa_validator import _validate_aoa_and_project_nulls

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _prepare_null_steering_descriptors_jit(
    hijack_flags: np.ndarray,         # (num_channels,) bool_
    angular_errors: np.ndarray,       # (num_channels,) float64
    phase_diffs: np.ndarray,          # (num_channels, 3) float64
    bearing_confidence: np.ndarray,   # (num_channels,) float64 (output)
    null_depth_demand: np.ndarray,    # (num_channels,) float64 (output)
    exclusion_mask: np.ndarray,       # (num_channels,) bool_ (output)
    trust_weights: np.ndarray,        # (num_channels,) float64 (output)
    suppression_coeffs: np.ndarray,   # (num_channels, 4) complex64 (output)
    fisher_margin_rad: float
):
    """
    Zero-Heap Numba JIT Kernel:
    1. Transforms geodesic angular errors into threat bearing confidences.
    2. Calculates decibel-scaled null-depth demands based on angular separation ratios.
    3. Builds binary beam exclusion masks for tracking loop isolation.
    4. Modulates channel trust weights (1.0 for secure, scaling to 0.0 for spoofed).
    5. Formulates conjugate steer suppression coefficients for spatial null injection.
    """
    num_channels = hijack_flags.shape[0]
    
    for m in range(num_channels):
        err = angular_errors[m]
        is_threat = hijack_flags[m]
        
        if is_threat:
            ratio = err / (fisher_margin_rad + 1e-12)
            
            # Threat bearing confidence (exponential growth with error ratio)
            bearing_confidence[m] = 1.0 - math.exp(-ratio)
            
            # Decibel null-depth demand (scales between 45dB and 70dB)
            db_demand = 45.0 + 25.0 * (1.0 - math.exp(-ratio + 1.0))
            if db_demand > 70.0:
                db_demand = 70.0
            null_depth_demand[m] = db_demand
            
            # Channel exclusion mask
            exclusion_mask[m] = True
            
            # Trust weight decay mapping
            trust_weights[m] = 1.0 / (1.0 + ratio * ratio)
            
            # Conjugate steering suppression vector: s^H = [1, e^-jpsi1, e^-jpsi2, e^-jpsi3]
            psi1 = phase_diffs[m, 0]
            psi2 = phase_diffs[m, 1]
            psi3 = phase_diffs[m, 2]
            
            suppression_coeffs[m, 0] = 1.0 + 0.0j
            suppression_coeffs[m, 1] = math.cos(psi1) - 1j * math.sin(psi1)
            suppression_coeffs[m, 2] = math.cos(psi2) - 1j * math.sin(psi2)
            suppression_coeffs[m, 3] = math.cos(psi3) - 1j * math.sin(psi3)
        else:
            bearing_confidence[m] = 0.0
            null_depth_demand[m] = 0.0
            exclusion_mask[m] = False
            trust_weights[m] = 1.0
            
            for k in range(4):
                suppression_coeffs[m, k] = 0.0 + 0.0j


class AoAEnforcementBridge:
    """
    SpaceShield Angle-of-Arrival Enforcement Bridge.
    Validates physical wave arrival paths, maps them to array nulling structures,
    and updates global telemetry registers without dynamic memory footprints.
    """
    def __init__(
        self,
        num_channels: int = 4,
        element_spacing: float = 0.127,  # meters
        lambda_val: float = 0.254,       # meters
        default_fisher_margin_deg: float = 5.0
    ):
        self.num_channels = num_channels
        self.element_spacing = element_spacing
        self.lambda_val = lambda_val
        self.default_fisher_margin_rad = math.radians(default_fisher_margin_deg)
        self.active_fisher_margin_rad = self.default_fisher_margin_rad
        
        # Pre-allocated zero-allocation NumPy arrays (buffer reuse across strides)
        self.hijack_flags = np.zeros(self.num_channels, dtype=np.bool_)
        self.angular_errors = np.zeros(self.num_channels, dtype=np.float64)
        
        self.bearing_confidence = np.zeros(self.num_channels, dtype=np.float64)
        self.null_depth_demand = np.zeros(self.num_channels, dtype=np.float64)
        self.exclusion_mask = np.zeros(self.num_channels, dtype=np.bool_)
        self.trust_weights = np.ones(self.num_channels, dtype=np.float64)
        self.suppression_coeffs = np.zeros((self.num_channels, 4), dtype=np.complex64)
        
        # Pre-warm compiler
        self._warmup()

    def _warmup(self):
        """Forces compilation of Numba kernels."""
        dummy_diffs = np.zeros((self.num_channels, 3), dtype=np.float64)
        dummy_true = np.zeros((self.num_channels, 2), dtype=np.float64)
        dummy_weights = np.ones((self.num_channels, 4), dtype=np.complex64) * 0.5
        
        _validate_aoa_and_project_nulls(
            dummy_diffs,
            dummy_true,
            dummy_weights,
            self.hijack_flags,
            self.angular_errors,
            self.element_spacing,
            self.lambda_val,
            self.active_fisher_margin_rad
        )
        _prepare_null_steering_descriptors_jit(
            self.hijack_flags,
            self.angular_errors,
            dummy_diffs,
            self.bearing_confidence,
            self.null_depth_demand,
            self.exclusion_mask,
            self.trust_weights,
            self.suppression_coeffs,
            self.active_fisher_margin_rad
        )
        self.hijack_flags.fill(False)
        self.angular_errors.fill(0.0)
        self.bearing_confidence.fill(0.0)
        self.null_depth_demand.fill(0.0)
        self.exclusion_mask.fill(False)
        self.trust_weights.fill(1.0)
        self.suppression_coeffs.fill(0.0 + 0.0j)

    def execute_enforcement_stride(
        self,
        phase_diffs: np.ndarray,
        true_az_el: np.ndarray,
        beamforming_weights: np.ndarray
    ) -> tuple:
        """
        Processes a single geometric tracking stride:
        1. Invokes the geometric validation kernel.
        2. Computes the null-steering descriptors in-place.
        """
        # Step 1: Run geometric spatial look angle validation and update weights inline
        _validate_aoa_and_project_nulls(
            phase_diffs,
            true_az_el,
            beamforming_weights,
            self.hijack_flags,
            self.angular_errors,
            self.element_spacing,
            self.lambda_val,
            self.active_fisher_margin_rad
        )
        
        # Step 2: Compute steering null enforcement parameters
        _prepare_null_steering_descriptors_jit(
            self.hijack_flags,
            self.angular_errors,
            phase_diffs,
            self.bearing_confidence,
            self.null_depth_demand,
            self.exclusion_mask,
            self.trust_weights,
            self.suppression_coeffs,
            self.active_fisher_margin_rad
        )
        
        return (
            self.bearing_confidence,
            self.null_depth_demand,
            self.exclusion_mask,
            self.trust_weights,
            self.suppression_coeffs
        )

    # =========================================================================
    # GLRT ANOMALY INTEROPERABILITY
    # =========================================================================

    def interoperate_with_glrt(self, glrt_detector, covariance_anomaly: bool):
        """
        Dynamic revalidation and mitigation escalation trigger:
        If spatial_glrt_detector.py reports a covariance anomaly (Sphericity score
        exceeds statistical threshold), we escalate by narrowing our Fisher Margin
        radially, causing the physical layer check to operate with heightened sensitivity.
        """
        if covariance_anomaly:
            # Narrow look angle allowance window by 60% to force revalidation of margins
            self.active_fisher_margin_rad = self.default_fisher_margin_rad * 0.40
        else:
            # Restore to standard default configuration bounds
            self.active_fisher_margin_rad = self.default_fisher_margin_rad

    # =========================================================================
    # TELEMETRY EXPOSURE BRIDGE
    # =========================================================================

    def format_telemetry_payload(self) -> dict:
        """Constructs compact metric updates for dashboard_api.py."""
        return {
            "bearing_confidences": [float(x) for x in self.bearing_confidence],
            "null_demands_db": [float(x) for x in self.null_depth_demand],
            "exclusion_mask": [bool(x) for x in self.exclusion_mask],
            "trust_weights": [float(x) for x in self.trust_weights]
        }


# =========================================================================
# DETERMINISTIC SELF-TEST BLOCK
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Spatial AoA Enforcement Bridge")
    print("==================================================================")
    
    bridge = AoAEnforcementBridge(
        num_channels=4,
        element_spacing=0.127,
        lambda_val=0.254,
        default_fisher_margin_deg=5.0
    )
    
    # Define Satellites true target angles (Azimuth, Elevation in degrees)
    true_angles = np.array([
        [45.0, 30.0],
        [90.0, 60.0],
        [180.0, 45.0],
        [270.0, 15.0]
    ], dtype=np.float64)
    
    # 1. Clean-sky case: measured angles align perfectly with orbital path
    clean_phase_diffs = np.zeros((4, 3), dtype=np.float64)
    for m in range(4):
        az = math.radians(true_angles[m, 0])
        el = math.radians(true_angles[m, 1])
        ux = math.cos(el) * math.cos(az)
        uy = math.cos(el) * math.sin(az)
        clean_phase_diffs[m, 0] = ux * math.pi
        clean_phase_diffs[m, 1] = uy * math.pi
        clean_phase_diffs[m, 2] = (ux + uy) * math.pi

    # 2. Single-emitter jammer case: Ch 1 phase pulled off by 15 degrees
    jammer_phase_diffs = clean_phase_diffs.copy()
    az_j = math.radians(90.0 + 15.0)
    el_j = math.radians(60.0)
    ux_j = math.cos(el_j) * math.cos(az_j)
    uy_j = math.cos(el_j) * math.sin(az_j)
    jammer_phase_diffs[1, 0] = ux_j * math.pi
    jammer_phase_diffs[1, 1] = uy_j * math.pi
    jammer_phase_diffs[1, 2] = (ux_j + uy_j) * math.pi

    # 3. Spoof-plus-jammer scenario: Ch 1 pulled by 15 degrees, Ch 2 by 20 degrees
    spoof_jammer_phase_diffs = jammer_phase_diffs.copy()
    az_s = math.radians(180.0 + 20.0)
    el_s = math.radians(45.0 - 4.0)
    ux_s = math.cos(el_s) * math.cos(az_s)
    uy_s = math.cos(el_s) * math.sin(az_s)
    spoof_jammer_phase_diffs[2, 0] = ux_s * math.pi
    spoof_jammer_phase_diffs[2, 1] = uy_s * math.pi
    spoof_jammer_phase_diffs[2, 2] = (ux_s + uy_s) * math.pi

    # Allocate weights buffer
    weights = np.ones((4, 4), dtype=np.complex64) * 0.5

    # Run clean-sky check
    print("\n[*] Scenario 1: Clean Sky Simulation")
    conf, db_dem, mask, trust, coeffs = bridge.execute_enforcement_stride(clean_phase_diffs, true_angles, weights)
    print(f"    -> Exclusion Mask: {mask}")
    print(f"    -> Trust Weights:   {trust}")
    assert np.all(mask == False), "Clean sky misclassified!"

    # Run single-emitter jammer check
    print("\n[*] Scenario 2: Single-Emitter Jammer on Ch 1")
    conf, db_dem, mask, trust, coeffs = bridge.execute_enforcement_stride(jammer_phase_diffs, true_angles, weights)
    print(f"    -> Exclusion Mask: {mask}")
    print(f"    -> Trust Weights:   {trust}")
    print(f"    -> Null depth Ch 1: {db_dem[1]:.2f} dB")
    print(f"    -> Threat conf Ch 1: {conf[1]*100.0:.2f}%")
    assert mask[1] == True, "Jammer wavefront on Ch 1 missed!"
    assert trust[1] < 0.5, "Trust mapping failure on Ch 1!"

    # Run spoof-plus-jammer check
    print("\n[*] Scenario 3: Spoof + Jammer on Ch 1 & Ch 2")
    conf, db_dem, mask, trust, coeffs = bridge.execute_enforcement_stride(spoof_jammer_phase_diffs, true_angles, weights)
    print(f"    -> Exclusion Mask: {mask}")
    print(f"    -> Trust Weights:   {trust}")
    assert mask[1] == True and mask[2] == True, "Multi-emitter threat validation failed!"

    # Dynamic revalidation/escalation under GLRT anomaly
    print("\n[*] Scenario 4: GLRT Anomaly Escalation Test")
    print(f"    -> Default Fisher Margin: {math.degrees(bridge.active_fisher_margin_rad):.2f}°")
    bridge.interoperate_with_glrt(None, covariance_anomaly=True)
    print(f"    -> Escalated Fisher Margin: {math.degrees(bridge.active_fisher_margin_rad):.2f}°")
    assert bridge.active_fisher_margin_rad < bridge.default_fisher_margin_rad, "Escalation failed to narrow window!"

    # Benchmark test
    print("\n--- ENFORCEMENT BRIDGE BENCHMARK ---")
    print("[*] Running 15,000 warm continuous loops...")
    
    latencies = []
    for _ in range(15000):
        t0 = time.perf_counter()
        _ = bridge.execute_enforcement_stride(spoof_jammer_phase_diffs, true_angles, weights)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)
        
    avg_us = np.median(latencies)
    print(f"  Median Execution Latency: {avg_us:.3f} µs")
    
    if avg_us < 2.0:
        print("[PASSED] AoA Enforcement Bridge operates under the 2µs update limit.")
    else:
        print("[FAILED] Enforcement Bridge execution latency exceeded limit.")
