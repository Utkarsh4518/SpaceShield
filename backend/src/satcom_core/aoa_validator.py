"""
Task 57.5: Spatial Angle-of-Arrival Cross-Validator Block
SpaceShield High-Velocity Receiver DSP Subsystem

Geometric Angle-of-Arrival (AoA) validation and dynamic spatial nulling controller.
Reconstructs the arrival unit vector from multi-antenna phase difference telemetry,
cross-validates against true orbital ephemeris tracking trajectories, and projects
spatial nulling steering patterns on hijacked channels to protect baseband loops.
"""

import time
import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, boundscheck=False)
def _validate_aoa_and_project_nulls(
    phase_diffs: np.ndarray,          # (num_channels, 3) float64 (relative to Ant 0)
    true_az_el: np.ndarray,           # (num_channels, 2) float64 (true azimuth/elevation in degrees)
    beamforming_weights: np.ndarray,  # (num_channels, 4) complex64 (input/output)
    hijack_flags: np.ndarray,         # (num_channels,) bool_ (output)
    angular_errors: np.ndarray,       # (num_channels,) float64 (output)
    element_spacing: float,           # spacing between antenna elements (meters)
    lambda_val: float,                # carrier signal wavelength (meters)
    fisher_margin_rad: float          # threat detection threshold (radians)
):
    """
    Zero-Heap Numba JIT Kernel:
    1. Reconstructs measured 3D AoA vector from planar array phase-differences.
    2. Projects true satellite coordinates into 3D unit vectors.
    3. Calculates the angular difference (geodesic error) between measured and true paths.
    4. Triggers threat flags and dynamically projects steering weights orthogonal to 
       the spoofing steering vector to execute immediate spatial nulling.
    """
    num_channels = phase_diffs.shape[0]
    scaling = lambda_val / (2.0 * math.pi * element_spacing)
    
    for m in range(num_channels):
        psi1 = phase_diffs[m, 0]
        psi2 = phase_diffs[m, 1]
        psi3 = phase_diffs[m, 2] # Usually psi1 + psi2 under noise-free planar conditions
        
        # 1. Reconstruct 3D measured look direction cosines
        ux = psi1 * scaling
        uy = psi2 * scaling
        
        # Unit vector constraint: ux^2 + uy^2 + uz^2 = 1.0
        sum_sq = ux*ux + uy*uy
        if sum_sq > 1.0:
            # Over-determined or noisy phase wrap: Normalize to unit plane boundary
            norm_fact = math.sqrt(sum_sq)
            ux /= norm_fact
            uy /= norm_fact
            uz = 0.0
        else:
            uz = math.sqrt(1.0 - sum_sq)
            
        # 2. Project true satellite coordinates to 3D unit vector
        true_az_rad = math.radians(true_az_el[m, 0])
        true_el_rad = math.radians(true_az_el[m, 1])
        
        cos_el = math.cos(true_el_rad)
        tx = cos_el * math.cos(true_az_rad)
        ty = cos_el * math.sin(true_az_rad)
        tz = math.sin(true_el_rad)
        
        # 3. Calculate Geodesic Angular Error (dot product of unit vectors)
        dot_prod = ux*tx + uy*ty + uz*tz
        # Clamp bounds to prevent float inaccuracies breaking acos
        if dot_prod > 1.0:
            dot_prod = 1.0
        elif dot_prod < -1.0:
            dot_prod = -1.0
            
        ang_err = math.acos(dot_prod)
        angular_errors[m] = ang_err
        
        # 4. Fisher Margin Threshold Validation
        if ang_err > fisher_margin_rad:
            hijack_flags[m] = True
            
            # Dynamic Orthogonal Projection to place null at spoofing AoA
            # Construct phase steering vector for the spoofing arrival vector direction
            s0 = 1.0 + 0.0j
            s1 = math.cos(psi1) + 1j * math.sin(psi1)
            s2 = math.cos(psi2) + 1j * math.sin(psi2)
            s3 = math.cos(psi3) + 1j * math.sin(psi3)
            
            # Compute dot product: p = s^H * W
            # s^H is conjugate of s
            w0 = beamforming_weights[m, 0]
            w1 = beamforming_weights[m, 1]
            w2 = beamforming_weights[m, 2]
            w3 = beamforming_weights[m, 3]
            
            p = (1.0 * w0 + 
                 (math.cos(psi1) - 1j * math.sin(psi1)) * w1 + 
                 (math.cos(psi2) - 1j * math.sin(psi2)) * w2 + 
                 (math.cos(psi3) - 1j * math.sin(psi3)) * w3)
            
            # Project weights orthogonal to s (W_new = W - 0.25 * p * s)
            beamforming_weights[m, 0] = w0 - 0.25 * p * s0
            beamforming_weights[m, 1] = w1 - 0.25 * p * s1
            beamforming_weights[m, 2] = w2 - 0.25 * p * s2
            beamforming_weights[m, 3] = w3 - 0.25 * p * s3
        else:
            hijack_flags[m] = False


class SpatialAoAValidator:
    """
    SpaceShield Real-Time Spatial Angle-of-Arrival Geometric Validator.
    Protects antenna arrays from spatial hijacking by monitoring incoming wavefront 
    propagation angles and dynamically applying null steering vectors.
    """
    def __init__(
        self,
        num_channels: int = 4,
        element_spacing: float = 0.127,  # meters (typically half-wavelength)
        lambda_val: float = 0.254,       # meters (NavIC L5 / GPS L5 band)
        fisher_margin_deg: float = 5.0
    ):
        self.num_channels = num_channels
        self.element_spacing = element_spacing
        self.lambda_val = lambda_val
        self.fisher_margin_rad = math.radians(fisher_margin_deg)
        
        # Zero-allocation dynamic output matrices
        self.hijack_flags = np.zeros(self.num_channels, dtype=np.bool_)
        self.angular_errors = np.zeros(self.num_channels, dtype=np.float64)
        
        # Pre-warm JIT compiler
        self._warmup()

    def _warmup(self):
        """Forces compilation of Numba kernels using nominal geometries."""
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
            self.fisher_margin_rad
        )
        self.hijack_flags.fill(False)
        self.angular_errors.fill(0.0)

    def validate_spatial_integrity(
        self,
        phase_diffs: np.ndarray,
        true_az_el: np.ndarray,
        beamforming_weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validates phase differences against true coordinates.
        Updates beamforming weights inline if geometric deviations are detected.
        Returns:
            hijack_flags: boolean array flagging hijacked channels.
            angular_errors: geodesic error in radians.
        """
        _validate_aoa_and_project_nulls(
            phase_diffs,
            true_az_el,
            beamforming_weights,
            self.hijack_flags,
            self.angular_errors,
            self.element_spacing,
            self.lambda_val,
            self.fisher_margin_rad
        )
        return self.hijack_flags, self.angular_errors


if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Spatial AoA Cross-Validator")
    print("==================================================================")
    
    validator = SpatialAoAValidator(
        num_channels=4,
        element_spacing=0.127,
        lambda_val=0.254,
        fisher_margin_deg=5.0
    )
    
    # Setup test vectors
    # Let satellite true look angles (az, el in degrees) be:
    # Ch 0: (45.0, 30.0)
    # Ch 1: (90.0, 60.0)
    # Ch 2: (180.0, 45.0)
    # Ch 3: (270.0, 15.0)
    true_angles = np.array([
        [45.0, 30.0],
        [90.0, 60.0],
        [180.0, 45.0],
        [270.0, 15.0]
    ], dtype=np.float64)
    
    # Calculate target nominal phase differences for these directions
    # psi = 2*pi * d/lambda * cos(el) * coordinate
    # element spacing d = lambda/2, so scaling = pi
    nom_phase_diffs = np.zeros((4, 3), dtype=np.float64)
    for m in range(4):
        az_rad = math.radians(true_angles[m, 0])
        el_rad = math.radians(true_angles[m, 1])
        ux = math.cos(el_rad) * math.cos(az_rad)
        uy = math.cos(el_rad) * math.sin(az_rad)
        
        nom_phase_diffs[m, 0] = ux * math.pi
        nom_phase_diffs[m, 1] = uy * math.pi
        nom_phase_diffs[m, 2] = (ux + uy) * math.pi

    # Setup simulated actual phase diffs (introduce spoofing hijack on Ch 1)
    act_phase_diffs = nom_phase_diffs.copy()
    
    # Injected offset on Ch 1 (Spoofer steering wavefront from different angle)
    # Spoofer arriving from (120.0, 20.0)
    sp_az = math.radians(120.0)
    sp_el = math.radians(20.0)
    sp_ux = math.cos(sp_el) * math.cos(sp_az)
    sp_uy = math.cos(sp_el) * math.sin(sp_az)
    act_phase_diffs[1, 0] = sp_ux * math.pi
    act_phase_diffs[1, 1] = sp_uy * math.pi
    act_phase_diffs[1, 2] = (sp_ux + sp_uy) * math.pi

    # Initial uniform beamforming weights [0.5, 0.5, 0.5, 0.5]
    weights = np.ones((4, 4), dtype=np.complex64) * 0.5
    
    print("[*] Running Spatial AoA Cross-Validation...")
    flags, errors = validator.validate_spatial_integrity(act_phase_diffs, true_angles, weights)
    
    for ch in range(4):
        print(f"  Channel {ch}: Err = {math.degrees(errors[ch]):.2f}° | Hijacked = {flags[ch]}")
        print(f"    -> Weights: {weights[ch]}")
        
    assert flags[0] == False, "Channel 0 nominal flagged as hijacked!"
    assert flags[1] == True, "Channel 1 spoofing wave was missed!"
    
    # Check that weights for Channel 1 were indeed updated (to create a null)
    # The null steering check: W * steering_vector should be ~0
    psi1, psi2, psi3 = act_phase_diffs[1, 0], act_phase_diffs[1, 1], act_phase_diffs[1, 2]
    s = np.array([
        1.0 + 0.0j,
        math.cos(psi1) + 1j*math.sin(psi1),
        math.cos(psi2) + 1j*math.sin(psi2),
        math.cos(psi3) + 1j*math.sin(psi3)
    ], dtype=np.complex64)
    dot_val = np.abs(np.dot(s.conj(), weights[1]))
    print(f"  [+] Post-Nulling Steering Dot Product: {dot_val:.4e}")
    assert dot_val < 1e-6, "Spatial nulling projection failed to achieve orthogonality!"
    
    print("\n--- SPATIAL VALIDATOR BENCHMARK ---")
    print("[*] Simulating 10,000 continuous real-time cross-validation strides...")
    
    latencies = []
    for _ in range(10000):
        t0 = time.perf_counter()
        _, _ = validator.validate_spatial_integrity(act_phase_diffs, true_angles, weights)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)
        
    avg_us = np.median(latencies)
    p99_us = np.percentile(latencies, 99.0)
    
    print(f"  Median Execution Latency: {avg_us:.3f} µs")
    print(f"  P99 Execution Latency:    {p99_us:.3f} µs")
    
    if avg_us < 15.0:
        print("[PASSED] Spatial AoA Cross-Validator executes beneath 15µs limit.")
    else:
        print("[FAILED] Spatial AoA Cross-Validator overhead exceeded constraints.")
