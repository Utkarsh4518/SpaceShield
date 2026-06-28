"""
Task 61.2: Beam Null Verification Module
SpaceShield High-Velocity Receiver DSP Subsystem

Evaluates spatial steering response and beamforming weight null placement depth.
Simulates RF antenna array factor performance and computes suppression ratios.
"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _compute_power_gain_jit(
    weights: np.ndarray,      # (4,) complex64
    az: float,
    el: float,
    d: float,
    lambda_val: float
) -> float:
    """Computes the array factor power gain in a specific direction (az, el)."""
    az_rad = math.radians(az)
    el_rad = math.radians(el)
    cos_el = math.cos(el_rad)
    ux = cos_el * math.cos(az_rad)
    uy = cos_el * math.sin(az_rad)
    
    # Phase differences relative to Antenna 0
    k_d = 2.0 * math.pi * d / lambda_val
    phi1 = ux * k_d
    phi2 = uy * k_d
    phi3 = phi1 + phi2
    
    # Steering vector elements (standard form)
    a0 = 1.0 + 0.0j
    a1 = math.cos(phi1) + 1j * math.sin(phi1)
    a2 = math.cos(phi2) + 1j * math.sin(phi2)
    a3 = math.cos(phi3) + 1j * math.sin(phi3)
    
    # Beamformed response dot-product (conjugate weights: w^H * a)
    w0_c = weights[0].real - 1j * weights[0].imag
    w1_c = weights[1].real - 1j * weights[1].imag
    w2_c = weights[2].real - 1j * weights[2].imag
    w3_c = weights[3].real - 1j * weights[3].imag
    
    resp = w0_c*a0 + w1_c*a1 + w2_c*a2 + w3_c*a3
    
    power = resp.real*resp.real + resp.imag*resp.imag
    return power


@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _evaluate_null_performance_jit(
    weights: np.ndarray,      # (4,) complex64
    target_az: float,
    target_el: float,
    threat_az: float,
    threat_el: float,
    d: float,
    lambda_val: float
) -> tuple[float, float, float]:
    """
    Computes main-lobe gain, threat null-depth, and peak sidelobe suppression.
    Avoids dynamic allocation inside the evaluation loop.
    """
    # 1. Main Lobe Gain
    main_gain = _compute_power_gain_jit(weights, target_az, target_el, d, lambda_val)
    main_db = 10.0 * math.log10(max(main_gain, 1e-12))
    
    # 2. Threat Null Depth
    threat_gain = _compute_power_gain_jit(weights, threat_az, threat_el, d, lambda_val)
    null_db = 10.0 * math.log10(max(threat_gain, 1e-12))
    
    # 3. Peak Sidelobe Scan
    peak_sidelobe = 1e-12
    # Scan coarse grid outside the main beam region (defined as > 15 degrees separation)
    for az in range(0, 360, 5):
        for el in range(10, 90, 5):
            # Calculate angular separation
            az_rad = math.radians(az)
            el_rad = math.radians(el)
            t_az_rad = math.radians(target_az)
            t_el_rad = math.radians(target_el)
            
            cos_sep = (math.cos(el_rad)*math.cos(az_rad)*math.cos(t_el_rad)*math.cos(t_az_rad) +
                       math.cos(el_rad)*math.sin(az_rad)*math.cos(t_el_rad)*math.sin(t_az_rad) +
                       math.sin(el_rad)*math.sin(t_el_rad))
            
            sep = math.acos(max(-1.0, min(1.0, cos_sep)))
            if math.degrees(sep) > 15.0:
                gain = _compute_power_gain_jit(weights, float(az), float(el), d, lambda_val)
                if gain > peak_sidelobe:
                    peak_sidelobe = gain
                    
    sidelobe_db = 10.0 * math.log10(peak_sidelobe)
    sidelobe_suppression = main_db - sidelobe_db
    
    return main_db, null_db, sidelobe_suppression


class BeamNullVerification:
    """
    Simulates physical phased-array reception and calculates gain,
    suppression metrics, and SNR curves based on beamforming steering weights.
    """
    def __init__(
        self,
        element_spacing: float = 0.127,
        lambda_val: float = 0.254
    ):
        self.d = element_spacing
        self.lambda_val = lambda_val

    def evaluate_spatial_response(
        self,
        weights: np.ndarray,  # (4,) complex64
        target_az_el: tuple[float, float],
        threat_az_el: tuple[float, float]
    ) -> dict:
        """
        Runs JIT evaluator and returns metrics as a dictionary for the control plane.
        """
        main_db, null_db, sidelobe_suppr = _evaluate_null_performance_jit(
            weights.astype(np.complex64),
            target_az_el[0], target_az_el[1],
            threat_az_el[0], threat_az_el[1],
            self.d, self.lambda_val
        )
        
        # Calculate residual interference ratio
        # Interference before nulling: weights = uniform [0.25, 0.25, 0.25, 0.25]
        uniform_weights = np.ones(4, dtype=np.complex64) * 0.25
        before_null_gain = _compute_power_gain_jit(
            uniform_weights, threat_az_el[0], threat_az_el[1], self.d, self.lambda_val
        )
        after_null_gain = _compute_power_gain_jit(
            weights, threat_az_el[0], threat_az_el[1], self.d, self.lambda_val
        )
        residual_ratio = after_null_gain / max(before_null_gain, 1e-12)

        return {
            "main_lobe_gain_db": main_db,
            "threat_null_depth_db": null_db,
            "sidelobe_suppression_db": sidelobe_suppr,
            "residual_interference_ratio": residual_ratio
        }


# =========================================================================
# DETERMINISTIC SIMULATION HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Beam Null Verification Harness")
    print("==================================================================")
    
    verifier = BeamNullVerification(element_spacing=0.127, lambda_val=0.254)
    
    target = (45.0, 30.0)
    threat = (120.0, 50.0)
    
    # 1. Clean Sky (Uniform pointing weights directed to target)
    print("[*] Case 1: Clean Sky Simulation...")
    # Generate steering weights for target azimuth/elevation
    az_rad = math.radians(target[0])
    el_rad = math.radians(target[1])
    cos_el = math.cos(el_rad)
    ux = cos_el * math.cos(az_rad)
    uy = cos_el * math.sin(az_rad)
    k_d = 2.0 * math.pi * verifier.d / verifier.lambda_val
    phi1 = ux * k_d
    phi2 = uy * k_d
    phi3 = phi1 + phi2
    
    # Steering vector to beamform toward target (gain = 1.0)
    w_target = np.array([
        0.25,
        0.25 * (math.cos(phi1) + 1j * math.sin(phi1)),
        0.25 * (math.cos(phi2) + 1j * math.sin(phi2)),
        0.25 * (math.cos(phi3) + 1j * math.sin(phi3))
    ], dtype=np.complex64)
    
    metrics = verifier.evaluate_spatial_response(w_target, target, threat)
    print(f"    -> Main-Lobe Gain:        {metrics['main_lobe_gain_db']:.2f} dB")
    print(f"    -> Threat Null Gain:      {metrics['threat_null_depth_db']:.2f} dB")
    print(f"    -> Sidelobe Suppression:  {metrics['sidelobe_suppression_db']:.2f} dB")
    
    assert abs(metrics['main_lobe_gain_db']) < 0.1, "Main lobe gain should be 0 dB for perfect beamforming!"
    print("    -> Clean sky checks: [PASSED]")
    
    # 2. Spoof + Jammer Nulling (Orthogonal null projection directed to threat)
    print("\n[*] Case 2: Threat Spatial Nulling Simulation...")
    # Spoofing steering vector
    psi1 = math.radians(threat[0]) * 0.5 # mock phase difference
    psi2 = math.radians(threat[1]) * 0.5
    psi3 = psi1 + psi2
    
    s0 = 1.0 + 0.0j
    s1 = math.cos(psi1) + 1j * math.sin(psi1)
    s2 = math.cos(psi2) + 1j * math.sin(psi2)
    s3 = math.cos(psi3) + 1j * math.sin(psi3)
    
    p = (w_target[0]*s0.conjugate() + w_target[1]*s1.conjugate() + w_target[2]*s2.conjugate() + w_target[3]*s3.conjugate())
    
    # Project orthogonal null (weights - 0.25 * p * s)
    w_nulled = w_target - 0.25 * p * np.array([s0, s1, s2, s3], dtype=np.complex64)
    
    # Determine the actual threat angles that correspond to the phase differences psi1, psi2
    # psi = 2 * pi * d * u / lambda -> u = psi * lambda / (2 * pi * d)
    scaling = verifier.lambda_val / (2.0 * math.pi * verifier.d)
    threat_ux = psi1 * scaling
    threat_uy = psi2 * scaling
    threat_uz = math.sqrt(1.0 - (threat_ux**2 + threat_uy**2))
    
    # Back-project threat azimuth and elevation
    threat_el_rad = math.asin(threat_uz)
    threat_az_rad = math.atan2(threat_uy, threat_ux)
    threat_actual = (math.degrees(threat_az_rad), math.degrees(threat_el_rad))
    
    metrics_nulled = verifier.evaluate_spatial_response(w_nulled, target, threat_actual)
    print(f"    -> Nulled Main-Lobe Gain: {metrics_nulled['main_lobe_gain_db']:.2f} dB")
    print(f"    -> Nulled Threat Gain:    {metrics_nulled['threat_null_depth_db']:.2f} dB")
    print(f"    -> Residual Ratio:        {metrics_nulled['residual_interference_ratio']}")
    
    assert metrics_nulled['threat_null_depth_db'] < -40.0, "Null depth should be below -40 dB!"
    assert metrics_nulled['residual_interference_ratio'] < 1e-4, "Suppression ratio failed!"
    print("    -> Threat spatial nulling checks: [PASSED]")
    
    print("\n[+] Phased-array beam null verification successfully completed.")
