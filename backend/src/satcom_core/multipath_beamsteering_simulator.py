"""
Task 62.1: Phased-Array Multipath Beamsteering Simulator Module
SpaceShield High-Velocity Receiver DSP Subsystem

Simulates time-varying multipath fading, Doppler phase slips, and delay spreads
to evaluate beamforming steering weights stability under ionospheric distortions.
"""

import json
import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _simulate_multipath_stride_jit(
    weights: np.ndarray,      # (4,) complex64
    path_u_cos: np.ndarray,   # (K, 2) float64
    path_atten: np.ndarray,   # (K,) float64
    path_phase: np.ndarray,   # (K,) float64
    path_doppler: np.ndarray, # (K,) float64
    t_steps: np.ndarray,      # (N_steps,) float64
    d: float,
    lambda_val: float,
    out_gains: np.ndarray     # (N_steps,) float64 (output)
):
    """
    Zero-Heap Numba JIT simulation kernel:
    Accumulates time-varying composite multipath wavefront vectors and computes
    instantaneous power gain response.
    """
    num_paths = path_atten.shape[0]
    num_steps = t_steps.shape[0]
    k_d = 2.0 * math.pi * d / lambda_val
    
    # Pre-conjugate weights for standard beamforming (w^H * h)
    w0_c = weights[0].real - 1j * weights[0].imag
    w1_c = weights[1].real - 1j * weights[1].imag
    w2_c = weights[2].real - 1j * weights[2].imag
    w3_c = weights[3].real - 1j * weights[3].imag
    
    for s in range(num_steps):
        t = t_steps[s]
        
        # Accumulate composite wavefront across the 4 elements
        h0 = 0.0 + 0.0j
        h1 = 0.0 + 0.0j
        h2 = 0.0 + 0.0j
        h3 = 0.0 + 0.0j
        
        for p in range(num_paths):
            ux = path_u_cos[p, 0]
            uy = path_u_cos[p, 1]
            a = path_atten[p]
            theta = path_phase[p] + 2.0 * math.pi * path_doppler[p] * t
            
            # Complex amplitude of path
            c_amp = a * (math.cos(theta) + 1j * math.sin(theta))
            
            # Phase differences relative to Antenna 0
            phi1 = ux * k_d
            phi2 = uy * k_d
            phi3 = phi1 + phi2
            
            # Phase steering vector components
            a0 = 1.0 + 0.0j
            a1 = math.cos(phi1) + 1j * math.sin(phi1)
            a2 = math.cos(phi2) + 1j * math.sin(phi2)
            a3 = math.cos(phi3) + 1j * math.sin(phi3)
            
            # Accumulate path contribution
            h0 += c_amp * a0
            h1 += c_amp * a1
            h2 += c_amp * a2
            h3 += c_amp * a3
            
        # Beamformed output: y = w^H * h
        resp = w0_c * h0 + w1_c * h1 + w2_c * h2 + w3_c * h3
        out_gains[s] = resp.real*resp.real + resp.imag*resp.imag


class MultipathBeamsteeringSimulator:
    """
    Models time-varying reflected paths, attenation, phase offsets, and Doppler
    shifts to evaluate beamsteering null integrity.
    """
    def __init__(
        self,
        element_spacing: float = 0.127,
        lambda_val: float = 0.254,
        max_paths: int = 5
    ):
        self.d = element_spacing
        self.lambda_val = lambda_val
        self.max_paths = max_paths
        
        # Pre-allocated arrays for JIT safety
        self.path_u_cos = np.zeros((self.max_paths, 2), dtype=np.float64)
        self.path_atten = np.zeros(self.max_paths, dtype=np.float64)
        self.path_phase = np.zeros(self.max_paths, dtype=np.float64)
        self.path_doppler = np.zeros(self.max_paths, dtype=np.float64)
        
        # Pre-warm compilation
        self._warmup()

    def _warmup(self):
        w = np.ones(4, dtype=np.complex64) * 0.25
        t = np.zeros(1, dtype=np.float64)
        g = np.zeros(1, dtype=np.float64)
        _simulate_multipath_stride_jit(
            w, self.path_u_cos[:2], self.path_atten[:2],
            self.path_phase[:2], self.path_doppler[:2],
            t, self.d, self.lambda_val, g
        )

    def run_simulation(
        self,
        weights: np.ndarray,      # (4,) complex64
        target_az_el: tuple[float, float],
        reflections: list[dict],  # list of reflections parameters
        num_steps: int = 100,
        time_interval: float = 0.001
    ) -> np.ndarray:
        """
        Executes JIT multipath simulation.
        Returns time-series power gain metrics.
        """
        # Set Direct LOS Path at index 0
        az_rad = math.radians(target_az_el[0])
        el_rad = math.radians(target_az_el[1])
        cos_el = math.cos(el_rad)
        self.path_u_cos[0, 0] = cos_el * math.cos(az_rad)
        self.path_u_cos[0, 1] = cos_el * math.sin(az_rad)
        self.path_atten[0] = 1.0  # Normalized Direct LOS amplitude
        self.path_phase[0] = 0.0
        self.path_doppler[0] = 0.0
        
        # Set reflected components
        num_reflections = len(reflections)
        num_paths = min(1 + num_reflections, self.max_paths)
        
        for idx in range(num_paths - 1):
            ref = reflections[idx]
            r_az_rad = math.radians(ref["az"])
            r_el_rad = math.radians(ref["el"])
            cos_r_el = math.cos(r_el_rad)
            
            self.path_u_cos[idx + 1, 0] = cos_r_el * math.cos(r_az_rad)
            self.path_u_cos[idx + 1, 1] = cos_r_el * math.sin(r_az_rad)
            self.path_atten[idx + 1] = ref["attenuation"]
            self.path_phase[idx + 1] = ref.get("phase", 0.0)
            self.path_doppler[idx + 1] = ref.get("doppler", 0.0)
            
        t_steps = np.arange(num_steps, dtype=np.float64) * time_interval
        out_gains = np.zeros(num_steps, dtype=np.float64)
        
        # Run JIT Jitter loop
        _simulate_multipath_stride_jit(
            weights.astype(np.complex64),
            self.path_u_cos[:num_paths],
            self.path_atten[:num_paths],
            self.path_phase[:num_paths],
            self.path_doppler[:num_paths],
            t_steps,
            self.d,
            self.lambda_val,
            out_gains
        )
        return out_gains


# =========================================================================
# DETERMINISTIC SIMULATION HARNESS & REPORTING
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Phased-Array Multipath Simulator")
    print("==================================================================")
    
    sim = MultipathBeamsteeringSimulator(max_paths=5)
    
    # Target satellite pointing coordinates
    target_coords = (45.0, 30.0)
    
    # Generate weights pointing to target
    az_rad = math.radians(target_coords[0])
    el_rad = math.radians(target_coords[1])
    cos_el = math.cos(el_rad)
    ux = cos_el * math.cos(az_rad)
    uy = cos_el * math.sin(az_rad)
    k_d = 2.0 * math.pi * sim.d / sim.lambda_val
    phi1 = ux * k_d
    phi2 = uy * k_d
    phi3 = phi1 + phi2
    
    w_target = np.array([
        0.25,
        0.25 * (math.cos(phi1) + 1j * math.sin(phi1)),
        0.25 * (math.cos(phi2) + 1j * math.sin(phi2)),
        0.25 * (math.cos(phi3) + 1j * math.sin(phi3))
    ], dtype=np.complex64)
    
    report_data = {}
    
    # 1. Clean Sky Test
    print("[*] Case 1: Clean Sky Simulation...")
    gains_clean = sim.run_simulation(w_target, target_coords, [], num_steps=50)
    gains_clean_db = 10.0 * np.log10(np.maximum(gains_clean, 1e-12))
    
    clean_mean = np.mean(gains_clean_db)
    clean_std = np.std(gains_clean_db)
    print(f"    -> Mean Gain: {clean_mean:.2f} dB | Std: {clean_std:.4f} dB")
    assert abs(clean_mean) < 0.1 and clean_std < 1e-6, "Clean sky response should be static 0 dB!"
    
    report_data["clean_sky"] = {"mean_gain_db": clean_mean, "std_gain_db": clean_std}
    
    # 2. Mild Multipath Test
    print("\n[*] Case 2: Mild Multipath Simulation (2 weak reflections)...")
    reflections_mild = [
        {"az": 48.0, "el": 32.0, "attenuation": 0.1, "doppler": 2.0},
        {"az": 42.0, "el": 28.0, "attenuation": 0.05, "doppler": 5.0}
    ]
    gains_mild = sim.run_simulation(w_target, target_coords, reflections_mild, num_steps=50)
    gains_mild_db = 10.0 * np.log10(np.maximum(gains_mild, 1e-12))
    
    mild_mean = np.mean(gains_mild_db)
    mild_std = np.std(gains_mild_db)
    print(f"    -> Mean Gain: {mild_mean:.2f} dB | Std: {mild_std:.4f} dB")
    assert mild_std > 0.0, "Mild fading should display non-zero variance!"
    
    report_data["mild_multipath"] = {"mean_gain_db": mild_mean, "std_gain_db": mild_std}
    
    # 3. Severe Multipath Test
    print("\n[*] Case 3: Severe Multipath Simulation (3 strong reflections)...")
    reflections_severe = [
        {"az": 55.0, "el": 38.0, "attenuation": 0.4, "doppler": 10.0},
        {"az": 35.0, "el": 22.0, "attenuation": 0.3, "doppler": 25.0},
        {"az": 45.0, "el": 15.0, "attenuation": 0.2, "doppler": 50.0}
    ]
    gains_severe = sim.run_simulation(w_target, target_coords, reflections_severe, num_steps=50)
    gains_severe_db = 10.0 * np.log10(np.maximum(gains_severe, 1e-12))
    
    severe_mean = np.mean(gains_severe_db)
    severe_std = np.std(gains_severe_db)
    print(f"    -> Mean Gain: {severe_mean:.2f} dB | Std: {severe_std:.4f} dB")
    assert severe_std > mild_std, "Severe fading variance should exceed mild fading variance!"
    
    report_data["severe_multipath"] = {"mean_gain_db": severe_mean, "std_gain_db": severe_std}
    
    # Export metrics report for off-line analysis
    output_path = "backend/src/satcom_core/multipath_simulation_report.json"
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)
        
    print(f"\n[+] Multipath simulation data saved successfully to {output_path}")
    print("[+] Phased-array multipath beamsteering simulator validation complete.")
