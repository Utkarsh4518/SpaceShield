"""
Task 63.1: Adaptive Phased-Array Null Tracker Module
SpaceShield High-Velocity Receiver DSP Subsystem

Implements Frost's Linearly Constrained LMS adaptive algorithm to track
moving spatial nulls under high-Doppler fading conditions.
"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _adaptive_null_track_jit(
    weights: np.ndarray,      # (4,) complex64 (input/output)
    target_a: np.ndarray,     # (4,) complex64 (target steering vector)
    x_history: np.ndarray,    # (N, 4) complex64 (received signal vectors)
    mu: float,
    N_steps: int,
    out_powers: np.ndarray    # (N,) float64 (output)
):
    """
    Frost's Linearly Constrained LMS Algorithm:
    Minimizes output power while keeping target gain fixed at 1.0.
    Operates in-place on pre-allocated weights array.
    """
    for n in range(N_steps):
        # 1. Ingest received vector x
        x0 = x_history[n, 0]
        x1 = x_history[n, 1]
        x2 = x_history[n, 2]
        x3 = x_history[n, 3]
        
        # 2. Compute beamformer output: y = w^H * x
        w0_c = weights[0].real - 1j * weights[0].imag
        w1_c = weights[1].real - 1j * weights[1].imag
        w2_c = weights[2].real - 1j * weights[2].imag
        w3_c = weights[3].real - 1j * weights[3].imag
        
        y = w0_c * x0 + w1_c * x1 + w2_c * x2 + w3_c * x3
        
        # Record instantaneous output power
        out_powers[n] = y.real*y.real + y.imag*y.imag
        
        y_conj = y.real - 1j * y.imag
        
        # 3. Weight adaptation step: v = w - mu * y^* * x
        v0 = weights[0] - mu * y_conj * x0
        v1 = weights[1] - mu * y_conj * x1
        v2 = weights[2] - mu * y_conj * x2
        v3 = weights[3] - mu * y_conj * x3
        
        # 4. Subspace projection: w = P * v + f
        # P * v = v - 0.25 * target_a * (target_a^H * v)
        a0_c = target_a[0].real - 1j * target_a[0].imag
        a1_c = target_a[1].real - 1j * target_a[1].imag
        a2_c = target_a[2].real - 1j * target_a[2].imag
        a3_c = target_a[3].real - 1j * target_a[3].imag
        
        inner = a0_c * v0 + a1_c * v1 + a2_c * v2 + a3_c * v3
        
        weights[0] = v0 - 0.25 * inner * target_a[0] + 0.25 * target_a[0]
        weights[1] = v1 - 0.25 * inner * target_a[1] + 0.25 * target_a[1]
        weights[2] = v2 - 0.25 * inner * target_a[2] + 0.25 * target_a[2]
        weights[3] = v3 - 0.25 * inner * target_a[3] + 0.25 * target_a[3]


class AdaptiveNullTracker:
    """
    Manages adaptive null tracking iterations.
    Tracks spoofer/jammer look-angle variations.
    """
    def __init__(self, mu: float = 0.01):
        self.mu = mu
        
    def track_null(
        self,
        initial_weights: np.ndarray,  # (4,) complex64
        target_steering: np.ndarray,  # (4,) complex64
        received_history: np.ndarray  # (N, 4) complex64
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs JIT Frost algorithm update loop.
        Returns (weights, output_power_history).
        """
        w_out = initial_weights.copy().astype(np.complex64)
        out_powers = np.zeros(received_history.shape[0], dtype=np.float64)
        _adaptive_null_track_jit(
            w_out,
            target_steering.astype(np.complex64),
            received_history.astype(np.complex64),
            self.mu,
            received_history.shape[0],
            out_powers
        )
        return w_out, out_powers


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Adaptive Null Tracker Harness")
    print("==================================================================")
    
    tracker = AdaptiveNullTracker(mu=0.03)
    
    # 1. Target pointing direction: (0, 0) azimuth/elevation
    # Target steering vector has unit elements
    target_a = np.ones(4, dtype=np.complex64)
    
    # Generate static target pointing weights (W^H * target_a = 1.0)
    w_init = np.ones(4, dtype=np.complex64) * 0.25
    
    # Helper to generate time-varying spoofer signal under Doppler slip
    def generate_doppler_data(doppler_hz: float, num_steps: int = 1500, dt: float = 0.001) -> np.ndarray:
        history = np.zeros((num_steps, 4), dtype=np.complex64)
        
        # Initial spoofer phase differences
        psi1_0 = math.radians(60.0)
        psi2_0 = math.radians(45.0)
        
        for n in range(num_steps):
            # Time-varying phase differences due to Doppler look-angle shift
            offset = 2.0 * math.pi * doppler_hz * n * dt
            psi1 = psi1_0 + offset
            psi2 = psi2_0 + offset
            psi3 = psi1 + psi2
            
            s0 = 1.0 + 0.0j
            s1 = math.cos(psi1) + 1j * math.sin(psi1)
            s2 = math.cos(psi2) + 1j * math.sin(psi2)
            s3 = math.cos(psi3) + 1j * math.sin(psi3)
            
            # Interference source wave amplitude
            i_amp = 2.0 * (math.cos(10.0 * n * dt) + 1j * math.sin(10.0 * n * dt))
            
            # Channel wave vector
            history[n, 0] = i_amp * s0 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            history[n, 1] = i_amp * s1 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            history[n, 2] = i_amp * s2 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            history[n, 3] = i_amp * s3 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            
        return history

    # Scenario 1: Low Doppler (0.1 Hz)
    print("[*] Scenario 1: Low Doppler Adaptive Tracking...")
    x_low = generate_doppler_data(0.1)
    w_low, p_low = tracker.track_null(w_init, target_a, x_low)
    
    p_init = p_low[0]
    p_final = np.mean(p_low[-100:])
    improvement = 10.0 * math.log10(max(p_init, 1e-12) / max(p_final, 1e-12))
    
    print(f"    -> Initial Interference Power: {p_init:.4f}")
    print(f"    -> Adaptive Tracked Power:     {p_final:.4f}")
    print(f"    -> Suppression Improvement:     {improvement:.2f} dB")
    assert improvement > 35.0, "LMS failed to converge and null the interference!"
    print("    -> Low Doppler convergence: [PASSED]")
    
    # Scenario 2: Moderate Doppler (0.5 Hz)
    print("\n[*] Scenario 2: Moderate Doppler Adaptive Tracking...")
    x_mod = generate_doppler_data(0.5)
    w_mod, p_mod = tracker.track_null(w_init, target_a, x_mod)
    p_init_mod = p_mod[0]
    p_final_mod = np.mean(p_mod[-100:])
    improvement_mod = 10.0 * math.log10(max(p_init_mod, 1e-12) / max(p_final_mod, 1e-12))
    print(f"    -> Suppression Improvement:     {improvement_mod:.2f} dB")
    assert improvement_mod > 20.0, "Failed to track moderate Doppler shifts!"
    print("    -> Moderate Doppler checks: [PASSED]")

    # Scenario 3: Severe Doppler (2.0 Hz)
    print("\n[*] Scenario 3: Severe Doppler Adaptive Tracking...")
    x_sev = generate_doppler_data(2.0)
    w_sev, p_sev = tracker.track_null(w_init, target_a, x_sev)
    p_init_sev = p_sev[0]
    p_final_sev = np.mean(p_sev[-100:])
    improvement_sev = 10.0 * math.log10(max(p_init_sev, 1e-12) / max(p_final_sev, 1e-12))
    print(f"    -> Suppression Improvement:     {improvement_sev:.2f} dB")
    assert improvement_sev > 5.0, "Failed to track severe Doppler shifts!"
    print("    -> Severe Doppler checks: [PASSED]")

    print("\n[+] Adaptive null tracker validation successfully completed.")
