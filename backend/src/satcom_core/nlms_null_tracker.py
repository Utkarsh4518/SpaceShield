"""
Task 64.2: Normalized LMS Phased-Array Null Tracker Module
SpaceShield High-Velocity Receiver DSP Subsystem

Replaces standard LMS with an NLMS update loop to stabilize convergence
and null maintenance across high power swings and fading bursts.
"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _adaptive_null_track_nlms_jit(
    weights: np.ndarray,      # (4,) complex64 (input/output)
    target_a: np.ndarray,     # (4,) complex64 (target steering vector)
    x_history: np.ndarray,    # (N, 4) complex64 (received signal vectors)
    mu0: float,
    epsilon: float,
    N_steps: int,
    out_powers: np.ndarray    # (N,) float64 (output)
):
    """
    Frost's Linearly Constrained Normalized LMS (NLMS) Algorithm.
    Computes time-varying step sizes: mu[n] = mu0 / (||x[n]||^2 + epsilon).
    Preserves target gain constraint w^H * target_a = 1.0.
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
        out_powers[n] = y.real*y.real + y.imag*y.imag
        
        # 3. Compute vector power ||x||^2
        x_power = (
            x0.real*x0.real + x0.imag*x0.imag +
            x1.real*x1.real + x1.imag*x1.imag +
            x2.real*x2.real + x2.imag*x2.imag +
            x3.real*x3.real + x3.imag*x3.imag
        )
        
        # 4. Apply step-size normalization
        mu = mu0 / (x_power + epsilon)
        
        y_conj = y.real - 1j * y.imag
        
        # 5. Adaptive step: v = w - mu * y^* * x
        v0 = weights[0] - mu * y_conj * x0
        v1 = weights[1] - mu * y_conj * x1
        v2 = weights[2] - mu * y_conj * x2
        v3 = weights[3] - mu * y_conj * x3
        
        # 6. Subspace projection: w = P * v + f
        a0_c = target_a[0].real - 1j * target_a[0].imag
        a1_c = target_a[1].real - 1j * target_a[1].imag
        a2_c = target_a[2].real - 1j * target_a[2].imag
        a3_c = target_a[3].real - 1j * target_a[3].imag
        
        inner = a0_c * v0 + a1_c * v1 + a2_c * v2 + a3_c * v3
        
        weights[0] = v0 - 0.25 * inner * target_a[0] + 0.25 * target_a[0]
        weights[1] = v1 - 0.25 * inner * target_a[1] + 0.25 * target_a[1]
        weights[2] = v2 - 0.25 * inner * target_a[2] + 0.25 * target_a[2]
        weights[3] = v3 - 0.25 * inner * target_a[3] + 0.25 * target_a[3]


class NLMSNullTracker:
    """
    Handles Normalized LMS adaptive null steering operations.
    Maintains spatial filtering convergence speed across wide SNR swings.
    """
    def __init__(self, mu0: float = 0.05, epsilon: float = 1e-4):
        self.mu0 = mu0
        self.epsilon = epsilon
        
    def track_null(
        self,
        initial_weights: np.ndarray,  # (4,) complex64
        target_steering: np.ndarray,  # (4,) complex64
        received_history: np.ndarray  # (N, 4) complex64
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs JIT NLMS algorithm update loop.
        Returns (weights, output_power_history).
        """
        w_out = initial_weights.copy().astype(np.complex64)
        out_powers = np.zeros(received_history.shape[0], dtype=np.float64)
        _adaptive_null_track_nlms_jit(
            w_out,
            target_steering.astype(np.complex64),
            received_history.astype(np.complex64),
            self.mu0,
            self.epsilon,
            received_history.shape[0],
            out_powers
        )
        return w_out, out_powers


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: NLMS Null Tracker Validation")
    print("==================================================================")
    
    tracker = NLMSNullTracker(mu0=0.08, epsilon=1e-3)
    
    target_a = np.ones(4, dtype=np.complex64)
    w_init = np.ones(4, dtype=np.complex64) * 0.25
    
    # Helper to generate signal wave history with variable interference amplitude
    def generate_signal_data(amplitude: float, fading: bool = False, steps: int = 1500) -> np.ndarray:
        history = np.zeros((steps, 4), dtype=np.complex64)
        rad = math.radians(45.0)
        s0 = 1.0 + 0.0j
        s1 = math.cos(rad) + 1j * math.sin(rad)
        s2 = math.cos(rad * 1.5) + 1j * math.sin(rad * 1.5)
        s3 = math.cos(rad * 2.0) + 1j * math.sin(rad * 2.0)
        
        for n in range(steps):
            current_amp = amplitude
            if fading:
                # Alternating multi-path fade sweep
                current_amp = amplitude * (0.5 + 0.5 * math.sin(2.0 * math.pi * 5.0 * n * 0.001))
                
            i_amp = current_amp * (math.cos(10.0 * n * 0.001) + 1j * math.sin(10.0 * n * 0.001))
            history[n, 0] = i_amp * s0 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            history[n, 1] = i_amp * s1 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            history[n, 2] = i_amp * s2 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            history[n, 3] = i_amp * s3 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            
        return history

    # Scenario 1: Low SNR / Weak Interference (amplitude = 0.2)
    print("[*] Scenario 1: Low SNR Tracking (Weak Jammer)...")
    x_low = generate_signal_data(0.2)
    w_low, p_low = tracker.track_null(w_init, target_a, x_low)
    
    p_init = p_low[0]
    p_final = np.mean(p_low[-100:])
    improvement = 10.0 * math.log10(max(p_init, 1e-12) / max(p_final, 1e-12))
    print(f"    -> Suppression Improvement:     {improvement:.2f} dB")
    assert improvement > 20.0, "NLMS failed to converge under weak signals!"
    print("    -> Low SNR convergence: [PASSED]")
    
    # Scenario 2: High SNR / Strong Jamming (amplitude = 25.0)
    # A standard LMS tracker with fixed step size would blow up under 25.0 amplitude
    print("\n[*] Scenario 2: High SNR / Power Jamming Surge (Stable Check)...")
    x_high = generate_signal_data(25.0)
    w_high, p_high = tracker.track_null(w_init, target_a, x_high)
    
    p_init_high = p_high[0]
    p_final_high = np.mean(p_high[-100:])
    improvement_high = 10.0 * math.log10(max(p_init_high, 1e-12) / max(p_final_high, 1e-12))
    print(f"    -> Suppression Improvement:     {improvement_high:.2f} dB")
    assert improvement_high > 35.0, "NLMS diverged or failed to suppress strong jamming!"
    print("    -> High SNR tracking: [PASSED]")

    # Scenario 3: Rapid Fading Channel
    print("\n[*] Scenario 3: Fading Channel Tracking...")
    x_fade = generate_signal_data(5.0, fading=True)
    w_fade, p_fade = tracker.track_null(w_init, target_a, x_fade)
    
    p_init_fade = p_fade[0]
    p_final_fade = np.mean(p_fade[-100:])
    improvement_fade = 10.0 * math.log10(max(p_init_fade, 1e-12) / max(p_final_fade, 1e-12))
    print(f"    -> Suppression Improvement:     {improvement_fade:.2f} dB")
    assert improvement_fade > 25.0, "NLMS failed under rapid fading variations!"
    print("    -> Fading convergence: [PASSED]")

    print("\n[+] NLMS null tracker validation complete.")
