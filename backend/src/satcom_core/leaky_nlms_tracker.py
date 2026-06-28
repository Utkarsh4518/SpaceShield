"""
Task 65.1: Leaky NLMS Phased-Array Null Tracker Module
SpaceShield High-Velocity Receiver DSP Subsystem

Extends NLMS with a leaky weight update rule: v = alpha * w - mu * y^* * x.
Prevents weight accumulation and numerical drift during multi-hour runs.
"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _adaptive_null_track_leaky_nlms_jit(
    weights: np.ndarray,      # (4,) complex64 (input/output)
    target_a: np.ndarray,     # (4,) complex64 (target steering vector)
    x_history: np.ndarray,    # (N, 4) complex64 (received signal vectors)
    mu0: float,
    epsilon: float,
    alpha: float,             # Leakage factor (e.g. 0.9995)
    N_steps: int,
    out_powers: np.ndarray    # (N,) float64 (output)
):
    """
    Frost's Constrained Leaky NLMS algorithm update loop.
    v = alpha * w - mu * y_conj * x
    w = P * v + f
    """
    for n in range(N_steps):
        x0 = x_history[n, 0]
        x1 = x_history[n, 1]
        x2 = x_history[n, 2]
        x3 = x_history[n, 3]
        
        w0_c = weights[0].real - 1j * weights[0].imag
        w1_c = weights[1].real - 1j * weights[1].imag
        w2_c = weights[2].real - 1j * weights[2].imag
        w3_c = weights[3].real - 1j * weights[3].imag
        
        y = w0_c * x0 + w1_c * x1 + w2_c * x2 + w3_c * x3
        out_powers[n] = y.real*y.real + y.imag*y.imag
        
        x_power = (
            x0.real*x0.real + x0.imag*x0.imag +
            x1.real*x1.real + x1.imag*x1.imag +
            x2.real*x2.real + x2.imag*x2.imag +
            x3.real*x3.real + x3.imag*x3.imag
        )
        
        mu = mu0 / (x_power + epsilon)
        y_conj = y.real - 1j * y.imag
        
        # Leaky update
        v0 = alpha * weights[0] - mu * y_conj * x0
        v1 = alpha * weights[1] - mu * y_conj * x1
        v2 = alpha * weights[2] - mu * y_conj * x2
        v3 = alpha * weights[3] - mu * y_conj * x3
        
        a0_c = target_a[0].real - 1j * target_a[0].imag
        a1_c = target_a[1].real - 1j * target_a[1].imag
        a2_c = target_a[2].real - 1j * target_a[2].imag
        a3_c = target_a[3].real - 1j * target_a[3].imag
        
        inner = a0_c * v0 + a1_c * v1 + a2_c * v2 + a3_c * v3
        
        weights[0] = v0 - 0.25 * inner * target_a[0] + 0.25 * target_a[0]
        weights[1] = v1 - 0.25 * inner * target_a[1] + 0.25 * target_a[1]
        weights[2] = v2 - 0.25 * inner * target_a[2] + 0.25 * target_a[2]
        weights[3] = v3 - 0.25 * inner * target_a[3] + 0.25 * target_a[3]


class LeakyNLMSTracker:
    """
    Handles Leaky Normalized LMS adaptive null steering operations.
    Bounds weight growth to prevent numerical drift in long runs.
    """
    def __init__(self, mu0: float = 0.05, epsilon: float = 1e-4, alpha: float = 0.9995):
        self.mu0 = mu0
        self.epsilon = epsilon
        self.alpha = alpha
        
    def track_null(
        self,
        initial_weights: np.ndarray,  # (4,) complex64
        target_steering: np.ndarray,  # (4,) complex64
        received_history: np.ndarray  # (N, 4) complex64
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs JIT Leaky NLMS algorithm update loop.
        Returns (weights, output_power_history).
        """
        w_out = initial_weights.copy().astype(np.complex64)
        out_powers = np.zeros(received_history.shape[0], dtype=np.float64)
        _adaptive_null_track_leaky_nlms_jit(
            w_out,
            target_steering.astype(np.complex64),
            received_history.astype(np.complex64),
            self.mu0,
            self.epsilon,
            self.alpha,
            received_history.shape[0],
            out_powers
        )
        return w_out, out_powers


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Leaky NLMS Tracker Validation")
    print("==================================================================")
    
    tracker = LeakyNLMSTracker(mu0=0.08, epsilon=1e-3, alpha=0.999)
    
    target_a = np.ones(4, dtype=np.complex64)
    w_init = np.ones(4, dtype=np.complex64) * 0.25
    
    # Helper to generate signal wave history with variable interference amplitude
    def generate_signal_data(amplitude: float, spike: bool = False, steps: int = 1500) -> np.ndarray:
        history = np.zeros((steps, 4), dtype=np.complex64)
        rad = math.radians(45.0)
        s0 = 1.0 + 0.0j
        s1 = math.cos(rad) + 1j * math.sin(rad)
        s2 = math.cos(rad * 1.5) + 1j * math.sin(rad * 1.5)
        s3 = math.cos(rad * 2.0) + 1j * math.sin(rad * 2.0)
        
        for n in range(steps):
            current_amp = amplitude
            if spike and 500 <= n <= 550:
                # 50-step high-intensity interference spike
                current_amp = amplitude * 50.0
                
            i_amp = current_amp * (math.cos(10.0 * n * 0.001) + 1j * math.sin(10.0 * n * 0.001))
            history[n, 0] = i_amp * s0 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            history[n, 1] = i_amp * s1 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            history[n, 2] = i_amp * s2 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            history[n, 3] = i_amp * s3 + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
            
        return history

    # Scenario 1: Sustained Jamming Convergence
    print("[*] Scenario 1: Sustained Jamming Tracking...")
    x_sust = generate_signal_data(3.0)
    w_sust, p_sust = tracker.track_null(w_init, target_a, x_sust)
    
    p_init = p_sust[0]
    p_final = np.mean(p_sust[-100:])
    improvement = 10.0 * math.log10(max(p_init, 1e-12) / max(p_final, 1e-12))
    print(f"    -> Suppression Improvement:     {improvement:.2f} dB")
    assert improvement > 25.0, "Leaky NLMS failed to converge on sustained jamming!"
    print("    -> Sustained jamming check: [PASSED]")
    
    # Scenario 2: Transient Spike & Post-Spike Recovery
    print("\n[*] Scenario 2: Transient Spike & Recovery...")
    x_spike = generate_signal_data(2.0, spike=True)
    w_spike, p_spike = tracker.track_null(w_init, target_a, x_spike)
    
    # Compare power during spike (steps 500-550) and after spike (steps 700-800)
    p_spike_zone = np.mean(p_spike[500:550])
    p_recovery_zone = np.mean(p_spike[700:800])
    recovery_ratio = 10.0 * math.log10(max(p_spike_zone, 1e-12) / max(p_recovery_zone, 1e-12))
    print(f"    -> Power during Spike:          {p_spike_zone:.4f}")
    print(f"    -> Power after Recovery:        {p_recovery_zone:.4f}")
    print(f"    -> Post-Spike Recovery Depth:   {recovery_ratio:.2f} dB")
    assert recovery_ratio > 30.0, "Leaky NLMS failed to recover after a power spike!"
    print("    -> Transient spike recovery: [PASSED]")

    # Scenario 3: Long-Run Drift Accumulation (checking weight magnitude bounds)
    print("\n[*] Scenario 3: Long-Run Weight Stability check...")
    # Run a long 5000-step loop
    x_long = generate_signal_data(2.0, steps=5000)
    w_long, _ = tracker.track_null(w_init, target_a, x_long)
    weight_norm = np.linalg.norm(w_long)
    print(f"    -> Long-run final weight norm:  {weight_norm:.4f}")
    # Normal weight norm should be stable around 0.5 to 1.5, never expanding indefinitely
    assert 0.1 < weight_norm < 2.0, "Weights accumulated drift or overflowed!"
    print("    -> Long-run weight stability: [PASSED]")

    print("\n[+] Leaky NLMS tracker validation complete.")
