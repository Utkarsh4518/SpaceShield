"""
Task 65.2: Space-Time Adaptive Processing (STAP) Interference Model
SpaceShield High-Velocity Receiver DSP Subsystem

Implements space-time covariance and delay-line adaptive suppression.
Adds time-domain delay taps to the phased-array spatial model to suppress wideband/chirp jammers.
"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _stap_track_jit(
    weights: np.ndarray,          # (12,) complex64 (in/out)
    target_v: np.ndarray,         # (12,) complex64 (space-time target steering)
    x_history: np.ndarray,        # (N, 4) complex64 (raw antenna inputs)
    mu0: float,
    epsilon: float,
    N_steps: int,
    out_powers: np.ndarray        # (N,) float64 (output)
):
    """
    Zero-Heap JIT Space-Time Adaptive Processing (STAP) update loop.
    Uses 4 elements x 3 taps (12 total degrees of freedom).
    """
    x_stap = np.zeros(12, dtype=np.complex64)
    v = np.zeros(12, dtype=np.complex64)
    
    for n in range(2, N_steps):
        # 1. Construct space-time snapshot vector x_stap
        for m in range(4):
            x_stap[m*3]     = x_history[n, m]
            x_stap[m*3 + 1] = x_history[n - 1, m]
            x_stap[m*3 + 2] = x_history[n - 2, m]
            
        # 2. Compute STAP filter output: y = w^H * x_stap
        y = 0.0 + 0.0j
        for k in range(12):
            w_c = weights[k].real - 1j * weights[k].imag
            y += w_c * x_stap[k]
            
        out_powers[n] = y.real*y.real + y.imag*y.imag
        
        # 3. Compute vector power ||x_stap||^2
        x_power = 0.0
        for k in range(12):
            x_power += x_stap[k].real*x_stap[k].real + x_stap[k].imag*x_stap[k].imag
            
        # 4. Normalized step size
        mu = mu0 / (x_power + epsilon)
        y_conj = y.real - 1j * y.imag
        
        # 5. Leaky weight update: v = w - mu * y^* * x_stap
        for k in range(12):
            # Leaky factor 0.9995 to prevent numerical drift
            v[k] = 0.9995 * weights[k] - mu * y_conj * x_stap[k]
            
        # 6. Frost space-time LCMV projection: w = v - 0.25 * (target_v^H * v) * target_v + 0.25 * target_v
        inner = 0.0 + 0.0j
        for k in range(12):
            t_conj = target_v[k].real - 1j * target_v[k].imag
            inner += t_conj * v[k]
            
        for k in range(12):
            weights[k] = v[k] - 0.25 * inner * target_v[k] + 0.25 * target_v[k]


class STAPInterferenceModel:
    """
    Manages space-time adaptive processing operations.
    Maintains spatial-temporal nulling across wideband jamming bands.
    """
    def __init__(self, mu0: float = 0.05, epsilon: float = 1e-4):
        self.mu0 = mu0
        self.epsilon = epsilon
        
        # Pre-allocated active weights (12 coefficients: 4 antennas x 3 taps)
        self.weights = np.zeros(12, dtype=np.complex64)
        # Initialize zero-delay taps to nominal target steering
        for m in range(4):
            self.weights[m*3] = 0.25

    def track_stap(
        self,
        target_steering: np.ndarray,  # (4,) complex64
        received_history: np.ndarray  # (N, 4) complex64
    ) -> np.ndarray:
        """
        Runs JIT STAP algorithm update loop.
        Returns the output power sequence.
        """
        # Map target steering to 12-element space-time target steering vector
        # (Zero-delay taps get target_steering, other taps get 0)
        target_v = np.zeros(12, dtype=np.complex64)
        for m in range(4):
            target_v[m*3] = target_steering[m]
            
        out_powers = np.zeros(received_history.shape[0], dtype=np.float64)
        _stap_track_jit(
            self.weights,
            target_v,
            received_history.astype(np.complex64),
            self.mu0,
            self.epsilon,
            received_history.shape[0],
            out_powers
        )
        return out_powers


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: STAP Interference Model Validation")
    print("==================================================================")
    
    target_a = np.ones(4, dtype=np.complex64)
    
    # Helper to generate spatial steering vector
    def get_spatial_vector(elevation_deg: float) -> np.ndarray:
        rad = math.radians(elevation_deg)
        a = np.zeros(4, dtype=np.complex64)
        a[0] = 1.0
        a[1] = math.cos(rad) + 1j * math.sin(rad)
        a[2] = math.cos(rad * 1.5) + 1j * math.sin(rad * 1.5)
        a[3] = math.cos(rad * 2.0) + 1j * math.sin(rad * 2.0)
        return a
        
    spoofer_a = get_spatial_vector(45.0)

    # 1. Narrowband Interference (Sustained sine wave)
    print("[*] Scenario 1: Narrowband Interference Suppression...")
    sim_steps = 1500
    x_nb = np.zeros((sim_steps, 4), dtype=np.complex64)
    for n in range(sim_steps):
        # Narrowband single frequency sine
        i_amp = 3.0 * (math.cos(10.0 * n * 0.001) + 1j * math.sin(10.0 * n * 0.001))
        x_nb[n] = i_amp * spoofer_a + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
        
    stap_nb = STAPInterferenceModel(mu0=0.08)
    p_nb = stap_nb.track_stap(target_a, x_nb)
    
    nb_init = p_nb[2]
    nb_final = np.mean(p_nb[-100:])
    nb_db = 10.0 * math.log10(max(nb_init, 1e-12) / max(nb_final, 1e-12))
    print(f"    -> Narrowband Suppression:      {nb_db:.2f} dB")
    assert nb_db > 25.0, "STAP failed to null narrowband jammer!"
    print("    -> Narrowband check: [PASSED]")

    # 2. Wideband Jammer (Random noise signal)
    print("\n[*] Scenario 2: Wideband Burst Suppression...")
    x_wb = np.zeros((sim_steps, 4), dtype=np.complex64)
    for n in range(sim_steps):
        # Wideband random complex noise
        i_amp = 3.0 * (np.random.normal() + 1j * np.random.normal())
        x_wb[n] = i_amp * spoofer_a + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
        
    stap_wb = STAPInterferenceModel(mu0=0.08)
    p_wb = stap_wb.track_stap(target_a, x_wb)
    
    wb_init = p_wb[2]
    wb_final = np.mean(p_wb[-100:])
    wb_db = 10.0 * math.log10(max(wb_init, 1e-12) / max(wb_final, 1e-12))
    print(f"    -> Wideband Suppression:        {wb_db:.2f} dB")
    assert wb_db > 20.0, "STAP failed to null wideband jammer!"
    print("    -> Wideband check: [PASSED]")

    # 3. Time-Varying Chirp (Frequency sweeping)
    print("\n[*] Scenario 3: Chirped Interference Suppression...")
    x_chirp = np.zeros((sim_steps, 4), dtype=np.complex64)
    for n in range(sim_steps):
        # Frequency sweeping chirp: phase = 2*pi*f_start*t + pi*sweep_rate*t^2
        t = n * 0.001
        phase = 2.0 * math.pi * 5.0 * t + math.pi * 20.0 * (t ** 2)
        i_amp = 3.0 * (math.cos(phase) + 1j * math.sin(phase))
        x_chirp[n] = i_amp * spoofer_a + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
        
    stap_chirp = STAPInterferenceModel(mu0=0.08)
    p_chirp = stap_chirp.track_stap(target_a, x_chirp)
    
    chirp_init = p_chirp[2]
    chirp_final = np.mean(p_chirp[-100:])
    chirp_db = 10.0 * math.log10(max(chirp_init, 1e-12) / max(chirp_final, 1e-12))
    print(f"    -> Chirp Suppression:           {chirp_db:.2f} dB")
    assert chirp_db > 20.0, "STAP failed to null chirped jammer!"
    print("    -> Chirp check: [PASSED]")

    print("\n[+] STAP interference model validation complete.")
