"""
Task 67.2: FFT STAP Processor Module
SpaceShield High-Velocity Receiver DSP Subsystem

Transforms the time-domain multi-tap adaptive beamformer into the frequency domain.
Runs independent spatial nulling on each FFT bin to suppress wideband/chirped jammers.
"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _fft_stap_process_block_jit(
    x_block: np.ndarray,          # (16, 4) complex64 input block
    dft_mat: np.ndarray,          # (16, 16) complex64
    idft_mat: np.ndarray,         # (16, 16) complex64
    weights: np.ndarray,          # (16, 4) complex64 (4 weights per bin)
    target_a: np.ndarray,         # (4,) complex64 target steering
    mu: float,
    epsilon: float,
    y_block_out: np.ndarray       # (16,) complex64 output block
):
    """
    Performs block-by-block FFT STAP:
    1. Transforms time domain block to frequency domain for all 4 channels.
    2. Performs separate NLMS updates on each bin to place spatial nulls.
    3. Transforms output bins back to time domain.
    """
    # 1. Transform time block to frequency block (16 bins, 4 channels)
    X_freq = np.zeros((16, 4), dtype=np.complex64)
    for c in range(4):
        for f in range(16):
            val = 0.0 + 0.0j
            for t in range(16):
                val += dft_mat[f, t] * x_block[t, c]
            X_freq[f, c] = val
            
    # 2. Process each frequency bin independently
    Y_freq = np.zeros(16, dtype=np.complex64)
    v = np.zeros(4, dtype=np.complex64)
    
    for f in range(16):
        # Extract spatial vector for bin f
        x0 = X_freq[f, 0]
        x1 = X_freq[f, 1]
        x2 = X_freq[f, 2]
        x3 = X_freq[f, 3]
        
        # Output y = w^H * x
        w0_c = weights[f, 0].real - 1j * weights[f, 0].imag
        w1_c = weights[f, 1].real - 1j * weights[f, 1].imag
        w2_c = weights[f, 2].real - 1j * weights[f, 2].imag
        w3_c = weights[f, 3].real - 1j * weights[f, 3].imag
        
        y = w0_c * x0 + w1_c * x1 + w2_c * x2 + w3_c * x3
        Y_freq[f] = y
        
        # Power estimation for normalization
        x_power = (
            x0.real*x0.real + x0.imag*x0.imag +
            x1.real*x1.real + x1.imag*x1.imag +
            x2.real*x2.real + x2.imag*x2.imag +
            x3.real*x3.real + x3.imag*x3.imag
        )
        
        bin_mu = mu / (x_power + epsilon)
        y_conj = y.real - 1j * y.imag
        
        # Leaky LMS update for bin weights
        v[0] = 0.9999 * weights[f, 0] - bin_mu * y_conj * x0
        v[1] = 0.9999 * weights[f, 1] - bin_mu * y_conj * x1
        v[2] = 0.9999 * weights[f, 2] - bin_mu * y_conj * x2
        v[3] = 0.9999 * weights[f, 3] - bin_mu * y_conj * x3
        
        # Frost target projection
        a0_c = target_a[0].real - 1j * target_a[0].imag
        a1_c = target_a[1].real - 1j * target_a[1].imag
        a2_c = target_a[2].real - 1j * target_a[2].imag
        a3_c = target_a[3].real - 1j * target_a[3].imag
        
        inner = a0_c * v[0] + a1_c * v[1] + a2_c * v[2] + a3_c * v[3]
        
        weights[f, 0] = v[0] - 0.25 * inner * target_a[0] + 0.25 * target_a[0]
        weights[f, 1] = v[1] - 0.25 * inner * target_a[1] + 0.25 * target_a[1]
        weights[f, 2] = v[2] - 0.25 * inner * target_a[2] + 0.25 * target_a[2]
        weights[f, 3] = v[3] - 0.25 * inner * target_a[3] + 0.25 * target_a[3]
        
    # 3. Transform filtered frequency block back to time domain (IFFT)
    for t in range(16):
        val = 0.0 + 0.0j
        for f in range(16):
            val += idft_mat[t, f] * Y_freq[f]
        y_block_out[t] = val


class FFTSTAPProcessor:
    """
    Manages FFT-based Overlap-Safe Space-Time Adaptive Processing.
    Filters wideband and chirped signals in frequency bins.
    """
    def __init__(self, mu: float = 0.05, epsilon: float = 1e-4):
        self.mu = mu
        self.epsilon = epsilon
        
        # Initialize weights (16 bins x 4 channels)
        self.weights = np.zeros((16, 4), dtype=np.complex64)
        for f in range(16):
            self.weights[f, :] = 0.25
            
        # Precompute DFT and IDFT matrices of size 16
        self.dft_mat = np.zeros((16, 16), dtype=np.complex64)
        self.idft_mat = np.zeros((16, 16), dtype=np.complex64)
        
        for f in range(16):
            for t in range(16):
                angle = -2.0 * math.pi * f * t / 16.0
                self.dft_mat[f, t] = math.cos(angle) + 1j * math.sin(angle)
                
                angle_inv = 2.0 * math.pi * f * t / 16.0
                self.idft_mat[f, t] = (math.cos(angle_inv) + 1j * math.sin(angle_inv)) / 16.0

    def process_telemetry_history(
        self,
        target_steering: np.ndarray,  # (4,) complex64
        received_history: np.ndarray  # (N, 4) complex64
    ) -> np.ndarray:
        """
        Processes entire time-domain history in blocks of 16.
        Returns clean output time-domain signal.
        """
        N = received_history.shape[0]
        num_blocks = N // 16
        
        out_signal = np.zeros(num_blocks * 16, dtype=np.complex64)
        x_block = np.zeros((16, 4), dtype=np.complex64)
        y_block_out = np.zeros(16, dtype=np.complex64)
        
        # Process block-by-block
        for b in range(num_blocks):
            # Ingest 16 samples block
            x_block[:, :] = received_history[b*16 : (b+1)*16, :]
            
            # Execute JIT step
            _fft_stap_process_block_jit(
                x_block,
                self.dft_mat,
                self.idft_mat,
                self.weights,
                target_steering.astype(np.complex64),
                self.mu,
                self.epsilon,
                y_block_out
            )
            
            # Write to output buffer
            out_signal[b*16 : (b+1)*16] = y_block_out
            
        return out_signal


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: FFT STAP Processor Validation")
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
    sim_steps = 1600  # Must be multiple of 16

    # 1. Narrowband Jammer
    print("[*] Scenario 1: Narrowband Suppression...")
    x_nb = np.zeros((sim_steps, 4), dtype=np.complex64)
    for n in range(sim_steps):
        i_amp = 3.0 * (math.cos(10.0 * n * 0.001) + 1j * math.sin(10.0 * n * 0.001))
        x_nb[n] = i_amp * spoofer_a + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
        
    stap_nb = FFTSTAPProcessor(mu=0.08)
    out_nb = stap_nb.process_telemetry_history(target_a, x_nb)
    
    p_init = np.mean(np.abs(x_nb[:32, 0])**2)
    p_final = np.mean(np.abs(out_nb[-32:])**2)
    nb_db = 10.0 * math.log10(max(p_init, 1e-12) / max(p_final, 1e-12))
    print(f"    -> Narrowband Suppression:      {nb_db:.2f} dB")
    assert nb_db > 20.0, "FFT STAP failed on narrowband jammer!"
    print("    -> Narrowband check: [PASSED]")

    # 2. Wideband Jammer
    print("\n[*] Scenario 2: Wideband Burst Suppression...")
    x_wb = np.zeros((sim_steps, 4), dtype=np.complex64)
    for n in range(sim_steps):
        i_amp = 3.0 * (np.random.normal() + 1j * np.random.normal())
        x_wb[n] = i_amp * spoofer_a + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
        
    stap_wb = FFTSTAPProcessor(mu=0.08)
    out_wb = stap_wb.process_telemetry_history(target_a, x_wb)
    
    p_init_wb = np.mean(np.abs(x_wb[:32, 0])**2)
    p_final_wb = np.mean(np.abs(out_wb[-32:])**2)
    wb_db = 10.0 * math.log10(max(p_init_wb, 1e-12) / max(p_final_wb, 1e-12))
    print(f"    -> Wideband Suppression:        {wb_db:.2f} dB")
    assert wb_db > 15.0, "FFT STAP failed on wideband jammer!"
    print("    -> Wideband check: [PASSED]")

    # 3. Chirp Interference
    print("\n[*] Scenario 3: Chirped Suppression...")
    x_chirp = np.zeros((sim_steps, 4), dtype=np.complex64)
    for n in range(sim_steps):
        t = n * 0.001
        phase = 2.0 * math.pi * 5.0 * t + math.pi * 20.0 * (t ** 2)
        i_amp = 3.0 * (math.cos(phase) + 1j * math.sin(phase))
        x_chirp[n] = i_amp * spoofer_a + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
        
    stap_chirp = FFTSTAPProcessor(mu=0.08)
    out_chirp = stap_chirp.process_telemetry_history(target_a, x_chirp)
    
    p_init_chirp = np.mean(np.abs(x_chirp[:32, 0])**2)
    p_final_chirp = np.mean(np.abs(out_chirp[-32:])**2)
    chirp_db = 10.0 * math.log10(max(p_init_chirp, 1e-12) / max(p_final_chirp, 1e-12))
    print(f"    -> Chirp Suppression:           {chirp_db:.2f} dB")
    assert chirp_db > 18.0, "FFT STAP failed on chirped jammer!"
    print("    -> Chirp check: [PASSED]")

    print("\n[+] FFT STAP processor validation complete.")
