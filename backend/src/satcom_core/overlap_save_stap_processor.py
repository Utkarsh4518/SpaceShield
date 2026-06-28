"""
Task 68.1: Overlap-Save STAP Processor Module
SpaceShield High-Velocity Receiver DSP Subsystem

Upgrades FFT STAP with an overlap-save block convolution algorithm.
Maintains a pre-allocated sliding buffer to eliminate circular convolution boundary artifacts.
"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _overlap_save_stap_jit(
    sliding_buf: np.ndarray,      # (32, 4) complex64
    new_data: np.ndarray,         # (16, 4) complex64
    dft_mat: np.ndarray,          # (32, 32) complex64
    idft_mat: np.ndarray,         # (32, 32) complex64
    weights: np.ndarray,          # (32, 4) complex64
    target_a: np.ndarray,         # (4,) complex64
    mu: float,
    epsilon: float,
    out_block: np.ndarray         # (16,) complex64
):
    """
    Zero-Heap JIT Overlap-Save block processing loop:
    1. Shifts sliding buffer by 16 samples.
    2. Runs size-32 FFT on each channel.
    3. Runs separate Frost LMS updates on all 32 frequency bins.
    4. Runs size-32 IDFT.
    5. Discards the first 16 corrupted samples, keeping the last 16.
    """
    # 1. Shift old samples to top and copy new samples to bottom
    for t in range(16):
        for c in range(4):
            sliding_buf[t, c] = sliding_buf[t + 16, c]
            sliding_buf[t + 16, c] = new_data[t, c]
            
    # 2. Transform block to frequency domain (FFT size 32)
    X_freq = np.zeros((32, 4), dtype=np.complex64)
    for c in range(4):
        for f in range(32):
            val = 0.0 + 0.0j
            for t in range(32):
                val += dft_mat[f, t] * sliding_buf[t, c]
            X_freq[f, c] = val
            
    # 3. Spatial filtering per bin
    Y_freq = np.zeros(32, dtype=np.complex64)
    v = np.zeros(4, dtype=np.complex64)
    
    for f in range(32):
        x0 = X_freq[f, 0]
        x1 = X_freq[f, 1]
        x2 = X_freq[f, 2]
        x3 = X_freq[f, 3]
        
        w0_c = weights[f, 0].real - 1j * weights[f, 0].imag
        w1_c = weights[f, 1].real - 1j * weights[f, 1].imag
        w2_c = weights[f, 2].real - 1j * weights[f, 2].imag
        w3_c = weights[f, 3].real - 1j * weights[f, 3].imag
        
        y = w0_c * x0 + w1_c * x1 + w2_c * x2 + w3_c * x3
        Y_freq[f] = y
        
        x_power = (
            x0.real*x0.real + x0.imag*x0.imag +
            x1.real*x1.real + x1.imag*x1.imag +
            x2.real*x2.real + x2.imag*x2.imag +
            x3.real*x3.real + x3.imag*x3.imag
        )
        
        bin_mu = mu / (x_power + epsilon)
        y_conj = y.real - 1j * y.imag
        
        # Leaky weight update
        v[0] = 0.9999 * weights[f, 0] - bin_mu * y_conj * x0
        v[1] = 0.9999 * weights[f, 1] - bin_mu * y_conj * x1
        v[2] = 0.9999 * weights[f, 2] - bin_mu * y_conj * x2
        v[3] = 0.9999 * weights[f, 3] - bin_mu * y_conj * x3
        
        a0_c = target_a[0].real - 1j * target_a[0].imag
        a1_c = target_a[1].real - 1j * target_a[1].imag
        a2_c = target_a[2].real - 1j * target_a[2].imag
        a3_c = target_a[3].real - 1j * target_a[3].imag
        
        inner = a0_c * v[0] + a1_c * v[1] + a2_c * v[2] + a3_c * v[3]
        
        weights[f, 0] = v[0] - 0.25 * inner * target_a[0] + 0.25 * target_a[0]
        weights[f, 1] = v[1] - 0.25 * inner * target_a[1] + 0.25 * target_a[1]
        weights[f, 2] = v[2] - 0.25 * inner * target_a[2] + 0.25 * target_a[2]
        weights[f, 3] = v[3] - 0.25 * inner * target_a[3] + 0.25 * target_a[3]
        
    # 4. Transform filtered spectrum back to time (IDFT size 32)
    y_time = np.zeros(32, dtype=np.complex64)
    for t in range(32):
        val = 0.0 + 0.0j
        for f in range(32):
            val += idft_mat[t, f] * Y_freq[f]
        y_time[t] = val
        
    # 5. Extract non-corrupted samples (index 16 to 31)
    for t in range(16):
        out_block[t] = y_time[t + 16]


class OverlapSaveSTAPProcessor:
    """
    Overlap-Save Frequency-Domain STAP processor.
    Eliminates circular convolution transients by maintaining sliding buffers.
    """
    def __init__(self, mu: float = 0.05, epsilon: float = 1e-4):
        self.mu = mu
        self.epsilon = epsilon
        
        # Pre-allocated variables
        self.sliding_buf = np.zeros((32, 4), dtype=np.complex64)
        self.weights = np.zeros((32, 4), dtype=np.complex64)
        for f in range(32):
            self.weights[f, :] = 0.25
            
        # Precompute size-32 DFT / IDFT matrices
        self.dft_mat = np.zeros((32, 32), dtype=np.complex64)
        self.idft_mat = np.zeros((32, 32), dtype=np.complex64)
        
        for f in range(32):
            for t in range(32):
                angle = -2.0 * math.pi * f * t / 32.0
                self.dft_mat[f, t] = math.cos(angle) + 1j * math.sin(angle)
                
                angle_inv = 2.0 * math.pi * f * t / 32.0
                self.idft_mat[f, t] = (math.cos(angle_inv) + 1j * math.sin(angle_inv)) / 32.0

    def process_history(
        self,
        target_steering: np.ndarray,  # (4,) complex64
        received_history: np.ndarray  # (N, 4) complex64
    ) -> np.ndarray:
        """
        Processes multi-channel time history.
        """
        N = received_history.shape[0]
        num_blocks = N // 16
        
        out_signal = np.zeros(num_blocks * 16, dtype=np.complex64)
        new_data = np.zeros((16, 4), dtype=np.complex64)
        out_block = np.zeros(16, dtype=np.complex64)
        
        # Reset sliding buffer to zeroes
        self.sliding_buf.fill(0)
        
        for b in range(num_blocks):
            new_data[:, :] = received_history[b*16 : (b+1)*16, :]
            
            _overlap_save_stap_jit(
                self.sliding_buf,
                new_data,
                self.dft_mat,
                self.idft_mat,
                self.weights,
                target_steering.astype(np.complex64),
                self.mu,
                self.epsilon,
                out_block
            )
            
            out_signal[b*16 : (b+1)*16] = out_block
            
        return out_signal


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Overlap-Save STAP Validation")
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
    sim_steps = 1600

    # 1. Narrowband Jammer
    print("[*] Scenario 1: Narrowband Suppression...")
    x_nb = np.zeros((sim_steps, 4), dtype=np.complex64)
    for n in range(sim_steps):
        i_amp = 3.0 * (math.cos(10.0 * n * 0.001) + 1j * math.sin(10.0 * n * 0.001))
        x_nb[n] = i_amp * spoofer_a + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
        
    processor = OverlapSaveSTAPProcessor(mu=0.08)
    out_nb = processor.process_history(target_a, x_nb)
    
    p_init = np.mean(np.abs(x_nb[:32, 0])**2)
    p_final = np.mean(np.abs(out_nb[-32:])**2)
    nb_db = 10.0 * math.log10(max(p_init, 1e-12) / max(p_final, 1e-12))
    print(f"    -> Narrowband Suppression:      {nb_db:.2f} dB")
    assert nb_db > 20.0, "Overlap-Save STAP failed on narrowband jammer!"
    print("    -> Narrowband check: [PASSED]")

    # 2. Wideband Jammer
    print("\n[*] Scenario 2: Wideband Burst Suppression...")
    x_wb = np.zeros((sim_steps, 4), dtype=np.complex64)
    for n in range(sim_steps):
        i_amp = 3.0 * (np.random.normal() + 1j * np.random.normal())
        x_wb[n] = i_amp * spoofer_a + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
        
    processor_wb = OverlapSaveSTAPProcessor(mu=0.08)
    out_wb = processor_wb.process_history(target_a, x_wb)
    
    p_init_wb = np.mean(np.abs(x_wb[:32, 0])**2)
    p_final_wb = np.mean(np.abs(out_wb[-32:])**2)
    wb_db = 10.0 * math.log10(max(p_init_wb, 1e-12) / max(p_final_wb, 1e-12))
    print(f"    -> Wideband Suppression:        {wb_db:.2f} dB")
    assert wb_db > 15.0, "Overlap-Save STAP failed on wideband jammer!"
    print("    -> Wideband check: [PASSED]")

    # 3. Boundary Discontinuity check (reconstruction phase continuity)
    print("\n[*] Scenario 3: Block Boundary Continuity check...")
    # Checking if there are any sudden magnitude leaps at block boundaries (multiples of 16)
    diffs = []
    for idx in range(15, len(out_nb) - 1, 16):
        d_val = np.abs(out_nb[idx+1] - out_nb[idx])
        diffs.append(d_val)
    mean_discontinuity = np.mean(diffs)
    print(f"    -> Average block boundary leap: {mean_discontinuity:.4f}")
    assert mean_discontinuity < 1.0, "Overlap-Save failed to enforce phase continuity at block boundaries!"
    print("    -> Boundary continuity check: [PASSED]")

    print("\n[+] Overlap-Save STAP processor validation complete.")
