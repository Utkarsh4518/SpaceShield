"""
Task 48.1: RF Front-End Impairment Emulator Utility
Zero-Allocation, Numba-Accelerated Physical RF Channel Simulation
"""

import time
import math
import numpy as np
from numba import njit

# ==============================================================================
# JIT-Compiled Simulation Kernels
# ==============================================================================

@njit(fastmath=True, cache=True)
def _emulate_impairments_kernel(
    data: np.ndarray,
    phase_noise_std: np.float32,
    a_fade: np.float32,
    b_fade: np.float32,
    k_factor: np.float32,
    state: np.ndarray,
    noise_phase: np.ndarray,
    noise_fade_r: np.ndarray,
    noise_fade_i: np.ndarray
) -> None:
    """
    Applies simulated hardware oscillator phase noise (white frequency random-walk)
    and flat Rician/Rayleigh fading (AR(1) complex process) to incoming channel strides.
    Operates strictly in-place with zero runtime heap modifications.
    Uses a fast circular lookup table for noise and Taylor approximation for phase updates.
    All hot path operations use single-precision float32 values to prevent double-precision
    cast overhead and enable optimal SIMD compilation.
    
    Args:
        data: (num_channels, N) complex64 stride view.
        phase_noise_std: Standard deviation of phase increment per sample.
        a_fade: AR(1) fading transition coefficient.
        b_fade: AR(1) fading noise scaling coefficient.
        k_factor: Rician K-factor (ratio of LOS to diffuse power). If 0, pure Rayleigh.
        state: Pre-allocated float64 array of shape (6,) storing state:
               - state[0]: phasor_r (real part of oscillator phasor)
               - state[1]: phasor_i (imag part of oscillator phasor)
               - state[2]: g_r (real part of fading)
               - state[3]: g_i (imag part of fading)
               - state[4]: phase_idx (index for circular phase noise table)
               - state[5]: fade_idx (index for circular fading noise table)
        noise_phase: (65536,) float32 pre-generated phase noise array.
        noise_fade_r: (65536,) float32 pre-generated fading real noise array.
        noise_fade_i: (65536,) float32 pre-generated fading imag noise array.
    """
    N = data.shape[1]
    
    # Restore state values into single-precision float32 scalars
    p_r = np.float32(state[0])
    p_i = np.float32(state[1])
    g_r = np.float32(state[2])
    g_i = np.float32(state[3])
    phase_idx = int(state[4])
    fade_idx = int(state[5])
    
    # Rician power distribution scaling factors
    los_scale = np.float32(math.sqrt(k_factor / (k_factor + 1.0)))
    diff_scale = np.float32(math.sqrt(1.0 / (k_factor + 1.0)))
    
    # Standard deviation for independent real/imag fading noise elements (power divided by 2)
    noise_std = np.float32(0.7071067811865475)
    
    TABLE_MASK = 65535 # 2^16 - 1
    
    for k in range(N):
        # 1. Update Carrier Phase Noise using Taylor Approximation
        idx_p = (phase_idx + k) & TABLE_MASK
        w = np.float32(noise_phase[idx_p]) * phase_noise_std
        
        # cos(w) ~ 1 - w^2 / 2, sin(w) ~ w (extremely accurate for w << 1)
        dw_r = np.float32(1.0) - np.float32(0.5) * w * w
        dw_i = w
        
        # Complex update of phasor: p = p * dw
        new_p_r = p_r * dw_r - p_i * dw_i
        new_p_i = p_r * dw_i + p_i * dw_r
        p_r = new_p_r
        p_i = new_p_i
        
        # Normalize phasor every 64 samples to eliminate numerical drift accumulation
        if (k & 63) == 0:
            norm_p = np.float32(math.sqrt(p_r * p_r + p_i * p_i))
            p_r /= norm_p
            p_i /= norm_p
        
        # 2. Update Rayleigh Fading Diffuse Component (Complex AR(1) process)
        idx_f = (fade_idx + k) & TABLE_MASK
        w_r = np.float32(noise_fade_r[idx_f]) * noise_std
        w_i = np.float32(noise_fade_i[idx_f]) * noise_std
        
        g_r = a_fade * g_r + b_fade * w_r
        g_i = a_fade * g_i + b_fade * w_i
        
        # 3. Construct Rician fading coefficient: h_k = los_scale + diff_scale * g_k
        h_r = los_scale + diff_scale * g_r
        h_i = diff_scale * g_i
        
        # 4. Total impairment multiplier: m_k = h_k * p_k
        m_r = h_r * p_r - h_i * p_i
        m_i = h_r * p_i + h_i * p_r
        
        # 5. Apply in-place to all 4 receiver channels (zero allocation, manual indexing)
        x0 = data[0, k]
        x1 = data[1, k]
        x2 = data[2, k]
        x3 = data[3, k]
        
        x0_r, x0_i = x0.real, x0.imag
        x1_r, x1_i = x1.real, x1.imag
        x2_r, x2_i = x2.real, x2.imag
        x3_r, x3_i = x3.real, x3.imag
        
        data[0, k] = complex(x0_r * m_r - x0_i * m_i, x0_r * m_i + x0_i * m_r)
        data[1, k] = complex(x1_r * m_r - x1_i * m_i, x1_r * m_i + x1_i * m_r)
        data[2, k] = complex(x2_r * m_r - x2_i * m_i, x2_r * m_i + x2_i * m_r)
        data[3, k] = complex(x3_r * m_r - x3_i * m_i, x3_r * m_i + x3_i * m_r)
            
    # Save states back for the next stride block
    state[0] = np.float64(p_r)
    state[1] = np.float64(p_i)
    state[2] = np.float64(g_r)
    state[3] = np.float64(g_i)
    state[4] = (phase_idx + N) & TABLE_MASK
    state[5] = (fade_idx + N) & TABLE_MASK


# ==============================================================================
# RF Front-End Impairment Emulator
# ==============================================================================

class RfFrontendEmulator:
    """
    RF Front-End Impairment Emulator.
    Simulates oscillator phase noise and Rayleigh/Rician flat fading channels.
    Operates in-place on pre-allocated complex64 arrays.
    """
    def __init__(
        self,
        sample_rate_hz: float = 2.0e6,
        phase_noise_std: float = 0.005,
        doppler_spread_hz: float = 50.0,
        rice_k_factor: float = 4.0
    ):
        self.sample_rate = sample_rate_hz
        self.phase_noise_std = phase_noise_std
        self.rice_k_factor = rice_k_factor
        
        # Calculate AR(1) fading coefficients
        if doppler_spread_hz > 0:
            self.a_fade = math.exp(-2.0 * math.pi * doppler_spread_hz / sample_rate_hz)
            self.b_fade = math.sqrt(1.0 - self.a_fade * self.a_fade)
        else:
            self.a_fade = 1.0
            self.b_fade = 0.0
            
        # Pre-allocate circular lookup tables for normal random noise (65536 elements)
        # Using a deterministic seed to ensure test repeatability
        rng = np.random.default_rng(0x5EED)
        self.noise_phase = rng.normal(0.0, 1.0, 65536).astype(np.float32)
        self.noise_fade_r = rng.normal(0.0, 1.0, 65536).astype(np.float32)
        self.noise_fade_i = rng.normal(0.0, 1.0, 65536).astype(np.float32)
        
        # Pre-allocated state register (phasor_r, phasor_i, g_r, g_i, phase_idx, fade_idx)
        self.state = np.zeros(6, dtype=np.float64)
        
        # Initialize active state
        self.state[0] = 1.0  # phasor_r
        self.state[1] = 0.0  # phasor_i
        self.state[2] = 1.0  # fading g_r
        self.state[3] = 0.0  # fading g_i
        
        self._warmup()
        
    def _warmup(self):
        """Forces Numba JIT compilation trace."""
        dummy_data = np.ones((4, 1024), dtype=np.complex64)
        _emulate_impairments_kernel(
            dummy_data,
            np.float32(self.phase_noise_std),
            np.float32(self.a_fade),
            np.float32(self.b_fade),
            np.float32(self.rice_k_factor),
            self.state,
            self.noise_phase,
            self.noise_fade_r,
            self.noise_fade_i
        )
        # Reset state after warmup
        self.state[0] = 1.0
        self.state[1] = 0.0
        self.state[2] = 1.0
        self.state[3] = 0.0
        self.state[4] = 0.0
        self.state[5] = 0.0
        
    def emulate_impairments(self, data: np.ndarray):
        """
        Intercepts and injects RF impairments in-place into the complex64 stride block.
        
        Args:
            data: shape (4, N) complex64 numpy array.
        """
        if data.ndim != 2 or data.shape[0] != 4:
            raise ValueError("Input data must have shape (4, N).")
            
        _emulate_impairments_kernel(
            data,
            np.float32(self.phase_noise_std),
            np.float32(self.a_fade),
            np.float32(self.b_fade),
            np.float32(self.rice_k_factor),
            self.state,
            self.noise_phase,
            self.noise_fade_r,
            self.noise_fade_i
        )


# ==============================================================================
# Standalone Diagnostic Verification Harness
# ==============================================================================

if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield RF Ingestion: Hardware Impairment Emulator")
    print("==================================================================")
    
    # Configure emulator: 2 MSPS, 0.005 rad phase noise, 100 Hz Doppler, 3.0 Rician factor
    emulator = RfFrontendEmulator(
        sample_rate_hz=2.0e6,
        phase_noise_std=0.005,
        doppler_spread_hz=100.0,
        rice_k_factor=3.0
    )
    print("[PASS] Impairment Emulator Initialized & JIT Compiled.")
    
    # Generate mock 4-channel signal (e.g. constant pilot tones at 0.5 amplitude)
    # Using chunk size 2048 to benchmark latency under the strict 15us limit
    chunk_size = 2048
    num_channels = 4
    
    data_raw = np.ones((num_channels, chunk_size), dtype=np.complex64) * 0.5
    data_test = data_raw.copy()
    
    # 1. Run emulator on the test stride
    t_start = time.perf_counter_ns()
    emulator.emulate_impairments(data_test)
    t_end = time.perf_counter_ns()
    latency_us = (t_end - t_start) / 1000.0
    
    print(f"\n[EVAL] Stride Processing Latency (Chunk Size: {chunk_size} samples):")
    print(f"    -> First Run Ingestion Latency: {latency_us:.3f} microseconds")
    
    # Run 1000 iterations to measure average performance
    latencies = []
    for _ in range(1000):
        data_bench = data_raw.copy()
        t1 = time.perf_counter_ns()
        emulator.emulate_impairments(data_bench)
        t2 = time.perf_counter_ns()
        latencies.append((t2 - t1) / 1000.0)
        
    avg_us = sum(latencies) / len(latencies)
    max_us = np.percentile(latencies, 99.0)
    print(f"    -> Average Execution Latency:  {avg_us:.3f} microseconds")
    print(f"    -> P99 Max Execution Jitter:   {max_us:.3f} microseconds")
    
    # 2. Check Power Normalization over multiple chunks to ensure statistical convergence
    power_ratios = []
    emulator_power = RfFrontendEmulator(
        sample_rate_hz=2.0e6,
        phase_noise_std=0.005,
        doppler_spread_hz=100.0,
        rice_k_factor=3.0
    )
    # Run audit over 500 chunks to verify average power conservation of fading channel
    for _ in range(500):
        data_chunk = np.ones((num_channels, chunk_size), dtype=np.complex64) * 0.5
        emulator_power.emulate_impairments(data_chunk)
        power_raw = np.mean(np.abs(0.5) ** 2)
        power_chunk = np.mean(np.abs(data_chunk) ** 2)
        power_ratios.append(power_chunk / power_raw)
        
    mean_power_ratio = float(np.mean(power_ratios))
    print(f"\n[EVAL] Power Integrity Audit:")
    print(f"    -> Mean Power Scaling Ratio (500 chunks): {mean_power_ratio:.6f} (Expected: ~1.00)")
    
    # 3. Verify Phase Noise Accumulation (isolating it by setting K to be very large)
    emulator_pn = RfFrontendEmulator(
        sample_rate_hz=2.0e6,
        phase_noise_std=0.005,
        doppler_spread_hz=100.0,
        rice_k_factor=1e9 # effectively disable fading phase variance
    )
    data_pn = np.ones((num_channels, chunk_size), dtype=np.complex64) * 0.5
    emulator_pn.emulate_impairments(data_pn)
    phases = np.angle(data_pn[0, :])
    phase_diffs = np.diff(phases)
    # Handle wrapping
    phase_diffs = (phase_diffs + np.pi) % (2.0 * np.pi) - np.pi
    observed_pn_std = float(np.std(phase_diffs))
    print(f"\n[EVAL] Phase Noise Random Walk Audit:")
    print(f"    -> Target Phase Noise Std:  {emulator_pn.phase_noise_std:.6f} rad")
    print(f"    -> Observed Phase Step Std: {observed_pn_std:.6f} rad (Expected: ~{emulator_pn.phase_noise_std:.6f})")
    
    # Assert correctness
    latency_ok = avg_us < 15.0
    power_ok = abs(mean_power_ratio - 1.0) < 0.15 # Allow standard statistical variance for 500 chunks
    phase_noise_ok = abs(observed_pn_std - emulator_pn.phase_noise_std) < 0.001
    
    print("\n[VERIFY] Performance and DSP Model Verification:")
    if latency_ok:
        print("[PASS] Impairment execution runs well under the 15.0 microsecond limit.")
    else:
        print("[FAIL] Ingestion pipeline execution breached the 15.0 microsecond real-time limit.")
        
    if power_ok:
        print("[PASS] Fading channel power is correctly normalized.")
    else:
        print("[FAIL] Fading channel introduced abnormal signal amplification or attenuation.")
        
    if phase_noise_ok:
        print("[PASS] Phase noise standard deviation is accurate.")
    else:
        print("[FAIL] Phase noise standard deviation deviated significantly from target.")
        
    assert latency_ok, "Failing due to latency limit breach"
    assert power_ok, "Failing due to fading power normalization breach"
    assert phase_noise_ok, "Failing due to phase noise verification breach"
    print("==================================================================")
