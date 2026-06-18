import time
import numpy as np
from numba import njit
import scipy.linalg as la

@njit(fastmath=True, cache=True)
def _synthesize_kernel(aligned_stride, weights, output_buffer, num_channels, stride_length):
    """
    Numba JIT Kernel: Vectorized coherent aperture combining.
    aligned_stride: (M, N) complex64
    weights: (M,) complex64
    output_buffer: (N,) complex64
    """
    for n in range(stride_length):
        val = 0.0 + 0j
        for c in range(num_channels):
            # Apply weights: conjugate of weight times aligned signal
            # Standard beamforming: y = w^H * x
            val += weights[c].conjugate() * aligned_stride[c, n]
        output_buffer[n] = val


class CoherentApertureSynthesizer:
    """
    Coherent Aperture Synthesizer Subsystem.
    Combines phase-aligned multi-aperture input signals to maximize the signal-to-noise ratio
    using Maximal Ratio Combining (MRC) or MVDR-based null steering.
    Guarantees zero heap allocation in hot-path.
    """
    def __init__(self, num_channels: int = 4, stride_length: int = 4096, diagonal_loading: float = 1e-4):
        self.num_channels = num_channels
        self.stride_length = stride_length
        self.diagonal_loading = diagonal_loading
        
        # Pre-allocated active weights (default to MRC normalized weights)
        self.weights = np.ones(self.num_channels, dtype=np.complex64) / np.float32(np.sqrt(self.num_channels))
        
        # Pre-allocated output and scratch buffers to prevent garbage collection spikes
        self._output_buffer = np.zeros(self.stride_length, dtype=np.complex64)
        self._identity = np.eye(self.num_channels, dtype=np.complex64)
        self._loaded_R = np.zeros((self.num_channels, self.num_channels), dtype=np.complex64)
        
        # JIT Warmup
        self._warmup()

    def _warmup(self):
        """Warms up the Numba JIT compiler with mock inputs."""
        mock_aligned = np.random.randn(self.num_channels, self.stride_length).astype(np.complex64)
        _synthesize_kernel(
            mock_aligned, self.weights, self._output_buffer,
            self.num_channels, self.stride_length
        )

    def set_weights(self, weights: np.ndarray):
        """Sets custom combining weights."""
        np.copyto(self.weights, weights)

    def set_mrc_weights(self):
        """Sets uniform weights matching the theoretical MRC bound for aligned, equal-power channels."""
        # Standardize weights to 1/sqrt(M) so that the output noise power equals the input noise power,
        # which makes calculating the 10*log10(M) gain very clean.
        self.weights.fill(1.0 / np.sqrt(self.num_channels))

    def update_mvdr_weights(self, R: np.ndarray, target_sv: np.ndarray):
        """
        Calculates MVDR beamforming weights to suppress spatial interference while preserving the target.
        Formula: w = (R^-1 * a) / (a^H * R^-1 * a)
        """
        # Apply diagonal loading
        np.multiply(self._identity, self.diagonal_loading, out=self._loaded_R)
        np.add(self._loaded_R, R, out=self._loaded_R)
        
        try:
            R_inv = la.inv(self._loaded_R)
        except la.LinAlgError:
            # Fallback to MRC weights on singular matrix
            self.set_mrc_weights()
            return
            
        # numerator = R_inv @ target_sv
        numerator = np.dot(R_inv, target_sv)
        
        # denominator = target_sv^H @ R_inv @ target_sv
        denominator = np.vdot(target_sv, numerator)
        
        # w = numerator / denominator
        if abs(denominator) > 1e-12:
            self.weights[:] = numerator / denominator
        else:
            self.set_mrc_weights()

    def synthesize(self, aligned_stride: np.ndarray) -> tuple:
        """
        Coherently synthesizes (combines) multi-aperture signals.
        Returns:
            synthesized_view (np.ndarray): Pre-allocated combined buffer.
            exec_us (float): Latency in microseconds.
        """
        t0 = time.perf_counter()
        
        _synthesize_kernel(
            aligned_stride, self.weights, self._output_buffer,
            self.num_channels, self.stride_length
        )
        
        exec_us = (time.perf_counter() - t0) * 1e6
        return self._output_buffer, exec_us


# --- Rapid Verification & Benchmarking Harness ---
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Coherent Aperture Synthesizer")
    print("==================================================================")
    
    M = 4
    N = 4096
    synthesizer = CoherentApertureSynthesizer(num_channels=M, stride_length=N)
    
    # 1. Verify MRC SNR Gain
    print("[*] Verifying MRC Combining Gain...")
    rng = np.random.default_rng(42)
    # Target signal
    target_signal = rng.choice([1+1j, 1-1j, -1+1j, -1-1j], N) / np.sqrt(2.0)
    
    # Generate 4 independent noise channels
    noise_power = 0.5
    noise = (rng.normal(0, np.sqrt(noise_power / 2), (M, N)) + 
             1j * rng.normal(0, np.sqrt(noise_power / 2), (M, N))).astype(np.complex64)
             
    # Input channels: perfectly aligned signals + independent noise
    aligned_stride = np.zeros((M, N), dtype=np.complex64)
    for c in range(M):
        aligned_stride[c, :] = target_signal + noise[c, :]
        
    synthesizer.set_mrc_weights()
    combined, exec_us = synthesizer.synthesize(aligned_stride)
    
    # Calculate SNR of Channel 0
    signal_power = np.mean(np.abs(target_signal)**2)
    input_noise_power_est = np.mean(np.abs(aligned_stride[0, :] - target_signal)**2)
    input_snr = signal_power / input_noise_power_est
    input_snr_db = 10 * np.log10(input_snr)
    
    # Combined output: the signal amplitude is weighted by sum(w_c)
    # Since w_c = 1/sqrt(M), signal is scaled by M/sqrt(M) = sqrt(M).
    # Power is scaled by M. Combined noise power is equal to input noise power.
    # So SNR is scaled by M.
    output_noise_power_est = np.mean(np.abs(combined - np.sqrt(M) * target_signal)**2)
    output_snr = (np.sqrt(M) * signal_power) / output_noise_power_est # Wait, signal power in combined is scaled by M
    # Let's calculate actual SNR from combined signal directly:
    # Combined signal is np.sqrt(M) * target_signal, power is M * signal_power
    output_signal_power = np.mean(np.abs(np.sqrt(M) * target_signal)**2)
    output_snr = output_signal_power / output_noise_power_est
    output_snr_db = 10 * np.log10(output_snr)
    
    snr_gain = output_snr_db - input_snr_db
    print(f" [>] Input SNR: {input_snr_db:.2f} dB")
    print(f" [>] Output SNR: {output_snr_db:.2f} dB")
    print(f" [>] Measured Gain: {snr_gain:.2f} dB (Theoretical MRC Limit: {10*np.log10(M):.2f} dB)")
    
    # 2. Verify Jammer Nulling response
    print("\n[*] Verifying Jammer Null-Steering Response...")
    target_sv = np.ones(M, dtype=np.complex64) # Target at broadside
    jammer_angle_deg = 30.0
    jammer_sv = np.exp(-1j * np.pi * np.sin(np.radians(jammer_angle_deg)) * np.arange(M)).astype(np.complex64)
    
    # Covariance matrix with high power Jammer
    jammer_power = 1e4
    R = jammer_power * np.outer(jammer_sv, jammer_sv.conj()) + np.eye(M, dtype=np.complex64)
    
    synthesizer.update_mvdr_weights(R, target_sv)
    w_opt = synthesizer.weights
    
    # Response calculations
    target_response = np.abs(np.vdot(w_opt, target_sv))
    jammer_response = np.abs(np.vdot(w_opt, jammer_sv))
    
    null_depth_db = 20 * np.log10(jammer_response / (target_response + 1e-12))
    print(f" [>] Target Response (Gain): {target_response:.4f}")
    print(f" [>] Jammer Response (Gain): {jammer_response:.6f}")
    print(f" [>] Jammer Null Depth: {null_depth_db:.2f} dB")
    
    # 3. Latency Check
    latencies = []
    for _ in range(1000):
        _, us = synthesizer.synthesize(aligned_stride)
        latencies.append(us)
        
    avg_latency = np.mean(latencies)
    import sys
    is_rt_capable = sys.platform.startswith('linux')
    latency_limit = 10.0 if is_rt_capable else 25.0
    print(f"\n [>] Average Synthesis Latency: {avg_latency:.4f} us (Limit: <{latency_limit:.1f} us)")
    
    assert abs(snr_gain - 6.02) < 0.2, "MRC gain verification mismatch."
    assert null_depth_db < -40.0, "Jammer null-steering depth insufficient."
    assert avg_latency < latency_limit, f"Synthesis latency {avg_latency:.2f} us exceeded limit of {latency_limit:.1f} us."
    print("\n[PASSED] Coherent Aperture Synthesizer compiled and verified successfully!")
    print("==================================================================")
