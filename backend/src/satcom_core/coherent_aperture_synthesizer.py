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

@njit(fastmath=True, cache=True)
def _estimate_noise_powers_kernel(aligned, noise_powers, num_channels, stride_len):
    """
    Numba JIT Kernel: Estimates noise power per channel using the spatial mean as signal estimate,
    and applies linear algebra correction to obtain unbiased estimators.
    aligned: (M, N) complex64
    noise_powers: (M,) float32 (output)
    """
    # 1. Compute raw residual variances V
    V = np.zeros(num_channels, dtype=np.float32)
    for n in range(stride_len):
        sig_sum = 0.0 + 0j
        for c in range(num_channels):
            sig_sum += aligned[c, n]
        sig_est = sig_sum / num_channels
        
        for c in range(num_channels):
            diff = aligned[c, n] - sig_est
            V[c] += diff.real**2 + diff.imag**2
            
    for c in range(num_channels):
        V[c] /= stride_len
        
    # 2. Convert V to unbiased noise powers
    # Formula: sigma_c^2 = (V_c - sum(V)/(M*(M-1))) / (1 - 2/M)
    sum_V = 0.0
    for c in range(num_channels):
        sum_V += V[c]
        
    denom = 1.0 - 2.0 / num_channels
    factor = 1.0 / (num_channels * (num_channels - 1.0))
    for c in range(num_channels):
        val = (V[c] - sum_V * factor) / denom
        # Ensure non-negative noise power
        if val < 1e-6:
            val = 1e-6
        noise_powers[c] = val

@njit(fastmath=True, cache=True)
def _compute_inverse_noise_weights_kernel(noise_powers, weights, num_channels):
    """
    Numba JIT Kernel: Computes MRC weights from estimated channel noise powers.
    weights are normalized such that they sum to 1.0 (to preserve signal amplitude).
    """
    total_inv_power = 0.0
    for c in range(num_channels):
        inv_power = 1.0 / (noise_powers[c] + 1e-12)
        weights[c] = inv_power
        total_inv_power += inv_power
        
    for c in range(num_channels):
        weights[c] /= total_inv_power


class CoherentApertureSynthesizer:
    """
    Coherent Aperture Synthesizer Subsystem.
    Combines phase-aligned multi-aperture input streams to maximize the signal-to-noise ratio
    using noise-power weighted MRC or MVDR/LCMV beamforming to suppress grating lobes and aliases.
    Guarantees zero heap allocation in hot-path.
    """
    def __init__(self, num_channels: int = 4, stride_length: int = 4096, diagonal_loading: float = 1e-4):
        self.num_channels = num_channels
        self.stride_length = stride_length
        self.diagonal_loading = diagonal_loading
        
        # Pre-allocated weights and state arrays
        self.weights = np.ones(self.num_channels, dtype=np.complex64) / np.float32(np.sqrt(self.num_channels))
        self.noise_powers = np.zeros(self.num_channels, dtype=np.float32)
        
        # Pre-allocated output and scratch buffers to prevent garbage collection spikes
        self._output_buffer = np.zeros(self.stride_length, dtype=np.complex64)
        self._identity = np.eye(self.num_channels, dtype=np.complex64)
        self._loaded_R = np.zeros((self.num_channels, self.num_channels), dtype=np.complex64)
        
        # Pre-allocated matrices for LCMV (max 3 constraints: target + 2 grating lobes/aliases)
        self._max_constraints = 3
        self._C_H = np.zeros((self._max_constraints, self.num_channels), dtype=np.complex64)
        self._R_inv_C = np.zeros((self.num_channels, self._max_constraints), dtype=np.complex64)
        self._CH_R_inv_C = np.zeros((self._max_constraints, self._max_constraints), dtype=np.complex64)
        self._temp_vector = np.zeros(self._max_constraints, dtype=np.complex64)
        
        # JIT Warmup
        self._warmup()

    def _warmup(self):
        """Warms up Numba JIT kernels with mock inputs."""
        mock_aligned = np.random.randn(self.num_channels, self.stride_length).astype(np.complex64)
        _estimate_noise_powers_kernel(
            mock_aligned, self.noise_powers, self.num_channels, self.stride_length
        )
        _compute_inverse_noise_weights_kernel(
            self.noise_powers, self.weights, self.num_channels
        )
        _synthesize_kernel(
            mock_aligned, self.weights, self._output_buffer,
            self.num_channels, self.stride_length
        )
        # Reset defaults
        self.weights.fill(1.0 / np.sqrt(self.num_channels))
        self.noise_powers.fill(0.0)

    def update_mrc_weights_from_data(self, aligned_stride: np.ndarray):
        """
        Estimates channel noise powers from the aligned input data and updates
        the combining weights according to the inverse noise-power MRC formulation.
        """
        _estimate_noise_powers_kernel(
            aligned_stride, self.noise_powers, self.num_channels, self.stride_length
        )
        _compute_inverse_noise_weights_kernel(
            self.noise_powers, self.weights, self.num_channels
        )

    def set_weights(self, weights: np.ndarray):
        """Sets custom combining weights."""
        np.copyto(self.weights, weights)

    def set_uniform_weights(self):
        """Sets uniform MRC weights (1/sqrt(M))."""
        self.weights.fill(1.0 / np.sqrt(self.num_channels))

    def set_mrc_weights(self):
        """Sets uniform MRC weights (1/sqrt(M))."""
        self.set_uniform_weights()

    def update_mvdr_weights(self, R: np.ndarray, target_sv: np.ndarray):
        """
        Calculates MVDR beamforming weights to suppress spatial interference while preserving the target.
        Formula: w = (R^-1 * a) / (a^H * R^-1 * a)
        """
        np.multiply(self._identity, self.diagonal_loading, out=self._loaded_R)
        np.add(self._loaded_R, R, out=self._loaded_R)
        
        try:
            R_inv = la.inv(self._loaded_R)
        except la.LinAlgError:
            self.set_uniform_weights()
            return
            
        numerator = np.dot(R_inv, target_sv)
        denominator = np.vdot(target_sv, numerator)
        
        if abs(denominator) > 1e-12:
            self.weights[:] = numerator / denominator
        else:
            self.set_uniform_weights()

    def update_lcmv_weights(self, R: np.ndarray, C: np.ndarray, f: np.ndarray):
        """
        Calculates LCMV weights to suppress grating lobes and aliases while preserving the target.
        Formula: w = R^-1 * C * (C^H * R^-1 * C)^-1 * f
        C: (M, K) complex64 constraint matrix (first column is target steering vector, other columns are grating lobes/aliases)
        f: (K,) complex64 response vector (e.g. [1.0, 0.0, ...])
        """
        K = C.shape[1]
        
        np.multiply(self._identity, self.diagonal_loading, out=self._loaded_R)
        np.add(self._loaded_R, R, out=self._loaded_R)
        
        try:
            R_inv = la.inv(self._loaded_R)
        except la.LinAlgError:
            self.set_uniform_weights()
            return
            
        # R^-1 @ C
        np.matmul(R_inv, C, out=self._R_inv_C[:, :K])
        
        # C^H @ R^-1 @ C
        np.conjugate(C.T, out=self._C_H[:K, :])
        np.matmul(self._C_H[:K, :], self._R_inv_C[:, :K], out=self._CH_R_inv_C[:K, :K])
        
        try:
            inner_inv = la.inv(self._CH_R_inv_C[:K, :K])
        except la.LinAlgError:
            self.set_uniform_weights()
            return
            
        # temp = inner_inv @ f
        np.matmul(inner_inv, f, out=self._temp_vector[:K])
        
        # w = (R^-1 @ C) @ temp
        np.matmul(self._R_inv_C[:, :K], self._temp_vector[:K], out=self.weights)

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
    
    # 1. Verify Inverse Noise Power MRC Weights
    print("[*] Verifying Inverse Noise-Power MRC Weights...")
    rng = np.random.default_rng(42)
    target_signal = rng.choice([1+1j, 1-1j, -1+1j, -1-1j], N) / np.sqrt(2.0)
    
    # Generate noise with different powers on channels
    true_noise_powers = np.array([0.2, 0.4, 0.1, 0.8], dtype=np.float32)
    noise = np.zeros((M, N), dtype=np.complex64)
    for c in range(M):
        p = true_noise_powers[c]
        noise[c, :] = (rng.normal(0, np.sqrt(p / 2), N) + 
                       1j * rng.normal(0, np.sqrt(p / 2), N)).astype(np.complex64)
                       
    aligned_stride = np.zeros((M, N), dtype=np.complex64)
    for c in range(M):
        aligned_stride[c, :] = target_signal + noise[c, :]
        
    synthesizer.update_mrc_weights_from_data(aligned_stride)
    
    print(" [>] True Noise Powers:      ", true_noise_powers)
    print(" [>] Estimated Noise Powers: ", synthesizer.noise_powers)
    print(" [>] Calculated MRC Weights: ", synthesizer.weights.real)
    
    # Expected weights based on inverse noise powers:
    inv_powers = 1.0 / true_noise_powers
    expected_weights = inv_powers / np.sum(inv_powers)
    print(" [>] Expected MRC Weights:   ", expected_weights)
    
    assert np.allclose(synthesizer.noise_powers, true_noise_powers, atol=0.05), "Noise power estimation error too high."
    assert np.allclose(synthesizer.weights.real, expected_weights, atol=0.05), "MRC weights calculation mismatch."
    
    # 2. Verify Grating Lobe Suppression using LCMV
    print("\n[*] Verifying LCMV Grating Lobe & Anti-Aliasing Protection Mask...")
    target_sv = np.ones(M, dtype=np.complex64) # Target at broadside
    grating_lobe_angle = 45.0
    grating_lobe_sv = np.exp(-1j * np.pi * np.sin(np.radians(grating_lobe_angle)) * np.arange(M)).astype(np.complex64)
    
    # Construct C and f
    C = np.zeros((M, 2), dtype=np.complex64)
    C[:, 0] = target_sv
    C[:, 1] = grating_lobe_sv
    f = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex64) # Target=1.0, Grating Lobe=0.0
    
    # Environment covariance matrix (thermal noise only)
    R = np.eye(M, dtype=np.complex64)
    
    synthesizer.update_lcmv_weights(R, C, f)
    w_lcmv = synthesizer.weights
    
    target_response = np.abs(np.vdot(w_lcmv, target_sv))
    grating_lobe_response = np.abs(np.vdot(w_lcmv, grating_lobe_sv))
    
    print(f" [>] Target Response (Gain):        {target_response:.4f} (Expected: 1.0)")
    print(f" [>] Grating Lobe Response (Gain):  {grating_lobe_response:.6e} (Expected: 0.0)")
    
    # 3. Latency Check
    latencies = []
    for _ in range(1000):
        _, us = synthesizer.synthesize(aligned_stride)
        latencies.append(us)
        
    avg_latency = np.mean(latencies)
    import sys
    is_rt_capable = sys.platform.startswith('linux')
    latency_limit = 12.0 if is_rt_capable else 25.0
    print(f"\n [>] Average Synthesis Latency: {avg_latency:.4f} us (Limit: <{latency_limit:.1f} us)")
    
    # Assertions
    assert abs(target_response - 1.0) < 1e-5, "LCMV target response not unity."
    assert grating_lobe_response < 1e-5, "Grating lobe not nulled."
    assert avg_latency < latency_limit, "Synthesis latency exceeded limit."
    
    print("\n[PASSED] Coherent Aperture Synthesizer compiled and verified successfully!")
    print("==================================================================")
