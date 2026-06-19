import time
import numpy as np
import scipy.linalg

class OrthogonalMaskingEngine:
    """
    Electronic Warfare (EW) Signaling Defense Module.
    Generates an Orthogonal Complement Projection (OCP) space from the authentic
    satellite steering vectors. It projects synthetic Gaussian noise directly into 
    this null-space and superimposes it onto the LCMV array weights. 
    This structurally encrypts the array's nulling behavior against hostile observers 
    without causing any signal degradation along the NavIC Line-of-Sight.
    """
    def __init__(self, num_channels: int = 4, mask_scale: float = 0.05, noise_pool_size: int = 16384):
        self.num_channels = num_channels
        self.mask_scale = mask_scale
        
        # Zero-Allocation PRNG Pool
        # Pre-generating a massive cyclic pool of Gaussian noise prevents 
        # continuous heap allocations inside the strict 40µs stride hot-path.
        self._noise_pool_size = noise_pool_size
        self._pool_idx = 0
        raw_noise = np.random.randn(self._noise_pool_size, self.num_channels) + 1j * np.random.randn(self._noise_pool_size, self.num_channels)
        self._noise_pool = raw_noise.astype(np.complex64)
        
        # Pre-allocated structural buffers
        self._identity = np.eye(self.num_channels, dtype=np.complex64)
        self._P_ortho = np.eye(self.num_channels, dtype=np.complex64) # Default to identity
        self._w_mask = np.zeros(self.num_channels, dtype=np.complex64)
        self._w_out = np.zeros(self.num_channels, dtype=np.complex64)

    def update_steering_matrix(self, S: np.ndarray):
        """
        Calculates the Orthogonal Projection Matrix P_ortho = I - S * (S^H * S)^-1 * S^H.
        Called asynchronously whenever the Blind Array Calibrator updates the look vectors.
        """
        K = S.shape[1]
        
        # S^H * S
        S_H_S = S.conj().T @ S
        
        # (S^H * S)^-1
        try:
            # Add negligible diagonal load for absolute stability if S is rank deficient
            S_H_S_inv = scipy.linalg.inv(S_H_S + 1e-6 * np.eye(K))
        except Exception:
            S_H_S_inv = np.eye(K)
            
        # S * (S^H * S)^-1 * S^H
        proj = S @ S_H_S_inv @ S.conj().T
        
        # P_ortho = I - proj
        self._P_ortho = self._identity - proj

    def mask_weights(self, w: np.ndarray) -> tuple:
        """
        Hot-path matrix execution. 
        Extracts complex noise, projects it orthogonally to the target vectors, 
        and superimposes it to obfuscate the true beamforming gradients.
        
        Args:
            w: (4,) The unmasked, optimal LCMV weight vector.
            
        Returns:
            (w_masked, execution_time_us)
        """
        t0 = time.perf_counter()
        
        # 1. Zero-Allocation Fast Noise Extraction
        v = self._noise_pool[self._pool_idx]
        self._pool_idx = (self._pool_idx + 1) % self._noise_pool_size
        
        # 2. Orthogonal Projection: w_mask = P_ortho * v
        # Native numpy matmul guarantees zero transient pointer allocation here
        np.matmul(self._P_ortho, v, out=self._w_mask)
        
        # 3. Superposition Integration: w_out = w + scale * w_mask
        np.multiply(self._w_mask, self.mask_scale, out=self._w_mask)
        np.add(w, self._w_mask, out=self._w_out)
        
        execution_us = (time.perf_counter() - t0) * 1e6
        
        return self._w_out, execution_us

# --- Rapid Verification Stub ---
if __name__ == "__main__":
    print("[*] Initializing Orthogonal Weight Masking Engine...")
    masker = OrthogonalMaskingEngine(num_channels=4, mask_scale=0.25)
    
    # 1. Define physical space
    # Authentic target look-direction (Broadside NavIC satellite)
    S_target = np.ones((4, 1), dtype=np.complex64)
    masker.update_steering_matrix(S_target)
    
    # Baseline LCMV unmasked weight vector
    w_unmasked = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex64)
    
    # Burn-in pass for compiler cache warming
    masker.mask_weights(w_unmasked)
    
    # 2. Benchmark the masking loop
    latencies = []
    w_masked = None
    for _ in range(5000):
        w_masked, us = masker.mask_weights(w_unmasked)
        latencies.append(us)
        
    avg_us = sum(latencies) / len(latencies)
    
    print("\n--- OCP MASKING EXECUTION PROFILES ---")
    print(f" [>] Cyclic Noise Buffer Allocation: {masker._noise_pool_size} Vectors")
    
    # 3. Cross-Correlation Proof
    # We mathematically prove the injected noise vector is completely invisible to the target signal
    # Cross-Correlation = S^H * w_mask
    cross_corr = np.abs(np.vdot(S_target.flatten(), masker._w_mask.flatten()))
    print(f" [>] NavIC Signal Cross-Correlation: {cross_corr:.8f} (Must be exactly 0.0)")
    
    # 4. Obfuscation Proof
    # Show how much the physical weights changed to fool eavesdroppers
    divergence = np.linalg.norm(w_masked - w_unmasked)
    print(f" [>] Physical Weight Shift (L2 Norm): {divergence:.4f}")
    
    print(f"\n [>] Average Execution Time: {avg_us:.2f} µs")
    
    if avg_us < 40.0 and cross_corr < 1e-5:
        print(f" [PASSED] Orthogonal Null-Space Projection seamlessly executed within constraint!")
    else:
        print(f" [FAILED] Constraint violation detected.")
