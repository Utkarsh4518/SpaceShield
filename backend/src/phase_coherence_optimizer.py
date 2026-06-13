import numpy as np
import logging

logger = logging.getLogger("PhaseCoherenceOptimizer")
logger.setLevel(logging.INFO)

class PhaseCoherenceOptimizer:
    """
    High-Throughput Phase & Amplitude Array Calibration Engine.
    Continuously normalizes multi-channel phase offsets and amplitude imbalances
    against a defined reference channel (Channel 0) without triggering garbage collection.
    """
    def __init__(self, num_channels: int = 4, alpha: float = 0.05):
        """
        Initializes the optimizer with pre-allocated zero-allocation buffers.
        
        Parameters:
            num_channels (int): Antenna array channel count (M).
            alpha (float): Exponential moving average smoothing factor.
        """
        self.M = num_channels
        self.alpha = alpha
        
        # Pre-allocated structural buffers
        # Complex correction coefficients: (M, 1) broadcastable vector
        self.correction_coeffs = np.ones((self.M, 1), dtype=np.complex64)
        
        # Scratch buffers for calculation to avoid runtime allocations
        self._ch0_conj = None
        self._cross_corr = np.zeros(self.M, dtype=np.complex64)
        self._ref_power = 0.0
        
    def update_coefficients(self, X: np.ndarray):
        """
        Block-Averaged Least-Squares Estimation (Slow-Path Update).
        Estimates the relative inter-channel amplitude and phase error relative to Channel 0.
        
        Parameters:
            X (np.ndarray): Shape (M, N), incoming physical phase-aligned I/Q frames.
        """
        N = X.shape[1]
        
        # Allocate scratch buffers dynamically upon first frame size lock
        if self._ch0_conj is None or self._ch0_conj.shape[0] != N:
            self._ch0_conj = np.zeros(N, dtype=np.complex64)
            
        # 1. Isolate Reference Channel (Channel 0)
        # We calculate X[i] * conj(X[0]) for the cross-correlation phase difference.
        np.conjugate(X[0, :], out=self._ch0_conj)
        
        # Reference power for normalization (amplitude calibration)
        self._ref_power = np.mean(X[0, :].real**2 + X[0, :].imag**2) + 1e-12
        
        # 2. Block-Averaged Correlation Estimate
        for i in range(1, self.M):
            # Compute cross-correlation E[X_i * X_0^*]
            # Fast vectorized dot product
            cross_val = np.dot(X[i, :], self._ch0_conj) / N
            
            # The ideal complex correction coefficient to align X[i] to X[0]:
            # w_i = conj(cross_val) / ref_power
            # We track the inverse scaling required to normalize incoming arrays
            target_coeff = np.conj(cross_val) / self._ref_power
            
            # 3. Rolling Exponential Smoothing
            # Smoothly transition coefficients to avoid jarring phase jumps in the tracker
            self.correction_coeffs[i, 0] = (1.0 - self.alpha) * self.correction_coeffs[i, 0] + self.alpha * target_coeff
            
        # Note: Channel 0 is the reference, so its coefficient remains 1.0 + 0j
        self.correction_coeffs[0, 0] = 1.0 + 0.0j

    def apply_compensation(self, X: np.ndarray):
        """
        Ultra-Fast Execution Pipeline (Hot-Path).
        Applies the complex correction array entirely in-place to normalize the steering vector.
        
        Parameters:
            X (np.ndarray): Shape (M, N) complex64 data. Modifies in-place.
        """
        # Element-wise broadcasting multiplication in-place (ZERO memory allocations)
        X *= self.correction_coeffs


# --- Rapid Verification Stub ---
if __name__ == "__main__":
    import time
    
    print("[*] Testing PhaseCoherenceOptimizer...")
    M, N = 4, 4096
    
    optimizer = PhaseCoherenceOptimizer(num_channels=M)
    
    # Generate authentic signal
    rng = np.random.default_rng(42)
    t = np.arange(N)
    # Carrier
    s = np.exp(1j * 2 * np.pi * 0.05 * t).astype(np.complex64)
    
    # Steering Vector with Random Static Hardware Phase Errors and Gain Imbalances
    # Channel 0 is reference: gain=1.0, phase=0.0
    true_gains = np.array([1.0, 0.8, 1.2, 0.9])
    true_phases = np.array([0.0, np.pi/4, -np.pi/3, np.pi/6])
    
    # Construct base array view
    X_raw = np.empty((M, N), dtype=np.complex64)
    for i in range(M):
        error_coeff = true_gains[i] * np.exp(1j * true_phases[i])
        X_raw[i, :] = s * error_coeff
        
    # Add thermal noise
    X_raw += (rng.normal(0, 0.05, (M, N)) + 1j * rng.normal(0, 0.05, (M, N))).astype(np.complex64)
    
    print(f"[+] Initial Hardware Phase Imbalance (Ch1 vs Ch0): {true_phases[1]:.4f} rad")
    
    # Execute batch convergence
    for _ in range(50):
        optimizer.update_coefficients(X_raw)
        
    est_phase_correction = np.angle(optimizer.correction_coeffs[1, 0])
    print(f"[+] Estimated Correction Phase for Ch1: {est_phase_correction:.4f} rad (Should approximate {-true_phases[1]:.4f})")
    
    # Measure Fast-Path Latency (In-Place Multiplication)
    t0 = time.perf_counter()
    optimizer.apply_compensation(X_raw)
    t_us = (time.perf_counter() - t0) * 1e6
    
    print(f"[+] Zero-Allocation In-Place execution time: {t_us:.2f} µs for ({M}, {N}) blocks.")
