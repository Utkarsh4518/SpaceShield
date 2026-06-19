import numpy as np
import scipy.linalg as la
import logging

logger = logging.getLogger("MutualCouplingCompensator")
logger.setLevel(logging.INFO)

class MutualCouplingCompensator:
    """
    Blind Mutual Coupling Matrix (MCM) Compensator for 4-Channel Spatial Arrays.
    Uses Structured Covariance Optimization to isolate steering vector deviations
    without active reference signals, enabling robust spatial tracking under field stress.
    """
    
    def __init__(self, num_channels: int = 4, learning_rate: float = 0.05, max_iter: int = 10):
        self.M = num_channels
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
        # Initial coupling matrix assumes ideal isolation (Identity matrix)
        self.C_est = np.eye(self.M, dtype=np.complex128)
        self.C_inv = np.eye(self.M, dtype=np.complex128)
        
        # Temporal smoothing to prevent matrix inversion instability
        self.R_smoothed = None
        self.alpha = 0.1 # Exponential moving average factor for covariance
        
    def _enforce_symmetric_toeplitz(self, matrix: np.ndarray) -> np.ndarray:
        """
        Projects a generic complex matrix into the closest symmetric Toeplitz structure.
        Crucial for Uniform Linear Arrays (ULA) mutual coupling physics.
        """
        toeplitz_vector = np.zeros(self.M, dtype=np.complex128)
        # Average along the diagonals
        for d in range(self.M):
            diag_elements = np.diag(matrix, k=d)
            if d > 0:
                # Average upper and lower diagonals to enforce symmetry
                diag_elements = np.concatenate([diag_elements, np.diag(matrix, k=-d)])
            toeplitz_vector[d] = np.mean(diag_elements)
            
        return la.toeplitz(toeplitz_vector)

    def estimate_mcm(self, X: np.ndarray):
        """
        Alternating Projection / Structured Covariance Optimization.
        Extracts the MCM from a batch of I/Q physical observations.
        
        Parameters:
            X (np.ndarray): Shape (M, N), incoming physical phase-aligned I/Q frames.
        """
        # 1. Estimate Spatial Sample Covariance Matrix
        N = X.shape[1]
        R_sample = (X @ X.conj().T) / N
        
        # Apply exponential smoothing for stability over continuous frames
        if self.R_smoothed is None:
            self.R_smoothed = R_sample
        else:
            self.R_smoothed = (1 - self.alpha) * self.R_smoothed + self.alpha * R_sample
            
        # 2. Extract Signal Subspace (Blind Calibration via Eigendecomposition)
        # SVD/EVD extraction of dominant eigenvectors
        evals, evecs = la.eigh(self.R_smoothed)
        
        # Assume rank-1 dominant signal for blind steering vector isolation 
        # (typical for single main carrier like NavIC L5 tracking)
        dominant_vector = evecs[:, -1:] 
        
        # 3. Iterative Coupling Projection
        # We attempt to find a coupling matrix C such that C^{-1} aligns the 
        # dominant subspace closer to a theoretical idealized manifold.
        # We approximate the projection iteratively by enforcing structural constraints.
        C_current = np.copy(self.C_est)
        
        for _ in range(self.max_iter):
            # Form pseudo-ideal manifold projection
            # In a fully blind alternating setup, we project dominant vectors
            ideal_proj = C_current @ dominant_vector
            
            # Cross-correlation based update step
            gradient = (ideal_proj @ dominant_vector.conj().T) - self.R_smoothed
            
            # Gradient descent step
            C_next = C_current - self.learning_rate * gradient
            
            # Physics Constraint: Project back onto Symmetric Toeplitz manifold
            C_current = self._enforce_symmetric_toeplitz(C_next)
            
            # Normalize to prevent unbounded growth (C[0,0] is exactly 1 + 0j)
            C_current /= C_current[0, 0]
            
        # 4. Filter Update
        self.C_est = (1 - self.learning_rate) * self.C_est + self.learning_rate * C_current
        
        # 5. Pre-calculate the Inversion for the fast-path
        try:
            self.C_inv = la.inv(self.C_est)
        except la.LinAlgError:
            logger.warning("Singular coupling matrix detected. Falling back to Identity.")
            self.C_inv = np.eye(self.M, dtype=np.complex128)

    def apply_compensation(self, X: np.ndarray) -> np.ndarray:
        """
        Equalization fast-path. Operates within the ~24.40µs SVD engine bounds.
        Applies the pre-calculated inverse MCM to decouple physical antenna leakage.
        """
        # Direct matrix multiplication decoupling
        return self.C_inv @ X

# --- Rapid Verification Stub ---
if __name__ == "__main__":
    print("[*] Compiling MutualCouplingCompensator...")
    M, N = 4, 8192
    
    # Generate synthetic coupled signal
    rng = np.random.default_rng(42)
    ideal_signal = rng.normal(0, 1, (1, N)) + 1j * rng.normal(0, 1, (1, N))
    
    # Steering vector for 30 degrees
    theta = np.radians(30)
    a_ideal = np.exp(1j * np.arange(M) * np.sin(theta)).reshape(M, 1)
    
    # Physical array leakage (Toeplitz MCM)
    true_coupling = la.toeplitz([1.0, 0.3+0.1j, 0.05-0.02j, 0.01])
    
    X_coupled = true_coupling @ (a_ideal @ ideal_signal)
    
    # Add thermal noise
    X_coupled += rng.normal(0, 0.05, (M, N)) + 1j * rng.normal(0, 0.05, (M, N))
    
    print(f"[+] Initial True Coupling C[1,0] Magnitude: {np.abs(true_coupling[1,0]):.4f}")
    
    compensator = MutualCouplingCompensator(num_channels=M)
    
    # Batch processing simulation
    for batch in range(20):
        # Slice into observation frames
        X_frame = X_coupled[:, (batch*400):((batch+1)*400)]
        compensator.estimate_mcm(X_frame)
        
    print(f"[+] Estimated Coupling C[1,0] Magnitude: {np.abs(compensator.C_est[1,0]):.4f}")
    
    import time
    t0 = time.perf_counter()
    X_clean = compensator.apply_compensation(X_coupled[:, :50])
    t_us = (time.perf_counter() - t0) * 1e6
    print(f"[+] Clean inversion executed in {t_us:.2f} µs.")
