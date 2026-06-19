"""
Task 38.2: SpaceShield Array Manifold Self-Healer
Adaptive Spatial Signature Calibration Engine

Utilizes localized Subspace Projection mapping to perform coordinate-free 
steering vector self-healing. Restores degraded antenna phase centers dynamically
by projecting nominal target signatures strictly onto the empirical signal subspace.
"""

import numpy as np
from numba import njit, prange

@njit(parallel=False, fastmath=True, boundscheck=False, cache=True)
def _project_steering_vectors(
    V_pool: np.ndarray,
    lambdas_pool: np.ndarray,
    nominal_steers: np.ndarray,
    corrected_steers: np.ndarray,
    num_anomalies: int,
    snr_threshold: float
):
    """
    Constructs an orthogonal projector P_sig per frequency anomaly.
    Evaluates subspace dimensionality dynamically via SNR bounds, then maps
    and normalizes the nominal vectors to eliminate component drift.
    """
    for b in range(num_anomalies):
        # 1. Estimate Number of Signals (K) via Eigenvalue Thresholding
        noise_floor = (lambdas_pool[b, 2] + lambdas_pool[b, 3]) / 2.0
        
        K = 0
        for i in range(4):
            if lambdas_pool[b, i] > snr_threshold * noise_floor:
                K += 1
                
        # If no signal space is detected or subspace covers everything, bypass
        if K == 0 or K == 4:
            for c in range(4):
                corrected_steers[b, c] = nominal_steers[b, c]
            continue
            
        # 2. Orthogonal Projector P_sig = V_s * V_s^H applied to a_nom
        # a_corr = V_s * (V_s^H * a_nom)
        for c in range(4):
            corrected_steers[b, c] = 0.0 + 0j
            
        # Inner product (V_s^H * a_nom) for each signal subspace vector
        for k in range(K):
            inner_prod = 0.0 + 0j
            for i in range(4):
                # V^H is conjugate transpose -> conj(V[i, k])
                inner_prod += np.conj(V_pool[b, i, k]) * nominal_steers[b, i]
                
            # Outer multiplication V_s * inner_prod
            for i in range(4):
                corrected_steers[b, i] += V_pool[b, i, k] * inner_prod
                
        # 3. Normalization to maintain array gain constraints
        norm_sq = np.float32(0.0)
        for i in range(4):
            val = corrected_steers[b, i]
            norm_sq += np.float32(val.real * val.real + val.imag * val.imag)
            
        if norm_sq > np.float32(1e-12):
            inv_norm = np.float32(1.0 / np.sqrt(norm_sq))
            for i in range(4):
                corrected_steers[b, i] *= inv_norm
        else:
            for i in range(4):
                corrected_steers[b, i] = nominal_steers[b, i]


class ManifoldSelfHealer:
    """
    SpaceShield Array Calibration Engine.
    Operates sequentially after the SVD Subspace Clipper. Extracts the dominant
    spatial eigenvectors and eliminates physical hardware mismatch by projecting 
    theoretical Keplerian steering vectors deeply into the true signal subspace.
    """
    def __init__(self, channels: int = 4, max_anomalies: int = 64, snr_db_threshold: float = 10.0):
        self.channels = channels
        self.max_anomalies = max_anomalies
        self.snr_threshold = 10.0 ** (snr_db_threshold / 10.0)
        
        # Zero-allocation coordinate buffers
        self.corrected_steer_pool = np.zeros((self.max_anomalies, self.channels), dtype=np.complex64)
        
        self._warmup()
        
    def _warmup(self):
        """Forces LLVM compilation trace"""
        dummy_v = np.zeros((1, self.channels, self.channels), dtype=np.complex64)
        dummy_l = np.ones((1, self.channels), dtype=np.float32)
        dummy_s = np.ones((1, self.channels), dtype=np.complex64)
        
        _project_steering_vectors(
            dummy_v, dummy_l, dummy_s, self.corrected_steer_pool, 1, self.snr_threshold
        )
        
    def heal_manifold(self, V_pool: np.ndarray, lambdas_pool: np.ndarray, nominal_steers: np.ndarray, num_active: int) -> np.ndarray:
        """
        Takes raw SVD eigenvector pools and theoretically computed nominal steering vectors,
        correcting physical mismatch via orthogonal signal-subspace projections.
        """
        if num_active > self.max_anomalies:
            raise ValueError("Anomalies exceed Manifold Self-Healer bounds.")
            
        if num_active == 0:
            return self.corrected_steer_pool[:0]
            
        _project_steering_vectors(
            V_pool, lambdas_pool, nominal_steers, self.corrected_steer_pool, 
            num_active, self.snr_threshold
        )
        
        return self.corrected_steer_pool[:num_active]
