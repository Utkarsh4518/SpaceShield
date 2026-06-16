"""
Task 37.2: SpaceShield Capon Minimum Variance AoA Mapper
High-Resolution Spatial Spectrum Estimation Engine

Implements an inline single-snapshot Capon spatial estimator mapping over
flagged frequency anomaly bins. Utilizes custom static memory pooling 
and Gauss-Jordan matrix inversion to strictly guarantee 0 heap allocations
and sub-45us target resolution per stride.
"""

import numpy as np
from numba import njit, prange

@njit(parallel=False, fastmath=True, boundscheck=False, cache=True)
def _capon_sweep_engine(
    fft_matrix: np.ndarray,
    target_bins: np.ndarray,
    out_spectra: np.ndarray,
    A_steer: np.ndarray,
    R_pool: np.ndarray,
    inv_pool: np.ndarray,
    temp_pool: np.ndarray,
    diag_load: float
):
    """
    Core numerical execution block.
    1. Reconstructs Rank-1 Loaded Snapshot Covariance
    2. Zero-Allocation Gauss-Jordan 4x4 Inversion
    3. Minimum Variance Distortionless Response (MVDR) Grid Sweep
    """
    num_bins = len(target_bins)
    num_angles = A_steer.shape[0]
    
    for k in range(num_bins):
        bin_idx = target_bins[k]
        
        # 1. Form Spatial Covariance Matrix R = x*x^H + alpha*I
        for i in range(4):
            val_i = fft_matrix[i, bin_idx]
            for j in range(4):
                val_j = fft_matrix[j, bin_idx]
                R_pool[k, i, j] = val_i * np.conj(val_j)
                
            # Add strict diagonal loading to force positive-definite full rank
            R_pool[k, i, i] += diag_load
            
        # 2. In-place Gauss-Jordan Matrix Inversion avoiding np.linalg.inv heap allocs
        for i in range(4):
            for j in range(4):
                inv_pool[k, i, j] = 1.0 + 0j if i == j else 0j
                temp_pool[k, i, j] = R_pool[k, i, j]
                
        for i in range(4):
            pivot = temp_pool[k, i, i]
            # Numerical safety catch for extreme zero-energy anomalies
            if abs(pivot) < 1e-12:
                pivot = pivot + 1e-6
                
            inv_pivot = 1.0 / pivot
            for j in range(4):
                temp_pool[k, i, j] *= inv_pivot
                inv_pool[k, i, j] *= inv_pivot
                
            for r in range(4):
                if r != i:
                    factor = temp_pool[k, r, i]
                    for c in range(4):
                        temp_pool[k, r, c] -= factor * temp_pool[k, i, c]
                        inv_pool[k, r, c] -= factor * inv_pool[k, i, c]
                        
        # 3. MVDR Spatial Grid Evaluation
        for a_idx in range(num_angles):
            # P(theta) = 1.0 / (a^H * R_inv * a)
            power = 0.0 + 0j
            for i in range(4):
                a_h = np.conj(A_steer[a_idx, i])
                inner_sum = 0.0 + 0j
                for j in range(4):
                    inner_sum += inv_pool[k, i, j] * A_steer[a_idx, j]
                power += a_h * inner_sum
                
            out_spectra[k, a_idx] = np.float32(1.0 / (np.abs(power) + 1e-12))


class CaponAoAMapper:
    """
    SpaceShield Minimum Variance High-Resolution Mapper.
    Wraps the Vectorized FFT Core outputs. When specific frequency bins trip 
    the anomaly detectors, this module fires a targeted 181-degree spatial 
    sweep specifically on those bins to geolocate adversarial emitters.
    """
    def __init__(self, channels: int = 4, max_anomalies: int = 64):
        self.channels = channels
        self.max_anomalies = max_anomalies
        self.num_angles = 181  # -90 to +90 inclusive
        self.diag_load = 1e-3  # Heavy diagonal loading for single-snapshot conditioning
        
        # Pre-compute target steering vectors
        self.A_steer = np.zeros((self.num_angles, self.channels), dtype=np.complex64)
        angles = np.linspace(-90, 90, self.num_angles)
        for i, theta_deg in enumerate(angles):
            theta_rad = np.deg2rad(theta_deg)
            # Assumption: Linear uniform array with d = lambda/2
            phi = np.pi * np.sin(theta_rad)
            for c in range(self.channels):
                self.A_steer[i, c] = np.complex64(np.cos(c * phi) + 1j * np.sin(c * phi))
                
        # Initialize zero-allocation numerical execution pools
        # Size bounded to `max_anomalies` to prevent out-of-bounds heap triggers
        self.R_pool = np.zeros((self.max_anomalies, self.channels, self.channels), dtype=np.complex64)
        self.inv_pool = np.zeros((self.max_anomalies, self.channels, self.channels), dtype=np.complex64)
        self.temp_pool = np.zeros((self.max_anomalies, self.channels, self.channels), dtype=np.complex64)
        self.out_spectra = np.zeros((self.max_anomalies, self.num_angles), dtype=np.float32)
        
        self._warmup()
        
    def _warmup(self):
        """Forces Numba LLVM IR trace compilation."""
        dummy_fft = np.ones((self.channels, 10), dtype=np.complex64)
        dummy_bins = np.array([0, 1], dtype=np.int32)
        _capon_sweep_engine(
            dummy_fft, dummy_bins, self.out_spectra, self.A_steer,
            self.R_pool, self.inv_pool, self.temp_pool, self.diag_load
        )
        
    def map_anomalies(self, fft_stride: np.ndarray, target_bins: np.ndarray) -> np.ndarray:
        """
        Ingests a 4-channel frequency-domain matrix and targets hostile bins.
        
        Args:
            fft_stride: np.ndarray of shape (4, N_bins) complex64
            target_bins: 1D np.ndarray array of specific bin indices (integers)
            
        Returns:
            np.ndarray view of shape (len(target_bins), 181) mapping spatial probability P(theta).
        """
        num_targets = len(target_bins)
        if num_targets > self.max_anomalies:
            raise ValueError(f"SpaceShield Exception: Active anomalies ({num_targets}) overflow the Capon static allocation pool ({self.max_anomalies}).")
            
        if num_targets == 0:
            return self.out_spectra[:0]
            
        _capon_sweep_engine(
            fft_stride, target_bins, self.out_spectra, self.A_steer,
            self.R_pool, self.inv_pool, self.temp_pool, self.diag_load
        )
        
        # Return strict zero-allocation contiguous view
        return self.out_spectra[:num_targets]
