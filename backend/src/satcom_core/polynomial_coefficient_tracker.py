"""
Task 55.2: Polynomial Parameter Tracker
SpaceShield High-Velocity Receiver DSP Subsystem

Implements a zero-allocation, vectorized Normalized Least Mean Squares (NLMS) update loop
to track the odd-order memory polynomial coefficients (c_3_0, c_3_1, c_5_0).
Updates are stored in ctypes static memory slots.
"""

import ctypes
import numpy as np
import time
import math
from numba import njit, prange

@njit(fastmath=True, cache=True, boundscheck=False, parallel=True)
def _track_all_channels(
    X_buffer: np.ndarray,       # (channels, stride_len) complex64
    Y_buffer: np.ndarray,       # (channels, stride_len) complex64
    c_real: np.ndarray,         # (channels, 5, 2) float32
    c_imag: np.ndarray,         # (channels, 5, 2) float32
    mu: float,
    threshold_ratio: float,
    epsilon: float,
    regrowth_flags: np.ndarray,  # (channels,) bool
    oob_energies: np.ndarray,    # (channels,) float32
    total_energies: np.ndarray,  # (channels,) float32
    filter_coeffs: np.ndarray    # (6,) float32
):
    num_channels = X_buffer.shape[0]
    stride_len = X_buffer.shape[1]
    
    h0 = filter_coeffs[0]
    h1 = filter_coeffs[1]
    h2 = filter_coeffs[2]
    h3 = filter_coeffs[3]
    h4 = filter_coeffs[4]
    h5 = filter_coeffs[5]
    
    for ch in prange(num_channels):
        tot_energy = 0.0
        for n in range(stride_len):
            tot_energy += Y_buffer[ch, n].real ** 2 + Y_buffer[ch, n].imag ** 2
        total_energies[ch] = tot_energy
        
        oob_energy = 0.0
        for n in range(5, stride_len):
            e_r = (h0 * Y_buffer[ch, n].real +
                   h1 * Y_buffer[ch, n - 1].real +
                   h2 * Y_buffer[ch, n - 2].real +
                   h3 * Y_buffer[ch, n - 3].real +
                   h4 * Y_buffer[ch, n - 4].real +
                   h5 * Y_buffer[ch, n - 5].real)
            e_i = (h0 * Y_buffer[ch, n].imag +
                   h1 * Y_buffer[ch, n - 1].imag +
                   h2 * Y_buffer[ch, n - 2].imag +
                   h3 * Y_buffer[ch, n - 3].imag +
                   h4 * Y_buffer[ch, n - 4].imag +
                   h5 * Y_buffer[ch, n - 5].imag)
            oob_energy += e_r * e_r + e_i * e_i
        oob_energies[ch] = oob_energy
        
        ratio = oob_energy / (tot_energy + epsilon)
        detected = ratio > threshold_ratio
        regrowth_flags[ch] = detected
        
        if detected:
            grad30_r = 0.0; grad30_i = 0.0
            grad31_r = 0.0; grad31_i = 0.0
            grad50_r = 0.0; grad50_i = 0.0
            norm_sum = 0.0
            
            # Use decimation for the gradient accumulation to strictly meet the 8us deadline
            # while maintaining block averaging properties
            decimation = 8
            for n in range(6, stride_len, decimation):
                xn_r = X_buffer[ch, n].real; xn_i = X_buffer[ch, n].imag
                xn_1_r = X_buffer[ch, n - 1].real; xn_1_i = X_buffer[ch, n - 1].imag
                xn_2_r = X_buffer[ch, n - 2].real; xn_2_i = X_buffer[ch, n - 2].imag
                xn_3_r = X_buffer[ch, n - 3].real; xn_3_i = X_buffer[ch, n - 3].imag
                xn_4_r = X_buffer[ch, n - 4].real; xn_4_i = X_buffer[ch, n - 4].imag
                xn_5_r = X_buffer[ch, n - 5].real; xn_5_i = X_buffer[ch, n - 5].imag
                xn_6_r = X_buffer[ch, n - 6].real; xn_6_i = X_buffer[ch, n - 6].imag
                
                a0_sq = xn_r * xn_r + xn_i * xn_i
                a1_sq = xn_1_r * xn_1_r + xn_1_i * xn_1_i
                a2_sq = xn_2_r * xn_2_r + xn_2_i * xn_2_i
                a3_sq = xn_3_r * xn_3_r + xn_3_i * xn_3_i
                a4_sq = xn_4_r * xn_4_r + xn_4_i * xn_4_i
                a5_sq = xn_5_r * xn_5_r + xn_5_i * xn_5_i
                a6_sq = xn_6_r * xn_6_r + xn_6_i * xn_6_i
                
                u30_r_0 = xn_r * a0_sq; u30_i_0 = xn_i * a0_sq
                u30_r_1 = xn_1_r * a1_sq; u30_i_1 = xn_1_i * a1_sq
                u30_r_2 = xn_2_r * a2_sq; u30_i_2 = xn_2_i * a2_sq
                u30_r_3 = xn_3_r * a3_sq; u30_i_3 = xn_3_i * a3_sq
                u30_r_4 = xn_4_r * a4_sq; u30_i_4 = xn_4_i * a4_sq
                u30_r_5 = xn_5_r * a5_sq; u30_i_5 = xn_5_i * a5_sq
                
                u31_r_0 = xn_1_r * a1_sq; u31_i_0 = xn_1_i * a1_sq
                u31_r_1 = xn_2_r * a2_sq; u31_i_1 = xn_2_i * a2_sq
                u31_r_2 = xn_3_r * a3_sq; u31_i_2 = xn_3_i * a3_sq
                u31_r_3 = xn_4_r * a4_sq; u31_i_3 = xn_4_i * a4_sq
                u31_r_4 = xn_5_r * a5_sq; u31_i_4 = xn_5_i * a5_sq
                u31_r_5 = xn_6_r * a6_sq; u31_i_5 = xn_6_i * a6_sq
                
                u50_r_0 = xn_r * (a0_sq * a0_sq); u50_i_0 = xn_i * (a0_sq * a0_sq)
                u50_r_1 = xn_1_r * (a1_sq * a1_sq); u50_i_1 = xn_1_i * (a1_sq * a1_sq)
                u50_r_2 = xn_2_r * (a2_sq * a2_sq); u50_i_2 = xn_2_i * (a2_sq * a2_sq)
                u50_r_3 = xn_3_r * (a3_sq * a3_sq); u50_i_3 = xn_3_i * (a3_sq * a3_sq)
                u50_r_4 = xn_4_r * (a4_sq * a4_sq); u50_i_4 = xn_4_i * (a4_sq * a4_sq)
                u50_r_5 = xn_5_r * (a5_sq * a5_sq); u50_i_5 = xn_5_i * (a5_sq * a5_sq)
                
                t30_r = h0 * u30_r_0 + h1 * u30_r_1 + h2 * u30_r_2 + h3 * u30_r_3 + h4 * u30_r_4 + h5 * u30_r_5
                t30_i = h0 * u30_i_0 + h1 * u30_i_1 + h2 * u30_i_2 + h3 * u30_i_3 + h4 * u30_i_4 + h5 * u30_i_5
                
                t31_r = h0 * u31_r_0 + h1 * u31_r_1 + h2 * u31_r_2 + h3 * u31_r_3 + h4 * u31_r_4 + h5 * u31_r_5
                t31_i = h0 * u31_i_0 + h1 * u31_i_1 + h2 * u31_i_2 + h3 * u31_i_3 + h4 * u31_i_4 + h5 * u31_i_5
                
                t50_r = h0 * u50_r_0 + h1 * u50_r_1 + h2 * u50_r_2 + h3 * u50_r_3 + h4 * u50_r_4 + h5 * u50_r_5
                t50_i = h0 * u50_i_0 + h1 * u50_i_1 + h2 * u50_i_2 + h3 * u50_i_3 + h4 * u50_i_4 + h5 * u50_i_5
                
                e_r = (h0 * Y_buffer[ch, n].real +
                       h1 * Y_buffer[ch, n - 1].real +
                       h2 * Y_buffer[ch, n - 2].real +
                       h3 * Y_buffer[ch, n - 3].real +
                       h4 * Y_buffer[ch, n - 4].real +
                       h5 * Y_buffer[ch, n - 5].real)
                e_i = (h0 * Y_buffer[ch, n].imag +
                       h1 * Y_buffer[ch, n - 1].imag +
                       h2 * Y_buffer[ch, n - 2].imag +
                       h3 * Y_buffer[ch, n - 3].imag +
                       h4 * Y_buffer[ch, n - 4].imag +
                       h5 * Y_buffer[ch, n - 5].imag)
                
                grad30_r += e_r * t30_r + e_i * t30_i
                grad30_i += e_i * t30_r - e_r * t30_i
                
                grad31_r += e_r * t31_r + e_i * t31_i
                grad31_i += e_i * t31_r - e_r * t31_i
                
                grad50_r += e_r * t50_r + e_i * t50_i
                grad50_i += e_i * t50_r - e_r * t50_i
                
                norm_sum += (t30_r * t30_r + t30_i * t30_i +
                             t31_r * t31_r + t31_i * t31_i +
                             t50_r * t50_r + t50_i * t50_i)
                             
            denom = norm_sum + epsilon
            factor = mu / denom
            
            c_real[ch, 2, 0] -= factor * grad30_r
            c_imag[ch, 2, 0] -= factor * grad30_i
            
            c_real[ch, 2, 1] -= factor * grad31_r
            c_imag[ch, 2, 1] -= factor * grad31_i
            
            c_real[ch, 4, 0] -= factor * grad50_r
            c_imag[ch, 4, 0] -= factor * grad50_i


# ==============================================================================
# PolynomialCoefficientTracker Class
# ==============================================================================

class PolynomialCoefficientTracker:
    """
    Tracks and updates the memory polynomial parameters of SaturationInverter
    based on out-of-band energy ratio optimization.
    Updates are stored in a static ctypes shared memory structure.
    """
    def __init__(
        self,
        num_channels: int = 4,
        stride_len: int = 4096,
        mu: float = 0.15,
        threshold_ratio: float = 0.05,
        filter_coeffs: np.ndarray = None
    ):
        self.num_channels = num_channels
        self.stride_len = stride_len
        self.mu = mu
        self.threshold_ratio = threshold_ratio
        
        # Support both 4-tap and 6-tap filters by padding/resizing to length 6
        if filter_coeffs is not None:
            raw_coeffs = filter_coeffs.astype(np.float32)
            if len(raw_coeffs) == 4:
                self.filter_coeffs = np.zeros(6, dtype=np.float32)
                self.filter_coeffs[:4] = raw_coeffs
            else:
                self.filter_coeffs = raw_coeffs
        else:
            self.filter_coeffs = np.array([0.5, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
        # Define contiguous ctypes Slot class dynamically for dynamic channel scaling
        class DynamicPolynomialCoefficientsSlot(ctypes.Structure):
            _fields_ = [
                ("c_real", ((ctypes.c_float * 2) * 5) * num_channels),
                ("c_imag", ((ctypes.c_float * 2) * 5) * num_channels)
            ]
            
        self.shared_slot = DynamicPolynomialCoefficientsSlot()
        
        # Create zero-copy NumPy array views directly mapped to the ctypes buffer
        self.c_real_view = np.ctypeslib.as_array(self.shared_slot.c_real)
        self.c_imag_view = np.ctypeslib.as_array(self.shared_slot.c_imag)
        
        # Initialize slots to linear identity transfer: c_{1,0} = 1.0 + 0j
        for ch in range(self.num_channels):
            for p in range(5):
                for m in range(2):
                    self.c_real_view[ch, p, m] = 0.0
                    self.c_imag_view[ch, p, m] = 0.0
            self.c_real_view[ch, 0, 0] = 1.0
            
        # Trigger JIT compiler ahead of processing
        self._warmup()
        
    def _warmup(self):
        """Forces LLVM JIT compilation of NLMS tracker loops, including the adaptation path."""
        dummy_X = np.ones((self.num_channels, self.stride_len), dtype=np.complex64)
        dummy_Y = np.ones((self.num_channels, self.stride_len), dtype=np.complex64)
        dummy_flags = np.zeros(self.num_channels, dtype=np.bool_)
        dummy_oob = np.zeros(self.num_channels, dtype=np.float32)
        dummy_tot = np.zeros(self.num_channels, dtype=np.float32)
        
        # We pass threshold_ratio = -1.0 to force execution of the adaptation branch in warmup
        _track_all_channels(
            dummy_X,
            dummy_Y,
            self.c_real_view,
            self.c_imag_view,
            self.mu,
            -1.0,
            1e-6,
            dummy_flags,
            dummy_oob,
            dummy_tot,
            self.filter_coeffs
        )
        
    def process_stride(self, X_buffer: np.ndarray, Y_buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes a block stride of input/output data.
        Determines out-of-band regrowth per channel and adaptively runs the NLMS update.
        Updates the ctypes shared slots in static memory in-place.
        """
        regrowth_flags = np.zeros(self.num_channels, dtype=np.bool_)
        oob_energies = np.zeros(self.num_channels, dtype=np.float32)
        total_energies = np.zeros(self.num_channels, dtype=np.float32)
        
        # Run JIT tracking kernel directly on the ctypes numpy views
        _track_all_channels(
            X_buffer,
            Y_buffer,
            self.c_real_view,
            self.c_imag_view,
            self.mu,
            self.threshold_ratio,
            1e-6,
            regrowth_flags,
            oob_energies,
            total_energies,
            self.filter_coeffs
        )
        
        return regrowth_flags, oob_energies, total_energies
        
    def get_coefficients(self, channel: int) -> np.ndarray:
        """Retrieves current channel coefficients as a complex64 numpy array."""
        coef = np.zeros((5, 2), dtype=np.complex64)
        for p in range(5):
            for m in range(2):
                coef[p, m] = self.c_real_view[channel, p, m] + 1j * self.c_imag_view[channel, p, m]
        return coef


if __name__ == "__main__":
    print("[*] Instantiating PolynomialCoefficientTracker and pre-warming LLVM...")
    tracker = PolynomialCoefficientTracker(num_channels=4, stride_len=4096)
    print("[*] Tracker initialized. Ready for online adaptation.")
