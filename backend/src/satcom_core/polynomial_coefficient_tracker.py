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
from numba import njit

# parallel=True/prange was removed: this kernel only ever runs over the fixed
# M=4 antenna array (see README Sec 1.1), and profiling showed Numba's
# thread-pool dispatch for a 4-way prange region costs ~100-130us on its own
# -- roughly 3x the actual per-channel compute. A serial njit loop measures
# ~3x faster end-to-end for this channel count. If num_channels grows well
# past the core count in a future revision, parallel=True should be
# re-evaluated against the same benchmark methodology.
@njit(fastmath=True, cache=True, boundscheck=False, parallel=False)
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
    filter_coeffs: np.ndarray    # (num_taps,) float32 -- was hardcoded to 6
):
    # Generalized from a fixed 6-tap manual unroll to an arbitrary tap count.
    # A 6-tap FIR has exactly 5 real degrees of freedom, fully consumed by 3
    # required spectral nulls (DC + a real-valued band each for the target
    # and jammer regions in tests/saturation_linearization_verifier.py costs
    # 1 + 2 + 2 = 5 DOF) -- there is provably zero freedom left to also shape
    # the passband, so any 6-tap filter meeting those nulls is forced into a
    # wildly uneven passband (measured: 30x gain spread, peaking at Nyquist).
    # This generalization was built to test the hypothesis that this
    # imbalance was the cause of that test's near-0dB IMD suppression
    # result. It wasn't: a longer, well-conditioned (near-flat passband)
    # filter measured no real improvement, and a follow-up exhaustive-ish
    # search over the model's entire 3-parameter coefficient space showed
    # the achievable ceiling is ~0.05dB regardless of filter shape or step
    # size -- a model-capacity limit (single memory tap, 2 nonlinear
    # orders), not an OOB-detector design problem. Full writeup in
    # tests/saturation_linearization_verifier.py. This generalized,
    # variable-tap kernel is kept regardless: it's correct, verified
    # bit-exact against the old 6-tap path (see
    # tests/test_polynomial_coefficient_tracker.py), and is the
    # infrastructure a future higher-order model would need anyway.
    num_channels = X_buffer.shape[0]
    stride_len = X_buffer.shape[1]
    num_taps = filter_coeffs.shape[0]

    for ch in range(num_channels):
        tot_energy = 0.0
        for n in range(stride_len):
            tot_energy += Y_buffer[ch, n].real ** 2 + Y_buffer[ch, n].imag ** 2
        total_energies[ch] = tot_energy

        oob_energy = 0.0
        for n in range(num_taps - 1, stride_len):
            e_r = 0.0
            e_i = 0.0
            for k in range(num_taps):
                hk = filter_coeffs[k]
                e_r += hk * Y_buffer[ch, n - k].real
                e_i += hk * Y_buffer[ch, n - k].imag
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
            for n in range(num_taps, stride_len, decimation):
                e_r = 0.0; e_i = 0.0
                t30_r = 0.0; t30_i = 0.0
                t31_r = 0.0; t31_i = 0.0
                t50_r = 0.0; t50_i = 0.0

                for k in range(num_taps):
                    hk = filter_coeffs[k]

                    y_r = Y_buffer[ch, n - k].real
                    y_i = Y_buffer[ch, n - k].imag
                    e_r += hk * y_r
                    e_i += hk * y_i

                    # tap-0 delay regressor (dY(n-k)/dc30, dY(n-k)/dc50): X at n-k
                    x0_r = X_buffer[ch, n - k].real
                    x0_i = X_buffer[ch, n - k].imag
                    a0_sq = x0_r * x0_r + x0_i * x0_i
                    u30_r = x0_r * a0_sq; u30_i = x0_i * a0_sq
                    u50_r = x0_r * (a0_sq * a0_sq); u50_i = x0_i * (a0_sq * a0_sq)
                    t30_r += hk * u30_r; t30_i += hk * u30_i
                    t50_r += hk * u50_r; t50_i += hk * u50_i

                    # tap-1 delay regressor (dY(n-k)/dc31): X at n-k-1
                    x1_r = X_buffer[ch, n - k - 1].real
                    x1_i = X_buffer[ch, n - k - 1].imag
                    a1_sq = x1_r * x1_r + x1_i * x1_i
                    u31_r = x1_r * a1_sq; u31_i = x1_i * a1_sq
                    t31_r += hk * u31_r; t31_i += hk * u31_i

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

    mu default lowered from 0.15 to 0.05: swept mu in [0.002, 1.0] against the
    two-tone regrowth fixture and the combined-norm block update diverges (OOB
    ratio increases past its own minimum) for any mu >= ~0.06, including the
    prior 0.15 default. 0.05 is the largest step size that stays monotonically
    convergent through the fixture's adaptation window.
    """
    def __init__(
        self,
        num_channels: int = 4,
        stride_len: int = 4096,
        mu: float = 0.05,
        threshold_ratio: float = 0.05,
        filter_coeffs: np.ndarray = None
    ):
        self.num_channels = num_channels
        self.stride_len = stride_len
        self.mu = mu
        self.threshold_ratio = threshold_ratio
        
        # filter_coeffs is now an arbitrary-length FIR (see _track_all_channels
        # for why a fixed 6-tap filter was a hard design ceiling, not just an
        # implementation detail). Any length >= 1 works; no padding needed.
        if filter_coeffs is not None:
            self.filter_coeffs = np.ascontiguousarray(filter_coeffs, dtype=np.float32)
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
