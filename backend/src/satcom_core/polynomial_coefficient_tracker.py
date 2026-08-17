"""
Task 55.2: Polynomial Parameter Tracker
SpaceShield High-Velocity Receiver DSP Subsystem

Zero-allocation Normalized Least-Mean-Squares (NLMS) tracker for the
configurable memory-polynomial coefficients of SaturationInverter:

    y[n] = sum_{p=0}^{P-1} sum_{k=0}^{K-1} c[p,k] * x[n-k] * |x[n-k]|^(2p)

It minimises the out-of-band (OOB) filtered energy J = sum_n |e[n]|^2, where
e[n] = sum_j h_j y[n-j] is a configurable FIR high/stop-band probe. The
gradient below is the exact Wirtinger gradient dJ/dc*[p,k] and is checked
numerically by tests/test_nlms_predistorter_objective.py.

Phase 3: generalised from the old fixed {c30, c31, c50} update (one effective
memory tap) to every (p, k) slot except the anchored linear/current tap
c[0,0]=1. The capacity study (see the Phase 3 report and memory_polynomial.py)
showed memory depth K, not nonlinear order P, is what unlocks real suppression:
K=1 stalls near 0 dB (its OOB optimum is counterproductive in-band), K>=3
reaches ~+10-14 dB with well-conditioned coefficients.
"""

import ctypes
import numpy as np
from numba import njit


# parallel=True/prange stays OFF: profiling showed Numba's thread-pool dispatch
# for a 4-way prange region costs ~100-130us on its own -- ~3x the actual
# per-channel compute for the fixed M=4 array. A serial njit loop is ~3x faster
# end-to-end here. Re-evaluate against the same benchmark only if num_channels
# grows well past the host core count.
@njit(fastmath=True, cache=True, boundscheck=False, parallel=False)
def _track_all_channels(
    X_buffer: np.ndarray,       # (channels, stride_len) complex64
    Y_buffer: np.ndarray,       # (channels, stride_len) complex64
    c_real: np.ndarray,         # (channels, num_orders, num_taps) float32
    c_imag: np.ndarray,         # (channels, num_orders, num_taps) float32
    mu: float,
    threshold_ratio: float,
    epsilon: float,
    decimation: int,
    regrowth_flags: np.ndarray,  # (channels,) bool
    oob_energies: np.ndarray,    # (channels,) float32
    total_energies: np.ndarray,  # (channels,) float32
    filter_coeffs: np.ndarray,   # (num_filter_taps,) float32
    tr_scratch: np.ndarray,      # (num_orders, num_taps) f32 -- regressor real
    ti_scratch: np.ndarray,      # (num_orders, num_taps) f32 -- regressor imag
    gr_scratch: np.ndarray,      # (num_orders, num_taps) f32 -- grad real
    gi_scratch: np.ndarray,      # (num_orders, num_taps) f32 -- grad imag
):
    """
    NLMS update of the memory-polynomial coefficients.

    Objective J = sum_n |e[n]|^2, e[n] = sum_j h_j Y[n-j]. The regressor for
    coefficient c[p,k] is t_{p,k}[n] = sum_j h_j * psi_{p,k}[n-j] with
    psi_{p,k}[m] = x[m-k]*|x[m-k]|^(2p). The Wirtinger gradient
    dJ/dc*[p,k] = sum_n e[n] conj(t_{p,k}[n]); NLMS normalises by the total
    regressor energy. A running power recurrence builds |x|^(2p) across the
    order loop so it is never re-powered. Verified against finite differences
    in tests/test_nlms_predistorter_objective.py.

    NOTE on latency: the dominant cost of this kernel is the out-of-band energy
    probe below -- a full FIR over the stride, O(stride_len * num_filter_taps)
    per channel, run every stride for detection. It scales with the OOB filter
    length, so the short production probe (a few taps) is ~100us on this host
    while a long analysis filter (e.g. the 65-tap firwin2 in the integration
    verifier) is proportionally slower. The decimated gradient is comparatively
    cheap.
    """
    num_channels = X_buffer.shape[0]
    stride_len = X_buffer.shape[1]
    num_filter_taps = filter_coeffs.shape[0]
    num_orders = c_real.shape[1]
    num_taps = c_real.shape[2]

    for ch in range(num_channels):
        # --- total energy of the linearized output ---
        tot_energy = 0.0
        for n in range(stride_len):
            yr = Y_buffer[ch, n].real
            yi = Y_buffer[ch, n].imag
            tot_energy += yr * yr + yi * yi
        total_energies[ch] = tot_energy

        # --- out-of-band (FIR-filtered) energy ---
        oob_energy = 0.0
        for n in range(num_filter_taps - 1, stride_len):
            e_r = 0.0
            e_i = 0.0
            for j in range(num_filter_taps):
                hj = filter_coeffs[j]
                e_r += hj * Y_buffer[ch, n - j].real
                e_i += hj * Y_buffer[ch, n - j].imag
            oob_energy += e_r * e_r + e_i * e_i
        oob_energies[ch] = oob_energy

        ratio = oob_energy / (tot_energy + epsilon)
        detected = ratio > threshold_ratio
        regrowth_flags[ch] = detected

        if not detected:
            continue

        # --- NLMS gradient over the decimated block ---
        for p in range(num_orders):
            for k in range(num_taps):
                gr_scratch[p, k] = 0.0
                gi_scratch[p, k] = 0.0
        norm_sum = 0.0

        start = num_filter_taps + num_taps  # ensures n-j-k >= 0 everywhere below
        for n in range(start, stride_len, decimation):
            # e[n] = sum_j h_j Y[n-j]  and  t_{p,k}[n] = sum_j h_j psi_{p,k}[n-j]
            e_r = 0.0
            e_i = 0.0
            for p in range(num_orders):
                for k in range(num_taps):
                    tr_scratch[p, k] = 0.0
                    ti_scratch[p, k] = 0.0
            for j in range(num_filter_taps):
                hj = filter_coeffs[j]
                m = n - j
                e_r += hj * Y_buffer[ch, m].real
                e_i += hj * Y_buffer[ch, m].imag
                for k in range(num_taps):
                    xk = X_buffer[ch, m - k]
                    xr = xk.real
                    xi = xk.imag
                    a2 = xr * xr + xi * xi
                    ap = 1.0  # |x[m-k]|^(2p), running power over the order loop
                    for p in range(num_orders):
                        tr_scratch[p, k] += hj * (xr * ap)
                        ti_scratch[p, k] += hj * (xi * ap)
                        ap *= a2

            for p in range(num_orders):
                for k in range(num_taps):
                    if p == 0 and k == 0:
                        continue
                    tr = tr_scratch[p, k]
                    ti = ti_scratch[p, k]
                    gr_scratch[p, k] += e_r * tr + e_i * ti
                    gi_scratch[p, k] += e_i * tr - e_r * ti
                    norm_sum += tr * tr + ti * ti

        denom = norm_sum + epsilon
        factor = mu / denom
        for p in range(num_orders):
            for k in range(num_taps):
                if p == 0 and k == 0:
                    continue
                c_real[ch, p, k] -= factor * gr_scratch[p, k]
                c_imag[ch, p, k] -= factor * gi_scratch[p, k]


# ==============================================================================
# PolynomialCoefficientTracker Class
# ==============================================================================

class PolynomialCoefficientTracker:
    """
    Tracks and updates the memory-polynomial parameters of SaturationInverter by
    NLMS minimisation of out-of-band filtered energy. Coefficients live in a
    contiguous ctypes shared-memory slot mirrored by zero-copy NumPy views.

    mu default is 0.5: with the NLMS normalisation the stable region is mu < 2,
    and the Phase 3 reachability study found ~0.5 converges the expanded
    (K>=3-tap) model to its OOB optimum within a few thousand strides while
    keeping coefficients O(2). The old 0.05 default was tuned to a one-tap model
    whose OOB optimum was itself counterproductive, so no step size helped there.
    """
    def __init__(
        self,
        num_channels: int = 4,
        stride_len: int = 4096,
        mu: float = 0.5,
        threshold_ratio: float = 0.05,
        filter_coeffs: np.ndarray = None,
        num_orders: int = 2,
        num_taps: int = 3,
        decimation: int = 8,
    ):
        self.num_channels = num_channels
        self.stride_len = stride_len
        self.mu = mu
        self.threshold_ratio = threshold_ratio
        self.num_orders = num_orders
        self.num_taps = num_taps
        self.decimation = decimation

        # OOB probe FIR: arbitrary length >= 1 (this is the spectral filter, a
        # separate concept from the polynomial memory depth num_taps).
        if filter_coeffs is not None:
            self.filter_coeffs = np.ascontiguousarray(filter_coeffs, dtype=np.float32)
        else:
            self.filter_coeffs = np.array([0.5, -0.5], dtype=np.float32)

        # ctypes shared slot, shape (num_channels, num_orders, num_taps) x2 (re/im)
        class DynamicPolynomialCoefficientsSlot(ctypes.Structure):
            _fields_ = [
                ("c_real", ((ctypes.c_float * num_taps) * num_orders) * num_channels),
                ("c_imag", ((ctypes.c_float * num_taps) * num_orders) * num_channels),
            ]

        self.shared_slot = DynamicPolynomialCoefficientsSlot()
        self.c_real_view = np.ctypeslib.as_array(self.shared_slot.c_real)
        self.c_imag_view = np.ctypeslib.as_array(self.shared_slot.c_imag)

        # Linear-identity initialisation: c[0,0] = 1+0j, rest 0.
        self.c_real_view[...] = 0.0
        self.c_imag_view[...] = 0.0
        self.c_real_view[:, 0, 0] = 1.0

        # Preallocated per-channel scratch (zero-heap kernel): regressor and
        # gradient accumulators, both (orders x taps).
        self._tr = np.zeros((num_orders, num_taps), dtype=np.float32)
        self._ti = np.zeros((num_orders, num_taps), dtype=np.float32)
        self._gr = np.zeros((num_orders, num_taps), dtype=np.float32)
        self._gi = np.zeros((num_orders, num_taps), dtype=np.float32)

        self._warmup()

    def _warmup(self):
        """JIT-compile the tracker kernel (adaptation path included) ahead of time."""
        dummy_X = np.ones((self.num_channels, self.stride_len), dtype=np.complex64)
        dummy_Y = np.ones((self.num_channels, self.stride_len), dtype=np.complex64)
        dummy_flags = np.zeros(self.num_channels, dtype=np.bool_)
        dummy_oob = np.zeros(self.num_channels, dtype=np.float32)
        dummy_tot = np.zeros(self.num_channels, dtype=np.float32)

        # Snapshot + restore coefficients: warmup forces the adaptation branch
        # (threshold_ratio = -1.0) and must not perturb the initialised state.
        saved_r = self.c_real_view.copy()
        saved_i = self.c_imag_view.copy()
        _track_all_channels(
            dummy_X, dummy_Y, self.c_real_view, self.c_imag_view,
            self.mu, -1.0, 1e-6, self.decimation,
            dummy_flags, dummy_oob, dummy_tot, self.filter_coeffs,
            self._tr, self._ti, self._gr, self._gi,
        )
        self.c_real_view[...] = saved_r
        self.c_imag_view[...] = saved_i

    def process_stride(self, X_buffer: np.ndarray, Y_buffer: np.ndarray):
        """
        Processes one input/output stride: measures OOB regrowth per channel and,
        where detected, runs one NLMS coefficient update in-place on the ctypes
        shared slots. Returns (regrowth_flags, oob_energies, total_energies).
        """
        regrowth_flags = np.zeros(self.num_channels, dtype=np.bool_)
        oob_energies = np.zeros(self.num_channels, dtype=np.float32)
        total_energies = np.zeros(self.num_channels, dtype=np.float32)

        _track_all_channels(
            X_buffer, Y_buffer, self.c_real_view, self.c_imag_view,
            self.mu, self.threshold_ratio, 1e-6, self.decimation,
            regrowth_flags, oob_energies, total_energies, self.filter_coeffs,
            self._tr, self._ti, self._gr, self._gi,
        )
        return regrowth_flags, oob_energies, total_energies

    def get_coefficients(self, channel: int) -> np.ndarray:
        """Current channel coefficients as a (num_orders, num_taps) complex64 array."""
        return (self.c_real_view[channel] + 1j * self.c_imag_view[channel]).astype(np.complex64)


if __name__ == "__main__":
    print("[*] Instantiating PolynomialCoefficientTracker and pre-warming LLVM...")
    tracker = PolynomialCoefficientTracker(num_channels=4, stride_len=4096)
    print(f"    model: num_orders={tracker.num_orders}, num_taps={tracker.num_taps}, "
          f"adaptive coeffs={tracker.num_orders * tracker.num_taps - 1}")
    print("[*] Tracker initialized. Ready for online adaptation.")
