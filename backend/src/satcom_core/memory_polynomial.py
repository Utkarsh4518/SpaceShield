"""
Shared memory-polynomial layout & reference math for the SpaceShield
predistortion loop (Task 55 / Phase 3).

Both SaturationInverter (forward evaluation) and PolynomialCoefficientTracker
(NLMS gradient) are driven by the SAME model defined here, so their coefficient
indexing can never silently diverge. The Numba kernels in those two modules
re-implement the exact math below for speed; the pure-NumPy functions here are
the single source of truth the unit tests check both kernels against.

MODEL
-----
Configurable memory polynomial (a.k.a. the memory-polynomial subset of a
Volterra series -- diagonal kernels only):

    y[n] = sum_{p=0}^{P-1} sum_{k=0}^{K-1} c[p, k] * x[n-k] * |x[n-k]|^(2p)

    * K = num_taps      : memory depth (k = 0 is the current sample).
    * P = num_orders    : number of odd nonlinear orders. Order index p maps to
                          polynomial order 2p+1, i.e. p=0 -> linear (|x|^0),
                          p=1 -> cubic (|x|^2), p=2 -> quintic (|x|^4), ...
    * c has shape (P, K) complex per channel. c[0, 0] is the linear/current-tap
      coefficient; it is anchored to 1+0j (the gain reference) and NOT adapted
      by the tracker. Every other slot is a free complex parameter.

Samples with n-k < 0 contribute nothing (zero state before the block start),
matching the Numba kernels' delay-line initialisation.

WHY THIS SHAPE (Phase 3 capacity study)
---------------------------------------
The previous model was a fixed sparse subset {c10, c30, c31, c50}: effectively
ONE useful memory tap. A least-squares capacity sweep + NLMS reachability study
(see the Phase 3 engineering report) showed the binding limitation is MEMORY
DEPTH, not nonlinear order:

    * K=1 tap  : the OOB-energy optimum is *counterproductive* in-band (~-16 dB);
                 no step size helps -> this is why the old loop stalled near 0 dB.
    * K=3 taps : well-conditioned optimum, NLMS reaches ~+10-14 dB in-band IMD
                 suppression on the jammer scenario with |c| ~ O(2).
    * Nonlinear order beyond cubic (P>2) adds <0.1 dB robustly for that scenario
      and only inflates the basis condition number, so it is configurable but
      not the default lever.

The dense (P, K) subset with the old {c10, c30, c31, c50} slots populated and
the rest zero reproduces the previous model bit-for-bit (see
tests/test_saturation_inverter.py::legacy regression).
"""

import numpy as np

# The previous fixed model, expressed as (num_orders=3, num_taps=2) with only
# these (p, k) slots non-zero. Handy for legacy regression tests.
LEGACY_NUM_ORDERS = 3
LEGACY_NUM_TAPS = 2
LEGACY_ACTIVE_SLOTS = ((0, 0), (1, 0), (1, 1), (2, 0))  # c10, c30, c31, c50


def default_coefficients(channels: int, num_orders: int, num_taps: int) -> np.ndarray:
    """Linear-identity initialisation: c[0,0]=1+0j (pass-through), rest 0."""
    c = np.zeros((channels, num_orders, num_taps), dtype=np.complex64)
    c[:, 0, 0] = 1.0 + 0.0j
    return c


def basis_regressor(x: np.ndarray, p: int, k: int) -> np.ndarray:
    """
    psi_{p,k}[n] = x[n-k] * |x[n-k]|^(2p), 1-D complex, zero-padded for n-k < 0.

    This is exactly the regressor whose coefficient is c[p, k]; the tracker's
    gradient is built from FIR-filtered copies of these.
    """
    xk = np.zeros_like(x)
    if k == 0:
        xk = x.astype(np.complex128, copy=True)
    else:
        xk = xk.astype(np.complex128)
        xk[k:] = x[:-k]
    a2 = np.abs(xk) ** 2
    return xk * (a2 ** p)


def evaluate(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy reference for the memory-polynomial output y[n].

    X            : (channels, stride_len) complex
    coefficients : (channels, num_orders, num_taps) complex
    returns      : (channels, stride_len) complex
    """
    channels, stride_len = X.shape
    _, num_orders, num_taps = coefficients.shape
    Y = np.zeros_like(X, dtype=np.complex128)
    for ch in range(channels):
        for k in range(num_taps):
            psi_base = np.zeros(stride_len, dtype=np.complex128)
            if k == 0:
                psi_base[:] = X[ch]
            else:
                psi_base[k:] = X[ch, :-k]
            a2 = np.abs(psi_base) ** 2
            for p in range(num_orders):
                Y[ch] += coefficients[ch, p, k] * psi_base * (a2 ** p)
    return Y.astype(X.dtype)
