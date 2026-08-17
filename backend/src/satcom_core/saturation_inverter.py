"""
Task 55.1: Inline Saturation Linearization Inversion Engine
SpaceShield High-Velocity Receiver DSP Subsystem

Applies a zero-allocation, vectorized memory-polynomial model to invert
non-linear receiver clipping / IMD regrowth:

    y[n] = sum_{p=0}^{P-1} sum_{k=0}^{K-1} c[p, k] * x[n-k] * |x[n-k]|^(2p)

The model is now CONFIGURABLE in both memory depth (num_taps = K) and nonlinear
order (num_orders = P); see memory_polynomial.py for the shared layout and the
Phase 3 capacity study that motivated the expansion from the old fixed
{c10, c30, c31, c50} subset (one effective memory tap) to K>=3 taps.
"""

import numpy as np
import time
from numba import njit

import memory_polynomial as mp


# parallel=True/prange was intentionally NOT used: this kernel runs over the
# fixed M=4 antenna array, and Numba's thread-pool dispatch for a 4-way prange
# region measured ~3x slower end-to-end than a serial loop for that channel
# count (same finding as the tracker kernel). Revisit only if num_channels grows
# well past the host core count.
@njit(fastmath=True, cache=True, boundscheck=False, parallel=False)
def _apply_linearization(
    X_buffer: np.ndarray,      # (channels, stride_len) complex64
    Y_buffer: np.ndarray,      # (channels, stride_len) complex64
    c_real: np.ndarray,        # (channels, num_orders, num_taps) float32
    c_imag: np.ndarray,        # (channels, num_orders, num_taps) float32
):
    """
    Zero-heap Numba kernel evaluating the configurable memory polynomial
        y[n] = sum_{p,k} c[p,k] * x[n-k] * |x[n-k]|^(2p)
    with x[n-k] = 0 for n-k < 0 (delay-line cold start). Uses a Horner
    recurrence over the nonlinear order p so |x|^(2p) is never re-powered.
    """
    num_channels = X_buffer.shape[0]
    stride_len = X_buffer.shape[1]
    num_orders = c_real.shape[1]
    num_taps = c_real.shape[2]

    for ch in range(num_channels):
        for n in range(stride_len):
            acc_r = 0.0
            acc_i = 0.0
            k_max = num_taps if n + 1 >= num_taps else n + 1
            for k in range(k_max):
                xk = X_buffer[ch, n - k]
                xr = xk.real
                xi = xk.imag
                a2 = xr * xr + xi * xi
                # Horner over orders: poly = c[P-1] ; poly = poly*a2 + c[p]
                pr = c_real[ch, num_orders - 1, k]
                pi = c_imag[ch, num_orders - 1, k]
                for p in range(num_orders - 2, -1, -1):
                    pr = pr * a2 + c_real[ch, p, k]
                    pi = pi * a2 + c_imag[ch, p, k]
                # term = x[n-k] * poly(|x[n-k]|^2)
                acc_r += xr * pr - xi * pi
                acc_i += xr * pi + xi * pr
            Y_buffer[ch, n] = acc_r + 1j * acc_i


class SaturationInverter:
    """
    SpaceShield saturation-linearization interface.

    Configurable memory-polynomial linearizer. Public surface preserved:
    ``coefficients`` (complex64, now shape (channels, num_orders, num_taps)),
    ``linearize_stride(X)``, ``c_real`` / ``c_imag`` float32 views.
    """
    def __init__(
        self,
        channels: int = 4,
        stride_len: int = 4096,
        coefficients: np.ndarray = None,
        num_orders: int = 2,
        num_taps: int = 3,
    ):
        self.channels = channels
        self.stride_len = stride_len

        if coefficients is not None:
            coefficients = np.asarray(coefficients)
            if coefficients.ndim != 3 or coefficients.shape[0] != channels:
                raise ValueError(
                    "coefficients must have shape (channels, num_orders, num_taps)"
                )
            self.num_orders = coefficients.shape[1]
            self.num_taps = coefficients.shape[2]
            self._coefficients = coefficients.astype(np.complex64)
        else:
            self.num_orders = num_orders
            self.num_taps = num_taps
            self._coefficients = mp.default_coefficients(channels, num_orders, num_taps)

        # Pre-allocate zero-heap output buffer
        self.Y_buffer = np.zeros((self.channels, self.stride_len), dtype=np.complex64)

        # Split into contiguous real/imag float32 views for register-friendly JIT
        self.c_real = np.ascontiguousarray(self._coefficients.real, dtype=np.float32)
        self.c_imag = np.ascontiguousarray(self._coefficients.imag, dtype=np.float32)

        self._warmup()

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        value = np.asarray(value).astype(np.complex64)
        self._coefficients = value
        self.num_orders = value.shape[1]
        self.num_taps = value.shape[2]
        self.c_real = np.ascontiguousarray(value.real, dtype=np.float32)
        self.c_imag = np.ascontiguousarray(value.imag, dtype=np.float32)

    def _warmup(self):
        """Forces LLVM JIT compilation ahead of processing to avoid latency ceilings."""
        dummy_X = np.ones((self.channels, self.stride_len), dtype=np.complex64)
        _apply_linearization(dummy_X, self.Y_buffer, self.c_real, self.c_imag)

    def linearize_stride(self, X_buffer: np.ndarray) -> np.ndarray:
        """
        Processes a raw input complex sample stride, applying the memory-polynomial
        correction. Overwrites and returns the internal pre-allocated Y_buffer.
        """
        _apply_linearization(X_buffer, self.Y_buffer, self.c_real, self.c_imag)
        return self.Y_buffer


if __name__ == "__main__":
    print("[*] Instantiating SaturationInverter and pre-warming LLVM compiler...")
    inverter = SaturationInverter()
    print(f"    model: num_orders={inverter.num_orders}, num_taps={inverter.num_taps}, "
          f"coeffs={inverter.num_orders * inverter.num_taps}")

    mock_X = (np.random.randn(4, 4096) + 1j * np.random.randn(4, 4096)).astype(np.complex64)

    print("[*] Running 1,000 benchmark strides...")
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = inverter.linearize_stride(mock_X)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    avg_us = np.mean(latencies)
    p99_us = np.percentile(latencies, 99.0)

    print("\n--- SATURATION INVERTER PERFORMANCE HUD ---")
    print(f"  Average Stride Latency: {avg_us:.2f} us")
    print(f"  P99 Stride Latency:      {p99_us:.2f} us")
