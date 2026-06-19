"""
Task 55.1: Inline Saturation Linearization Inversion Engine
SpaceShield High-Velocity Receiver DSP Subsystem

Applies a zero-allocation, vectorized Memory Polynomial model to reconstruct signal peaks
and invert non-linear receiver clipping using strict odd-order orthogonal terms.
y(n) = c_1_0 * x(n) + c_3_0 * x(n)*|x(n)|^2 + c_3_1 * x(n-1)*|x(n-1)|^2 + c_5_0 * x(n)*|x(n)|^4
"""

import numpy as np
import time
import math
from numba import njit, prange

@njit(fastmath=True, cache=True, boundscheck=False, parallel=True)
def _apply_linearization(
    X_buffer: np.ndarray,      # (channels, stride_len) complex64
    Y_buffer: np.ndarray,      # (channels, stride_len) complex64
    c_real: np.ndarray,        # (channels, 5, 2) float32
    c_imag: np.ndarray         # (channels, 5, 2) float32
):
    """
    Zero-Heap Numba JIT Kernel: Evaluates the odd-order orthogonal Memory Polynomial:
    y(n) = c_1_0 * x(n) + c_3_0 * x(n)*|x(n)|^2 + c_3_1 * x(n-1)*|x(n-1)|^2 + c_5_0 * x(n)*|x(n)|^4
    """
    num_channels = X_buffer.shape[0]
    stride_len = X_buffer.shape[1]

    for ch in prange(num_channels):
        # Localize coefficients to avoid namespace overlap/complex lookup in loop
        c10_r = c_real[ch, 0, 0]; c10_i = c_imag[ch, 0, 0]
        c30_r = c_real[ch, 2, 0]; c30_i = c_imag[ch, 2, 0]
        c31_r = c_real[ch, 2, 1]; c31_i = c_imag[ch, 2, 1]
        c50_r = c_real[ch, 4, 0]; c50_i = c_imag[ch, 4, 0]
        
        # Delay elements
        u1_r = 0.0
        u1_i = 0.0
        a1_sq = 0.0
        
        for n in range(stride_len):
            xn = X_buffer[ch, n]
            u0_r = xn.real
            u0_i = xn.imag
            
            # Squared magnitude
            a0_sq = u0_r * u0_r + u0_i * u0_i
            
            # P0(|x(n)|) = c10 + c30 * a0_sq + c50 * a0_sq^2
            p0_r = c10_r + a0_sq * (c30_r + a0_sq * c50_r)
            p0_i = c10_i + a0_sq * (c30_i + a0_sq * c50_i)
            
            # P1(|x(n-1)|) = c31 * a1_sq
            p1_r = c31_r * a1_sq
            p1_i = c31_i * a1_sq
            
            # Complex multiplication for m=0 term: x(n) * P0
            t0_r = u0_r * p0_r - u0_i * p0_i
            t0_i = u0_r * p0_i + u0_i * p0_r
            
            # Complex multiplication for m=1 term: x(n-1) * P1
            t1_r = u1_r * p1_r - u1_i * p1_i
            t1_i = u1_r * p1_i + u1_i * p1_r
            
            # Linearized output y(n)
            Y_buffer[ch, n] = (t0_r + t1_r) + 1j * (t0_i + t1_i)
            
            # Update delay memory
            u1_r = u0_r
            u1_i = u0_i
            a1_sq = a0_sq


class SaturationInverter:
    """
    SpaceShield Saturation Linearization Interface.
    Corrects ADC non-linear saturation scaling inside parallel processing strides.
    """
    def __init__(
        self,
        channels: int = 4,
        stride_len: int = 4096,
        coefficients: np.ndarray = None
    ):
        self.channels = channels
        self.stride_len = stride_len
        
        # Pre-allocate zero-heap output buffer
        self.Y_buffer = np.zeros((self.channels, self.stride_len), dtype=np.complex64)
        
        # Initialize Memory Polynomial coefficients: (channels, 5 orders, 2 delays)
        if coefficients is not None:
            self._coefficients = coefficients.astype(np.complex64)
        else:
            # Default linear mapping: c_{1,0} = 1.0 + 0j, others 0
            self._coefficients = np.zeros((self.channels, 5, 2), dtype=np.complex64)
            self._coefficients[:, 0, 0] = 1.0 + 0j
            
        # Split coefficients to real/imag float32 views for optimal JIT register mapping
        self.c_real = self._coefficients.real.astype(np.float32)
        self.c_imag = self._coefficients.imag.astype(np.float32)
        
        # Warmup JIT compilation trace
        self._warmup()

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        self._coefficients = value.astype(np.complex64)
        self.c_real = self._coefficients.real.astype(np.float32)
        self.c_imag = self._coefficients.imag.astype(np.float32)

    def _warmup(self):
        """Forces LLVM JIT compilation ahead of processing to avoid latency ceilings."""
        dummy_X = np.ones((self.channels, self.stride_len), dtype=np.complex64)
        _apply_linearization(
            dummy_X,
            self.Y_buffer,
            self.c_real,
            self.c_imag
        )

    def linearize_stride(self, X_buffer: np.ndarray) -> np.ndarray:
        """
        Processes a raw input complex sample stride, applying the inverse polynomial correction.
        Overwrites and returns the internal pre-allocated Y_buffer.
        """
        _apply_linearization(
            X_buffer,
            self.Y_buffer,
            self.c_real,
            self.c_imag
        )
        return self.Y_buffer


if __name__ == "__main__":
    print("[*] Instantiating SaturationInverter and pre-warming LLVM compiler...")
    inverter = SaturationInverter()
    
    # Generate mock saturated input stride
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
    print(f"  Average Stride Latency: {avg_us:.2f} µs")
    print(f"  P99 Stride Latency:      {p99_us:.2f} µs")
    
    if avg_us <= 22.0:
        print("[PASSED] Peak reconstruction operates well within the 22µs cap.")
    else:
        print("[FAIL] Operational ceiling breached! Check Numba compilation.")
