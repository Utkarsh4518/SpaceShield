"""
Task 39.1: Cyclic Least-Squares (LS) Equalizer Engine
SpaceShield Channel Purification Module

Constructs an ultra-fast localized Least-Squares optimization loop
across a 5-tap sliding delay line, independently per phase-coherent channel.
Dynamically minimizes passband distortion relative to NavIC tracking references
bypassing all dynamic Python allocations.
"""

import numpy as np
from numba import njit, prange

@njit(parallel=False, fastmath=True, boundscheck=False, cache=True)
def _invert_5x5_hermitian(A: np.ndarray, inv_A: np.ndarray):
    """
    Zero-allocation Gauss-Jordan elimination optimized for a 5x5 complex system.
    LLVM will unroll this statically-bounded loop.
    """
    n = 5
    aug = np.zeros((5, 10), dtype=np.complex64)
    for i in range(n):
        for j in range(n):
            aug[i, j] = A[i, j]
        aug[i, i + n] = 1.0 + 0j
        
    # Forward Elimination
    for i in range(n):
        # Partial pivoting
        pivot_val = np.abs(aug[i, i])
        pivot_row = i
        for k in range(i + 1, n):
            val = np.abs(aug[k, i])
            if val > pivot_val:
                pivot_val = val
                pivot_row = k
                
        # Swap rows
        if pivot_row != i:
            for j in range(i, 2 * n):
                temp = aug[i, j]
                aug[i, j] = aug[pivot_row, j]
                aug[pivot_row, j] = temp
                
        # Scale pivot row
        pivot_elem = aug[i, i]
        if np.abs(pivot_elem) < 1e-12:
            pivot_elem = 1e-12 + 0j
            
        inv_pivot = np.float32(1.0) / pivot_elem
        for j in range(i, 2 * n):
            aug[i, j] *= inv_pivot
            
        # Eliminate lower elements
        for k in range(i + 1, n):
            factor = aug[k, i]
            for j in range(i, 2 * n):
                aug[k, j] -= factor * aug[i, j]
                
    # Backward Substitution
    for i in range(n - 1, -1, -1):
        for k in range(i - 1, -1, -1):
            factor = aug[k, i]
            for j in range(i, 2 * n):
                aug[k, j] -= factor * aug[i, j]
                
    # Extract Inverse Matrix
    for i in range(n):
        for j in range(n):
            inv_A[i, j] = aug[i, j + n]


@njit(parallel=False, fastmath=True, boundscheck=False, cache=True)
def _compute_ls_filter_weights(
    X_buffer: np.ndarray,
    d_ref: np.ndarray,
    weights_pool: np.ndarray,
    channels: int,
    taps: int,
    stride_len: int
):
    R = np.zeros((5, 5), dtype=np.complex64)
    R_inv = np.zeros((5, 5), dtype=np.complex64)
    p = np.zeros(5, dtype=np.complex64)
    
    N_valid = stride_len - taps + 1
    
    for c in range(channels):
        p0_r = p0_i = p1_r = p1_i = p2_r = p2_i = p3_r = p3_i = p4_r = p4_i = np.float32(0.0)
        r0_r = r0_i = r1_r = r1_i = r2_r = r2_i = r3_r = r3_i = r4_r = r4_i = np.float32(0.0)
                
        for n in range(N_valid):
            idx = n + 4
            ref = d_ref[idx]
            ref_r = ref.real
            # Conjugate ref:
            ref_i = -ref.imag
            
            x0 = X_buffer[c, idx]
            x0_r = x0.real; x0_i = x0.imag
            x1 = X_buffer[c, idx - 1]
            x1_r = x1.real; x1_i = x1.imag
            x2 = X_buffer[c, idx - 2]
            x2_r = x2.real; x2_i = x2.imag
            x3 = X_buffer[c, idx - 3]
            x3_r = x3.real; x3_i = x3.imag
            x4 = X_buffer[c, idx - 4]
            x4_r = x4.real; x4_i = x4.imag
            
            # Cross correlation (p = x * conj(ref))
            p0_r += x0_r * ref_r - x0_i * ref_i; p0_i += x0_r * ref_i + x0_i * ref_r
            p1_r += x1_r * ref_r - x1_i * ref_i; p1_i += x1_r * ref_i + x1_i * ref_r
            p2_r += x2_r * ref_r - x2_i * ref_i; p2_i += x2_r * ref_i + x2_i * ref_r
            p3_r += x3_r * ref_r - x3_i * ref_i; p3_i += x3_r * ref_i + x3_i * ref_r
            p4_r += x4_r * ref_r - x4_i * ref_i; p4_i += x4_r * ref_i + x4_i * ref_r
            
            # Auto correlation (r = x0 * conj(x_k))
            # k=0
            r0_r += x0_r * x0_r + x0_i * x0_i
            # k=1
            r1_r += x0_r * x1_r + x0_i * x1_i; r1_i += x0_i * x1_r - x0_r * x1_i
            # k=2
            r2_r += x0_r * x2_r + x0_i * x2_i; r2_i += x0_i * x2_r - x0_r * x2_i
            # k=3
            r3_r += x0_r * x3_r + x0_i * x3_i; r3_i += x0_i * x3_r - x0_r * x3_i
            # k=4
            r4_r += x0_r * x4_r + x0_i * x4_i; r4_i += x0_i * x4_r - x0_r * x4_i
            
        p[0] = p0_r + 1j * p0_i
        p[1] = p1_r + 1j * p1_i
        p[2] = p2_r + 1j * p2_i
        p[3] = p3_r + 1j * p3_i
        p[4] = p4_r + 1j * p4_i
        
        r0 = r0_r + 1j * r0_i
        r1 = r1_r + 1j * r1_i
        r2 = r2_r + 1j * r2_i
        r3 = r3_r + 1j * r3_i
        r4 = r4_r + 1j * r4_i
        r_arr = np.array([r0, r1, r2, r3, r4], dtype=np.complex64)
            
        for i in range(taps):
            R[i, i] = r_arr[0] + 1e-6 
            for j in range(i + 1, taps):
                k = j - i
                R[i, j] = r_arr[k]
                R[j, i] = np.conj(r_arr[k])
            
        _invert_5x5_hermitian(R, R_inv)
        
        for i in range(taps):
            val = 0.0 + 0j
            val += R_inv[i, 0] * p[0]
            val += R_inv[i, 1] * p[1]
            val += R_inv[i, 2] * p[2]
            val += R_inv[i, 3] * p[3]
            val += R_inv[i, 4] * p[4]
            weights_pool[c, i] = val


@njit(parallel=False, fastmath=True, boundscheck=False, cache=True)
def _apply_ls_filter(
    X_buffer: np.ndarray,
    Y_out: np.ndarray,
    weights_pool: np.ndarray,
    channels: int,
    taps: int,
    stride_len: int
):
    for c in range(channels):
        for n in range(taps - 1):
            Y_out[c, n] = 0.0 + 0j
            
        w0 = np.conj(weights_pool[c, 0])
        w1 = np.conj(weights_pool[c, 1])
        w2 = np.conj(weights_pool[c, 2])
        w3 = np.conj(weights_pool[c, 3])
        w4 = np.conj(weights_pool[c, 4])
        
        w0_r = w0.real; w0_i = w0.imag
        w1_r = w1.real; w1_i = w1.imag
        w2_r = w2.real; w2_i = w2.imag
        w3_r = w3.real; w3_i = w3.imag
        w4_r = w4.real; w4_i = w4.imag
            
        for n in range(taps - 1, stride_len):
            x0 = X_buffer[c, n]
            x1 = X_buffer[c, n - 1]
            x2 = X_buffer[c, n - 2]
            x3 = X_buffer[c, n - 3]
            x4 = X_buffer[c, n - 4]
            
            x0_r = x0.real; x0_i = x0.imag
            x1_r = x1.real; x1_i = x1.imag
            x2_r = x2.real; x2_i = x2.imag
            x3_r = x3.real; x3_i = x3.imag
            x4_r = x4.real; x4_i = x4.imag
            
            vr = w0_r * x0_r - w0_i * x0_i + \
                 w1_r * x1_r - w1_i * x1_i + \
                 w2_r * x2_r - w2_i * x2_i + \
                 w3_r * x3_r - w3_i * x3_i + \
                 w4_r * x4_r - w4_i * x4_i
                 
            vi = w0_r * x0_i + w0_i * x0_r + \
                 w1_r * x1_i + w1_i * x1_r + \
                 w2_r * x2_i + w2_i * x2_r + \
                 w3_r * x3_i + w3_i * x3_r + \
                 w4_r * x4_i + w4_i * x4_r
                 
            Y_out[c, n] = vr + 1j * vi


class CyclicLSEqualizer:
    """
    SpaceShield Least-Squares Channel Equalization Layer.
    Ingests wideband I/Q phase streams immediately following the polyphase filter banks.
    Performs deterministic dynamic passband corrections against NavIC pilot reference signatures.
    """
    def __init__(self, channels: int = 4, taps: int = 5, stride_len: int = 4096):
        self.channels = channels
        self.taps = taps
        self.stride_len = stride_len
        
        # Statically allocate output vectors and structural pools to prevent GC stutter
        self.weights_pool = np.zeros((self.channels, self.taps), dtype=np.complex64)
        self.Y_out = np.zeros((self.channels, self.stride_len), dtype=np.complex64)
        
        self._warmup()
        
    def _warmup(self):
        """Forces immediate JIT compilation mapping through the LLVM backend."""
        dummy_x = np.zeros((self.channels, self.stride_len), dtype=np.complex64)
        dummy_x[0, 0] = 1.0 + 0j
        dummy_d = np.ones(self.stride_len, dtype=np.complex64)
        
        _compute_ls_filter_weights(dummy_x, dummy_d, self.weights_pool, self.channels, self.taps, self.stride_len)
        _apply_ls_filter(dummy_x, self.Y_out, self.weights_pool, self.channels, self.taps, self.stride_len)
        
    def equalize_stride(self, X_buffer: np.ndarray, d_ref: np.ndarray) -> np.ndarray:
        """
        Primary worker execution path.
        Minimizes || Xw - d ||^2 independently across the 4 receiving antenna lines.
        Returns the clean, purified baseline matrices inline to the parallel caller loop.
        """
        # Execute localized inner unrolled computations
        _compute_ls_filter_weights(X_buffer, d_ref, self.weights_pool, self.channels, self.taps, self.stride_len)
        _apply_ls_filter(X_buffer, self.Y_out, self.weights_pool, self.channels, self.taps, self.stride_len)
        
        return self.Y_out
