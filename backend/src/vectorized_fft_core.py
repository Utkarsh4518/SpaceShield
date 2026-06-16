"""
Task 37.1: SpaceShield Vectorized FFT Core
Radix-4 Split-Radix Pre-Computed Frequency Domain Transformer

Implements an ultra-low latency, zero-allocation radix-4 Fast Fourier Transform.
Utilizes Numba-fused unrolled loop constructs and manually vectorized float32 
butterfly paths to satisfy strict 18us execution barriers over a 4-channel matrix.
"""

import numpy as np
from numba import njit, prange

@njit(cache=True)
def _generate_radix4_rev(n: int) -> np.ndarray:
    """Pre-computes a base-4 digit reversal array to eliminate runtime index tracking."""
    rev = np.zeros(n, dtype=np.int32)
    shift = int(np.round(np.log2(n)))
    num_digits = shift // 2
    for i in range(n):
        res = 0
        temp = i
        for _ in range(num_digits):
            res = (res << 2) | (temp & 3)
            temp >>= 2
        rev[i] = res
    return rev

@njit(cache=True)
def _generate_twiddles(N: int):
    """
    Pre-computes and flattens all Radix-4 Twiddle Factors (W1, W2, W3) across all stages.
    Enforces strict float32/complex64 numerical bounds for high-speed LLVM execution.
    """
    num_stages = int(np.round(np.log2(N) / 2))
    
    # Calculate the exact scalar memory required for all sequential stages
    total_twiddles = 0
    L_dummy = 4
    for _ in range(num_stages):
        total_twiddles += L_dummy // 4
        L_dummy *= 4
        
    W1 = np.zeros(total_twiddles, dtype=np.complex64)
    W2 = np.zeros(total_twiddles, dtype=np.complex64)
    W3 = np.zeros(total_twiddles, dtype=np.complex64)
    
    offset = 0
    L = 4
    for _ in range(num_stages):
        L_quarter = L // 4
        for i in range(L_quarter):
            angle = -2.0 * np.pi * i / L
            W1[offset + i] = np.complex64(np.cos(angle) + 1j * np.sin(angle))
            W2[offset + i] = np.complex64(np.cos(2 * angle) + 1j * np.sin(2 * angle))
            W3[offset + i] = np.complex64(np.cos(3 * angle) + 1j * np.sin(3 * angle))
        offset += L_quarter
        L *= 4
        
    return W1, W2, W3, num_stages


@njit(parallel=False, fastmath=True, boundscheck=False, cache=True)
def _radix4_fft_core(
    in_buffer: np.ndarray,
    out_buffer: np.ndarray,
    rev_indices: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    W3: np.ndarray,
    num_channels: int,
    N: int,
    num_stages: int
):
    """
    Core SIMD engine executing a Radix-4 discrete forward transform.
    Aggressively avoids Python complex float128 abstraction leaks.
    """
    for c in range(num_channels):
        # 1. Zero-Allocation base-4 memory copy
        for i in range(N):
            out_buffer[c, i] = in_buffer[c, rev_indices[i]]
            
        # 2. Strided Radix-4 Butterfly network
        offset = 0
        L = 4
        for _ in range(num_stages):
            L_quarter = L // 4
            
            for block in range(0, N, L):
                for i in range(L_quarter):
                    idx0 = block + i
                    idx1 = idx0 + L_quarter
                    idx2 = idx1 + L_quarter
                    idx3 = idx2 + L_quarter
                    
                    w1 = W1[offset + i]
                    w2 = W2[offset + i]
                    w3 = W3[offset + i]
                    
                    # Manual float32 vector multiplications bypassing CPython object overhead
                    B_val = out_buffer[c, idx1]
                    B_r = np.float32(B_val.real * w1.real - B_val.imag * w1.imag)
                    B_i = np.float32(B_val.real * w1.imag + B_val.imag * w1.real)
                    
                    C_val = out_buffer[c, idx2]
                    C_r = np.float32(C_val.real * w2.real - C_val.imag * w2.imag)
                    C_i = np.float32(C_val.real * w2.imag + C_val.imag * w2.real)
                    
                    D_val = out_buffer[c, idx3]
                    D_r = np.float32(D_val.real * w3.real - D_val.imag * w3.imag)
                    D_i = np.float32(D_val.real * w3.imag + D_val.imag * w3.real)
                    
                    A_val = out_buffer[c, idx0]
                    A_r = A_val.real
                    A_i = A_val.imag
                    
                    # First sub-stage adds
                    t0_r = np.float32(A_r + C_r)
                    t0_i = np.float32(A_i + C_i)
                    t1_r = np.float32(A_r - C_r)
                    t1_i = np.float32(A_i - C_i)
                    
                    t2_r = np.float32(B_r + D_r)
                    t2_i = np.float32(B_i + D_i)
                    t3_r = np.float32(B_r - D_r)
                    t3_i = np.float32(B_i - D_i)
                    
                    # Final crossover & complex multiplication via phase logic
                    out_buffer[c, idx0] = (t0_r + t2_r) + 1j * (t0_i + t2_i)
                    out_buffer[c, idx1] = (t1_r + t3_i) + 1j * (t1_i - t3_r)
                    out_buffer[c, idx2] = (t0_r - t2_r) + 1j * (t0_i - t2_i)
                    out_buffer[c, idx3] = (t1_r - t3_i) + 1j * (t1_i + t3_r)
                    
            offset += L_quarter
            L *= 4


class VectorizedFFTCore:
    """
    SpaceShield High-Speed Spectral Mapper.
    Intercepts phase-aligned complex strides and executes parallel Radix-4 transforms
    designed specifically for real-time Spatio-Temporal EW integration.
    """
    def __init__(self, channels: int = 4, n_points: int = 4096):
        self.channels = channels
        self.N = n_points
        
        # Verify strict power of 4 boundary
        if self.N == 0 or (self.N & (self.N - 1)) != 0 or int(np.round(np.log2(self.N))) % 2 != 0:
            raise ValueError(f"VectorizedFFTCore demands a strict Power of 4 size. Received: {self.N}")
            
        self.rev_indices = _generate_radix4_rev(self.N)
        self.W1, self.W2, self.W3, self.num_stages = _generate_twiddles(self.N)
        
        # Pre-allocate zero-heap output transformation matrix
        self.out_buffer = np.zeros((self.channels, self.N), dtype=np.complex64)
        
        self._warmup()
        
    def _warmup(self):
        """Eliminates LLVM IR JIT lag to prevent dropping EW streams on initial cold starts."""
        dummy_in = np.zeros((self.channels, self.N), dtype=np.complex64)
        _radix4_fft_core(
            dummy_in, self.out_buffer, self.rev_indices, 
            self.W1, self.W2, self.W3, self.channels, self.N, self.num_stages
        )
        self.out_buffer.fill(0)
        
    def execute_transform(self, iq_stride: np.ndarray) -> np.ndarray:
        """
        Executes a 4-channel Radix-4 Forward Transform mapping continuous
        time-domain signals into isolated frequency bins.
        """
        if iq_stride.shape != (self.channels, self.N):
            raise ValueError(f"Strand size mismatch. Expected ({self.channels}, {self.N}).")
            
        _radix4_fft_core(
            iq_stride, self.out_buffer, self.rev_indices, 
            self.W1, self.W2, self.W3, self.channels, self.N, self.num_stages
        )
        
        return self.out_buffer
