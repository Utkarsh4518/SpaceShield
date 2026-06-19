"""
Task 36.1: SpaceShield Polyphase Decimator
High-Throughput Alias-Free Sub-Band Isolation Engine

Implements an M-channel noble identities decomposition structure.
Features a Kaiser-windowed low-pass prototype matrix (stopband >60 dB).
Enforces absolute zero-allocation with strict 32-bit floating precision.
"""

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Filter Design Mathematics (Dependency-Free / Pre-Computation)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _i0(x):
    """Modified Bessel function of the first kind (zeroth order)"""
    ans = 1.0
    term = 1.0
    x2 = (x / 2.0)**2
    for k in range(1, 50):
        term *= x2 / (k * k)
        ans += term
        if term < 1e-15 * ans:
            break
    return ans

@njit(cache=True)
def _kaiser_window(N, beta):
    """Generates a Kaiser window for FIR attenuation bounds"""
    alpha = (N - 1) / 2.0
    w = np.zeros(N, dtype=np.float32)
    denom = _i0(beta)
    for n in range(N):
        val = (n - alpha) / alpha
        w[n] = np.float32(_i0(beta * np.sqrt(1.0 - val**2)) / denom)
    return w

@njit(cache=True)
def _design_lowpass_prototype(M, N, beta=6.0):
    """Generates a normalized Kaiser-sinc lowpass prototype filter"""
    alpha = (N - 1) / 2.0
    w = _kaiser_window(N, beta)
    h = np.zeros(N, dtype=np.float32)
    fc = 1.0 / M
    for n in range(N):
        if n == alpha:
            h[n] = np.float32(2.0 * fc)
        else:
            x = 2.0 * np.pi * fc * (n - alpha)
            h[n] = np.float32(np.sin(x) / (np.pi * (n - alpha)))
    
    h = h * w
    
    # Enforce flat passband DC gain of exactly 1.0
    h_sum = np.sum(h)
    for n in range(N):
        h[n] = np.float32(h[n] / h_sum)
    return h


# ---------------------------------------------------------------------------
# SIMD Polyphase Core & Memory Mapping
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, boundscheck=False, cache=True)
def _shift_and_load_buffer(
    in_buffer: np.ndarray,
    new_data: np.ndarray,
    history_len: int,
    stride_size: int,
    num_channels: int
):
    """
    Bypasses Numpy C-API overhead to guarantee zero heap allocations.
    In-place slides the continuous memory window backwards.
    """
    for c in prange(num_channels):
        # 1. Slide historical tail
        for i in range(history_len):
            in_buffer[c, i] = in_buffer[c, i + stride_size]
            
        # 2. Inject incoming stride
        for i in range(stride_size):
            in_buffer[c, history_len + i] = new_data[c, i]


@njit(parallel=True, fastmath=True, boundscheck=False, cache=True)
def _polyphase_decimate_core(
    in_buffer: np.ndarray,
    out_buffer: np.ndarray,
    h_poly: np.ndarray,
    M: int,
    L_sub: int,
    num_channels: int,
    num_outputs: int
):
    """
    M-channel noble identities decomposition matrix logic.
    Computes directly on float32 and complex64 bounded tensors.
    """
    for c in prange(num_channels):
        for m in range(num_outputs):
            base_idx = m * M
            acc_real = np.float32(0.0)
            acc_imag = np.float32(0.0)
            
            # Parallel branch convolution mapped to sub-filters
            for k in range(M):
                b_real = np.float32(0.0)
                b_imag = np.float32(0.0)
                for n in range(L_sub):
                    idx = base_idx + n * M + k
                    val = in_buffer[c, idx]
                    weight = h_poly[k, n]
                    
                    b_real += val.real * weight
                    b_imag += val.imag * weight
                    
                acc_real += b_real
                acc_imag += b_imag
                
            out_buffer[c, m] = acc_real + 1j * acc_imag


# ---------------------------------------------------------------------------
# Hardware Intercept Class
# ---------------------------------------------------------------------------
class PolyphaseDecimator:
    """
    SpaceShield Sub-Band Isolation Engine.
    Intercepts raw 4-channel complex64 I/Q strides from the SoapyReceiverBridge.
    Applies aggressive >60dB anti-aliasing bounds before fractional rate reduction.
    """
    def __init__(self, channels: int = 4, stride_size: int = 4096, decimation_factor: int = 4, filter_taps: int = None):
        self.channels = channels
        self.stride_size = stride_size
        self.M = decimation_factor
        
        if self.stride_size % self.M != 0:
            raise ValueError("SpaceShield alignment violation: Stride size must be cleanly divisible by decimation factor.")
            
        self.num_outputs = self.stride_size // self.M
        
        # Determine strict FIR prototype length
        # Default ~96 taps for M=4 guarantees an aggressive Kaiser transition band
        if filter_taps is None:
            base_taps = 24 * self.M
        else:
            base_taps = filter_taps
            
        # N must precisely align to an M-channel boundary
        self.N = int(np.ceil(base_taps / self.M) * self.M)
        self.L_sub = self.N // self.M
        
        # Design prototype and prepare the flipped convolution kernel
        h_prototype = _design_lowpass_prototype(self.M, self.N, beta=6.0)
        h_rev = h_prototype[::-1].copy()
        
        # Synthesize the Polyphase Matrix Engine
        self.h_poly = np.zeros((self.M, self.L_sub), dtype=np.float32)
        for k in range(self.M):
            for n in range(self.L_sub):
                self.h_poly[k, n] = h_rev[n * self.M + k]
                
        # Initialize zero-allocation continuous memory mapping
        self.history_len = self.N - 1
        self.buffer_len = self.history_len + self.stride_size
        
        self.in_buffer = np.zeros((self.channels, self.buffer_len), dtype=np.complex64)
        self.out_buffer = np.zeros((self.channels, self.num_outputs), dtype=np.complex64)
        
        self._warmup()
        
    def _warmup(self):
        """Pre-heats Numba LLVM IR to prevent 20us execution ceiling violations on frame 1."""
        dummy_stride = np.zeros((self.channels, self.stride_size), dtype=np.complex64)
        _shift_and_load_buffer(
            self.in_buffer, dummy_stride, 
            self.history_len, self.stride_size, self.channels
        )
        _polyphase_decimate_core(
            self.in_buffer, self.out_buffer, self.h_poly, 
            self.M, self.L_sub, self.channels, self.num_outputs
        )
        
    def decimate_stride(self, iq_stride: np.ndarray) -> np.ndarray:
        """
        Ingests a 4-channel matrix and applies polyphase decimation.
        
        Args:
            iq_stride: np.ndarray of shape (channels, stride_size) in complex64.
            
        Returns:
            np.ndarray view of decimated output mapped into shape (channels, stride_size // M).
        """
        if iq_stride.shape != (self.channels, self.stride_size):
            raise ValueError(f"Strand shape mismatch. Expected ({self.channels}, {self.stride_size}), got {iq_stride.shape}")
            
        # 1. Engage zero-allocation shift
        _shift_and_load_buffer(
            self.in_buffer, iq_stride, 
            self.history_len, self.stride_size, self.channels
        )
        
        # 2. Fire Polyphase convolution network
        _polyphase_decimate_core(
            self.in_buffer, self.out_buffer, self.h_poly,
            self.M, self.L_sub, self.channels, self.num_outputs
        )
        
        return self.out_buffer
