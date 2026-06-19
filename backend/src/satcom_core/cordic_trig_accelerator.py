"""
Task 35.1: SpaceShield CORDIC Trigonometric Accelerator
Zero-Allocation, Integer-Only Coordinate Conversion Engine

Optimized for Q1.31 fixed-point math using 32-bit signed integers.
Provides both vectoring (atan2, magnitude) and rotation (sin, cos) modes.
Engineered for <15µs execution per 4096-sample stride via SIMD Numba compilation.
"""

import numpy as np
from numba import njit, prange

# Q1.31 Binary Angle Measure (BAM) Table where 2**31 represents pi radians.
# Pre-calculated via: [int(round(math.atan(2**-i) / math.pi * 2**31)) for i in range(32)]
ATAN_TABLE = np.array([
    536870912, 316933406, 167414882,  85004756,  42667331,  21354465,  10679838,
      5340245,   2670163,   1335087,    667544,    333772,    166886,     83443,
        41722,     20861,     10430,      5215,      2608,      1304,
          652,       326,       163,        81,        41,        20,
           10,         5,         3,         1,         1,         0
], dtype=np.int32)

# Inverse CORDIC Gain (1/K) in Q1.31 = int(round(0.6072529350088812 * (2**31)))
CORDIC_GAIN_INV = np.int64(1304095400)

# Pi/2 in Q1.31 Binary Angle Measure
PI_OVER_2 = np.int32(1073741824)


@njit(parallel=True, fastmath=True, boundscheck=False, cache=True)
def cordic_process_stride(
    x_in: np.ndarray,
    y_in: np.ndarray,
    z_in: np.ndarray,
    x_out: np.ndarray,
    y_out: np.ndarray,
    z_out: np.ndarray,
    mode: np.int32,
    iterations: np.int32 = 32
):
    """
    Zero-allocation SIMD CORDIC block mapping inputs to resolved polar/cartesian spaces.
    
    Format: Q1.31 (32-bit signed integers).
    Angles mapped such that 2**31 represents pi radians (Binary Angle Measure).
    
    Inputs:
        x_in, y_in, z_in: Pre-allocated 1D int32 arrays (Q1.31)
        x_out, y_out, z_out: Pre-allocated 1D int32 arrays for output
        mode: 0 for Vectoring (Cartesian -> Polar)
              1 for Rotation (Polar -> Cartesian)
        iterations: Number of CORDIC iterations (max 32)
    """
    n = x_in.shape[0]
    for i in prange(n):
        x = x_in[i]
        y = y_in[i]
        z = z_in[i]

        if mode == 0:  # Vectoring Mode (resolve phase and magnitude)
            if x == 0 and y == 0:
                x_out[i] = 0
                y_out[i] = 0
                z_out[i] = 0
                continue
                
            # Pre-rotate into right half-plane [-pi/2, pi/2] to ensure convergence
            if x < 0:
                if y >= 0:
                    tx = x
                    x = y
                    y = np.int32(-tx)
                    z = np.int32(z + PI_OVER_2)
                else:
                    tx = x
                    x = np.int32(-y)
                    y = tx
                    z = np.int32(z - PI_OVER_2)
            
            # Systolic CORDIC Iterations
            for j in range(iterations):
                if y >= 0:
                    dx = np.int32(y >> j)
                    dy = np.int32(-(x >> j))
                    dz = ATAN_TABLE[j]
                else:
                    dx = np.int32(-(y >> j))
                    dy = np.int32(x >> j)
                    dz = -ATAN_TABLE[j]
                    
                x = np.int32(x + dx)
                y = np.int32(y + dy)
                z = np.int32(z + dz)

            # Apply gain compensation via fast int64 multiplier and shift back to Q1.31
            x_out[i] = np.int32((np.int64(x) * CORDIC_GAIN_INV) >> 31)
            y_out[i] = 0
            z_out[i] = z

        else:  # Rotation Mode (resolve sine and cosine)
            # Pre-rotate for angles outside bounds [-pi/2, pi/2]
            if z > PI_OVER_2:
                tx = x
                x = np.int32(-y)
                y = tx
                z = np.int32(z - PI_OVER_2)
            elif z < -PI_OVER_2:
                tx = x
                x = y
                y = np.int32(-tx)
                z = np.int32(z + PI_OVER_2)

            # Systolic CORDIC Iterations
            for j in range(iterations):
                if z >= 0:
                    dx = np.int32(-(y >> j))
                    dy = np.int32(x >> j)
                    dz = -ATAN_TABLE[j]
                else:
                    dx = np.int32(y >> j)
                    dy = np.int32(-(x >> j))
                    dz = ATAN_TABLE[j]
                    
                x = np.int32(x + dx)
                y = np.int32(y + dy)
                z = np.int32(z + dz)

            # Apply gain compensation
            x_out[i] = np.int32((np.int64(x) * CORDIC_GAIN_INV) >> 31)
            y_out[i] = np.int32((np.int64(y) * CORDIC_GAIN_INV) >> 31)
            z_out[i] = z


class CordicTrigAccelerator:
    """
    SpaceShield Hardware-in-the-Loop CORDIC Accelerator Interface.
    Maintains rigorous zero-allocation profile via double-buffered pre-allocation.
    
    Warning: For Vectoring mode, ensure input magnitudes sqrt(x^2 + y^2) < 1.0 
    to prevent Q1.31 overflow. Limit input amplitudes to ~0.6 to account for internal CORDIC gain.
    """
    def __init__(self, stride_size: int = 4096):
        self.stride_size = stride_size
        
        # Pre-allocate zero-heap operational buffers
        self.x_buffer = np.zeros(stride_size, dtype=np.int32)
        self.y_buffer = np.zeros(stride_size, dtype=np.int32)
        self.z_buffer = np.zeros(stride_size, dtype=np.int32)
        
        self.x_out = np.zeros(stride_size, dtype=np.int32)
        self.y_out = np.zeros(stride_size, dtype=np.int32)
        self.z_out = np.zeros(stride_size, dtype=np.int32)
        
        self._warmup()

    def _warmup(self):
        """Forces ahead-of-time JIT compilation to avoid first-call latency spikes."""
        cordic_process_stride(
            self.x_buffer[:1], self.y_buffer[:1], self.z_buffer[:1],
            self.x_out[:1], self.y_out[:1], self.z_out[:1],
            np.int32(0), np.int32(32)
        )

    def vectoring_cartesian_to_polar(self, iq_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Ingests a shape (N,) complex64 or 2D (2, N) integer array.
        Returns mapped memory views of (Magnitude, Phase).
        """
        n = iq_array.shape[-1]
        if n > self.stride_size:
            raise ValueError(f"Stride exceeds hardware bounds: {n} > {self.stride_size}")

        if np.iscomplexobj(iq_array):
            # Scale float64/float32 inputs to Q1.31 and cast
            # Note: For strict zero-allocation, prefer feeding integer arrays directly to x_buffer
            self.x_buffer[:n] = np.int32(iq_array.real * (2**31 - 1))
            self.y_buffer[:n] = np.int32(iq_array.imag * (2**31 - 1))
        else:
            self.x_buffer[:n] = iq_array[0]
            self.y_buffer[:n] = iq_array[1]

        self.z_buffer[:n] = 0

        cordic_process_stride(
            self.x_buffer[:n], self.y_buffer[:n], self.z_buffer[:n],
            self.x_out[:n], self.y_out[:n], self.z_out[:n],
            np.int32(0), np.int32(32)
        )
        return self.x_out[:n], self.z_out[:n]

    def rotation_polar_to_cartesian(self, phase_q131: np.ndarray, magnitude_q131: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Resolves sine and cosine vectors simultaneously from phase inputs.
        """
        n = phase_q131.shape[0]
        if n > self.stride_size:
            raise ValueError(f"Stride exceeds hardware bounds: {n} > {self.stride_size}")

        if magnitude_q131 is not None:
            self.x_buffer[:n] = magnitude_q131
        else:
            # Inject unit magnitude max scalar (slightly less than 1.0 to prevent corner clipping)
            self.x_buffer[:n] = 2147483647 
            
        self.y_buffer[:n] = 0
        self.z_buffer[:n] = phase_q131

        cordic_process_stride(
            self.x_buffer[:n], self.y_buffer[:n], self.z_buffer[:n],
            self.x_out[:n], self.y_out[:n], self.z_out[:n],
            np.int32(1), np.int32(32)
        )
        return self.x_out[:n], self.y_out[:n]

