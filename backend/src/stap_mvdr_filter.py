import time
import numpy as np
from numba import njit

def _cholesky_solve_kernel(R, a, L, y, v, w):
    """
    Solves R_ST * v = a_ST using static Cholesky decomposition L * L^H
    and forward/backward substitution. Outputs optimal weights w = v / (a^H * v).
    Zero-heap execution.
    """
    # 1. Reset lower-triangular working buffer L to 0
    for i in range(12):
        for j in range(12):
            L[i, j] = 0.0 + 0j
            
    # 2. Compute Cholesky factor L: R = L * L^H
    for i in range(12):
        for j in range(i + 1):
            s = 0.0 + 0j
            for k in range(j):
                s += L[i, k] * np.conj(L[j, k])
            if i == j:
                val = R[i, i] - s
                val_real = val.real
                # Clamping bounds to prevent floating-point underflow singularities
                if val_real < 1e-12:
                    val_real = 1e-12
                L[i, i] = np.complex64(np.sqrt(val_real) + 0j)
            else:
                L[i, j] = (R[i, j] - s) / L[j, j]
                
    # 3. Forward substitution: L * y = a
    for i in range(12):
        s = 0.0 + 0j
        for k in range(i):
            s += L[i, k] * y[k]
        y[i] = (a[i] - s) / L[i, i]
        
    # 4. Backward substitution: L^H * v = y
    for i in range(11, -1, -1):
        s = 0.0 + 0j
        for k in range(i + 1, 12):
            s += np.conj(L[k, i]) * v[k]
        v[i] = (y[i] - s) / L[i, i]
        
    # 5. Compute denominator scaling factor: beta = a^H * v
    beta = 0.0 + 0j
    for k in range(12):
        beta += np.conj(a[k]) * v[k]
        
    beta_val = beta.real
    if beta_val < 1e-12:
        beta_val = 1e-12
        
    # 6. Normalize weights w = v / beta
    for k in range(12):
        w[k] = v[k] / beta_val


@njit(fastmath=True, cache=True)
