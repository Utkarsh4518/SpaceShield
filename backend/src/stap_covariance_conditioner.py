import time
import numpy as np
from numba import njit

def _compute_stap_covariance_kernel(X, R_ST, R0, R1, R2, base_load, trace_scale):
    """
    Numba-optimized kernel for computing the joint space-time covariance matrix R_ST
    using a block-Toeplitz formulation for maximum instruction-level speed.
    """
    # 1. Reset pre-allocated block matrices
    for i in range(4):
        for j in range(4):
            R0[i, j] = 0.0 + 0j
            R1[i, j] = 0.0 + 0j
            R2[i, j] = 0.0 + 0j

    # 2. Compute the 4x4 spatial cross-covariance blocks
    for i in range(4):
        # j < i: only compute cross-lags R1 and R2
        for j in range(i):
            acc_R1 = 0.0 + 0j
            acc_R2 = 0.0 + 0j
            
            # Boundary t=0, 1 (Circular wrap)
            val_i_0 = X[i, 0]; val_j_4095 = X[j, 4095]; val_j_4094 = X[j, 4094]
            val_i_1 = X[i, 1]; val_j_0 = X[j, 0]; val_j_4095_b = X[j, 4095]
            
            acc_R1 += val_i_0 * np.conj(val_j_4095)
            acc_R2 += val_i_0 * np.conj(val_j_4094)
            acc_R1 += val_i_1 * np.conj(val_j_0)
            acc_R2 += val_i_1 * np.conj(val_j_4095_b)
            
            # Main sample loop (branch-free)
            for t in range(2, 4096):
                val_i = X[i, t]
                acc_R1 += val_i * np.conj(X[j, t - 1])
                acc_R2 += val_i * np.conj(X[j, t - 2])
                
            R1[i, j] = acc_R1 / 4096.0
            R2[i, j] = acc_R2 / 4096.0

        # j >= i: compute spatial covariance R0 and cross-lags R1 and R2
        for j in range(i, 4):
            acc_R0 = 0.0 + 0j
            acc_R1 = 0.0 + 0j
            acc_R2 = 0.0 + 0j
            
            # Boundary t=0, 1 (Circular wrap)
            val_i_0 = X[i, 0]; val_j_0 = X[j, 0]; val_j_4095 = X[j, 4095]; val_j_4094 = X[j, 4094]
            val_i_1 = X[i, 1]; val_j_1 = X[j, 1]; val_j_0_b = X[j, 0]; val_j_4095_b = X[j, 4095]
            
            acc_R0 += val_i_0 * np.conj(val_j_0)
            acc_R0 += val_i_1 * np.conj(val_j_1)
            acc_R1 += val_i_0 * np.conj(val_j_4095)
            acc_R1 += val_i_1 * np.conj(val_j_0_b)
            acc_R2 += val_i_0 * np.conj(val_j_4094)
            acc_R2 += val_i_1 * np.conj(val_j_4095_b)
            
            if j == i:
                # Diagonal case: real-only optimization for R0
                acc_diag = val_i_0.real * val_i_0.real + val_i_0.imag * val_i_0.imag
                acc_diag += val_i_1.real * val_i_1.real + val_i_1.imag * val_i_1.imag
                for t in range(2, 4096):
                    val_i = X[i, t]
                    acc_diag += val_i.real * val_i.real + val_i.imag * val_i.imag
                    acc_R1 += val_i * np.conj(X[i, t - 1])
                    acc_R2 += val_i * np.conj(X[i, t - 2])
                R0[i, i] = np.complex64(acc_diag / 4096.0)
            else:
                for t in range(2, 4096):
                    val_i = X[i, t]
                    acc_R0 += val_i * np.conj(X[j, t])
                    acc_R1 += val_i * np.conj(X[j, t - 1])
                    acc_R2 += val_i * np.conj(X[j, t - 2])
                R0[i, j] = acc_R0 / 4096.0
                R0[j, i] = np.conj(R0[i, j])
                
            R1[i, j] = acc_R1 / 4096.0
            R2[i, j] = acc_R2 / 4096.0

    # 3. Assemble the full 12x12 Block-Toeplitz R_ST matrix
    for p_row in range(3):
        row_offset = p_row * 4
        for p_col in range(3):
            col_offset = p_col * 4
            lag = p_row - p_col
            if lag == 0:
                for i in range(4):
                    for j in range(4):
                        R_ST[row_offset + i, col_offset + j] = R0[i, j]
            elif lag == 1:
                for i in range(4):
                    for j in range(4):
                        R_ST[row_offset + i, col_offset + j] = np.conj(R1[j, i])
            elif lag == 2:
                for i in range(4):
                    for j in range(4):
                        R_ST[row_offset + i, col_offset + j] = np.conj(R2[j, i])
            elif lag == -1:
                for i in range(4):
                    for j in range(4):
                        R_ST[row_offset + i, col_offset + j] = R1[i, j]
            elif lag == -2:
                for i in range(4):
                    for j in range(4):
                        R_ST[row_offset + i, col_offset + j] = R2[i, j]

    # 4. Compute space-time Matrix Trace (real part)
    tr = 0.0
    for k in range(12):
        tr += R_ST[k, k].real

    # 5. Calculate regularized LSMI diagonal loading factor
    alpha = base_load + trace_scale * tr

    # 6. Apply diagonal loading in-place
    for k in range(12):
        R_ST[k, k] += alpha + 0j


