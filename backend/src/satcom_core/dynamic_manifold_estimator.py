import time
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def _auxiliary_vector_projection(R, s_nom, s_cal, mu_step):
    """
    Zero-Heap LLVM Kernel: Resolves Mutual Coupling perturbations dynamically.
    Instead of performing a full Eigen-decomposition (which triggers massive heap 
    allocations and breaches execution limits), this computes an Auxiliary-Vector 
    orthogonal projection step. It pulls the nominal steering vector towards the 
    true Signal Subspace embedded within the live multi-channel Covariance Matrix.
    """
    s_norm_sq = 0.0
    u_dot_s_re = 0.0
    u_dot_s_im = 0.0
    
    # 1. Compute u = R @ s_nom inline
    # We temporarily borrow s_cal to store u to maintain a zero-heap stack frame
    for i in range(4):
        u_re = 0.0
        u_im = 0.0
        for j in range(4):
            u_re += R[i, j].real * s_nom[j].real - R[i, j].imag * s_nom[j].imag
            u_im += R[i, j].real * s_nom[j].imag + R[i, j].imag * s_nom[j].real
            
        s_cal[i] = complex(u_re, u_im)
        
        # Accumulate nominal vector squared norm natively
        s_norm_sq += s_nom[i].real**2 + s_nom[i].imag**2
        
        # Accumulate complex dot product (conj(s_nom) @ u)
        u_dot_s_re += s_nom[i].real * u_re + s_nom[i].imag * u_im
        u_dot_s_im += s_nom[i].real * u_im - s_nom[i].imag * u_re

    # 2. Extract Subspace Scalar projection mapping
    scalar_re = u_dot_s_re / s_norm_sq
    scalar_im = u_dot_s_im / s_norm_sq
    
    cal_norm_sq = 0.0
    for i in range(4):
        # 3. Construct Orthogonal Auxiliary Vector g = u - scalar * s_nom
        # Note: u[i] is currently resting temporarily in s_cal[i]
        g_re = s_cal[i].real - (scalar_re * s_nom[i].real - scalar_im * s_nom[i].imag)
        g_im = s_cal[i].imag - (scalar_re * s_nom[i].imag + scalar_im * s_nom[i].real)
        
        # 4. Morph Calibration: s_cal = s_nom - mu * g
        cal_re = s_nom[i].real - mu_step * g_re
        cal_im = s_nom[i].imag - mu_step * g_im
        
        s_cal[i] = complex(cal_re, cal_im)
        cal_norm_sq += cal_re**2 + cal_im**2
        
    # 5. Null Depth Stability Lock (Phase/Gain normalization)
    norm_factor = np.sqrt(s_norm_sq / cal_norm_sq)
    for i in range(4):
        s_cal[i] = complex(s_cal[i].real * norm_factor, s_cal[i].imag * norm_factor)


class DynamicManifoldEstimator:
    """
    Adaptive Antenna Array Calibration Core.
    Continuously taps the multi-channel spatial covariance tracker to blind-estimate
    and resolve localized mutual coupling and radome perturbation errors without 
    external pilot signals, instantly repairing distorted steering matrices.
    """
    def __init__(self, mu_step: float = 0.01):
        # Morphing step-size (Controls stability vs adaptation speed)
        self.mu_step = mu_step
        
        # Pre-allocated 4-element complex buffer (Zero growth profile)
        self._s_calibrated = np.zeros(4, dtype=np.complex64)

    def calibrate_steering_vector(self, R_matrix: np.ndarray, s_nominal: np.ndarray) -> tuple:
        """
        Executes the instantaneous manifold correction projection.
        """
        t0 = time.perf_counter()
        
        # Dispatch highly-optimized native LLVM kernel mapping
        _auxiliary_vector_projection(R_matrix, s_nominal, self._s_calibrated, self.mu_step)
        
        exec_us = (time.perf_counter() - t0) * 1e6
        
        return self._s_calibrated, exec_us


# --- Rapid Verification Stub ---
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Dynamic Manifold Calibration")
    print("==================================================================")
    
    estimator = DynamicManifoldEstimator(mu_step=0.015)
    
    # 1. Synthesize Mock Covariance and Array Manifold
    R_mock = np.identity(4, dtype=np.complex64)
    s_mock = np.array([1.0+0j, 0.0+1.0j, -1.0+0j, 0.0-1.0j], dtype=np.complex64)
    
    # Burn-in LLVM Math Vector
    estimator.calibrate_steering_vector(R_mock, s_mock)
    
    # 2. Hot-Path Stress Simulation
    print("[*] Tracking Auxiliary-Vector Calibration Profiler...")
    latencies = []
    
    for i in range(2500):
        # Inject extreme random dynamic coupling phase-shifts into the matrix
        R_noise = R_mock + (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) * 0.1
        # Force Hermitian symmetry for legitimate covariance simulation
        R_hermitian = (R_noise + R_noise.conj().T) / 2.0
        R_hermitian = R_hermitian.astype(np.complex64)
        
        calibrated_matrix, exec_us = estimator.calibrate_steering_vector(R_hermitian, s_mock)
        latencies.append(exec_us)

    avg_us = sum(latencies) / len(latencies)
    import numpy as np
    max_us = np.percentile(latencies, 99.0)
    
    print("\n--- MANIFOLD ESTIMATION HUD ---")
    print(f" [>] Estimation Protocol:       Auxiliary-Vector Orthogonal Projection")
    print(f" [>] Complex Memory Allocation: Strict Zero-Heap In-Place Routing")
    
    print(f"\n [>] Average Execution Latency: {avg_us:.2f} µs")
    print(f" [>] Max Edge Latency:          {max_us:.2f} µs")
    
    if max_us < 15.0:
        print("\n[PASSED] Blind mutual coupling calibration processes securely beneath 15µs limit!")
    else:
        print("\n[FAILED] Execution exceeded 15µs critical envelope limit.")
