"""
Task 46.2: Adaptive Phased Array Spatial Perturbation Smoothing and Bounding Filter
Zero-Allocation, Numba-Accelerated Geodesic Interpolation Engine
"""

import time
import math
import numpy as np
from numba import njit

# ==============================================================================
# JIT-Compiled Coordinate and Projection Kernels
# ==============================================================================

@njit(fastmath=True, cache=True)
def _build_rotation_matrix(yaw: float, pitch: float, roll: float, R: np.ndarray):
    """
    Constructs the 3D rotation correction matrix R = Rx(-roll) * Ry(-pitch) * Rz(-yaw)
    in-place to avoid heap allocations.
    """
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)
    
    # Row 0
    R[0, 0] = cp * cy
    R[0, 1] = cp * sy
    R[0, 2] = -sp
    
    # Row 1
    R[1, 0] = -cr * sy + sr * sp * cy
    R[1, 1] = cr * cy + sr * sp * sy
    R[1, 2] = sr * cp
    
    # Row 2
    R[2, 0] = sr * sy + cr * sp * cy
    R[2, 1] = -sr * cy + cr * sp * sy
    R[2, 2] = cr * cp


@njit(fastmath=True, cache=True)
def _clamp_steering_kernel(
    s_old: np.ndarray,
    s_new: np.ndarray,
    yaw_old: float,
    yaw_new: float,
    pitch_old: float,
    pitch_new: float,
    roll_old: float,
    roll_new: float,
    threshold: float,
    s_clamped: np.ndarray,
    angles_clamped: np.ndarray,
    R_clamped: np.ndarray,
    R_raw: np.ndarray
) -> bool:
    """
    Evaluates the angular step delta between s_old and s_new.
    Aligns global phase of s_new to s_old to eliminate phase ambiguity.
    Clamps the step precisely to the threshold if it is exceeded.
    Updates s_clamped, angles_clamped, R_clamped, and R_raw.
    Returns True if clamping was applied, False otherwise.
    Operates strictly in-place with zero heap allocations.
    """
    # 1. Intercept and construct the raw un-clamped rotation matrix
    _build_rotation_matrix(yaw_new, pitch_new, roll_new, R_raw)
    
    # 2. Compute complex inner product: rho = s_old^H * s_new
    rho = 0.0 + 0j
    for m in range(4):
        rho += np.conj(s_old[m]) * s_new[m]
        
    r = abs(rho)
    
    # Prevent numerical issues where r exceeds 1.0 slightly
    if r > 1.0:
        r = 1.0
        
    theta = math.acos(r)
    
    # 3. Check threshold violation and apply geodesic projection
    clamped_applied = False
    
    # Phase alignment factor to resolve global phase ambiguity: s'_new = s_new * conj(rho)/r
    phase_corr = 1.0 + 0j
    if r > 1e-12:
        phase_corr = np.conj(rho) / r
        
    if theta > threshold:
        clamped_applied = True
        
        # Geodesic SLERP interpolation factor
        t = threshold / theta
        
        # SLERP weights
        sin_theta = math.sin(theta)
        w_old = math.sin((1.0 - t) * theta) / sin_theta
        w_new = math.sin(t * theta) / sin_theta
        
        for m in range(4):
            s_prime = s_new[m] * phase_corr
            s_clamped[m] = w_old * s_old[m] + w_new * s_prime
            
        # Interpolate angles linearly
        yaw_c = yaw_old + t * (yaw_new - yaw_old)
        pitch_c = pitch_old + t * (pitch_new - pitch_old)
        roll_c = roll_old + t * (roll_new - roll_old)
    else:
        # Step within bounds. Retain phase alignment to avoid sudden phase jumps.
        for m in range(4):
            s_clamped[m] = s_new[m] * phase_corr
            
        yaw_c = yaw_new
        pitch_c = pitch_new
        roll_c = roll_new
        
    # 4. Normalize s_clamped to maintain precise unit norm gain constraint
    norm_sq = 0.0
    for m in range(4):
        val = s_clamped[m]
        norm_sq += val.real * val.real + val.imag * val.imag
        
    if norm_sq > 1e-12:
        inv_norm = 1.0 / math.sqrt(norm_sq)
        for m in range(4):
            s_clamped[m] *= inv_norm
    else:
        # Fallback to s_old if normalization fails
        for m in range(4):
            s_clamped[m] = s_old[m]
            
    # 5. Wrap angles to [-pi, pi]
    while yaw_c > math.pi: yaw_c -= 2.0 * math.pi
    while yaw_c < -math.pi: yaw_c += 2.0 * math.pi
    while pitch_c > math.pi: pitch_c -= 2.0 * math.pi
    while pitch_c < -math.pi: pitch_c += 2.0 * math.pi
    while roll_c > math.pi: roll_c -= 2.0 * math.pi
    while roll_c < -math.pi: roll_c += 2.0 * math.pi
    
    angles_clamped[0] = yaw_c
    angles_clamped[1] = pitch_c
    angles_clamped[2] = roll_c
    
    # 6. Build the clamped rotation matrix in-place
    _build_rotation_matrix(yaw_c, pitch_c, roll_c, R_clamped)
    
    return clamped_applied


# ==============================================================================
# Adaptive Spatial Perturbation Limiter
# ==============================================================================

class SpatialPerturbationLimiter:
    """
    Embedded Signal Estimation Bounding Filter.
    Intercepts raw correction matrices and optimized steering vectors.
    Clamps phase and angular step adjustments to a strict threshold
    using geodesic projections on the complex unit sphere.
    Maintains a strictly zero-heap, sub-microsecond processing pipeline.
    """
    def __init__(self, threshold: float = 0.005):
        self.threshold = threshold
        
        # Pre-allocated active state registers (static variables)
        self.s_active = np.zeros(4, dtype=np.complex64)
        self.angles_active = np.zeros(3, dtype=np.float32)
        self.R_active = np.eye(3, dtype=np.float32)
        
        # Pre-allocated outputs (returned as views to prevent allocations)
        self.s_clamped = np.zeros(4, dtype=np.complex64)
        self.angles_clamped = np.zeros(3, dtype=np.float32)
        self.R_clamped = np.zeros((3, 3), dtype=np.float32)
        self.R_raw = np.zeros((3, 3), dtype=np.float32)
        
        # Set default broadside starting vector
        self.s_active[:] = 0.5 + 0j
        
        self._warmup()
        
    def _warmup(self):
        """Forces Numba JIT compilation trace."""
        dummy_s_old = np.ones(4, dtype=np.complex64) / 2.0
        dummy_s_new = np.ones(4, dtype=np.complex64) / 2.0
        _clamp_steering_kernel(
            dummy_s_old, dummy_s_new,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            self.threshold,
            self.s_clamped,
            self.angles_clamped,
            self.R_clamped,
            self.R_raw
        )
        
    def reset(self, s_init: np.ndarray, yaw_init: float = 0.0, pitch_init: float = 0.0, roll_init: float = 0.0):
        """
        Resets the active state of the filter.
        
        Args:
            s_init: (4,) complex64 initial steering vector.
            yaw_init: Initial yaw angle.
            pitch_init: Initial pitch angle.
            roll_init: Initial roll angle.
        """
        if s_init.shape != (4,):
            raise ValueError("Steering vector must have shape (4,).")
            
        norm = np.linalg.norm(s_init)
        if norm > 1e-12:
            self.s_active[:] = s_init / norm
        else:
            self.s_active[:] = s_init
            
        self.angles_active[0] = yaw_init
        self.angles_active[1] = pitch_init
        self.angles_active[2] = roll_init
        
        _build_rotation_matrix(yaw_init, pitch_init, roll_init, self.R_active)
        
    def limit_perturbation(
        self,
        s_new: np.ndarray,
        yaw_new: float,
        pitch_new: float,
        roll_new: float
    ) -> tuple:
        """
        Evaluates the step adjustment between s_active and s_new.
        Performs geodesic projection clamping if the step exceeds the threshold.
        Updates internal active state with clamped values and returns views.
        
        Args:
            s_new: (4,) complex64 newly adjusted target steering vector.
            yaw_new: Optimized yaw angle.
            pitch_new: Optimized pitch angle.
            roll_new: Optimized roll angle.
            
        Returns:
            Tuple: (s_clamped, R_raw, R_clamped, angles_clamped, clamped_applied)
                   - s_clamped: (4,) complex64 clamped and phase-aligned steering vector view.
                   - R_raw: (3, 3) float32 raw un-clamped rotation matrix view.
                   - R_clamped: (3, 3) float32 clamped rotation matrix view.
                   - angles_clamped: (3,) float32 clamped angles view.
                   - clamped_applied: bool indicating if damping was active.
        """
        clamped_applied = _clamp_steering_kernel(
            self.s_active,
            s_new,
            self.angles_active[0], yaw_new,
            self.angles_active[1], pitch_new,
            self.angles_active[2], roll_new,
            self.threshold,
            self.s_clamped,
            self.angles_clamped,
            self.R_clamped,
            self.R_raw
        )
        
        # Update active state in-place to ensure zero heap allocation
        self.s_active[:] = self.s_clamped
        self.angles_active[:] = self.angles_clamped
        self.R_active[:] = self.R_clamped
        
        return self.s_clamped, self.R_raw, self.R_clamped, self.angles_clamped, clamped_applied


# ==============================================================================
# Standalone Diagnostic Verification Harness
# ==============================================================================

if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Calibration Layer: Spatial Perturbation Limiter")
    print("==================================================================")
    
    limiter = SpatialPerturbationLimiter(threshold=0.005)
    print("[PASS] Spatial Perturbation Limiter Initialized & JIT Compiled.")
    
    # Let's initialize with a broadside steering vector (all ones / 2)
    s_init = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex64)
    limiter.reset(s_init, yaw_init=0.0, pitch_init=0.0, roll_init=0.0)
    
    # Orthogonal component to generate custom perturbations
    s_ortho = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.complex64)
    
    # 1. Test a small update (angular delta < 0.005 radians)
    theta_small = 0.002
    s_small = math.cos(theta_small) * s_init + math.sin(theta_small) * s_ortho
    
    s_clamp1, R_raw1, R_clamp1, angles_clamp1, applied1 = limiter.limit_perturbation(
        s_small, yaw_new=0.002, pitch_new=0.0, roll_new=0.0
    )
    
    rho1 = np.dot(s_init.conj(), s_clamp1)
    angle_step1 = math.acos(min(1.0, abs(rho1)))
    print(f"\n[EVAL] Small Update Test (Target Step: {theta_small:.6f} rad):")
    print(f"    -> Clamping Applied: {applied1}")
    print(f"    -> Resulting Step:   {angle_step1:.6f} rad")
    print(f"    -> Clamped Angles:   {angles_clamp1}")
    
    # Reset limiter back to s_init
    limiter.reset(s_init, yaw_init=0.0, pitch_init=0.0, roll_init=0.0)
    
    # 2. Test a large update (angular delta > 0.005 radians)
    theta_large = 0.02
    s_large = math.cos(theta_large) * s_init + math.sin(theta_large) * s_ortho
    
    s_clamp2, R_raw2, R_clamp2, angles_clamp2, applied2 = limiter.limit_perturbation(
        s_large, yaw_new=0.02, pitch_new=0.0, roll_new=0.0
    )
    
    rho2 = np.dot(s_init.conj(), s_clamp2)
    angle_step2 = math.acos(min(1.0, abs(rho2)))
    print(f"\n[EVAL] Large Update Test (Target Step: {theta_large:.6f} rad):")
    print(f"    -> Clamping Applied: {applied2}")
    print(f"    -> Resulting Step:   {angle_step2:.6f} rad (Expected: 0.005000)")
    print(f"    -> Clamped Angles:   {angles_clamp2} (Expected Yaw: 0.005)")
    
    # Assert correctness (with float32 precision limits accounted for)
    assert abs(angle_step2 - 0.005) < 5e-5, f"Clamped step is {angle_step2}, expected 0.005"
    assert abs(angles_clamp2[0] - 0.005) < 1e-5, f"Clamped yaw is {angles_clamp2[0]}, expected 0.005"
    
    # 3. Test phase ambiguity resilience
    phase_shift = np.exp(1j * 1.234)
    s_large_phase_shifted = s_large * phase_shift
    
    # Reset limiter back to s_init
    limiter.reset(s_init, yaw_init=0.0, pitch_init=0.0, roll_init=0.0)
    
    s_clamp3, R_raw3, R_clamp3, angles_clamp3, applied3 = limiter.limit_perturbation(
        s_large_phase_shifted, yaw_new=0.02, pitch_new=0.0, roll_new=0.0
    )
    
    rho3 = np.dot(s_init.conj(), s_clamp3)
    angle_step3 = math.acos(min(1.0, abs(rho3)))
    print(f"\n[EVAL] Phase Ambiguity Test:")
    print(f"    -> Clamping Applied: {applied3}")
    print(f"    -> Resulting Step:   {angle_step3:.6f} rad (Expected: 0.005000)")
    print(f"    -> Inner Product Phase: {np.angle(rho3):.6f} rad (Expected: 0.000000)")
    assert abs(np.angle(rho3)) < 1e-5, f"Phase was not aligned, angle is {np.angle(rho3)}"
    assert abs(angle_step3 - 0.005) < 5e-5, f"Clamped step is {angle_step3}, expected 0.005"
    
    # 4. Execution Latency Benchmarks
    print("\n[EVAL] Running 10,000 performance sweeps...")
    latencies = []
    
    for i in range(10000):
        # alternate between small and large step to test both branches
        step = 0.01 if i % 2 == 0 else 0.001
        s_test = math.cos(step) * s_clamp2 + math.sin(step) * s_ortho
        
        t_start = time.perf_counter_ns()
        limiter.limit_perturbation(s_test, yaw_new=step, pitch_new=0.0, roll_new=0.0)
        t_end = time.perf_counter_ns()
        
        latencies.append((t_end - t_start) / 1000.0)
        
    avg_us = sum(latencies) / len(latencies)
    max_us = np.percentile(latencies, 99.0)
    
    print(f"    -> Average Latency: {avg_us:.3f} microseconds")
    print(f"    -> P99 Max Jitter:  {max_us:.3f} microseconds")
    
    # Assert performance bounds
    latency_ok = avg_us < 8.0
    print("\n[VERIFY] Performance and Correctness Tests:")
    if latency_ok:
        print("[PASS] Perturbation limiter executed well within the 8.0 microsecond window.")
    else:
        print("[FAIL] Limiter execution breached the 8.0 microsecond real-time envelope.")
        
    print("[PASS] Step clamping logic verified successfully.")
    print("==================================================================")
