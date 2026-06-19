"""
Task 49.1: MUSIC Spatial Projection Engine
Zero-Allocation, Vectorized 2D Multiple Signal Classification
"""

import time
import math
import numpy as np
from numba import njit

# ==============================================================================
# JIT-Compiled MUSIC Kernels
# ==============================================================================

@njit(fastmath=True, cache=True)
def _precompute_steering_vectors_kernel(
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    geometry: np.ndarray,
    A_lut: np.ndarray
) -> None:
    """
    JIT-compiled precomputation of spatial steering vectors over the coordinate grid.
    Fills a static complex64 array of shape (M_az, M_el, 4) in-place.
    """
    M_az = len(az_grid)
    M_el = len(el_grid)
    scale = np.float32(0.5) # 1/sqrt(4) for a 4-antenna receiver
    
    for i in range(M_az):
        az = np.float32(az_grid[i])
        for j in range(M_el):
            el = np.float32(el_grid[j])
            # Wave wavenumber components (w.r.t wavelength)
            kx = np.float32(math.cos(az) * math.cos(el))
            ky = np.float32(math.sin(az) * math.cos(el))
            kz = np.float32(math.sin(el))
            
            for m in range(4):
                xm = np.float32(geometry[m, 0])
                ym = np.float32(geometry[m, 1])
                zm = np.float32(geometry[m, 2])
                
                # Phase shift relative to virtual phase center (0,0,0)
                psi = -np.float32(2.0 * math.pi) * (xm * kx + ym * ky + zm * kz)
                
                # Assign to LUT in-place
                A_lut[i, j, m] = complex(scale * np.float32(math.cos(psi)), scale * np.float32(math.sin(psi)))


@njit(fastmath=True, cache=True)
def _music_projection_kernel(
    A: np.ndarray,
    E_n: np.ndarray,
    pseudospectrum: np.ndarray
) -> None:
    """
    Vectorized 2D MUSIC pseudospectrum projection over precomputed steering vectors.
    Operates in-place on pre-allocated complex64 arrays with zero runtime allocations.
    Uses manual float32 real/imaginary math to avoid complex object boxing/unboxing overhead.
    
    Args:
        A: Precomputed steering vector LUT of shape (M_az, M_el, 4) complex64.
        E_n: Noise subspace eigenvectors of shape (4, num_noise) complex64.
        pseudospectrum: Pseudospectrum array of shape (M_az, M_el) complex64.
    """
    M_az = A.shape[0]
    M_el = A.shape[1]
    num_noise = E_n.shape[1]
    
    for i in range(M_az):
        for j in range(M_el):
            den = np.float32(0.0)
            
            for col in range(num_noise):
                # Manual conjugate dot product: E_n[:, col]^H * A[i, j, :]
                val_r = (E_n[0, col].real * A[i, j, 0].real + E_n[0, col].imag * A[i, j, 0].imag +
                         E_n[1, col].real * A[i, j, 1].real + E_n[1, col].imag * A[i, j, 1].imag +
                         E_n[2, col].real * A[i, j, 2].real + E_n[2, col].imag * A[i, j, 2].imag +
                         E_n[3, col].real * A[i, j, 3].real + E_n[3, col].imag * A[i, j, 3].imag)
                         
                val_i = (E_n[0, col].real * A[i, j, 0].imag - E_n[0, col].imag * A[i, j, 0].real +
                         E_n[1, col].real * A[i, j, 1].imag - E_n[1, col].imag * A[i, j, 1].real +
                         E_n[2, col].real * A[i, j, 2].imag - E_n[2, col].imag * A[i, j, 2].real +
                         E_n[3, col].real * A[i, j, 3].imag - E_n[3, col].imag * A[i, j, 3].real)
                         
                # Add squared magnitude of projection
                den += val_r * val_r + val_i * val_i
                
            if den > np.float32(1e-15):
                val_out = np.float32(1.0) / den
            else:
                val_out = np.float32(1e15)
                
            # Store in static complex64 grid array view (setting imaginary part to zero)
            pseudospectrum[i, j] = complex(val_out, np.float32(0.0))


# ==============================================================================
# MUSIC Spatial Projector Class
# ==============================================================================

class MusicSpatialProjector:
    """
    Massively Parallel 2D MUSIC Spatial Spectrum Identification Engine.
    Executes a high-resolution angular grid sweep across configured Azimuth (-90° to +90°)
    and Elevation (0° to 90°) coordinates to map physical incoming signals.
    """
    def __init__(self, step_az_deg: float = 3.0, step_el_deg: float = 3.0):
        # Configure angular grid vectors
        self.angles_az_deg = np.arange(-90.0, 90.0 + 1e-5, step_az_deg).astype(np.float32)
        self.angles_el_deg = np.arange(0.0, 90.0 + 1e-5, step_el_deg).astype(np.float32)
        
        self.angles_az_rad = np.radians(self.angles_az_deg).astype(np.float32)
        self.angles_el_rad = np.radians(self.angles_el_deg).astype(np.float32)
        
        self.M_az = len(self.angles_az_deg)
        self.M_el = len(self.angles_el_deg)
        
        # Pre-allocate default 0.5-wavelength ULA geometry (M=4)
        self.geometry = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]
        ], dtype=np.float32)
        
        # Pre-allocate lookup table of steering vectors (M_az, M_el, 4)
        self.A_lut = np.zeros((self.M_az, self.M_el, 4), dtype=np.complex64)
        
        # Pre-allocate static complex64 pseudospectrum output matrix
        self.pseudospectrum = np.zeros((self.M_az, self.M_el), dtype=np.complex64)
        
        # Precompute LUT values
        self._recompute_lut()
        
        # Perform warmup JIT compilation trace
        self._warmup()
        
    def _recompute_lut(self):
        """Precomputes steering vectors over the coordinate grid."""
        _precompute_steering_vectors_kernel(
            self.angles_az_rad,
            self.angles_el_rad,
            self.geometry,
            self.A_lut
        )
        
    def _warmup(self):
        """Forces Numba compilation."""
        dummy_En = np.ones((4, 3), dtype=np.complex64)
        _music_projection_kernel(self.A_lut, dummy_En, self.pseudospectrum)
        self.pseudospectrum.fill(0.0)
        
    def set_geometry(self, custom_geometry: np.ndarray) -> None:
        """
        Updates the physical array geometry and recomputes the steering vector LUT.
        
        Args:
            custom_geometry: shape (4, 3) float32 array in wavelengths.
        """
        if custom_geometry.shape != (4, 3):
            raise ValueError("Geometry must be of shape (4, 3).")
        np.copyto(self.geometry, custom_geometry.astype(np.float32))
        self._recompute_lut()
        
    def project_music(self, E_n: np.ndarray) -> tuple:
        """
        Ingests the orthogonal noise-subspace eigenvectors and executes
        the JIT-compiled MUSIC grid sweep in-place.
        
        Args:
            E_n: (4, num_noise) complex64 noise-subspace matrix.
            
        Returns:
            (pseudospectrum, execution_time_us)
        """
        if E_n.ndim != 2 or E_n.shape[0] != 4:
            raise ValueError("E_n must have shape (4, num_noise).")
            
        t0 = time.perf_counter()
        _music_projection_kernel(self.A_lut, E_n, self.pseudospectrum)
        execution_us = (time.perf_counter() - t0) * 1e6
        
        return self.pseudospectrum, execution_us


# ==============================================================================
# Standalone Diagnostic Verification Harness
# ==============================================================================

if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Spatial Processing: MUSIC 2D Spatial Projector")
    print("==================================================================")
    
    # 1. Instantiate the projector
    # Use a Planar square geometry so that elevation is observable
    custom_planar_geometry = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0]
    ], dtype=np.float32)
    
    projector = MusicSpatialProjector(step_az_deg=3.0, step_el_deg=3.0)
    projector.set_geometry(custom_planar_geometry)
    print("[PASS] MUSIC Spatial Projector Initialized & JIT Compiled.")
    
    # 2. Synthesize an arriving target signal and compute its noise subspace
    # Target DOA: Azimuth = +30 degrees, Elevation = +45 degrees
    target_az = np.radians(30.0)
    target_el = np.radians(45.0)
    
    # Analytical steering vector for target DOA
    kx = math.cos(target_az) * math.cos(target_el)
    ky = math.sin(target_az) * math.cos(target_el)
    kz = math.sin(target_el)
    
    a_target = np.zeros((4, 1), dtype=np.complex64)
    scale = 0.5
    for m in range(4):
        xm, ym, zm = custom_planar_geometry[m]
        psi = -2.0 * math.pi * (xm * kx + ym * ky + zm * kz)
        a_target[m, 0] = scale * (math.cos(psi) + 1j * math.sin(psi))
        
    # Extract noise subspace (orthogonal complement of a_target) using QR decomposition
    Q, R = np.linalg.qr(a_target, mode='complete')
    # The remaining columns (from index 1 onwards) span the noise subspace
    E_n = Q[:, 1:].astype(np.complex64)
    print(f"[*] Target Signal Steering Vector (DOA: Azimuth 30°, Elevation 45°):")
    print(f"    a_target = {a_target.ravel()}")
    print(f"[*] Orthogonal Noise Subspace E_n (shape: {E_n.shape}):")
    for col in range(E_n.shape[1]):
        print(f"    Vector {col}: {E_n[:, col]}")
        # Verify orthogonality
        dot = np.vdot(a_target, E_n[:, col])
        print(f"    -> Orthogonality check (a^H * e_n): {abs(dot):.3e} (Expected: ~0.0)")
        
    # 3. Run the MUSIC projection
    pseudospectrum, us = projector.project_music(E_n)
    
    # Identify Peak coordinate in pseudospectrum
    max_idx = np.argmax(np.real(pseudospectrum))
    peak_az_idx, peak_el_idx = np.unravel_index(max_idx, pseudospectrum.shape)
    peak_az = projector.angles_az_deg[peak_az_idx]
    peak_el = projector.angles_el_deg[peak_el_idx]
    
    print("\n--- 2D MUSIC SPECTRUM SWEEP RESULTS ---")
    print(f" [>] Azimuthal Sweep Scope:  -90 to +90 Degrees")
    print(f" [>] Elevation Sweep Scope:  0 to +90 Degrees")
    print(f" [>] Grid Step Resolution:   3.0 Degrees")
    print(f" [>] Projected Grid Dimensions: {pseudospectrum.shape[0]}x{pseudospectrum.shape[1]}")
    print(f" [>] Target DOA Location:    Azimuth = +30.0°, Elevation = +45.0°")
    print(f" [>] Peak MUSIC Detected:    Azimuth = {peak_az:+.1f}°, Elevation = {peak_el:+.1f}°")
    
    accuracy_ok = (abs(peak_az - 30.0) < 1e-3) and (abs(peak_el - 45.0) < 1e-3)
    if accuracy_ok:
        print(" [PASS] Peak MUSIC spatial detection matches true target location exactly.")
    else:
        print(" [FAIL] Peak MUSIC spatial detection shifted from true target location.")
        
    # 4. Latency Benchmark Sweep
    latencies = []
    for _ in range(2000):
        _, us_bench = projector.project_music(E_n)
        latencies.append(us_bench)
        
    avg_us = sum(latencies) / len(latencies)
    max_us = np.percentile(latencies, 99.0)
    print(f"\n[EVAL] Stride Processing Latency:")
    print(f"    -> Average Execution Latency:  {avg_us:.3f} microseconds")
    print(f"    -> P99 Max Execution Jitter:   {max_us:.3f} microseconds")
    
    latency_ok = avg_us < 30.0
    if latency_ok:
        print("[PASS] 2D MUSIC grid sweep runs well under the 30.0 microsecond ceiling.")
    else:
        print("[FAIL] 2D MUSIC grid sweep breached the 30.0 microsecond real-time limit.")
        
    assert accuracy_ok, "Failing due to MUSIC resolution peak mismatch"
    assert latency_ok, "Failing due to latency limit breach"
    print("==================================================================")
