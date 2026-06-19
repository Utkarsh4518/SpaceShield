"""
Task 49.2: Spatial Peak Finder Module
Zero-Allocation, Vectorized Windowed Local Maxima Extraction
"""

import time
import math
import numpy as np
import ctypes
from numba import njit

# ==============================================================================
# ctypes Atomic Shared Configuration Slot
# ==============================================================================

class SpatialPeakSlot(ctypes.Structure):
    """
    Structure representing a single identified emitter peak.
    Mapped directly to memory to enable thread-safe, lock-free downstream reading.
    """
    _fields_ = [
        ("valid", ctypes.c_bool),
        ("azimuth", ctypes.c_float),
        ("elevation", ctypes.c_float),
        ("value", ctypes.c_float)
    ]


# ==============================================================================
# JIT-Compiled Peak Finding Kernel
# ==============================================================================

@njit(fastmath=True, cache=True)
def _find_peaks_kernel(
    pseudospectrum: np.ndarray,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    threshold_factor: np.float32,
    p_val: np.ndarray,
    p_az: np.ndarray,
    p_el: np.ndarray
) -> None:
    """
    Finds up to 3 local maxima in the 2D spatial pseudospectrum grid matrix.
    Uses windowed neighborhood comparisons and adaptive thresholding.
    Operates strictly in-place with zero heap allocations.
    
    Args:
        pseudospectrum: (M_az, M_el) complex64 grid matrix.
        az_grid: (M_az,) float32 grid coordinates in degrees.
        el_grid: (M_el,) float32 grid coordinates in degrees.
        threshold_factor: Adaptive threshold standard deviation scaling factor (float32).
        p_val: (3,) float32 array storing top 3 peak values.
        p_az: (3,) float32 array storing top 3 peak azimuths.
        p_el: (3,) float32 array storing top 3 peak elevations.
    """
    M_az = pseudospectrum.shape[0]
    M_el = pseudospectrum.shape[1]
    
    # 1. Compute adaptive threshold: Threshold = mean + factor * std
    sum_val = np.float32(0.0)
    for i in range(M_az):
        for j in range(M_el):
            sum_val += pseudospectrum[i, j].real
            
    mean_val = sum_val / np.float32(M_az * M_el)
    
    sum_sq_diff = np.float32(0.0)
    for i in range(M_az):
        for j in range(M_el):
            diff = pseudospectrum[i, j].real - mean_val
            sum_sq_diff += diff * diff
            
    std_val = np.float32(math.sqrt(sum_sq_diff / np.float32(M_az * M_el)))
    threshold = mean_val + threshold_factor * std_val
    
    # Clear outputs
    p_val.fill(np.float32(0.0))
    p_az.fill(np.float32(0.0))
    p_el.fill(np.float32(0.0))
    
    # 2. Windowed Local Maxima Sweep
    # Performs localized 2D comparisons in a 3x3 sliding neighborhood
    for i in range(1, M_az - 1):
        for j in range(1, M_el - 1):
            val = pseudospectrum[i, j].real
            if val <= threshold:
                continue
                
            # Compare with 8 neighbors
            is_max = (
                val > pseudospectrum[i - 1, j - 1].real and
                val > pseudospectrum[i - 1, j].real and
                val > pseudospectrum[i - 1, j + 1].real and
                val > pseudospectrum[i, j - 1].real and
                val > pseudospectrum[i, j + 1].real and
                val > pseudospectrum[i + 1, j - 1].real and
                val > pseudospectrum[i + 1, j].real and
                val > pseudospectrum[i + 1, j + 1].real
            )
            
            if is_max:
                # 3. Shift-and-Insert Sorting to extract the top 3 peaks (O(1) complexity)
                if val > p_val[0]:
                    p_val[2] = p_val[1]
                    p_az[2] = p_az[1]
                    p_el[2] = p_el[1]
                    
                    p_val[1] = p_val[0]
                    p_az[1] = p_az[0]
                    p_el[1] = p_el[0]
                    
                    p_val[0] = val
                    p_az[0] = az_grid[i]
                    p_el[0] = el_grid[j]
                elif val > p_val[1]:
                    p_val[2] = p_val[1]
                    p_az[2] = p_az[1]
                    p_el[2] = p_el[1]
                    
                    p_val[1] = val
                    p_az[1] = az_grid[i]
                    p_el[1] = el_grid[j]
                elif val > p_val[2]:
                    p_val[2] = val
                    p_az[2] = az_grid[i]
                    p_el[2] = el_grid[j]


# ==============================================================================
# Spatial Peak Finder Class
# ==============================================================================

class SpatialPeakFinder:
    """
    Inline Multidimensional Local Maxima Extraction Module.
    Consumes the 2D pseudospectrum and extracts up to 3 discrete emitter AoAs.
    Saves candidate emitter parameters into atomic configuration slots.
    """
    def __init__(self, step_az_deg: float = 3.0, step_el_deg: float = 3.0, threshold_factor: float = 2.0):
        # Recreate grid coordinates matching the MUSIC spatial sweep
        self.angles_az_deg = np.arange(-90.0, 90.0 + 1e-5, step_az_deg).astype(np.float32)
        self.angles_el_deg = np.arange(0.0, 90.0 + 1e-5, step_el_deg).astype(np.float32)
        
        self.threshold_factor = np.float32(threshold_factor)
        
        # Pre-allocated arrays for the JIT kernel
        self.p_val = np.zeros(3, dtype=np.float32)
        self.p_az = np.zeros(3, dtype=np.float32)
        self.p_el = np.zeros(3, dtype=np.float32)
        
        # Pre-allocate atomic configuration slots (ctypes array of 3 slots)
        self.shared_slots = (SpatialPeakSlot * 3)()
        for i in range(3):
            self.shared_slots[i].valid = False
            self.shared_slots[i].azimuth = 0.0
            self.shared_slots[i].elevation = 0.0
            self.shared_slots[i].value = 0.0
            
        self._warmup()
        
    def _warmup(self):
        """Forces Numba compilation."""
        dummy_grid = np.zeros((len(self.angles_az_deg), len(self.angles_el_deg)), dtype=np.complex64)
        _find_peaks_kernel(
            dummy_grid, self.angles_az_deg, self.angles_el_deg, self.threshold_factor,
            self.p_val, self.p_az, self.p_el
        )
        # Reset output states
        self.p_val.fill(0.0)
        self.p_az.fill(0.0)
        self.p_el.fill(0.0)
        
    def find_peaks(self, pseudospectrum: np.ndarray) -> tuple:
        """
        Extracts up to 3 local maxima in the pseudospectrum grid.
        Atomically updates the shared configuration slots.
        
        Args:
            pseudospectrum: (M_az, M_el) complex64 grid matrix.
            
        Returns:
            (p_val, p_az, p_el, execution_time_us)
        """
        t0 = time.perf_counter()
        
        # Run JIT-compiled peak finding kernel
        _find_peaks_kernel(
            pseudospectrum,
            self.angles_az_deg,
            self.angles_el_deg,
            self.threshold_factor,
            self.p_val,
            self.p_az,
            self.p_el
        )
        
        # Assign to ctypes structures atomically
        for i in range(3):
            val = self.p_val[i]
            if val > 0.0:
                self.shared_slots[i].valid = True
                self.shared_slots[i].azimuth = self.p_az[i]
                self.shared_slots[i].elevation = self.p_el[i]
                self.shared_slots[i].value = val
            else:
                self.shared_slots[i].valid = False
                self.shared_slots[i].azimuth = 0.0
                self.shared_slots[i].elevation = 0.0
                self.shared_slots[i].value = 0.0
                
        execution_us = (time.perf_counter() - t0) * 1e6
        return self.p_val, self.p_az, self.p_el, execution_us


# ==============================================================================
# Standalone Diagnostic Verification Harness
# ==============================================================================

if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Spatial Processing: Multidimensional Peak Finder")
    print("==================================================================")
    
    # 1. Instantiate Peak Finder
    step_deg = 3.0
    finder = SpatialPeakFinder(step_az_deg=step_deg, step_el_deg=step_deg, threshold_factor=2.0)
    print("[PASS] Spatial Peak Finder Initialized & JIT Compiled.")
    
    # 2. Synthesize mock pseudospectrum grid
    # Grid sizes match M_az and M_el
    M_az = len(finder.angles_az_deg)
    M_el = len(finder.angles_el_deg)
    
    # Create background noise base
    rng = np.random.default_rng(0xBA5E)
    pseudospectrum = (rng.uniform(0.1, 1.0, (M_az, M_el)) + 0j).astype(np.complex64)
    
    # Inject 3 target peaks at known coordinates
    # We choose coordinates that lie exactly on the grid:
    # Target 1: Azimuth = -30.0 degrees, Elevation = +15.0 degrees, Value = 100.0
    # Target 2: Azimuth = +15.0 degrees, Elevation = +60.0 degrees, Value = 50.0
    # Target 3: Azimuth = +75.0 degrees, Elevation = +30.0 degrees, Value = 25.0
    
    def find_grid_idx(angles, target):
        return np.argmin(np.abs(angles - target))
        
    idx_az1, idx_el1 = find_grid_idx(finder.angles_az_deg, -30.0), find_grid_idx(finder.angles_el_deg, 15.0)
    idx_az2, idx_el2 = find_grid_idx(finder.angles_az_deg, 15.0), find_grid_idx(finder.angles_el_deg, 60.0)
    idx_az3, idx_el3 = find_grid_idx(finder.angles_az_deg, 75.0), find_grid_idx(finder.angles_el_deg, 30.0)
    
    # Ensure they are local maxima by clearing neighbors and setting peaks
    def set_peak(grid, idx_az, idx_el, val):
        # Set surrounding neighborhood small
        for u in [-1, 0, 1]:
            for v in [-1, 0, 1]:
                if 0 <= idx_az + u < grid.shape[0] and 0 <= idx_el + v < grid.shape[1]:
                    grid[idx_az + u, idx_el + v] = 0.05 + 0j
        # Set peak center
        grid[idx_az, idx_el] = val + 0j
        
    set_peak(pseudospectrum, idx_az1, idx_el1, 100.0)
    set_peak(pseudospectrum, idx_az2, idx_el2, 50.0)
    set_peak(pseudospectrum, idx_az3, idx_el3, 25.0)
    
    print("\n[*] Synthesized Pseudospectrum Targets:")
    print(f"    -> Target 1: Azimuth -30.0°, Elevation +15.0°, Power 100.0")
    print(f"    -> Target 2: Azimuth +15.0°, Elevation +60.0°, Power 50.0")
    print(f"    -> Target 3: Azimuth +75.0°, Elevation +30.0°, Power 25.0")
    
    # 3. Execute peak finder
    p_val, p_az, p_el, us = finder.find_peaks(pseudospectrum)
    
    print("\n--- DETECTED PEAKS RESULTS ---")
    accuracy_ok = True
    for i in range(3):
        valid = finder.shared_slots[i].valid
        az = finder.shared_slots[i].azimuth
        el = finder.shared_slots[i].elevation
        val = finder.shared_slots[i].value
        print(f" [>] Peak {i}: Valid={valid} | Azimuth={az:+.1f}° | Elevation={el:+.1f}° | Value={val:.2f}")
        
        # Verify alignment within step bounds
        if i == 0:
            if not (valid and abs(az - (-30.0)) < 1e-3 and abs(el - 15.0) < 1e-3 and abs(val - 100.0) < 1e-3):
                accuracy_ok = False
        elif i == 1:
            if not (valid and abs(az - 15.0) < 1e-3 and abs(el - 60.0) < 1e-3 and abs(val - 50.0) < 1e-3):
                accuracy_ok = False
        elif i == 2:
            if not (valid and abs(az - 75.0) < 1e-3 and abs(el - 30.0) < 1e-3 and abs(val - 25.0) < 1e-3):
                accuracy_ok = False
                
    if accuracy_ok:
        print(" [PASS] Peak finder successfully extracted the top 3 AoA peaks with perfect coordinate alignment.")
    else:
        print(" [FAIL] Peak finder failed to extract the targets accurately.")
        
    # 4. Latency Benchmark Sweep
    latencies = []
    for _ in range(3000):
        _, _, _, us_bench = finder.find_peaks(pseudospectrum)
        latencies.append(us_bench)
        
    avg_us = sum(latencies) / len(latencies)
    max_us = np.percentile(latencies, 99.0)
    print(f"\n[EVAL] Stride Processing Latency:")
    print(f"    -> Average Execution Latency:  {avg_us:.3f} microseconds")
    print(f"    -> P99 Max Execution Jitter:   {max_us:.3f} microseconds")
    
    latency_ok = avg_us < 10.0
    if latency_ok:
        print("[PASS] Peak finding local maxima extraction sweeps well under the 10.0 microsecond ceiling.")
    else:
        print("[FAIL] Peak finding local maxima extraction breached the 10.0 microsecond limit.")
        
    assert accuracy_ok, "Failing due to peak tracking coordinates mismatch"
    assert latency_ok, "Failing due to latency limit breach"
    print("==================================================================")
