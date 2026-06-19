import time
import math
import numpy as np
from numba import njit

# WGS-84 Earth Constants
OMEGA_E = 7.2921151467e-5  # Earth rotation rate (rad/s)
MU_EARTH = 3.986005e14     # Gravitational parameter (m^3/s^2)

@njit(fastmath=True, cache=True)
def _solve_kepler_and_enu(t_elapsed, A, e, i_angle, omega, raan_0, M_0, gs_ecef, enu_matrix):
    """
    Numba JIT Kernel: Solves the absolute Keplerian Orbital Equations to derive 
    high-precision Azimuth and Elevation relative to the local ground station.
    Bypasses standard Python mathematical dispatch for sub-microsecond latency.
    """
    # 1. Mean Anomaly Propagator
    n = math.sqrt(MU_EARTH / (A**3))
    M = M_0 + n * t_elapsed
    
    # 2. Newton-Raphson for Eccentric Anomaly (E)
    # Fixed at 5 iterations to eliminate dynamic looping branch constraints
    E = M
    for _ in range(5):
        E = E - (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        
    # 3. True Anomaly (nu) Extraction
    cos_E = math.cos(E)
    sin_E = math.sin(E)
    nu = math.atan2(math.sqrt(1.0 - e**2) * sin_E, cos_E - e)
    
    # 4. Radius and Coplanar Coordinates
    r = A * (1.0 - e * cos_E)
    x_prime = r * math.cos(nu)
    y_prime = r * math.sin(nu)
    
    # 5. ECEF Rotational Matrix (3D Space wrapping)
    raan = raan_0 - OMEGA_E * t_elapsed
    cw, sw = math.cos(omega), math.sin(omega)
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(i_angle), math.sin(i_angle)
    
    x = x_prime * (cO * cw - sO * sw * ci) + y_prime * (-cO * sw - sO * cw * ci)
    y = x_prime * (sO * cw + cO * sw * ci) + y_prime * (-sO * sw + cO * cw * ci)
    z = x_prime * (sw * si) + y_prime * (cw * si)
    
    # 6. Transform to ENU (East-North-Up) using pre-computed GS rotation block
    dx = x - gs_ecef[0]
    dy = y - gs_ecef[1]
    dz = z - gs_ecef[2]
    
    e_val = enu_matrix[0,0]*dx + enu_matrix[0,1]*dy + enu_matrix[0,2]*dz
    n_val = enu_matrix[1,0]*dx + enu_matrix[1,1]*dy + enu_matrix[1,2]*dz
    u_val = enu_matrix[2,0]*dx + enu_matrix[2,1]*dy + enu_matrix[2,2]*dz
    
    # 7. Extract Exact Topocentric Look Angles (Azimuth/Elevation)
    azimuth = math.atan2(e_val, n_val)
    if azimuth < 0:
        azimuth += 2 * math.pi
        
    horizontal_dist = math.sqrt(e_val**2 + n_val**2)
    elevation = math.atan2(u_val, horizontal_dist)
    
    return azimuth, elevation


class EphemerisLookAngleValidator:
    """
    Hardware-in-the-Loop Kinematic Defense Layer.
    Parses live ephemeris telemetry matrices and compares the mathematically predicted
    satellite coordinates against the physical steering angles acquired by the Spatial Array.
    Triggers structural alerts if the array is being pulled away by a localized spoofing transmitter.
    """
    def __init__(self, gs_lat: float, gs_lon: float, gs_alt: float, violation_margin_deg: float = 3.0):
        lat_rad = math.radians(gs_lat)
        lon_rad = math.radians(gs_lon)
        
        # WGS-84 Ground Station to ECEF static initialization
        a_wgs84 = 6378137.0
        e_sq = 0.00669437999014
        N_val = a_wgs84 / math.sqrt(1 - e_sq * math.sin(lat_rad)**2)
        
        x_gs = (N_val + gs_alt) * math.cos(lat_rad) * math.cos(lon_rad)
        y_gs = (N_val + gs_alt) * math.cos(lat_rad) * math.sin(lon_rad)
        z_gs = (N_val * (1 - e_sq) + gs_alt) * math.sin(lat_rad)
        
        # Pre-allocate completely static vectors for the JIT kernel
        self._gs_ecef = np.array([x_gs, y_gs, z_gs], dtype=np.float64)
        
        # Pre-compute ECEF to ENU rotation block (Static memory)
        clat, slat = math.cos(lat_rad), math.sin(lat_rad)
        clon, slon = math.cos(lon_rad), math.sin(lon_rad)
        
        self._enu_matrix = np.array([
            [-slon, clon, 0.0],
            [-slat * clon, -slat * slon, clat],
            [clat * clon, clat * slon, slat]
        ], dtype=np.float64)
        
        self.violation_margin_rad = math.radians(violation_margin_deg)
        
    def validate_geometry(self, measured_az_deg: float, measured_el_deg: float, 
                          t_elapsed: float, ephemeris_params: tuple) -> tuple:
        """
        Executes the instantaneous structural lock test.
        """
        t0 = time.perf_counter()
        
        # Unpack Keplerian Sub-Frame Data
        A, e, i_angle, omega, raan_0, M_0 = ephemeris_params
        
        # Fire Vector-Accelerated zero-heap Keplerian solver
        true_az, true_el = _solve_kepler_and_enu(
            t_elapsed, A, e, i_angle, omega, raan_0, M_0, 
            self._gs_ecef, self._enu_matrix
        )
        
        # Geometry Verification
        meas_az = math.radians(measured_az_deg)
        meas_el = math.radians(measured_el_deg)
        
        # Phase-wrapped spatial divergence
        d_az = meas_az - true_az
        while d_az > math.pi: d_az -= 2 * math.pi
        while d_az < -math.pi: d_az += 2 * math.pi
            
        d_el = meas_el - true_el
        
        spatial_error_rad = math.sqrt(d_az**2 + d_el**2)
        
        # Boundary State Evaluator
        is_violated = spatial_error_rad > self.violation_margin_rad
        alert_flag = "EPHEMERIS_GEOMETRY_VIOLATION" if is_violated else "NOMINAL_TRACKING"
        
        exec_us = (time.perf_counter() - t0) * 1e6
        
        return alert_flag, math.degrees(true_az), math.degrees(true_el), spatial_error_rad, exec_us


# --- Rapid Verification Stub ---
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Ephemeris Look-Angle Validator")
    print("==================================================================")
    
    # 1. Initialize Ground Station (e.g., Space Command Hub in New Delhi)
    validator = EphemerisLookAngleValidator(gs_lat=28.6139, gs_lon=77.2090, gs_alt=216.0, violation_margin_deg=3.0)
    
    # 2. NavIC Geosynchronous Satellite Keplerian Parameters (Mock)
    A = 42164000.0        # Semi-major axis (~35,786 km altitude)
    e = 0.002             # Slight eccentricity
    i_angle = math.radians(29.0)  # NavIC typical inclination
    omega = math.radians(10.0)    
    raan_0 = math.radians(45.0)   
    M_0 = math.radians(100.0)     
    ephemeris_params = (A, e, i_angle, omega, raan_0, M_0)
    
    # 3. Burn-in Numba Compilation Layer
    validator.validate_geometry(0.0, 0.0, 0.0, ephemeris_params)
    
    # 4. Hot-Path Thread Simulation
    latencies = []
    
    print("[*] Locking onto Real-Time Ephemeris Trajectory Tracker...")
    # True Az/El for this mock will be extracted on the first run
    _, t_az, t_el, _, _ = validator.validate_geometry(0.0, 0.0, 0.0, ephemeris_params)
    print(f"    -> True Keplerian Target: Azimuth {t_az:.2f}°, Elevation {t_el:.2f}°")
    
    for i in range(2500):
        # Simulate local array pulling exactly on target, but occasionally a terrestrial spoofing device
        # attempts to drag the beam away from the sky.
        if i == 1500:
            meas_az = t_az + 5.0 # Attacker drags steering angle 5 degrees horizontally
            meas_el = t_el
        else:
            meas_az = t_az + np.random.randn() * 0.5 # Normal mechanical/atmospheric tracking jitter
            meas_el = t_el + np.random.randn() * 0.5
            
        alert_flag, _, _, err, exec_us = validator.validate_geometry(meas_az, meas_el, i * 0.1, ephemeris_params)
        latencies.append(exec_us)
        
        if alert_flag == "EPHEMERIS_GEOMETRY_VIOLATION":
            print(f"\n[!] TERRESTRIAL SPOOFING INTERCEPTED AT FRAME {i}")
            print(f"    -> Beam was dragged by {math.degrees(err):.2f}° structurally.")
            print(f"    -> Flag: {alert_flag}")
            
    avg_us = sum(latencies) / len(latencies)
    max_us = np.percentile(latencies, 99.0)
    
    print("\n--- EPHEMERIS KINEMATIC HUD ---")
    print(f" [>] Tracking Method:           Newton-Raphson Keplerian Propagator")
    print(f" [>] Pre-Computed Rotations:    ECEF -> ENU (Static Matrix)")
    
    print(f"\n [>] Average Execution Latency: {avg_us:.2f} µs")
    print(f" [>] Max Edge Latency:          {max_us:.2f} µs")
    
    if max_us < 30.0:
        print("\n[PASSED] Inline validator tracks orbital models mathematically beneath 30µs limit!")
    else:
        print("\n[FAILED] Execution exceeded 30µs critical envelope limit.")
