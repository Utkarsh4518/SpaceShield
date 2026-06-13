import time
import numpy as np
import logging

logger = logging.getLogger("DopplerDriftTracker")
logger.setLevel(logging.INFO)

class DopplerDriftTracker:
    """
    Orbital Mechanics Kinematic Discriminator.
    Implements a multi-dimensional Alpha-Beta-Gamma filter to track 
    Doppler state variables natively across all 4 spatial channels.
    Since the NavIC constellation resides in Geosynchronous/Geostationary 
    (GSO/GEO) orbits, their relative radial acceleration toward the receiver 
    is mathematically infinitesimal. Any detected Doppler 'Jerk' or high 
    acceleration indicates a terrestrial spoofer (e.g., a drone or 
    moving vehicle) spoofing the signal, triggering an immediate anomaly flag.
    """
    def __init__(self, num_channels: int = 4, stride_dt: float = 0.001024, 
                 alpha: float = 0.1, beta: float = 0.005, gamma: float = 0.0001,
                 acceleration_threshold_hz_s2: float = 2.5):
        self.num_channels = num_channels
        self.dt = stride_dt
        self.dt_sq = self.dt ** 2
        
        # Kinematic State Vectors (4 channels)
        # x_0: Doppler Offset (Hz)
        # x_1: Doppler Velocity / Drift Rate (Hz/s)
        # x_2: Doppler Acceleration / Jerk (Hz/s^2)
        self.x_0 = np.zeros(self.num_channels, dtype=np.float32)
        self.x_1 = np.zeros(self.num_channels, dtype=np.float32)
        self.x_2 = np.zeros(self.num_channels, dtype=np.float32)
        
        # Filter Gains
        self.alpha = np.float32(alpha)
        self.beta_scaled = np.float32(beta / self.dt)
        self.gamma_scaled = np.float32((2.0 * gamma) / self.dt_sq)
        
        # Threat Classification bounds
        self.accel_threshold = np.float32(acceleration_threshold_hz_s2)
        
        # Pre-allocated arrays for zero-heap operations
        self._x_0_pred = np.zeros(self.num_channels, dtype=np.float32)
        self._x_1_pred = np.zeros(self.num_channels, dtype=np.float32)
        self._residual = np.zeros(self.num_channels, dtype=np.float32)

    def process_doppler_observations(self, doppler_hz_array: np.ndarray) -> tuple:
        """
        Hot-path execution tracker. Runs instantly after carrier_lock_flywheel extracts 
        the phase boundaries to maintain constant situational awareness.
        
        Args:
            doppler_hz_array: (4,) Float32 array containing instantaneous Doppler measurements.
            
        Returns:
            (is_spoofing_flag, execution_us)
        """
        t0 = time.perf_counter()
        
        # 1. Kinematic Prediction
        # x_0_pred = x_0 + x_1 * dt + 0.5 * x_2 * dt^2
        np.multiply(self.x_2, 0.5 * self.dt_sq, out=self._x_0_pred)
        self._x_0_pred += (self.x_1 * self.dt)
        self._x_0_pred += self.x_0
        
        # x_1_pred = x_1 + x_2 * dt
        np.multiply(self.x_2, self.dt, out=self._x_1_pred)
        self._x_1_pred += self.x_1
        
        # 2. Measurement Residual
        np.subtract(doppler_hz_array, self._x_0_pred, out=self._residual)
        
        # 3. State Update
        # x_0 = x_0_pred + alpha * r
        self.x_0 = self._x_0_pred + (self.alpha * self._residual)
        
        # x_1 = x_1_pred + beta * r
        self.x_1 = self._x_1_pred + (self.beta_scaled * self._residual)
        
        # x_2 = x_2_pred + gamma * r
        self.x_2 = self.x_2 + (self.gamma_scaled * self._residual)
        
        # 4. Kinematic Orbital Classification
        # GEO satellites physically cannot accelerate rapidly toward a receiver
        # Check if the absolute Doppler Acceleration exceeds terrestrial physics threshold
        spoofing_flags = np.abs(self.x_2) > self.accel_threshold
        is_spoofing = bool(np.any(spoofing_flags))
        
        execution_us = (time.perf_counter() - t0) * 1e6
        
        return is_spoofing, execution_us

# --- Rapid Verification Stub ---
if __name__ == "__main__":
    print("[*] Initializing Kinematic Orbital Discriminator...")
    tracker = DopplerDriftTracker(num_channels=4, stride_dt=0.001024, acceleration_threshold_hz_s2=2.5)
    
    # 1. Simulate Clean NavIC GSO Baseline (Very slow natural drift, ~0.001 Hz/s^2)
    # Give the filter 5000 strides to converge
    base_doppler = np.array([1200.0, 1200.0, 1200.0, 1200.0], dtype=np.float32)
    t = 0.0
    
    for _ in range(5000):
        # Sine wave creating very slow GEO orbital drift mechanics
        geo_doppler = base_doppler + np.sin(t * 0.01) * 10.0
        tracker.process_doppler_observations(geo_doppler)
        t += 0.001024
        
    print("\n--- NAVIC GSO CONVERGED KINEMATICS ---")
    print(f" [>] Tracked Mean Doppler:       {np.mean(tracker.x_0):.2f} Hz")
    print(f" [>] Tracked Radial Velocity:    {np.mean(tracker.x_1):.4f} Hz/s")
    print(f" [>] Tracked Orbital Jerk:       {np.mean(tracker.x_2):.4f} Hz/s^2")
    
    # 2. Benchmark the Filter tensor matrix latency
    latencies = []
    for _ in range(2000):
        _, us = tracker.process_doppler_observations(base_doppler)
        latencies.append(us)
        
    avg_us = sum(latencies) / len(latencies)
    print(f" [>] Matrix Filter Execution:    {avg_us:.2f} µs")
    
    # 3. Simulate High-Acceleration Moving UAV Spoofer
    # A drone rapidly accelerates radially toward the receiver (+5g equivalent radial turn)
    print("\n[*] Injecting Kinematic 'KINEMATIC_SPOOFING_VECTOR' (Terrestrial Drone Profile)...")
    
    # Reset filter state to allow clear reading
    tracker.x_1[:] = 0.0
    tracker.x_2[:] = 0.0
    
    is_spoofed = False
    spoofed_stride = 0
    
    spoofer_doppler = base_doppler.copy()
    
    for stride in range(100): # Spoofer accelerates aggressively
        # Adding parabolic acceleration to the Doppler
        spoofer_doppler += (0.5 * 15.0 * (0.001024 ** 2)) * (stride ** 2) 
        
        is_spoofed, us = tracker.process_doppler_observations(spoofer_doppler)
        if is_spoofed:
            spoofed_stride = stride
            break
            
    print("\n--- THREAT CLASSIFICATION MATRIX ---")
    print(f" [>] Measured Spoofer Jerk:      {np.max(tracker.x_2):.4f} Hz/s^2")
    print(f" [>] Trigger Latency:            Detected in {spoofed_stride} strides ({(spoofed_stride * 1.024):.2f} ms)")
    print(f" [>] Anomaly Class output:       {'KINEMATIC_SPOOFING_VECTOR' if is_spoofed else 'CLEAN'}")
    
    if avg_us < 20.0 and is_spoofed:
        print(f"\n[PASSED] Kinematic Orbital Discriminator effectively trapped the terrestrial threat within execution bounds!")
    else:
        print(f"\n[FAILED] Constraints breached.")
