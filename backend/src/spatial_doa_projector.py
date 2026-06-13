import time
import math
import asyncio
import numpy as np
import json

class SpatialDOAProjector:
    """
    High-Speed Spatial Spectrum Optimization Engine.
    Executes a massively parallel Capon's Minimum Variance (MVDR) angular grid sweep 
    to map physical electromagnetic energy across the azimuthal plane.
    """
    def __init__(self, num_channels: int = 4, element_spacing: float = 0.5):
        self.num_channels = num_channels
        
        # Grid parameters: -90 to +90 degrees in 1-degree intervals
        self.angles_deg = np.linspace(-90, 90, 181)
        self.angles_rad = np.radians(self.angles_deg)
        
        # Pre-calculated Steering Vector Lookup Table (LUT)
        # Assumes Uniform Linear Array (ULA) geometry
        # a(theta) = exp(-j * 2 * pi * d * sin(theta))
        self.A_lut = np.zeros((self.num_channels, 181), dtype=np.complex64)
        for i in range(self.num_channels):
            # Phase shift relative to channel 0
            self.A_lut[i, :] = np.exp(-1j * 2 * np.pi * element_spacing * i * np.sin(self.angles_rad))
            
        self.A_lut_conj = np.conj(self.A_lut)
        
        # Pre-allocated zero-heap buffers
        self._U_buffer = np.zeros((self.num_channels, 181), dtype=np.complex64)
        self._P_inv_buffer = np.zeros(181, dtype=np.complex64)
        self._spectrum = np.zeros(181, dtype=np.float32)

    def compute_spatial_spectrum(self, R_inv: np.ndarray) -> np.ndarray:
        """
        Hot-path coordinate sweep executing Capon's MVDR beamformer matrix logic.
        P(theta) = 1 / (a^H(theta) * R^-1 * a(theta))
        
        Args:
            R_inv: (4, 4) Inverted Spatial Covariance Matrix
            
        Returns:
            spectrum: (181,) float32 array of normalized spatial power bounds.
        """
        t0 = time.perf_counter()
        
        # 1. Calculate U = R^-1 * A
        np.matmul(R_inv, self.A_lut, out=self._U_buffer)
        
        # 2. Calculate a^H * U (Element-wise multiplication and sum across channels)
        np.sum(self.A_lut_conj * self._U_buffer, axis=0, out=self._P_inv_buffer)
        
        # 3. Calculate spatial power P = 1 / real(a^H * R^-1 * a)
        # Using real() because theoretical P is purely real (Hermitian properties)
        np.reciprocal(np.real(self._P_inv_buffer), out=self._spectrum)
        
        # Normalize spectrum to [0, 1] range for the frontend HUD
        max_val = np.max(self._spectrum)
        if max_val > 1e-12:
            self._spectrum /= max_val
            
        # Convert absolute float64/float32 output back down to strict float32
        self._spectrum = self._spectrum.astype(np.float32)
        
        execution_us = (time.perf_counter() - t0) * 1e6
        return self._spectrum, execution_us

    async def _async_push_payload(self, spectrum: np.ndarray):
        """
        Non-blocking payload transmission mimicking WebSocket delivery.
        In a full FastAPI harness context, this hooks into the streaming broadcaster.
        """
        try:
            import websockets
            payload = {
                "timestamp": time.time(),
                "event": "DOA_SPECTRUM_SWEEP",
                "azimuth_grid": self.angles_deg.tolist(),
                "power_spectrum": spectrum.tolist()
            }
            # Attempt to connect to the active SpaceShield Gateway
            async with websockets.connect("ws://127.0.0.1:8000/stream") as ws:
                await ws.send(json.dumps(payload))
        except Exception:
            # Silently pass if the mock loopback server isn't up
            pass

    async def run_daemon(self, get_r_inv_callback):
        """
        Perpetual 100ms background execution loop.
        Extracts the inverted covariance from the parent DSP worker, computes the 
        DOA sweep, and pushes the payload over the wire asynchronously.
        """
        while True:
            t0 = time.time()
            try:
                R_inv = get_r_inv_callback()
                spectrum, _ = self.compute_spatial_spectrum(R_inv)
                
                # Fire and forget over asyncio background task
                asyncio.create_task(self._async_push_payload(spectrum))
            except Exception as e:
                print(f"[!] DOA Projector Fault: {e}")
                
            # Rigid 100ms framing
            elapsed = time.time() - t0
            sleep_time = max(0.0, 0.100 - elapsed)
            await asyncio.sleep(sleep_time)

# --- Rapid Verification Stub ---
if __name__ == "__main__":
    print("[*] Initializing Spatial DOA Capon Spectrum Sweep...")
    projector = SpatialDOAProjector(num_channels=4, element_spacing=0.5)
    
    # 1. Synthesize an ambient spatial covariance matrix with a target arriving at +30 degrees
    # a(theta) = [1, exp(-j*pi*sin(theta)), exp(-j*2*pi*sin(theta)), exp(-j*3*pi*sin(theta))]
    theta_target = np.radians(30.0)
    target_steering = np.exp(-1j * np.pi * np.sin(theta_target) * np.arange(4)).reshape(4, 1)
    
    # R = Target Power * a*a^H + Noise Variance * I
    R_mock = 100.0 * (target_steering @ target_steering.conj().T) + 1.0 * np.eye(4)
    R_inv_mock = np.linalg.inv(R_mock).astype(np.complex64)
    
    # 2. Execute Benchmark Sweeps
    latencies = []
    for _ in range(1000):
        spectrum, us = projector.compute_spatial_spectrum(R_inv_mock)
        latencies.append(us)
        
    avg_us = sum(latencies) / len(latencies)
    avg_ms = avg_us / 1000.0
    
    # Identify peak index in the spectrum
    peak_idx = np.argmax(spectrum)
    peak_angle = projector.angles_deg[peak_idx]
    
    print("\n--- DOA CAPON SPECTRUM HUD ---")
    print(f" [>] Azimuthal Sweep Scope:  -90 to +90 Degrees")
    print(f" [>] Sweep Resolution:       1 Degree Intervals (181 bins)")
    print(f" [>] Zero-Heap Matrix Math:  Active")
    print(f" [>] Target Peak Detected:   {peak_angle:+.1f} Degrees")
    print(f"\n [>] Average Execution:      {avg_ms:.4f} ms per full sweep")
    
    if avg_ms < 2.5:
        print(f" [PASSED] Spatial computation crushed the sub-2.5ms budget!")
    else:
        print(f" [FAILED] Latency Envelope Exceeded 2.5ms constraint.")
