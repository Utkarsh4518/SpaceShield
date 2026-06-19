import time
import threading
import numpy as np
import scipy.linalg
import logging

logger = logging.getLogger("BlindArrayCalibrator")
logger.setLevel(logging.INFO)

class BlindArrayCalibrator:
    """
    Adaptive Steering Vector Tracking Engine.
    Operates as an asynchronous background daemon, utilizing dominant eigenvector 
    extraction (Power Iteration / Eigendecomposition) to track micro-phase drifts 
    across the physical array structure. This continuously re-anchors the expected 
    look-direction of NavIC satellites, preventing target-signal cancellation 
    (Signal-of-Interest degradation) inside the MVDR spatial nulling filters.
    """
    def __init__(self, num_channels: int = 4, tracking_rate_hz: float = 10.0):
        self.num_channels = num_channels
        self.tracking_interval = 1.0 / tracking_rate_hz
        
        # Lock-free atomic swap buffers for thread safety
        self._latest_R = np.eye(self.num_channels, dtype=np.complex64)
        self._new_R_available = False
        
        # The true calibrated steering vector expected by downstream equalizers
        # Factory Genesis Initialization: Broadside Nominal Look
        self._calibrated_steering_vector = np.ones((self.num_channels, 1), dtype=np.complex64)
        
        # Background Tracking Daemon
        self._running = True
        self._worker_thread = threading.Thread(target=self._background_optimization_daemon, daemon=True)
        self._worker_thread.start()

    def ingest_tracking_matrix(self, R: np.ndarray):
        """
        O(1) Lock-free ingestion. 
        Taps the spatial covariance matrix out of the phase coherence pipeline.
        """
        # Shallow copy pointer swap is atomic in CPython
        self._latest_R = R
        self._new_R_available = True

    def get_calibrated_steering_vector(self) -> np.ndarray:
        """Returns the dynamically tracked phase-drift offset vector."""
        return self._calibrated_steering_vector

    def _background_optimization_daemon(self):
        """
        Decoupled Low-Priority Execution Loop.
        Uses Power Iteration or Eigendecomposition to extract the principal subspace.
        """
        # Lower thread priority if operating on POSIX architecture
        try:
            import os
            if hasattr(os, 'nice'):
                os.nice(10)
        except Exception:
            pass
            
        while self._running:
            t0 = time.time()
            
            if self._new_R_available:
                self._new_R_available = False
                
                # Snapshot the current matrix to avoid asynchronous tearing
                R_snapshot = self._latest_R.copy()
                
                try:
                    # 1. Dominant Eigenvector Extraction
                    # Since NavIC GEO/GSO satellites define the primary physical spatial signature 
                    # during clean ambient conditions, the dominant eigenvector perfectly models 
                    # the physical array phase drift.
                    
                    # We utilize the highly optimized Hermitian solver
                    eigvals, eigvecs = scipy.linalg.eigh(R_snapshot)
                    
                    # Sort indices, largest eigenvalue first
                    idx = np.argsort(eigvals)[::-1]
                    dominant_vec = eigvecs[:, idx[0]].reshape(self.num_channels, 1)
                    
                    # 2. Phase Normalization (Anchor to Channel 0)
                    # We rotate the entire vector so that Channel 0 sits at precisely 0-degrees phase.
                    phase_anchor = np.angle(dominant_vec[0, 0])
                    rotation_complex = np.exp(-1j * phase_anchor)
                    dominant_vec *= rotation_complex
                    
                    # 3. Vector Scaling
                    # Enforce unity magnitude per element
                    dominant_vec = dominant_vec / np.abs(dominant_vec)
                    
                    # 4. Atomic Swap for the main DSP thread
                    self._calibrated_steering_vector = dominant_vec.astype(np.complex64)
                    
                except Exception as e:
                    logger.error(f"[!] Background calibration extraction failed: {e}")
                    
            # Yield CPU back to the primary matrix execution workers
            elapsed = time.time() - t0
            sleep_time = max(0.001, self.tracking_interval - elapsed)
            time.sleep(sleep_time)

    def shutdown(self):
        self._running = False
        self._worker_thread.join()

# --- Rapid Verification Stub ---
if __name__ == "__main__":
    print("[*] Initializing Adaptive Blind Array Calibrator (Tracking at 10Hz)...")
    calibrator = BlindArrayCalibrator(num_channels=4, tracking_rate_hz=10.0)
    
    # 1. Baseline Genesis State
    baseline = calibrator.get_calibrated_steering_vector()
    print("\n--- BASELINE STEERING VECTOR ---")
    for i, w in enumerate(baseline):
        print(f" Ch[{i}]: {w[0].real:+.4f} {w[0].imag:+.4f}j  | Phase: {np.degrees(np.angle(w[0])):+6.2f}")
        
    # 2. Induce a severe synthetic thermal phase drift across the hardware array
    print("\n[*] Injecting 45-Degree Phase Drift (+45, +90, +135) to ambient covariance...")
    
    # Generate nominal signal vector with the exact phase drift
    # Ch 0: 0, Ch 1: 45, Ch 2: 90, Ch 3: 135
    true_phases = np.radians([0.0, 45.0, 90.0, 135.0])
    v_drift = np.exp(1j * true_phases).reshape(4, 1)
    
    # Construct drifted covariance matrix and feed it to the daemon
    R_drifted = 100.0 * (v_drift @ v_drift.conj().T) + 1.0 * np.eye(4)
    calibrator.ingest_tracking_matrix(R_drifted)
    
    # Wait strictly for the background thread to poll, execute extraction, and lock
    time.sleep(0.15)
    
    calibrated = calibrator.get_calibrated_steering_vector()
    print("\n--- DYNAMICALLY RE-CALIBRATED STEERING VECTOR ---")
    
    success = True
    for i, w in enumerate(calibrated):
        phase_deg = np.degrees(np.angle(w[0]))
        print(f" Ch[{i}]: {w[0].real:+.4f} {w[0].imag:+.4f}j  | Phase: {phase_deg:+6.2f}")
        
        # Verify alignment within 1 degree
        if abs(phase_deg - np.degrees(true_phases[i])) > 1.0:
            success = False
            
    if success:
        print(f"\n[PASSED] Blind Spatial Extraction successfully neutralized the thermal phase drift autonomously.")
    else:
        print(f"\n[FAILED] Steering Vector tracking lost coherence.")
        
    calibrator.shutdown()
