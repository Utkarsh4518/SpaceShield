import time
import numpy as np
import numba
import logging

logger = logging.getLogger("SICCorrelationCanceller")
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------------
# Zero-Heap Numba JIT Mathematics
# --------------------------------------------------------------------------------
@numba.njit(fastmath=True, cache=True)
def _fast_sic_cancellation(x_real, x_imag, replica_real, replica_imag):
    """
    Executes an instantaneous cross-correlation to extract the true amplitude and 
    carrier phase of the hostile signal component, constructs the localized replica, 
    and applies negative superposition (subtraction) in-place.
    
    Args:
        x_real: (4, 4096) Float32 array representing the real component of the Baseband I/Q.
        x_imag: (4, 4096) Float32 array representing the imag component of the Baseband I/Q.
        replica_real: (4096,) Float32 array representing the normalized interferer code signature.
        replica_imag: (4096,) Float32 array representing the normalized interferer code signature.
    """
    num_channels = x_real.shape[0]
    stride_len = x_real.shape[1]
    
    # Process independently per channel
    for ch in range(num_channels):
        # 1. Project the signal onto the interferer replica to extract Amplitude and Phase
        dot_r = 0.0
        dot_i = 0.0
        
        for i in range(stride_len):
            v_r = x_real[ch, i]
            v_i = x_imag[ch, i]
            rep_r = replica_real[i]
            rep_i = -replica_imag[i] # Conjugate for cross-correlation
            
            # Complex dot product: V * conj(Replica)
            dot_r += (v_r * rep_r - v_i * rep_i)
            dot_i += (v_r * rep_i + v_i * rep_r)
            
        # Normalize the extracted correlation coefficient
        C_real = dot_r / stride_len
        C_imag = dot_i / stride_len
        
        # 2. Reconstruct the localized replica and Execute synchronized Vector Subtraction
        for i in range(stride_len):
            rep_r = replica_real[i]
            rep_i = replica_imag[i]
            
            # Complex multiply: C * Replica
            recon_r = C_real * rep_r - C_imag * rep_i
            recon_i = C_real * rep_i + C_imag * rep_r
            
            # 3. Successive Interference Cancellation (In-Place Subtraction)
            x_real[ch, i] -= recon_r
            x_imag[ch, i] -= recon_i

# --------------------------------------------------------------------------------
# Modular Python Class Envelope
# --------------------------------------------------------------------------------
class SICCorrelationCanceller:
    """
    Successive Interference Cancellation (SIC) Engine.
    Operates immediately before the Spatial GLRT matrix logic to physically strip 
    dominant cross-correlation interference (like a high-power meaconing burst or 
    continuous-wave spoofer) directly out of the raw IQ samples. 
    It prevents spatial matrices from becoming structurally biased by false PRN codes.
    """
    def __init__(self, num_channels: int = 4, stride_length: int = 4096):
        self.num_channels = num_channels
        self.stride_length = stride_length
        
        # Simulated Hostile Tracking Parameter Cache
        self._target_replica = np.zeros(self.stride_length, dtype=np.complex64)

    def load_interferer_parameters(self, hostile_replica: np.ndarray):
        """
        Dynamically flagged by edge_inference_engine.py. 
        Loads the regenerated localized code replica of the dominant interferer.
        """
        np.copyto(self._target_replica, hostile_replica)

    def execute_sic_stride(self, X: np.ndarray) -> tuple:
        """
        Hot-path execution function. Calculates instantaneous channel-wise cross-correlation 
        and extracts the hostile payload natively in C memory space.
        
        Args:
            X: (4, 4096) Complex64 baseband I/Q array. Modifies IN PLACE.
            
        Returns:
            (X_clean, execution_time_us)
        """
        t0 = time.perf_counter()
        
        # Execute C-compiled SIC subtraction block using pure float32 scalar math
        _fast_sic_cancellation(
            X.real,
            X.imag,
            self._target_replica.real,
            self._target_replica.imag
        )
        
        execution_us = (time.perf_counter() - t0) * 1e6
        
        return X, execution_us

# --- Rapid Verification Stub ---
if __name__ == "__main__":
    print("[*] Initializing SIC Correlation Canceller...")
    canceller = SICCorrelationCanceller(num_channels=4, stride_length=4096)
    
    # 1. Synthesize the clean ambient background (Thermal Noise + Weak NavIC Target)
    ambient_X = (np.random.randn(4, 4096) + 1j * np.random.randn(4, 4096)).astype(np.complex64)
    
    # 2. Synthesize a Dominant Code-Division Spoofer (Hostile Replica)
    # The hostile signal has an arbitrary PRN sequence and massive spatial phase/amplitude differences
    hostile_prn = np.sign(np.random.randn(4096)).astype(np.float32)
    hostile_replica = hostile_prn + 1j * 0.0 # BPSK signal
    
    canceller.load_interferer_parameters(hostile_replica)
    
    # Inject massive +30dB hostile power into the ambient signal (Simulating near-field spoofing)
    # Channel 0: Amplitude 50, Phase 0
    # Channel 1: Amplitude 50, Phase 90 deg
    # Channel 2: Amplitude 50, Phase 180 deg
    # Channel 3: Amplitude 50, Phase 270 deg
    hostile_amplitudes = np.array([50.0, 50.0j, -50.0, -50.0j], dtype=np.complex64).reshape(4, 1)
    attack_X = ambient_X + (hostile_amplitudes * hostile_replica)
    
    # Pre-calculate the cross-correlation sidelobe leakage of the RAW infected signal
    # Cross-Correlation = dot(X, conj(Replica))
    infected_dot = np.sum(attack_X * np.conj(hostile_replica), axis=1) / 4096.0
    infected_power_db = 10 * np.log10(np.abs(infected_dot)**2 + 1e-12)
    
    # Burn-in LLVM caches
    canceller.execute_sic_stride(attack_X.copy())
    
    # 3. Benchmarking
    latencies = []
    clean_X = None
    for _ in range(5000):
        # We pass a copy to isolate memory mutation for benchmarking iterations
        mock_X = attack_X.copy()
        clean_X, us = canceller.execute_sic_stride(mock_X)
        latencies.append(us)
        
    avg_us = sum(latencies) / len(latencies)
    
    print("\n--- SIC EXECUTION PROFILES ---")
    print(f" [>] Background Monitoring Latency: {avg_us:.2f} µs")
    
    # Calculate Residual Sidelobe Leakage after SIC Vector Subtraction
    cleaned_dot = np.sum(clean_X * np.conj(hostile_replica), axis=1) / 4096.0
    cleaned_power_db = 10 * np.log10(np.abs(cleaned_dot)**2 + 1e-12)
    
    suppression_db = infected_power_db - cleaned_power_db
    mean_suppression = np.mean(suppression_db)
    
    print(f"\n--- SIDELOBE CANCELLATION METRICS ---")
    for ch in range(4):
        print(f" [Ch {ch}] Hostile Infection Vector: {infected_dot[ch].real:+.2f} {infected_dot[ch].imag:+.2f}j  ({infected_power_db[ch]:+.2f} dB)")
        print(f" [Ch {ch}] Purified Signal Residual: {cleaned_dot[ch].real:+.2f} {cleaned_dot[ch].imag:+.2f}j  ({cleaned_power_db[ch]:+.2f} dB)")
        print(f" [Ch {ch}] Subtraction Suppression:  {suppression_db[ch]:+.2f} dB")
        
    if avg_us < 45.0 and mean_suppression > 26.0:
        print(f"\n[PASSED] Sub-45µs SIC Subtraction successfully annihilated the hostile PRN correlation!")
    else:
        print(f"\n[FAILED] Constraints breached.")
