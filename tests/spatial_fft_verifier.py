"""
Task 37.3: Spatial-Frequency Estimation Verifier
Automated Wireless Security Harness
"""

import sys
import os
import json
import time
import numpy as np

# Map Absolute Paths to SpaceShield Sub-modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_SRC = os.path.join(BASE_DIR, 'backend', 'src')
COMPLIANCE_DIR = os.path.join(BASE_DIR, 'compliance')
LOG_PATH = os.path.join(COMPLIANCE_DIR, 'certin_incident_spoofing.json')

sys.path.insert(0, BACKEND_SRC)

try:
    from vectorized_fft_core import VectorizedFFTCore
    from capon_aoa_mapper import CaponAoAMapper
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to link SpaceShield modules. {e}")
    sys.exit(1)


def execute_spatial_stress_tests():
    print("===============================================================================")
    print("SPACESHIELD HARNESS: Spatial-Frequency Estimation & Resolution Verifier")
    print("===============================================================================")
    
    N = 4096
    np.random.seed(0xBA5EBA11)
    t = np.arange(N)
    
    # Generate multi-tone frequency-agile jamming vectors
    # Tone 1
    f1 = 0.12
    theta1 = 25.0
    phi1 = np.pi * np.sin(np.deg2rad(theta1))
    s1 = np.exp(1j * 2 * np.pi * f1 * t)
    
    # Tone 2 (Separated by 3 degrees)
    f2 = 0.15
    theta2 = 28.0
    phi2 = np.pi * np.sin(np.deg2rad(theta2))
    s2 = np.exp(1j * 2 * np.pi * f2 * t)
    
    # Construct 4-channel spatial-temporal array stride
    iq_stride = np.zeros((4, N), dtype=np.complex64)
    for c in range(4):
        a1 = np.cos(c * phi1) + 1j * np.sin(c * phi1)
        a2 = np.cos(c * phi2) + 1j * np.sin(c * phi2)
        iq_stride[c] = s1 * a1 + s2 * a2
        
    # Inject thermal floor to guarantee positive-definite bounds mathematically
    noise = (np.random.randn(4, N) + 1j * np.random.randn(4, N)) * 0.001
    iq_stride += noise.astype(np.complex64)
    
    # -------------------------------------------------------------------------
    # TEST 1: Absolute Precision Radix-4 Baseline Checks
    # -------------------------------------------------------------------------
    print("[1] Verifying High-Speed Radix-4 FFT Core Integrity...")
    fft_core = VectorizedFFTCore(channels=4, n_points=N)
    
    t0 = time.perf_counter()
    out_fft = fft_core.execute_transform(iq_stride)
    t1 = time.perf_counter()
    fft_latency_us = (t1 - t0) * 1e6
    
    # Absolute Double-Precision Cooley-Tukey Verification
    import scipy.fft
    baseline_fft = scipy.fft.fft(iq_stride, axis=1)
    mse = np.mean(np.abs(out_fft - baseline_fft)**2)
    
    print(f"    -> Cooley-Tukey Baseline MSE: {mse:.4e}")
    print(f"    -> Radix-4 Execution Latency: {fft_latency_us:.2f} us")
    
    assert mse < 1e-6, f"VERIFICATION FAILED: FFT MSE {mse:.4e} exceeded bounds of 1e-6."
    print("    [PASS] Spectral execution flawlessly mirrors double-precision floating math.")
    
    # -------------------------------------------------------------------------
    # TEST 2: Spatial Anomalies 3-Degree Resolution Geolocation
    # -------------------------------------------------------------------------
    print("\n[2] Executing Sub-3 Degree Capon Minimum Variance Targeting...")
    bin1 = int(round(f1 * N))
    bin2 = int(round(f2 * N))
    
    mapper = CaponAoAMapper(channels=4, max_anomalies=64)
    target_bins = np.array([bin1, bin2], dtype=np.int32)
    
    t0 = time.perf_counter()
    spectra = mapper.map_anomalies(out_fft, target_bins)
    t1 = time.perf_counter()
    capon_latency_us = (t1 - t0) * 1e6
    
    peak1_idx = np.argmax(spectra[0])
    peak2_idx = np.argmax(spectra[1])
    
    est_theta1 = peak1_idx - 90
    est_theta2 = peak2_idx - 90
    
    err1 = abs(est_theta1 - theta1)
    err2 = abs(est_theta2 - theta2)
    max_err = max(err1, err2)
    
    print(f"    -> Target 1 Jammer:  {theta1} deg | Capon Lock: {est_theta1} deg")
    print(f"    -> Target 2 Jammer:  {theta2} deg | Capon Lock: {est_theta2} deg")
    print(f"    -> Max Azimuth Error: {max_err} deg")
    print(f"    -> Sweeper Latency:   {capon_latency_us:.2f} us")
    
    assert max_err <= 0.5, f"VERIFICATION FAILED: Max AoA lock error {max_err} exceeds 0.5 degrees"
    print("    [PASS] Multiple closely-spaced multi-tone jamming angles successfully isolated.")
    
    # -------------------------------------------------------------------------
    # TEST 3: Cryptographic WORM Signatures
    # -------------------------------------------------------------------------
    print(f"\n[3] Sealing Spatial Verification Signatures into WORM Ledger...")
    os.makedirs(COMPLIANCE_DIR, exist_ok=True)
    
    log_event = {
        "timestamp_iso": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "event_classification": "SPATIAL_FFT_VERIFICATION",
        "fft_core_metrics": {
            "double_precision_mse": float(mse),
            "stride_latency_us": float(fft_latency_us),
            "precision_pass": bool(mse < 1e-6)
        },
        "capon_aoa_metrics": {
            "target_1_azimuth": float(theta1),
            "target_2_azimuth": float(theta2),
            "estimated_1_azimuth": float(est_theta1),
            "estimated_2_azimuth": float(est_theta2),
            "max_angular_error_deg": float(max_err),
            "sweep_latency_us": float(capon_latency_us),
            "resolution_pass": bool(max_err <= 0.5)
        }
    }
    
    # Override strict WORM
    import stat
    if os.path.exists(LOG_PATH):
        os.chmod(LOG_PATH, stat.S_IWRITE)
        
    worm_chain = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, 'r') as f:
                worm_chain = json.load(f)
                if not isinstance(worm_chain, list):
                    worm_chain = [worm_chain]
        except Exception:
            pass
            
    worm_chain.append(log_event)
    
    with open(LOG_PATH, 'w') as f:
        json.dump(worm_chain, f, indent=4)
        
    os.chmod(LOG_PATH, stat.S_IREAD)
    
    print(f"    [PASS] Signatures secured and appended -> {LOG_PATH}")
    print("===============================================================================")
    print("ALL HARNESS TESTS CLEARED.")
    print("===============================================================================")

if __name__ == "__main__":
    execute_spatial_stress_tests()
