import sys
import os
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_SRC = os.path.join(BASE_DIR, 'backend', 'src')
sys.path.insert(0, BACKEND_SRC)

from cyclic_ls_equalizer import CyclicLSEqualizer

def test_equalizer_latency():
    equalizer = CyclicLSEqualizer(channels=4, taps=5, stride_len=4096)
    
    # Generate some dummy data
    np.random.seed(42)
    X_buffer = np.random.randn(4, 4096) + 1j * np.random.randn(4, 4096)
    X_buffer = X_buffer.astype(np.complex64)
    
    # Create reference as delayed version of channel 0 plus noise
    d_ref = np.zeros(4096, dtype=np.complex64)
    d_ref[2:] = X_buffer[0, :-2]
    d_ref += 0.1 * (np.random.randn(4096) + 1j * np.random.randn(4096))
    d_ref = d_ref.astype(np.complex64)
    
    # Warmup
    Y = equalizer.equalize_stride(X_buffer, d_ref)
    
    # Measure latency
    runs = 100
    t0 = time.perf_counter()
    for _ in range(runs):
        Y = equalizer.equalize_stride(X_buffer, d_ref)
    t1 = time.perf_counter()
    
    avg_us = ((t1 - t0) / runs) * 1e6
    print(f"Average Execution Latency: {avg_us:.2f} us")
    
    # Check if the output approximates the reference.
    # The MSE for channel 0 should be small.
    # MSE: 
    valid_slice = slice(5-1, 4096)
    mse0 = np.mean(np.abs(Y[0, valid_slice] - d_ref[valid_slice])**2)
    mse1 = np.mean(np.abs(Y[1, valid_slice] - d_ref[valid_slice])**2)
    
    print(f"MSE Channel 0 (where ref is embedded): {mse0:.4f}")
    print(f"MSE Channel 1 (random): {mse1:.4f}")

if __name__ == '__main__':
    test_equalizer_latency()
