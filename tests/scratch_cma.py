import sys
import os
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_SRC = os.path.join(BASE_DIR, 'backend', 'src')
sys.path.insert(0, BACKEND_SRC)

from cma_blind_equalizer import CMABlindEqualizer

def test_cma_latency():
    equalizer = CMABlindEqualizer(channels=4, taps=5, stride_len=4096, mu=1e-4)
    
    # Generate BPSK dummy data (modulus ~ 1.0) with some noise
    np.random.seed(42)
    bpsk = np.sign(np.random.randn(4, 4096)) + 1j * np.sign(np.random.randn(4, 4096))
    bpsk /= np.sqrt(2)
    noise = (np.random.randn(4, 4096) + 1j * np.random.randn(4, 4096)) * 0.1
    X_buffer = (bpsk + noise).astype(np.complex64)
    
    # Warmup
    X_copy = X_buffer.copy()
    Y = equalizer.equalize_stride(X_copy)
    
    # Measure latency
    runs = 1000
    t0 = time.perf_counter()
    for _ in range(runs):
        Y = equalizer.equalize_stride(X_copy)
    t1 = time.perf_counter()
    
    avg_us = ((t1 - t0) / runs) * 1e6
    print(f"Average Execution Latency: {avg_us:.2f} us")

if __name__ == '__main__':
    test_cma_latency()
