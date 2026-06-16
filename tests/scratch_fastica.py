import sys
import os
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_SRC = os.path.join(BASE_DIR, 'backend', 'src')
sys.path.insert(0, BACKEND_SRC)

from fastica_separator import FastICASeparator

def test_fastica_latency():
    separator = FastICASeparator(stride_len=4096, num_iters=10)
    
    # Generate some dummy data
    np.random.seed(42)
    X_buffer = np.random.randn(4, 4096) + 1j * np.random.randn(4, 4096)
    X_buffer = X_buffer.astype(np.complex64)
    
    # Mix signals
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    A = A.astype(np.complex64)
    X_mixed = A @ X_buffer
    
    # Warmup
    X_copy = X_mixed.copy()
    Y = separator.separate_stride(X_copy)
    
    # Measure latency
    runs = 100
    t0 = time.perf_counter()
    for _ in range(runs):
        X_copy = X_mixed.copy()
        Y = separator.separate_stride(X_copy)
    t1 = time.perf_counter()
    
    avg_us = ((t1 - t0) / runs) * 1e6
    print(f"Average Execution Latency: {avg_us:.2f} us")

if __name__ == '__main__':
    test_fastica_latency()
