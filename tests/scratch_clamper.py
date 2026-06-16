import sys
import os
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_SRC = os.path.join(BASE_DIR, 'backend', 'src')
sys.path.insert(0, BACKEND_SRC)

from multipath_tap_clamper import MultipathTapClamper

def test_clamper_latency():
    clamper = MultipathTapClamper(channels=4, taps=5, isolation_db=15.0)
    
    # Generate some dummy weights
    weights = np.zeros((4, 5), dtype=np.complex64)
    # Primary LoS peak at tap 0
    weights[:, 0] = 1.0 + 1j
    # Noise tap below 15dB (-15dB = 0.0316 power ratio)
    weights[:, 1] = 0.1 + 0.1j # Power = 0.02 (Peak power = 2.0). Ratio = 0.01 < 0.0316
    # Echo tap above 15dB
    weights[:, 2] = 0.5 + 0.5j # Power = 0.5. Ratio = 0.25 > 0.0316
    
    # Warmup
    clamper.enforce_sparsity(weights)
    
    # Assert correctness
    assert weights[0, 1] == 0.0 + 0j
    assert weights[0, 2] != 0.0 + 0j
    
    # Measure latency
    runs = 1000
    t0 = time.perf_counter()
    for _ in range(runs):
        clamper.enforce_sparsity(weights)
    t1 = time.perf_counter()
    
    avg_us = ((t1 - t0) / runs) * 1e6
    print(f"Average Execution Latency: {avg_us:.2f} us")

if __name__ == '__main__':
    test_clamper_latency()
