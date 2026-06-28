"""
Task 66.1: Live SDR Timestamp Bridge Module
SpaceShield High-Velocity Receiver DSP Subsystem

Binds SDR hardware timestamp ticks and DMA buffer telemetry to firmware_loopback_sync.
Evaluates delay, jitter, and slips in a zero-allocation JIT hot path.
"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _evaluate_live_sdr_bridge_jit(
    tx_hw_ts: int,
    rx_hw_ts: int,
    buf_occupancy: float,
    seq_idx: int,
    dma_stall_flag: int,
    delay_history: np.ndarray,
    history_idx: int,
    history_len: int,
    clock_freq: float,
    prev_seq: int,
    slip_threshold: int
) -> tuple[int, float, float, int]:
    """
    JIT-compiled SDR hardware telemetry evaluator:
    Returns (sync_state, delay_seconds, jitter_seconds, slips).
    """
    # 1. Convert ticks difference to seconds
    ticks_diff = rx_hw_ts - tx_hw_ts
    delay_s = ticks_diff / clock_freq
    
    # 2. Track sample slips
    slips = 0
    if prev_seq != -1:
        if seq_idx > prev_seq:
            slips = seq_idx - prev_seq - 1
            
    # 3. Store delay in ring buffer
    delay_history[history_idx] = delay_s
    
    # 4. Compute mean and jitter (standard deviation)
    d_sum = 0.0
    for i in range(history_len):
        d_sum += delay_history[i]
    mean_delay = d_sum / history_len
    
    var_sum = 0.0
    for i in range(history_len):
        diff = delay_history[i] - mean_delay
        var_sum += diff * diff
    jitter_s = math.sqrt(var_sum / history_len)
    
    # 5. Determine state transitions:
    # 0: NOMINAL, 1: WARNING, 2: SLIP, 3: DESYNC
    if dma_stall_flag == 1 or slips > slip_threshold or delay_s > 0.015:
        sync_state = 3  # DESYNC
    elif slips > 0:
        sync_state = 2  # SLIP
    elif jitter_s > 0.0025 or buf_occupancy > 80.0 or delay_s > 0.008:
        sync_state = 1  # WARNING
    else:
        sync_state = 0  # NOMINAL
        
    return sync_state, delay_s, jitter_s, slips


class LiveSDRTimestampBridge:
    """
    Binds live SDR hardware timestamp counters and DMA telemetry to loopback synchronization.
    Runs on a zero-allocation JIT update path.
    """
    def __init__(self, clock_freq: float = 125_000_000.0, history_len: int = 100, slip_threshold: int = 2):
        self.clock_freq = clock_freq
        self.history_len = history_len
        self.slip_threshold = slip_threshold
        
        # Pre-allocated arrays and counters
        self.delay_history = np.zeros(history_len, dtype=np.float64)
        self.history_idx = 0
        self.prev_seq = -1

    def process_hw_stride(
        self,
        tx_hw_ts: int,
        rx_hw_ts: int,
        buf_occupancy: float,
        seq_idx: int,
        dma_stall_flag: int
    ) -> dict:
        """
        Ingests a single hardware telemetry stride.
        """
        sync_state, delay_s, jitter_s, slips = _evaluate_live_sdr_bridge_jit(
            tx_hw_ts,
            rx_hw_ts,
            buf_occupancy,
            seq_idx,
            dma_stall_flag,
            self.delay_history,
            self.history_idx,
            self.history_len,
            self.clock_freq,
            self.prev_seq,
            self.slip_threshold
        )
        
        # Shift indexes
        self.history_idx = (self.history_idx + 1) % self.history_len
        self.prev_seq = seq_idx
        
        return {
            "sync_state": sync_state,
            "delay_seconds": delay_s,
            "jitter_seconds": jitter_s,
            "sample_slips": slips
        }


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Live SDR Timestamp Bridge Harness")
    print("==================================================================")
    
    bridge = LiveSDRTimestampBridge(clock_freq=125_000_000.0, history_len=10)
    
    # 1. Stable Timing
    print("[*] Scenario 1: Stable Hardware Clock Ingestion...")
    # 2ms delay = 250,000 ticks at 125 MHz
    base_tx = 10_000_000
    base_rx = 10_250_000
    
    for step in range(10):
        res = bridge.process_hw_stride(
            base_tx + step * 125_000,  # 1ms step increment
            base_rx + step * 125_000,
            20.0,  # Low buffer occupancy (20%)
            step + 1,
            0      # No stalls
        )
        
    print(f"    -> Sync State: {res['sync_state']} | Delay: {res['delay_seconds']*1000:.2f} ms | Jitter: {res['jitter_seconds']*1000:.4f} ms")
    assert res['sync_state'] == 0, "Stable hardware clock check failed!"
    print("    -> Stable operation: [PASSED]")
    
    # 2. Slow Drift (Jitter increase)
    print("\n[*] Scenario 2: Slow Timing Drift & Jitter Accumulation...")
    # Gradually add delay noise (3ms = 375,000 ticks)
    for step in range(10, 15):
        noise = int(800_000 * math.sin(step))
        res = bridge.process_hw_stride(
            base_tx + step * 125_000,
            base_rx + step * 125_000 + noise,
            25.0,
            step + 1,
            0
        )
        
    print(f"    -> Sync State: {res['sync_state']} | Jitter: {res['jitter_seconds']*1000:.2f} ms")
    assert res['sync_state'] == 1, "Failed to classify Drift Warning state!"
    print("    -> Drift detection: [PASSED]")
    
    # 3. Sudden Clock Slip (Sequence jump)
    print("\n[*] Scenario 3: Abrupt Clock Slip...")
    res = bridge.process_hw_stride(
        base_tx + 15 * 125_000,
        base_rx + 15 * 125_000,
        30.0,
        18,  # skipped sequence index (expects 16, gets 18)
        0
    )
    print(f"    -> Sync State: {res['sync_state']} | Slips: {res['sample_slips']}")
    assert res['sync_state'] == 2 and res['sample_slips'] == 2, "Failed to detect clock slip event!"
    print("    -> Clock slip detection: [PASSED]")

    # 4. Burst Stall (DMA Stall flag active)
    print("\n[*] Scenario 4: DMA Transport Stall...")
    res = bridge.process_hw_stride(
        base_tx + 16 * 125_000,
        base_rx + 16 * 125_000,
        95.0,  # High buffer occupancy
        19,
        1      # DMA Stall active!
    )
    print(f"    -> Sync State: {res['sync_state']}")
    assert res['sync_state'] == 3, "Failed to detect DMA desync stall state!"
    print("    -> DMA stall detection: [PASSED]")

    print("\n[+] Live SDR timestamp bridge validation complete.")
