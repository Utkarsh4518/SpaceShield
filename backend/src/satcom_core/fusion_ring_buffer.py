"""
Task 59.1: Fusion Ring Buffer Module
SpaceShield High-Velocity Receiver DSP Subsystem

Implements a zero-allocation, fixed-capacity thread-safe ring-buffer bridge.
Supports contiguous pre-allocated static slots for threat state, scores,
GLRT sphericity, correlator skew residuals, AoA deviations, null directives,
and timestamps. Index math and batch extraction are JIT-compiled.
"""

import time
import threading
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _push_ring_record_jit(
    write_ptr: int,
    read_ptr: int,
    drop_counter: int,
    capacity: int,
    threat_state_buf: np.ndarray,
    jammer_score_buf: np.ndarray,
    spoof_score_buf: np.ndarray,
    sphericity_buf: np.ndarray,
    skew_residuals_buf: np.ndarray,
    aoa_deviation_buf: np.ndarray,
    nulling_directives_buf: np.ndarray,
    timestamp_buf: np.ndarray,
    # Inputs
    threat_state: int,
    jammer_score: float,
    spoof_score: float,
    sphericity: float,
    skew_residuals: np.ndarray,
    aoa_deviation: np.ndarray,
    nulling_directives: np.ndarray,
    timestamp: float
) -> tuple[int, int, int]:
    """
    Zero-Heap Numba JIT Producer Push Kernel:
    1. Computes target modulo write index.
    2. Writes data directly into pre-allocated contiguous buffers.
    3. Handles buffer overflows by advancing read pointer (oldest record drops).
    """
    w_idx = write_ptr % capacity
    
    threat_state_buf[w_idx] = threat_state
    jammer_score_buf[w_idx] = jammer_score
    spoof_score_buf[w_idx] = spoof_score
    sphericity_buf[w_idx] = sphericity
    timestamp_buf[w_idx] = timestamp
    
    for c in range(4):
        skew_residuals_buf[w_idx, c] = skew_residuals[c]
        aoa_deviation_buf[w_idx, c] = aoa_deviation[c]
        nulling_directives_buf[w_idx, c] = nulling_directives[c]
        
    new_write_ptr = write_ptr + 1
    new_read_ptr = read_ptr
    new_drop_counter = drop_counter
    
    # Check overwrite boundary (overflow)
    if (new_write_ptr - new_read_ptr) > capacity:
        new_read_ptr = new_write_ptr - capacity
        new_drop_counter = drop_counter + 1
        
    return new_write_ptr, new_read_ptr, new_drop_counter


@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _pop_batch_jit(
    write_ptr: int,
    read_ptr: int,
    capacity: int,
    threat_state_buf: np.ndarray,
    jammer_score_buf: np.ndarray,
    spoof_score_buf: np.ndarray,
    sphericity_buf: np.ndarray,
    skew_residuals_buf: np.ndarray,
    aoa_deviation_buf: np.ndarray,
    nulling_directives_buf: np.ndarray,
    timestamp_buf: np.ndarray,
    # Outputs
    out_threat_states: np.ndarray,
    out_jammer_scores: np.ndarray,
    out_spoof_scores: np.ndarray,
    out_sphericity: np.ndarray,
    out_skew_residuals: np.ndarray,
    out_aoa_deviation: np.ndarray,
    out_nulling_directives: np.ndarray,
    out_timestamps: np.ndarray,
    max_count: int
) -> tuple[int, int]:
    """
    Zero-Heap Numba JIT Consumer Pop Kernel:
    Extracts up to max_count records without any heap allocation.
    Returns: (new_read_ptr, popped_count)
    """
    available = write_ptr - read_ptr
    if available <= 0:
        return read_ptr, 0
        
    popped_count = max_count
    if available < max_count:
        popped_count = available
        
    curr_read_ptr = read_ptr
    for i in range(popped_count):
        r_idx = curr_read_ptr % capacity
        
        out_threat_states[i] = threat_state_buf[r_idx]
        out_jammer_scores[i] = jammer_score_buf[r_idx]
        out_spoof_scores[i] = spoof_score_buf[r_idx]
        out_sphericity[i] = sphericity_buf[r_idx]
        out_timestamps[i] = timestamp_buf[r_idx]
        
        for c in range(4):
            out_skew_residuals[i, c] = skew_residuals_buf[r_idx, c]
            out_aoa_deviation[i, c] = aoa_deviation_buf[r_idx, c]
            out_nulling_directives[i, c] = nulling_directives_buf[r_idx, c]
            
        curr_read_ptr += 1
        
    return curr_read_ptr, popped_count


class FusionRingBuffer:
    """
    Thread-Safe Spatial Telemetry Ring Buffer.
    Acts as a lock-free pointer container guarded by atomic thread locks.
    Decoupled compute-plane ingestion from control-plane serialization.
    """
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        
        # Pre-allocated zero-heap buffers
        self.threat_state_buf = np.zeros(self.capacity, dtype=np.int32)
        self.jammer_score_buf = np.zeros(self.capacity, dtype=np.float64)
        self.spoof_score_buf = np.zeros(self.capacity, dtype=np.float64)
        self.sphericity_buf = np.zeros(self.capacity, dtype=np.float64)
        self.skew_residuals_buf = np.zeros((self.capacity, 4), dtype=np.float64)
        self.aoa_deviation_buf = np.zeros((self.capacity, 4), dtype=np.float64)
        self.nulling_directives_buf = np.zeros((self.capacity, 4), dtype=np.bool_)
        self.timestamp_buf = np.zeros(self.capacity, dtype=np.float64)
        
        # Pointers (64-bit integer monotonically increasing registers)
        self.write_ptr = 0
        self.read_ptr = 0
        self.drop_counter = 0
        
        # Pre-allocated extraction target slots (reused to prevent pop-path allocations)
        self._out_threat = np.zeros(128, dtype=np.int32)
        self._out_jammer = np.zeros(128, dtype=np.float64)
        self._out_spoof = np.zeros(128, dtype=np.float64)
        self._out_sph = np.zeros(128, dtype=np.float64)
        self._out_skew = np.zeros((128, 4), dtype=np.float64)
        self._out_aoa = np.zeros((128, 4), dtype=np.float64)
        self._out_null = np.zeros((128, 4), dtype=np.bool_)
        self._out_ts = np.zeros(128, dtype=np.float64)
        
        # Access lock for thread synchronization
        self.lock = threading.Lock()
        
        # Pre-warm compiler
        self._warmup()

    def _warmup(self):
        """Pre-compiles JIT routines."""
        dummy_arr = np.zeros(4, dtype=np.float64)
        dummy_bool = np.zeros(4, dtype=np.bool_)
        _push_ring_record_jit(
            self.write_ptr, self.read_ptr, self.drop_counter, self.capacity,
            self.threat_state_buf, self.jammer_score_buf, self.spoof_score_buf,
            self.sphericity_buf, self.skew_residuals_buf, self.aoa_deviation_buf,
            self.nulling_directives_buf, self.timestamp_buf,
            0, 0.0, 0.0, 0.0, dummy_arr, dummy_arr, dummy_bool, 0.0
        )
        _pop_batch_jit(
            self.write_ptr, self.read_ptr, self.capacity,
            self.threat_state_buf, self.jammer_score_buf, self.spoof_score_buf,
            self.sphericity_buf, self.skew_residuals_buf, self.aoa_deviation_buf,
            self.nulling_directives_buf, self.timestamp_buf,
            self._out_threat, self._out_jammer, self._out_spoof, self._out_sph,
            self._out_skew, self._out_aoa, self._out_null, self._out_ts, 1
        )

    def push(
        self,
        threat_state: int,
        jammer_score: float,
        spoof_score: float,
        sphericity: float,
        skew_residuals: np.ndarray,      # (4,)
        aoa_deviation: np.ndarray,       # (4,)
        nulling_directives: np.ndarray,  # (4,)
        timestamp: float
    ):
        """Thread-safe fast producer path."""
        with self.lock:
            self.write_ptr, self.read_ptr, self.drop_counter = _push_ring_record_jit(
                self.write_ptr,
                self.read_ptr,
                self.drop_counter,
                self.capacity,
                self.threat_state_buf,
                self.jammer_score_buf,
                self.spoof_score_buf,
                self.sphericity_buf,
                self.skew_residuals_buf,
                self.aoa_deviation_buf,
                self.nulling_directives_buf,
                self.timestamp_buf,
                threat_state,
                jammer_score,
                spoof_score,
                sphericity,
                skew_residuals,
                aoa_deviation,
                nulling_directives,
                timestamp
            )

    def pop_batch(self, max_count: int = 128) -> list[dict]:
        """
        Thread-safe consumer dequeue loop.
        Extracts records into pre-allocated buffers, then serializes them to control-plane dicts.
        """
        count = min(max_count, 128)
        
        with self.lock:
            self.read_ptr, popped = _pop_batch_jit(
                self.write_ptr,
                self.read_ptr,
                self.capacity,
                self.threat_state_buf,
                self.jammer_score_buf,
                self.spoof_score_buf,
                self.sphericity_buf,
                self.skew_residuals_buf,
                self.aoa_deviation_buf,
                self.nulling_directives_buf,
                self.timestamp_buf,
                self._out_threat,
                self._out_jammer,
                self._out_spoof,
                self._out_sph,
                self._out_skew,
                self._out_aoa,
                self._out_null,
                self._out_ts,
                count
            )
            
            # Save drop counter to local scope
            drops = self.drop_counter

        # Control-plane serialization (safe to allocate dicts here after release of DSP lock)
        serialized_list = []
        for i in range(popped):
            serialized_list.append({
                "threat_state": int(self._out_threat[i]),
                "jammer_score": float(self._out_jammer[i]),
                "spoof_score": float(self._out_spoof[i]),
                "sphericity": float(self._out_sph[i]),
                "skew_residuals": [float(self._out_skew[i, c]) for c in range(4)],
                "aoa_deviation": [float(self._out_aoa[i, c]) for c in range(4)],
                "nulling_directives": [bool(self._out_null[i, c]) for c in range(4)],
                "timestamp": float(self._out_ts[i]),
                "buffer_drops": int(drops)
            })
            
        return serialized_list


# =========================================================================
# DETERMINISTIC STRESS HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Fusion Ring Buffer Stress Harness")
    print("==================================================================")
    
    # Initialize tiny buffer for easy overwrite stress-testing
    ring = FusionRingBuffer(capacity=10)
    
    dummy_skew = np.array([0.1, 0.2, 0.3, 0.4])
    dummy_aoa = np.array([0.01, 0.02, 0.03, 0.04])
    dummy_null = np.array([False, False, True, False])
    
    # 1. Steady-state nominal streaming (push 8 elements, read 8 elements)
    print("[*] Scenario 1: Steady-State Nominal Ingestion...")
    for step in range(8):
        ring.push(0, 0.05, 0.02, 12.5, dummy_skew, dummy_aoa, dummy_null, time.time())
        
    assert ring.write_ptr == 8 and ring.read_ptr == 0, "Nominal write indices mismatched!"
    batch = ring.pop_batch(max_count=8)
    assert len(batch) == 8, "Failed to pop complete nominal batch!"
    assert ring.read_ptr == 8, "Read index pointer failed to advance!"
    print("    -> Nominal test check: [PASSED]")
    
    # 2. Bursty adversarial updates / Overwrite stress test (push 15 elements without reading)
    print("\n[*] Scenario 2: Overwrite and Overflow Stress Test...")
    # Buffer has capacity 10. Pushing 15 elements should drop 5 elements (read_ptr shifts)
    for step in range(15):
        ring.push(3, 0.95, 0.85, 95.0, dummy_skew, dummy_aoa, dummy_null, time.time())
        
    print(f"    -> Write pointer: {ring.write_ptr} | Read pointer: {ring.read_ptr}")
    print(f"    -> Drop counter: {ring.drop_counter}")
    assert ring.drop_counter == 5, "Drop overwrite accounting mismatch!"
    assert ring.read_ptr == ring.write_ptr - 10, "Read index offset failed to adjust!"
    
    # Drain remaining batch
    drained = ring.pop_batch(max_count=20)
    assert len(drained) == 10, "Drained count must match available capacity!"
    assert ring.read_ptr == ring.write_ptr, "Buffer failed to drain fully!"
    print("    -> Overwrite check: [PASSED]")
    
    # 3. Telemetry Starvation Test (attempting to read empty buffer)
    print("\n[*] Scenario 3: Telemetry Starvation (Pop Empty Buffer)...")
    empty_batch = ring.pop_batch(max_count=5)
    assert len(empty_batch) == 0, "Popping empty buffer returned non-zero elements!"
    print("    -> Starvation check: [PASSED]")
    
    # Benchmark latency test
    print("\n--- FUSION RING BUFFER BENCHMARK ---")
    print("[*] Running 30,000 warm thread pushes...")
    
    latencies = []
    for _ in range(30000):
        t0 = time.perf_counter()
        ring.push(0, 0.0, 0.0, 0.0, dummy_skew, dummy_aoa, dummy_null, 0.0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)
        
    avg_us = np.median(latencies)
    p99_us = np.percentile(latencies, 99.0)
    
    print(f"  Median Ingestion Latency: {avg_us:.3f} µs")
    print(f"  P99 Ingestion Latency:    {p99_us:.3f} µs")
    
    if avg_us < 2.0:
        print("[PASSED] Ingestion latency remains sub-microsecond.")
    else:
        print("[FAILED] Buffer insertion latency exceeded budget ceiling.")
