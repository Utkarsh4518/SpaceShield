"""
Task 62.3: Telemetry Payload Compressor Module
SpaceShield High-Velocity Receiver DSP Subsystem

Provides JIT-compiled marker-based run-length encoding (RLE) for 109-byte binary telemetry.
Achieves high compression ratios on repeated zeros while avoiding heap allocations.
"""

import time
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _compress_rle_jit(
    src: np.ndarray,      # (109,) uint8
    dst: np.ndarray,      # (250,) uint8 (pre-allocated)
    marker: int = 0xFE
) -> int:
    """
    Zero-Heap Numba JIT RLE Compressor:
    Scans the source bytearray and collapses duplicate runs >= 4 bytes.
    Escapes instances of the marker byte.
    Returns the compressed byte length.
    """
    src_len = src.shape[0]
    write_idx = 0
    i = 0
    
    while i < src_len:
        val = src[i]
        
        # Count identical runs
        run_len = 1
        while (i + run_len) < src_len and src[i + run_len] == val and run_len < 255:
            run_len += 1
            
        if val == marker:
            # Escape marker
            dst[write_idx] = marker
            dst[write_idx + 1] = run_len
            dst[write_idx + 2] = marker
            write_idx += 3
            i += run_len
        elif run_len >= 4:
            # Pack run
            dst[write_idx] = marker
            dst[write_idx + 1] = run_len
            dst[write_idx + 2] = val
            write_idx += 3
            i += run_len
        else:
            # Copy literally
            for r in range(run_len):
                dst[write_idx] = val
                write_idx += 1
            i += run_len
            
    return write_idx


@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _decompress_rle_jit(
    src: np.ndarray,      # (comp_size,) uint8
    comp_size: int,
    dst: np.ndarray,      # (109,) uint8 (pre-allocated)
    marker: int = 0xFE
) -> int:
    """
    Zero-Heap Numba JIT RLE Decompressor:
    Unrolls marker sequences back to original layout.
    Returns decompressed byte size (must equal 109).
    """
    read_idx = 0
    write_idx = 0
    
    while read_idx < comp_size:
        val = src[read_idx]
        
        if val == marker:
            run_len = src[read_idx + 1]
            run_val = src[read_idx + 2]
            
            for r in range(run_len):
                dst[write_idx + r] = run_val
                
            write_idx += run_len
            read_idx += 3
        else:
            dst[write_idx] = val
            write_idx += 1
            read_idx += 1
            
    return write_idx


class TelemetryPayloadCompressor:
    """
    Handles lightweight telemetry packet compression.
    Ensures safe round-trips for the 109-byte SpaceShield frames.
    """
    def __init__(self, marker: int = 0xFE):
        self.marker = marker
        
        # Pre-allocated buffers to prevent GC thrashing in hot loops
        self._compress_buf = np.zeros(250, dtype=np.uint8)
        self._decompress_buf = np.zeros(109, dtype=np.uint8)
        
        # Warmup compiler
        self._warmup()

    def _warmup(self):
        dummy_src = np.zeros(109, dtype=np.uint8)
        _compress_rle_jit(dummy_src, self._compress_buf, self.marker)
        _decompress_rle_jit(self._compress_buf, 3, self._decompress_buf, self.marker)
        self._compress_buf.fill(0)
        self._decompress_buf.fill(0)

    def compress(self, raw_frame: bytes) -> bytes:
        """Compresses a 109-byte frame to a variable-length byte string."""
        src_arr = np.frombuffer(raw_frame, dtype=np.uint8)
        comp_len = _compress_rle_jit(src_arr, self._compress_buf, self.marker)
        return bytes(self._compress_buf[:comp_len])

    def decompress(self, compressed_frame: bytes) -> bytes:
        """Decompresses back to the original 109-byte structure."""
        src_arr = np.frombuffer(compressed_frame, dtype=np.uint8)
        dec_len = _decompress_rle_jit(src_arr, len(compressed_frame), self._decompress_buf, self.marker)
        return bytes(self._decompress_buf[:dec_len])


# =========================================================================
# DETERMINISTIC BENCHMARK HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Telemetry Payload Compressor Benchmark")
    print("==================================================================")
    
    compressor = TelemetryPayloadCompressor()
    
    import struct
    
    # 1. Nominal payload (lots of zeros in residuals and metrics)
    nominal_frame = struct.pack(
        "<B iddd dddd dddd bbbb d i",
        1, 0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,  # 32 bytes of zeros
        0.0, 0.0, 0.0, 0.0,  # 32 bytes of zeros
        0, 0, 0, 0,
        time.time(),
        0
    )
    
    # 2. Critical Alert payload (has non-zero values, fewer consecutive zeros)
    alert_frame = struct.pack(
        "<B iddd dddd dddd bbbb d i",
        1, 3, 0.95, 0.82, 64.5,
        0.1, 0.2, 0.3, 0.4,
        0.01, 0.02, 0.03, 0.04,
        0, 1, 1, 0,
        time.time(),
        5
    )
    
    # Functional verification
    print("[*] Performing compressor functional round-trip check...")
    comp_nominal = compressor.compress(nominal_frame)
    dec_nominal = compressor.decompress(comp_nominal)
    
    print(f"    -> Nominal Frame Size:    {len(nominal_frame)} bytes")
    print(f"    -> Compressed Nominal:    {len(comp_nominal)} bytes")
    assert dec_nominal == nominal_frame, "Nominal frame round-trip corrupted!"
    
    comp_alert = compressor.compress(alert_frame)
    dec_alert = compressor.decompress(comp_alert)
    print(f"    -> Compressed Alert:      {len(comp_alert)} bytes")
    assert dec_alert == alert_frame, "Alert frame round-trip corrupted!"
    print("    -> Functional verification: [PASSED]")

    # Run Benchmark Speed Run (50,000 operations)
    print("\n[*] Starting 50,000 loop speed run...")
    t0 = time.perf_counter()
    for _ in range(50000):
        c = compressor.compress(nominal_frame)
        _ = compressor.decompress(c)
    t1 = time.perf_counter()
    
    avg_us = (t1 - t0) * 1e6 / 50000
    comp_ratio = len(nominal_frame) / len(comp_nominal)
    
    print("\n--- COMPRESSION PERF HUD ---")
    print(f"  Avg Encode+Decode Latency:    {avg_us:.3f} µs per cycle")
    print(f"  Nominal Compression Ratio:    {comp_ratio:.2f}x")
    print(f"  Nominal Bandwidth Saving:     {(1.0 - 1.0/comp_ratio)*100:.1f}%")
    
    if avg_us < 6.0:
        print("\n[PASSED] Telemetry compression operates within low-latency bounds.")
    else:
        print("\n[FAILED] Compression latency exceeded performance limits.")
