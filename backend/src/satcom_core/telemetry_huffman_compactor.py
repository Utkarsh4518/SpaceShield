"""
Task 63.2: Telemetry Huffman Compactor Module
SpaceShield High-Velocity Receiver DSP Subsystem

Applies a static prefix-free Huffman coding pass on top of RLE compressed
binary telemetry payloads. Achieves high throughput using JIT-compiled bit-shifting.
"""

import time
import numpy as np
from numba import njit

from telemetry_payload_compressor import TelemetryPayloadCompressor

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _huffman_encode_jit(
    src: np.ndarray,      # (src_len,) uint8
    src_len: int,
    dst: np.ndarray       # (max_dst_len,) uint8 (pre-allocated)
) -> int:
    """
    Zero-Heap JIT Huffman Encoder:
    Converts src byte symbols to variable-length prefix bits.
    Returns the size of the packed bytes.
    """
    bit_buf = 0
    bit_count = 0
    write_idx = 0
    
    for i in range(src_len):
        val = src[i]
        
        # Static prefix-code assignments:
        # val == 0   -> 0b0 (len 1)
        # val == 254 -> 0b10 (len 2) (RLE marker)
        # other      -> 0b11 + 8-bit val (len 10)
        if val == 0:
            code = 0
            code_len = 1
        elif val == 254:
            code = 2
            code_len = 2
        else:
            code = (3 << 8) | int(val)
            code_len = 10
            
        bit_buf = (bit_buf << code_len) | code
        bit_count += code_len
        
        while bit_count >= 8:
            bit_count -= 8
            byte_val = (bit_buf >> bit_count) & 0xFF
            dst[write_idx] = byte_val
            write_idx += 1
            bit_buf = bit_buf & ((1 << bit_count) - 1)
            
    if bit_count > 0:
        byte_val = (bit_buf << (8 - bit_count)) & 0xFF
        dst[write_idx] = byte_val
        write_idx += 1
        
    return write_idx


@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _huffman_decode_jit(
    src: np.ndarray,      # (src_len,) uint8
    src_len: int,
    expected_dec_len: int,
    dst: np.ndarray       # (expected_dec_len,) uint8 (pre-allocated)
) -> int:
    """
    Zero-Heap JIT Huffman Decoder:
    Decodes bitstream into original byte arrays without tree traversing.
    Returns decoded length.
    """
    bit_buf = 0
    bit_count = 0
    read_idx = 0
    write_idx = 0
    
    while write_idx < expected_dec_len:
        while bit_count < 10 and read_idx < src_len:
            byte_val = src[read_idx]
            read_idx += 1
            bit_buf = (bit_buf << 8) | byte_val
            bit_count += 8
            
        bit1 = (bit_buf >> (bit_count - 1)) & 1
        if bit1 == 0:
            dst[write_idx] = 0
            write_idx += 1
            bit_count -= 1
            bit_buf = bit_buf & ((1 << bit_count) - 1)
            continue
            
        bit2 = (bit_buf >> (bit_count - 2)) & 3
        if bit2 == 2:
            dst[write_idx] = 254
            write_idx += 1
            bit_count -= 2
            bit_buf = bit_buf & ((1 << bit_count) - 1)
            continue
        elif bit2 == 3:
            literal_val = (bit_buf >> (bit_count - 10)) & 0xFF
            dst[write_idx] = literal_val
            write_idx += 1
            bit_count -= 10
            bit_buf = bit_buf & ((1 << bit_count) - 1)
            continue
        else:
            break
            
    return write_idx


class TelemetryHuffmanCompactor:
    """
    Performs joint RLE + Huffman compression.
    Operates on pre-allocated buffers with zero heap allocations.
    """
    def __init__(self):
        self.rle_compressor = TelemetryPayloadCompressor()
        
        # Pre-allocated arrays
        self._huff_compress_buf = np.zeros(250, dtype=np.uint8)
        self._huff_decompress_buf = np.zeros(250, dtype=np.uint8)
        
        # Warmup compiler
        self._warmup()

    def _warmup(self):
        dummy_src = np.zeros(20, dtype=np.uint8)
        _huffman_encode_jit(dummy_src, 20, self._huff_compress_buf)
        _huffman_decode_jit(self._huff_compress_buf, 5, 20, self._huff_decompress_buf)
        self._huff_compress_buf.fill(0)
        self._huff_decompress_buf.fill(0)

    def compress(self, raw_frame: bytes) -> bytes:
        """Applies RLE then Huffman compression."""
        # 1. Run-length encode
        rle_bytes = self.rle_compressor.compress(raw_frame)
        rle_arr = np.frombuffer(rle_bytes, dtype=np.uint8)
        
        # 2. Huffman encode JIT
        huff_len = _huffman_encode_jit(rle_arr, len(rle_bytes), self._huff_compress_buf)
        
        # Return compressed payload along with RLE length header (1 byte)
        header = bytes([len(rle_bytes)])
        return header + bytes(self._huff_compress_buf[:huff_len])

    def decompress(self, comp_payload: bytes) -> bytes:
        """Applies Huffman decompression then RLE decompression."""
        rle_len = comp_payload[0]
        huff_bytes = comp_payload[1:]
        
        huff_arr = np.frombuffer(huff_bytes, dtype=np.uint8)
        
        # 1. Huffman decode JIT
        dec_len = _huffman_decode_jit(
            huff_arr, len(huff_bytes), rle_len, self._huff_decompress_buf
        )
        
        # 2. RLE decode
        rle_bytes = bytes(self._huff_decompress_buf[:dec_len])
        return self.rle_compressor.decompress(rle_bytes)


# =========================================================================
# DETERMINISTIC COMPRESSION BENCHMARK HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Huffman Compactor Benchmark")
    print("==================================================================")
    
    compactor = TelemetryHuffmanCompactor()
    
    import struct
    
    # 1. Nominal payload (simulates active telemetry with alternating values)
    nominal_frame = struct.pack(
        "<B iddd dddd dddd bbbb d i",
        1, 0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0, 1, 0, 1,
        12345.67,  # static timestamp
        0
    )
    
    # 2. Critical Alert payload (dense metrics)
    alert_frame = struct.pack(
        "<B iddd dddd dddd bbbb d i",
        1, 3, 0.95, 0.82, 64.5,
        0.1, 0.2, 0.3, 0.4,
        0.01, 0.02, 0.03, 0.04,
        0, 1, 1, 0,
        time.time(),
        5
    )

    # 1. Functional correctness validation
    print("[*] Performing compactor functional verification...")
    c_nominal = compactor.compress(nominal_frame)
    d_nominal = compactor.decompress(c_nominal)
    assert d_nominal == nominal_frame, "Nominal Huffman round-trip corrupted!"
    
    c_alert = compactor.compress(alert_frame)
    d_alert = compactor.decompress(c_alert)
    assert d_alert == alert_frame, "Alert Huffman round-trip corrupted!"
    print("    -> Functional verification: [PASSED]")
    
    # 2. Compression Ratio Comparisons
    print("\n[*] Comparing Payload Sizes (in bytes):")
    print(f"    -> Raw Frame:             {len(nominal_frame)} B")
    
    rle_nominal_len = len(compactor.rle_compressor.compress(nominal_frame))
    print(f"    -> Nominal (RLE only):     {rle_nominal_len} B")
    print(f"    -> Nominal (RLE+Huffman):  {len(c_nominal)} B")
    
    rle_alert_len = len(compactor.rle_compressor.compress(alert_frame))
    print(f"    -> Alert (RLE only):       {rle_alert_len} B")
    print(f"    -> Alert (RLE+Huffman):    {len(c_alert)} B")
    
    # 3. Speed Run Benchmark (50,000 runs)
    print("\n[*] Starting 50,000 loop speed run...")
    t0 = time.perf_counter()
    for _ in range(50000):
        c = compactor.compress(nominal_frame)
        _ = compactor.decompress(c)
    t1 = time.perf_counter()
    
    avg_us = (t1 - t0) * 1e6 / 50000
    comp_ratio = len(nominal_frame) / len(c_nominal)
    
    print("\n--- COMPACTOR SPEED & COMPRESSION HUD ---")
    print(f"  Avg Encode+Decode Latency:    {avg_us:.3f} µs per cycle")
    print(f"  Nominal Compression Ratio:    {comp_ratio:.2f}x")
    print(f"  Nominal Bandwidth Saving:     {(1.0 - 1.0/comp_ratio)*100:.1f}%")
    
    assert len(c_nominal) < rle_nominal_len, "Huffman pass failed to compress nominal RLE further!"
    assert avg_us < 6.0, "Compactor cycle latency exceeded limit!"
    print("\n[PASSED] Huffman compactor validation successfully completed.")
