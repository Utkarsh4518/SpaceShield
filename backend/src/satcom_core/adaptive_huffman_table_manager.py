"""
Task 63.3 / 64.3: Adaptive Huffman Table Manager Module
SpaceShield High-Velocity Receiver DSP Subsystem

Provides dynamic switching between nominal and critical Huffman tables
using a 1-byte packed header: MSB denotes table ID, remaining bits denote RLE length.
"""

import time
import numpy as np
from numba import njit

from telemetry_payload_compressor import TelemetryPayloadCompressor

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _adaptive_huffman_encode_jit(
    src: np.ndarray,      # (src_len,) uint8
    src_len: int,
    table_id: int,
    dst: np.ndarray       # (max_dst_len,) uint8
) -> int:
    """
    Zero-Heap JIT Huffman Encoder with Dynamic Tables:
    Table 0: Optimized for Nominal (Zero-heavy)
    Table 1: Optimized for Critical (Alert-heavy)
    """
    bit_buf = 0
    bit_count = 0
    write_idx = 0
    
    for i in range(src_len):
        val = src[i]
        
        code = 0
        code_len = 0
        
        if table_id == 0:
            # Table 0: Nominal (Zero-heavy)
            if val == 0:
                code = 0
                code_len = 1
            elif val == 254:
                code = 2
                code_len = 2
            else:
                code = (3 << 8) | int(val)
                code_len = 10
        else:
            # Table 1: Critical (Alert-heavy)
            if val == 3:
                code = 0
                code_len = 1
            elif val == 1:
                code = 2
                code_len = 2
            elif val == 0:
                code = 6  # 0b110
                code_len = 3
            elif val == 254:
                code = 14  # 0b1110
                code_len = 4
            else:
                code = (15 << 8) | int(val)
                code_len = 12
                
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
def _adaptive_huffman_decode_jit(
    src: np.ndarray,      # (src_len,) uint8
    src_len: int,
    table_id: int,
    expected_dec_len: int,
    dst: np.ndarray       # (expected_dec_len,) uint8
) -> int:
    """
    Zero-Heap JIT Huffman Decoder with Dynamic Tables:
    Decodes bitstream into original byte arrays based on table_id.
    """
    bit_buf = 0
    bit_count = 0
    read_idx = 0
    write_idx = 0
    
    while write_idx < expected_dec_len:
        while bit_count < 12 and read_idx < src_len:
            byte_val = src[read_idx]
            read_idx += 1
            bit_buf = (bit_buf << 8) | byte_val
            bit_count += 8
            
        if table_id == 0:
            # Decode Table 0
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
        else:
            # Decode Table 1
            bit1 = (bit_buf >> (bit_count - 1)) & 1
            if bit1 == 0:
                dst[write_idx] = 3
                write_idx += 1
                bit_count -= 1
                bit_buf = bit_buf & ((1 << bit_count) - 1)
                continue
                
            bit2 = (bit_buf >> (bit_count - 2)) & 3
            if bit2 == 2:
                dst[write_idx] = 1
                write_idx += 1
                bit_count -= 2
                bit_buf = bit_buf & ((1 << bit_count) - 1)
                continue
                
            bit3 = (bit_buf >> (bit_count - 3)) & 7
            if bit3 == 6:
                dst[write_idx] = 0
                write_idx += 1
                bit_count -= 3
                bit_buf = bit_buf & ((1 << bit_count) - 1)
                continue
                
            bit4 = (bit_buf >> (bit_count - 4)) & 15
            if bit4 == 14:
                dst[write_idx] = 254
                write_idx += 1
                bit_count -= 4
                bit_buf = bit_buf & ((1 << bit_count) - 1)
                continue
            elif bit4 == 15:
                literal_val = (bit_buf >> (bit_count - 12)) & 0xFF
                dst[write_idx] = literal_val
                write_idx += 1
                bit_count -= 12
                bit_buf = bit_buf & ((1 << bit_count) - 1)
                continue
            else:
                break
                
    return write_idx


class AdaptiveHuffmanTableManager:
    """
    Manages RLE and multi-table Huffman compression.
    Detects frame threat level and switches coding tables automatically.
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
        _adaptive_huffman_encode_jit(dummy_src, 20, 0, self._huff_compress_buf)
        _adaptive_huffman_encode_jit(dummy_src, 20, 1, self._huff_compress_buf)
        _adaptive_huffman_decode_jit(self._huff_compress_buf, 5, 0, 20, self._huff_decompress_buf)
        _adaptive_huffman_decode_jit(self._huff_compress_buf, 5, 1, 20, self._huff_decompress_buf)
        self._huff_compress_buf.fill(0)
        self._huff_decompress_buf.fill(0)

    def compress(self, raw_frame: bytes) -> bytes:
        """Compresses payload, selecting Table 1 for threat level 3, otherwise Table 0."""
        # 1. Detect threat level
        threat_state = raw_frame[1]  # Second byte denotes threat level in frame schema
        table_id = 1 if threat_state == 3 else 0
        
        # 2. Run-length encode
        rle_bytes = self.rle_compressor.compress(raw_frame)
        rle_arr = np.frombuffer(rle_bytes, dtype=np.uint8)
        
        # 3. Huffman encode JIT
        huff_len = _adaptive_huffman_encode_jit(
            rle_arr, len(rle_bytes), table_id, self._huff_compress_buf
        )
        
        # Packed header: MSB contains table_id, remaining bits contain rle_len
        header = bytes([len(rle_bytes) | (table_id << 7)])
        return header + bytes(self._huff_compress_buf[:huff_len])

    def decompress(self, comp_payload: bytes) -> bytes:
        """Extracts table ID and RLE length, then decompresses."""
        header_byte = comp_payload[0]
        table_id = (header_byte >> 7) & 1
        rle_len = header_byte & 127
        
        huff_bytes = comp_payload[1:]
        huff_arr = np.frombuffer(huff_bytes, dtype=np.uint8)
        
        # 1. Huffman decode JIT
        dec_len = _adaptive_huffman_decode_jit(
            huff_arr, len(huff_bytes), table_id, rle_len, self._huff_decompress_buf
        )
        
        # 2. RLE decode
        rle_bytes = bytes(self._huff_decompress_buf[:dec_len])
        return self.rle_compressor.decompress(rle_bytes)


# =========================================================================
# DETERMINISTIC COMPRESSION BENCHMARK HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Adaptive Huffman Manager Benchmark")
    print("==================================================================")
    
    manager = AdaptiveHuffmanTableManager()
    
    import struct
    
    # 1. Nominal payload (threat_state = 0, zero-heavy)
    nominal_frame = struct.pack(
        "<B iddd dddd dddd bbbb d i",
        1, 0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0, 1, 0, 1,
        12345.67,
        0
    )
    
    # 2. Critical Alert payload (threat_state = 3, alert-heavy)
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
    print("[*] Performing adaptive manager functional verification...")
    c_nominal = manager.compress(nominal_frame)
    d_nominal = manager.decompress(c_nominal)
    assert d_nominal == nominal_frame, "Nominal adaptive round-trip corrupted!"
    
    c_alert = manager.compress(alert_frame)
    d_alert = manager.decompress(c_alert)
    assert d_alert == alert_frame, "Alert adaptive round-trip corrupted!"
    print("    -> Functional verification: [PASSED]")
    
    # 2. Compression Ratio Comparisons
    print("\n[*] Comparing Payload Sizes (in bytes):")
    print(f"    -> Raw Frame:             {len(nominal_frame)} B")
    
    rle_nominal_len = len(manager.rle_compressor.compress(nominal_frame))
    print(f"    -> Nominal (RLE only):     {rle_nominal_len} B")
    print(f"    -> Nominal (Adaptive):     {len(c_nominal)} B")
    
    rle_alert_len = len(manager.rle_compressor.compress(alert_frame))
    print(f"    -> Alert (RLE only):       {rle_alert_len} B")
    print(f"    -> Alert (Adaptive):       {len(c_alert)} B")
    
    # 3. Speed Run Benchmark (50,000 runs)
    print("\n[*] Starting 50,000 loop speed run...")
    t0 = time.perf_counter()
    for _ in range(50000):
        c = manager.compress(nominal_frame)
        _ = manager.decompress(c)
    t1 = time.perf_counter()
    
    avg_us = (t1 - t0) * 1e6 / 50000
    comp_ratio = len(nominal_frame) / len(c_nominal)
    
    print("\n--- ADAPTIVE COMPACTOR SPEED & COMPRESSION HUD ---")
    print(f"  Avg Encode+Decode Latency:    {avg_us:.3f} µs per cycle")
    print(f"  Nominal Compression Ratio:    {comp_ratio:.2f}x")
    print(f"  Nominal Bandwidth Saving:     {(1.0 - 1.0/comp_ratio)*100:.1f}%")
    
    assert len(c_nominal) < rle_nominal_len, "Adaptive pass failed to compress nominal RLE further!"
    assert avg_us < 6.0, "Adaptive compactor cycle latency exceeded limit!"
    print("\n[PASSED] Adaptive Huffman compactor validation successfully completed.")
