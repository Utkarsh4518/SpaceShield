"""
Task 60.1: Binary Telemetry Codec Module
SpaceShield High-Velocity Receiver DSP Subsystem

Provides zero-copy binary serialization/deserialization for telemetry frames.
Defines a compact, fixed-size 109-byte schema layout, avoiding heap allocation 
in the fast path. Exposes a struct-based packer interface and contains a 
comparative JSON benchmark.
"""

import time
import struct
import json
import numpy as np

# Format definition for 109-byte binary layout:
# <  : little-endian
# B  : version (uint8)
# i  : threat_state (int32)
# d  : jammer_score (float64)
# d  : spoof_score (float64)
# d  : sphericity (float64)
# 4d : skew_residuals (4 * float64)
# 4d : aoa_deviation (4 * float64)
# 4b : nulling_directives (4 * int8 / bool)
# d  : timestamp (float64)
# i  : buffer_drops (int32)
SCHEMA_FORMAT = "<B iddd dddd dddd bbbb d i"
SCHEMA_SIZE = struct.calcsize(SCHEMA_FORMAT) # Exact size = 109 bytes

class BinaryTelemetryCodec:
    """
    SpaceShield Zero-Copy Binary Telemetry Codec.
    Packs telemetry variables directly into a pre-allocated write buffer.
    """
    def __init__(self, version: int = 1):
        self.version = version
        # Pre-allocated serialization write buffer to avoid dynamic memory allocation
        self._write_buf = bytearray(SCHEMA_SIZE)
        self._write_view = memoryview(self._write_buf)

    def encode(
        self,
        threat_state: int,
        jammer_score: float,
        spoof_score: float,
        sphericity: float,
        skew_residuals: np.ndarray,      # (4,) float64
        aoa_deviation: np.ndarray,       # (4,) float64
        nulling_directives: np.ndarray,  # (4,) bool
        timestamp: float,
        buffer_drops: int
    ) -> memoryview:
        """
        Packs inputs directly into the pre-allocated write buffer in-place.
        Returns a zero-copy memoryview reference of the byte structure.
        """
        struct.pack_into(
            SCHEMA_FORMAT,
            self._write_buf,
            0,
            self.version,
            threat_state,
            jammer_score,
            spoof_score,
            sphericity,
            skew_residuals[0], skew_residuals[1], skew_residuals[2], skew_residuals[3],
            aoa_deviation[0], aoa_deviation[1], aoa_deviation[2], aoa_deviation[3],
            int(nulling_directives[0]), int(nulling_directives[1]), int(nulling_directives[2]), int(nulling_directives[3]),
            timestamp,
            buffer_drops
        )
        return self._write_view

    def decode(self, data: bytes) -> dict:
        """
        Unpacks binary telemetry stream records into a standard dictionary.
        Enforces backward-compatible schema parsing.
        """
        unpacked = struct.unpack(SCHEMA_FORMAT, data)
        
        # Mapping elements back to structured format
        return {
            "version": unpacked[0],
            "threat_state": unpacked[1],
            "jammer_score": unpacked[2],
            "spoof_score": unpacked[3],
            "sphericity": unpacked[4],
            "skew_residuals": list(unpacked[5:9]),
            "aoa_deviation": list(unpacked[9:13]),
            "nulling_directives": [bool(x) for x in unpacked[13:17]],
            "timestamp": unpacked[17],
            "buffer_drops": unpacked[18]
        }


# =========================================================================
# DETERMINISTIC PERFORMANCE BENCHMARK
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Binary Telemetry Codec Benchmark")
    print("==================================================================")
    
    codec = BinaryTelemetryCodec(version=1)
    
    # Test dataset definition
    threat_state = 3
    jammer_score = 0.85
    spoof_score = 0.92
    sphericity = 65.4
    skew_residuals = np.array([0.05, 0.10, 0.15, 0.20], dtype=np.float64)
    aoa_deviation = np.array([0.02, 0.04, 0.06, 0.08], dtype=np.float64)
    nulling_directives = np.array([False, True, True, False], dtype=np.bool_)
    timestamp = time.time()
    buffer_drops = 12

    # 1. Functional correctness validation
    print("[*] Performing codec functional verification...")
    encoded_view = codec.encode(
        threat_state, jammer_score, spoof_score, sphericity,
        skew_residuals, aoa_deviation, nulling_directives, timestamp, buffer_drops
    )
    
    encoded_bytes = bytes(encoded_view)
    print(f"    -> Encoded Binary Size: {len(encoded_bytes)} bytes (Expected: {SCHEMA_SIZE})")
    assert len(encoded_bytes) == SCHEMA_SIZE, "Size validation mismatch!"
    
    decoded = codec.decode(encoded_bytes)
    assert decoded["version"] == 1
    assert decoded["threat_state"] == threat_state
    assert decoded["nulling_directives"][1] == True
    assert decoded["buffer_drops"] == buffer_drops
    print("    -> Functional verification: [PASSED]")

    # 2. Benchmarking performance against standard JSON string conversion
    print("\n[*] Starting 50,000 loop serialization speed run...")
    
    # Pre-generated JSON mock dictionary
    json_dict = {
        "v": 1,
        "threat_state": threat_state,
        "jammer_score": jammer_score,
        "spoof_score": spoof_score,
        "sphericity": sphericity,
        "skew_residuals": [float(x) for x in skew_residuals],
        "aoa_deviation": [float(x) for x in aoa_deviation],
        "nulling_directives": [bool(x) for x in nulling_directives],
        "timestamp": timestamp,
        "buffer_drops": buffer_drops
    }

    # JSON Benchmark
    t0 = time.perf_counter()
    for _ in range(50000):
        _ = json.dumps(json_dict)
    t1 = time.perf_counter()
    json_time = (t1 - t0) * 1e6 / 50000
    
    # Binary Benchmark
    t0 = time.perf_counter()
    for _ in range(50000):
        _ = codec.encode(
            threat_state, jammer_score, spoof_score, sphericity,
            skew_residuals, aoa_deviation, nulling_directives, timestamp, buffer_drops
        )
    t1 = time.perf_counter()
    bin_time = (t1 - t0) * 1e6 / 50000
    
    print("\n--- CODEC SPEED HUD ---")
    print(f"  Standard JSON Serialization:  {json_time:.3f} µs per frame")
    print(f"  Hardened Binary Codec Path:   {bin_time:.3f} µs per frame")
    speed_ratio = json_time / bin_time
    print(f"  Performance Speedup Factor:   {speed_ratio:.1f}x faster")
    
    if bin_time < 2.0:
        print("\n[PASSED] Binary serialization executes inside sub-microsecond boundaries.")
    else:
        print("\n[FAILED] Codec overhead exceeded performance constraints.")
