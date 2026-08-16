"""
Binary Telemetry Codec Protocol Consistency Tests
Verifies the single authoritative wire format shared by the JSON (/stream)
and binary (/stream/binary) telemetry transports: encode/decode round-trip,
malformed frames, missing fields, boundary values, and version handling.
"""

import os
import sys
import struct

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_SRC = os.path.join(BASE_DIR, 'backend', 'src', 'satcom_core')
sys.path.insert(0, BACKEND_SRC)

from binary_telemetry_codec import BinaryTelemetryCodec, SCHEMA_FORMAT, SCHEMA_SIZE


def _sample_fields():
    return dict(
        threat_state=3,
        jammer_score=0.85,
        spoof_score=0.92,
        sphericity=65.4,
        skew_residuals=np.array([0.05, 0.10, 0.15, 0.20], dtype=np.float64),
        aoa_deviation=np.array([0.02, 0.04, 0.06, 0.08], dtype=np.float64),
        nulling_directives=np.array([False, True, True, False], dtype=np.bool_),
        timestamp=1700000000.123456,
        buffer_drops=12,
    )


def test_roundtrip_matches_input():
    codec = BinaryTelemetryCodec(version=1)
    fields = _sample_fields()
    encoded = bytes(codec.encode(**fields))
    assert len(encoded) == SCHEMA_SIZE

    decoded = codec.decode(encoded)
    assert decoded["version"] == 1
    assert decoded["threat_state"] == fields["threat_state"]
    assert abs(decoded["jammer_score"] - fields["jammer_score"]) < 1e-9
    assert abs(decoded["spoof_score"] - fields["spoof_score"]) < 1e-9
    assert abs(decoded["sphericity"] - fields["sphericity"]) < 1e-9
    assert decoded["skew_residuals"] == list(fields["skew_residuals"])
    assert decoded["aoa_deviation"] == list(fields["aoa_deviation"])
    assert decoded["nulling_directives"] == [False, True, True, False]
    assert abs(decoded["timestamp"] - fields["timestamp"]) < 1e-6
    assert decoded["buffer_drops"] == fields["buffer_drops"]
    print("[PASS] Encode/decode round-trip preserves all fields.")


def test_malformed_frame_wrong_size_raises():
    codec = BinaryTelemetryCodec(version=1)
    fields = _sample_fields()
    encoded = bytes(codec.encode(**fields))

    truncated = encoded[:-1]
    try:
        codec.decode(truncated)
        raised = False
    except struct.error:
        raised = True
    assert raised, "Truncated frame should raise struct.error, not decode silently."

    padded = encoded + b"\x00"
    try:
        codec.decode(padded)
        raised = False
    except struct.error:
        raised = True
    assert raised, "Over-length frame should raise struct.error, not decode silently."
    print("[PASS] Malformed (wrong-size) frames raise struct.error rather than silently decoding.")


def test_empty_frame_raises():
    codec = BinaryTelemetryCodec(version=1)
    try:
        codec.decode(b"")
        raised = False
    except struct.error:
        raised = True
    assert raised, "Empty frame should raise struct.error."
    print("[PASS] Empty frame raises struct.error.")


def test_boundary_values():
    codec = BinaryTelemetryCodec(version=1)
    fields = _sample_fields()
    fields.update(
        threat_state=-2147483648,  # int32 min
        jammer_score=0.0,
        spoof_score=float("inf"),
        sphericity=0.0,
        skew_residuals=np.zeros(4, dtype=np.float64),
        aoa_deviation=np.array([1e300, -1e300, 0.0, -0.0], dtype=np.float64),
        nulling_directives=np.array([True, True, True, True], dtype=np.bool_),
        timestamp=0.0,
        buffer_drops=2147483647,  # int32 max
    )
    encoded = bytes(codec.encode(**fields))
    decoded = codec.decode(encoded)
    assert decoded["threat_state"] == -2147483648
    assert decoded["buffer_drops"] == 2147483647
    assert decoded["spoof_score"] == float("inf")
    assert decoded["nulling_directives"] == [True, True, True, True]
    print("[PASS] Boundary values (int32 extremes, inf, zero) survive encode/decode.")


def test_buffer_drops_overflow_raises():
    codec = BinaryTelemetryCodec(version=1)
    fields = _sample_fields()
    fields["buffer_drops"] = 2**31  # one past int32 max
    try:
        codec.encode(**fields)
        raised = False
    except struct.error:
        raised = True
    assert raised, "buffer_drops beyond int32 range should raise struct.error at encode time."
    print("[PASS] Out-of-range int32 field raises struct.error at encode time (fails loud, not silently truncated).")


def test_version_field_is_first_byte_and_round_trips():
    for version in (0, 1, 255):
        codec = BinaryTelemetryCodec(version=version)
        fields = _sample_fields()
        encoded = bytes(codec.encode(**fields))
        assert encoded[0] == version, "version must be the first byte of the wire format."
        decoded = codec.decode(encoded)
        assert decoded["version"] == version
    print("[PASS] Version field occupies byte 0 and round-trips for representable uint8 values.")


def test_schema_size_is_stable_109_bytes():
    # This is the documented, authoritative frame size shared by both
    # HardenedWebSocketRuntime (byte-offset threat_state extraction) and
    # TelemetryDispatcher.broadcast_binary. If this ever changes, every
    # consumer that hardcodes offsets [1:5] for threat_state must be
    # updated in lockstep.
    assert SCHEMA_SIZE == 109
    assert struct.calcsize(SCHEMA_FORMAT) == 109
    print("[PASS] Wire schema remains the documented fixed 109-byte layout.")


if __name__ == "__main__":
    test_roundtrip_matches_input()
    test_malformed_frame_wrong_size_raises()
    test_empty_frame_raises()
    test_boundary_values()
    test_buffer_drops_overflow_raises()
    test_version_field_is_first_byte_and_round_trips()
    test_schema_size_is_stable_109_bytes()
    print("\n[PASSED] All binary telemetry codec protocol tests cleared.")
