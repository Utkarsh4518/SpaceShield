"""
Task 66.2: Anomaly Event Tagging Module
SpaceShield High-Velocity Receiver DSP Subsystem

Disambiguates RF interference from transport stalls and packs findings
into a compact 32-bit packed event tag to prevent telemetry overhead.
"""

import math
import numpy as np
from numba import njit

# Primary Anomaly Codes
CODE_CLEAN_OPERATION = 0
CODE_PURE_RF_DEGRADATION = 1
CODE_PURE_TRANSPORT_STALL = 2
CODE_MIXED_FAULT_CONDITION = 3
CODE_HANDOVER_IN_PROGRESS = 4

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _classify_and_pack_tag_jit(
    loopback_latency_s: float,
    dma_stall_flag: int,
    buf_occupancy: float,
    null_depth_db: float,
    handover_active: int,
    seq_idx: int
) -> int:
    """
    Zero-Heap JIT anomaly source classifier and packet tagger.
    Packs indicators into a 32-bit unsigned integer:
      - Bits 0-2 (3 bits): Primary anomaly classification code.
      - Bits 3-5 (3 bits): Timing warning state.
      - Bits 6-8 (3 bits): DMA alarm state.
      - Bits 9-11 (3 bits): RF alarm state.
      - Bits 12-19 (8 bits): Frame sequence offset LSB.
      - Bits 20-31 (12 bits): Absolute null depth (0-100 dB).
    """
    # 1. Determine component failure indicators
    rf_bad = 1 if null_depth_db >= -20.0 else 0
    transport_bad = 1 if (loopback_latency_s >= 0.008 or dma_stall_flag == 1 or buf_occupancy > 80.0) else 0
    
    # 2. Compute primary classification
    if handover_active == 1:
        primary_code = CODE_HANDOVER_IN_PROGRESS
    elif rf_bad == 1 and transport_bad == 1:
        primary_code = CODE_MIXED_FAULT_CONDITION
    elif rf_bad == 0 and transport_bad == 1:
        primary_code = CODE_PURE_TRANSPORT_STALL
    elif rf_bad == 1 and transport_bad == 0:
        primary_code = CODE_PURE_RF_DEGRADATION
    else:
        primary_code = CODE_CLEAN_OPERATION
        
    # 3. Pack fields
    p_code = primary_code & 0x7
    t_state = (1 if loopback_latency_s >= 0.004 else 0) & 0x7
    d_state = (1 if (dma_stall_flag == 1 or buf_occupancy > 80.0) else 0) & 0x7
    r_state = rf_bad & 0x7
    seq_field = seq_idx & 0xFF
    
    # Map null depth to positive bounded 12-bit integer: range [-100, 0] dB mapped to [0, 100]
    db_val = int(min(max(-null_depth_db, 0.0), 100.0)) & 0xFFF
    
    packed = (
        (db_val << 20) |
        (seq_field << 12) |
        (r_state << 9) |
        (d_state << 6) |
        (t_state << 3) |
        p_code
    )
    return packed


@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _unpack_tag_jit(packed: int) -> tuple[int, int, int, int, int, int]:
    """
    JIT-compiled unpacker:
    Returns (primary_code, timing_state, dma_state, rf_state, seq_idx_lsb, null_depth_abs).
    """
    p_code = packed & 0x7
    t_state = (packed >> 3) & 0x7
    d_state = (packed >> 6) & 0x7
    r_state = (packed >> 9) & 0x7
    seq_field = (packed >> 12) & 0xFF
    db_val = (packed >> 20) & 0xFFF
    return p_code, t_state, d_state, r_state, seq_field, db_val


class AnomalyEventTagger:
    """
    Disambiguates and logs anomaly events.
    Runs on an allocation-free update path.
    """
    def __init__(self):
        pass

    def generate_tag(
        self,
        loopback_latency_s: float,
        dma_stall_flag: int,
        buf_occupancy: float,
        null_depth_db: float,
        handover_active: int,
        seq_idx: int
    ) -> int:
        """
        Generates a packed 32-bit anomaly event tag.
        """
        return _classify_and_pack_tag_jit(
            loopback_latency_s,
            dma_stall_flag,
            buf_occupancy,
            null_depth_db,
            handover_active,
            seq_idx
        )

    def unpack_tag(self, packed: int) -> dict:
        """
        Unpacks a 32-bit packed tag into a metadata dictionary.
        """
        p_code, t_state, d_state, r_state, seq_field, db_val = _unpack_tag_jit(packed)
        return {
            "primary_code": p_code,
            "timing_state": t_state,
            "dma_state": d_state,
            "rf_state": r_state,
            "seq_idx_lsb": seq_field,
            "null_depth_db": -float(db_val)
        }


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Anomaly Event Tagging Validation")
    print("==================================================================")
    
    tagger = AnomalyEventTagger()
    
    # 1. Clean operation
    print("[*] Scenario 1: Clean Operation...")
    tag_clean = tagger.generate_tag(
        loopback_latency_s=0.002,
        dma_stall_flag=0,
        buf_occupancy=15.0,
        null_depth_db=-45.0,
        handover_active=0,
        seq_idx=101
    )
    meta_clean = tagger.unpack_tag(tag_clean)
    print(f"    -> Code: {meta_clean['primary_code']} | Null: {meta_clean['null_depth_db']:.1f} dB")
    assert meta_clean["primary_code"] == CODE_CLEAN_OPERATION, "Failed to classify clean operation!"
    print("    -> Clean check: [PASSED]")
    
    # 2. Pure RF Degradation (interference)
    print("\n[*] Scenario 2: Pure RF Degradation...")
    tag_rf = tagger.generate_tag(
        loopback_latency_s=0.002,
        dma_stall_flag=0,
        buf_occupancy=15.0,
        null_depth_db=-12.5,  # Shallow null depth (interference)
        handover_active=0,
        seq_idx=102
    )
    meta_rf = tagger.unpack_tag(tag_rf)
    print(f"    -> Code: {meta_rf['primary_code']} | Null: {meta_rf['null_depth_db']:.1f} dB")
    assert meta_rf["primary_code"] == CODE_PURE_RF_DEGRADATION, "Failed to classify pure RF anomaly!"
    print("    -> RF degradation check: [PASSED]")

    # 3. Pure Transport Stall
    print("\n[*] Scenario 3: Pure Transport Stall...")
    tag_trans = tagger.generate_tag(
        loopback_latency_s=0.009,  # high latency
        dma_stall_flag=1,         # DMA stall active
        buf_occupancy=92.0,        # high buffer occupancy
        null_depth_db=-42.0,       # null is healthy
        handover_active=0,
        seq_idx=103
    )
    meta_trans = tagger.unpack_tag(tag_trans)
    print(f"    -> Code: {meta_trans['primary_code']} | Timing State: {meta_trans['timing_state']}")
    assert meta_trans["primary_code"] == CODE_PURE_TRANSPORT_STALL, "Failed to classify pure transport stall!"
    print("    -> Transport stall check: [PASSED]")

    # 4. Mixed Fault Conditions (both RF and Transport bad)
    print("\n[*] Scenario 4: Mixed Fault Condition...")
    tag_mixed = tagger.generate_tag(
        loopback_latency_s=0.009,
        dma_stall_flag=1,
        buf_occupancy=92.0,
        null_depth_db=-15.0,  # shallow null AND dma stall!
        handover_active=0,
        seq_idx=104
    )
    meta_mixed = tagger.unpack_tag(tag_mixed)
    print(f"    -> Code: {meta_mixed['primary_code']}")
    assert meta_mixed["primary_code"] == CODE_MIXED_FAULT_CONDITION, "Failed to classify mixed fault conditions!"
    print("    -> Mixed fault check: [PASSED]")

    print("\n[+] Anomaly event tagging validation complete.")
