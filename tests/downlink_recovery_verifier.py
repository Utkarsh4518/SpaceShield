#!/usr/bin/env python3
"""
SpaceShield: Protocol Compliance Verification Suite
Description: Rigorous validation harness to prove that the Dual-Stage Forward Error 
             Correction (ECC) matrix successfully tracks, isolates, and repairs heavy 
             packet degradation (up to 12 destroyed bytes) inside the strict 60µs 
             baseband execution window.
"""

import os
import sys
import time
import json
import hashlib
import random
import numpy as np

# Dynamically link the backend modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'backend', 'src'))
sys.path.append(os.path.join(BASE_DIR, 'tests'))

from soft_decision_decoder import SoftDecisionDecoder

class DownlinkRecoveryVerifier:
    def __init__(self):
        self.decoder = SoftDecisionDecoder(rs_blocks_per_stride=1)
        self.ledger_path = os.path.join(BASE_DIR, 'compliance', 'certin_incident_spoofing.json')

    def _commit_to_worm_ledger(self, metrics: dict):
        """Appends the verification pass/fail matrix to the STQC WORM log."""
        metrics["timestamp"] = time.time()
        metrics["incident_type"] = "ECC_DOWNLINK_RECOVERY_VERIFICATION"
        
        last_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            last_hash = parsed[-1].get("hash", last_hash)
                        elif isinstance(parsed, dict):
                            last_hash = parsed.get("hash", last_hash)
            except json.JSONDecodeError:
                with open(self.ledger_path, "r") as f:
                    for line in reversed(f.readlines()):
                        if line.strip():
                            try:
                                last_hash = json.loads(line).get("hash", last_hash)
                                break
                            except: pass
                            
        metrics["previous_hash"] = last_hash
        raw_string = json.dumps(metrics, sort_keys=True)
        metrics["hash"] = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
        
        try:
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except PermissionError:
            print("[!] Permission Denied on WORM Ledger. Cannot commit test matrix.")

    def run_validation(self):
        print("==================================================================")
        print(" SpaceShield Protocol Execution: Downlink Recovery Verifier")
        print("==================================================================")
        
        # 1. Generate Clean Baseline Null-Vector Frame
        # The structural RS(255,223) test mock uses 0x00 as the nominal clean byte mapping.
        # So we synthesize soft probabilities completely grouped near 0.0
        clean_soft_bits = np.random.uniform(0.0, 0.2, self.decoder.soft_bit_length).astype(np.float32)
        
        # 2. Deliberately Inject Intense Code-Division Damage (12 Random Byte Errors)
        # We simulate a burst jammer wiping out exact chunks of the framing sequence
        target_byte_errors = 12
        corrupted_soft_bits = clean_soft_bits.copy()
        
        corrupted_byte_indices = random.sample(range(255), target_byte_errors)
        
        total_bit_errors = 0
        for byte_idx in corrupted_byte_indices:
            # Each byte maps to 8 bits. In Rate 1/2, each bit is represented by 2 soft probabilities.
            bit_offset = byte_idx * 8 * 2
            for i in range(16):
                # Force maximum soft-distance corruption mapping to hard '1' bits
                corrupted_soft_bits[bit_offset + i] = 1.0
                total_bit_errors += 1
                
        # Calculate raw Pre-Correction BER
        total_frame_bits = self.decoder.viterbi_byte_length * 8
        pre_correction_ber = total_bit_errors / total_frame_bits
        
        print(f"[*] Synthesizing Catastrophic RF Flooding Strike...")
        print(f"    -> Target Frame Size:   {self.decoder.viterbi_byte_length} bytes")
        print(f"    -> Destroyed Bytes:     {target_byte_errors} bytes")
        print(f"    -> Pre-Correction BER:  {pre_correction_ber:.4f} (Extremely Degraded)")
        
        # 3. Burn-in Numba Compilation
        self.decoder.decode_stride(clean_soft_bits)
        
        # 4. Route through the Vector-Accelerated Soft-Decision ECC Pipeline
        print("\n[*] Routing corrupted stream into Dual-Stage ECC Matrices...")
        
        rep_errors, recovered_payload, exec_us = self.decoder.decode_stride(corrupted_soft_bits)
        
        # 5. Extract Verification Bounds
        # Any byte != 0 is an uncorrected error in the final 223 byte payload.
        remaining_byte_errors = np.count_nonzero(recovered_payload)
        post_correction_ber = (remaining_byte_errors * 8) / (self.decoder.rs_payload_length * 8)
        
        print(f"\n--- ECC DOWNLINK VERIFICATION PROFILES ---")
        print(f" [>] Pre-Correction BER:   {pre_correction_ber:.4f}")
        print(f" [>] Post-Correction BER:  {post_correction_ber:.4f}")
        print(f" [>] Framing Corrections:  {rep_errors} bytes fully reconstructed")
        print(f" [>] Matrix Latency:       {exec_us:.2f} µs")
        
        ber_passed = post_correction_ber == 0.0
        latency_passed = exec_us < 60.0
        
        overall_pass = ber_passed and latency_passed
        
        metrics = {
            "pre_correction_ber": float(pre_correction_ber),
            "post_correction_ber": float(post_correction_ber),
            "repaired_bytes": int(rep_errors),
            "latency_us": float(exec_us),
            "status": "PASS" if overall_pass else "FAIL"
        }
        
        print("\n[*] Committing Protocol ECC boundaries to cryptographic compliance ledger...")
        self._commit_to_worm_ledger(metrics)
        
        if overall_pass:
            print("[+] SYSTEM ARCHITECTURE VERIFIED: 100% Payload Extracted Under Target 60µs Constraints!")
            sys.exit(0)
        else:
            print("[!] SYSTEM CRITICAL: PACKET DEGRADATION LEAKED PAST REED-SOLOMON PARITY CHECK.")
            sys.exit(1)

if __name__ == "__main__":
    verifier = DownlinkRecoveryVerifier()
    verifier.run_validation()
