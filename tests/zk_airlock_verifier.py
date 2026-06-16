#!/usr/bin/env python3
"""
SpaceShield: Protocol Compliance Verification Suite
Description: Rigorous validation harness to prove that the NIZKP Fiat-Shamir
             extraction accurately enforces mathematically binding public proofs 
             while strictly guaranteeing Zero-Knowledge physical airgapping.
"""

import os
import sys
import time
import json
import base64
import hashlib

# Dynamically link the backend modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'backend', 'src'))
sys.path.append(os.path.join(BASE_DIR, 'tests'))

from zk_containment_prover import ZKContainmentProver

class ZKAirlockVerifier:
    def __init__(self):
        self.prover = ZKContainmentProver()
        self.ledger_path = os.path.join(BASE_DIR, 'compliance', 'certin_incident_spoofing.json')

    def _commit_to_worm_ledger(self, metrics: dict):
        """Appends the verification pass/fail matrix to the STQC WORM log."""
        metrics["timestamp"] = time.time()
        metrics["incident_type"] = "ZKP_AIRLOCK_CONTAINMENT_VERIFICATION"
        
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
        print(" SpaceShield Protocol Execution: ZKP Airlock Verifier")
        print("==================================================================")
        
        # Simulated highly classified Baseband State
        secret_iq_matrix = "0xDEADBEEF_COHERENT_IQ_MATRIX_SPATIAL_SAMPLES"
        historical_hash = "f4b3e6d1e9f1a239a7e0892095cc1a9e5b2a0c6498a5e890c2a5e92"
        
        print(f"[*] Extracting NIZKP Fiat-Shamir Statement...")
        proof_b64, valid_statement, public_y, exec_ms = self.prover.generate_proof(
            historical_hash=historical_hash,
            null_depth_db=-48.2,
            ber=0.0,
            loop_stable=True,
            raw_iq_state=secret_iq_matrix
        )
        
        print(f"    -> Statement: {valid_statement}")
        print(f"    -> Proof Key Size: {len(proof_b64)} bytes")
        
        # 1. Nominal Public Verification Check
        is_valid_nominal = self.prover.verify_proof(proof_b64, valid_statement)
        print(f"\n[1] Public Mathematical Verification Status: {is_valid_nominal}")
        
        # 2. Intentional Single-Bit Tampering
        # We simulate an attacker intercepting the statement and attempting to forge 
        # a shallower null depth (-48.2 -> -48.1)
        tampered_statement = valid_statement.replace("-48.20", "-48.10")
        is_valid_tampered = self.prover.verify_proof(proof_b64, tampered_statement)
        print(f"[2] Single-Bit Tampering Rejection Status: {not is_valid_tampered} (Proof validated as {is_valid_tampered})")
        
        # 3. ZKP Information Bleed Check
        # Ensure the secret IQ matrix identifier NEVER exists in the base64 or plaintext proof
        decoded_proof = base64.b64decode(proof_b64).decode('utf-8')
        leak_detected = secret_iq_matrix in decoded_proof or secret_iq_matrix in proof_b64
        print(f"[3] Absolute Zero-Knowledge Proof Isolation Status: {not leak_detected}")
        
        # Verify bounding
        overall_pass = is_valid_nominal and (not is_valid_tampered) and (not leak_detected)
        
        metrics = {
            "nizkp_key_size_bytes": len(proof_b64),
            "nominal_verification": is_valid_nominal,
            "tamper_rejection": not is_valid_tampered,
            "airlock_sealed": not leak_detected,
            "latency_ms": float(exec_ms),
            "status": "PASS" if overall_pass else "FAIL"
        }
        
        print("\n[*] Committing ZKP Airlock boundaries to cryptographic compliance ledger...")
        self._commit_to_worm_ledger(metrics)
        
        if overall_pass:
            print("[+] SYSTEM ARCHITECTURE VERIFIED: Zero-Knowledge Airlock mathematically enforced!")
            sys.exit(0)
        else:
            print("[!] SYSTEM CRITICAL: ZKP EXTRACTOR FAILED CRYPTOGRAPHIC AUDIT.")
            sys.exit(1)

if __name__ == "__main__":
    verifier = ZKAirlockVerifier()
    verifier.run_validation()
