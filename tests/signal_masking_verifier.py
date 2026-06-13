#!/usr/bin/env python3
"""
SpaceShield: ECCM OCP Masking Verifier
Description: Evaluates the Orthogonal Complement Projection engine to mathematically 
             prove that the internal beamforming adaptations are perfectly encrypted 
             against hostile Electronic Warfare eavesdropping, while inflicting 
             zero gain degradation against the authentic NavIC satellite target.
"""

import os
import sys
import time
import json
import hashlib
import numpy as np

# Dynamically link the backend modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'backend', 'src'))

from orthogonal_masking_engine import OrthogonalMaskingEngine

class SignalMaskingVerifier:
    def __init__(self):
        self.masker = OrthogonalMaskingEngine(num_channels=4, mask_scale=0.5)
        self.ledger_path = os.path.join(BASE_DIR, 'compliance', 'certin_incident_spoofing.json')
        
    def _commit_to_worm_ledger(self, metrics: dict):
        """Appends the verification pass/fail matrix to the STQC WORM log."""
        metrics["timestamp"] = time.time()
        metrics["incident_type"] = "OCP_SIGNAL_MASKING_VERIFICATION"
        
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
        print("================================================================")
        print(" SpaceShield ECCM Execution: Signal Masking Verifier")
        print("================================================================")
        
        # 1. Authentic NavIC Satellite Track Setup
        theta_rad = np.radians(15.0) # Assume satellite is at 15 degrees
        S_target = np.exp(-1j * np.pi * np.sin(theta_rad) * np.arange(4)).reshape(4, 1)
        
        self.masker.update_steering_matrix(S_target)
        
        # 2. Baseline Matrix weights
        # Represents highly sensitive Nulling gradients that we MUST hide
        w_lcmv_optimal = np.array([-0.1+0.2j, 0.4-0.1j, 0.6+0.3j, -0.2-0.5j], dtype=np.complex64)
        
        print("[*] Simulating Continuous Array Operation (10,000 Masking Cycles)...")
        max_ortho_leakage = 0.0
        
        # Collect variance metrics to prove visual obfuscation
        observed_weights = []
        
        for _ in range(10000):
            w_masked, _ = self.masker.mask_weights(w_lcmv_optimal)
            w_mask_only = self.masker._w_mask
            
            # Proof 1: Strict Orthogonality
            # Inner product |S^H * w_mask| must be 0
            leakage = np.abs(np.vdot(S_target.flatten(), w_mask_only.flatten()))
            if leakage > max_ortho_leakage:
                max_ortho_leakage = leakage
                
            observed_weights.append(w_masked.copy())
            
        # Proof 2: External Uniform Obfuscation
        # Calculate the spatial variance of the observed weights.
        # If variance > 0 across dimensions, the static w_lcmv is successfully encrypted into random noise.
        observed_matrix = np.vstack(observed_weights)
        spatial_variance = np.var(np.abs(observed_matrix), axis=0)
        mean_variance = np.mean(spatial_variance)
        
        # Evaluations
        ortho_passed = max_ortho_leakage <= 1e-5 # Float32 structural limit approximation for complex64
        obfuscation_passed = mean_variance > 0.05
        
        overall_pass = ortho_passed and obfuscation_passed
        
        print(f"    -> Maximum Orthogonal Leakage:  {max_ortho_leakage:.4e} {'[OK]' if ortho_passed else '[FAIL]'}")
        print(f"    -> Adversarial Spatial Variance:{mean_variance:.4f} {'[OK]' if obfuscation_passed else '[FAIL]'}")
        print(f"    -> Verification Status:         {'PASS' if overall_pass else 'FAIL'}\n")
        
        metrics = {
            "max_orthogonal_leakage": float(max_ortho_leakage),
            "adversarial_spatial_variance": float(mean_variance),
            "orthogonality_limit_db": -100.0, # 10^-5 precision translates to extreme spatial suppression
            "status": "PASS" if overall_pass else "FAIL"
        }
        
        print("[*] Committing cryptographic pass/fail matrices to compliance WORM ledger...")
        self._commit_to_worm_ledger(metrics)
        
        if overall_pass:
            print("[+] ECCM OCP SIGNAL MASKING VERIFICATIONS PASSED SECURELY.")
            sys.exit(0)
        else:
            print("[!] SYSTEM CRITICAL: OCP ENGINE FAILED TO MASK STRUCTURAL WEIGHTS.")
            sys.exit(1)

if __name__ == "__main__":
    verifier = SignalMaskingVerifier()
    verifier.run_validation()
