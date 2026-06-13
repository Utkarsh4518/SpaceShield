#!/usr/bin/env python3
"""
SpaceShield: Cross-Correlation Anti-Jamming Verification Suite
Description: Synthesizes high-power (+40dB ISR) code-division spoofing environments 
             and mathematically verifies that the SIC Cancellation block cleanly 
             strips out orthogonal PRN sequences without perturbing the true NavIC payload.
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
sys.path.append(os.path.join(BASE_DIR, 'tests'))

from sic_correlation_canceller import SICCorrelationCanceller

class CrossCorrelationVerifier:
    def __init__(self):
        self.canceller = SICCorrelationCanceller(num_channels=4, stride_length=4096)
        self.ledger_path = os.path.join(BASE_DIR, 'compliance', 'certin_incident_spoofing.json')
        
    def _commit_to_worm_ledger(self, metrics: dict):
        """Appends the verification pass/fail matrix to the STQC WORM log."""
        metrics["timestamp"] = time.time()
        metrics["incident_type"] = "SIC_CROSS_CORRELATION_VERIFICATION"
        
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
        print(" SpaceShield Baseband Execution: Cross-Correlation Verifier")
        print("================================================================")
        
        # 1. Synthesize True NavIC Sequence and Malicious Spoofer Sequence
        # Using completely orthogonal pseudo-random noise (PRN) codes
        navic_prn = (np.sign(np.random.randn(4096)) + 0j).astype(np.complex64)
        spoofer_prn = (np.sign(np.random.randn(4096)) + 0j).astype(np.complex64)
        
        # 2. Construct Clean Ambient NavIC Baseline
        # Nominal target baseline (Amplitude = 1.0) with standard thermal noise
        thermal_noise = (np.random.randn(4, 4096) + 1j * np.random.randn(4, 4096)).astype(np.complex64) * 0.1
        ambient_X = thermal_noise.copy()
        
        # Inject NavIC payload across 4 spatial channels (Assume arriving roughly in-phase for simplicity)
        for ch in range(4):
            ambient_X[ch, :] += navic_prn * 1.0
            
        # Extract Baseline Authentic NavIC Peak Magnitude
        baseline_dots = np.sum(ambient_X * np.conj(navic_prn), axis=1) / 4096.0
        baseline_power_db = 10 * np.log10(np.abs(baseline_dots)**2 + 1e-12)
        print(f"[*] Authentic NavIC Baseline Power: {np.mean(baseline_power_db):.2f} dB (Nominal)")
        
        # 3. Simulate High-ISR Code-Division Attack (+40dB Hostile Power)
        print("[*] Initiating Catastrophic +40dB Code-Division Spoofing Blast...")
        attack_X = ambient_X.copy()
        
        # Inject massive +40dB amplitude (100x voltage multiplier) into the channels
        # Introduce some phase variations to test complex math bounds
        hostile_amplitudes = np.array([100.0, 100.0j, -100.0, -100.0j], dtype=np.complex64).reshape(4, 1)
        attack_X += (hostile_amplitudes * spoofer_prn)
        
        # Extract Sidelobe Interference Power BEFORE mitigation
        infected_dots = np.sum(attack_X * np.conj(spoofer_prn), axis=1) / 4096.0
        infected_power_db = 10 * np.log10(np.abs(infected_dots)**2 + 1e-12)
        
        # 4. Execute Successive Interference Cancellation (SIC) Layer
        print("[*] Engaging Zero-Heap SIC Superposition Matrices...")
        self.canceller.load_interferer_parameters(spoofer_prn)
        
        # Burn-in pass
        self.canceller.execute_sic_stride(attack_X.copy())
        
        # Hot path execution
        clean_X, exec_us = self.canceller.execute_sic_stride(attack_X)
        
        # 5. Extract Verification Sidelobe & Peak Math
        cleaned_spoofer_dots = np.sum(clean_X * np.conj(spoofer_prn), axis=1) / 4096.0
        cleaned_spoofer_power_db = 10 * np.log10(np.abs(cleaned_spoofer_dots)**2 + 1e-12)
        
        recovered_navic_dots = np.sum(clean_X * np.conj(navic_prn), axis=1) / 4096.0
        recovered_navic_power_db = 10 * np.log10(np.abs(recovered_navic_dots)**2 + 1e-12)
        
        suppression_db = infected_power_db - cleaned_spoofer_power_db
        payload_drift_db = recovered_navic_power_db - baseline_power_db
        
        print("\n--- CANCELLATION SUPPRESSION MATRICES ---")
        min_suppression = float('inf')
        max_drift = 0.0
        
        for ch in range(4):
            print(f" [Ch {ch}] Hostile Sidelobe: {infected_power_db[ch]:+.2f} dB -> {cleaned_spoofer_power_db[ch]:+.2f} dB (Suppression: {suppression_db[ch]:.2f} dB)")
            print(f"        NavIC Payload Peak: {baseline_power_db[ch]:+.2f} dB -> {recovered_navic_power_db[ch]:+.2f} dB (Drift: {payload_drift_db[ch]:+.4f} dB)")
            
            min_suppression = min(min_suppression, suppression_db[ch])
            max_drift = max(max_drift, abs(payload_drift_db[ch]))
            
        print(f"\n [>] Sub-System Latency: {exec_us:.2f} µs")
        
        # 6. Evaluation Verification Proof
        sidelobe_passed = min_suppression >= 26.0
        payload_passed = max_drift <= 0.5
        
        overall_pass = sidelobe_passed and payload_passed
        
        metrics = {
            "min_suppression_db": float(min_suppression),
            "max_payload_drift_db": float(max_drift),
            "execution_us": float(exec_us),
            "status": "PASS" if overall_pass else "FAIL"
        }
        
        print("\n[*] Committing SIC variance signatures to cryptographic compliance ledger...")
        self._commit_to_worm_ledger(metrics)
        
        if overall_pass:
            print("[+] SPREAD SPECTRUM CROSS-CORRELATION BOUNDARIES VERIFIED SECURELY.")
            sys.exit(0)
        else:
            print("[!] SYSTEM CRITICAL: INTERFERENCE CANCELLATION MATRICES COMPROMISED.")
            sys.exit(1)

if __name__ == "__main__":
    verifier = CrossCorrelationVerifier()
    verifier.run_validation()
