#!/usr/bin/env python3
"""
SpaceShield: Protocol Compliance Verification Suite
Description: Rigorous security stress-testing harness to prove that the 
             Ed25519 Secure Handshake Interceptor combined with the Lock-Free 
             Memory Nonce Cache successfully blocks Replay Attacks and Temporal 
             Meaconing without breaking hardware latency barriers.
"""

import os
import sys
import time
import json
import hashlib
import struct
from cryptography.hazmat.primitives.asymmetric import ed25519

# Dynamically link the backend modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'backend', 'src'))
sys.path.append(os.path.join(BASE_DIR, 'tests'))

from secure_handshake_interceptor import SecureHandshakeInterceptor

class HandshakeReplayVerifier:
    def __init__(self):
        self.ledger_path = os.path.join(BASE_DIR, 'compliance', 'certin_incident_spoofing.json')
        
    def _commit_to_worm_ledger(self, metrics: dict):
        """Appends the verification pass/fail matrix to the STQC WORM log."""
        metrics["timestamp"] = time.time()
        metrics["incident_type"] = "SECURE_HANDSHAKE_ATTACK_SIMULATION"
        
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
            except:
                pass
                            
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
        print(" SpaceShield Security Verification: Zero-Trust Handshake & Replay")
        print("==================================================================")
        
        # 1. Synthesize Ed25519 Cryptographic Keys for Authorized Target Node
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        node_id = 1
        interceptor = SecureHandshakeInterceptor(authorized_keys={node_id: public_key})
        
        # --- TEST 1: Legitimate Cryptographically Signed Payload ---
        print("\n[*] STAGE 1: Generating Valid Command Payload...")
        nonce = 1000
        timestamp_us = time.time() * 1000000.0
        payload_data = b'{"command": "SHUTDOWN_EXTERNAL_INTERFACE"}'
        
        message_block = struct.pack('<IQQ', node_id, nonce, int(timestamp_us)) + payload_data
        signature = private_key.sign(message_block)
        
        is_valid_1, flag_1, lat_1 = interceptor.intercept_payload(node_id, nonce, timestamp_us, signature, payload_data)
        
        print(f"    -> Interceptor Evaluation: {flag_1}")
        print(f"    -> Valid Verification Latency: {lat_1:.2f} µs")
        
        if not is_valid_1:
            print("[!] FATAL: Legitimate payload was incorrectly quarantined!")
            sys.exit(1)
            
            
        # --- TEST 2: Captured Network Replay Attack ---
        print("\n[*] STAGE 2: Capturing and Re-Transmitting Valid Payload (Replay Attack)...")
        # Attacker literally steals the exact bytes from the wire and blasts them back 
        # a split microsecond later.
        
        is_valid_2, flag_2, lat_2 = interceptor.intercept_payload(node_id, nonce, timestamp_us, signature, payload_data)
        
        print(f"    -> Interceptor Evaluation: {flag_2}")
        print(f"    -> Duplication Memory-Cache Latency: {lat_2:.2f} µs")
        
        if is_valid_2 or flag_2 != "REPLAY_ATTACK":
            print("[!] FATAL: Hardware memory cache leaked a replay attack!")
            sys.exit(1)
            
            
        # --- TEST 3: Expired Temporal Spoofing Attack ---
        print("\n[*] STAGE 3: Synthesizing Legitimate Payload with Expired Temporal Bounds...")
        # Attacker intercepts a valid payload, holds it in a buffer for 250ms, then transmits it 
        # attempting to desynchronize the active array tracking loops.
        nonce_3 = 1001
        
        # Simulate generating the payload 250ms in the past
        past_timestamp_us = (time.time() - 0.25) * 1000000.0 
        
        message_block_3 = struct.pack('<IQQ', node_id, nonce_3, int(past_timestamp_us)) + payload_data
        signature_3 = private_key.sign(message_block_3)
        
        is_valid_3, flag_3, lat_3 = interceptor.intercept_payload(node_id, nonce_3, past_timestamp_us, signature_3, payload_data)
        
        print(f"    -> Interceptor Evaluation: {flag_3}")
        print(f"    -> Temporal Bounding Check Latency: {lat_3:.2f} µs")
        
        if is_valid_3 or flag_3 != "EXPIRED_WINDOW":
            print("[!] FATAL: Interceptor allowed an expired payload to pass!")
            sys.exit(1)
            
        
        # --- TEST 4: Unauthorized Cryptographic Spoofing ---
        print("\n[*] STAGE 4: Synthesizing Unauthorized Key Forgery...")
        nonce_4 = 1002
        current_us = time.time() * 1000000.0
        
        # An attacker without the true private key generates their own
        rogue_private_key = ed25519.Ed25519PrivateKey.generate()
        message_block_4 = struct.pack('<IQQ', node_id, nonce_4, int(current_us)) + payload_data
        rogue_signature = rogue_private_key.sign(message_block_4)
        
        is_valid_4, flag_4, lat_4 = interceptor.intercept_payload(node_id, nonce_4, current_us, rogue_signature, payload_data)
        
        print(f"    -> Interceptor Evaluation: {flag_4}")
        print(f"    -> Asymmetric Forgery Check Latency: {lat_4:.2f} µs")
        
        if is_valid_4 or flag_4 != "INVALID_SIGNATURE":
            print("[!] FATAL: Interceptor accepted a forged Ed25519 signature!")
            sys.exit(1)
        
        
        # --- LATENCY VERIFICATION ---
        max_lat = max([lat_1, lat_2, lat_3, lat_4])
        print(f"\n[*] Maximum Security Bounding Latency: {max_lat:.2f} µs")
        
        metrics = {
            "legitimate_handshake_us": lat_1,
            "replay_intercept_us": lat_2,
            "temporal_intercept_us": lat_3,
            "forgery_intercept_us": lat_4,
            "max_ceiling_us": max_lat,
            "status": "PASS" if max_lat < 150.0 else "FAIL"
        }
        
        print("\n[*] Committing Security Handshake Performance to WORM ledger...")
        self._commit_to_worm_ledger(metrics)
        
        if max_lat < 150.0:
            print("[+] SYSTEM ARCHITECTURE VERIFIED: Ed25519 Security Hooks operate securely beneath 150µs!")
            sys.exit(0)
        else:
            print("[!] SYSTEM CRITICAL: Handshake layer breached execution time envelope.")
            sys.exit(1)


if __name__ == "__main__":
    verifier = HandshakeReplayVerifier()
    verifier.run_validation()
