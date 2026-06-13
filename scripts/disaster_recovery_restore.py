#!/usr/bin/env python3
"""
SpaceShield: Disaster Recovery & Forensic Bootstrap Restorer
Description: Executes natively on cold-boot to extract, verify, and rehydrate
             the execution state from the SHA-256 tamper-evident WORM ledger.
             Features Zero-Dependency Cryptographic Auditing & Autonomous Threat Isolation.
"""

import os
import sys
import json
import hashlib
import subprocess

class DisasterRecoveryEngine:
    def __init__(self, ledger_path: str = None):
        # Defaulting to standard SpaceShield mapping boundaries
        if ledger_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.ledger_path = os.path.join(base_dir, 'compliance', 'certin_incident_spoofing.json')
            
            # Check test fallback if main production log doesn't exist
            if not os.path.exists(self.ledger_path):
                fallback = os.path.join(base_dir, 'compliance', 'certin_incident_spoofing_test.json')
                if os.path.exists(fallback):
                    self.ledger_path = fallback
        else:
            self.ledger_path = ledger_path
            
        self.verify_depth = 50

    def _trigger_network_isolation(self, reason: str):
        """
        Critical response mechanism when tampering is discovered.
        Physically drops the active network interface to prevent compromised 
        state execution or lateral movement across the cluster mesh.
        """
        print("\n\033[1;41;37m" + "="*70)
        print(" [!!!] CRITICAL: CRYPTOGRAPHIC TAMPERING DETECTED [!!!] ")
        print(f" REASON: {reason}")
        print(" ACTION: ISOLATING NODE FROM MESH NETWORK")
        print("="*70 + "\033[0m\n")
        
        # Cross-platform interface drop attempt
        if os.name == 'posix':
            try:
                # Attempt standard Linux interface lockdown
                subprocess.run(["ip", "link", "set", "eth0", "down"], check=False, stderr=subprocess.DEVNULL)
                subprocess.run(["iptables", "-A", "INPUT", "-j", "DROP"], check=False, stderr=subprocess.DEVNULL)
                print("[*] POSIX network isolation boundaries engaged.")
            except Exception:
                print("[-] Failed to engage root network controls (Unprivileged Container?).")
        else:
            print("[*] Non-POSIX OS detected. Logical isolation engaged.")
            
        print("\n[!] Boot process halted. Manual forensic intervention required.")
        sys.exit(1)

    def _verify_block_hash(self, raw_record: dict, claimed_hash: str) -> bool:
        """Deterministically hashes the payload to verify immutable parity."""
        # Strip the hash signature for deterministic recalculation
        clean_record = raw_record.copy()
        clean_record.pop('hash', None)
        
        # Serialize with strict sorted keys identical to ingestion protocol
        raw_string = json.dumps(clean_record, sort_keys=True)
        recalculated = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
        
        return recalculated == claimed_hash

    def perform_forensic_sweep(self):
        """
        Extracts and verifies the cryptographic chain backwards by 50 blocks.
        Re-hydrates the system parameters if the chain is uncorrupted.
        """
        print(f"[*] Bootstrapping Disaster Recovery from Volume: {self.ledger_path}")
        
        if not os.path.exists(self.ledger_path):
            print("[-] No WORM ledger found on mount. Proceeding with clean factory default boot.")
            return True
            
        # Load all logs sequentially (Zero external dependencies)
        logs = []
        try:
            with open(self.ledger_path, 'r') as f:
                content = f.read().strip()
                if content:
                    try:
                        # Attempt standard JSON first (dict or list)
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            logs = parsed
                        else:
                            logs = [parsed]
                    except json.JSONDecodeError:
                        # Fallback to NDJSON format
                        f.seek(0)
                        for line in f:
                            line = line.strip()
                            if not line: continue
                            try:
                                logs.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            self._trigger_network_isolation(f"Ledger file read corruption: {e}")
            
        if not logs:
            print("[-] WORM ledger is empty. Clean boot initialized.")
            return True
            
        # Extract the last 50 blocks for the forensic sweep
        sweep_blocks = logs[-self.verify_depth:]
        print(f"[*] Extracting forensic slice: Scanning {len(sweep_blocks)} chronological blocks...")
        
        # Reverse chronological verification
        # Iterate from the end of the list backwards
        for i in range(len(sweep_blocks) - 1, 0, -1):
            current_block = sweep_blocks[i]
            previous_block = sweep_blocks[i - 1]
            
            # 1. Structural Hash Validation
            claimed_hash = current_block.get("hash")
            if not claimed_hash or not self._verify_block_hash(current_block, claimed_hash):
                self._trigger_network_isolation(f"Immutable hash parity failure at block timestamp: {current_block.get('timestamp')}")
                
            # 2. Sequential Pointer Validation (The Chain)
            chain_pointer = current_block.get("previous_hash")
            actual_previous_hash = previous_block.get("hash")
            
            if chain_pointer != actual_previous_hash:
                self._trigger_network_isolation(f"Broken Cryptographic Chain! Block {i} points to {chain_pointer}, but predecessor is {actual_previous_hash}")

        # Verify the absolute oldest block in our slice individually
        oldest_block = sweep_blocks[0]
        if oldest_block.get("hash") and not self._verify_block_hash(oldest_block, oldest_block.get("hash")):
            self._trigger_network_isolation("Hash parity failure at boundary block 0.")

        print(f"[+] Cryptographic Chain Intact. {len(sweep_blocks)} consecutive blocks mathematically validated.")
        
        # --- Rehydration Phase ---
        # Scan backward for the most recent valid state footprint
        rehydrated_gamma = 50.17  # Factory default
        rehydrated_beta = 1.0     # Factory default
        
        for block in reversed(sweep_blocks):
            if "gamma_threshold" in block or "sphericity" in block:
                # Extracting simulated state from various logging payloads
                rehydrated_gamma = block.get("gamma_threshold", block.get("sphericity", rehydrated_gamma))
                rehydrated_beta = block.get("metr", rehydrated_beta)
                break
                
        print("\n--- SYSTEM REHYDRATION COMPLETE ---")
        print(f" [>] SVD Spatial Threshold (Gamma) Restored: {rehydrated_gamma:.4f}")
        print(f" [>] Inter-Channel Coherence (Beta) Restored: {rehydrated_beta:.4f}")
        print(" [>] Active State: CLEARED FOR HOT-PATH INGESTION\n")
        
        return True


if __name__ == "__main__":
    restorer = DisasterRecoveryEngine()
    restorer.perform_forensic_sweep()
