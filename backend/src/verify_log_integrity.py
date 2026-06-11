#!/usr/bin/env python3
"""
SpaceShield: CERT-In log integrity verification utility.
Author: Antigravity AI
Version: 1.0.0

This utility validates the cryptographic hash chain of SpaceShield's 180-day
security log file to ensure compliance with the February 2026 CERT-In
Space Cyber Security Framework.
"""

import os
import sys
import json
import hashlib
import argparse

def verify_log_integrity(log_file_path):
    """
    Parses a SpaceShield WORM log file line by line, re-calculating the SHA-256
    hash for each entry, and validating the blockchain-like hash chain.
    """
    if not os.path.exists(log_file_path):
        print(f"[-] Error: Log file not found at: {log_file_path}")
        return False

    print("=" * 80)
    print("      SPACESHIELD CERT-IN SPACE CYBER SECURITY INTEGRITY AUDIT REPORT      ")
    print("=" * 80)
    print(f"Target Log File: {log_file_path}")
    print(f"Audit Timestamp: {hashlib.sha256().hexdigest()[:8]} (Verification Session ID)")
    print("-" * 80)

    prev_computed_hash = "0" * 64
    lines_processed = 0
    verification_errors = 0

    with open(log_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            lines_processed += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[!] [Line {line_num}] JSON Parse Error: {e}")
                verification_errors += 1
                continue

            # Extract fields for verification
            recorded_hash = entry.get("hash", "")
            recorded_prev_hash = entry.get("prev_hash", "")
            timestamp = entry.get("timestamp", "UNKNOWN")
            scenario = entry.get("monitoring_scenario", "UNKNOWN")
            verdict = entry.get("incident_details", {}).get("threat_category", "UNKNOWN")

            # Prepare entry for hash re-computation by removing the hash itself
            entry_copy = entry.copy()
            if "hash" in entry_copy:
                del entry_copy["hash"]

            # Re-serialize exactly as it was done during generation (sort_keys=True)
            serialized = json.dumps(entry_copy, sort_keys=True)
            computed_hash = hashlib.sha256(serialized.encode('utf-8')).hexdigest()

            # Perform validation checks
            hash_mismatch = (computed_hash != recorded_hash)
            chain_broken = (recorded_prev_hash != prev_computed_hash)

            status_str = "OK"
            if hash_mismatch or chain_broken:
                status_str = "FAIL"
                verification_errors += 1

            print(f"[{status_str}] Entry #{lines_processed} | Time: {timestamp} | Scenario: {scenario:<8} | Verdict: {verdict:<8}")

            if hash_mismatch:
                print(f"    └─ [ERROR] Current Hash Mismatch!")
                print(f"       Recorded: {recorded_hash}")
                print(f"       Computed: {computed_hash}")
            
            if chain_broken:
                print(f"    └─ [ERROR] Hash Chain Broken (prev_hash mismatch)!")
                print(f"       Recorded prev_hash: {recorded_prev_hash}")
                print(f"       Expected prev_hash: {prev_computed_hash}")

            # Keep track of the computed hash for the next line's prev_hash check
            # We use computed_hash (or recorded_hash depending on definition, but computed_hash tracks the correct chain)
            prev_computed_hash = computed_hash

    print("-" * 80)
    print("AUDIT SUMMARY:")
    print(f"  • Total Log Lines Scanned: {lines_processed}")
    print(f"  • Integrity Violations:   {verification_errors}")
    
    if verification_errors == 0 and lines_processed > 0:
        print("\n[+] RESULT: SECURE & COMPLIANT")
        print("    The cryptographic chain is perfectly intact. No tampering detected.")
        print("    Logs satisfy the CERT-In Space Cyber Security Framework requirements.")
        print("=" * 80)
        return True
    else:
        print("\n[!] ALERT: INTEGRITY COMPROMISED")
        print("    One or more log lines have been modified, deleted, or inserted out of sequence.")
        print("    Action required: Initiate ground station incident response protocol immediately.")
        print("=" * 80)
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify SpaceShield log file integrity and hash chains.")
    parser.add_argument('--log-file', type=str, default="data/spaceshield_180day_security.log",
                        help="Path to the rolling log file (default: data/spaceshield_180day_security.log)")
    args = parser.parse_args()
    
    # Resolve relative path from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    log_path = args.log_file
    if not os.path.isabs(log_path):
        log_path = os.path.join(project_root, log_path)

    success = verify_log_integrity(log_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
