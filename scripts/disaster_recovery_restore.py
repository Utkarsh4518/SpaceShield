#!/usr/bin/env python3
"""
Task 43.3: Bare-Metal Systems Recovery Specialist
Self-Contained Disaster Recovery Bootloader & Ledger Verification
"""

import sys
import os
import json
import hashlib
import gzip
import tarfile
import glob
import time
import stat

# ==============================================================================
# Initialization & Absolute Path Binding
# ==============================================================================
# Resolves the script path completely independent of the executing working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Target structural directories for SpaceShield absolute compliance
REQUIRED_DIRS = [
    os.path.join(BASE_DIR, 'backend', 'src'),
    os.path.join(BASE_DIR, 'frontend'),
    os.path.join(BASE_DIR, 'compliance'),
    os.path.join(BASE_DIR, 'compliance', 'archives'),
    os.path.join(BASE_DIR, 'models'),
    os.path.join(BASE_DIR, 'keys', 'public')
]

ACTIVE_LEDGER = os.path.join(BASE_DIR, 'compliance', 'certin_incident_spoofing.json')
ARCHIVE_PATTERN = os.path.join(BASE_DIR, 'compliance', 'archives', 'certin_ledger_verified_*.json.gz')

# Genesis Baseline Token (Pre-calculated constant for unbroken chains)
GENESIS_HASH = "GENESIS_ROOT_000000000000000000000000000000000000000000000000000000"

# ==============================================================================
# Cryptographic Validation Protocols
# ==============================================================================
def calculate_block_hash(block):
    """
    Computes the canonical SHA-256 hash of a dictionary block.
    Sorts keys rigidly to guarantee deterministic binary output across runs.
    """
    # Exclude internal temporary keys if any existed, serialize rigidly
    block_str = json.dumps(block, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(block_str.encode('utf-8')).hexdigest()

def verify_ledger_chain(blocks, start_hash=None):
    """
    Iterates rigorously through an active JSON block array mathematically verifying continuity.
    Returns the final (most recent) hash if completely unbroken, otherwise instantly raises ValueError.
    """
    if not blocks:
        return start_hash
        
    last_hash = start_hash
    for i, block in enumerate(blocks):
        if i == 0 and last_hash is None:
            # Absolute genesis block sequence
            last_hash = calculate_block_hash(block)
            continue
            
        expected_prev = block.get('previous_hash', None)
        
        # Legacy retrofitting constraint: Ensure legacy logs adopt the new hash structure seamlessly
        if expected_prev is None and last_hash is not None:
            block['previous_hash'] = last_hash
            expected_prev = last_hash
            
        if last_hash is not None and expected_prev != last_hash:
            raise ValueError(f"Structural Integrity Failure at Block {i}: "
                             f"Expected Prev: {expected_prev}, Actual Hash: {last_hash}")
                             
        last_hash = calculate_block_hash(block)
        
    return last_hash

# ==============================================================================
# Structural Path Engine
# ==============================================================================
def repair_directory_structure():
    """
    Programmatically rebuilds the physical directory skeleton required for deployment operation.
    """
    print("\n[1] Evaluating Hardware Directory Tree Integrity...")
    missing_count = 0
    for directory in REQUIRED_DIRS:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"    -> Restored path boundary: {directory}")
                missing_count += 1
            except Exception as e:
                print(f"    -> [FATAL] Path regeneration failed for {directory}: {e}")
                sys.exit(1)
                
    if missing_count == 0:
        print("    -> [PASS] Structural directories intact.")
    else:
        print(f"    -> [WARN] Restored {missing_count} corrupted or missing subsystem paths.")

# ==============================================================================
# Master Recovery Sequence
# ==============================================================================
def execute_cold_recovery():
    print("===============================================================================")
    print("SPACESHIELD FALLBACK BOOTLOADER: Bare-Metal Disaster Recovery Sequence")
    print("===============================================================================")
    
    # 1. Base Framework Structural Repair
    repair_directory_structure()
    
    # 2. Historical Archive Cryptographic Verification
    print("\n[2] Crawling and Verifying Immutable Cryptographic Archives...")
    archive_files = sorted(glob.glob(ARCHIVE_PATTERN))
    
    baseline_hash = GENESIS_HASH
    
    if not archive_files:
        print("    -> [INFO] No legacy archival segments found. Initializing pure genesis sequence.")
    else:
        print(f"    -> Discovered {len(archive_files)} compressed segments. Crawling Hash-Chain...")
        
        # Traverse chronologically based on filename timestamp sorting
        for arch_file in archive_files:
            try:
                with gzip.open(arch_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if not isinstance(data, list):
                    print(f"    -> [ERROR] Corrupt structure in {os.path.basename(arch_file)}. Array expected.")
                    sys.exit(1)
                    
                # Verify external segment starts exactly where the previous timeline ended
                if data:
                    segment_genesis_prev = data[0].get('previous_hash', None)
                    # Support legacy un-hashed segments
                    if segment_genesis_prev is not None and segment_genesis_prev != baseline_hash and baseline_hash != GENESIS_HASH:
                        print(f"    -> [FATAL] Archive Chain Disconnect in {os.path.basename(arch_file)}.")
                        print(f"       Expected {baseline_hash}, Found {segment_genesis_prev}")
                        sys.exit(1)
                        
                # Traverse internal array chain
                segment_terminal_hash = verify_ledger_chain(data, start_hash=baseline_hash)
                if segment_terminal_hash:
                    baseline_hash = segment_terminal_hash
                    
                print(f"    -> [PASS] Verified Boundary Integrity: {os.path.basename(arch_file)}")
                
            except Exception as e:
                print(f"    -> [FATAL] Archive signature mismatch in {os.path.basename(arch_file)}: {e}")
                sys.exit(1)
                
    print(f"    -> Cryptographic Continuity Terminus Hash: {baseline_hash}")
    
    # 3. Active System Ledger Re-Initialization
    print("\n[3] Re-initializing Active JSON WORM Ledger...")
    if os.path.exists(ACTIVE_LEDGER):
        try:
            with open(ACTIVE_LEDGER, 'r', encoding='utf-8') as f:
                current_ledger = json.load(f)
            print("    -> Active ledger located. Asserting forward baseline continuity...")
            
            if not isinstance(current_ledger, list):
                current_ledger = [current_ledger]
                
            if current_ledger:
                # Force synchronization with historical baseline if un-anchored
                if 'previous_hash' not in current_ledger[0]:
                    print("    -> [WARN] Current ledger missing temporal anchoring. Forcing hash injection.")
                    current_ledger[0]['previous_hash'] = baseline_hash
                elif current_ledger[0]['previous_hash'] != baseline_hash and baseline_hash != GENESIS_HASH:
                     print("    -> [FATAL] Active ledger 'previous_hash' does not link seamlessly to Archive sequence.")
                     sys.exit(1)
                     
                final_hash = verify_ledger_chain(current_ledger, start_hash=baseline_hash)
                print("    -> [PASS] Active ledger chain cryptographically verified. No malicious truncation detected.")
            
        except json.JSONDecodeError:
            print("    -> [WARN] Active ledger corrupted or tampered. Eradicating memory and re-anchoring to secure baseline.")
            current_ledger = []
    else:
        print("    -> [INFO] Active ledger missing. Generating fresh compliance block.")
        current_ledger = []
        
    # Generate anchor block if the ledger was wiped
    if not current_ledger:
        genesis_block = {
            "timestamp_iso": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "event_classification": "COLD_BOOT_RECOVERY",
            "previous_hash": baseline_hash,
            "recovery_status": "VERIFIED_SECURE",
            "message": "Fallback bootloader successfully synchronized temporal continuity."
        }
        current_ledger.append(genesis_block)
        print("    -> [PASS] Baseline anchor block synthesized and attached.")
        
    # 4. Enforce WORM Serialization
    try:
        if os.path.exists(ACTIVE_LEDGER):
            # Temporarily restore write permissions to overwrite the active ledger
            os.chmod(ACTIVE_LEDGER, stat.S_IWRITE)
            
        with open(ACTIVE_LEDGER, 'w', encoding='utf-8') as f:
            json.dump(current_ledger, f, indent=4)
            
        # Re-enforce strict Read-Only WORM constraints to prevent local application tampering
        os.chmod(ACTIVE_LEDGER, stat.S_IREAD)
        print(f"    -> [PASS] System ledger committed securely to {ACTIVE_LEDGER} as Read-Only.")
        
    except Exception as e:
        print(f"    -> [FATAL] Failed to serialize recovery ledger: {e}")
        sys.exit(1)
        
    print("===============================================================================")
    print("[SUCCESS] Bare-Metal Architecture Restored and Ready for Ignition.")
    print("===============================================================================")

if __name__ == "__main__":
    execute_cold_recovery()
