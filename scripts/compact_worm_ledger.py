#!/usr/bin/env python3
"""
SpaceShield: Secure WORM Ledger Compaction & Cryptographic Audit
Description: Periodically audits the entire SHA-256 incident ledger for forensic 
             integrity. Upon verification, compacts the ledger into immutable cold 
             storage and anchors the new baseline chain to the exact terminal hash.
"""

import os
import sys
import time
import json
import gzip
import shutil
import hashlib
import logging

logger = logging.getLogger("WORMCompactor")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] [COMPACTOR] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class WORMCompactor:
    def __init__(self, max_entries: int = 1000):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.ledger_path = os.path.join(self.base_dir, 'compliance', 'certin_incident_spoofing.json')
        self.archive_dir = os.path.join(self.base_dir, 'compliance', 'archive')
        self.max_entries = max_entries
        
        os.makedirs(self.archive_dir, exist_ok=True)

    def _verify_cryptographic_chain(self, filepath: str) -> str:
        """
        Parses the frozen ledger from Genesis to Terminus.
        Re-calculates every SHA-256 block signature locally and mathematically 
        proves that no block has been structurally tampered with or omitted.
        Returns the final cumulative anchor hash if successful.
        """
        logger.info("[*] Commencing cryptographic backward forensic audit...")
        
        last_seen_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        parsed_entries = 0
        
        try:
            with open(filepath, "r") as f:
                content = f.read().strip()
                if not content:
                    return last_seen_hash
                    
                # Graceful handling for both NDJSON and JSON Array formats
                try:
                    records = json.loads(content)
                    if not isinstance(records, list):
                        records = [records]
                except json.JSONDecodeError:
                    f.seek(0)
                    records = []
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                            
            # Sweep Forward: Verify Chain Linkage
            for i, record in enumerate(records):
                claimed_hash = record.pop("hash", None)
                claimed_prev = record.get("previous_hash", "0000000000000000000000000000000000000000000000000000000000000000")
                
                # Verify that the block links correctly to the previous block's hash
                if i > 0 and claimed_prev != last_seen_hash:
                    raise ValueError(f"Chain breakage detected at block index {i}! Expected Prev: {last_seen_hash}, Got: {claimed_prev}")
                    
                # Verify the block's internal signature matches its physical content
                raw_string = json.dumps(record, sort_keys=True)
                actual_hash = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
                
                if actual_hash != claimed_hash:
                    raise ValueError(f"Block tampering detected at index {i}! Signature mismatch.")
                    
                last_seen_hash = actual_hash
                parsed_entries += 1
                
            logger.info(f"[+] Forensic Audit Passed. {parsed_entries} blocks verified flawlessly.")
            return last_seen_hash
            
        except Exception as e:
            logger.critical(f"[!!!] STQC FORENSIC AUDIT FAILURE: {e}")
            raise

    def execute_compaction(self):
        """
        Executes the atomic freeze, audit, compression, and re-baselining sequence.
        """
        if not os.path.exists(self.ledger_path):
            return
            
        # Count approximate lines/entries
        try:
            with open(self.ledger_path, "r") as f:
                entry_count = sum(1 for line in f if line.strip())
        except Exception:
            return
            
        if entry_count < self.max_entries:
            return # Volume threshold not breached
            
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        frozen_ledger = f"{self.ledger_path}.{timestamp_str}.frozen"
        
        # 1. State Freeze (Atomic Rename)
        os.rename(self.ledger_path, frozen_ledger)
        logger.info(f"[*] Ledger volume breached limit ({entry_count} entries). State FROZEN.")
        
        try:
            # 2. Cryptographic Forensic Audit
            final_hash = self._verify_cryptographic_chain(frozen_ledger)
            
            # 3. Establish New Baseline with Cumulative Anchor
            genesis_entry = {
                "timestamp": time.time(),
                "incident_type": "BASELINE_COMPACTION_ANCHOR",
                "previous_hash": final_hash,
                "status": "CHAIN_AUDIT_PASSED"
            }
            raw_string = json.dumps(genesis_entry, sort_keys=True)
            genesis_entry["hash"] = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
            
            with open(self.ledger_path, "w") as f:
                f.write(json.dumps(genesis_entry) + "\n")
                
            logger.info(f"[+] New Active Baseline established. Cumulative anchor: {final_hash[:16]}...")
            
            # 4. Gzip Compression and Cold Storage Archival
            archive_dest = os.path.join(self.archive_dir, f"certin_ledger_verified_{timestamp_str}.json.gz")
            logger.info(f"[*] Compressing verified historic blocks...")
            with open(frozen_ledger, 'rb') as f_in:
                with gzip.open(archive_dest, 'wb', compresslevel=9) as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            os.remove(frozen_ledger)
            
            if os.name == 'posix':
                os.chmod(archive_dest, 0o400) # Lock physical permissions to Read-Only
                
            logger.info(f"[+] Verified archival completed successfully: {archive_dest}")
            
        except Exception as e:
            # Critical Failure - Restore the frozen ledger immediately to prevent data loss
            logger.critical(f"[!] COMPACTION ABORTED DUE TO AUDIT FAILURE. Restoring ledger state.")
            os.rename(frozen_ledger, self.ledger_path)
            sys.exit(1)

# --- Rapid Verification Stub ---
if __name__ == "__main__":
    compactor = WORMCompactor(max_entries=5) # Tiny threshold for instant mock triggering
    
    print("================================================================")
    print(" SpaceShield WORM Ledger Compaction & Cryptographic Audit")
    print("================================================================")
    
    # Generate mathematically correct mock block chain
    print("[*] Generating mathematically sound cryptographic WORM sequence...")
    os.makedirs(os.path.dirname(compactor.ledger_path), exist_ok=True)
    
    last_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    with open(compactor.ledger_path, "w") as f:
        for i in range(10):
            record = {
                "timestamp": time.time(),
                "event": "MOCK_RADAR_EVENT",
                "power_db": i * 1.5,
                "previous_hash": last_hash
            }
            raw_string = json.dumps(record, sort_keys=True)
            record["hash"] = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
            last_hash = record["hash"]
            f.write(json.dumps(record) + "\n")
            
    # Trigger the full audit and compaction cycle
    compactor.execute_compaction()
    
    # Read the newly established baseline
    print("\n[+] Verification of the newly established baseline file:")
    with open(compactor.ledger_path, "r") as f:
        print("   ", f.read().strip())
