#!/usr/bin/env python3
"""
SpaceShield: Container Health Verification Utility.
Author: Principal Cloud-Native Embedded Systems Architect
Version: 1.0.0

This utility parses the serialized Ground Station harness status file
to monitor pipeline execution statistics, checking for dropped blocks,
thread stalls, or queue saturation, reporting back to the Docker daemon.
"""

import os
import sys
import json
import time

STATUS_FILE = "/tmp/spaceshield_status.json"

def evaluate_health():
    if not os.path.exists(STATUS_FILE):
        print(f"[-] Health Check Failed: Status file missing at {STATUS_FILE}")
        return False
        
    try:
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)
            
        timestamp = status.get("timestamp", 0.0)
        proc_blocks = status.get("processed_blocks", 0)
        drop_blocks = status.get("dropped_blocks", 0)
        q_size = status.get("queue_size", 0)
        q_max = status.get("queue_max", 1000)
        num_workers = status.get("num_workers", 0)
        
        current_time = time.time()
        age_sec = current_time - timestamp
        
        # 1. Thread loop stall detection (stale status write)
        if age_sec > 10.0:
            print(f"[-] Health Check Failed: Stalled processing loop. Status file is stale by {age_sec:.2f} seconds.")
            return False
            
        # 2. Pipeline drops detection
        if drop_blocks > 0:
            print(f"[-] Health Check Failed: Block loss detected! Dropped Blocks: {drop_blocks}")
            return False
            
        # 3. Queue saturation check (stalling warning above 90% threshold)
        q_saturation_pct = (q_size / q_max) * 100.0 if q_max > 0 else 0.0
        if q_saturation_pct > 90.0:
            print(f"[-] Health Check Failed: Queue saturation threat! Ingestion queue is {q_saturation_pct:.1f}% saturated ({q_size}/{q_max} blocks).")
            return False
            
        # 4. Worker thread count validation
        if num_workers <= 0:
            print("[-] Health Check Failed: Active worker count is 0 or uninitialized.")
            return False
            
        print(f"[+] Health Check Passed: Pipeline running at line-rate. Processed: {proc_blocks} | Drops: {drop_blocks} | Queue size: {q_size} ({q_saturation_pct:.1f}%) | Workers: {num_workers}")
        return True
        
    except Exception as e:
        print(f"[-] Health Check Exception: {e}")
        return False

def main():
    success = evaluate_health()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
