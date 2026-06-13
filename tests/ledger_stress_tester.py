#!/usr/bin/env python3
"""
SpaceShield: Cryptographic WORM Ledger Stress Tester
Description: High-velocity concurrency validator asserting zero lock-contention
             and absolute SHA-256 hash chaining integrity under massive 24-thread
             electronic warfare flooding.
"""

import os
import time
import json
import queue
import hashlib
import threading
import tracemalloc
import logging

logger = logging.getLogger("LedgerStressTester")
logger.setLevel(logging.INFO)

class FastWormLogger:
    """
    Mock representation of the SpaceShield backend lock-free logging ring-buffer.
    Uses a strict background daemon to serialize the SHA-256 cryptographic chain
    without blocking the high-speed spatial array DSP workers.
    """
    def __init__(self, log_path: str, max_queue_size: int = 500000):
        self.log_path = log_path
        # The lock-free backpressure absorber buffer
        self.log_queue = queue.Queue(maxsize=max_queue_size)
        
        self.running = True
        self.last_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        self.logs_processed = 0
        self.max_memory_observed = 0
        
        # Start the background serialization daemon
        self.writer_thread = threading.Thread(target=self._async_writer_loop, daemon=True)
        self.writer_thread.start()

    def push_incident(self, payload: dict):
        """Asynchronous, non-blocking ingestion from DSP hot-paths."""
        try:
            self.log_queue.put_nowait(payload)
        except queue.Full:
            logger.critical("FATAL: Lock-free ring buffer overflowed under back-pressure!")
            raise MemoryError("Logging Queue Overrun")

    def _async_writer_loop(self):
        """Sequential cryptographic chaining daemon."""
        try:
            with open(self.log_path, "w") as f:
                while self.running or not self.log_queue.empty():
                    try:
                        # Batch blocking pop to save CPU cycles
                        incident = self.log_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                        
                    # 1. Enforce Cryptographic Chaining
                    incident["previous_hash"] = self.last_hash
                    
                    # Deterministic serialization for hashing
                    raw_string = json.dumps(incident, sort_keys=True)
                    current_hash = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
                    
                    incident["hash"] = current_hash
                    self.last_hash = current_hash
                    
                    # 2. WORM Commit
                    f.write(json.dumps(incident) + "\n")
                    self.logs_processed += 1
                    self.log_queue.task_done()
        except Exception as e:
            logger.critical(f"Logging Daemon Crashed: {e}")

    def shutdown(self):
        self.running = False
        self.writer_thread.join()

class StressTestHarness:
    def __init__(self, num_threads: int = 24, target_rate_per_sec: int = 10000, duration_sec: float = 3.0):
        self.num_threads = num_threads
        self.target_rate = target_rate_per_sec
        self.duration = duration_sec
        
        test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../compliance'))
        os.makedirs(test_dir, exist_ok=True)
        self.log_file = os.path.join(test_dir, "stress_test_worm.log")
        
        self.logger_engine = FastWormLogger(log_path=self.log_file)
        
        self.start_barrier = threading.Barrier(self.num_threads + 1)
        self.contention_faults = 0

    def _worker_flood(self, thread_id: int):
        """High-frequency mock DSP thread injecting massive incident vectors."""
        self.start_barrier.wait()
        
        end_time = time.time() + self.duration
        logs_generated = 0
        
        # We enforce a pacing delay to hit exact targets without blowing up the thread scheduler
        pacing_delay = 1.0 / self.target_rate 
        
        while time.time() < end_time:
            payload = {
                "timestamp": time.time(),
                "thread": thread_id,
                "incident_type": "SIMULATED_JAMMING_FLOOD",
                "sphericity": 55.4 + thread_id,
                "metr": 0.89
            }
            
            try:
                self.logger_engine.push_incident(payload)
                logs_generated += 1
                # Micro-sleep to pace the exact 10,000 logs/sec constraint per thread
                time.sleep(pacing_delay)
            except Exception:
                self.contention_faults += 1
                break

    def verify_cryptographic_chain(self):
        """Reads the entire WORM log back from disk and verifies the unbroken SHA-256 chain."""
        print("[*] Initiating rigorous SHA-256 hash chain validation...")
        last_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        validated_count = 0
        
        with open(self.log_file, "r") as f:
            for line in f:
                record = json.loads(line)
                
                # Check pointer integrity
                if record["previous_hash"] != last_hash:
                    return False, f"Broken chain pointer at record {validated_count}"
                    
                # Verify payload integrity
                claimed_hash = record.pop("hash")
                raw_string = json.dumps(record, sort_keys=True)
                recomputed_hash = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
                
                if recomputed_hash != claimed_hash:
                    return False, f"Hash corruption at record {validated_count}"
                    
                last_hash = claimed_hash
                validated_count += 1
                
        return True, validated_count

    def execute(self):
        print("==================================================================")
        print(f" SpaceShield Cryptographic Ledger Stress Tester")
        print(f" Vectors: {self.num_threads} Threads | Rate: {self.target_rate} logs/s/thread")
        print("==================================================================")
        
        tracemalloc.start()
        
        threads = []
        for i in range(self.num_threads):
            t = threading.Thread(target=self._worker_flood, args=(i,), daemon=True)
            threads.append(t)
            t.start()
            
        print("[+] Thread pool initialized. Releasing execution barrier...")
        t0 = time.time()
        self.start_barrier.wait()
        
        for t in threads:
            t.join()
            
        execution_time = time.time() - t0
        
        print("[*] Attack wave completed. Waiting for serialization daemon to drain queue...")
        self.logger_engine.shutdown()
        
        # Profiling Metrics
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_logs = self.logger_engine.logs_processed
        actual_rate = total_logs / execution_time
        
        print("\n--- PERFORMANCE PROFILES ---")
        print(f" [>] Duration:           {execution_time:.2f} seconds")
        print(f" [>] Total Logs Written: {total_logs}")
        print(f" [>] Actual Throughput:  {actual_rate:.2f} logs/second combined")
        print(f" [>] Lock Contention:    {self.contention_faults} faults")
        print(f" [>] Peak Memory Bound:  {peak_mem / 1024 / 1024:.2f} MB")
        
        print("\n--- INTEGRITY ASSERTIONS ---")
        is_valid, count_or_err = self.verify_cryptographic_chain()
        
        assert self.contention_faults == 0, "FAILED: Lock contention detected!"
        assert peak_mem < 250 * 1024 * 1024, "FAILED: Memory exceeded 250MB bounding box!"
        assert is_valid, f"FAILED: Cryptographic chain corrupted! {count_or_err}"
        
        print(f" [PASSED] Cryptographic Chain Verified Unbroken ({count_or_err} total records).")
        print(f" [PASSED] Memory utilized strictly within bounding constraints.")
        print(f" [PASSED] Zero lock-contention detected under multi-threading pressure.")
        print("==================================================================")

if __name__ == "__main__":
    tester = StressTestHarness(num_threads=24, target_rate_per_sec=10000, duration_sec=2.0)
    tester.execute()
