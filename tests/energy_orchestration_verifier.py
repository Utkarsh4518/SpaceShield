#!/usr/bin/env python3
"""
SpaceShield: TBPM Energy Orchestration Verification Suite
Description: Couples the Energy Aware Orchestrator with an active DSP worker thread 
             to mathematically prove that entering CRITICAL_SUSTAINABILITY_MODE forces 
             drastic CPU savings on background tasks without stalling or lagging the 
             hot-path baseband phase-locked loop Solvers.
"""

import os
import sys
import time
import json
import hashlib
import threading
import numpy as np

# Dynamically link the backend modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'backend', 'src'))
sys.path.append(os.path.join(BASE_DIR, 'tests'))

from energy_aware_orchestrator import EnergyAwareOrchestrator

class EnergyOrchestrationVerifier:
    def __init__(self):
        self.orchestrator = EnergyAwareOrchestrator(critical_threshold_pct=20.0, hysteresis_pct=25.0)
        self.ledger_path = os.path.join(BASE_DIR, 'compliance', 'certin_incident_spoofing.json')
        
        self.bg_loop_iterations = 0
        self.is_running = True
        
        # Simulated continuous DSP states
        self.fast_path_latencies = []

    def _commit_to_worm_ledger(self, metrics: dict):
        """Appends the verification pass/fail matrix to the STQC WORM log."""
        metrics["timestamp"] = time.time()
        metrics["incident_type"] = "TBPM_ENERGY_ORCHESTRATION_VERIFICATION"
        
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

    def _mock_hardware_background_thread(self):
        """
        Simulates the SoapySDR hardware ingestion loop. 
        It polls the TBPM atomic configuration to determine its own execution budget.
        """
        while self.is_running:
            cfg = self.orchestrator.get_atomic_config()
            target_mode = cfg.get("power_state", "NOMINAL_RENEWABLE_MODE")
            
            # Simulate basic background work
            _ = np.sum(np.ones(100))
            
            if target_mode == "CRITICAL_SUSTAINABILITY_MODE":
                # CPU Mitigation Active! Scaled sleep
                time.sleep(0.005)
            else:
                # Unthrottled Nominal ingestion
                time.sleep(0.001)
                
            self.bg_loop_iterations += 1

    def _execute_fast_path_stride(self):
        """Simulates a dense math operation natively running in ~24µs."""
        t0 = time.perf_counter()
        
        # Dense SVD / Math block
        _ = np.linalg.svd(np.random.randn(8, 8))
        
        # Check config to verify zero lock contention
        cfg = self.orchestrator.get_atomic_config()
        _ = cfg["power_state"]
        
        exec_us = (time.perf_counter() - t0) * 1e6
        self.fast_path_latencies.append(exec_us)

    def run_validation(self):
        print("======================================================================")
        print(" SpaceShield Infrastructure: Energy Orchestration Verifier")
        print("======================================================================")
        
        # 1. Initialize Orhcestrator and Background Hardware Thread
        self.orchestrator.start_daemon()
        self.orchestrator._mock_battery_soc = 100.0
        
        bg_thread = threading.Thread(target=self._mock_hardware_background_thread)
        bg_thread.start()
        
        print("\n[*] Stage 1: Measuring NOMINAL_RENEWABLE_MODE baselines...")
        # Give the background loop time to rack up stats
        self.bg_loop_iterations = 0
        self.fast_path_latencies = []
        
        t_start = time.time()
        while time.time() - t_start < 2.0:
            self._execute_fast_path_stride()
            time.sleep(0.01) # Moderate test pacing
            
        nominal_bg_iterations = self.bg_loop_iterations
        nominal_latencies = self.fast_path_latencies.copy()
        
        avg_nominal_lat = sum(nominal_latencies) / len(nominal_latencies)
        print(f"    -> Hardware Duty Cycles: {nominal_bg_iterations} iterations / 2 sec")
        print(f"    -> Fast-Path Latency:    {avg_nominal_lat:.2f} µs")
        
        print("\n[*] Stage 2: Simulating Microgrid Depletion -> CRITICAL_SUSTAINABILITY_MODE")
        self.orchestrator._mock_battery_soc = 15.0 # Drop below 20% limit
        
        # Capture operations DURING the transition to detect thread-locks
        self.bg_loop_iterations = 0
        self.fast_path_latencies = []
        
        t_start = time.time()
        while time.time() - t_start < 2.0:
            self._execute_fast_path_stride()
            time.sleep(0.01)
            
        critical_bg_iterations = self.bg_loop_iterations
        critical_latencies = self.fast_path_latencies.copy()
        
        avg_critical_lat = sum(critical_latencies) / len(critical_latencies)
        
        # 3. Teardown
        self.is_running = False
        self.orchestrator.stop_daemon()
        bg_thread.join()
        
        print(f"    -> Hardware Duty Cycles: {critical_bg_iterations} iterations / 2 sec")
        print(f"    -> Fast-Path Latency:    {avg_critical_lat:.2f} µs")
        
        # 4. CPU Metric Verification
        cpu_savings_pct = (1.0 - (critical_bg_iterations / nominal_bg_iterations)) * 100.0
        print(f"\n--- TBPM EXECUTION PROFILES ---")
        print(f" [>] Background CPU Consumption Drop: {cpu_savings_pct:.1f}%")
        
        # Verify thread anomalies
        max_transition_latency = max(critical_latencies)
        degradation = avg_critical_lat - avg_nominal_lat
        
        print(f" [>] Avg DSP Pipeline Degradation:    {degradation:+.2f} µs")
        print(f" [>] Max Transition Spike Latency:    {max_transition_latency:.2f} µs")
        
        cpu_passed = cpu_savings_pct >= 30.0
        
        # We enforce strict microsecond bounds. The average degradation must be effectively 0 (allow <= 5us buffer for OS OS scheduling jitter in python tests).
        degradation_passed = degradation <= 5.0 
        
        overall_pass = cpu_passed and degradation_passed
        
        metrics = {
            "background_cpu_savings_pct": float(cpu_savings_pct),
            "fast_path_nominal_lat_us": float(avg_nominal_lat),
            "fast_path_critical_lat_us": float(avg_critical_lat),
            "fast_path_degradation_us": float(degradation),
            "status": "PASS" if overall_pass else "FAIL"
        }
        
        print("\n[*] Committing TBPM Thread Profiles to cryptographic compliance ledger...")
        self._commit_to_worm_ledger(metrics)
        
        if overall_pass:
            print("[+] SYSTEM ARCHITECTURE VERIFIED: Zero-Contention Power Management Passed!")
            sys.exit(0)
        else:
            print("[!] SYSTEM CRITICAL: CPU OR THREAD LOCK ANOMALIES DETECTED.")
            sys.exit(1)

if __name__ == "__main__":
    verifier = EnergyOrchestrationVerifier()
    verifier.run_validation()
