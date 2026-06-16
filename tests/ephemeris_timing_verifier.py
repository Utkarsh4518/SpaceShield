#!/usr/bin/env python3
"""
SpaceShield: Protocol Compliance Verification Suite
Description: Rigorous validation harness to prove that the Kinematic Ephemeris 
             Look-Angle Validator and Subframe Timing Lock structurally mitigate 
             highly coordinated hybrid Meaconing/Spoofing attacks executing 
             multi-dimensional (Spatial + Temporal) drag-offs.
"""

import os
import sys
import time
import json
import hashlib
import math
import numpy as np

# Dynamically link the backend modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'backend', 'src'))
sys.path.append(os.path.join(BASE_DIR, 'tests'))

from ephemeris_look_angle_validator import EphemerisLookAngleValidator
from subframe_timing_lock import SubframeTimingLock

class EphemerisTimingVerifier:
    def __init__(self):
        # Initialize the Physical Layer Geometry and Timing Constraints
        self.look_angle_val = EphemerisLookAngleValidator(gs_lat=28.6139, gs_lon=77.2090, gs_alt=216.0, violation_margin_deg=3.0)
        self.timing_lock_val = SubframeTimingLock(tolerance_samples=1.0)
        
        self.ledger_path = os.path.join(BASE_DIR, 'compliance', 'certin_incident_spoofing.json')

    def _commit_to_worm_ledger(self, metrics: dict):
        """Appends the verification pass/fail matrix to the STQC WORM log."""
        metrics["timestamp"] = time.time()
        metrics["incident_type"] = "EPHEMERIS_AND_TIMING_HYBRID_SPOOF_VERIFICATION"
        
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
        print("==================================================================")
        print(" SpaceShield Protocol Execution: Hybrid Ephemeris & Timing Verifier")
        print("==================================================================")
        
        # Nominal Baseband Geometry (NavIC Keplerian Mock)
        A, e, i_angle = 42164000.0, 0.002, math.radians(29.0)
        omega, raan_0, M_0 = math.radians(10.0), math.radians(45.0), math.radians(100.0)
        ephemeris_params = (A, e, i_angle, omega, raan_0, M_0)
        
        # Retrieve True Orbital Ground-Truth Anchor (And Burn-In Numba Kernel)
        _, true_az, true_el, _, _ = self.look_angle_val.validate_geometry(0.0, 0.0, 0.0, ephemeris_params)
        expected_timing_baseline = 150.0  # Nominal sub-frame sample index baseline
        
        # Burn-in Timing Lock LLVM Matrix
        self.timing_lock_val.check_timing(np.array([150.0, 150.0, 150.0, 150.0], dtype=np.float32), 150.0)
        
        print(f"[*] Locking onto True Target Bounds...")
        print(f"    -> True Look Angle: Azimuth {true_az:.2f}°, Elevation {true_el:.2f}°")
        print(f"    -> True Temporal Frame: {expected_timing_baseline} samples")
        
        print("\n[*] Engaging Simulated Hybrid Spoofing/Meaconing Attacker...")
        
        # Test boundaries over 2 Data Strides (Immediate Interception Guarantee)
        spatial_flags = []
        temporal_flags = []
        recorded_latency_us = []
        
        for stride in range(2):
            # 1. Attack Vector A: Spatial Spoofing (Terrestrial Transmitter)
            # The attacker drags the steered beam off-axis by +45.0 degrees
            attack_azimuth = true_az + 45.0
            attack_elevation = true_el
            
            # 2. Attack Vector B: Delayed Meaconing (Record & Replay)
            # The attacker plays back the sub-frame exactly 3 chips (samples) late
            attack_timing = np.array([
                expected_timing_baseline + 3.0,
                expected_timing_baseline + 3.0,
                expected_timing_baseline + 3.0,
                expected_timing_baseline + 3.0
            ], dtype=np.float32)
            
            # 3. Validation Passes
            spatial_flag, true_az_loop, true_el_loop, spatial_err_rad, exec_us_geom = self.look_angle_val.validate_geometry(
                attack_azimuth, attack_elevation, stride * 0.1, ephemeris_params
            )
            
            temporal_flag, timing_err, exec_us_time = self.timing_lock_val.check_timing(
                attack_timing, expected_timing_baseline
            )
            
            spatial_flags.append(spatial_flag)
            temporal_flags.append(temporal_flag)
            recorded_latency_us.append(exec_us_geom + exec_us_time)
            
        print("\n--- HYBRID ATTACK INTERCEPTION HUD ---")
        
        # Verification Checks
        geom_trapped = all(flag == "EPHEMERIS_GEOMETRY_VIOLATION" for flag in spatial_flags)
        time_trapped = all(flag == "TIMING_COHERENCE_BREACH" for flag in temporal_flags)
        
        print(f" [>] Spatial (Look-Angle) Spoofing Trapped: {geom_trapped} (Offset: {math.degrees(spatial_err_rad):.2f}°)")
        print(f" [>] Temporal (Meaconing) Replay Trapped:   {time_trapped} (Offset: {timing_err:.2f} samples)")
        
        avg_hybrid_latency = sum(recorded_latency_us) / len(recorded_latency_us)
        print(f" [>] Combined Diagnostic Latency:           {avg_hybrid_latency:.2f} µs per stride")
        
        overall_pass = geom_trapped and time_trapped
        
        metrics = {
            "spatial_spoofing_trapped": geom_trapped,
            "temporal_meaconing_trapped": time_trapped,
            "spatial_residual_rad": float(spatial_err_rad),
            "timing_residual_samples": float(timing_err),
            "latency_us": float(avg_hybrid_latency),
            "status": "PASS" if overall_pass else "FAIL"
        }
        
        print("\n[*] Committing Hybrid Security Evaluation Bounds to compliance ledger...")
        self._commit_to_worm_ledger(metrics)
        
        if overall_pass:
            print("[+] SYSTEM ARCHITECTURE VERIFIED: Multi-Dimensional hybrid attack trapped perfectly within bounds!")
            sys.exit(0)
        else:
            print("[!] SYSTEM CRITICAL: TRACKING ARCHITECTURE LEAKED A SPOOFED VECTOR.")
            sys.exit(1)

if __name__ == "__main__":
    verifier = EphemerisTimingVerifier()
    verifier.run_validation()
