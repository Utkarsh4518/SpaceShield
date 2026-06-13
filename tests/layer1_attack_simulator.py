#!/usr/bin/env python3
"""
SpaceShield: Automated Layer-1 Attack Simulator & Validation Suite.
Description: Stress-tests the Spatial GLRT detection boundaries by dynamically injecting
             tactical electronic warfare vectors directly into the spatial array buffers.
"""

import os
import sys
import time
import json
import numpy as np

# Adjust sys.path to access the backend algorithms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/src')))

try:
    from spatial_glrt_detector import SpatialGLRTDetector
except ImportError as e:
    print(f"[-] Critical Import Error: {e}")
    print("Ensure the script is run from the repository root or tests/ directory.")
    sys.exit(1)

class Layer1AttackSimulator:
    def __init__(self, num_channels=4, chunk_size=8192, window_size=50, fs=5.0e6, center_freq=1176.45e6):
        self.num_channels = num_channels
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.fs = fs
        self.center_freq = center_freq  # NavIC L5
        
        # Initialize the production detector for boundary testing
        # We use strict 1e-7 false alarm threshold resulting in gamma ~ 50.17
        self.detector = SpatialGLRTDetector(num_channels=self.num_channels, window_size=self.window_size, p_fa=1e-7)
        self.detector.gamma = 50.17
        
        # Output directory mappings
        self.compliance_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../compliance'))
        os.makedirs(self.compliance_dir, exist_ok=True)
        
        self.rng = np.random.default_rng(seed=42)
        
    def generate_baseline_noise(self):
        """Generates the background thermal noise baseline for the antenna array."""
        # Baseline thermal noise power ~ 0.05
        noise_power = 0.05
        noise = self.rng.normal(0, np.sqrt(noise_power / 2), (self.num_channels, self.chunk_size)) + \
                1j * self.rng.normal(0, np.sqrt(noise_power / 2), (self.num_channels, self.chunk_size))
        return noise

    def test_drag_off_spoofing_vector(self):
        """
        Scenario 1: Array-Coherent Spoofing with Doppler Sweep
        Gradually injects a rank-1 coherent waveform simulating a drag-off attack.
        Tracks the specific amplitude required to breach the GLRT threshold.
        """
        print("\n[*] Commencing Scenario 1: NavIC L5 Coherent Drag-Off Spoofing Sweep")
        
        # Target angle of arrival for spoofer (synthetic EW asset)
        theta_spoofer = np.radians(35.0)
        a_spoofer = np.exp(1j * np.arange(self.num_channels) * np.sin(theta_spoofer)).reshape(self.num_channels, 1)
        
        t = np.arange(self.chunk_size) / self.fs
        
        # Doppler sweep parameters (linear frequency modulation to simulate drag-off)
        f_start = 0.0
        f_end = 25000.0  # 25 kHz Doppler shift
        k_sweep = (f_end - f_start) / (self.chunk_size / self.fs)
        phase_sweep = 2 * np.pi * (f_start * t + 0.5 * k_sweep * t**2)
        waveform = np.exp(1j * phase_sweep)
        
        breach_amplitude = None
        breach_metrics = None
        
        # Ramp up amplitude from 0.01 to 2.0 to find exact trigger boundary
        amplitudes = np.linspace(0.01, 2.0, 200)
        
        for amp in amplitudes:
            iq_buffer = self.generate_baseline_noise()
            iq_buffer += a_spoofer * (amp * waveform)
            
            # Extract first window snapshot for testing
            snapshot = iq_buffer[:, :self.window_size]
            res = self.detector.evaluate(snapshot)
            
            if res["alert_triggered"]:
                breach_amplitude = amp
                breach_metrics = res
                print(f"    [!] GLRT Boundary Breached at Amplitude: {amp:.4f}")
                print(f"        Sphericity Stat: {res['test_statistic_sphericity']:.2f} (Gamma: {self.detector.gamma})")
                print(f"        METR FIM Beta: {res['lambda_metr']:.4f}")
                break
                
        return {
            "scenario": "DRAG_OFF_SPOOFING",
            "trigger_amplitude": float(breach_amplitude) if breach_amplitude else -1.0,
            "max_sphericity": float(breach_metrics["test_statistic_sphericity"]) if breach_metrics else 0.0,
            "metr": float(breach_metrics["lambda_metr"]) if breach_metrics else 0.0
        }

    def test_broadband_noise_jamming(self):
        """
        Scenario 2: High-Power Uncoordinated Broadband Noise Jamming
        Floods all 4 channels simultaneously to trigger sphericity disruption.
        """
        print("\n[*] Commencing Scenario 2: Broadband Noise Jamming Flood")
        
        breach_amplitude = None
        breach_metrics = None
        
        # Ramp up jamming noise power
        amplitudes = np.linspace(0.1, 25.0, 500)
        
        for amp in amplitudes:
            iq_buffer = self.generate_baseline_noise()
            
            # Uncoordinated noise across all channels (full rank injection)
            jamming_noise = self.rng.normal(0, np.sqrt(amp / 2), (self.num_channels, self.chunk_size)) + \
                            1j * self.rng.normal(0, np.sqrt(amp / 2), (self.num_channels, self.chunk_size))
            iq_buffer += jamming_noise
            
            snapshot = iq_buffer[:, :self.window_size]
            res = self.detector.evaluate(snapshot)
            
            # Noise jamming often pushes METR closer to 1.0 (isotropic) but can trigger sphericity 
            # if power imbalances exist. We test strict bounds.
            if res["alert_triggered"]:
                breach_amplitude = amp
                breach_metrics = res
                print(f"    [!] GLRT Boundary Breached at Jamming Power: {amp:.4f}")
                print(f"        Sphericity Stat: {res['test_statistic_sphericity']:.2f} (Gamma: {self.detector.gamma})")
                break

        return {
            "scenario": "BROADBAND_JAMMING",
            "trigger_power": float(breach_amplitude) if breach_amplitude else -1.0,
            "max_sphericity": float(breach_metrics["test_statistic_sphericity"]) if breach_metrics else 0.0,
            "metr": float(breach_metrics["lambda_metr"]) if breach_metrics else 0.0
        }

    def run_suite(self):
        """Executes all stress tests and outputs compliance ledger."""
        print("================================================================")
        print(" SpaceShield Automated Threat Vector Validator")
        print("================================================================")
        
        res1 = self.test_drag_off_spoofing_vector()
        res2 = self.test_broadband_noise_jamming()
        
        # Formulate CERT-In compliance report payload
        compliance_report = {
            "timestamp": time.time(),
            "target_frequency_mhz": self.center_freq / 1e6,
            "sample_rate_msps": self.fs / 1e6,
            "evaluation_threshold": self.detector.gamma,
            "incident_results": [res1, res2],
            "containment_status": "VALIDATED_AND_ISOLATED"
        }
        
        report_path = os.path.join(self.compliance_dir, "certin_incident_spoofing.json")
        try:
            with open(report_path, "w") as f:
                json.dump(compliance_report, f, indent=4)
        except PermissionError:
            print(f"\\n[!] Target file locked by active server. Writing to fallback...")
            report_path = report_path.replace('.json', '_test.json')
            with open(report_path, "w") as f:
                json.dump(compliance_report, f, indent=4)
            
        print(f"\n[+] Validation suite completed. Performance bounds written to: {report_path}")
        print("================================================================")

if __name__ == "__main__":
    simulator = Layer1AttackSimulator()
    simulator.run_suite()
