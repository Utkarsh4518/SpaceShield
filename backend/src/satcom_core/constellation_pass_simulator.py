"""
Task 63.3: Constellation Pass Simulator Module
SpaceShield High-Velocity Receiver DSP Subsystem

Connects Phase 58-62 spatial, timing, and transport layers to a simulated
multi-satellite MEO/GEO orbital pass environment.
"""

import math
import struct
import numpy as np

from adaptive_null_tracker import AdaptiveNullTracker
from hil_flow_controller import HILFlowController
from telemetry_huffman_compactor import TelemetryHuffmanCompactor

class ConstellationPassSimulator:
    """
    Simulates time-varying look-angles, link SNR, and timing drifts over
    a synthetic 1-hour orbital pass scenario.
    """
    def __init__(self):
        self.null_tracker = AdaptiveNullTracker(mu=0.03)
        self.flow_controller = HILFlowController(history_window=10)
        self.compactor = TelemetryHuffmanCompactor()

    def run_scenario(self, scenario_name: str, steps: int = 360, dt: float = 10.0) -> dict:
        """
        Runs a synthetic scenario over 360 steps (1 hour total for dt = 10.0s).
        """
        # Metrics trackers
        link_continuity = []
        null_depths = []
        rate_factors = []
        compressed_sizes = []
        
        # Initial steering weights (pointing to zenith)
        weights = np.ones(4, dtype=np.complex64) * 0.25
        target_a = np.ones(4, dtype=np.complex64)
        
        # Base timing parameters
        base_sdr = 1000.0
        base_sys = 1000.001
        
        for step in range(steps):
            t = step * dt
            
            # 1. Define orbital trajectories and look-angles
            if scenario_name == "nominal_pass":
                # Smooth single MEO satellite pass
                elevation = 10.0 + 70.0 * math.sin(math.pi * t / 3600.0)  # 10 to 80 deg
                azimuth = 45.0 + 90.0 * (t / 3600.0)
                spoofer_ele = 20.0
                spoofer_azi = 150.0
                drift_noise = 0.0005 * math.sin(t / 100.0)
                stalls = False
                handover = False
                
            elif scenario_name == "rising_edge_pass":
                # High-velocity rising pass (steep angle change)
                elevation = 10.0 + 80.0 * (t / 1800.0) if t < 1800 else 90.0
                azimuth = 30.0 + 120.0 * (t / 3600.0)
                spoofer_ele = 15.0
                spoofer_azi = 160.0
                drift_noise = 0.001 * (t / 3600.0)
                stalls = False
                handover = False
                
            elif scenario_name == "obstruction_fading":
                # Sudden obstruction blockages between steps 150 and 180 (1500s to 1800s)
                elevation = 45.0
                azimuth = 90.0
                spoofer_ele = 25.0
                spoofer_azi = 130.0
                if 150 <= step <= 180:
                    drift_noise = 0.004  # Higher noise
                    stalls = True       # PCIe/USB timing stalls
                else:
                    drift_noise = 0.0005
                    stalls = False
                handover = False
                
            elif scenario_name == "handover_overlap":
                # Handover between setting Satellite 1 and rising Satellite 2 at step 180 (1800s)
                if step < 180:
                    elevation = 80.0 - 70.0 * (t / 1800.0)  # Setting S1
                    azimuth = 45.0
                else:
                    elevation = 10.0 + 70.0 * ((t - 1800.0) / 1800.0)  # Rising S2
                    azimuth = 225.0
                spoofer_ele = 20.0
                spoofer_azi = 150.0
                drift_noise = 0.0005
                stalls = False
                handover = (step == 180)  # Sudden handover clock slip
            
            else:  # recovery_steady_state
                # Return to nominal Zenith tracking
                elevation = 80.0
                azimuth = 180.0
                spoofer_ele = 10.0
                spoofer_azi = 0.0
                drift_noise = 0.0001
                stalls = False
                handover = False

            # Convert angles to radians
            target_rad = math.radians(elevation)
            spoofer_rad = math.radians(spoofer_ele)
            
            # 2. Phase-level spatial response (Steering vector generation)
            target_a[0] = 1.0
            target_a[1] = math.cos(target_rad) + 1j * math.sin(target_rad)
            target_a[2] = math.cos(target_rad * 1.5) + 1j * math.sin(target_rad * 1.5)
            target_a[3] = math.cos(target_rad * 2.0) + 1j * math.sin(target_rad * 2.0)
            
            spoofer_a = np.zeros(4, dtype=np.complex64)
            spoofer_a[0] = 1.0
            spoofer_a[1] = math.cos(spoofer_rad) + 1j * math.sin(spoofer_rad)
            spoofer_a[2] = math.cos(spoofer_rad * 1.5) + 1j * math.sin(spoofer_rad * 1.5)
            spoofer_a[3] = math.cos(spoofer_rad * 2.0) + 1j * math.sin(spoofer_rad * 2.0)
            
            # Generate local 5-stride history of received signals for adaptive nulling
            x_history = np.zeros((5, 4), dtype=np.complex64)
            for s in range(5):
                amp = 2.0 * (math.cos(10.0 * s * 0.001) + 1j * math.sin(10.0 * s * 0.001))
                x_history[s] = amp * spoofer_a + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
                
            # Perform JIT-based adaptive null tracking
            weights, _ = self.null_tracker.track_null(weights, target_a, x_history)
            
            # Evaluate spatial null depth
            w_c = weights.real - 1j * weights.imag
            null_resp = w_c[0]*spoofer_a[0] + w_c[1]*spoofer_a[1] + w_c[2]*spoofer_a[2] + w_c[3]*spoofer_a[3]
            null_db = 20.0 * math.log10(max(abs(null_resp), 1e-6))
            null_depths.append(null_db)
            
            # 3. Timing and HIL drift tracking
            sdr_ts = base_sdr + step * dt
            sys_ts = base_sys + step * dt + drift_noise
            
            if stalls:
                # Add delay stall
                sys_ts += 0.004
            if handover:
                # Timing slip jump
                sys_ts += 0.015
                
            res = self.flow_controller.process_stride(sdr_ts, sys_ts, step + 1)
            rate_factors.append(res["rate_factor"])
            link_continuity.append(elevation > 15.0)  # Operational continuity mask
            
            # 4. Packaging and compressing telemetry
            threat_state = 1 if abs(null_db) < 20.0 else 0
            payload = struct.pack(
                "<B iddd dddd dddd bbbb d i",
                1, threat_state, 1.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 1.0,
                0, 1, 0, 1,
                sys_ts,
                step + 1
            )
            
            comp_bytes = self.compactor.compress(payload)
            compressed_sizes.append(len(comp_bytes))
            
        return {
            "link_continuity_ratio": sum(link_continuity) / steps,
            "mean_null_depth_db": np.mean(null_depths),
            "worst_null_depth_db": np.max(null_depths),
            "mean_rate_factor": np.mean(rate_factors),
            "mean_compressed_size_bytes": np.mean(compressed_sizes)
        }


# =========================================================================
# DETERMINISTIC SIMULATION VALIDATION HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Constellation Pass Simulator Harness")
    print("==================================================================")
    
    sim = ConstellationPassSimulator()
    
    scenarios = [
        "nominal_pass",
        "rising_edge_pass",
        "obstruction_fading",
        "handover_overlap",
        "recovery_steady_state"
    ]
    
    for sc in scenarios:
        res = sim.run_scenario(sc)
        print(f"\n[*] Results for Scenario: '{sc}'")
        print(f"    -> Link Continuity Ratio:     {res['link_continuity_ratio']*100:.1f}%")
        print(f"    -> Mean Adaptive Null Depth:  {res['mean_null_depth_db']:.2f} dB")
        print(f"    -> Worst-Case Null Leak:      {res['worst_null_depth_db']:.2f} dB")
        print(f"    -> Mean SDR Rate Factor:      {res['mean_rate_factor']:.2f}x")
        print(f"    -> Mean Telemetry Packet:     {res['mean_compressed_size_bytes']:.1f} bytes")
        
        # Verify baseline bounds
        assert res['mean_null_depth_db'] < -15.0, "Null depth was not correctly suppressed!"
        assert res['mean_compressed_size_bytes'] < 50.0, "Huffman compaction failed to meet size limits!"
        
    print("\n[+] Constellation pass simulation validation complete.")
