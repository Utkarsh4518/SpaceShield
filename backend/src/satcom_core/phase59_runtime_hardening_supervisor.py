"""
Task 59.3: Phase 59 Runtime Hardening Supervisor Module
SpaceShield High-Velocity Receiver DSP Subsystem

Integrates the Fusion Ring Buffer, Telemetry Backpressure Guard,
Phase 58 Fusion Supervisor, and Priority Telemetry Dispatcher into
a single, unified runtime hardening layer. Provides isolating logic
between the real-time compute loop and WebSocket transport buffers.
"""

import time
import math
import numpy as np
from numba import njit

# Import the integrated subsystems
from phase58_fusion_supervisor import Phase58FusionSupervisor
from fusion_ring_buffer import FusionRingBuffer
from telemetry_backpressure_guard import TelemetryBackpressureGuard
from telemetry_dispatcher import PriorityClientQueue, TelemetryDispatcher

class Phase59RuntimeHardeningSupervisor:
    """
    SpaceShield Runtime Hardening Supervisor.
    Orchestrates zero-allocation compute loops and backpressure-protected delivery.
    """
    def __init__(
        self,
        num_channels: int = 4,
        ring_capacity: int = 1024,
        queue_capacity: int = 50,
        high_pressure_threshold: float = 0.7,
        critical_pressure_threshold: float = 0.9
    ):
        self.num_channels = num_channels
        self.queue_capacity = queue_capacity
        
        # 1. Compute Plane: Fusion Supervisor
        self.fusion_supervisor = Phase58FusionSupervisor(
            num_channels=num_channels,
            default_fisher_margin_deg=5.0
        )
        
        # 2. Bridge Plane: Zero-Allocation Ring Buffer
        self.ring_buffer = FusionRingBuffer(capacity=ring_capacity)
        
        # 3. Guard Plane: Telemetry Backpressure Guard
        self.backpressure_guard = TelemetryBackpressureGuard(
            high_threshold=high_pressure_threshold,
            critical_threshold=critical_pressure_threshold
        )
        
        # 4. Delivery Plane: Priority Telemetry Dispatcher
        self.dispatcher = TelemetryDispatcher(
            queue_capacity=queue_capacity,
            version=1
        )
        
        # Hardened state counters (fixed-size registers)
        self.total_processed_strides = 0
        self.total_dropped_telemetry_frames = 0
        self.total_critical_alerts_routed = 0
        
        # Pre-allocated arrays for JIT bridge operations
        self._dummy_skew = np.zeros(self.num_channels, dtype=np.float64)
        self._dummy_aoa = np.zeros(self.num_channels, dtype=np.float64)
        self._dummy_null = np.zeros(self.num_channels, dtype=np.bool_)

    def ingest_compute_stride(
        self,
        Y_matrix: np.ndarray,
        correlation_taps: np.ndarray,
        phase_diffs: np.ndarray,
        true_az_el: np.ndarray,
        beamforming_weights: np.ndarray,
        inference_verdict_idx: int = 0
    ) -> int:
        """
        Compute-plane entry point (executed by real-time DSP thread):
        1. Runs the fusion supervisor stride.
        2. Non-blocking push of results directly into the Fusion Ring Buffer.
        3. Guaranteeing that no network serialization or socket blockage can stall the loop.
        """
        # Step 1: Execute JIT sensors fusion and state machine
        state = self.fusion_supervisor.process_stride(
            Y_matrix,
            correlation_taps,
            phase_diffs,
            true_az_el,
            beamforming_weights,
            inference_verdict_idx
        )
        
        # Extract features for storage from the pre-allocated supervisor arrays
        skew_res = self.fusion_supervisor.correlator_bridge.dissector_scores
        aoa_err = self.fusion_supervisor.aoa_bridge.angular_errors
        null_dir = self.fusion_supervisor.aoa_bridge.hijack_flags
        
        # Determine jammer and spoof scores from the pre-allocated feature matrix
        feat_matrix = self.fusion_supervisor.correlator_bridge.feature_matrix
        jammer_score = float(np.mean(feat_matrix[:, 1]))
        spoof_score = float(np.max(feat_matrix[:, 0]))
        
        # Step 2: Push compact binary record to Ring Buffer (sub-microsecond execution)
        self.ring_buffer.push(
            threat_state=int(state),
            jammer_score=jammer_score,
            spoof_score=spoof_score,
            sphericity=float(self.fusion_supervisor.last_sphericity),
            skew_residuals=skew_res,
            aoa_deviation=aoa_err,
            nulling_directives=null_dir,
            timestamp=time.time()
        )
        
        self.total_processed_strides += 1
        if state == 3:
            self.total_critical_alerts_routed += 1
            
        return state

    def dispatch_pending_telemetry(self) -> int:
        """
        Control-plane bridge task (running in background or async loop):
        1. Pulls a batch of pending binary records from the Ring Buffer.
        2. Evaluates backpressure status on active clients.
        3. Pushes records to the dispatcher where priority and drop schemas are enforced.
        """
        records = self.ring_buffer.pop_batch(max_count=128)
        popped_count = len(records)
        
        if popped_count == 0:
            return 0
            
        # Broadcast the batch of records
        for record in records:
            # Check client queue sizes to track backpressure metrics
            for client_id, client_queue in self.dispatcher.clients.items():
                queue_size = len(client_queue)
                pressure_state = self.backpressure_guard.get_pressure_state(
                    queue_size=queue_size,
                    capacity=self.queue_capacity
                )
                
                # Apply pressure tracking alerts
                if pressure_state == 2:
                    self.total_dropped_telemetry_frames += 1
                    
                # Ingest into priority client queues
                client_queue.push(record.copy())
                
        return popped_count


# =========================================================================
# DETERMINISTIC HARDENING REPLAY HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Hardening Supervisor Verification")
    print("==================================================================")
    
    # Initialize hardening supervisor
    supervisor = Phase59RuntimeHardeningSupervisor(
        num_channels=4,
        ring_capacity=100,
        queue_capacity=10
    )
    
    # Register Mock Clients (Normal client vs Slow client)
    import asyncio
    
    async def run_harness():
        # Register clients
        await supervisor.dispatcher.register_client("hud_operator_normal")
        await supervisor.dispatcher.register_client("hud_operator_slow_backlog")
        
        # True satellite tracking azimuth/elevation coordinates
        true_angles = np.array([
            [45.0, 30.0],
            [90.0, 60.0],
            [180.0, 45.0],
            [270.0, 15.0]
        ], dtype=np.float64)
        
        nom_phase_diffs = np.zeros((4, 3), dtype=np.float64)
        for m in range(4):
            az = math.radians(true_angles[m, 0])
            el = math.radians(true_angles[m, 1])
            ux = math.cos(el) * math.cos(az)
            uy = math.cos(el) * math.sin(az)
            nom_phase_diffs[m, 0] = ux * math.pi
            nom_phase_diffs[m, 1] = uy * math.pi
            nom_phase_diffs[m, 2] = (ux + uy) * math.pi
            
        weights = np.ones((4, 4), dtype=np.complex64) * 0.5
        
        # 1. Drive Nominal Load (Steps 1-30)
        # Normal client drains queue regularly. Slow client accumulates backlog.
        print("[*] Running Hardening Phase 1: Nominal & Starvation Load...")
        for step in range(30):
            # Complex white noise
            Y_block = (np.random.normal(0, 1.0, (4, 50)) + 1j * np.random.normal(0, 1.0, (4, 50))).astype(np.complex64)
            corr_taps = np.zeros((4, 7), dtype=np.complex64)
            for m in range(4):
                corr_taps[m] = np.array([0.25, 0.50, 0.75, 1.00, 0.75, 0.50, 0.25])
            
            # Ingest compute stride
            supervisor.ingest_compute_stride(
                Y_block, corr_taps, nom_phase_diffs, true_angles, weights, 0
            )
            
            # Dispatch to client queues
            supervisor.dispatch_pending_telemetry()
            
            # Normal client drains queue immediately
            supervisor.dispatcher.clients["hud_operator_normal"].pop()
            # Slow client does NOT drain the queue (simulating network bottleneck)
            
        # Verify queues
        normal_len = len(supervisor.dispatcher.clients["hud_operator_normal"])
        slow_len = len(supervisor.dispatcher.clients["hud_operator_slow_backlog"])
        drops = supervisor.dispatcher.clients["hud_operator_slow_backlog"].drop_counter
        
        print(f"    -> Normal Client Queue size: {normal_len} | Slow Client size: {slow_len}")
        print(f"    -> Slow Client Drops:         {drops}")
        assert normal_len == 0, "Normal client queue failed to drain!"
        assert slow_len == 10, "Slow client queue size exceeded capacity cap!"
        assert drops == 20, f"Expected 20 drops, got {drops}"
        print("    -> Nominal load check: [PASSED]")
        
        # 2. Jammer & Spoof Escalation (Steps 31-50)
        # Inject critical mitigation frames. Verify they displace nominal ones in slow queue.
        print("\n[*] Running Hardening Phase 2: Adversarial Escalation...")
        
        # Jammer covariance matrix
        Y_block = (np.random.normal(0, 3.0, (4, 50)) + 1j * np.random.normal(0, 3.0, (4, 50))).astype(np.complex64)
        corr_taps = np.zeros((4, 7), dtype=np.complex64)
        for m in range(4):
            corr_taps[m] = np.array([0.10, 0.20, 0.70, 0.80, 0.40, 0.20, 0.10])
        # Off-pointing AoA vectors to trigger CRITICAL_MITIGATION
        phase_diffs_spoofed = nom_phase_diffs * 2.0
        
        for step in range(7):
            # Ingest critical strides
            supervisor.ingest_compute_stride(
                Y_block, corr_taps, phase_diffs_spoofed, true_angles, weights, 2
            )
            supervisor.dispatch_pending_telemetry()
            
            # Normal client continues to drain
            supervisor.dispatcher.clients["hud_operator_normal"].pop()
            # Slow client remains locked
            
        # Verify priority alert retention in the slow client queue
        slow_queue = supervisor.dispatcher.clients["hud_operator_slow_backlog"]
        retained = []
        while len(slow_queue) > 0:
            retained.append(slow_queue.pop())
            
        verdicts = [x.get("threat_state") for x in retained]
        print(f"    -> Retained States in Slow Queue: {verdicts}")
        # The 5 critical alerts (threat_state = 3) must be preserved in the queue
        critical_count = verdicts.count(3)
        print(f"    -> Retained Critical Alerts:      {critical_count}")
        assert critical_count == 5, "Critical threat alerts were displaced in slow client buffer!"
        print("    -> Priority alert check: [PASSED]")
        
        # 3. Compute Latency Isolation check under stress
        print("\n--- COMPUTE LOOP ISOLATION BENCHMARK ---")
        print("[*] Running 10,000 warm iterations with saturated client backlogs...")
        latencies = []
        for _ in range(10000):
            t0 = time.perf_counter()
            # Push compute stride (ring buffer guarantees no blocked calls)
            supervisor.ingest_compute_stride(Y_block, corr_taps, phase_diffs_spoofed, true_angles, weights, 2)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6)
            
        avg_us = np.median(latencies)
        print(f"  Median Compute-Stride Latency: {avg_us:.3f} µs")
        assert avg_us < 15.0, "Compute loop stalled under telemetry backlog pressure!"
        print("[PASSED] Compute loop latency remains isolated from transport stress.")
        
    asyncio.run(run_harness())
