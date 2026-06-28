"""
Task 61.1: Hardened WebSocket Runtime Module
SpaceShield High-Velocity Receiver DSP Subsystem

Provides a highly optimized, low-overhead WebSocket transport plane.
Operates on raw binary frames, using direct byte-indexing to check threat states
and managing bounded per-client queues.
"""

import time
import struct
import asyncio
from collections import deque
import numpy as np

class BinaryPriorityQueue:
    """
    Highly optimized queue managing raw binary frames.
    Extracts the threat_state directly from bytes [1:5] to avoid deserialization.
    """
    def __init__(self, max_capacity: int = 50):
        self.max_capacity = max_capacity
        self.queue = deque()
        self.drop_count = 0
        self.critical_drop_count = 0

    def push(self, binary_frame: bytes):
        """
        Pushes binary telemetry frame.
        Offset 1-5 contains the 32-bit threat_state.
        """
        b1 = binary_frame[1]
        b2 = binary_frame[2]
        b3 = binary_frame[3]
        b4 = binary_frame[4]
        # Fast byte translation (little endian)
        threat_state = b1 + (b2 << 8) + (b3 << 16) + (b4 << 24)
        if threat_state > 0x7fffffff:
            threat_state -= 0x100000000
            
        is_critical = (threat_state == 3)

        if len(self.queue) < self.max_capacity:
            self.queue.append(binary_frame)
            return

        # Handle overflow
        if is_critical:
            # Find oldest non-critical frame to drop
            non_crit_idx = -1
            for idx, frame in enumerate(self.queue):
                f_state = frame[1] + (frame[2] << 8) + (frame[3] << 16) + (frame[4] << 24)
                if f_state > 0x7fffffff:
                    f_state -= 0x100000000
                if f_state != 3:
                    non_crit_idx = idx
                    break
            
            if non_crit_idx != -1:
                del self.queue[non_crit_idx]
                self.queue.append(binary_frame)
                self.drop_count += 1
            else:
                # Discard oldest critical frame
                self.queue.popleft()
                self.queue.append(binary_frame)
                self.critical_drop_count += 1
        else:
            self.drop_count += 1

    def pop(self) -> bytes:
        if self.queue:
            return self.queue.popleft()
        return None

    def __len__(self):
        return len(self.queue)


class HardenedWebSocketRuntime:
    """
    High-capacity WebSocket connection manager.
    Coordinates non-blocking fan-out to hundreds of concurrent HUD clients.
    """
    def __init__(self, queue_capacity: int = 50):
        self.queue_capacity = queue_capacity
        self.clients = {}  # client_id -> BinaryPriorityQueue
        self.lock = asyncio.Lock()
        
        # Diagnostics
        self.total_broadcasts = 0
        self.total_dropped_frames = 0
        self.total_critical_drops = 0

    async def register_client(self, client_id: str) -> BinaryPriorityQueue:
        async with self.lock:
            q = BinaryPriorityQueue(max_capacity=self.queue_capacity)
            self.clients[client_id] = q
            return q

    async def unregister_client(self, client_id: str):
        async with self.lock:
            if client_id in self.clients:
                del self.clients[client_id]

    async def broadcast_binary(self, binary_frame: bytes):
        """
        Fast fan-out of raw bytes to all connected clients.
        Executed asynchronously to prevent blocking the DSP worker threads.
        """
        self.total_broadcasts += 1
        async with self.lock:
            for client_id, q in self.clients.items():
                q.push(binary_frame)
                # Keep overall counters updated
                self.total_dropped_frames += q.drop_count
                self.total_critical_drops += q.critical_drop_count
                q.drop_count = 0
                q.critical_drop_count = 0


# =========================================================================
# DETERMINISTIC LOAD TEST HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Hardened WebSocket Delivery Plane Load Test")
    print("==================================================================")
    
    # 109-byte schema layout helpers
    # Format: B (version=1), i (threat_state), d (jammer), d (spoof), d (sphericity), 4d (skew), 4d (aoa), 4b (nulling), d (timestamp), i (drops)
    def create_mock_binary_frame(threat_state: int) -> bytes:
        return struct.pack(
            "<B iddd dddd dddd bbbb d i",
            1, threat_state, 0.1, 0.2, 14.5,
            0.1, 0.2, 0.3, 0.4,
            0.01, 0.02, 0.03, 0.04,
            0, 1, 1, 0,
            time.time(),
            0
        )
        
    async def execute_load_tests():
        runtime = HardenedWebSocketRuntime(queue_capacity=20)
        
        # Scenario 1: Burst Fan-Out (200 concurrent clients)
        print("[*] Scenario 1: Simulating 200 Concurrent Clients Burst...")
        for c in range(200):
            await runtime.register_client(f"operator_hud_{c}")
            
        t_start = time.perf_counter()
        # Broadcast 100 frames to all 200 clients (20,000 deliveries total)
        for step in range(100):
            frame = create_mock_binary_frame(threat_state=0)
            await runtime.broadcast_binary(frame)
            
        t_end = time.perf_counter()
        tot_time_ms = (t_end - t_start) * 1000
        per_frame_us = (t_end - t_start) * 1e6 / 100
        
        print(f"    -> 20,000 deliveries completed in {tot_time_ms:.2f} ms")
        print(f"    -> Stride broadcast overhead:      {per_frame_us:.2f} µs")
        assert per_frame_us < 2000.0, "Broadcast stride overhead exceeded safety limits!"
        
        # Scenario 2: Slow-Client Backlog & Priority Alert Coalescing
        print("\n[*] Scenario 2: Slow-Client Backlog & Priority Alert Ingestion...")
        # Clear dispatcher clients
        runtime.clients.clear()
        
        # Register a single slow client
        q_slow = await runtime.register_client("slow_client")
        
        # Push 20 nominal frames to fill the queue
        for i in range(20):
            await runtime.broadcast_binary(create_mock_binary_frame(threat_state=0))
            
        # Push 10 critical alerts (threat_state = 3) + 10 nominal frames (threat_state = 0)
        for i in range(10):
            await runtime.broadcast_binary(create_mock_binary_frame(threat_state=3))
            await runtime.broadcast_binary(create_mock_binary_frame(threat_state=0))
            
        # Verify drops and priorities
        print(f"    -> Slow client queue size: {len(q_slow)} (Expected: 20)")
        print(f"    -> Dropped nominal frames: {q_slow.drop_count}")
        print(f"    -> Dropped critical alerts: {q_slow.critical_drop_count}")
        
        assert len(q_slow) == 20
        # The 10 critical alerts should have displaced 10 nominal frames in the queue
        retained_states = []
        while len(q_slow) > 0:
            frame = q_slow.pop()
            b_state = frame[1] + (frame[2] << 8) + (frame[3] << 16) + (frame[4] << 24)
            retained_states.append(b_state)
            
        critical_retained = retained_states.count(3)
        print(f"    -> Critical Alerts Retained in Backlog: {critical_retained}")
        assert critical_retained == 10, "Critical alerts were displaced by routine status traffic!"
        print("    -> Priority alert check: [PASSED]")
        
        # Scenario 3: Reconnect-After-Drop Behavior
        print("\n[*] Scenario 3: Reconnect and Queue Rebuilding...")
        # Unregister and re-register
        await runtime.unregister_client("slow_client")
        assert "slow_client" not in runtime.clients
        q_new = await runtime.register_client("slow_client")
        assert len(q_new) == 0
        print("    -> Reconnect validation check: [PASSED]")

    asyncio.run(execute_load_tests())
    print("\n[+] Hardened WebSocket runtime validation successfully completed.")
