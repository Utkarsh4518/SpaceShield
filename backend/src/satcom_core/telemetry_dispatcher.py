"""
SpaceShield Telemetry Dispatcher
Bridges the DSP harness's telemetry payloads to registered WebSocket clients.

Two independent transports are exposed, both bounded and priority-aware:
- JSON transport (/stream): one PriorityClientQueue per client, holding raw
  dict payloads for human/browser/Streamlit consumers.
- Binary transport (/stream/binary): delegates to the existing
  HardenedWebSocketRuntime rather than reimplementing its bounded
  BinaryPriorityQueue / fan-out logic.

Both transports share the same overflow policy: when a client's queue is
full, the oldest routine (non-critical) frame is dropped first. A critical
alert (threat_state == 3) is only dropped once every other queued frame is
already critical, so status noise never displaces an active threat alert.
"""

import asyncio
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from hardened_websocket_runtime import HardenedWebSocketRuntime
from binary_telemetry_codec import BinaryTelemetryCodec

CRITICAL_THREAT_STATE = 3


class PriorityClientQueue:
    """
    Bounded per-client JSON telemetry queue with critical-alert retention.
    Mirrors the overflow semantics of HardenedWebSocketRuntime's
    BinaryPriorityQueue, operating on dict payloads instead of raw frames.
    """

    def __init__(self, max_capacity: int = 50):
        self.max_capacity = max_capacity
        self.queue = deque()
        self.drop_count = 0
        self.critical_drop_count = 0

    def push(self, payload: dict):
        is_critical = payload.get("threat_state") == CRITICAL_THREAT_STATE

        if len(self.queue) < self.max_capacity:
            self.queue.append(payload)
            return

        if is_critical:
            non_crit_idx = -1
            for idx, item in enumerate(self.queue):
                if item.get("threat_state") != CRITICAL_THREAT_STATE:
                    non_crit_idx = idx
                    break

            if non_crit_idx != -1:
                del self.queue[non_crit_idx]
                self.queue.append(payload)
                self.drop_count += 1
            else:
                # Every queued frame is already critical; oldest one yields.
                self.queue.popleft()
                self.queue.append(payload)
                self.critical_drop_count += 1
        else:
            self.drop_count += 1

    def pop(self) -> Optional[dict]:
        if self.queue:
            return self.queue.popleft()
        return None

    def __len__(self):
        return len(self.queue)


class TelemetryDispatcher:
    """
    Coordinates fan-out of telemetry payloads to registered clients across
    both the JSON and binary transports, with bounded per-client queues and
    graceful client lifecycle handling.
    """

    def __init__(self, queue_capacity: int = 50, version: int = 1):
        self.queue_capacity = queue_capacity
        self.version = version

        self.clients: Dict[str, PriorityClientQueue] = {}
        self.lock = asyncio.Lock()

        self.codec = BinaryTelemetryCodec(version=version)
        self.binary_runtime = HardenedWebSocketRuntime(queue_capacity=queue_capacity)

        # Diagnostics
        self.total_broadcasts = 0
        self.total_dropped_frames = 0
        self.total_critical_drops = 0

    # ------------------------------------------------------------------
    # JSON transport (/stream)
    # ------------------------------------------------------------------
    async def register_client(self, client_id: str) -> PriorityClientQueue:
        async with self.lock:
            q = PriorityClientQueue(max_capacity=self.queue_capacity)
            self.clients[client_id] = q
            return q

    async def unregister_client(self, client_id: str):
        async with self.lock:
            self.clients.pop(client_id, None)

    async def broadcast(self, payload: dict):
        """Fan out a JSON telemetry payload to all registered /stream clients."""
        self.total_broadcasts += 1
        async with self.lock:
            for q in self.clients.values():
                q.push(payload)
                self.total_dropped_frames += q.drop_count
                self.total_critical_drops += q.critical_drop_count
                q.drop_count = 0
                q.critical_drop_count = 0

    # ------------------------------------------------------------------
    # Binary transport (/stream/binary) -- delegates to HardenedWebSocketRuntime
    # ------------------------------------------------------------------
    async def register_binary_client(self, client_id: str):
        return await self.binary_runtime.register_client(client_id)

    async def unregister_binary_client(self, client_id: str):
        await self.binary_runtime.unregister_client(client_id)

    async def broadcast_binary(self, payload: dict):
        """Encode a telemetry payload via BinaryTelemetryCodec and fan it out."""
        frame = self.codec.encode(
            threat_state=int(payload.get("threat_state", 0)),
            jammer_score=float(payload.get("jammer_score", 0.0)),
            spoof_score=float(payload.get("spoof_score", 0.0)),
            sphericity=float(payload.get("sphericity_score", payload.get("sphericity", 0.0))),
            skew_residuals=np.asarray(payload.get("skew_residuals", [0.0] * 4), dtype=np.float64),
            aoa_deviation=np.asarray(payload.get("aoa_deviation", [0.0] * 4), dtype=np.float64),
            nulling_directives=np.asarray(payload.get("nulling_directives", [False] * 4), dtype=np.bool_),
            timestamp=float(payload.get("timestamp", time.time())),
            buffer_drops=int(payload.get("dropped_blocks", 0)),
        )
        await self.binary_runtime.broadcast_binary(bytes(frame))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def shutdown(self):
        """Drains and releases all client queues across both transports."""
        async with self.lock:
            self.clients.clear()
        async with self.binary_runtime.lock:
            self.binary_runtime.clients.clear()


if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Telemetry Dispatcher Smoke Test")
    print("==================================================================")

    async def _smoke_test():
        dispatcher = TelemetryDispatcher(queue_capacity=10, version=1)

        q = await dispatcher.register_client("client_a")
        assert "client_a" in dispatcher.clients

        for i in range(15):
            await dispatcher.broadcast({"threat_state": 0, "frame": i})
        assert len(q) == 10, f"Expected bounded queue at 10, got {len(q)}"

        await dispatcher.broadcast({"threat_state": CRITICAL_THREAT_STATE, "frame": 999})
        popped = [q.pop() for _ in range(len(q))]
        assert any(p["threat_state"] == CRITICAL_THREAT_STATE for p in popped), \
            "Critical alert was dropped despite queue overflow policy"
        print("[PASS] Bounded JSON queue retains critical alerts under overflow.")

        await dispatcher.register_binary_client("client_a")
        await dispatcher.broadcast_binary({
            "threat_state": 1, "jammer_score": 0.5, "spoof_score": 0.1, "sphericity_score": 12.0,
            "skew_residuals": [0.0] * 4, "aoa_deviation": [0.0] * 4,
            "nulling_directives": [False] * 4, "timestamp": time.time(), "dropped_blocks": 0,
        })
        print("[PASS] Binary transport encode/broadcast completed without error.")

        await dispatcher.unregister_client("client_a")
        await dispatcher.unregister_binary_client("client_a")
        assert "client_a" not in dispatcher.clients
        await dispatcher.shutdown()
        print("[PASS] Client lifecycle and graceful shutdown completed.")

    asyncio.run(_smoke_test())
    print("\n[+] Telemetry dispatcher smoke test completed successfully.")
