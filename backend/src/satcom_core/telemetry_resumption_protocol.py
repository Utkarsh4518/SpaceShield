"""
Task 60.3: Telemetry Resumption Protocol Module
SpaceShield High-Velocity Receiver DSP Subsystem

Implements an automatic reconnect and catch-up protocol for the operator HUD.
Generates monotonic resumption tokens, supports REST-based catch-up retrieval
of missed critical frames, and reconciles duplicates or out-of-order delivery.
"""

import time
from collections import deque

class TeleumptionToken:
    """Monotonic resumption token helper using sequence index and timestamps."""
    @staticmethod
    def generate(seq_num: int, timestamp: float) -> str:
        # Construct monotonic token string: "timestamp_ms:seq_num"
        return f"{int(timestamp * 1000)}:{seq_num}"

    @staticmethod
    def parse(token: str) -> tuple[int, int]:
        parts = token.split(":")
        if len(parts) != 2:
            return 0, 0
        return int(parts[0]), int(parts[1])


class TelemetryResumptionServer:
    """
    Server-side component maintaining a bounded cache of past dispatched frames.
    Allows clients to query missed critical alerts following socket drops.
    """
    def __init__(self, cache_limit: int = 200):
        self.cache_limit = cache_limit
        # Bounded cache storing the last N telemetry payloads
        self.history = deque(maxlen=self.cache_limit)
        self.sequence_counter = 0

    def cache_frame(self, payload: dict) -> str:
        """Assigns a monotonic token and caches the frame in the ring cache."""
        self.sequence_counter += 1
        timestamp = payload.get("timestamp") or time.time()
        
        token = TeleumptionToken.generate(self.sequence_counter, timestamp)
        payload["resumption_token"] = token
        payload["seq"] = self.sequence_counter
        
        self.history.append(payload.copy())
        return token

    def retrieve_catchup(self, last_token: str, only_critical: bool = True) -> list[dict]:
        """
        Retrieves missed frames dispatched after last_token.
        Can be configured to filter only critical mitigation alerts.
        """
        if not last_token:
            return []
            
        last_ts, last_seq = TeleumptionToken.parse(last_token)
        catchup_frames = []
        
        for frame in self.history:
            f_ts, f_seq = TeleumptionToken.parse(frame["resumption_token"])
            # Filter frames that were sent after the last known sequence number
            if f_seq > last_seq:
                is_critical = frame.get("threat_state") == 3 or frame.get("threat_verdict") == "CRITICAL_MITIGATION"
                if not only_critical or is_critical:
                    catchup_frames.append(frame.copy())
                    
        return catchup_frames


class TelemetryResumptionClient:
    """
    Client-side operator HUD reconciler.
    Filters out duplicates and orders out-of-order arrivals before rendering.
    """
    def __init__(self):
        # Store seen sequence numbers to prevent duplicate processing
        self.seen_sequences = set()
        # Out-of-order buffer holding frames to sort
        self.reorder_buffer = []

    def ingest_frame(self, frame: dict) -> bool:
        """
        Ingests a frame (from WebSocket or REST catch-up).
        Returns True if the frame is new (non-duplicate), and False if suppressed.
        """
        seq = frame.get("seq")
        if seq is None:
            return True # If no sequence key, pass through
            
        if seq in self.seen_sequences:
            return False # Suppressed duplicate!
            
        self.seen_sequences.add(seq)
        self.reorder_buffer.append(frame)
        return True

    def flush_sorted_frames(self) -> list[dict]:
        """Sorts the reorder buffer by sequence number and drains it."""
        self.reorder_buffer.sort(key=lambda x: x.get("seq", 0))
        flushed = self.reorder_buffer.copy()
        self.reorder_buffer.clear()
        return flushed


# =========================================================================
# DETERMINISTIC DISCONNECT & CATCH-UP HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Telemetry Resumption Protocol Harness")
    print("==================================================================")
    
    server = TelemetryResumptionServer(cache_limit=50)
    client = TelemetryResumptionClient()
    
    # 1. Nominal Operation: Streaming and Ingesting
    print("[*] Scenario 1: Streaming under Nominal Conditions...")
    tokens = []
    for step in range(5):
        frame = {"threat_state": 0, "timestamp": time.time(), "data": f"nominal_{step}"}
        token = server.cache_frame(frame)
        tokens.append(token)
        client.ingest_frame(frame)
        
    flushed = client.flush_sorted_frames()
    assert len(flushed) == 5, "Failed to ingest nominal frames!"
    print("    -> Nominal stream check: [PASSED]")
    
    # 2. Client Disconnect & Starvation: Client drops off, misses 10 frames
    print("\n[*] Scenario 2: Client Disconnect & Socket Loss Simulation...")
    last_processed_token = tokens[-1] # Client holds token 5
    print(f"    -> Client disconnected. Last seen token: {last_processed_token}")
    
    # Server continues caching frames while client is offline
    missed_critical_tokens = []
    for step in range(5, 15):
        # We push some nominal and some critical mitigation frames
        is_crit = (step % 3 == 0) # Steps 6, 9, 12 are critical
        state = 3 if is_crit else 0
        verdict = "CRITICAL_MITIGATION" if is_crit else "NOMINAL"
        
        frame = {
            "threat_state": state,
            "threat_verdict": verdict,
            "timestamp": time.time(),
            "data": f"backlog_{step}"
        }
        token = server.cache_frame(frame)
        if is_crit:
            missed_critical_tokens.append(token)
            
    print(f"    -> Missed critical frame count: {len(missed_critical_tokens)}")
    
    # 3. Client Reconnection & REST Catch-up: Client queries REST catch-up stream
    print("\n[*] Scenario 3: Client Reconnection and REST Catch-up...")
    # Client requests missed critical frames using its last known token
    rest_catchup_frames = server.retrieve_catchup(last_processed_token, only_critical=True)
    assert len(rest_catchup_frames) == 3, f"Expected 3 critical frames, got {len(rest_catchup_frames)}"
    
    # Client ingests the REST catch-up frames
    for frame in rest_catchup_frames:
        client.ingest_frame(frame)
        
    print("    -> REST catch-up frames successfully ingested by client.")
    
    # 4. Out-of-Order & Duplicate Suppression
    print("\n[*] Scenario 4: Reconciling Out-of-Order & Duplicates...")
    # WebSocket reconnects, and sends some overlapping frames (which client already got via REST)
    # Plus some newer nominal frames in out-of-order sequence:
    # Say, the live WebSocket sends step 12 (already got via REST) and step 15 (new)
    dup_frame = server.history[12] # Index 12 corresponds to seq 13 (step 12, critical)
    new_frame = {
        "threat_state": 0,
        "threat_verdict": "NOMINAL",
        "timestamp": time.time(),
        "data": "live_reconnect_15"
    }
    server.cache_frame(new_frame)
    
    # Ingest duplicate and new frame
    dup_ingest = client.ingest_frame(dup_frame)
    new_ingest = client.ingest_frame(new_frame)
    
    assert dup_ingest == False, "Duplicate suppression failed! Permitted duplicate frame."
    assert new_ingest == True, "Failed to ingest new frame after reconnect!"
    
    # Sort and flush client HUD
    sorted_hud = client.flush_sorted_frames()
    retained_indices = [x["seq"] for x in sorted_hud]
    print(f"    -> HUD Display sequence indices: {retained_indices}")
    
    # Expected sequence indices: 6 (critical), 9 (critical), 12 (critical), 16 (new nominal)
    # Verification checks
    assert 7 in retained_indices and 10 in retained_indices and 13 in retained_indices, "Missed critical frames in HUD!"
    assert retained_indices == sorted(retained_indices), "HUD frames are out of order!"
    print("    -> Reconnection and duplicate suppression check: [PASSED]")
    print("\n[+] Telemetry resumption protocol verified successfully across all disconnect scenarios.")
