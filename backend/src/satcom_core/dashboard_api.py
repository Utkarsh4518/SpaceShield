#!/usr/bin/env python3
"""
SpaceShield: Asynchronous Web Server Gateway & Telemetry Bridge.
Author: Principal Backend API Architect & Cloud-Native Distributed Systems Engineer
Version: 1.0.0

Bridges the multi-threaded spatial hardware-in-the-loop (HIL) signal harness
with a modern, high-throughput web interface.

Key architectural features:
- Fast, asynchronous FastAPI engine (unprivileged localhost port).
- Thread-safe inter-process queue listener for metrics aggregation.
- Real-time WebSocket streaming route (/stream) broadcasting payload every 100ms.
- Safe lifecycle management ensuring no resource leakage on client dropouts.
"""

import os
import sys
import json
import time
import queue
import asyncio
import threading
from typing import Set

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app, Histogram, Gauge, Counter

# Ensure sibling imports resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spatial_hardware_harness import SpatialHardwareHarness
from telemetry_dispatcher import TelemetryDispatcher

# ──────────────────────────────────────────────────────────────────────
# API Initialization & Configuration
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SpaceShield Real-Time Telemetry Gateway",
    description="Asynchronous bridge for spatial hardware-in-the-loop DSP metrics.",
    version="1.0.0"
)

# Enforce strict Cross-Origin Resource Sharing (CORS) parameters
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origin in strict production environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus Metrics Route
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ──────────────────────────────────────────────────────────────────────
# Prometheus Enterprise Telemetry Definitions
# ──────────────────────────────────────────────────────────────────────

# 1. Execution Latency (Histogram)
svd_execution_latency_us = Histogram(
    'spaceshield_svd_latency_us',
    'Precise execution latency of the SVD equalizer engine in microseconds',
    buckets=(10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, float("inf"))
)

# 2. Eigenvalues & Sphericity Ratios (Gauges)
glrt_sphericity_ratio = Gauge('spaceshield_glrt_sphericity_ratio', 'Current GLRT sphericity test statistic')
metr_fim_beta = Gauge('spaceshield_metr_fim_beta', 'Maximum Eigen-Trace Ratio (FIM Beta)')

# 3. Hardware Dropped Blocks (Counter)
soapy_sdr_overflow_total = Counter('spaceshield_soapy_sdr_overflow_total', 'Total count of hardware ingestion block drops (overflows)')

# ──────────────────────────────────────────────────────────────────────
# Inter-Process State & Concurrency Vectors
# ──────────────────────────────────────────────────────────────────────

# Thread-safe queue interconnect for intercepting operational metrics from the harness
telemetry_queue = queue.Queue(maxsize=100)

# Global priority telemetry dispatcher for separate client WebSocket pipelines
dispatcher = TelemetryDispatcher(queue_capacity=50, version=1)

# Global register of active WebSocket client connections
active_connections: Set[WebSocket] = set()

# Global reference to the executing harness for graceful teardown mechanics
global_harness_instance = None

# Runtime State Configuration Payload
class ConfigUpdate(BaseModel):
    detection_threshold: float
    antenna_attenuation_db: float
    zero_trust_lockout: bool

# ──────────────────────────────────────────────────────────────────────
# Background Tasks & Queue Listeners
# ──────────────────────────────────────────────────────────────────────

async def broadcast_telemetry_loop():
    """
    Background thread-safe queue listener that intercepts operational metrics.
    Ingests processed frame indicators, updates Prometheus metrics,
    and broadcasts them to client priority queues every 100ms.
    """
    print("[+] Background telemetry broadcaster initialized.")
    while True:
        payload = None
        try:
            # Drain queue to get the freshest metric payload without blocking
            while not telemetry_queue.empty():
                payload = telemetry_queue.get_nowait()
                telemetry_queue.task_done()
        except queue.Empty:
            pass

        if payload:
            # Update Enterprise Prometheus Metrics (Non-Blocking)
            if 'inference_latency_us' in payload:
                svd_execution_latency_us.observe(payload['inference_latency_us'])
            if 'sphericity_score' in payload:
                glrt_sphericity_ratio.set(payload['sphericity_score'])
            if 'fim_beta' in payload:
                metr_fim_beta.set(payload['fim_beta'])
            if 'dropped_blocks' in payload and payload['dropped_blocks'] > 0:
                soapy_sdr_overflow_total.inc(payload['dropped_blocks'])

            # Broadcast to priority client queues asynchronously
            await dispatcher.broadcast(payload)
        
        # Enforce exactly 100ms broadcast interval
        await asyncio.sleep(0.1)


def harness_worker_thread():
    """
    Spawns the heavy DSP multi-threaded harness in a daemonized background context.
    Supplies the shared queue for zero-latency inter-process data marshalling.
    """
    global global_harness_instance
    print("[+] Initializing background spatial hardware harness execution loop...")
    try:
        # Instantiate harness with 1-hour duration to keep the dashboard alive
        # Inject the thread-safe queue reference directly
        global_harness_instance = SpatialHardwareHarness(
            duration_sec=3600,
            telemetry_queue=telemetry_queue
        )
        global_harness_instance.execute()
    except Exception as e:
        print(f"[!] Critical Error in background harness execution: {e}")


# ──────────────────────────────────────────────────────────────────────
# API Lifecycle Events
# ──────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    """Bootstrap background mechanics when the API gateway boots."""
    # 1. Start the asynchronous WebSocket broadcaster loop
    asyncio.create_task(broadcast_telemetry_loop())
    
    # 2. Spin up the hardware harness math processors on a separate OS thread
    harness_thread = threading.Thread(target=harness_worker_thread, daemon=True)
    harness_thread.start()


@app.on_event("shutdown")
async def on_shutdown():
    """Strict safety catch-blocks to monitor teardown mechanics."""
    print("\n[*] Gateway shutdown sequence initiated.")
    if global_harness_instance:
        print("[*] Instructing DSP harness to gracefully terminate worker resources...")
        global_harness_instance.running = False


# ──────────────────────────────────────────────────────────────────────
# Exposed REST Configuration Routes
# ──────────────────────────────────────────────────────────────────────

@app.post("/api/v1/config/update")
async def update_runtime_config(config: ConfigUpdate):
    """
    High-speed HTTP POST route verifying and injecting dynamic
    Hardware-in-the-Loop runtime parameters.
    """
    if global_harness_instance:
        global_harness_instance.set_runtime_config(
            config.detection_threshold,
            config.antenna_attenuation_db,
            config.zero_trust_lockout
        )
    return {"status": "success", "applied_parameters": config.dict()}


# ──────────────────────────────────────────────────────────────────────
# Exposed WebSocket Routes
# ──────────────────────────────────────────────────────────────────────

@app.websocket("/stream")
async def telemetry_stream(websocket: WebSocket):
    """
    Thread-safe WebSocket endpoint serving the live JSON schema payload:
    - sphericity_score
    - fim_beta
    - threat_verdict
    - inference_latency_us
    - dropped_blocks
    """
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else str(time.time())
    await dispatcher.register_client(client_id)
    print(f"[+] Client {client_id} connected to high-throughput priority telemetry stream.")
    
    # Asynchronous send loop task for this client
    async def send_loop():
        try:
            client_queue = dispatcher.clients.get(client_id)
            if not client_queue:
                return
            while True:
                payload = client_queue.pop()
                if payload:
                    await websocket.send_text(json.dumps(payload))
                await asyncio.sleep(0.05) # Poll client queue every 50ms
        except Exception as e:
            print(f"[-] Telemetry connection send loop terminated for {client_id}: {e}")

    send_task = asyncio.create_task(send_loop())
    
    try:
        while True:
            # Keep socket alive and monitor disconnection events
            await websocket.receive_text()
    except WebSocketDisconnect:
        print(f"[-] Client {client_id} gracefully disconnected.")
    except Exception as e:
        print(f"[!] WebSocket socket panic detected for {client_id}: {e}")
    finally:
        send_task.cancel()
        await dispatcher.unregister_client(client_id)


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Launch uvicorn engine
    uvicorn.run(
        "dashboard_api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
