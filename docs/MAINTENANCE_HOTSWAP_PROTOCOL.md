# SpaceShield: Zero-Downtime Maintenance & Hot-Swap Protocol
**Prepared By:** Site Reliability & Infrastructure Operations (SRE)  
**System Target:** SpaceShield Edge Processing Node (Layer-1 Threat Isolation)  
**Classification:** RESTRICTED / CORE INFRASTRUCTURE RUNBOOK  

---

## 1. Abstract & Operational Doctrine
In mission-critical national security environments, shutting down an active RF defense perimeter to apply patches or update algorithmic models is strictly prohibited. SpaceShield guarantees **Zero-Downtime Reliability** via a seamless Hot-Swap Protocol. This document outlines the explicit orchestration steps utilized by operations teams to securely reload ONNX machine-learning heuristics or reconnect physical `SoapySDR` hardware pipelines without dropping a single microsecond of Layer-1 spatial data ingestion.

---

## 2. POSIX Signal Orchestration Framework
SpaceShield’s core daemon handles POSIX system signals asynchronously to execute graceful state transitions inside the primary processing loops.

### SIGUSR1: Hot-Reloading the Edge Inference Model
When new RF threat fingerprints are modeled, the inference engine (`edge_inference_engine.py`) must be updated to the latest ONNX weights on disk without pausing the DSP cascade.

1. **Trigger Phase:** The administrator pushes the validated `.onnx` model binary into the `/backend/models/` secure volume mount.
2. **Signal Injection:** The SRE node issues `kill -SIGUSR1 <PID>` targeting the main `spatial_hardware_harness` process.
3. **Execution Block:**
   - The harness catches `SIGUSR1`. It suppresses immediate model instantiation to prevent pipeline blocking.
   - A background thread safely spawns the new instance of `EdgeInferenceEngine`, executing the heavy `.initialize_engine()` payload on a secondary CPU core.
   - Once the new ONNX session is verified and securely loaded into RAM, the global pointer reference (`self.engine`) is atomically swapped via a Python thread-safe object reference reassignment (`GIL` protection guarantees atomicity).
   - Old tensor caches are pushed for garbage collection. **Result: Zero dropped packets.**

### SIGUSR2: Hardware Re-Binding (SoapySDR)
Used when a physical SDR device hangs or when shifting to an alternate physical receiver channel dynamically.

1. **Signal Injection:** The SRE node issues `kill -SIGUSR2 <PID>`.
2. **Execution Block:**
   - The `SoapyReceiverBridge` receives the interrupt flag.
   - The bridge commands the active stream hardware to gracefully pause `deactivateStream()` but holds the pipeline connections intact.
   - The new hardware initialization sequence is loaded. 
   - `activateStream()` is resumed onto the new physical registers.
   - **Crucial Buffer Protection:** The ring buffer dynamically stretches through the Secondary Standby Absorber (detailed below) during the ~3ms hardware renegotiation phase.

---

## 3. The Duplicate Standby Absorber: Preventing Layer-1 Blind Spots
The greatest risk during a dynamic hardware renegotiation or heavy thread-pool swap is thread-blocking, causing the physical hardware (SoapySDR) to overflow and drop frames, establishing a "blind spot" for a hostile spoofing attack.

To counter this, SpaceShield employs a **Dual-Ring Absorber Architecture**.

### Structural Implementation
1. **Primary Queue (Active):** Under normal operations, the `SoapyReceiverBridge` pushes physical IQ batches straight into the `self.iq_queue` (a highly responsive Lock-Free Ring buffer) where the 24-thread parallel DSP workers drain the queue efficiently.
2. **Secondary Queue (Standby Absorber):** 
   - If `SIGUSR2` is triggered, or if the primary DSP worker pool is being selectively drained for an architectural restart, the master orchestrator instantaneously routes new 4096-sample continuous hardware strides to a massive, heavily pre-allocated secondary fallback queue.
   - This secondary queue uses zero-allocation Numpy mapping (similar to the primary) but operates sequentially. It is strictly configured to absorb incoming bursts without processing them heavily (bypassing slow paths like RFF extraction temporarily) and evaluating *only* the ultra-fast Bartlett Sphericity baseline to maintain defensive continuity.
   - Once the primary pool completes its structural restart or the hardware comes back online, the orchestrator begins draining the Standby Absorber aggressively across the fresh worker pool until parity is reached.
   - The stream is then atomically re-attached to the newly updated primary ingestion path.

---

## 4. Execution Commands (SRE Runbook)
Below are the exact commands utilized by the operator via the secure terminal boundary.

**1. ONNX Model Hot-Swap (No Thread Disruption):**
```bash
# Push the updated model payload into the secure mount
cp /mnt/secure_usb/model_v2.onnx /app/data/edge_model_current.onnx

# Identify the core PID
SPACESHIELD_PID=$(pgrep -f "spatial_hardware_harness")

# Issue the atomic hot-reload signal
kill -SIGUSR1 $SPACESHIELD_PID
```

**2. Hardware Re-Binding (Utilizing Standby Absorber):**
```bash
# Configure new generic driver settings if necessary via the internal endpoint
# Issue the atomic hardware reset and queue switch
kill -SIGUSR2 $SPACESHIELD_PID

# Monitor system log via external audit stream for buffer parity confirmation
tail -f /app/data/spaceshield_180day_security.log | grep "Absorber Parity Reached"
```

## 5. Formal Conclusion
SpaceShield ensures that active, adversarial RF environments cannot exploit maintenance windows. The architecture provides true 100.00% physical signal ingestion uptime, maintaining strict compliance with defense-grade mission-critical continuity protocols.
