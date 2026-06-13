# SpaceShield: Hardware-in-the-Loop (HIL) Execution Playbook
**Prepared By:** Site Reliability & Infrastructure Operations (SRE)  
**Classification:** RESTRICTED / CORE INFRASTRUCTURE RUNBOOK  
**Objective:** End-to-End Validation Protocol for the SpaceShield Air-Gapped EW Array

---

## 1. Phase I: Physical RF Cabling Topography
To execute the physical Layer-1 spatial matrices with sub-millisecond precision, the 4-channel antenna topography must be physically disciplined to an external atomic or OCXO baseline.

1. **Geometry Layout:** Deploy the 4-element phase-coherent antenna array in a Uniform Linear Array (ULA) configuration with precise $\lambda/2$ element spacing calibrated for the target NavIC L5 operating carrier frequency (1176.45 MHz).
2. **Timing Discipline:** 
   - Connect the external **10MHz Reference Clock** (10 MHz OUT) directly into the `10 MHz REF IN` ports across all utilized Software-Defined Radios (SDRs).
   - Feed the active **1-PPS (Pulse Per Second)** synchronization line into the `PPS IN` ports.
3. **Cabling Topology:** Ensure all 4 coaxial RF cables connecting the antenna elements to the low-noise amplifiers (LNAs) and subsequent SDR receiver channels are exact matched-length precision cables. Even millimeter discrepancies in phase length translate to artificial spatial steering vector corruption.
4. **Environment Boot:** Initiate the core orchestration daemon targeting the active interface.
   ```bash
   python backend/src/spatial_hardware_harness.py --live --fs 5.0e6
   ```

## 2. Phase II: Spatial Calibration & Phase Optimization
Before declaring the system "Mission Ready," operators must actively compensate for the inescapable thermal drift and LNA hardware gain imbalances inherent to COTS RF hardware.

1. **Trigger Calibration Mode:** Ensure the array is pointed toward a "clean" zenith view with an unobstructed line-of-sight to an authentic NavIC GEO/GSO satellite.
2. **Engage Optimizer:** The system will dynamically funnel multi-channel baseband matrices (`X_raw`) into the zero-allocation `PhaseCoherenceOptimizer`.
3. **Observation:** Monitor the terminal output. The engine automatically locks onto `Channel 0` as the fixed complex phase reference (1.0 + 0j). It will apply an exponential moving average (EMA) across 50 iterations to dynamically build the `correction_coeffs` matrix.
4. **Verification:** Validate that the sub-microsecond in-place multiplication routine (`X *= self.correction_coeffs`) actively drives the measured inter-channel baseline phase offsets (e.g. `Ch1 vs Ch0`) toward mathematically perfect zero radians prior to engagement.

## 3. Phase III: Real-Time Threat Injection Profiling
Once baseline parity is verified, invoke the live testing suites to assault the active system. Execute the automated simulator to blast synthetic anomalies directly into the `secure_ipc_bridge` ring buffers.

```bash
python tests/layer1_attack_simulator.py
```

### Telemetry & HUD Profiling:
- **Mobile-Responsive HUD:** Open the frontend dashboard on a monitoring tablet. As the simulation ramps injection amplitudes, observe the Threat Verdict transition fluidly from `CLEAR` to `JAMMING` or `SPOOFING`.
- **Prometheus Enterprise Integration:**
  - View the **`spaceshield_svd_latency_us`** Histogram. Even under heavy multi-vector spoofing flooding, the equalizer engine must execute and log sub-50µs execution brackets consistently.
  - Monitor the **`spaceshield_glrt_sphericity_ratio`** Gauge. Watch it mathematically breach the absolute `50.17` threshold precisely when the drag-off vector asserts spatial coherence across the array matrix.
  - Monitor the **`spaceshield_soapy_sdr_overflow_total`** Counter. Validate that the Dual-Ring POSIX Standby Absorber absorbs all back-pressure, keeping this overflow metric strictly at `0`.

## 4. Phase IV: Cryptographic Incident & STQC Verification
Following a simulated assault, the system immediately drops into rigorous, tamper-evident forensic reporting to meet absolute compliance standards.

1. **Isolate the Secure Ledger:** Navigate to the hardened host-volume bound out of the Docker container.
   ```bash
   cd compliance/
   cat certin_incident_spoofing.json
   ```
2. **Audit Hash Linkages:** Manually parse or automate a sweep over the JSON incident records. Verify the `hash` and `previous_hash` nodes. In a compliant defense system, the SHA-256 cryptographic chain must be 100% mathematically continuous, proving that no individual log was dropped, re-ordered, or maliciously purged by an external threat actor.
3. **STQC Sign-Off:** Confirm the system correctly mapped the physical amplitude trigger boundaries (e.g., Amplitude: 0.1900) into immutable storage with zero lock-contention warnings (as validated by `tests/ledger_stress_tester.py`). Sign off the final validation ledger for immediate operational production scaling.
