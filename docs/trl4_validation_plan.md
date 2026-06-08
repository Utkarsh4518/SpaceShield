# SpaceShield: TRL 4 Hardware-in-the-Loop (HIL) Validation Plan

This document outlines the 12-week experimental roadmap required to transition the SpaceShield RF Threat Intelligence Platform from **TRL 3 (Proof-of-Concept)** to **TRL 4 (Component/Process Validation in Laboratory Environment)**. 

Completing this validation plan is a prerequisite for defense procurement pathways, including the **iDEX Open Challenge (May 2026)**.

---

## 1. Laboratory Testbed & Hardware Setup

To validate our Edge-AI anomaly classification and physical-layer threat detection without broadcasting live signals (violating WPC rules), the HIL laboratory setup is constructed inside a shielded RF environment (Faraday cage).

```
                      +─────────────────────────────────────────+
                      │          RF Shielded Enclosure          │
                      │                                         │
+────────────────+    │  +─────────────────+                    │
│   NavIC/SDR    │    │  │  Co-Axial RF    │                    │
│ Signal Sim /   │───┼─►│  Attenuators &  │                    │
│ Spoofer Source │    │  │  Combiner Network│                    │
+────────────────+    │  +─────────────────+                    │
                      │           │                             │
                      │           ▼                             │
                      │  +─────────────────+    +────────────+  │
                      │  │  Ettus USRP B210│───►│ Edge-AI GPU│  │
                      │  │   (RF Frontend) │    │  (Jetson)  │  │
                      │  +─────────────────+    +────────────+  │
                      +─────────────────────────────────────────+
```

### Hardware Components
1. **RF Front-end**: Ettus USRP B210 Software-Defined Radio (dual-channel transceiver covering 70 MHz to 6 GHz, streaming up to 56 MSPS in real-time).
2. **Signal Source Generator**: RF signal simulator generating coherent NavIC L5 (1176.45 MHz) and S-band (2492.028 MHz) waveforms.
3. **RF Attack Emulator**: Second USRP device configured as a tactical broadband jammer or a coherent spoofer generating dynamic Doppler "drag-off" sweeps.
4. **Interconnection**: High-shielded SMA co-axial cabling, inline passive step attenuators, and RF signal combiners inside a shielded metal enclosure.
5. **Edge Processor**: NVIDIA Jetson Orin Nano (8GB) executing the containerized SpaceShield Edge-AI agent.

---

## 2. 12-Week HIL Validation Schedule

```
Week 1-2      Week 3-4      Week 5-6      Week 7-8      Week 9-10     Week 11-12
[ Setup ] ──► [ RFF ]   ──► [ GLR ]   ──► [ ITU ]   ──► [ Edge ]  ──► [ Audit ]
  Lab RF        Mixer         Doppler       I/N Limit     Orin GPU      iDEX Signoff
  USRP HIL      Signature     Drift         -6 dB Alert   Real-time     TRL-4 Report
```

### Phase 1: Testbed Integration & Baseline Calibration (Weeks 1-2)
- **Week 1: Physical Linkage & Shielding Calibration**
  - Install the Ettus USRP B210 and connect SMA cables within the shielded Faraday enclosure.
  - Verify baseline noise floors and measure thermal noise variance to establish the standard $\sigma^2$ baseline.
- **Week 2: NavIC Carrier Acquisition**
  - Stream synthesized clean NavIC L5 (1176.45 MHz) and S-band (2492.028 MHz) signals.
  - Calibrate the IQ capture pipeline in python, checking for packet drops at a target sampling rate of $fs = 5\text{ MSPS}$.

### Phase 2: RF Fingerprinting (RFF) Classifier Validation (Weeks 3-4)
- **Week 3: Device Imperfection Profiling**
  - Generate signals from 5 different SDR units (transmitters) to profile hardware imperfections (IQ gain imbalance, phase skew, DC leakage).
  - Train our baseline XGBoost and complex-valued CNN models on the extracted RFF dataset.
- **Week 4: RFF Classification Integrity Test**
  - Validate device fingerprinting accuracy under varying SNR levels (0 dB to 20 dB).
  - Verify that the CNN classifier meets the targeted validation threshold ($\ge 99\%$ accuracy under line-of-sight conditions).

### Phase 3: GLR Stability Anomaly Validation (Weeks 5-6)
- **Week 5: Drag-off Spoofing Emulation**
  - Deploy a spoofing transmitter to generate carrier-coherent signals with a slowly drifting frequency offset (from 0.1 Hz/s up to 10 Hz/s).
  - Measure the response of the receiver's tracking loop loop-filter outputs.
- **Week 6: Generalized Likelihood Ratio (GLR) Tuning**
  - Feed Doppler rate variance into the GLRT detector.
  - Tune the detection threshold $\gamma$ to ensure a false-alarm rate $P_{\text{fa}} = 10^{-7}$ is strictly maintained, while achieving $99.5\%$ probability of detection for drag-off sweeps exceeding $1.5\text{ Hz/s}$ drift rate.

### Phase 4: ITU-R M.1902-2 Limit Compliance Verification (Weeks 7-8)
- **Week 7: Noise-to-Interference Stress Tests**
  - Introduce broadband white noise jamming. Gradually ramp up the jammer power from $I/N = -20\text{ dB}$ up to $+30\text{ dB}$.
  - Map tracking lock loss margins.
- **Week 8: Protection Limit Alert Verification**
  - Verify that the SpaceShield software agent triggers a `VIOLATION` alert within $<10\text{ ms}$ of the $I/N$ ratio crossing the $-6\text{ dB}$ threshold.
  - Ensure the alert correctly maps to spectral flatness characteristics.

### Phase 5: Real-time Edge Deployment & Latency Audits (Weeks 9-10)
- **Week 9: NVIDIA Jetson Microservice Packaging**
  - Containerize the `rf_threat_simulator` into a Docker microservice optimized for the NVIDIA Jetson Orin Nano edge platform using TensorRT.
  - Profile the CPU/GPU memory footprint and power consumption.
- **Week 10: Real-time Ingestion Latency Benchmarks**
  - Run continuous, real-time loop tests over 24-hour periods.
  - Benchmark inline inspection latency: target average must remain sub-millisecond (e.g., $<0.8\text{ ms}$ processing time per IQ frame batch).

### Phase 6: Compliance Logs & Audit Signoff (Weeks 11-12)
- **Week 11: CERT-In compliance Integration & Forensics**
  - Integrate the WORM security log architecture. Perform vulnerability tests trying to overwrite the `spaceshield_180day_security.log`.
  - Validate that the 6-hour reporting JSON payload is correctly pushed to local directories upon alert triggers.
- **Week 12: Final TRL 4 Documentation & iDEX Review**
  - Compile the HIL test outputs, performance graphs, and compliance logs.
  - Submit the validation dossier to Indian defense auditing bodies for official iDEX TRL 4 certification.

---

## 3. Key TRL 4 Performance Pass Criteria

To successfully obtain TRL 4 sign-off, the physical testbed must satisfy:
1. **GLR Spoofing Alert Speed:** Time to detect a dynamic drag-off spoofing attack must be $\le 100\text{ ms}$.
2. **RFF Device Verification:** Probability of detecting an unauthorized transmitter based on phase noise and CFO profile must exceed $98.5\%$.
3. **ITU Threshold Reliability:** Alert on $I/N \ge -6\text{ dB}$ must exhibit zero false negatives.
4. **Log Tamper Prevention:** The secure WORM log must remain locked against unauthorized terminal modification.
