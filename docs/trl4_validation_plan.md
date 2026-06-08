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
                      │  +─────────────────+                    │
+────────────────+    │  │  Co-Axial RF    │                    │
│   NavIC/SDR    │    │  │  Attenuators &  │                    │
│ Signal Sim /   │───┼─►│  Combiner Network│                    │
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

## 2. 12-Week HIL Validation Schedule (Phase 2)

```
Week 1-2      Week 3-4      Week 5-6      Week 7-8      Week 9-10     Week 11-12
[ Setup ] ──► [ RFF ]   ──► [ GLR ]   ──► [ ITU ]   ──► [ Edge ]  ──► [ Audit ]
  Lab RF        Mixer         Doppler       I/N Limit     Orin GPU      iDEX Signoff
  USRP HIL      Signature     Drift         -6 dB Alert   Real-time     TRL-4 Report
```

### Week 1: Physical Linkage & Shielding Calibration
*   **Engineering Focus:** RF path loss calibration, Faraday cage isolation measurement, and establishing thermal noise floor baseline ($\sigma^2$).
*   **Hardware/Software Integration:** Interconnecting Ettus USRP B210 TX/RX ports with SMA coaxial cables, attenuators (30 dB step), and combiner networks inside the shielding box. Interfacing with UHD drivers.
*   **Success Criteria/KPIs:** Achieve $>80\text{ dB}$ signal isolation envelope inside the Faraday cage. Establish nominal thermal noise floor at $-110\text{ dBm/Hz}$ without spurious leakage.

### Week 2: NavIC Carrier Ingestion & Streaming Optimization
*   **Engineering Focus:** IQ stream chunking, high-throughput buffer allocation, and thread-safe streaming to prevent packet loss.
*   **Hardware/Software Integration:** Configuring UHD source block to stream NavIC L5 (1176.45 MHz) and S-band (2492.028 MHz) at $5\text{ MSPS}$ sample rate. Tuning system socket buffers (`sysctl` network buffers) in Linux/Jetson.
*   **Success Criteria/KPIs:** Zero overflow (`O`) or underflow (`U`) packets over a 4-hour continuous streaming run. Maintain latency baseline under $5\text{ ms}$ for IQ buffer ingestion.

### Week 3: RF Fingerprinting (RFF) Dataset Gathering & Modeling
*   **Engineering Focus:** Compiling physical hardware signatures (DC offset, IQ gain imbalance, quadrature phase skew) from 5 distinct USRP/SDR units.
*   **Hardware/Software Integration:** Running Automated Feature Extraction pipeline to collect features (estimated CFO, phase noise, imbalance parameters). Formatting dataset for training XGBoost and Complex-Valued CNN models.
*   **Success Criteria/KPIs:** Collect $\ge 100,000$ distinct IQ bursts per SDR device. Achieve validation accuracy $\ge 95\%$ on offline training set.

### Week 4: RFF Classifier Optimization & Hardening
*   **Engineering Focus:** Hyperparameter tuning of RFF CNN classifier and testing model performance against low SNR environments.
*   **Hardware/Software Integration:** Converting trained PyTorch CNN and XGBoost models to ONNX formats and optimizing execution paths using TensorRT on the host.
*   **Success Criteria/KPIs:** Achieve CNN classification accuracy $\ge 99.0\%$ under line-of-sight conditions (SNR $\ge 15\text{ dB}$) and $\ge 98.0\%$ at low SNR levels (down to $3\text{ dB}$).

### Week 5: Coherent Spoofing & Doppler Drag-off Emulation
*   **Engineering Focus:** Emulating dynamic multi-generator spoofing scenarios (Doppler drag-off, power advantage, code-phase alignment).
*   **Hardware/Software Integration:** Integrating second USRP transmitter configured as a spoofer with tracking loop emulation software (GNU Radio NavIC tracking blocks).
*   **Success Criteria/KPIs:** Simulate successful pseudorange rate drag-off sweeps drifting from $0.1\text{ Hz/s}$ to $10\text{ Hz/s}$ without triggering standard receiver tracking lock loss initially.

### Week 6: Generalized Likelihood Ratio (GLR) Integration & Tuning
*   **Engineering Focus:** Tuning the GLRT anomaly detection statistic for NavIC GEO/GSO carrier stability.
*   **Hardware/Software Integration:** Interfacing tracking loop loop-filter frequency output with the GLR calculation block in Python.
*   **Success Criteria/KPIs:** Establish detection threshold $\gamma$ keeping False Alarm Rate $P_{fa} \le 10^{-7}$ while maintaining a Probability of Detection $P_d \ge 99.5\%$ for drag-off drift rates exceeding $1.5\text{ Hz/s}$.

### Week 7: ITU-R M.1902-2 Interference Mapping
*   **Engineering Focus:** High-power broadband white noise jamming and sweep jamming impact assessment on receiver tracking loops.
*   **Hardware/Software Integration:** Configuring RF combiner network to overlay broadband noise with legitimate NavIC signals. Monitoring carrier-to-noise ratio ($C/N_0$) drop-off in tracking blocks.
*   **Success Criteria/KPIs:** Map the exact tracking lock threshold drop-off curves from $I/N = -20\text{ dB}$ to $+30\text{ dB}$. Define clear boundaries for jamming state transitions.

### Week 8: Protection Limit Alerts & Feature Correlation
*   **Engineering Focus:** Optimizing response time for ITU-R alert triggers and correlating interference power with spectral flatness metrics (Wiener Entropy).
*   **Hardware/Software Integration:** Interfacing the Feature Extraction engine with the Threat Decision Matrix in `rf_threat_simulator.py`.
*   **Success Criteria/KPIs:** Time elapsed between $I/N$ crossing the $-6\text{ dB}$ threshold and warning alert generation must be under $10\text{ ms}$. Ensure spectral flatness drops below $0.5$ in presence of narrowband jamming.

### Week 9: Edge Microservice Packaging & Containerization
*   **Engineering Focus:** Code footprint reduction, containerization, and configuring system resource quotas.
*   **Hardware/Software Integration:** Containerizing SpaceShield Edge agent into a rootless Docker microservice. Optimizing ONNX runtime engine path on the NVIDIA Jetson Orin Nano GPU using CUDA/TensorRT execution providers.
*   **Success Criteria/KPIs:** Package container size kept under $1.2\text{ GB}$. Limit RAM utilization to $<2.0\text{ GB}$ and CPU load to $<30\%$ of available NVIDIA Orin cores.

### Week 10: Edge Latency Benchmarking & Stress Testing
*   **Engineering Focus:** Inline processing latency optimization, memory leak testing, and long-term stability profiling.
*   **Hardware/Software Integration:** Running continuous Loopback Hardware-in-the-Loop tests over 48 hours on Jetson Orin Nano.
*   **Success Criteria/KPIs:** Average inline inspection latency (IQ ingestion to threat verdict output) must remain under $1.0\text{ ms}$ per 1024-sample batch. No memory leaks detected over 48 hours of continuous operation.

### Week 11: CERT-In Compliance Log Auditing
*   **Engineering Focus:** Log security auditing, WORM state verification, and forensic hash chain integrity validation.
*   **Hardware/Software Integration:** Running vulnerability script tests trying to bypass WORM locks. Executing `verify_log_integrity.py` on logs generated during active stress testing.
*   **Success Criteria/KPIs:** Validate that all incident logs comply with the 6-hour reporting window. Enforce `chmod 0444` successfully, preventing log mutation or line-insertion attacks.

### Week 12: Final iDEX TRL 4 Review & Sign-off
*   **Engineering Focus:** Dossier compilation, verification results packaging, and certification demonstration preparation.
*   **Hardware/Software Integration:** Packaging system metrics, logs, and screenshots into an official auditing package. Preparing live lab demonstration scripts.
*   **Success Criteria/KPIs:** Complete compilation of TRL 4 validation report. Zero errors returned by audit test scripts. Secure approval dossier signoff for the iDEX delegation review.

---

## 3. Key TRL 4 Performance Pass Criteria

To successfully obtain TRL 4 sign-off, the physical testbed must satisfy:
1. **GLR Spoofing Alert Speed:** Time to detect a dynamic drag-off spoofing attack must be $\le 100\text{ ms}$.
2. **RFF Device Verification:** Probability of detecting an unauthorized transmitter based on phase noise and CFO profile must exceed $98.5\%$.
3. **ITU Threshold Reliability:** Alert on $I/N \ge -6\text{ dB}$ must exhibit zero false negatives.
4. **Log Tamper Prevention:** The secure WORM log must remain locked against unauthorized terminal modification.
