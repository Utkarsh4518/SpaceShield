# NON-BINDING LETTER OF INTENT (LOI) & MEMORANDUM OF UNDERSTANDING (MOU)
## COOPERATIVE PILOT EVALUATION AND COMMERCIAL TRANSITION AGREEMENT
**Document Reference:** LOI-SS-2026-PILOT-TMPL  
**Date of Template Issue:** June 10, 2026  

---

This **Letter of Intent and Memorandum of Understanding** (hereinafter referred to as the "LOI" or "MOU") constitutes a framework for a collaborative pilot program and is entered into by and between:

1. **SPACESHIELD SECURITY SOLUTIONS PRIVATE LIMITED** (hereinafter referred to as the **"Disclosing Party"** or **"SpaceShield"**), a deep-tech space cyber-defense enterprise specializing in spatiotemporal array processing and physical-layer threat detection; and
2. **THE PRIVATE COOPERATIVE SIGNATORY** (hereinafter referred to as the **"Operator"**), an emerging private SatCom Operator, Ground-Station-as-a-Service (GSaaS) Platform, or Critical Infrastructure Provider.

SpaceShield and the Operator may hereinafter be referred to individually as a **"Party"** and collectively as the **"Parties."**

---

### RECITALS

**WHEREAS**, the Operator manages critical ground-segment satellite reception infrastructure subject to regional signal disruptions, jamming, and spoofing vulnerabilities; and

**WHEREAS**, SpaceShield has developed a software-defined, multi-antenna Layer-1 signal integrity defense agent capable of real-time spatial calibration, detection of spatiotemporal anomalies, and machine learning feature inference (the "SpaceShield Engine"); and

**WHEREAS**, the Parties desire to establish a non-binding framework to conduct an on-site pilot evaluation of the SpaceShield Engine within the Operator’s receiver processing path to validate performance metrics and establish a commercial pathway.

**NOW, THEREFORE, the Parties align on the following terms and objectives:**

---

### SECTION 1: STATEMENT OF SHARED INTENT

1. **Pilot Objective**: The Parties intend to execute a collaborative, sandboxed engineering pilot to integrate and evaluate the SpaceShield software-defined Layer-1 signal defense agent directly within the Operator’s ground segment receiver loop.
2. **Threat Monitoring Focus**: The pilot shall evaluate the SpaceShield Engine’s capabilities in identifying, classifying, and alerting against:
   - **Regional Electronic Warfare (EW)** and wideband noise jamming;
   - **Coherent and Non-Coherent Spoofing Threats** targeting satellite navigation and communication telemetry links;
   - **Localized In-Band Jamming Anisotropy** and antenna array impairments.
3. **Operational Alignment**: The SpaceShield Engine will sit as a passive or active front-end processor positioned between the Software-Defined Radio (SDR) / digitizer ingestion layer and the downstream demodulation and processing blocks.

---

### SECTION 2: TECHNICAL SCOPE OF PILOT EVALUATION

The pilot validation program shall be assessed against a strict set of Key Performance Indicators (KPIs) based on verified physical-layer benchmarks. The Operator will verify that the integrated agent meets the following engineering thresholds under continuous workload simulation:

1. **Ingestion Processing and Throughput**:
   - The SpaceShield Engine must ingest and unpack complex I/Q data packets at a continuous physical sampling clock rate of **$2.0\text{ MSPS}$** (Megasamples Per Second) per channel across a 4-antenna receiver array configuration.
   - **Zero Block Drops**: The ingestion queue must sustain zero packet or block loss during continuous operation, validating the efficiency of the memory recycling pool and CPU-bound thread scheduling.
2. **Calibration and Processing Latency**:
   - **Multi-Antenna Calibration**: The Blind Phase-Coherence Calibration solver (EVD/SVD) must estimate gain and phase cable offsets within a sub-millisecond execution envelope (target $\tau_{\text{calib}} \le 1.0\text{ ms}$).
   - **Real-Time Equalization**: The fast-path broadcast equalization layer must execute in the microsecond range (target $\tau_{\text{equalize}} \le 100\text{ µs}$) per $8192$-sample packet to fit comfortably within the $4.096\text{ ms}$ processing deadline.
3. **Edge Tensor Inference**:
   - The compiled FP16 half-precision ONNX classifier graph must execute classification inference within a sub-millisecond sanity threshold (target $\tau_{\text{inference}} \le 1.0\text{ ms}$ on CPU/GPU hardware), minimizing overall tracking-loop group delay.

---

### SECTION 3: DATA RESIDENCY AND LOG SOVEREIGNTY ALIGNMENT

To satisfy strict sovereign space asset security directives, the evaluation deployment will adhere to the following data storage and auditing standards:

1. **Indian Space Hardening & CERT-In Compliance**:
   - All logging operations generated during the pilot will run locally within the Operator’s secure ground station domain.
   - **WORM Log Chaining**: The logging subsystem will generate Write-Once-Read-Many (WORM) security logs. Each incident entry will be cryptographically chained to its predecessor using SHA-256 hashes to guarantee historical immutability.
2. **Log Verification**: The Operator will have access to SpaceShield’s validation wrappers (`verify_log_integrity.py`) to independently verify the chain. Any single-bit tampering or payload deviation must trigger a termination code `1`, satisfying the **CERT-In 2026 Space Cyber Security Framework Guidelines**.
3. **Data Retention**: All raw digital I/Q recordings and telemetry alerts generated during the pilot remain under the exclusive physical control of the Operator, with no external phone-home telemetry transmission unless explicitly authorized in writing.

---

### SECTION 4: PILOT TIMELINE & COMMERCIAL TRANSITION OPTION

1. **Phase I: Sandbox Audit (Days 1–30)**:
   - SpaceShield will provide the Operator with a 30-day, zero-cost subscription license token enabling full features within the ground station container environment.
   - The Operator will deploy the containerized microservice and verify the execution KPIs defined in Section 2 using simulated and recorded over-the-air (OTA) captures.
2. **Phase II: Commercial Transition Trigger**:
   - Upon successful verification of the Section 2 KPIs at the end of the 30-day sandbox evaluation, the Operator shall have the option to transition to a paid Tier 2 production subscription.
3. **Tiered Commercial License Architecture**:
   - Commercial licenses will scale under a tiered Annual Software License structure, determined by the Operator's satellite constellation size, throughput bandwidth, and geographic deployment scale:
     
| License Tier | constellation Range | Ingestion Cap (Per GS) | SLA Support Level | Annual License Fee |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Emerging Operator)** | 1 – 3 Active Satellites | Up to $2.0\text{ MSPS}$ | Business Hours (Email/Slack) | On Request |
| **Tier 2 (Defense / Commercial)** | 4 – 12 Active Satellites | Up to $5.0\text{ MSPS}$ | 24/7/365 Critical Response | $XX,XXX / Year |
| **Tier 3 (Constellation Core)** | 13+ Satellites / GSaaS | Unlimited | Dedicated Mission Team | Custom Enterprise |

---

### SECTION 5: INTELLECTUAL PROPERTY & CONFIDENTIALITY

1. **IP Ownership**: SpaceShield retains all rights, titles, and interests in and to the SpaceShield Engine, source code, ONNX classifier models, and associated documentation. No intellectual property rights are transferred to the Operator under this pilot.
2. **Confidential Information**: The Parties agree that all technical data, calibration methodologies, algorithmic structures, and results of the pilot evaluation constitute Confidential Information and shall not be disclosed to any third party without prior written consent.

---

### SECTION 6: NON-BINDING NATURE & GOVERNING LAW

1. **Non-Binding Framework**: With the exception of **Section 3 (Data Residency)**, **Section 5 (Intellectual Property & Confidentiality)**, and **Section 6 (Non-Binding Nature)**, which are binding obligations upon signature, this document is a statement of intent only. Neither Party is legally obligated to enter into a commercial contract or license agreement as a result of this LOI/MOU.
2. **Governing Law**: This LOI/MOU and any subsequent definitive agreements shall be governed by, construed, and enforced in accordance with the laws of the Republic of India.

---

**IN WITNESS WHEREOF, the Parties hereto have executed this Letter of Intent as of the date first written below.**

```
For: SPACESHIELD SECURITY SOLUTIONS PVT. LTD.   For: [THE PRIVATE COOPERATIVE SIGNATORY]


_____________________________________________   _____________________________________________
Name:                                           Name:
Title:                                          Title:
Date:                                           Date:
```
