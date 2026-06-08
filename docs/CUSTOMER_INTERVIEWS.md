# SpaceShield Market Validation Questionnaires

This document contains the targeted, highly technical questionnaire scripts designed to guide SpaceShield's **30-to-50 structured customer interview campaign**. These questions are formulated to uncover hidden operational friction, integrate our software agent seamlessly into existing protocol stacks, and validate our engineering roadmap against real-world pain points.

---

## 1. Ground Station Operators & GSaaS Platforms

*   **Audience Profile:** Chief Technology Officers (CTOs), Infrastructure Engineers, and Ground Station Operations Managers.
*   **Focus Area:** Protocol stacks (VITA 49, CCSDS), visibility into Level 1 RF anomalies, interface points, and data latency tolerances.

### Interview Script & Question Bank

1.  **Ingestion & SDR Protocols:** When digitizing raw RF downlinks, which protocols are standard in your architecture (e.g., VITA 49.0/49.2, digital IF, or raw IQ streams over TCP/UDP)? What is your typical operational sampling rate (MSPS) and bit-depth configuration?
2.  **Level 1 RF Visibility:** What mechanisms or automated monitoring scripts do you currently run at the physical layer (Level 1) to identify signal anomalies, spectral noise, or carrier fluctuations before demodulation?
3.  **Anomalous Ingestion Routing:** If a receiver node experiences unexpected phase fluctuations or out-of-band energy spikes, does your control software automatically flag the packet stream, or does it continue streaming unvalidated frames to downstream customer networks?
4.  **Processing Latency Budgets:** What is the maximum latency budget (in milliseconds) allocated for real-time signal validation at the edge before demodulator frame processing delays become unacceptable?
5.  **Virtualization & Containerization:** To what extent is your ground station running virtualized software-defined infrastructure (e.g., Kubernetes containers at the edge, AWS Ground Station APIs) versus dedicated local FPGA/ASIC hardware cards?
6.  **Multi-Tenant RF Isolation:** In a multi-tenant Ground-Station-as-a-Service (GSaaS) model, how do you mathematically or physically isolate raw digitized streams to guarantee that interference on one receiver path doesn't compromise or leak into another client's virtual channel?
7.  **Ancillary Telemetry Availability:** Do your active tracking loops expose local receiver statistics (e.g., PLL/DLL loop-filter outputs, Carrier-to-Noise ratio ($C/N_0$) variations, Automatic Gain Control (AGC) levels) to external monitoring scripts?
8.  **SDR Integration Interfaces:** If you were to deploy an inline software agent (like SpaceShield) to inspect digitized I/Q streams, would it be easier to integrate as a pluggable UHD source block, a GNU Radio block, or a network proxy intercepting virtualized VITA 49 IP packets?
9.  **Anomalous Incident Workflows:** Once an RF anomaly or interference pattern is flagged, what automated workflow takes place? How are operators notified, and how long does it typically take to transition to an alternate receiver node?

---

## 2. Private SatCom Startups & Operators

*   **Audience Profile:** Operations Leads, RF System Architects, and Network Optimization Engineers.
*   **Focus Area:** SLA exposures during regional jamming, carrier lock loss responses, operational impacts of spoofing, and mitigation costs.

### Interview Script & Question Bank

1.  **SLA Financial Exposures:** How are your Service Level Agreements (SLAs) structured regarding link availability? What is the estimated hourly financial or operational impact if a ground station downlink experiences an outage due to localized L5 or S-band jamming?
2.  **Carrier Lock Loss Mechanisms:** When a remote ground terminal or receiver loses carrier lock (e.g., due to noise injection), what automated diagnostic routine runs to isolate whether the root cause is atmospheric fading, local blockage, or intentional jamming?
3.  **Spoofing Detection Realities:** If an adversary successfully executes a coherent, low-power Doppler drag-off spoofing attack that shifts your receiver's internal clocks without dropping the signal, how long would it take your current systems to notice the timing drift?
4.  **Mitigation CapEx vs. OpEx:** What has been your historical approach to mitigating RF threats? Have you evaluated multi-million dollar spatial filtering antenna arrays (CRPA), or are you actively looking for low-cost, software-defined solutions that run on standard SDRs?
5.  **Telemetry Demodulation Vulnerabilities:** Have you encountered scenarios where corrupted baseband packets or out-of-sequence frame numbers bypassed your receiver's physical demodulator and reached your central network database?
6.  **Interference Tracking History:** Can you describe a historical instance where a satellite terminal experienced signal degradation near border regions or industrial zones? How did you localize the interference, and how long did the triage process take?
7.  **Spectrum Health Dashboards:** What metrics do your network operations center (NOC) teams rely on to assess link health in real-time? Are they limited to post-demodulation bit-error rate (BER) and packet loss, or do they have real-time access to raw spectral flatness and $I/N$ ratios?
8.  **Compliance Reporting Overhead:** How much engineering effort is spent compiling incident logs after an RF anomaly? Would an automated compliance reporting generator (conforming to CERT-In or local space security standards) save significant overhead during post-mortem audits?
9.  **Fallback Ingress Routing:** When a primary satellite downlink is jammed, how do your systems coordinate routing traffic to alternative orbital paths (e.g., LEO-to-GEO handovers) or terrestrial fiber fallbacks, and what data loss is acceptable during the switchover window?

---

## 3. Defense Procurement Officers & System Integrators

*   **Audience Profile:** Security Directors, Defense Acquisition Managers, and Embedded System Security Architects.
*   **Focus Area:** MIL-STD hardening expectations, air-gapped system isolation rules, secure logging loops, and regulatory barriers.

### Interview Script & Question Bank

1.  **Standard Compliance (MIL-STD):** What specific physical-layer and environmental hardening specifications (e.g., MIL-STD-810H, MIL-STD-461G for electromagnetic compatibility) are mandatory for signal validation units entering your strategic ground networks?
2.  **Air-Gapped Operational Constraints:** Since strategic ground terminals must operate in completely air-gapped environments, how do you handle software updates, model retraining packages, or signature database updates without network access?
3.  **Audit Trail Cryptography:** What are your core requirements for secure logging and audit trails in defense installations? Do you mandate hardware security modules (HSM) or FIPS 140-3 validated cryptographic components to secure system event logs from root-level alteration?
4.  **Edge Compute Footprints:** When deploying threat classification software alongside receiver stations, what are the strict processing constraints (e.g., SWaP-C: Size, Weight, Power, and Cost) for hardware integration? Can you support GPU-accelerated co-processors like NVIDIA Jetson Orin or custom FPGAs?
5.  **False Alarm Thresholds:** In strategic military communications, what is your tolerance for false alarms? For example, if a signal processing anomaly detector has a false-alarm rate of $10^{-5}$ resulting in manual triage triggers, is that considered acceptable or operationally disruptive?
6.  **Signal Classification Frameworks:** Does your team prefer rule-based signal processing classifiers (like the Generalized Likelihood Ratio Test) because they are mathematically explainable, or are you ready to authorize deep learning models (CNNs/LSTMs) for RF fingerprinting if they provide higher classification accuracy?
7.  **Supply Chain Verification (SBOM):** What level of validation is required for external software libraries (e.g., CycloneDX SBOM, static code analysis, memory-safety audits) before a software package is authorized to run on national-security ground controllers?
8.  **Active Countermeasures vs. Passive Alerts:** Is your procurement focus strictly on passive threat intelligence (detecting and logging jamming/spoofing anomalies), or do you actively look for software that coordinates dynamic countermeasures like frequency hopping or adaptive spatial nulling?
9.  **Certification Bottlenecks:** What is the largest regulatory or testing bottleneck you face when trying to certify and deploy software updates to tactical edge receivers? How does the process change when the software involves cryptographic components or incident reporting modules?

---

## 4. Critical Infrastructure Managers (Aviation & Maritime)

*   **Audience Profile:** Port Operations Directors, Aviation Telecommunications Specialists, and GNSS Timing System Engineers.
*   **Focus Area:** Timing synchronization dependencies, history of tracking degradation, operational impact, and localized mitigation.

### Interview Script & Question Bank

1.  **GNSS Timing Dependency:** To what extent do your critical operational subsystems (e.g., power grid synchronization, maritime port scheduling, telecommunications cellular handovers) rely on GPS/NavIC timing pulses? What is the maximum timing drift (in microseconds) your systems can tolerate before failing?
2.  **GNSS Tracking Outage Incidents:** Can you detail a recent occurrence of GNSS signal degradation or timing anomalies at your facility? How long did the degradation last, and what were the cascading effects on downstream control platforms?
3.  **Local Spoofing Scenarios:** Have you observed instances of coordinate-shift spoofing (e.g., vessels appearing inland or aircraft reporting incorrect GPS altitudes) in your operations? How did your control operators identify the anomaly, and was it flagged automatically or manually?
4.  **Hardware Clock Backups:** What backup timing references (e.g., rubidium atomic clocks, oven-controlled crystal oscillators (OCXO), or PTP network feeds) are installed in your infrastructure, and how long can they maintain required precision during a prolonged GNSS denial event?
5.  **Local RF Jammer Vectors:** Are you concerned about localized, low-cost illegal jammers (e.g., personal privacy jammers used by delivery drivers near facilities) disrupting critical regional logistics, and how do you monitor for these low-power signals?
6.  **Regulatory Reporting Workflows:** When a GNSS anomaly or interference spike is detected, what local regulatory or safety agencies must be notified, and what telemetry data (e.g., location, frequency, signal strength) are you required to supply?
7.  **User Interface Preferences:** When an operator is managing active maritime cargo loading or air traffic operations, how should RF threat warnings be displayed? Do they require detailed spectral plots or simply a binary "Red/Green" channel reliability index integrated into their existing SCADA/GIS dashboard?
8.  **Demarcation of Responsibility:** When a timing clock drift or position leap occurs, how do your technical teams isolate whether the error is a satellite constellation anomaly (e.g., ephemeris error) versus a malicious local RF attack?
9.  **Retrofitting Feasibility:** Would it be feasible to deploy an inline SDR-based signal validation unit between your existing GNSS antennas and your master clock servers, or are you locked into proprietary antenna-receiver hardware stacks?
