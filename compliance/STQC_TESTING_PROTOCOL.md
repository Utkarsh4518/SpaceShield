# STQC TECHNICAL EVALUATION & COMPLIANCE PROTOCOL

**DOCUMENT CLASSIFICATION:** RESTRICTED / COMPLIANCE AUDIT  
**PROJECT NOMENCLATURE:** SpaceShield Layer-1 Ground Station Defense Agent  
**GOVERNING DIRECTIVE:** CERT-In 2026 Space Cyber Security Framework Guidelines  

## 1. PROBABILITY OF FALSE ALARM ($P_{\text{fa}}$) STRESS GUIDELINES

### 1.1 Objective
To empirically validate that the spatial Generalised Likelihood Ratio Test (GLRT) maintains strict adherence to the engineering design parameter limit of $P_{\text{fa}} = 10^{-7}$ under extreme thermal noise environments.

### 1.2 Execution Procedure
1. **Extended Baseline Provisioning**: Execute the data generation suite to synthesize a contiguous, multi-hour $\mathcal{H}_0$ (Nominal Propagation State) observation matrix.
   ```bash
   python src/generate_initial_dataset.py --duration 3600 --snr 5.0
   ```
2. **Thermal Noise Saturation**: Ensure the generation suite is parameterized to a heavily degraded Carrier-to-Noise Ratio (SNR = 5.0 dB) to saturate the Bartlett-corrected test statistic.
3. **Execution**: Stream the generated `nominal_reception.bin` directly into the `SpatialHardwareHarness` using the offline playback descriptor.
   ```bash
   python src/spatial_hardware_harness.py --playback-file data/nominal_reception.bin
   ```

### 1.3 Mathematical Verification Criteria
During execution, the examiner must monitor the **Sphericity LLR Stat** emitted to the CLI HUD or the WebSocket telemetry API.
*   **Threshold Limit ($\gamma$)**: The spatial detector utilizes a Bartlett-corrected log-likelihood evaluation. For a uniform linear array of $M=4$ channels across an $N=50$ temporal window with $P_{\text{fa}} = 10^{-7}$, the theoretical Chi-squared threshold limit is statically bounded at $\gamma = 50.17$.
*   **Success Metric**: Across $10^7$ discrete spatial evaluations, the recorded `max_sphericity` must strictly evaluate to $< 50.17$ during $\mathcal{H}_0$ operations. A single false positive alert during nominal signal profiling constitutes an evaluation failure.

---

## 2. INFERENCE LATENCY SANITY THRESHOLDS

### 2.1 Objective
To certify that the SpaceShield digital signal processing pipeline achieves bounded, real-time deterministic execution without triggering ingestion buffer overflows or tracking-loop packet drops.

### 2.2 Execution Procedure
1. Initialize the SpaceShield graphical telemetry dashboard via the FastAPI gateway.
   ```bash
   python src/dashboard_api.py
   ```
2. Open the control room HUD via `src/index.html`.
3. Locate the **Performance Profiling Grid** panel situated on the right-hand column of the dashboard matrix.

### 2.3 Verification Criteria
*   **Hardline Constraint**: At an ADC sampling clock of $f_s = 2.0\text{ MSPS}$ and a chunk size of $8192$ samples, the real-time buffer arrival window is strictly $4.096\text{ ms}$. 
*   **Micro-Readout Audit**: The examiner must observe the live metrics stream to confirm the following performance ceilings:
    *   **Blind Calibration Equalization**: Must resolve in $\approx 24.40\text{ \mu s}$.
    *   **Quantized FP16 ONNX Execution**: The Edge-AI latency readout must evaluate to $\approx 199.72\text{ \mu s}$ (under active GPU/TensorRT provisioning) or remain safely under $2000.00\text{ \mu s}$ under fallback CPU execution.
    *   **Queue Sequence Integrity**: The `dropped_blocks` metric must remain permanently locked at **0**. Any increment in this parameter proves critical pipeline backpressure and constitutes a real-time compliance failure.

---

## 3. WORM BLOCKCHAIN TAMPER-EVIDENCE TESTS

### 3.1 Objective
To prove total data sovereignty, forensic non-repudiation, and structural integrity of the Write-Once-Read-Many (WORM) incident logging chain against insider threats or persistent lateral movement.

### 3.2 Execution Procedure
1. Navigate to the secured compliance data vault: `data/spaceshield_180day_security.log`.
2. **Intentional Tampering Vector**: Utilizing a standard text editor or `sed` stream editor, isolate an arbitrary historical log entry.
3. Modify a single character inside the `sim_features` JSON blob (e.g., altering `rms_power_db` from `-15.2` to `-15.3`).
4. **Initiate Hash Verification**: Run the sovereign cryptography verification sequence.
   ```bash
   python src/verify_log_integrity.py
   ```

### 3.3 Verification Criteria
*   The script must actively recompute the SHA-256 cascading hashes block-by-block.
*   **Success Metric**: Upon encountering the manipulated line, the script must catch the structural hash payload mismatch, immediately halt execution, raise a **CRITICAL ALERT: CHAIN OF CUSTODY BROKEN** exception, and abort with `exit code 1`.

---

## 4. SUPPLY CHAIN TRACEABILITY VERIFICATION

### 4.1 Objective
To certify that the containerized application layer is free of untrusted third-party package dependencies, zero-day CVE vulnerabilities, or open-source license drifts.

### 4.2 Execution Procedure
1. Extract the active Python kernel dependency tree originating from the target edge deployment environment.
   ```bash
   pip freeze > current_env.txt
   ```
2. Locate the master software bill of materials (SBOM) manifest located at `compliance/sbom.json` (CycloneDX format).
3. **Traceability Mapping**: Audit the manifest components against the active `current_env.txt` output.

### 4.3 Verification Criteria
*   **Strict Adherence**: Every active package listed in the environment freeze (e.g., `numpy`, `scipy`, `onnxruntime`, `fastapi`, `uvicorn`) must possess a direct, matching node inside the `sbom.json` manifest.
*   **Constraint Check**: The STQC auditor must verify that no unauthorized, dynamically resolved dependencies exist. The container kernel must remain entirely air-gapped, pulling zero upstream dependencies at runtime. Any package discovered in the environment not explicitly declared in the CycloneDX structural layout constitutes a critical supply-chain breach.
