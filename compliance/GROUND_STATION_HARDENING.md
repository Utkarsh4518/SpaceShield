# SpaceShield Ground Station Hardening and Configuration Guide

**Document Classification**: Restricted / Operational Security Standard  
**Regulatory Alignment**: India Space Domain Cyber Security Mandates (2026)  
**Target Host Segment**: Mission-Critical Ground Station SDR & Telemetry Nodes  

---

## 1. Least-Privilege Telemetry Access (RBAC)

To safeguard spatiotemporal array statistics and real-time threat telemetry, ground station nodes must enforce Role-Based Access Control (RBAC) across both administrative and interactive telemetry layers.

### 1.1 RBAC Role Matrix
We define three explicit personnel tiers for interacting with the live ANSI Head-Up Display (HUD) and underlying metrics arrays:

| Role Identifier | Role Name | Authorized Assets | Scope of Permissions |
| :--- | :--- | :--- | :--- |
| **ROLE_INGEST_SUP** | Ingestion Supervisor | [binary_file_player.py](file:///c:/Users/Utkarsh/Desktop/SpaceShield/src/binary_file_player.py), raw SDR capture arrays | Configures hardware sampling clocks ($f_s = 2.0$/$5.0$ MSPS), maps input capture file pointers, and manages sample block ingestion parameters. No access to threat classification weights. |
| **ROLE_COMP_AUDITOR** | Security Compliance Auditor | [verify_log_integrity.py](file:///c:/Users/Utkarsh/Desktop/SpaceShield/src/verify_log_integrity.py), [spaceshield_180day_security.log](file:///c:/Users/Utkarsh/Desktop/SpaceShield/data/spaceshield_180day_security.log) | Reads WORM compliance logs, runs cryptographic ledger integrity audits, and handles reporting outputs to regulatory bodies. |
| **ROLE_SYS_OPS** | System Operations Engineer | [spatial_hardware_harness.py](file:///c:/Users/Utkarsh/Desktop/SpaceShield/src/spatial_hardware_harness.py), HUD Dashboard, Engine | Launches execution pipelines, monitors real-time Bartlett sphericity and METR statistics, reviews threat alerts, and manages worker threads. Restricted to read-only views of core logging directories. |

### 1.2 Access Enforcement Controls
*   **System Group Boundaries**: Ground station operating systems must maintain segregated local groups corresponding to the three tiers: `spaceshield-ingest`, `spaceshield-audit`, and `spaceshield-ops`.
*   **Process Permissions**: The [spatial_hardware_harness.py](file:///c:/Users/Utkarsh/Desktop/SpaceShield/src/spatial_hardware_harness.py) executable must run under a service account with minimal filesystem privileges, specifically restricted from writing outside `compliance/` and `data/` directories.

---

## 2. HUD & Dashboard Session Hardening

Interactive console displays utilizing virtual terminal escape sequences are susceptible to session hijacking and unauthorized command execution. The following protocols harden console and gRPC sessions:

### 2.1 Multi-Factor Token-Based Authentication (CLI/gRPC)
*   **Remote Console Access**: Direct SSH access to the terminal displaying the live HUD must require multi-factor authentication (MFA) utilizing FIPS 140-3 hardware security tokens (e.g., YubiKey) bound to SSH certificate authority keys.
*   **gRPC Session Hardening**: Any remote monitoring adapter exposing metrics from [spatial_hardware_harness.py](file:///c:/Users/Utkarsh/Desktop/SpaceShield/src/spatial_hardware_harness.py) must utilize gRPC over TLS (gRPCS) with Mutual TLS (mTLS) enabled. Connections must validate client certificates issued by the Space Segment Certificate Authority (SSCA).
*   **Token Expiration**: CLI execution tokens for diagnostic runs are restricted to a maximum time-to-live (TTL) of **15 minutes**, requiring automated session re-authentication.

---

## 3. Cryptographic Storage Enforcement (WORM)

To guarantee that forensic audit trails remain completely immune to post-compromise ledger wiping or log alteration, SpaceShield enforces a strict Write-Once-Read-Many (WORM) storage architecture for [spaceshield_180day_security.log](file:///c:/Users/Utkarsh/Desktop/SpaceShield/data/spaceshield_180day_security.log).

### 3.1 Cryptographic Hash-Chaining (Blockchain-Lite)
Every security event logged by the harness is hashed using SHA-256 and cryptographically chained to the preceding entry:
$$\text{Hash}_k = \text{SHA-256}(\text{Serialized\_Record}_k \mathbin{\Vert} \text{Hash}_{k-1})$$
Any deletion, insertion, or modification of historical records breaks the hash chain, immediately alerting offline validation audits.

### 3.2 System-Level WORM Locking
The log file must be configured with append-only system attributes immediately after creation to prevent deletion or truncation:

#### Linux Segment Deployments:
Enforce the append-only attribute (`+a`) using the ext4/xfs filesystem controls:
```bash
# Set write permissions to root only, and lock file as append-only
chown root:spaceshield-audit /data/spaceshield_180day_security.log
chmod 660 /data/spaceshield_180day_security.log
chattr +a /data/spaceshield_180day_security.log
```

#### Windows Segment Deployments (PowerShell):
Disable inheritance and strip "Write Data/Add File" and "Delete" permissions for general users, permitting only the system service account to append:
```powershell
$Path = "C:\Users\Utkarsh\Desktop\SpaceShield\data\spaceshield_180day_security.log"
$Acl = Get-Acl $Path
$Acl.SetAccessRuleProtection($True, $True) # Remove inheritance
# Configure Append-Only (CreateFiles/AppendData) for service account, remove Write/Delete
$Rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    "spaceshield-service",
    "CreateFiles,AppendData,Read",
    "Allow"
)
$Acl.AddAccessRule($Rule)
Set-Acl $Path $Acl
```

---

## 4. Air-Gapped Compliance Verification

In the event of a suspected security incident, regulatory auditors must perform verification offline in an isolated, air-gapped environment to prevent compromised hosts from spoofing validation checks.

### Operational Verification Standard (Step-by-Step)

1.  **Auditing Media Preparation**:
    *   Initialize a secure, write-blocked external storage medium (e.g., a hardware-encrypted USB drive).
    *   Clone [verify_log_integrity.py](file:///c:/Users/Utkarsh/Desktop/SpaceShield/src/verify_log_integrity.py) and copy the log file [spaceshield_180day_security.log](file:///c:/Users/Utkarsh/Desktop/SpaceShield/data/spaceshield_180day_security.log) from the target ground station node onto the external medium.

2.  **Environment Isolation**:
    *   Power on an audit workstation that is physically disconnected from all networks (Ethernet, Wi-Fi, Bluetooth). Ensure it is running a verified, secure OS image.
    *   Connect the write-blocked auditing media.

3.  **Auditor Validation Execution**:
    *   Copy the validation scripts and log logs to the local scratch space of the air-gapped machine.
    *   Execute the verification wrapper:
        ```bash
        python verify_log_integrity.py --log-file spaceshield_180day_security.log
        ```

4.  **Verifying Cryptographic Ledger Integrity**:
    *   Review the output checks. Confirm that the script returns:
        `AUDIT SUMMARY: Total Log Lines Scanned: N | Integrity Violations: 0 | [+] RESULT: SECURE & COMPLIANT`
    *   If the script reports a mismatch error at a specific entry number (e.g., `[FAIL] Hash mismatch at Entry #402`), immediately extract that timestamp and cross-reference active telemetry frames to locate the post-compromise altering window.

5.  **Audit Sign-off**:
    *   Wipe the air-gapped machine's local RAM and temp space.
    *   Generate a signed, cryptographic audit report detailing scanned ranges and verification results to satisfy compliance standards.
