#!/usr/bin/env python3
"""
SpaceShield: Cryptographic License Key Verification Module.
Author: Principal Cloud-Native Embedded Systems Architect
Version: 1.0.0

Verifies cryptographically signed license tokens to enforce subscription models
for Tier 2 Commercial and Defense deployments, preventing unauthorized usage
of the Ground Station spatiotemporal processing pipeline.
"""

import os
import sys
import json
import base64
import hashlib
from datetime import datetime

# STQC Auditing Public Key Parameters
RSA_N = 0x7ac92795f74531d6927d97793319f5fec3e2b1dd22b595144de89dd7f40fffb396924d7960a31d97c5644e1423bffd9b31d11f74042bb6ec17b7ff280983f26d
RSA_E = 0x10001

DEFAULT_LICENSE_PATH = "compliance/license.lic"

def verify_license_token(token_str):
    """
    Decodes and cryptographically verifies the signature and expiration of a license token.
    
    Parameters:
      token_str (str): Base64-encoded JSON license token string.
      
    Returns:
      dict: Decoded payload dictionary if valid, None otherwise.
    """
    try:
        # 1. Base64 Decode the envelope
        envelope_data = base64.b64decode(token_str.strip()).decode('utf-8')
        envelope = json.loads(envelope_data)
        
        payload_b64 = envelope.get("payload")
        signature_hex = envelope.get("signature")
        
        if not payload_b64 or not signature_hex:
            print("[-] License Verification Error: Missing payload or signature in envelope.")
            return None
            
        # 2. Cryptographic Signature Verification (RSA Asymmetric Signature Check)
        # Re-compute the SHA-256 hash of the base64 payload
        h_bytes = hashlib.sha256(payload_b64.encode('utf-8')).digest()
        h_int_expected = int.from_bytes(h_bytes, byteorder='big')
        
        # Recover signed hash via RSA modular exponentiation
        sig_int = int(signature_hex, 16)
        h_int_recovered = pow(sig_int, RSA_E, RSA_N)
        
        if h_int_recovered != h_int_expected:
            print("[-] Cryptographic Error: License signature is invalid! Unauthorized modification detected.")
            return None
            
        # 3. Decode Payload Content
        payload_data = base64.b64decode(payload_b64).decode('utf-8')
        payload = json.loads(payload_data)
        
        # 4. Subscription/Expiration Validation
        expires_str = payload.get("expires")
        if not expires_str:
            print("[-] License Verification Error: Missing expiration date in payload.")
            return None
            
        expires_date = datetime.strptime(expires_str, "%Y-%m-%d")
        current_date = datetime.now()
        
        if current_date > expires_date:
            print(f"[-] Subscription Expired: License expired on {expires_str}. Current date: {current_date.strftime('%Y-%m-%d')}")
            return None
            
        return payload
        
    except Exception as e:
        print(f"[-] License Verification Exception: {e}")
        return None

def run_license_audit():
    """
    Checks for license validation via Environment Variable or local compliance file.
    
    Returns:
      bool: True if a valid license is present and verified, False otherwise.
    """
    print("=" * 80)
    print("             SPACESHIELD CRYPTOGRAPHIC SUBSCRIPTION LICENSE AUDITOR            ")
    print("=" * 80)
    
    # Check Environment Variable first
    license_token = os.environ.get("SPACESHIELD_LICENSE_KEY")
    source = "ENVIRONMENT (SPACESHIELD_LICENSE_KEY)"
    
    # Fallback to local file path
    if not license_token:
        # Check standard project root compliance folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        file_path = os.path.join(project_root, DEFAULT_LICENSE_PATH)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    license_token = f.read().strip()
                source = f"FILE ({DEFAULT_LICENSE_PATH})"
            except Exception as e:
                print(f"[-] Failed to read license file: {e}")
                
    if not license_token:
        print("[-] License Audit Failure: No license token found in environment or file.")
        print("    Please configure SPACESHIELD_LICENSE_KEY or place license.lic in compliance/.")
        print("=" * 80)
        return False
        
    print(f"[*] Audit Source: {source}")
    payload = verify_license_token(license_token)
    
    if payload:
        print("[+] LICENSE VALIDATED SUCCESSFULLY:")
        print(f"  • Customer Name:  {payload.get('customer')}")
        print(f"  • Tier Level:     {payload.get('tier')}")
        print(f"  • Issued Date:    {payload.get('issued')}")
        print(f"  • Expiration Date: {payload.get('expires')} [ACTIVE]")
        print("=" * 80)
        return True
    else:
        print("[-] LICENSE VALIDATION FAILED. Operation suspended.")
        print("=" * 80)
        return False

def main():
    success = run_license_audit()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
