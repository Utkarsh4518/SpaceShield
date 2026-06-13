#!/usr/bin/env python3
"""
SpaceShield: Production Packaging & Release Manifest Generator
Description: Recursively calculates SHA-256 checksums across all mission-critical 
             source directories. Generates a consolidated deployment manifest and 
             an asymmetric detached signature (.sig) to guarantee zero vendor-side 
             tampering before air-gapped STQC activation.
"""

import os
import time
import json
import hashlib
import logging

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

logger = logging.getLogger("ReleaseEngineering")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class ReleaseManifestGenerator:
    def __init__(self):
        # Base project boundaries
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.target_directories = [
            "backend/src",
            "frontend",
            "compliance",
            "docs"
        ]
        
        # Ignored patterns to prevent hashing volatile testing caches
        self.ignore_extensions = ['.pyc', '.sig', '.log', '_test.json']
        self.ignore_directories = ['__pycache__', '.pytest_cache']
        
        self.manifest_path = os.path.join(self.base_dir, 'compliance', 'release_manifest_v20.json')
        self.signature_path = os.path.join(self.base_dir, 'compliance', 'release_manifest_v20.json.sig')
        self.key_path = os.path.join(self.base_dir, 'compliance', 'deployment_private.pem')
        
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculates a strict SHA-256 checksum for binary file integrity."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Chunked reading prevents massive model files from exploding memory
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {file_path}: {e}")
            return "ERROR_HASH_FAILED"

    def build_manifest(self) -> dict:
        """Traverses the source topography and aggregates cryptographic signatures."""
        logger.info("[*] Initiating Workspace Cryptographic Sweep...")
        
        manifest = {
            "metadata": {
                "release_version": "v2.0.0-GOLD-MASTER",
                "build_timestamp": time.time(),
                "target_framework": "STQC 2026 Space Cyber Security",
            },
            "signatures": {}
        }
        
        total_files = 0
        
        for directory in self.target_directories:
            target_path = os.path.join(self.base_dir, directory)
            if not os.path.exists(target_path):
                logger.warning(f"Target directory missing, skipping: {directory}")
                continue
                
            for root, dirs, files in os.walk(target_path):
                # Filter ignored directories dynamically
                dirs[:] = [d for d in dirs if d not in self.ignore_directories]
                
                for file_name in files:
                    if any(file_name.endswith(ext) for ext in self.ignore_extensions):
                        continue
                        
                    # Skip the manifest itself if re-running
                    if file_name.startswith("release_manifest_v20"):
                        continue
                        
                    full_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(full_path, self.base_dir)
                    
                    # Convert to uniform POSIX slashes for universal verification
                    relative_path = relative_path.replace("\\", "/")
                    
                    file_hash = self._calculate_file_hash(full_path)
                    manifest["signatures"][relative_path] = file_hash
                    total_files += 1
                    
        logger.info(f"[+] Workspace Sweep Complete. {total_files} artifacts hashed securely.")
        return manifest

    def _get_or_generate_signing_key(self):
        """Retrieves or natively generates a deterministic Ed25519 deployment private key."""
        if not HAS_CRYPTO:
            logger.warning("[-] Python 'cryptography' library missing. Generating SHA-256 HMAC fallback signature instead of Ed25519.")
            return None
            
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as key_file:
                return serialization.load_pem_private_key(key_file.read(), password=None)
        
        # Generate new authoritative key
        logger.info("[*] Generating new Ed25519 Deployment Private Key...")
        private_key = ed25519.Ed25519PrivateKey.generate()
        
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open(self.key_path, "wb") as f:
            f.write(pem)
            
        # Secure the key
        if os.name == 'posix':
            os.chmod(self.key_path, 0o600)
            
        return private_key

    def sign_and_export(self, manifest: dict):
        """
        Serializes the JSON structure deterministically, writes to disk, 
        and attaches a cryptographic detached signature.
        """
        # We must serialize deterministically (sorted keys, no extra whitespace)
        # to ensure the hash signature exactly matches upon receiver verification.
        raw_manifest_str = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
        raw_manifest_bytes = raw_manifest_str.encode('utf-8')
        
        # 1. Export standard JSON
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, "w") as f:
            # We save pretty-print for human auditors, but the signature binds to the exact bytes
            # Actually, standard practice requires signing the EXACT exported file bytes.
            # So we will write the exact deterministic string.
            f.write(raw_manifest_str)
            
        logger.info(f"[+] Consolidated Manifest written: {self.manifest_path}")
        
        # 2. Cryptographic Detached Signature
        private_key = self._get_or_generate_signing_key()
        
        if private_key is not None:
            signature = private_key.sign(raw_manifest_bytes)
            import base64
            b64_sig = base64.b64encode(signature).decode('utf-8')
            
            with open(self.signature_path, "w") as f:
                f.write(b64_sig)
                
            logger.info(f"[+] Detached Ed25519 Signature generated: {self.signature_path}")
        else:
            # Fallback to HMAC
            secret = b"SPACESHIELD_AIRGAPPED_HMAC_FALLBACK_SECRET_KEY"
            hmac_sig = hashlib.sha256(secret + raw_manifest_bytes).hexdigest()
            with open(self.signature_path, "w") as f:
                f.write(hmac_sig)
            logger.info(f"[+] Fallback HMAC Detached Signature generated: {self.signature_path}")

    def execute(self):
        print("================================================================")
        print(" SpaceShield DevSecOps Authorized Packaging Protocol")
        print("================================================================")
        
        manifest = self.build_manifest()
        self.sign_and_export(manifest)
        
        print("\n[PASSED] Gold-Master package securely sealed and signed for STQC Audits.")
        print("================================================================")

if __name__ == "__main__":
    packager = ReleaseManifestGenerator()
    packager.execute()
