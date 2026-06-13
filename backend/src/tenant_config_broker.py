import json
import base64
import logging
from enum import Enum
from typing import Dict, Any, Optional

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logging.warning("cryptography package missing. Asymmetric validation will fallback to highly restricted stubs.")

# Configure Module Logger
logger = logging.getLogger("TenantConfigBroker")
logger.setLevel(logging.INFO)

class OperationalTier(Enum):
    DEFENSE_ADMIN = 1       # Complete array/DSP authority
    COMMERCIAL_OPERATOR = 2 # Telemetry access & logging toggles
    SYSTEM_AUDITOR = 3      # WORM ledger extraction only
    UNAUTHORIZED = 4

class TenantConfigBroker:
    """
    Air-Gapped Role-Based Configuration Broker.
    Implements tokenless asymmetric cryptographic validation for multi-tenant isolation
    without relying on external database connections or OAuth networks.
    """
    
    def __init__(self):
        # In a real air-gapped deployment, these public keys would be burned into 
        # ROM or loaded from an immutable hardware security module (HSM).
        # We simulate this via a pre-shared hardware trust registry.
        self._trust_registry: Dict[str, dict] = {}
        self._initialize_trust_registry()
        
    def _initialize_trust_registry(self):
        """Hard-codes the sovereign public key registry representing physical terminal identities."""
        if not HAS_CRYPTO:
            return
            
        # Example deterministic private keys for demonstration (DO NOT USE IN PROD)
        # In production, ONLY the public bytes are stored here.
        admin_priv = ed25519.Ed25519PrivateKey.from_private_bytes(b'0'*32)
        operator_priv = ed25519.Ed25519PrivateKey.from_private_bytes(b'1'*32)
        auditor_priv = ed25519.Ed25519PrivateKey.from_private_bytes(b'2'*32)
        
        self._trust_registry = {
            "terminal_alpha_admin": {
                "public_key": admin_priv.public_key(),
                "tier": OperationalTier.DEFENSE_ADMIN
            },
            "terminal_beta_operator": {
                "public_key": operator_priv.public_key(),
                "tier": OperationalTier.COMMERCIAL_OPERATOR
            },
            "terminal_gamma_auditor": {
                "public_key": auditor_priv.public_key(),
                "tier": OperationalTier.SYSTEM_AUDITOR
            }
        }

    def verify_request_signature(self, terminal_id: str, payload: str, signature_b64: str) -> OperationalTier:
        """
        Validates the asymmetric Ed25519 signature of the incoming JSON payload.
        Returns the operational tier bounded to the terminal ID.
        """
        if not HAS_CRYPTO:
            logger.error("Rejecting request: Cryptography library missing. System locked.")
            return OperationalTier.UNAUTHORIZED

        if terminal_id not in self._trust_registry:
            logger.warning(f"Unauthorized terminal ID rejected: {terminal_id}")
            return OperationalTier.UNAUTHORIZED
            
        try:
            signature_bytes = base64.b64decode(signature_b64)
            pub_key: ed25519.Ed25519PublicKey = self._trust_registry[terminal_id]["public_key"]
            
            # Verify the signature against the raw payload bytes
            pub_key.verify(signature_bytes, payload.encode('utf-8'))
            
            tier = self._trust_registry[terminal_id]["tier"]
            logger.info(f"Signature verified for {terminal_id}. Granted Tier: {tier.name}")
            return tier
            
        except InvalidSignature:
            logger.warning(f"Cryptographic signature mismatch for terminal {terminal_id}. REJECTED.")
            return OperationalTier.UNAUTHORIZED
        except Exception as e:
            logger.error(f"Malformed signature payload: {e}")
            return OperationalTier.UNAUTHORIZED

    def process_config_update(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main routing function intercepted from the /api/v1/config/update FastAPI endpoint.
        Validates the cryptographic envelope and enforces strict RBAC tier routing.
        """
        terminal_id = request_payload.get("terminal_id", "")
        signature_b64 = request_payload.get("signature", "")
        raw_json_str = request_payload.get("raw_payload", "{}")
        
        # 1. Cryptographic Authentication
        tier = self.verify_request_signature(terminal_id, raw_json_str, signature_b64)
        
        if tier == OperationalTier.UNAUTHORIZED:
            return {"status": "error", "reason": "UNAUTHORIZED_SIGNATURE_OR_TERMINAL"}
            
        # Parse inner payload safely after signature validation
        try:
            command_data = json.loads(raw_json_str)
        except json.JSONDecodeError:
            return {"status": "error", "reason": "MALFORMED_INNER_PAYLOAD"}
            
        action = command_data.get("action", "")
        
        # 2. Strict Role-Based Access Control (Authorization Routing)
        if tier == OperationalTier.DEFENSE_ADMIN:
            return self._handle_defense_admin(action, command_data)
            
        elif tier == OperationalTier.COMMERCIAL_OPERATOR:
            return self._handle_commercial_operator(action, command_data)
            
        elif tier == OperationalTier.SYSTEM_AUDITOR:
            return self._handle_system_auditor(action, command_data)

        return {"status": "error", "reason": "CRITICAL_STATE_ERROR"}

    def _handle_defense_admin(self, action: str, data: dict) -> dict:
        """Tier 1: Complete hardware and DSP configuration authority."""
        allowed_actions = ["UPDATE_ARRAY_GEOMETRY", "SET_SVD_WEIGHTS", "UPDATE_GLRT_THRESHOLD"]
        
        if action not in allowed_actions:
            return {"status": "error", "reason": "INVALID_ADMIN_ACTION", "action": action}
            
        # Example execution block
        if action == "UPDATE_GLRT_THRESHOLD":
            new_gamma = data.get("gamma", 50.17)
            logger.warning(f"[DEFENSE_ADMIN] Overriding GLRT Spatial Threshold to {new_gamma}")
            return {"status": "success", "executed_action": action, "new_gamma": new_gamma}
            
        return {"status": "success", "executed_action": action}

    def _handle_commercial_operator(self, action: str, data: dict) -> dict:
        """Tier 2: Restricted to live telemetry and non-destructive configurations."""
        allowed_actions = ["VIEW_TELEMETRY", "TOGGLE_LOGGING_LEVEL"]
        
        if action not in allowed_actions:
            logger.warning(f"Commercial Operator attempted restricted action: {action}")
            return {"status": "error", "reason": "UNAUTHORIZED_FOR_TIER", "action": action}
            
        if action == "TOGGLE_LOGGING_LEVEL":
            level = data.get("level", "INFO")
            logger.info(f"[COMMERCIAL_OPERATOR] Toggling stream logging to {level}")
            return {"status": "success", "executed_action": action, "logging_level": level}

        return {"status": "success", "executed_action": action}

    def _handle_system_auditor(self, action: str, data: dict) -> dict:
        """Tier 3: Strictly limited to cryptographic log extraction."""
        allowed_actions = ["EXTRACT_WORM_LEDGER"]
        
        if action not in allowed_actions:
            logger.warning(f"System Auditor attempted restricted action: {action}")
            return {"status": "error", "reason": "UNAUTHORIZED_FOR_TIER", "action": action}
            
        if action == "EXTRACT_WORM_LEDGER":
            target_file = data.get("target_file", "certin_incident_spoofing.json")
            logger.info(f"[SYSTEM_AUDITOR] Initiating secure extraction protocol for {target_file}")
            return {"status": "success", "executed_action": action, "target_file": target_file, "payload_ready": True}

        return {"status": "success", "executed_action": action}

# --- Rapid Verification Stub ---
if __name__ == "__main__":
    print("[*] Testing TenantConfigBroker Cryptographic Validation...")
    broker = TenantConfigBroker()
    
    if HAS_CRYPTO:
        # Mock a valid Defense Admin payload
        admin_priv = ed25519.Ed25519PrivateKey.from_private_bytes(b'0'*32)
        test_payload = json.dumps({"action": "UPDATE_GLRT_THRESHOLD", "gamma": 45.0})
        sig = admin_priv.sign(test_payload.encode('utf-8'))
        sig_b64 = base64.b64encode(sig).decode('utf-8')
        
        req = {
            "terminal_id": "terminal_alpha_admin",
            "raw_payload": test_payload,
            "signature": sig_b64
        }
        
        result = broker.process_config_update(req)
        print(f"[+] Admin Access Test Result: {result}")
        
        # Test Operator breaching scope
        op_priv = ed25519.Ed25519PrivateKey.from_private_bytes(b'1'*32)
        test_payload2 = json.dumps({"action": "UPDATE_GLRT_THRESHOLD"})
        sig2 = base64.b64encode(op_priv.sign(test_payload2.encode('utf-8'))).decode('utf-8')
        req2 = {
            "terminal_id": "terminal_beta_operator",
            "raw_payload": test_payload2,
            "signature": sig2
        }
        
        result2 = broker.process_config_update(req2)
        print(f"[+] Operator Breach Test Result: {result2}")
    else:
        print("[-] Cryptography module not installed. Tests bypassed.")
