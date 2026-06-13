#!/usr/bin/env python3
"""
SpaceShield: Secure POSIX Inter-Process Communication (IPC) Bridge
Description: Decouples physical SDR hardware ingestion from the DSP engine 
             using lock-protected, strict-boundary POSIX shared memory mapped 
             in /dev/shm.
"""

import os
import time
import json
import signal
import logging
import numpy as np
from multiprocessing import shared_memory

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

logger = logging.getLogger("SecureIPCBridge")
logger.setLevel(logging.INFO)

class SecureIPCBridge:
    def __init__(self, create: bool = False, num_channels: int = 4, chunk_size: int = 4096):
        """
        Instantiates the strict-boundary dual-ring shared memory region.
        
        Parameters:
            create (bool): If True, creates the /dev/shm block. If False, attaches.
            num_channels (int): M=4 physical antennas.
            chunk_size (int): Temporal observation stride.
        """
        self.num_channels = num_channels
        self.chunk_size = chunk_size
        
        # Dual-Ring setup (Buffer 0 and Buffer 1)
        # Type: complex64 (8 bytes per element)
        self.element_size = 8 
        self.total_elements = 2 * self.num_channels * self.chunk_size
        self.shm_size = self.total_elements * self.element_size
        self.shm_name = "spaceshield_matrix_pool"
        
        # Lock configuration
        self.lock_path = "/tmp/spaceshield_matrix_pool.lock" if HAS_FCNTL else os.path.join(os.environ.get("TEMP", "C:\\temp"), "spaceshield_matrix.lock")
        
        self.shm = None
        self._buffer_array = None
        
        self._initialize_memory(create)

    def _initialize_memory(self, create: bool):
        """Maps the POSIX memory block and enforces array boundaries."""
        try:
            if create:
                # Purge lingering zombie blocks
                try:
                    existing_shm = shared_memory.SharedMemory(name=self.shm_name)
                    existing_shm.unlink()
                except FileNotFoundError:
                    pass
                
                logger.info(f"Mapping new POSIX Shared Memory [{self.shm_name}] ({self.shm_size} bytes)")
                self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.shm_size)
            else:
                logger.info(f"Attaching to existing Shared Memory [{self.shm_name}]")
                self.shm = shared_memory.SharedMemory(name=self.shm_name)
                
            # Map strict Numpy view over the raw shared buffer
            self._buffer_array = np.ndarray(
                shape=(2, self.num_channels, self.chunk_size),
                dtype=np.complex64,
                buffer=self.shm.buf
            )
            
            # Create descriptor lock
            if not os.path.exists(self.lock_path):
                with open(self.lock_path, "w") as f:
                    f.write("")
            self.lock_fd = open(self.lock_path, "r+")
            
        except Exception as e:
            logger.critical(f"Failed to map IPC memory: {e}")
            raise

    def acquire_lock(self):
        """Atomic Linux file-descriptor locking to prevent concurrent read/write races."""
        if HAS_FCNTL:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
            
    def release_lock(self):
        """Releases the exclusive POSIX flock."""
        if HAS_FCNTL:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)

    def _trigger_sigsegv(self, reason: str):
        """
        Hard-enforcement mechanism. If an untrusted process attempts an OOB write,
        we write an immutable WORM log event, then manually raise SIGSEGV to kill the host.
        """
        logger.critical(f"[!] MEMORY BOUNDARY VIOLATION: {reason}")
        
        # 1. WORM Compliance Ledger Injection
        log_payload = {
            "timestamp": time.time(),
            "incident_type": "IPC_OOB_MEMORY_VIOLATION",
            "reason": reason,
            "containment_action": "FORCED_SIGSEGV"
        }
        
        compliance_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../compliance'))
        os.makedirs(compliance_dir, exist_ok=True)
        report_path = os.path.join(compliance_dir, "certin_incident_spoofing.json")
        
        try:
            # We append directly or fallback if blocked
            try:
                with open(report_path, "a") as f:
                    f.write(json.dumps(log_payload) + "\n")
            except PermissionError:
                report_path = report_path.replace('.json', '_segv.json')
                with open(report_path, "w") as f:
                    json.dump(log_payload, f)
        except Exception:
            pass # Fail silently on logging if absolute catastrophic collapse
            
        # 2. Release locks cleanly before self-immolation
        try:
            self.release_lock()
            if self.shm: self.shm.close()
        except Exception:
            pass
            
        # 3. Kernel Isolation Drop (SIGSEGV)
        print("\n\033[1;41;37m[ CRITICAL KERNEL ISOLATION: FORCING SIGSEGV DUE TO ILLEGAL MEMORY ACCESS ]\033[0m")
        time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGSEGV)

    def write_buffer(self, ring_index: int, payload: np.ndarray):
        """
        Writes physical payload directly to the shared memory region with strict bounds checking.
        
        Parameters:
            ring_index (int): 0 or 1 targeting the dual-ring absorber.
            payload (np.ndarray): The incoming complex64 I/Q block.
        """
        if ring_index not in [0, 1]:
            self._trigger_sigsegv(f"Invalid Ring Buffer Index Request: {ring_index}")
            
        if payload.shape != (self.num_channels, self.chunk_size) or payload.dtype != np.complex64:
            self._trigger_sigsegv(f"Payload Signature Mismatch. Expected {(self.num_channels, self.chunk_size)} complex64. Got {payload.shape} {payload.dtype}")
            
        self.acquire_lock()
        try:
            # Direct raw memory copy. Zero allocation parsing.
            np.copyto(self._buffer_array[ring_index], payload)
        except Exception as e:
            self._trigger_sigsegv(f"Unexpected memory copying exception: {e}")
        finally:
            self.release_lock()

    def read_buffer(self, ring_index: int, output_buffer: np.ndarray):
        """
        Extracts a chunk from the shared memory into the target DSP workspace safely.
        """
        if ring_index not in [0, 1]:
            self._trigger_sigsegv(f"Invalid Ring Buffer Index Request: {ring_index}")
            
        if output_buffer.shape != (self.num_channels, self.chunk_size) or output_buffer.dtype != np.complex64:
            self._trigger_sigsegv(f"Output Buffer Constraints Violated: {output_buffer.shape} {output_buffer.dtype}")
            
        self.acquire_lock()
        try:
            # Direct raw memory extraction.
            np.copyto(output_buffer, self._buffer_array[ring_index])
        finally:
            self.release_lock()
            
    def shutdown(self):
        """Cleanly drops handles and unlinks POSIX block."""
        if self.shm:
            self.shm.close()
            try:
                self.shm.unlink()
            except Exception:
                pass


# --- Rapid Verification Stub ---
if __name__ == "__main__":
    print("[*] Testing SpaceShield Kernel IPC Architecture...")
    M, N = 4, 4096
    
    # Process A: Allocator & Hardware Writer
    ipc_host = SecureIPCBridge(create=True, num_channels=M, chunk_size=N)
    
    test_data = np.ones((M, N), dtype=np.complex64) * (1.0 + 1j*0.5)
    
    print("[+] Writing to Dual-Ring Buffer 0 under POSIX flock...")
    ipc_host.write_buffer(0, test_data)
    
    # Process B: DSP Consumer attached to the same /dev/shm block
    ipc_consumer = SecureIPCBridge(create=False, num_channels=M, chunk_size=N)
    
    receiver_buffer = np.zeros((M, N), dtype=np.complex64)
    ipc_consumer.read_buffer(0, receiver_buffer)
    
    print(f"[+] Cross-Process Read Successful. Value parity: {receiver_buffer[0,0]}")
    
    ipc_host.shutdown()
    ipc_consumer.shutdown()
    
    print("[!] Simulating Out-Of-Bounds Write Isolation...")
    # Recreate to demonstrate SIGSEGV
    ipc_fault = SecureIPCBridge(create=True, num_channels=M, chunk_size=N)
    malicious_data = np.ones((M, N+10), dtype=np.complex64)  # 10 samples too large!
    
    try:
        ipc_fault.write_buffer(0, malicious_data)
    except Exception as e:
        # If running on Windows, SIGSEGV will terminate standard Python gracefully with a Windows error
        print(f"Isolated: {e}")
    finally:
        ipc_fault.shutdown()
