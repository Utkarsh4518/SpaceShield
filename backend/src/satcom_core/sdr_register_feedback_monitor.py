"""
Task 68.2: SDR Register Feedback Monitor Module
SpaceShield High-Velocity Receiver DSP Subsystem

Reads readback status from SDR hardware to confirm command completion.
Ensures safety locks are held until PLL rate-change and gain trim are acknowledged.
"""

import time
import numpy as np
from numba import njit

# Acknowledgment Address Map
REG_SAMPLE_RATE = 0x0010
REG_ACK_SAMPLE_RATE = 0x0080

REG_HANDOVER_CTRL = 0x0020
REG_ACK_HANDOVER = 0x0081

REG_TRACKER_GAIN = 0x0030
REG_ACK_GAIN = 0x0082

REG_SAFETY_INTERLOCK = 0x00F0
REG_ACK_SAFETY = 0x0083

STATUS_CLEAN = 0
STATUS_PENDING = 1
STATUS_TIMEOUT = 2

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _check_register_acknowledgment_jit(
    registers: np.ndarray,          # (256,) int64
    pending_writes: np.ndarray,     # (4, 3) int64 -> each row: [reg_addr, ack_addr, write_timestamp_ns]
    now_ns: int,
    timeout_ns: int
) -> int:
    """
    Zero-Heap JIT confirmation of command completion.
    Returns status: 0 (Clean), 1 (Pending), 2 (Timeout/Stall).
    """
    status = STATUS_CLEAN
    for i in range(4):
        reg_addr = pending_writes[i, 0]
        if reg_addr == -1:
            continue
            
        ack_addr = pending_writes[i, 1]
        write_ts = pending_writes[i, 2]
        
        # Check if readback matches requested value
        if registers[reg_addr] == registers[ack_addr]:
            pending_writes[i, 0] = -1  # Clear pending status
            continue
            
        # Check timeout
        if now_ns - write_ts > timeout_ns:
            status = STATUS_TIMEOUT
        else:
            if status != STATUS_TIMEOUT:
                status = STATUS_PENDING
                
    return status


class SDRRegisterFeedbackMonitor:
    """
    Monitors SDR register write acknowledgments.
    Provides hardware status confirmation before safety lock releases.
    """
    def __init__(self, timeout_ms: float = 20.0):
        self.timeout_ns = int(timeout_ms * 1_000_000.0)
        
        # Pre-allocated pending writes table
        # Rows: 0: Sample Rate, 1: Handover, 2: Gain, 3: Safety Interlock
        self.pending_writes = np.ones((4, 3), dtype=np.int64) * -1
        
        # Map registers
        self.pending_writes[0, 0] = -1
        self.pending_writes[0, 1] = REG_ACK_SAMPLE_RATE
        
        self.pending_writes[1, 0] = -1
        self.pending_writes[1, 1] = REG_ACK_HANDOVER
        
        self.pending_writes[2, 0] = -1
        self.pending_writes[2, 1] = REG_ACK_GAIN
        
        self.pending_writes[3, 0] = -1
        self.pending_writes[3, 1] = REG_ACK_SAFETY

    def register_write_issued(self, reg_addr: int):
        """Signals that a write command has been dispatched to hardware."""
        now_ns = time.perf_counter_ns()
        if reg_addr == REG_SAMPLE_RATE:
            self.pending_writes[0, 0] = REG_SAMPLE_RATE
            self.pending_writes[0, 2] = now_ns
        elif reg_addr == REG_HANDOVER_CTRL:
            self.pending_writes[1, 0] = REG_HANDOVER_CTRL
            self.pending_writes[1, 2] = now_ns
        elif reg_addr == REG_TRACKER_GAIN:
            self.pending_writes[2, 0] = REG_TRACKER_GAIN
            self.pending_writes[2, 2] = now_ns
        elif reg_addr == REG_SAFETY_INTERLOCK:
            self.pending_writes[3, 0] = REG_SAFETY_INTERLOCK
            self.pending_writes[3, 2] = now_ns

    def check_status(self, registers: np.ndarray) -> int:
        """
        Runs JIT check on all pending registers.
        """
        now_ns = time.perf_counter_ns()
        return _check_register_acknowledgment_jit(
            registers,
            self.pending_writes,
            now_ns,
            self.timeout_ns
        )


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: SDR Feedback Monitor Validation")
    print("==================================================================")
    
    # 20ms timeout
    monitor = SDRRegisterFeedbackMonitor(timeout_ms=20.0)
    mock_registers = np.zeros(256, dtype=np.int64)
    
    # Initialize nominal registers
    mock_registers[REG_SAMPLE_RATE] = 1
    mock_registers[REG_ACK_SAMPLE_RATE] = 1
    
    # 1. Successful Acknowledgment
    print("[*] Scenario 1: Successful Write Acknowledgment...")
    # Write sample rate 2
    mock_registers[REG_SAMPLE_RATE] = 2
    monitor.register_write_issued(REG_SAMPLE_RATE)
    
    # Check pending
    st1 = monitor.check_status(mock_registers)
    print(f"    -> Status: {st1} (Expected: 1 - Pending)")
    assert st1 == STATUS_PENDING, "Write should be pending confirmation!"
    
    # Simulated hardware acknowledges write
    mock_registers[REG_ACK_SAMPLE_RATE] = 2
    st2 = monitor.check_status(mock_registers)
    print(f"    -> Status: {st2} (Expected: 0 - Clean)")
    assert st2 == STATUS_CLEAN, "Write should be clean after acknowledgment!"
    print("    -> Acknowledgment cycle: [PASSED]")
    
    # 2. Timeout / Stalled Hardware
    print("\n[*] Scenario 2: Timeout Handling (Hardware Stall)...")
    # Dispatched gain change
    mock_registers[REG_TRACKER_GAIN] = 204
    monitor.register_write_issued(REG_TRACKER_GAIN)
    
    # Force mock timeout (backdate the write timestamp by 25ms)
    monitor.pending_writes[2, 2] = time.perf_counter_ns() - int(25 * 1_000_000.0)
    
    st3 = monitor.check_status(mock_registers)
    print(f"    -> Status: {st3} (Expected: 2 - Timeout)")
    assert st3 == STATUS_TIMEOUT, "Feedback monitor failed to signal timeout!"
    print("    -> Timeout and stall handling: [PASSED]")

    print("\n[+] SDR feedback monitor validation complete.")
