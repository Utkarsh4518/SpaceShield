"""
Task 67.3: SDR Register Actuator Module
SpaceShield High-Velocity Receiver DSP Subsystem

Translates control actions into bounded register updates.
Enforces minimum write intervals to protect PLL loops from oscillation.
"""

import time
import numpy as np
from numba import njit

# Register Map Addresses
REG_SAMPLE_RATE = 0x0010      # Valid: 1 (125MHz), 2 (62.5MHz), 4 (31.25MHz)
REG_HANDOVER_CTRL = 0x0020    # Valid: 0 (Postponed), 1 (Allowed)
REG_TRACKER_GAIN = 0x0030     # Raw step size representation (e.g. 819 for 0.08)
REG_SAFETY_INTERLOCK = 0x00F0  # Valid: 0 (Unlocked), 1 (Locked)

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _push_command_jit(
    address: int,
    value: int,
    ts_ns: int,
    queue: np.ndarray,      # (32, 3) int64
    q_write_idx: int,
    q_len: int
) -> int:
    """Pushes command onto circular queue."""
    queue[q_write_idx, 0] = address
    queue[q_write_idx, 1] = value
    queue[q_write_idx, 2] = ts_ns
    return (q_write_idx + 1) % q_len


@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _process_queue_jit(
    queue: np.ndarray,              # (32, 3) int64
    q_read_idx: int,
    q_write_idx: int,
    q_len: int,
    hardware_registers: np.ndarray, # (256,) int64
    last_write_timestamps: np.ndarray, # (256,) int64
    min_write_interval_ns: int,
    interlock_active: int
) -> tuple[int, int]:
    """
    Processes all queued commands up to q_write_idx.
    Enforces interlocks and minimum write interval bounds.
    """
    processed = 0
    read_idx = q_read_idx
    
    while read_idx != q_write_idx:
        addr = queue[read_idx, 0]
        val = queue[read_idx, 1]
        ts_ns = queue[read_idx, 2]
        
        # 1. Check address boundaries
        if addr < 0 or addr >= 256:
            read_idx = (read_idx + 1) % q_len
            continue
            
        # 2. Check safety interlock
        if interlock_active == 1 and addr != REG_SAFETY_INTERLOCK:
            read_idx = (read_idx + 1) % q_len
            continue
            
        # 3. Check minimum write interval to prevent PLL oscillation
        last_ts = last_write_timestamps[addr]
        if addr != REG_SAFETY_INTERLOCK and ts_ns - last_ts < min_write_interval_ns:
            # Drop/defer
            read_idx = (read_idx + 1) % q_len
            continue
            
        # Apply register write
        hardware_registers[addr] = val
        last_write_timestamps[addr] = ts_ns
        processed += 1
        
        read_idx = (read_idx + 1) % q_len
        
    return read_idx, processed


class SDRRegisterActuator:
    """
    Translates closed-loop decisions to hardware register adjustments.
    Applies safety locks and debounces commands.
    """
    def __init__(self, min_write_interval_ms: float = 50.0):
        self.min_write_interval_ns = int(min_write_interval_ms * 1_000_000.0)
        self.q_len = 32
        
        # Pre-allocated arrays
        self.command_queue = np.zeros((32, 3), dtype=np.int64)
        self.q_read_idx = 0
        self.q_write_idx = 0
        
        self.hardware_registers = np.zeros(256, dtype=np.int64)
        self.last_write_timestamps = np.zeros(256, dtype=np.int64)
        
        # Initialize default registers
        self.hardware_registers[REG_SAMPLE_RATE] = 1
        self.hardware_registers[REG_HANDOVER_CTRL] = 1
        self.hardware_registers[REG_TRACKER_GAIN] = 819
        self.hardware_registers[REG_SAFETY_INTERLOCK] = 0

    def push_mitigation_action(self, action_code: int) -> int:
        """
        Translates mitigation action code to register commands and pushes to queue.
        """
        now_ns = time.perf_counter_ns()
        
        if action_code == 1:  # ACTION_REDUCE_SAMPLE_RATE
            # Reduce rate command (value = 2, representing 62.5 MHz)
            self.q_write_idx = _push_command_jit(
                REG_SAMPLE_RATE, 2, now_ns, self.command_queue, self.q_write_idx, self.q_len
            )
        elif action_code == 2:  # ACTION_POSTPONE_HANDOVER
            # Postpone/lock handover (value = 0)
            self.q_write_idx = _push_command_jit(
                REG_HANDOVER_CTRL, 0, now_ns, self.command_queue, self.q_write_idx, self.q_len
            )
        elif action_code == 3:  # ACTION_REDUCE_TRACKER_GAIN
            # Reduce LMS tracking gain (value = 204, representing 0.02)
            self.q_write_idx = _push_command_jit(
                REG_TRACKER_GAIN, 204, now_ns, self.command_queue, self.q_write_idx, self.q_len
            )
        elif action_code == 4:  # ACTION_ESCALATE_TO_OPERATOR
            # Trigger safety interlock lock (value = 1)
            self.q_write_idx = _push_command_jit(
                REG_SAFETY_INTERLOCK, 1, now_ns, self.command_queue, self.q_write_idx, self.q_len
            )
            
        return self.q_write_idx

    def process_actuations(self) -> int:
        """
        Executes queued register writes.
        """
        interlock_active = int(self.hardware_registers[REG_SAFETY_INTERLOCK])
        
        new_read_idx, num_processed = _process_queue_jit(
            self.command_queue,
            self.q_read_idx,
            self.q_write_idx,
            self.q_len,
            self.hardware_registers,
            self.last_write_timestamps,
            self.min_write_interval_ns,
            interlock_active
        )
        
        self.q_read_idx = new_read_idx
        return num_processed

    def clear_safety_interlock(self) -> int:
        """Manually unlocks register interface."""
        now_ns = time.perf_counter_ns()
        self.q_write_idx = _push_command_jit(
            REG_SAFETY_INTERLOCK, 0, now_ns, self.command_queue, self.q_write_idx, self.q_len
        )
        # Process immediately to unlock registers
        new_read_idx, num_processed = _process_queue_jit(
            self.command_queue,
            self.q_read_idx,
            self.q_write_idx,
            self.q_len,
            self.hardware_registers,
            self.last_write_timestamps,
            self.min_write_interval_ns,
            0  # temporarily bypass check to allow unlock
        )
        self.q_read_idx = new_read_idx
        return num_processed


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: SDR Register Actuator Validation")
    print("==================================================================")
    
    # Actuator with 10 ms interval for faster testing
    actuator = SDRRegisterActuator(min_write_interval_ms=10.0)
    
    # 1. Stable operation
    print("[*] Scenario 1: Nominal writes...")
    # Reduce rate
    actuator.push_mitigation_action(1)
    processed = actuator.process_actuations()
    print(f"    -> Processed: {processed} | REG_SAMPLE_RATE: {actuator.hardware_registers[REG_SAMPLE_RATE]}")
    assert processed == 1 and actuator.hardware_registers[REG_SAMPLE_RATE] == 2, "Failed nominal write!"
    print("    -> Nominal write: [PASSED]")
    
    # 2. Repeated trigger burst (flip-flop protection)
    print("\n[*] Scenario 2: Flip-Flop Protection...")
    # Push two commands to REG_SAMPLE_RATE immediately
    actuator.push_mitigation_action(1)
    actuator.push_mitigation_action(1)
    processed = actuator.process_actuations()
    print(f"    -> Processed writes: {processed} (Expected: 0 or 1, due to interval restriction)")
    # Since interval has not elapsed, the second write must be deferred or ignored
    assert processed <= 1, "Flip-flop protection failed to throttle rapid writes!"
    print("    -> Flip-flop throttle check: [PASSED]")
    
    # 3. Safety Interlock
    print("\n[*] Scenario 3: Safety Interlock Locking...")
    # Escalate to operator locks interface
    actuator.push_mitigation_action(4)
    actuator.process_actuations()
    print(f"    -> REG_SAFETY_INTERLOCK state: {actuator.hardware_registers[REG_SAFETY_INTERLOCK]}")
    assert actuator.hardware_registers[REG_SAFETY_INTERLOCK] == 1, "Failed to lock safety interlock!"
    
    # Attempt to write to sample rate while locked (should be blocked)
    actuator.push_mitigation_action(1)
    processed = actuator.process_actuations()
    print(f"    -> Processed writes while locked: {processed}")
    assert processed == 0, "Safety interlock allowed write while locked!"
    print("    -> Safety interlock: [PASSED]")

    # 4. Recovery
    print("\n[*] Scenario 4: Manual Interlock Unlock & Recovery...")
    actuator.clear_safety_interlock()
    print(f"    -> Unlocked REG_SAFETY_INTERLOCK state: {actuator.hardware_registers[REG_SAFETY_INTERLOCK]}")
    assert actuator.hardware_registers[REG_SAFETY_INTERLOCK] == 0, "Failed to unlock interlock!"
    print("    -> Unlock recovery: [PASSED]")

    print("\n[+] SDR register actuator validation complete.")
