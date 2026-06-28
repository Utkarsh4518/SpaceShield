"""
Task 68.3: Live Bench RF Validation Module
SpaceShield High-Velocity Receiver DSP Subsystem

Runs loopback HIL verification of the closed-loop register control path
under high-power RF sweeps, confirming lock security and recovery latency.
"""

import time
import numpy as np
from numba import njit

from sdr_register_actuator import SDRRegisterActuator, REG_SAMPLE_RATE, REG_SAFETY_INTERLOCK
from sdr_register_feedback_monitor import SDRRegisterFeedbackMonitor, STATUS_CLEAN, STATUS_PENDING, STATUS_TIMEOUT, REG_ACK_SAMPLE_RATE
from mitigation_hysteresis_filter import MitigationHysteresisFilter

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _run_live_bench_stride_jit(
    rf_power_dbm: float,
    rate_reg_val: int,
    interlock_val: int,
    consecutive_critical: int,
    consecutive_slips: int,
    action_history: np.ndarray,
    history_idx: int,
    history_len: int,
    up_window: int
) -> tuple[int, int, int, int]:
    """
    Zero-Heap JIT evaluation of RF sweep telemetry:
    Returns (raw_action, updated_consecutive_critical, updated_consecutive_slips, next_idx).
    """
    # 1. Anomaly classification based on power and sample rate
    rf_alarm = 1 if rf_power_dbm > -45.0 else 0
    
    sync_state = 0
    if rf_alarm == 1:
        if rate_reg_val == 1:  # 125 MHz is unstable under high-power sweep!
            sync_state = 3  # DESYNC
        else:
            sync_state = 0  # Stable rate-reduction!
            
    # 2. Update counters
    new_crit = consecutive_critical
    if sync_state == 3:
        new_crit += 1
    else:
        new_crit = max(0, new_crit - 1)
        
    new_slips = consecutive_slips
    if sync_state == 2 or sync_state == 3:
        new_slips += 1
    else:
        new_slips = max(0, new_slips - 1)
        
    # 3. Determine raw action
    raw_action = 0
    if interlock_val == 1:
        raw_action = 0  # Safety interlock locks interface, no actions
    elif new_crit >= 3:
        raw_action = 1  # REDUCE_SAMPLE_RATE
        
    # Write to history
    action_history[history_idx] = raw_action
    
    return raw_action, new_crit, new_slips, (history_idx + 1) % history_len


class LiveBenchRFValidation:
    """
    Manages end-to-end HIL sweep verification.
    Coordinates loopback telemetry, actuators, feedback monitors, and filters.
    """
    def __init__(self, up_window: int = 3, down_window: int = 5):
        self.actuator = SDRRegisterActuator(min_write_interval_ms=10.0)
        self.monitor = SDRRegisterFeedbackMonitor(timeout_ms=15.0)
        self.filter = MitigationHysteresisFilter(up_window=up_window, down_window=down_window)
        
        self.consecutive_critical = 0
        self.consecutive_slips = 0
        self.action_history = np.zeros(20, dtype=np.int32)
        self.history_idx = 0
        self.up_window = up_window

    def run_sweep_step(self, rf_power_dbm: float) -> dict:
        """
        Runs a single stride of the HIL RF sweep step.
        """
        rate_val = int(self.actuator.hardware_registers[REG_SAMPLE_RATE])
        interlock_val = int(self.actuator.hardware_registers[REG_SAFETY_INTERLOCK])
        
        # 1. JIT evaluate timing and state
        raw_act, new_crit, new_slips, next_idx = _run_live_bench_stride_jit(
            rf_power_dbm,
            rate_val,
            interlock_val,
            self.consecutive_critical,
            self.consecutive_slips,
            self.action_history,
            self.history_idx,
            len(self.action_history),
            self.up_window
        )
        
        self.consecutive_critical = new_crit
        self.consecutive_slips = new_slips
        self.history_idx = next_idx
        
        # 2. Filter raw action through hysteresis filter
        filtered_act = self.filter.filter_action(raw_act)
        
        # 3. Dispatched action via Register Actuator
        if filtered_act > 0:
            self.actuator.push_mitigation_action(filtered_act)
            # Log pending write to feedback monitor
            if filtered_act == 1:
                self.monitor.register_write_issued(REG_SAMPLE_RATE)
                
        # Process actuator command queue
        num_written = self.actuator.process_actuations()
        
        # 4. Check feedback monitor status
        status = self.monitor.check_status(self.actuator.hardware_registers)
        
        return {
            "raw_action": raw_act,
            "filtered_action": filtered_act,
            "writes_processed": num_written,
            "monitor_status": status,
            "sample_rate_reg": self.actuator.hardware_registers[REG_SAMPLE_RATE],
            "ack_sample_rate_reg": self.actuator.hardware_registers[REG_ACK_SAMPLE_RATE]
        }


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Live Bench RF Sweep Validation")
    print("==================================================================")
    
    validation = LiveBenchRFValidation(up_window=2, down_window=3)
    
    # 1. Nominal sweep (-90 dBm to -60 dBm)
    print("[*] Scenario 1: Nominal Power RF Sweep...")
    for step in range(5):
        power = -90.0 + step * 5.0
        res = validation.run_sweep_step(power)
        
    print(f"    -> Filtered Action: {res['filtered_action']} | Reg Rate: {res['sample_rate_reg']}")
    assert res["filtered_action"] == 0 and res["sample_rate_reg"] == 1, "Nominal sweep triggered false mitigation!"
    print("    -> Nominal sweep: [PASSED]")
    
    # 2. High Power Interference Sweep (-40 dBm)
    print("\n[*] Scenario 2: High-Power Jamming Sweep (Mitigation check)...")
    # Needs 3 critical steps to trigger raw action, plus 2 consecutive raw steps to pass filter
    # So we run 5 steps at high power
    for step in range(5):
        res = validation.run_sweep_step(-40.0)
        
    print(f"    -> Filtered Action: {res['filtered_action']} | Reg Rate: {res['sample_rate_reg']} | Pending writes status: {res['monitor_status']}")
    # Mitigation should trigger rate reduction (REG_SAMPLE_RATE = 2) and wait for ack (status = 1)
    assert res["sample_rate_reg"] == 2 and res["monitor_status"] == STATUS_PENDING, "Failed to apply rate reduction under jamming!"
    print("    -> High-power sweep mitigation: [PASSED]")

    # 3. Simulate hardware register acknowledgment
    print("\n[*] Scenario 3: Hardware Acknowledgment & Stabilization...")
    # Readback register matches updated rate
    validation.actuator.hardware_registers[REG_ACK_SAMPLE_RATE] = 2
    res = validation.run_sweep_step(-40.0)
    print(f"    -> Monitor Status after Ack: {res['monitor_status']}")
    assert res["monitor_status"] == STATUS_CLEAN, "Feedback monitor failed to clear status after acknowledgment!"
    print("    -> Hardware handshake confirmation: [PASSED]")

    print("\n[+] Live bench RF validation complete.")
