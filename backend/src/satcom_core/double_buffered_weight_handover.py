"""
Task 64.1: Double-Buffered Weight Handover Module
SpaceShield High-Velocity Receiver DSP Subsystem

Implements double-buffered active/shadow weight registers to eliminate
null-leak windows during rapid satellite handovers.
"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _handover_update_jit(
    active_weights: np.ndarray,   # (4,) complex64 (in/out)
    shadow_weights: np.ndarray,   # (4,) complex64 (in/out)
    target_active: np.ndarray,    # (4,) complex64
    target_shadow: np.ndarray,    # (4,) complex64
    x_active: np.ndarray,         # (N, 4) complex64
    x_shadow: np.ndarray,         # (N, 4) complex64
    mu: float,
    N_steps: int,
    settling_counter: int,
    settling_threshold: float
) -> tuple[int, int]:
    """
    JIT-compiled active/shadow weight adaptation and switch logic.
    Returns (switch_verdict, updated_settling_counter).
    """
    # 1. Adapt active weights on the active channel
    for n in range(N_steps):
        xa = x_active[n]
        ya = active_weights[0].conjugate()*xa[0] + active_weights[1].conjugate()*xa[1] + active_weights[2].conjugate()*xa[2] + active_weights[3].conjugate()*xa[3]
        ya_c = ya.conjugate()
        
        va0 = active_weights[0] - mu * ya_c * xa[0]
        va1 = active_weights[1] - mu * ya_c * xa[1]
        va2 = active_weights[2] - mu * ya_c * xa[2]
        va3 = active_weights[3] - mu * ya_c * xa[3]
        
        inner_a = target_active[0].conjugate()*va0 + target_active[1].conjugate()*va1 + target_active[2].conjugate()*va2 + target_active[3].conjugate()*va3
        
        active_weights[0] = va0 - 0.25 * inner_a * target_active[0] + 0.25 * target_active[0]
        active_weights[1] = va1 - 0.25 * inner_a * target_active[1] + 0.25 * target_active[1]
        active_weights[2] = va2 - 0.25 * inner_a * target_active[2] + 0.25 * target_active[2]
        active_weights[3] = va3 - 0.25 * inner_a * target_active[3] + 0.25 * target_active[3]

    # 2. Adapt shadow weights on the shadow channel
    for n in range(N_steps):
        xs = x_shadow[n]
        ys = shadow_weights[0].conjugate()*xs[0] + shadow_weights[1].conjugate()*xs[1] + shadow_weights[2].conjugate()*xs[2] + shadow_weights[3].conjugate()*xs[3]
        ys_c = ys.conjugate()
        
        vs0 = shadow_weights[0] - mu * ys_c * xs[0]
        vs1 = shadow_weights[1] - mu * ys_c * xs[1]
        vs2 = shadow_weights[2] - mu * ys_c * xs[2]
        vs3 = shadow_weights[3] - mu * ys_c * xs[3]
        
        inner_s = target_shadow[0].conjugate()*vs0 + target_shadow[1].conjugate()*vs1 + target_shadow[2].conjugate()*vs2 + target_shadow[3].conjugate()*vs3
        
        shadow_weights[0] = vs0 - 0.25 * inner_s * target_shadow[0] + 0.25 * target_shadow[0]
        shadow_weights[1] = vs1 - 0.25 * inner_s * target_shadow[1] + 0.25 * target_shadow[1]
        shadow_weights[2] = vs2 - 0.25 * inner_s * target_shadow[2] + 0.25 * target_shadow[2]
        shadow_weights[3] = vs3 - 0.25 * inner_s * target_shadow[3] + 0.25 * target_shadow[3]

    # 3. Calculate average shadow leakage power
    p_sum = 0.0
    for n in range(N_steps):
        xs = x_shadow[n]
        ys = shadow_weights[0].conjugate()*xs[0] + shadow_weights[1].conjugate()*xs[1] + shadow_weights[2].conjugate()*xs[2] + shadow_weights[3].conjugate()*xs[3]
        p_sum += ys.real*ys.real + ys.imag*ys.imag
    p_avg = p_sum / N_steps

    next_counter = settling_counter + 1
    
    # Trigger swap only after 10 adaptation iterations and if leakage is minimized
    if next_counter >= 10 and p_avg < settling_threshold:
        return 1, 0
        
    return 0, next_counter


class DoubleBufferedWeightHandover:
    """
    Manages active and standby beamforming channels.
    Stages incoming target weights in a shadow register to prevent spatial leakage.
    """
    def __init__(self, mu: float = 0.03, settling_threshold: float = 0.01):
        self.mu = mu
        self.settling_threshold = settling_threshold
        
        # Pre-allocated weights
        self.active_weights = np.ones(4, dtype=np.complex64) * 0.25
        self.shadow_weights = np.ones(4, dtype=np.complex64) * 0.25
        self.settling_counter = 0

    def process_stride(
        self,
        target_active: np.ndarray,
        target_shadow: np.ndarray,
        x_active: np.ndarray,
        x_shadow: np.ndarray
    ) -> bool:
        """
        Adapts active/shadow channels and triggers weight swap when settled.
        Returns True if a handover switch was executed.
        """
        switched, next_counter = _handover_update_jit(
            self.active_weights,
            self.shadow_weights,
            target_active.astype(np.complex64),
            target_shadow.astype(np.complex64),
            x_active.astype(np.complex64),
            x_shadow.astype(np.complex64),
            self.mu,
            x_active.shape[0],
            self.settling_counter,
            self.settling_threshold
        )
        
        self.settling_counter = next_counter
        
        if switched == 1:
            # Swap active weights
            self.active_weights = self.shadow_weights.copy()
            # Reset shadow to nominal standby steering
            self.shadow_weights = target_shadow.copy() * 0.25
            return True
            
        return False


# =========================================================================
# DETERMINISTIC HANDOVER VALIDATION HARNESS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Physical Layer: Double-Buffered Handover Validation")
    print("==================================================================")
    
    # Helper to generate spatial wavefront array vectors
    def get_steering_vector(elevation_deg: float) -> np.ndarray:
        rad = math.radians(elevation_deg)
        a = np.zeros(4, dtype=np.complex64)
        a[0] = 1.0
        a[1] = math.cos(rad) + 1j * math.sin(rad)
        a[2] = math.cos(rad * 1.5) + 1j * math.sin(rad * 1.5)
        a[3] = math.cos(rad * 2.0) + 1j * math.sin(rad * 2.0)
        return a

    # Helper to generate HIL received history
    def generate_hil_history(spoofer_a: np.ndarray, steps: int = 5) -> np.ndarray:
        x = np.zeros((steps, 4), dtype=np.complex64)
        for s in range(steps):
            amp = 2.0 * (math.cos(10.0 * s * 0.001) + 1j * math.sin(10.0 * s * 0.001))
            x[s] = amp * spoofer_a + np.random.normal(0, 0.01) + 1j * np.random.normal(0, 0.01)
        return x

    handover = DoubleBufferedWeightHandover(mu=0.03, settling_threshold=0.05)
    
    # Scenario 1: Smooth Handover Transition (Zenith Sat 1 -> Setting Sat 2)
    print("[*] Scenario 1: Overlapping Satellite Smooth Handover...")
    target_active = get_steering_vector(80.0)
    target_shadow = get_steering_vector(20.0)
    spoofer_a = get_steering_vector(50.0)
    
    switched = False
    for step in range(25):
        xa = generate_hil_history(spoofer_a)
        xs = generate_hil_history(spoofer_a)
        res = handover.process_stride(target_active, target_shadow, xa, xs)
        if res:
            switched = True
            break
            
    print(f"    -> Handover Switch Executed: {switched}")
    assert switched, "Failed to complete handover swap under overlapping visibility!"
    print("    -> Overlapping handover: [PASSED]")
    
    # Scenario 2: Abrupt 100-Degree Swap (Zenith -> Horizon)
    print("\n[*] Scenario 2: Abrupt >90-Degree Horizon-to-Zenith Swap...")
    target_active = get_steering_vector(10.0)
    target_shadow = get_steering_vector(110.0)
    
    switched = False
    for step in range(25):
        xa = generate_hil_history(spoofer_a)
        xs = generate_hil_history(spoofer_a)
        res = handover.process_stride(target_active, target_shadow, xa, xs)
        if res:
            switched = True
            break
            
    print(f"    -> Handover Switch Executed: {switched}")
    assert switched, "Failed to swap weights under abrupt wide-angle jump!"
    print("    -> Abrupt wide-angle swap: [PASSED]")

    print("\n[+] Double-buffered weight handover validation complete.")
