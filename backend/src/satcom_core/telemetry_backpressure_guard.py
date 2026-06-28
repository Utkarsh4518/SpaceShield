"""
Task 59.3: Telemetry Backpressure Guard Module
SpaceShield High-Velocity Receiver DSP Subsystem

Evaluates client queue pressure and makes JIT-compiled backpressure state transitions.
Assists the runtime supervisor in dropping and coalescing non-critical frames.
"""

import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True)
def _evaluate_pressure_state_jit(
    queue_size: int,
    capacity: int,
    high_threshold: float,
    critical_threshold: float
) -> int:
    """
    Zero-Heap Numba JIT Backpressure Evaluator:
    Calculates fill ratio and maps to pressure states:
    0: NOMINAL (fill_ratio < high_threshold)
    1: HIGH_PRESSURE (high_threshold <= fill_ratio < critical_threshold)
    2: CRITICAL_CONGESTION (fill_ratio >= critical_threshold)
    """
    if capacity <= 0:
        return 0
    
    fill_ratio = queue_size / capacity
    
    if fill_ratio >= critical_threshold:
        return 2
    elif fill_ratio >= high_threshold:
        return 1
    else:
        return 0


class TelemetryBackpressureGuard:
    """
    SpaceShield Telemetry Backpressure Guard.
    Monitors queue sizes and exposes state-driven backpressure controls.
    """
    def __init__(
        self,
        high_threshold: float = 0.7,
        critical_threshold: float = 0.9
    ):
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        
    def get_pressure_state(self, queue_size: int, capacity: int) -> int:
        """Returns the current pressure state (0, 1, 2) in sub-microsecond time."""
        return _evaluate_pressure_state_jit(
            queue_size,
            capacity,
            self.high_threshold,
            self.critical_threshold
        )
