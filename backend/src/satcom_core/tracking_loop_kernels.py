"""
Task 57.5: Fused Tracking-Loop Glue Kernels
SpaceShield High-Velocity Receiver DSP Subsystem

Phase 5: the tracking loop's per-cycle "glue" -- phase synchronisation,
discriminator, Kalman-prediction compounding, and the constant SNR feed -- was
a sequence of tiny NumPy expressions on 4-element arrays. Profiling
(tests/tracking_loop_verifier.py, isolated per-op timers) showed this glue cost
~9 us median, essentially all NumPy dispatch/allocation overhead rather than
arithmetic, split as sync ~2.7us + discriminator ~2.3us + prediction ~2.5us +
SNR alloc ~1.7us. The PRN synth (~25us, Phase 4 LUT), EML correlator (~8.6us),
and Kalman update (~0.8us) are already Numba kernels and are NOT touched here.

This module fuses that glue into two zero-allocation Numba kernels, one on each
side of the synth+correlator pair (which sit in the middle of the data flow, so
a single kernel is not possible):

  * _sync_replica_drivers  : states -> (code_phases, code_steps) for the synth
  * _discriminate_and_track: (I_E, I_L) -> discriminator, absolute measurement
                              compounding, then the EXISTING Kalman update

The Kalman math is reused verbatim by calling _update_kalman_loops from
kalman_loop_filter (njit-to-njit, inlined -- no extra dispatch, no divergence).
The arithmetic mirrors the previous inline NumPy exactly (same operation order),
so the state trajectory and discriminator trace match the pre-fusion loop to
floating-point tolerance (verified in tests/test_tracking_loop_fusion.py).
"""

import numpy as np
from numba import njit

from kalman_loop_filter import _update_kalman_loops


# fastmath is intentionally OFF for these two glue kernels: they are tiny
# (4-element) loops where fastmath buys no speed, and disabling it keeps the
# division (steps = freq/sample_rate, discriminator) and the compounding
# associativity bit-identical to the NumPy expressions they replace. The nested
# Kalman kernel keeps its own (unchanged) fastmath setting.
@njit(fastmath=False, cache=True, boundscheck=False)
def _sync_replica_drivers(
    states: np.ndarray,       # (targets, 3) [phase, freq, accel]
    code_length: int,
    sample_rate: float,
    phases_out: np.ndarray,   # (targets,) float64 -- written in place
    steps_out: np.ndarray,    # (targets,) float64 -- written in place
):
    """Replaces `code_phases = states[:,0] % code_length` and
    `steps = states[:,1] / sample_rate` (two NumPy allocations) with in-place
    writes. Numba's float `%` matches NumPy's (floor-based, non-negative for a
    positive divisor), verified for the negative initial-phase case."""
    for m in range(states.shape[0]):
        phases_out[m] = states[m, 0] % code_length
        steps_out[m] = states[m, 1] / sample_rate


@njit(fastmath=False, cache=True, boundscheck=False)
def _discriminate_and_track(
    I_E: np.ndarray,          # (targets,) early correlation magnitude
    I_L: np.ndarray,          # (targets,) late correlation magnitude
    states: np.ndarray,       # (targets, 3) -- updated in place
    covariances: np.ndarray,  # (targets, 6) -- updated in place
    T: float,
    half_T2: float,           # 0.5 * (T**2), precomputed to match the old inline expr
    q_accel: float,
    base_R: float,
    max_alpha: float,
    max_beta: float,
    max_gamma: float,
    snr: np.ndarray,          # (targets,) linear SNR feed
    disc_out: np.ndarray,     # (targets,) float64 -- discriminator error (for the trace)
    z_scratch: np.ndarray,    # (targets,) float64 -- absolute measurement scratch
):
    """Fuses the discriminator, the absolute-measurement compounding, and the
    Kalman predict/update. Reproduces, in order:
        disc      = 0.5 * (I_L - I_E) / (I_E + I_L + 1e-12)
        x0_p      = states[:,0] + T*states[:,1] + 0.5*T^2*states[:,2]
        absolute_z= x0_p + disc
        Kalman update with measurement absolute_z
    then leaves the discriminator error in disc_out for the caller's trace."""
    for m in range(states.shape[0]):
        ie = I_E[m]
        il = I_L[m]
        d = 0.5 * (il - ie) / (ie + il + 1e-12)
        disc_out[m] = d
        z_scratch[m] = states[m, 0] + T * states[m, 1] + half_T2 * states[m, 2] + d
    # Reuse the existing, verified Kalman kernel (inlined njit-to-njit call).
    _update_kalman_loops(
        states, covariances, z_scratch, snr,
        T, q_accel, base_R, max_alpha, max_beta, max_gamma,
    )


class FusedTrackingStep:
    """
    Holds the zero-allocation scratch buffers for the fused tracking-loop glue
    and drives the two kernels. Designed to slot directly into the timed region
    of the tracking loop in place of the inline NumPy expressions, without
    modifying the PRN synthesizer, EML correlator, or Kalman filter modules.
    """

    def __init__(self, targets: int, T: float, sample_rate: float, code_length: int):
        self.targets = targets
        self.T = T
        # Match the previous inline `0.5 * (T**2)` bit-for-bit.
        self.half_T2 = 0.5 * (T ** 2)
        self.sample_rate = sample_rate
        self.code_length = code_length

        self.phases = np.zeros(targets, dtype=np.float64)
        self.steps = np.zeros(targets, dtype=np.float64)
        self.disc = np.zeros(targets, dtype=np.float64)
        self._z = np.zeros(targets, dtype=np.float64)

        self._warmup()

    def _warmup(self):
        """Force JIT compilation of both kernels (including the nested Kalman)."""
        st = np.zeros((self.targets, 3), dtype=np.float64)
        st[:, 1] = self.sample_rate  # non-zero freq so steps are exercised
        cov = np.zeros((self.targets, 6), dtype=np.float64)
        ie = np.ones(self.targets); il = np.ones(self.targets)
        snr = np.ones(self.targets)
        _sync_replica_drivers(st, self.code_length, self.sample_rate, self.phases, self.steps)
        _discriminate_and_track(
            ie, il, st, cov, self.T, self.half_T2, 1.0, 0.1,
            0.8, 0.4, 0.1, snr, self.disc, self._z,
        )
        self.phases.fill(0.0); self.steps.fill(0.0); self.disc.fill(0.0)

    def sync_drivers(self, states: np.ndarray):
        """Compute code phases and code steps for the PRN synth from Kalman states."""
        _sync_replica_drivers(states, self.code_length, self.sample_rate, self.phases, self.steps)
        return self.phases, self.steps

    def discriminate_and_track(
        self, I_E, I_L, states, covariances,
        q_accel, base_R, max_alpha, max_beta, max_gamma, snr,
    ):
        """Discriminator + absolute-measurement compounding + Kalman update.
        Returns the discriminator error array (owned scratch; copy if retaining)."""
        _discriminate_and_track(
            I_E, I_L, states, covariances,
            self.T, self.half_T2, q_accel, base_R,
            max_alpha, max_beta, max_gamma, snr, self.disc, self._z,
        )
        return self.disc
