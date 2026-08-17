"""
LOCAL SIMULATION scenario state model for the SpaceShield console.

Extracted from app.py so the demo's signal-state behavior is deterministic,
reproducible, and unit-testable independently of Streamlit.

The deployed app (Streamlit Cloud) has no backend, so it runs in LOCAL
SIMULATION. Previously the simulated Bartlett-sphericity statistic was computed
from a hardcoded jammer term plus RNG noise and never read the selected
scenario, so the Scenario Control buttons changed only the URL while the system
stayed permanently in JAMMING. This module makes each scenario drive the two
spatial statistics the detector consumes; the existing detection thresholds
(`classify_verdict`) then produce the verdict. The verdict is never set directly.

Scenario -> statistics bands are chosen relative to the DEFAULT gamma (50) so the
unchanged threshold logic yields the intended verdict, while the gamma slider
still moves the decision boundary across the values:

    nominal  : sphericity below gamma           -> NORMAL           (isotropic METR ~0.25)
    jamming  : sphericity in (gamma, 1.5*gamma)  -> JAMMING          (anisotropic METR ~0.52)
    spoofing : sphericity above 1.5*gamma        -> CRITICAL SPOOFING (rank-1 METR ~0.90)

All values are deterministic functions of the frame counter (a small sine
ripple gives a "live" feel) -- reproducible for a given (scenario, tick,
attenuation), never random.
"""

import numpy as np

# Per-scenario means and (deterministic) ripple amplitudes for the two spatial
# statistics. sph = Bartlett sphericity LLR mean; metr = max-eigenvalue/trace.
SCENARIO_MODEL = {
    "nominal":  {"sph": 22.0, "sph_amp": 2.0, "metr": 0.25, "metr_amp": 0.006},
    "jamming":  {"sph": 62.0, "sph_amp": 2.5, "metr": 0.52, "metr_amp": 0.010},
    "spoofing": {"sph": 95.0, "sph_amp": 3.0, "metr": 0.90, "metr_amp": 0.010},
}
VALID_SCENARIOS = tuple(SCENARIO_MODEL.keys())
DEFAULT_SCENARIO = "nominal"
_NOMINAL_FLOOR = SCENARIO_MODEL["nominal"]["sph"]


def normalize_scenario(scenario: str) -> str:
    """Map an arbitrary query-param value to a known scenario (default nominal)."""
    return scenario if scenario in SCENARIO_MODEL else DEFAULT_SCENARIO


def simulate_scenario_state(scenario: str, tick: int, attenuation_db: float = 0.0) -> dict:
    """
    Deterministic LOCAL-SIMULATION spatial statistics for a scenario.

    Pure function of (scenario, tick, attenuation_db) -> no RNG, reproducible.

    Returns dict: {sphericity, metr, inference_latency_us}.

    Front-end antenna attenuation lowers the *excess* sphericity above the
    nominal floor (attenuating an interferer reduces its detectability) but never
    below the nominal baseline; capped at ~40% of the excess at full 30 dB so it
    cannot by itself flip the scenario class at low attenuation.
    """
    m = SCENARIO_MODEL[normalize_scenario(scenario)]
    phase = tick * 0.35

    atten = max(0.0, min(30.0, attenuation_db))
    atten_factor = 1.0 - 0.40 * (atten / 30.0)
    sph_mean = _NOMINAL_FLOOR + (m["sph"] - _NOMINAL_FLOOR) * atten_factor

    sphericity = max(0.0, sph_mean + m["sph_amp"] * float(np.sin(phase)))
    metr = min(1.0, max(0.0, m["metr"] + m["metr_amp"] * float(np.sin(phase * 1.3))))
    inference_latency_us = 199.7 + 2.0 * float(np.sin(phase * 0.5))

    return {
        "sphericity": sphericity,
        "metr": metr,
        "inference_latency_us": inference_latency_us,
    }


def classify_verdict(sphericity: float, gamma: float) -> str:
    """
    The console's detection decision logic (unchanged from the original inline
    thresholds): the Bartlett sphericity statistic vs the chi-squared boundary.

        sphericity > 1.5 * gamma -> CRITICAL SPOOFING
        sphericity > gamma       -> JAMMING
        otherwise                -> NORMAL

    This is the single source of truth for both the app and its tests; the
    verdict is always derived from the statistic, never set directly.
    """
    if sphericity > gamma * 1.5:
        return "CRITICAL SPOOFING"
    if sphericity > gamma:
        return "JAMMING"
    return "NORMAL"
