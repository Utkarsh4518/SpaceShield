"""
Focused tests for the SpaceShield console's LOCAL SIMULATION scenario state
(frontend/scenario_model.py).

These lock in the deployed-demo fix: the Scenario Control buttons must drive the
simulated spatial statistics (sphericity / METR) so that the existing detection
threshold logic produces distinct verdicts -- Nominal -> NORMAL,
Jamming -> JAMMING, Spoofing -> CRITICAL SPOOFING -- deterministically, and the
gamma threshold must still be able to flip the verdict. The verdict is always
derived from the statistic via classify_verdict (the detector is not bypassed).
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "frontend"))

from scenario_model import (
    simulate_scenario_state,
    classify_verdict,
    normalize_scenario,
    SCENARIO_MODEL,
    VALID_SCENARIOS,
    DEFAULT_SCENARIO,
)

DEFAULT_GAMMA = 50.0  # the console's default Detection Sensitivity slider value


def _verdict(scenario, tick, gamma=DEFAULT_GAMMA, attenuation=0.0):
    s = simulate_scenario_state(scenario, tick, attenuation)
    return classify_verdict(s["sphericity"], gamma), s


def test_default_is_nominal():
    assert DEFAULT_SCENARIO == "nominal"
    assert normalize_scenario("nominal") == "nominal"
    assert normalize_scenario("") == "nominal"
    assert normalize_scenario("bogus") == "nominal"          # unknown -> nominal
    assert normalize_scenario("jamming") == "jamming"
    assert set(VALID_SCENARIOS) == {"nominal", "jamming", "spoofing"}


def test_each_scenario_yields_its_intended_verdict_over_many_ticks():
    """At the default gamma, every scenario must hold its verdict across the
    ripple (no boundary flicker), with the expected METR regime."""
    expected = {
        "nominal": ("NORMAL", lambda m: m <= 0.40),      # isotropic
        "jamming": ("JAMMING", lambda m: 0.40 < m <= 0.60),  # anisotropic
        "spoofing": ("CRITICAL SPOOFING", lambda m: m > 0.60),  # eigen-collapse
    }
    for scenario, (want_verdict, metr_ok) in expected.items():
        verdicts, metrs, sphs = set(), [], []
        for tick in range(0, 120):
            v, s = _verdict(scenario, tick)
            verdicts.add(v)
            metrs.append(s["metr"])
            sphs.append(s["sphericity"])
        assert verdicts == {want_verdict}, (
            f"{scenario}: expected only {want_verdict}, got {verdicts} "
            f"(sph range [{min(sphs):.1f}, {max(sphs):.1f}])"
        )
        assert all(metr_ok(m) for m in metrs), f"{scenario}: METR out of regime: {min(metrs):.3f}-{max(metrs):.3f}"


def test_scenarios_are_distinct_from_each_other():
    n = simulate_scenario_state("nominal", 5)
    j = simulate_scenario_state("jamming", 5)
    s = simulate_scenario_state("spoofing", 5)
    # Sphericity strictly increases nominal < jamming < spoofing
    assert n["sphericity"] < j["sphericity"] < s["sphericity"]
    # METR strictly increases nominal < jamming < spoofing
    assert n["metr"] < j["metr"] < s["metr"]


def test_full_transition_sequence_nominal_jamming_spoofing_nominal():
    """The exact sequence the reviewer asked to verify: each hop changes the
    verdict, and returning to nominal restores NORMAL."""
    seq = ["nominal", "jamming", "spoofing", "nominal"]
    got = [_verdict(sc, tick=10)[0] for sc in seq]
    assert got == ["NORMAL", "JAMMING", "CRITICAL SPOOFING", "NORMAL"], got


def test_gamma_threshold_still_flips_the_verdict():
    """Raising gamma above the jamming sphericity must clear the alarm to
    NORMAL; lowering it below the nominal sphericity must raise an alarm. The
    detector logic, not the scenario, owns the verdict."""
    j = simulate_scenario_state("jamming", 3)["sphericity"]   # ~62
    assert classify_verdict(j, 50.0) == "JAMMING"
    assert classify_verdict(j, j + 10.0) == "NORMAL"          # threshold above signal -> cleared

    n = simulate_scenario_state("nominal", 3)["sphericity"]   # ~22
    assert classify_verdict(n, 50.0) == "NORMAL"
    assert classify_verdict(n, n - 5.0) == "JAMMING"          # over-tight threshold -> false alarm


def test_classify_verdict_boundaries():
    g = 50.0
    assert classify_verdict(g - 0.01, g) == "NORMAL"
    assert classify_verdict(g + 0.01, g) == "JAMMING"
    assert classify_verdict(g * 1.5 - 0.01, g) == "JAMMING"
    assert classify_verdict(g * 1.5 + 0.01, g) == "CRITICAL SPOOFING"


def test_state_is_deterministic_and_reproducible():
    """Same (scenario, tick, attenuation) -> identical output; no hidden RNG."""
    for scenario in VALID_SCENARIOS:
        for tick in (0, 1, 7, 42, 1000):
            for att in (0.0, 12.0, 30.0):
                a = simulate_scenario_state(scenario, tick, att)
                b = simulate_scenario_state(scenario, tick, att)
                assert a == b, f"non-deterministic at {scenario},{tick},{att}: {a} vs {b}"


def test_attenuation_dampens_interference_toward_baseline_monotonically():
    """Front-end attenuation lowers the jammer's sphericity monotonically, and
    at full attenuation drops it below the nominal-detection threshold -- but
    never below the nominal floor, and does not flip the class at 0 dB."""
    sph = [simulate_scenario_state("jamming", 0, att)["sphericity"] for att in (0.0, 10.0, 20.0, 30.0)]
    assert sph[0] > sph[1] > sph[2] > sph[3], f"attenuation not monotonic: {sph}"
    assert classify_verdict(sph[0], 50.0) == "JAMMING"        # 0 dB still jamming
    assert classify_verdict(sph[3], 50.0) == "NORMAL"         # full attenuation clears it
    assert sph[3] >= SCENARIO_MODEL["nominal"]["sph"] - 1e-9  # never below nominal floor


if __name__ == "__main__":
    test_default_is_nominal()
    test_each_scenario_yields_its_intended_verdict_over_many_ticks()
    test_scenarios_are_distinct_from_each_other()
    test_full_transition_sequence_nominal_jamming_spoofing_nominal()
    test_gamma_threshold_still_flips_the_verdict()
    test_classify_verdict_boundaries()
    test_state_is_deterministic_and_reproducible()
    test_attenuation_dampens_interference_toward_baseline_monotonically()
    print("[PASSED] All console scenario-state tests cleared.")
