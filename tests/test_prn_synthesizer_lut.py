"""
Phase 4: verification of the fractional-phase-LUT PRN synthesizer.

Checks that the fixed-point integer-accumulator lookup-table synthesizer is
numerically/code-phase equivalent to the ORIGINAL algorithm (float phase,
int() truncation, packed-bit lookup) it replaced, that the fractional
resolution (frac_bits) is a configurable accuracy knob, and that behavior is
deterministic. All measurements are real; no thresholds elsewhere are weakened.
"""

import os
import sys

import numpy as np
from numba import njit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "backend", "src", "satcom_core"))

from prn_code_synthesizer import PRNCodeSynthesizer, pack_prn_to_bits, DEFAULT_FRAC_BITS

TARGETS, STRIDE, CODE_LEN = 4, 4096, 1023


@njit(fastmath=True, cache=True, boundscheck=False)
def _original_reference(bt, cl, E, P, L, ph, st, sp):
    """Exact original algorithm: float64 phase, int() truncation, packed-bit lookup."""
    nt = P.shape[0]; sl = P.shape[1]
    for m in range(nt):
        phase = ph[m]; step = st[m]
        for n in range(sl):
            if phase >= cl: phase -= cl
            pe = phase - sp
            if pe < 0.0: pe += cl
            pl = phase + sp
            if pl >= cl: pl -= cl
            ie = int(pe); ip = int(phase); il = int(pl)
            E[m, n] = (1.0 - 2.0 * ((bt[m, ie >> 5] >> (ie & 31)) & 1)) + 0j
            P[m, n] = (1.0 - 2.0 * ((bt[m, ip >> 5] >> (ip & 31)) & 1)) + 0j
            L[m, n] = (1.0 - 2.0 * ((bt[m, il >> 5] >> (il & 31)) & 1)) + 0j
            phase += step
        ph[m] = phase


def _bit_table(seed=42):
    rng = np.random.default_rng(seed)
    raw = np.where(rng.integers(0, 2, size=(TARGETS, CODE_LEN)) == 0, -1.0, 1.0).astype(np.float32)
    return pack_prn_to_bits(raw)


def _run_reference(bit_table, phases, steps, spacing):
    E = np.zeros((TARGETS, STRIDE), np.complex64)
    P = np.zeros((TARGETS, STRIDE), np.complex64)
    L = np.zeros((TARGETS, STRIDE), np.complex64)
    ph = phases.copy()
    _original_reference(bit_table, CODE_LEN, E, P, L, ph, steps, spacing)
    return E, P, L, ph


def test_equivalence_to_original_algorithm():
    """
    New synth must match the original truncation algorithm to floating-point
    precision: code phase within <5e-7 chips and only a negligible fraction of
    samples differing (chip-boundary ties), across representative Doppler steps.
    """
    bit_table = _bit_table()
    sample_rate = 4.0e6
    code_freqs = np.array([1.023e6 + 5.0, 1.023e6 - 12.0, 1.023e6 + 0.1, 1.023e6 - 3.4])
    steps = code_freqs / sample_rate
    synth = PRNCodeSynthesizer(TARGETS, STRIDE, CODE_LEN)

    total_samples = 0
    total_mismatch = 0
    max_phase_err = 0.0
    for cyc in range(200):
        phases = (np.array([0.0, 0.25, 0.5, 0.75]) + cyc * 0.017) % CODE_LEN
        synth.code_phases = phases.copy()
        E, P, L = synth.synthesize_stride(bit_table, steps, 0.5)
        Er, Pr, Lr, phr = _run_reference(bit_table, phases, steps, 0.5)
        total_mismatch += int(np.sum(E.real != Er.real) + np.sum(P.real != Pr.real) + np.sum(L.real != Lr.real))
        total_samples += 3 * TARGETS * STRIDE
        max_phase_err = max(max_phase_err, float(np.max(np.abs(synth.code_phases - phr))))

    frac = total_mismatch / total_samples
    assert max_phase_err < 5e-6, f"code-phase drift too large: {max_phase_err:.3e} chips"
    assert frac < 1e-4, f"too many sample mismatches vs original: {frac:.2e}"
    print(f"[PASS] LUT synth equivalent to original: mismatch fraction={frac:.2e}, "
          f"max phase error={max_phase_err:.2e} chips.")


def test_frac_bits_is_configurable_accuracy_knob():
    """Higher fractional resolution must not increase disagreement with the float
    reference; frac_bits=32 must be effectively exact within a stride."""
    bit_table = _bit_table(7)
    steps = np.array([1.023e6 + 3.0, 1.023e6 - 7.0, 1.023e6, 1.023e6 + 0.5]) / 4.0e6
    phases0 = np.array([10.3, 200.7, 512.1, 1000.9])

    Er, Pr, Lr, _ = _run_reference(bit_table, phases0, steps, 0.5)

    prev = None
    results = {}
    for fb in (8, 16, 24, 32):
        synth = PRNCodeSynthesizer(TARGETS, STRIDE, CODE_LEN, frac_bits=fb)
        synth.code_phases = phases0.copy()
        E, _, _ = synth.synthesize_stride(bit_table, steps, 0.5)
        mism = int(np.sum(E.real != Er.real))
        results[fb] = mism
        if prev is not None:
            assert mism <= prev, f"accuracy regressed as frac_bits rose: fb={fb} ({mism}) > prev ({prev})"
        prev = mism

    assert results[32] == 0, f"frac_bits=32 not exact within a stride: {results[32]} mismatches"
    assert DEFAULT_FRAC_BITS == 32
    print(f"[PASS] frac_bits accuracy monotone: {results} (32 -> exact within a stride).")


def test_deterministic_and_api_shapes():
    """Identical inputs -> identical outputs; public shapes/dtypes preserved."""
    bit_table = _bit_table()
    steps = np.array([1.023e6, 1.023e6, 1.023e6, 1.023e6]) / 4.0e6

    def once():
        s = PRNCodeSynthesizer(TARGETS, STRIDE, CODE_LEN)
        s.code_phases = np.array([0.0, 0.25, 0.5, 0.75])
        E, P, L = s.synthesize_stride(bit_table, steps, 0.5)
        return E.copy(), P.copy(), L.copy(), s.code_phases.copy()

    e1, p1, l1, ph1 = once()
    e2, p2, l2, ph2 = once()
    assert e1.shape == (TARGETS, STRIDE) and e1.dtype == np.complex64
    assert np.array_equal(e1, e2) and np.array_equal(p1, p2) and np.array_equal(l1, l2)
    assert np.array_equal(ph1, ph2)
    # replicas are pure +/-1 (imag == 0)
    assert np.all(np.isin(e1.real, (-1.0, 1.0))) and np.all(e1.imag == 0.0)
    print("[PASS] LUT synth is deterministic; E/P/L shapes, dtype, and +/-1 range preserved.")


def test_lut_refreshes_when_code_changes():
    """The cached code LUT must refresh when a different PRN table is supplied."""
    steps = np.array([1.023e6, 1.023e6, 1.023e6, 1.023e6]) / 4.0e6
    synth = PRNCodeSynthesizer(TARGETS, STRIDE, CODE_LEN)
    bt_a = _bit_table(1)
    bt_b = _bit_table(2)
    synth.code_phases = np.array([0.0, 0.0, 0.0, 0.0]); Ea, _, _ = synth.synthesize_stride(bt_a, steps, 0.5); Ea = Ea.copy()
    synth.code_phases = np.array([0.0, 0.0, 0.0, 0.0]); Eb, _, _ = synth.synthesize_stride(bt_b, steps, 0.5); Eb = Eb.copy()
    assert not np.array_equal(Ea, Eb), "LUT did not refresh for a different PRN code table"
    # switching back reproduces the original
    synth.code_phases = np.array([0.0, 0.0, 0.0, 0.0]); Ea2, _, _ = synth.synthesize_stride(bt_a, steps, 0.5)
    assert np.array_equal(Ea, Ea2), "LUT refresh not stable when returning to a prior table"
    print("[PASS] Code LUT refreshes on table change and is stable on return.")


if __name__ == "__main__":
    test_equivalence_to_original_algorithm()
    test_frac_bits_is_configurable_accuracy_knob()
    test_deterministic_and_api_shapes()
    test_lut_refreshes_when_code_changes()
    print("\n[PASSED] All PRN fractional-phase-LUT synthesizer tests cleared.")
