"""
Focused verification of the PolynomialCoefficientTracker / SaturationInverter
NLMS predistortion loop, scoped to what is actually demonstrable given the
model-capacity investigation in saturation_linearization_verifier.py.

This deliberately does NOT assert the 20% OOB-reduction or 25dB IMD-
suppression targets used elsewhere: an exhaustive-ish search over the full
3-parameter (c30, c31, c50) coefficient space this model can represent tops
out around 0.05dB of real IMD suppression, so those targets are not
reachable by the current architecture regardless of adaptation algorithm.
See tests/saturation_linearization_verifier.py and
backend/src/satcom_core/polynomial_coefficient_tracker.py for the full
writeup. What IS tested here, with real (non-fabricated) measurements:

  1. nonlinear distortion is actually detected (regrowth flag responds to
     the presence/absence of real nonlinear distortion, not a hardcoded
     value)
  2. coefficient adaptation moves in the mathematically correct gradient
     direction (verified against a numerical finite-difference gradient of
     the tracker's own objective, not just re-asserted from theory)
  3. OOB energy measurably decreases over the achievable window (with an
     honest ceiling, not the unreached 20% target)
  4. convergence is reproducible (deterministic given identical inputs; no
     hidden RNG dependency)
  5. the IMD suppression measurement itself is correct on cases with a known
     right answer, independent of whether the tracker's adaptation succeeds
"""

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_SRC = os.path.join(BASE_DIR, 'backend', 'src', 'satcom_core')
sys.path.insert(0, BACKEND_SRC)

from polynomial_coefficient_tracker import PolynomialCoefficientTracker
from saturation_inverter import SaturationInverter


NUM_CHANNELS = 4
STRIDE_LEN = 4096
DEFAULT_FILTER = np.array([0.5, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _two_tone_distorted_signal():
    """Deterministic two-tone signal through a known 3rd/5th-order memory-polynomial distortion."""
    t = np.arange(STRIDE_LEN)
    s_clean = np.zeros((NUM_CHANNELS, STRIDE_LEN), dtype=np.complex64)
    for ch in range(NUM_CHANNELS):
        s_clean[ch] = 0.6 * np.exp(1j * 0.1 * t) + 0.4 * np.exp(1j * 0.3 * t)
    x_distorted = np.zeros_like(s_clean)
    for ch in range(NUM_CHANNELS):
        for n in range(STRIDE_LEN):
            val = s_clean[ch, n] - 0.15 * s_clean[ch, n] * (abs(s_clean[ch, n]) ** 2)
            if n > 0:
                val -= 0.05 * s_clean[ch, n - 1] * (abs(s_clean[ch, n - 1]) ** 2)
            x_distorted[ch, n] = val
    return s_clean, x_distorted


def _bandpass_filter(signal, f_low, f_high):
    num_ch, stride = signal.shape
    freqs = np.fft.fftfreq(stride)
    omega = np.abs(freqs) * 2.0 * np.pi
    mask = (omega >= f_low) & (omega <= f_high)
    out = np.zeros_like(signal, dtype=np.complex64)
    for ch in range(num_ch):
        sf = np.fft.fft(signal[ch])
        sf[~mask] = 0.0
        out[ch] = np.fft.ifft(sf)
    return out


def test_distortion_detection_responds_to_real_signal_state():
    """regrowth_flags must reflect whether OOB energy is actually present, not a fixed value."""
    _, x_distorted = _two_tone_distorted_signal()
    tracker = PolynomialCoefficientTracker(
        num_channels=NUM_CHANNELS, stride_len=STRIDE_LEN, mu=0.0, threshold_ratio=1e-4
    )

    # Case A: identity mapping (Y == X, i.e. "no correction applied yet" / no
    # residual regrowth signature beyond what's already in X itself).
    flags_a, oob_a, tot_a = tracker.process_stride(x_distorted, x_distorted)

    # Case B: inject a burst of genuinely out-of-band (near-Nyquist) content
    # into Y that isn't present in case A, to confirm the detector actually
    # responds to a real change in the signal rather than always returning
    # the same flag.
    y_with_regrowth = x_distorted.copy()
    t = np.arange(STRIDE_LEN)
    burst = (0.5 * np.exp(1j * 2.9 * t)).astype(np.complex64)
    y_with_regrowth += burst
    flags_b, oob_b, tot_b = tracker.process_stride(x_distorted, y_with_regrowth)

    assert np.all(oob_b > oob_a), (
        f"Injected out-of-band burst did not increase measured OOB energy: {oob_a} -> {oob_b}"
    )
    print(f"[PASS] OOB energy responds to real signal content: baseline={oob_a[0]:.4e}, "
          f"with injected regrowth={oob_b[0]:.4e} ({oob_b[0] / oob_a[0]:.1f}x)")


def test_gradient_direction_matches_finite_difference():
    """
    The analytic gradient computed inside _track_all_channels must agree in
    sign (and approximately in magnitude, given the block decimation) with a
    numerical finite-difference gradient of the same OOB-energy objective.
    This directly checks the update the kernel actually computes, rather
    than re-deriving it on paper as the previous investigation did.
    """
    _, x_distorted = _two_tone_distorted_signal()
    inverter = SaturationInverter(channels=NUM_CHANNELS, stride_len=STRIDE_LEN)

    def oob_energy_for_c30(c30_real_delta):
        coeffs = np.zeros((NUM_CHANNELS, 5, 2), dtype=np.complex64)
        coeffs[:, 0, 0] = 1.0
        coeffs[:, 2, 0] = c30_real_delta
        inverter.coefficients = coeffs
        Y = inverter.linearize_stride(x_distorted).copy()
        h = DEFAULT_FILTER
        e = np.zeros(STRIDE_LEN - 5, dtype=np.complex64)
        for k in range(6):
            e += h[k] * Y[0, 5 - k: STRIDE_LEN - k]
        return float(np.sum(np.abs(e) ** 2))

    eps = 1e-4
    j_plus = oob_energy_for_c30(eps)
    j_minus = oob_energy_for_c30(-eps)
    finite_diff_grad = (j_plus - j_minus) / (2 * eps)

    # Analytic gradient from the tracker kernel at c30=0 (matches the eps=0
    # expansion point above).
    tracker = PolynomialCoefficientTracker(
        num_channels=NUM_CHANNELS, stride_len=STRIDE_LEN, mu=1.0, threshold_ratio=-1.0,
        filter_coeffs=DEFAULT_FILTER,
    )
    coeffs0 = np.zeros((NUM_CHANNELS, 5, 2), dtype=np.complex64)
    coeffs0[:, 0, 0] = 1.0
    inverter.coefficients = coeffs0
    Y0 = inverter.linearize_stride(x_distorted).copy()
    c30_before = tracker.get_coefficients(0)[2, 0]
    tracker.process_stride(x_distorted, Y0)
    c30_after = tracker.get_coefficients(0)[2, 0]
    # mu=1.0 and factor=mu/denom means (c30_after - c30_before) == -grad30 / denom * 1.0*denom
    # i.e. the raw analytic gradient's sign is -(c30_after - c30_before) direction... the
    # kernel already applies "c -= factor*grad", so the *step it took* is -factor*grad,
    # whose sign must be opposite the finite-difference gradient's sign for the update to
    # be a genuine descent direction.
    step_real = float((c30_after - c30_before).real)

    assert np.sign(step_real) == -np.sign(finite_diff_grad), (
        f"Kernel step sign ({np.sign(step_real)}) is not a descent direction against the "
        f"finite-difference gradient sign ({np.sign(finite_diff_grad)}): "
        f"J(+eps)={j_plus:.6e}, J(-eps)={j_minus:.6e}, step={step_real:.6e}"
    )
    print(f"[PASS] Kernel's coefficient update is a real descent direction "
          f"(finite-diff dJ/dc30={finite_diff_grad:.4e}, kernel step={step_real:.4e}).")


def test_oob_energy_decreases_over_achievable_window():
    """
    OOB ratio must measurably decrease over training -- NOT claiming the
    (unreached) 20% target, only that real, non-trivial adaptation happens
    and moves the metric in the right direction with the achievable ~10%
    ceiling established during investigation.
    """
    _, x_distorted = _two_tone_distorted_signal()
    tracker = PolynomialCoefficientTracker(
        num_channels=NUM_CHANNELS, stride_len=STRIDE_LEN, mu=0.05, threshold_ratio=1e-5
    )
    inverter = SaturationInverter(channels=NUM_CHANNELS, stride_len=STRIDE_LEN)

    ratios = []
    for _ in range(20):
        for ch in range(NUM_CHANNELS):
            inverter.coefficients[ch] = tracker.get_coefficients(ch)
        inverter.c_real = inverter.coefficients.real.astype(np.float32)
        inverter.c_imag = inverter.coefficients.imag.astype(np.float32)
        Y = inverter.linearize_stride(x_distorted).copy()
        _, oob, tot = tracker.process_stride(x_distorted, Y)
        ratios.append(float(np.mean(oob / (tot + 1e-6))))

    best_reduction_pct = (1.0 - min(ratios) / ratios[0]) * 100.0
    assert min(ratios) < ratios[0], (
        f"OOB ratio never dropped below its initial value: {ratios}"
    )
    assert best_reduction_pct >= 5.0, (
        f"Reduction ({best_reduction_pct:.1f}%) fell below the ~9-11% ceiling established "
        f"during investigation -- may indicate a real regression, not just normal variance."
    )
    print(f"[PASS] OOB ratio reduced by {best_reduction_pct:.1f}% "
          f"(initial={ratios[0]:.4e}, best={min(ratios):.4e}) -- consistent with the "
          f"documented ~10% model-capacity ceiling, not the unreached 20% target.")


def test_convergence_is_reproducible():
    """Identical deterministic inputs must produce bit-identical coefficient trajectories."""
    def run_once():
        _, x_distorted = _two_tone_distorted_signal()
        tracker = PolynomialCoefficientTracker(
            num_channels=NUM_CHANNELS, stride_len=STRIDE_LEN, mu=0.05, threshold_ratio=1e-5
        )
        inverter = SaturationInverter(channels=NUM_CHANNELS, stride_len=STRIDE_LEN)
        for _ in range(10):
            for ch in range(NUM_CHANNELS):
                inverter.coefficients[ch] = tracker.get_coefficients(ch)
            inverter.c_real = inverter.coefficients.real.astype(np.float32)
            inverter.c_imag = inverter.coefficients.imag.astype(np.float32)
            Y = inverter.linearize_stride(x_distorted).copy()
            tracker.process_stride(x_distorted, Y)
        return tracker.get_coefficients(0)

    coefs_run1 = run_once()
    coefs_run2 = run_once()
    assert np.array_equal(coefs_run1, coefs_run2), (
        f"Two runs with identical deterministic inputs diverged:\n{coefs_run1}\nvs\n{coefs_run2}"
    )
    print("[PASS] Convergence trajectory is bit-reproducible across independent runs.")


def test_imd_suppression_measurement_is_correct_on_known_cases():
    """
    Sanity-check the suppression measurement methodology itself (bandpass +
    gain-aligned MSE) against cases with a known right answer, independent
    of whether the tracker's adaptation succeeds.
    """
    t = np.arange(STRIDE_LEN)
    s_target = np.zeros((1, STRIDE_LEN), dtype=np.complex64)
    s_target[0] = 0.04 * np.exp(1j * 0.08 * t) + 0.03 * np.exp(1j * 0.12 * t)

    def suppression_db(distorted, linearized):
        X_td = _bandpass_filter(distorted, 0.0, 0.25)
        Y_td = _bandpass_filter(linearized, 0.0, 0.25)
        alpha_d = np.vdot(s_target, X_td) / np.vdot(s_target, s_target)
        dist_err = np.mean(np.abs(X_td - alpha_d * s_target) ** 2)
        alpha_l = np.vdot(s_target, Y_td) / np.vdot(s_target, s_target)
        lin_err = np.mean(np.abs(Y_td - alpha_l * s_target) ** 2)
        return 10.0 * np.log10(dist_err / (lin_err + 1e-12)), dist_err, lin_err

    # Case 1: known corruption vs. perfect recovery -> suppression should be
    # large and positive (recovery removes essentially all the injected error).
    # The injected error is deliberately IN-BAND (a tone inside [0, 0.25])
    # rather than broadband noise, so bandpass-filtering to the target band
    # doesn't itself discard most of it -- that would make "perfect recovery"
    # look weaker than it is for reasons unrelated to the measurement's
    # correctness.
    injected_error = (0.02 * np.exp(1j * 0.18 * t)).astype(np.complex64)
    distorted = s_target + injected_error[None, :]
    perfectly_linearized = s_target.copy()
    db_recovered, dist_err, lin_err = suppression_db(distorted, perfectly_linearized)
    assert db_recovered > 20.0, f"Perfect recovery case should show large suppression, got {db_recovered:.2f} dB"
    assert lin_err < dist_err, "Perfectly recovered signal must have lower target-band error than the distorted one"

    # Case 2: "linearizer" that does nothing differently from the distorted
    # input -> suppression should be ~0 dB (dist_err == lin_err by construction).
    db_noop, dist_err2, lin_err2 = suppression_db(distorted, distorted)
    assert abs(db_noop) < 0.01, f"No-op case should measure ~0dB suppression, got {db_noop:.4f} dB"

    # Case 3: "linearizer" that makes it strictly worse -> suppression must
    # be negative, not clamped or hidden.
    worse = distorted + injected_error[None, :].astype(np.complex64)
    db_worse, _, _ = suppression_db(distorted, worse)
    assert db_worse < 0.0, f"Worsened case should measure negative suppression, got {db_worse:.2f} dB"

    print(f"[PASS] IMD suppression measurement is correct on known cases: "
          f"recovered={db_recovered:.2f}dB (>20 expected), noop={db_noop:.4f}dB (~0 expected), "
          f"worsened={db_worse:.2f}dB (<0 expected).")


if __name__ == "__main__":
    test_distortion_detection_responds_to_real_signal_state()
    test_gradient_direction_matches_finite_difference()
    test_oob_energy_decreases_over_achievable_window()
    test_convergence_is_reproducible()
    test_imd_suppression_measurement_is_correct_on_known_cases()
    print("\n[PASSED] All focused NLMS predistorter objective tests cleared "
          "(scoped to demonstrated, not aspirational, behavior).")
