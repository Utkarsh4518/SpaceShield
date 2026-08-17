# Phase 4 Engineering Report — Fractional-Phase LUT PRN Synthesizer

**Scope:** `backend/src/satcom_core/prn_code_synthesizer.py` and its tests. No
other DSP modules were modified. The predistorter (Phase 3) was not touched.

**Question:** can a precomputed fractional-phase PRN lookup table materially
reduce PRN synthesis latency, and does it bring the tracking loop under the
historical <23 µs target?

All numbers are honest, unscaled measurements from the deterministic
numpy/Numba simulation on the reference dev host (a Windows VM). Nothing here is
hardware-, HIL-, or real-time-target validated. Latency on a VM carries heavy
tail jitter from OS scheduling; medians are the stable figures, p95/p99/max are
reported as measured.

---

## 1. What changed

The PRN synthesizer's per-sample hot loop was reworked from:

- **before:** float64 phase accumulator, per-sample `int()` truncation, and
  per-sample bit-table extraction (`word>>5`, `bit&31`, shift, mask, `1-2*bit`)
  for each of Early/Prompt/Late;

to a **fractional-phase lookup table**:

- code phase carried as an **int64 in units of 2⁻ᶠʳᵃᶜ_ᵇⁱᵗˢ chips** — stepping is
  an integer add, and the chip index is a single right shift by `frac_bits`;
- Early/Prompt/Late values read directly from a **precomputed unpacked ±1 code
  LUT** (the packed table is expanded once and cached; re-expanded only when the
  code table changes);
- no per-sample floating point and no per-sample bit extraction.

`frac_bits` is a configurable constructor parameter (default 32) — the
"fractional-phase resolution" knob. The public API (`synthesize_stride`,
`code_phases` as float64, complex64 E/P/L) is unchanged.

**Design alternatives that were benchmarked and rejected** (see §5):
- *Oversampling the LUT itself* (storing R sub-chip copies of each chip): for
  pure BPSK ±1 chips the value depends only on the integer chip index, so
  oversampling adds no information and only wastes cache (R=256 → 4 MB table,
  measurably slower). The table stays length `code_length`; the fractional
  resolution lives in the accumulator, not the table.
- *float32 real output* / *no-prompt fast path*: only marginally faster and would
  change the public complex64 E/P/L contract. Not adopted.

---

## 2. PRN synthesizer latency — before / after

Measured over 2000–3000 warm strides, 4 targets × 4096 samples:

| | median | p95 | p99 | max |
|---|---|---|---|---|
| **before** (bit-packed float) | **~78 µs** | ~97 µs | ~137 µs | ~314 µs |
| **after** (fractional-phase LUT) | **~24–26 µs** | ~39–51 µs | ~49–82 µs | ~170–280 µs |

**≈ 3.2× median reduction.** (The p95/p99/max spread is dominated by VM
scheduling jitter, not the kernel; the median is the reliable figure.)

Cost attribution from the profiling variants (single stride, median):

| variant | median |
|---|---|
| V0 baseline (bit-packed, complex64 E/P/L) | 77 µs |
| unpacked-LUT read (removes bit extraction) | 51 µs |
| skip prompt (E/L only) | 61 µs |
| **fixed-point integer accumulator + LUT (adopted)** | **~24 µs** |

The dominant cost was the per-sample **float64 arithmetic + float→int
truncation**, eliminated by the integer accumulator; bit-extraction removal
(unpacked LUT) was secondary.

---

## 3. Tracking-loop latency — before / after

Full timed region of `tests/tracking_loop_verifier.py`, 2000 cycles, median µs:

| stage | before | after |
|---|---|---|
| phase sync (numpy) | 3.3 | 3.0 |
| **PRN synth** | **78.2** | **25.1** |
| EML correlator | 8.8 | 8.6 |
| discriminator + Kalman predict (numpy) | 6.8 | 6.5 |
| Kalman filter | 1.0 | 0.9 |
| **total loop** | **~98.6 (≈97–101)** | **~44.6 (≈44–45)** |

**≈ 2.2× loop median reduction** (`tracking_loop_verifier` reports 44.90 µs).

---

## 4. Tracking accuracy impact

Numerical/code-phase equivalence to the original truncation algorithm was
verified directly (`tests/test_prn_synthesizer_lut.py`):

- code phase tracks the float64 reference to **< 5e-7 chips**;
- only **~1e-5 of samples** differ, and only by a ±1-chip tie exactly on a chip
  boundary (sub-2⁻³² rounding) — not a tracking error;
- `frac_bits` is a clean accuracy knob: mismatches vs the float reference within
  a stride were `{8: 7618, 16: 56, 24: 0, 32: 0}` — 24 and 32 bits are exact.

End-to-end, the closed-loop verifier's **maximum residual tracking error is
0.011993 chips**, unchanged from the pre-LUT loop and well within the 0.02-chip
tolerance. The high-dynamic 4 g pull-in behavior is preserved.

---

## 5. Tests passed / failed

- `tests/test_prn_synthesizer_lut.py` — **4/4 pass** (equivalence, `frac_bits`
  resolution sweep, determinism/API shapes, LUT-refresh-on-code-change).
- Full pytest collection — **18/18 pass** (14 pre-existing + 4 new).
- `tests/tracking_loop_verifier.py` — **tracking-accuracy assertion passes**
  (0.011993 < 0.02); the **<23 µs latency assertion still fails** at ~45 µs. Its
  threshold was left **unchanged** (not weakened); this remains a documented,
  honestly-measured miss, now ~2× closer than before.
- No other module was modified, so the rest of the suite is unaffected.

---

## 6. Is <23 µs realistically achievable with the current architecture?

**No — not for the full loop on this host.** The synth is no longer the
blocker: with it at ~25 µs, the **non-synth floor is ~19 µs median** (correlator
~8.6 µs + per-cycle numpy glue for phase-sync/discriminator/Kalman-predict
~10 µs + Kalman ~1 µs). Even a hypothetical **zero-cost synth** leaves the loop
at ~19 µs median and ~60 µs p99, so <23 µs for the whole cycle is not reachable
here.

The remaining gap is **mixed**:
- *Algorithmic (irreducible-ish):* the correlator does real work over
  4 targets × 4096 complex samples (~8.6 µs, already a tuned single-pass Numba
  kernel), and the synth must still write 3 × 4 × 4096 complex64 samples.
- *Host / environment (not fundamental):* ~10 µs of the floor is Python/numpy
  per-call overhead on tiny 4-element arrays (phase sync, discriminator, Kalman
  predict), which would be near-zero in a fused kernel or on an embedded target;
  and the p95/p99/max tail (loop max spiked to ~400 µs–1.6 ms in places) is OS/VM
  scheduling jitter, not compute.

So the synth optimization was worth doing and is genuine (−53 µs off the loop),
but the <23 µs target is not realistic on this VM with the loop still stitched
together in Python.

---

## 7. Recommended next step

1. **Fuse the per-cycle numpy glue into one Numba kernel.** The phase-sync
   (`states % code_length`, `/sample_rate`), discriminator
   (`0.5*(I_L-I_E)/(I_E+I_L)`), and Kalman prediction are all tiny 4-element
   operations currently paying full numpy dispatch (~10 µs total). Folding them
   (and ideally the Kalman update) into a single kernel would remove most of the
   non-synth floor and bring the median toward ~30–33 µs. This is the highest-
   value remaining software change and touches only the tracking loop.
2. **Re-measure on a real embedded/bare-metal target** (or at least a
   non-virtualized host with CPU pinning). The p95/p99/max tail here is VM
   scheduling, not algorithm; a real-time target would collapse the tail and the
   interpreter overhead, and only there is a <23 µs claim testable honestly.
3. Keep `frac_bits=32`. It is exact within a stride, and lowering it buys no
   speed (resolution is accumulator-only, not table size) while adding chip-
   boundary error.

Do **not** pursue further micro-optimization of the synth kernel itself — at
~25 µs it is below the ~19 µs non-synth floor's relevance, and the loop, not the
synth, is now the limiter.
