# Phase 5 Engineering Report — Fusing the Tracking-Loop NumPy Glue

**Scope:** the per-cycle "glue" of the tracking loop — phase synchronisation,
discriminator, Kalman prediction/compounding, and the constant SNR feed. New
module `backend/src/satcom_core/tracking_loop_kernels.py`; integration in
`tests/tracking_loop_verifier.py`. The PRN synthesizer (Phase 4), EML
correlator, Kalman kernel, and predistorter were **not** modified.

**Question:** can the ~10 µs of tiny NumPy/Python per-cycle operations be safely
fused into Numba and actually removed?

**Answer:** yes. The glue drops ~80% (measured load-fair), the fused path is
**bit-identical** to the inline path, and tracking accuracy is unchanged. The
full loop still does not meet <23 µs — that target is structurally out of reach
because the two real-compute stages alone (PRN synth + correlator) already
exceed it — so the `tracking_loop_verifier` latency assertion is left unweakened
and remains a documented miss.

All numbers are honest, unscaled measurements from the deterministic
numpy/Numba simulation. **Important host-state caveat, quantified in §4:** this
session's host ran ~3.3× slower than the Phase-4 session (the *untouched* PRN
synth measured ~81 µs here vs ~25 µs in Phase 4 — a CPU clock/power-state
difference, not a code change), so absolute cycle latencies are inflated. The
robust, reproducible result is the **same-process A/B**, which is immune to that
host drift.

---

## 1. What changed

Profiling the timed region of `tracking_loop_verifier.py` with per-operation
timers (host in its fast state, PRN synth ~25 µs) attributed the glue as:

| glue op (inline NumPy) | median |
|---|---|
| phase sync (`states[:,0] % L`, `states[:,1]/sr`) | 2.7 µs |
| discriminator `0.5*(I_L-I_E)/(I_E+I_L+1e-12)` | 2.3 µs |
| Kalman-predict compounding (`x0_p`, `absolute_z`) | 2.5 µs |
| SNR feed `np.ones(N)*10.0` (per-cycle alloc) | 1.7 µs |
| **glue total** | **~9.2 µs** |

Each is a tiny operation on a 4-element array whose cost is essentially NumPy
dispatch/allocation overhead, not arithmetic. Two zero-allocation Numba kernels
replace them (one on each side of the synth+correlator pair, which sit in the
middle of the data flow, so a single kernel is not possible):

- `_sync_replica_drivers(states, code_length, sample_rate, phases_out, steps_out)`
- `_discriminate_and_track(I_E, I_L, states, covariances, T, half_T2, …, snr, disc_out, z)`
  — discriminator + absolute-measurement compounding + Kalman predict/update,
  reusing the **existing** `_update_kalman_loops` via an inlined njit-to-njit
  call (so the Kalman math cannot diverge). The constant SNR feed is hoisted to
  a preallocated buffer.

Both kernels use `fastmath=False` so the division and compounding associativity
stay bit-identical to the NumPy they replace (fastmath's fast-reciprocal
division was the only source of any difference and buys no speed on 4-element
loops).

---

## 2. Correctness — bit-identical

`tests/test_tracking_loop_fusion.py`, 2000-cycle closed loop with the 4 g step,
fused vs the exact inline NumPy on identical inputs:

| quantity | max difference over 2000 cycles |
|---|---|
| phase sync (code phases, code steps) | **0.0** |
| discriminator error | **0.0** |
| Kalman state trajectory | **0.0** |
| Kalman covariance trajectory | **0.0** |

End-to-end, the verifier's **maximum residual tracking error is 0.011993
chips** — identical to the pre-fusion value, still well within the 0.02-chip
tolerance. The fusion changes timing only, not behaviour.

---

## 3. Latency — glue before / after (the direct answer)

Because absolute cycle latency depends on host clock state (see §4), the
reliable measurement is a **same-process A/B**: inline glue vs fused glue run
back-to-back on identical inputs each iteration, so any host drift biases both
equally. Two independent runs:

| | P50 | P95 | P99 |
|---|---|---|---|
| inline glue (sync + disc + predict + SNR + Kalman) | ~17.2–17.6 µs | ~22–24 µs | ~61–77 µs |
| **fused glue** | **~3.4–3.5 µs** | ~4.7 µs | ~12–16 µs |
| reduction | **~80%** (−14 µs at this host state) | | |

At the Phase-4 host clock (fast state, measured early this session) the same
comparison is **glue 9.2 µs → 3.1 µs (−6.1 µs, −66%)**. Either way, the ~10 µs
of NumPy glue is genuinely removable, down to ~3 µs of real kernel work.

Breakdown of the fused glue (fast state): `sync_drivers` ~1.0 µs,
`discriminate_and_track` (incl. the nested Kalman update) ~2.1 µs.

---

## 4. Full-loop latency + the host-state caveat

**Host-state proof.** The PRN synthesizer was *not* touched this phase, yet it
measured ~81 µs median here vs ~25 µs in Phase 4 — a ~3.3× host-wide slowdown
(CPU clock/power state; 180 s idle did not recover it). Every Numba stage is
inflated by roughly the same factor, so absolute cycle latencies from this
session are **not** comparable to Phase 4's ~44.6 µs figure. The same-process
A/B below removes that confound.

**Same-process full-loop A/B** (identical synth+correlator, glue inline vs
fused, per-cycle interleaved, current host state), 2000 cycles:

| | P50 | P95 | P99 | max |
|---|---|---|---|---|
| inline-glue loop | ~209 µs | ~374 µs | ~566 µs | ~2697 µs |
| **fused-glue loop** | **~134 µs** | ~262 µs | ~427 µs | ~1526 µs |

Kalman state trajectories of the two loops matched to 0.0 (bit-identical).

**Expressed at the Phase-4 host clock** (the only apples-to-apples anchor to the
prior report): loop ~44.6 µs → **~38 µs** median (glue 9.2 → ~3 µs). The p95/p99
tails in every run are dominated by OS scheduling jitter, not compute.

---

## 5. Was <23 µs achieved? No — and it cannot be here

Removing the glue was the point of this phase and it succeeded, but it does not
reach the target and never could: the two irreducible real-compute stages alone
exceed it.

At the Phase-4 host clock:

| stage | median | touchable? |
|---|---|---|
| PRN synth (Phase-4 LUT) | ~25 µs | no (out of scope; already optimised) |
| EML correlator | ~8.6 µs | no (already a tuned single-pass Numba kernel) |
| fused glue (was ~9.2) | ~3 µs | **this phase** |
| **loop total** | **~38 µs** | |

`synth + correlator = ~33.6 µs > 23 µs` before any glue at all. The glue was
never the thing standing between the loop and 23 µs; it was ~9 µs of a ~44 µs
loop, now ~3 µs of a ~38 µs loop. **<23 µs is not achievable with this
architecture on this host**, and on the current (throttled) host it is much
further out. `tests/tracking_loop_verifier.py`'s 23 µs assertion is left
**unchanged** and remains a documented, honestly-measured miss.

---

## 6. Remaining bottleneck

1. **PRN synth (~25 µs at fast clock, ~65% of the optimised loop)** — the single
   largest stage. It writes 3 × 4 × 4096 complex64 replica samples per cycle;
   further reduction would require changing the E/P/L contract (e.g. real-valued
   replicas / dropping the unused prompt for the tracker), explicitly out of
   scope this phase and flagged in the Phase-4 report.
2. **EML correlator (~8.6 µs)** — real work over 4 × 4096 complex samples; already
   single-pass Numba.
3. **Host clock/power state** — the dominant *variable*: the same code runs
   ~3.3× faster or slower depending on the host's sustained clock. A stable
   real-time/embedded target (or a pinned high-performance host) is required
   before any absolute <23 µs claim is testable.

---

## 7. Tests

- `tests/test_tracking_loop_fusion.py` — **2/2 pass** (bit-identical equivalence
  over 2000 cycles; zero-allocation scratch reuse).
- Full pytest collection — **20/20 pass** (18 prior + 2 new).
- `tests/tracking_loop_verifier.py` — tracking-accuracy assertion **passes**
  (0.011993 < 0.02); the <23 µs latency assertion **fails** (unweakened,
  documented; further inflated by the current host state).
- Full standalone-script regression — see the run log; the only expected
  failures are the documented `tracking_loop_verifier` (latency) and
  `saturation_linearization_verifier` (Phase-3 25 dB), plus any host-timing-
  sensitive tests (`ipc_sync`, `energy_orchestration`) that are load-flaky and
  import none of this phase's modules.

---

## 8. Summary

The ~10 µs of tiny NumPy/Python glue in the tracking loop **can be removed** and
was: two zero-allocation Numba kernels reproduce the phase sync, discriminator,
and Kalman predict/update **bit-for-bit** while cutting that glue ~80% (to
~3 µs), leaving tracking accuracy untouched (0.011993 chips). At the Phase-4 host
clock the loop moves from ~44.6 µs to ~38 µs median. The <23 µs target remains
unreachable — the PRN synth (~25 µs) plus correlator (~8.6 µs) already exceed it,
so the loop, not the glue, is the limiter, and the synth is now the largest
remaining (out-of-scope) cost. The honest blocker beyond that is the host's
clock/power state, which varied ~3.3× within this work and must be pinned before
a <23 µs claim is even measurable.
