# Five-model surrogate-HMC performance repair result, 2026-10-12

The master program is executable and its repair phase has run. The repair is
**unfinished**: final-source full-size SIR accuracy and five-model practical
cost remain unresolved. No posterior HMC sampling or new admission ran in
this repair phase. The objective remains testing surrogate HMC, with a
deterministic proposal force and the declared independent Metropolis endpoint;
local filter checks are prerequisites, not a substitute for that objective.

## Implemented and reviewed

The consumer now carries all parameter sensitivities through one time
recursion, with shared primal particles, map factors, mixture quadratics and
Sinkhorn cost tiles. Mixture and transport algebra retain a leading batch
dimension. The current consumer groups four chains into two B=2 evaluations;
B=4 capacity is not established. Small fixed model-callback and batch tuples
are expanded during tracing. There is no runtime Python time/particle loop
in the repaired filter, and no complete T-step filter replay per parameter.
Repeated kernels use stable signatures and XLA. The independent endpoint
still iterates over chain positions inside a TensorFlow loop; the whole HMC
consumer has not become batch-native.

The score propagates explicit first-order forward sensitivities in TensorFlow.
Algebraically these are dual-number/JVP rules. This is the optional
covariance-guided finite filter; it is **not** canonical Contract-E LEDH
admission, and the analytical-jet abstraction does not make it one.

Two cancellation defects were repaired: common position/sensitivity offsets
in transport contractions and common sensitivity offsets in mixture-location
scores. The latter originally contracted large terms separately and then
subtracted them. Subtracting the common tangent before contraction preserves
the real-arithmetic formula and the primal mixture density. A componentwise
18-dimensional FP32 regression with a common tangent of 262144 failed before
this change and passes afterward against independently evaluated FP64 values.
The covariance quadratic was also factored without materializing a
parameter-by-batch-by-particle-pair-by-state array. These changes do not remove
the required quadratic-in-particle-count mixture/transport work.

The orchestrator preserves partial results and exact source snapshots on
timeout, checks XLA/host-callback status, records numerical traces, continues
independent models after one fails, separates compilation from warmed timing,
and writes a next-phase plan after each model. Non-finite diagnostic fields
are preserved explicitly rather than breaking JSON serialization. Diagnostic
completion is not reported as derivative or HMC success. Orchestration,
consumer wiring, failure isolation, and arithmetic checks are covered by the
focused regression suite. Future mathematical repairs still require a reviewed
implementation; the program does not automatically invent algorithms.

## Verified on the final source

- 59 CPU regression tests pass (75.44 seconds including startup).
- All five N64/T2/B2 FP64 unit cases pass old/new value-score parity and an
  independent finite-value derivative check (108.11 seconds total).
- All five kernels report XLA required, one trace, and no host callbacks.
- Source dependency hashes from these unit runs match the current source.
- 24 exact command forms in the narrow allow list were verified as allowed.

The CPU choice is explicit: CUDA_VISIBLE_DEVICES=-1. These are algebra and
wiring checks, not FP32 N4032/T50 evidence. Existing small off-centre,
nonzero-correction, and longer SIR checks precede the final mixture-centering
change and retain their own saved source; they cannot certify final-source
full-size behavior.

Evidence: artifacts/surrogate-hmc-cpu-regression-after-mixture-20261012-01/,
artifacts/surrogate-hmc-five-model-unit-after-mixture-20261012-01/, and
artifacts/surrogate-hmc-mixture-cancellation-before-20261012-01/.

## Earlier full-size GPU findings

These paired timings use N4032, FP32 without TF32, XLA, B=2 and frozen saved
controls. They precede the final arithmetic repairs. The table is descriptive,
based on two warmed paired repeats; it cannot rank methods statistically or
establish current-source speed. Ratios above one favor the repaired candidate.

| Model | Old median seconds | Candidate median seconds | Old / candidate | Allocator peak GiB |
|---|---:|---:|---:|---:|
| lgssm | 92.04 | 48.82 | 1.885 | 0.711 |
| ksc | 11.04 | 14.04 | 0.786 | 0.252 |
| predator_prey | 65.55 | 40.37 | 1.624 | 0.833 |
| range_bearing | 13.29 | 14.62 | 0.909 | 0.655 |

All four rows passed the then-declared old/new parity check. KSC and range
bearing became slower in these measurements: this is a repair trigger, not
a successful all-model performance result. Compilation was recorded separately.
The LGSSM candidate still took roughly 49 seconds per two-row score evaluation;
eliminating redundant parameter sweeps alone does not make HMC affordable.
Frozen old tuning controls are comparators, not fresh tuned settings after
source changes. The summary JSON links each measurement to its source result.

SIR N4032/T50 on the preceding FP32 source returned parameter-0 score
-50.706314 versus central finite differences -168.041992 (h=.005) and
-173.339828 (h=.001); all brackets completed 50 valid steps. This is a large
unresolved discrepancy. At T10 with the same FP32-quantized inputs promoted
to FP64, the score was -177.431673 and the first finite difference was
-174.102016 (h=.001). The width ladder timed out before convergence could be
checked. A full T50 FP64 attempt also timed out. Neither timeout supplies a
passing numerical result, and none of these figures uses the final
mixture-centering repair.

The small N64/T10 SIR failure had a different explanation: finite-difference
truncation at h=1e-5. Two converged Richardson estimates from h=1e-5,3e-6,1e-6
agreed with the analytical derivative under the existing strict tolerance.
That resolves the small-case check, not the full-size FP32 discrepancy.
Old/new scores alone cannot arbitrate SIR: the earlier directional baseline
itself changed materially with compiled graph layout.

Evidence: artifacts/surrogate-hmc-sir-final-source-fd-20261012-01/ (despite its
historical directory name, this is not the current source),
artifacts/surrogate-hmc-sir-prefix-fp64-fd-20261012-01/,
artifacts/surrogate-hmc-sir-full-fp64-reference-20261012-01/, and
artifacts/surrogate-hmc-sir-unit-convergence-20261012-01/.

## Review conclusions

The review checked the consumer-to-kernel call chain and numerical derivative
rules, not just the existence of an all-parameter helper. It found and corrected
an unreliable directional baseline interpretation, a single-width derivative
check, large-offset FP32 cancellation, timeout/result classification, and
source-provenance gaps. Local executable checks support the corrections.
Full N4032 final-source and end-to-end HMC execution remain untested.
The public HMC tuning interface and capability registry restrict this
position-only-force binding to conditional mechanics; there is no exact-score
artifact authority or canonical LEDH status from this work.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep implementation as an unpromoted candidate | Final small fixtures pass | No small-fixture numerical veto | Full-size FP32 behavior | Final-source SIR precision localization | Complete score repair |
| Continue shared-cost repair | Mixed earlier timings | KSC/range cost benefit absent | Final-source steady-state cost | Profile shared kernels and repeat paired timing | All-model speedup |
| Do not run posterior HMC yet | Full-size score/tuning criteria incomplete | SIR numerical discrepancy unresolved; source-specific tuning stale | Cost and force adequacy | Resolve derivative issues, retune each eligible scope | Posterior convergence or model validity |
| Pause further GPU attempts at current allocation | 3586.062/3600 seconds; 13/13 attempts | Budget continuation veto | Additional repair time | Allocate the next bounded repair phase | Rejection of surrogate HMC |

| Inference status | Finding |
|---|---|
| Hard veto screen | Earlier full-size SIR derivative mismatch; FP64 reference timeouts; final-source full-size checks absent |
| Statistically supported ranking | None |
| Descriptive-only differences | Four old/new timing ratios, memory, and finite-value/score gaps |
| Default readiness | No promotion; FP32/N4032 remain owner-requested testing settings |
| Next evidence needed | Converged same-input SIR derivative checks, final-source all-model checks, repeated timing, fresh tuning and full HMC evidence |

## Execution and continuation

The bounded allocation used **3586.062 GPU-wall seconds of 3600**, across
13 model attempts. 13.938 seconds remain, less than the smallest
useful next compile/run. CPU-only tests are accounted separately. Every GPU
run used the escalated fixed-interpreter/script form; no broad sandbox bypass
was installed. Per-run manifests preserve command, commit/source hashes,
conda interpreter, data identity, seed, devices, memory-growth settings,
precision, timing, and logs. The combined manifest and budget ledger are in
artifacts/surrogate-hmc-performance-repair-summary-20261012-01/summary.json.

The next work is **part of this master**, not a change in research direction:
first recheck final-source SIR at N4032 on short saved prefixes, then complete
T50 checks; repair KSC/range costs; finally repeat the full five-model checks
and perform fresh own-scope tuning where eligible. A proposed additional
7200-second GPU allocation is split 1800/1800/3600 seconds across those stages.
It is not yet authorized, does not guarantee closure, and does not include
posterior HMC. The prior and full adaptive-HMC campaign allocation remain
separate pending choices. See the machine-readable next-phase plan beside the
summary. Candidate failure remains a repair trigger; only the stated budget
stops GPU execution here.

The strongest alternative explanation is scope-dependent conditioning and
FP32 roundoff rather than a missing derivative term. Converged full-size FP64
finite-value checks followed by same-input FP32 localization would distinguish
that explanation. The weakest current evidence is the absence of any
full-size run after the final mixture repair. A clean final-source test would
resolve that numerical veto; it would still leave computational value and
posterior testing to establish.
