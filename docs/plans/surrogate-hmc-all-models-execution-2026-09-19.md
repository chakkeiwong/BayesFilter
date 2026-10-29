# Surrogate-HMC master execution — 2026-09-19

Continuation status, September 20: the public TensorFlow tuning limitation
described in the first-tranche sections below has been repaired and GPU
qualified. LG1 A/B/C retained comparisons have completed their planned looks;
see the [inference result](surrogate-hmc-lg1-inference-result-2026-09-20.md)
and [finite comparisons](surrogate-hmc-lg1-finite-result-2026-09-20.md).
The executable master now supports LG2 reference, full-consumer qualification
and scoped control calibration. Its qualified reference and consumer are
recorded in the [LG2 consumer result](surrogate-hmc-lg2-consumer-result-2026-09-20.md).
LG2 controls selected multiplier 0.01, but its inference forecast exceeded the
initial allocation; retention is deferred. C01 Gaussian mechanism controls then
completed: three arms are INCONCLUSIVE and zero-force tuning was rejected. See
the [control result](surrogate-hmc-gaussian-controls-result-2026-09-20.md).
Next, calibrate the batch-correlation uncertainty veto on independent exact
draws before using it for further model decisions. Closed criteria stay fixed.
Historical first-tranche blockers below must not be mistaken for current ones.
All-model coverage and scientific validation remain incomplete.

Active question: does HMC driven by the imperfect **full canonical DCTR force**
recover each model's parameter posterior, and at what total cost? The governing
[master](surrogate-hmc-all-models-master-program-2026-09-18.md) fixes the four
comparisons B/R, C/D, D/R and C/R. A filtering or force error alone cannot answer
that question or veto the B/R experiment.

The owner authorized revision, thorough review and execution on September 18.
The Opus reply withdraws F1. Separate frozen endpoint and force identities are
valid; each must remain internally consistent. Added precision and prior
reporting clarifications do not change the target or numerical criteria.

## First execution tranche and evidence contract

P0 builds a source-backed model register with explicit unresolved scopes. P1
tests the real LG1 DCTR consumer, prior/chart and parameter-dependent stationary
initialization, independent endpoint/force freezing, row independence, replay,
per-chain energy correction and actual GPU/XLA cost. Exact scalar Kalman
filtering and the shared corrected leapfrog provide the comparison authorities.
The DCTR call must resolve to `canonical_batch_fused_value_score` with all
corrections and Contract E–Chol enabled. No old reduced HMC adapter is eligible.

Primary P1 criterion: valid implementation, correct target/change of variables,
independent frozen bindings and a finite reversible proposal within measured
floating-point tolerances. Missing status, cross-chain coupling, changing
endpoint/cache, nonfinite accepted states, or a wrong law are continuation
vetoes. Filtering residuals, force disagreement, acceptance and speed are
explanatory; cost alone determines whether the declared inference schedule fits.
P1 cannot establish convergence, posterior recovery, or surrogate-HMC failure.

Start with attempt 4 diagnostic/calibration seeds, LG1 T=10, N=1024, K=1024,
four unequal chain positions, FP64 integration and Kalman endpoints, and FP32
DCTR with TF32/XLA. Run no claim inference before scoped DCTR selection and a
public admission-capable HMC tuning artifact exist. New source inspection found
the TensorFlow public-tuner configuration is explicitly diagnostic-only and
rejects XLA; its legacy admission path uses NumPy. This is an implementation
gap to test and repair, never permission to label a diagnostic as admission.

P1 baseline controls come from the post-invalidation September 17 full-route
calibration, as warm-start hypotheses only. The previous cap evaluation selected
16 for an LG1 diagnostic; neither those settings nor filter-screen outcomes are
inference admission. Early checks retain original validity and expose numerical
failures. Class C controls require fresh non-harm calibration before selection.
N=1024, four chains and FP64 integration come from master sections 9–10 and
remain scope-specific hypotheses. Uniform(-.95,.95), stationary x0, Q=1, R=.5
define the declared synthetic benchmark exactly. No empirical prior claim is
made. CPU checks deliberately hide GPUs; production-oriented probes require
trusted GPU access and verified memory growth before initialization.

Skeptical pre-run audit: the baseline is the same-law Kalman posterior; no
filter/probe metric grants posterior promotion; full-route status and binding
tests precede inference; source and dtype identities are explicit; unresolved
adapters stay in coverage. The diagnostic-tuner limitation is material and
prevents posterior launching until repaired. Proceed with P0/P1 mechanics and
timing, which answer the next readiness questions without concealing this gap.

Budget: master maximum 226800 GPU seconds / 54000 CPU-worker seconds, including
failures and compilation. Initial P0/P1 caps are 3600 GPU seconds / 7200 CPU
seconds, with individual GPU probes at most 900 seconds. Use one GPU scope at a
time and at most four CPU workers. Measured actual transition cost must support
the full frozen inference schedule with the master 1.5 safety factor; do not
launch a truncated posterior campaign as a substitute. The authorized two LG1
inference candidates retain indices 4 and 5. Mechanics-only checks do not spend
a candidate attempt, but every computation spends the time budget.

Outputs: `docs/plans/artifacts/surrogate-hmc-all-models-20260918T155904Z-01/`.
Preserve exact commands, environment, source hashes, logs, device/memory policy,
per-stage wall and CPU usage, `model-register.json`, scope cards, checks,
`decision.json` and `result.md`. The master driver must create new output
directories exclusively and fail closed for unimplemented inference stages.
Final reporting must distinguish incomplete implementation from a rejected
surrogate-force candidate. Follow the master's decision and inference tables.

## Bounded review follow-up

The first full GPU probe passed mechanics at four unequal positions. It spent
48.344 GPU seconds and 53.967 total CPU seconds, including inventory. The
four-chain L=2 transition took 1.711 seconds; the conditional 2000 warm-up plus
2000 retained schedule forecasts 2.868 GPU hours with the 1.5 factor, **excluding
tuning and reference work**. That cannot authorize P2's complete comparison.

One fresh P1 follow-up, also capped at 900 GPU seconds within the same tranche,
will exercise the actual public tuner with this full LG1 consumer. One mass,
one adaptation and one verification transition are interface diagnostics only.
Their unit scale, epsilon .01, target acceptance .7 and one leapfrog step are
explicit convenience hypotheses; broad acceptance bands test interface health,
not mixing or admission. The outer TensorFlow tuner requires non-XLA graph mode;
this is an explicit diagnostic exception, with the nested DCTR force still
GPU/XLA. The required outcome is a preserved diagnostic and rejection of a
retained-inference handoff. A missing admission capability is an implementation
continuation veto for inference, not rejection of the DCTR force or research
direction. No shared tuning authority will be silently changed.

The follow-up also compares scalar and batch calls, tests the effective force
used by the shared proposal, and preserves raw validity and nonfinite values.
The kernel already defines a deterministic zero-field extension on invalid
force rows; raw force failure must not be mistaken for invalid accepted state.
All raw diagnostics remain visible. Existing shared-kernel regression tests
check invalid-force fallback, endpoint status, kinetic energy and cache behavior.
CPU tests intentionally hide GPUs. No Jacobian test using implicit pfor runs.

Driver review repairs carry previous campaign charges into subsequent launches,
retain terminal accounting for launch errors and interruptions, and distinguish
job failure from incomplete implementation. Automatically discovered cost records
are under `artifacts/surrogate-hmc-all-models-*/accounting.json`; external test
charges can be supplied with `--previous-accounting`. Time limits supplied to
the CLI cap cumulative campaign use, not a fresh allocation per invocation.
Fresh follow-up output: `artifacts/surrogate-hmc-all-models-20260919-review-02/`.
