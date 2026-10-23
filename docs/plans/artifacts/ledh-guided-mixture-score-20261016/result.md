# Guided mixture comparison: numerical screen stopped the final stress case

The comparison reached336 valid rows before the identity-proposal arm failed
the shared reset's moment-consistency check. The failure occurs in the
persistent-drift/impulse condition, final seed100204, observation3. All four
score coordinates remain finite. CPU and GPU reproduce the rejection using
identical saved inputs. The discrepancy is small, about1e-13; this is not
evidence of materially inaccurate scores or failure of backward guidance.
It does prevent admitting this row under the predeclared validity screen.

## What was executed

The [plan](../../ledh-guided-mixture-score-comparison-20261016.md) compares
seven proposal arms through one shared consumer: identity, defensive global,
defensive local, equal mixture, calibrated current-only, fitted cSMC future,
and exact Gaussian future. Each condition has separate calibration,
validation and final seeds. Beta is frozen after independent calibration;
guide fitting uses independent pilots on each observation sequence. The
reference is the exact Kalman score of the matching predict-first model.
All four parameter directions are analytical in the candidate runtime.
OT remains epsilon1/five iterations, N32,T5, FP64 GPU/XLA, correction2.

Attempt01 completed all14 evaluation datasets in each of the first three
conditions and six datasets in condition3 (two validation, four final), with
seven arms each. Its next identity evaluation failed. The remaining final
rows were not run. The completeness check forbids dropping the failure or
reporting a paired ranking from this truncated comparison. This is an optional
mixture experiment, not canonical Contract-E/UKF admission or KDM training.

## Exact failure and localization

| Quantity at observation3 | GPU/XLA | CPU graph reference | Required limit |
| --- | ---: | ---: | ---: |
| Mean restoration error, after correction | 1.581e-13 | 1.956e-13 | CPU1.262e-13 |
| Relative covariance restoration error, after correction | 2.138e-13 | 4.825e-14 | 4.441e-14 |
| Effective sample size | 1.279 | 1.279 | Explanatory only |
| Largest normalized weight | .87556 | .87556 | Explanatory only |
| Reset minimum/maximum covariance eigenvalue ratio | 7.930e-6 | 7.930e-6 | Positive factorization margin |
| Transport column normalization error | 4.441e-16 | 4.441e-16 | Not the failed check |
| Coordinate correction displacement | .01621 | .01621 | .25 plus rounding allowance |

GPU02 records the original trace. CPU04 replays its serialized inputs;
CPU06 additionally exposes previously discarded reset diagnostics. Before
moment correction, CPU mean error is1.957e-13 and relative covariance error
4.860e-14, already exceeding their limits. The subsequent correction does
not cause the first exceedance. Its displacement check passes. The
pre-correction reset's finite-precision centering/recolouring is therefore
the next localization target. The eigenscale ratio and concentrated weights
suggest numerical amplification, but they do not prove its precise source.

The required mathematical target is exact restoration of the weighted mean
and covariance. The computed reset approximates these within the reported
floating-point residuals. Those residuals are too large for the implemented
screen, although tiny in absolute terms. Whether this is best repaired by a
more stable calculation or a derived conditioning-aware error bound remains
unchecked. No threshold was relaxed, no ridge was introduced, and no failed
score was accepted. Failure of this screen is not evidence against the
statistical idea or proof that the guard's threshold is well calibrated.

## Implementation and audit

New diagnostic wrappers call the existing mixture factory and map/density
functions, with frozen dynamic message coefficients. An executable wiring
test compares all four scores directly with that consumer and compares the
pilot target/densities with the same guided maps. The existing guide derivative
suite also passes. Fourteen unique focused tests pass: eight numerical/wiring
tests in CPU05 and six completeness/pairing/veto tests in CPU02. CPU devices
are explicitly hidden for those reference checks. Earlier repetitions are
not counted again.

Runtime changes in this stage append observability only: pre-correction mean
and covariance errors, the mean threshold, and moment/cap flags. The wrapper
also retains every direction's trace. Existing indices are preserved, controls
and numerical decisions are unchanged. The added fields were checked on CPU;
GPU02 predates the final additional fields. GPU behavior of the final logging
shape has not been rechecked, and no such claim is made.

Terminal skeptical audit: correct shared comparator and physical numerator;
same model/parameterization for Kalman; independent calibration and final data;
four conditional regimes; no optimized proxy promoted to score evidence;
failed final row retained; no post-failure tuning. Scientific ranking is
blocked by incompleteness, while diagnostic localization is complete.

Provenance limitations found and repaired for future runs: attempt01's
manifest labels the data stream1; source uses719. Also, the moment-correction
module is lazily imported during tracing and was omitted from the startup
loaded-module hash list. Its file is unchanged from the recorded Git HEAD,
SHA256 `91dac402c4594288df90f342112be41bd437a3dd72cf60ef0a705a7e32c03014`.
The runners now import it before hashing. Historical manifests remain intact;
these are explicit supplemental corrections, not retrospective relabeling.
GPU allocator current/peak bytes were not captured; memory growth, actual
RTX4080SUPER UUID, FP64/TF32/XLA and environment were captured.

## Decision

| Field | Result |
| --- | --- |
| Decision | Stop this comparison; continue reset-precision investigation |
| Primary criterion | Not assessable: final grid incomplete |
| Veto status | Shared moment-consistency screen rejects identity arm; no nonfinite score |
| Main uncertainty | Numerical source and justified error bound; model-score effect unevaluated |
| Next justified action | Diagnose centering and covariance recolouring on saved case; evaluate a bounded stability repair with analytical derivatives and healthy-case non-harm checks |
| Not concluded | Guide superiority, cSMC/KDM failure, canonical admission, default readiness, or materially harmful score error from the1e-13 moment residual |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Reproducible moment-screen failure; complete-comparison veto |
| Statistically supported ranking | None |
| Descriptive-only differences | Numerical residuals and concentration diagnostics; no arm ranking |
| Viable candidates | Guide mechanisms remain unassessed; this finite-precision baseline needs repair or a justified screen |
| Default readiness | Not established |
| Next evidence needed | Valid full comparison, conditional paired uncertainty, heuristic checks, complete cost accounting |

Post-run red team: the strongest alternative explanation is an overly strict
condition-independent tolerance rather than harmful reset error. A derived
error bound and downstream non-harm test could overturn a repair-first
interpretation. The weakest evidence is the absence of a completed final
comparison and per-arm total fitting/calibration cost. These limitations
prevent claims about guide effectiveness, cost efficiency, and population
quality. No bootstrap intervals were computed after the veto.

## Budget and evidence

Attempt01:230.850 GPU seconds. GPU02:19.345 seconds; plus preflight and
rounding, conservatively252/900 GPU seconds. Both GPU attempts are consumed.
CPU checks/replays are conservatively charged96/180 seconds. Unused wall
allowance is not a reason to bypass the attempt limit or numerical veto.
All processes have finished. No commit, push, merge, branch switch, or
environment mutation occurred.

Commands, seeds, source hashes and hardware are in `attempt01/manifest.json`,
`gpu02/manifest.json`, `cpu04/manifest.json`, and `cpu06/manifest.json`.
Raw valid rows are `attempt01/rows.jsonl`; failure is `attempt01/failure.json`;
reproducible input is `gpu02/inputs.json`. Logs and timed test results are in
`cpu01/`, `cpu02/`, `cpu03/`, `cpu05/` and the corresponding replay directories.
The [checkpoint](checkpoint.md) gives the next action.
