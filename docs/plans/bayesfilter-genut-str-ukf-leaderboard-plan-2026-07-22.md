# GenUT Chapter 18b Structural Leaderboard Plan

Date: 2026-07-22
Status: `AUTHORIZED_BOUNDED_EXECUTION_AFTER_SKEPTICAL_AUDIT`

## Research intent ledger

| Field | Frozen decision |
|---|---|
| Main question | Can the positive Gaussian-GenUT/Contract-E candidate evaluate the existing Chapter 18b `STR-UKF` target without changing its structural law, observation timing, frozen data, or source-probit parameterization, and produce finite value and manual recursive score evidence suitable for a GenUT candidate leaderboard entry? |
| Candidate | Existing non-fused `finite_value_score` route with a dimension-two positive Gaussian GenUT residual cloud and a target adapter that calls shared primitives from the established structural implementation. |
| Existing target | `STR-UKF-five-probit-T100-structural-innovation-v1`, frozen `T=100` data, parameter order `(rho_source_probit,sigma_source_probit,phi_source_probit,gamma_source_probit,R_source_probit)`. |
| Structural restriction | The only new transition shock is scalar `epsilon_t`. Every pre-reset propagated particle must satisfy `k_t-phi*k_(t-1)-gamma*m_t^2=0`. Contract E may reset the filtered posterior cloud in `(m,k)` after assimilation; it may not inject an independent process shock into `k`. |
| Comparator | Existing manual principal-square-root structural UKF likelihood value/score on the identical frozen data and source chart. It is a same-target deterministic approximation diagnostic, not an oracle and not a tuning target. |
| Expected failure mode | Structural timing mismatch, duplicated model equations drifting from the existing implementation, omitted chart derivative, observation-variance score error, incomplete recursive tangent, reset bias, score variance, dense `N^2` runtime, or GPU memory pressure. |
| Promotion criterion | Candidate leaderboard inclusion requires a valid route identity, `N>1000`, full `T=100`, FP32/TF32 GPU/XLA, manual recursive score, exact structural-residual hard gate, representative same-scalar FD audit, raw multi-seed values/scores, and honest comparator classification. |
| Promotion veto | Any artificial `k` process noise; transition before `y0`; structural residual above `2e-5` in FP32; nonfinite result; reset/marginal residual above `5e-4`; relative score-increment accounting residual above `2e-5`; FD relative error above 5%; missing GPU/XLA/TF32 or memory-growth evidence; `N<=1000`; stale route identity; or claim data used for control selection. |
| Continuation veto | The shared existing target cannot express the FP32 adapter without changing its law; the `N=1002,T=10` capacity probe exceeds 12 GiB or five minutes per seed; two localized implementation attempts fail; or the full bounded campaign would exceed 30 minutes. |
| Repair trigger | A local shape, dtype, XLA, diagnostic, serialization, or resource failure that leaves the scientific target and campaign budget unchanged. |
| Not concluded | No exact likelihood or score, no GenUT superiority, no unbiasedness, no HMC/default readiness, no high-dimensional scaling result, and no claim that UKF is truth. |

## Exact wiring

The existing implementation remains the authority for:

- source-probit bounds and chart;
- initial law `x_0 ~ N(0,diag(0.04,0.09))`;
- frozen observations and their hashes;
- `m_t=rho*m_(t-1)+sigma*epsilon_t`;
- `k_t=phi*k_(t-1)+gamma*m_t^2`;
- `y_t=m_t+k_t+e_t`, `e_t~N(0,R)`;
- the initial-observation-first ordering; and
- the manual principal-square-root UKF comparator.

The adapter must call dtype-generic primitives owned by
`bayesfilter/testing/structural_ukf_neutra_target_design_tf.py`. It must not
copy the physical chart or transition law into the GenUT module. The runtime
score is an explicit forward sensitivity through the identical finite GenUT
value program. Finite difference is diagnostic only.

The dimension-two Gaussian GenUT weights are `(1/3,1/6,1/6,1/6,1/6)`, so
`N=1002` is exactly representable and satisfies the repository numerical-test
minimum. Process noise has shape `[T,N,1]`; there is no second shock coordinate.

## Phases and bounded budget

1. Add shared dtype-generic structural chart/transition/residual primitives and
   preserve exact FP64 behavior of the established public endpoint.
2. Add an optional transition-residual callback to the generic GenUT candidate
   contract. Record the maximum residual immediately after transition and before
   observation weighting or Contract-E reset.
3. Add the `STR-UKF` GenUT adapter and focused CPU-hidden tests for transition,
   tangent, observation score, time-zero ordering, negative-control detection,
   same-scalar FD, no runtime autodiff, and no sample/Python loop in XLA code.
4. Run a trusted GPU/XLA `N=1002,T={2,10}` capacity probe. Continue only if the
   continuation veto does not fire.
5. Tune the full bounded controls `epsilon in {2,4}`, `sinkhorn_steps in {4,8}`,
   and `ridge in {1e-6,1e-5}` on disjoint simulated full-`T=100`
   calibration/validation data. Each dataset is evaluated at its own generating
   source parameter. Short prefixes are never used to select controls.
   Eligibility uses hard gates; selection minimizes held-out scaled conditional
   variance, not UKF discrepancy. Variance-ranked candidates receive the
   representative same-scalar FD audit in order until one passes; every selected
   candidate must pass, while unselected lower-ranked candidates need not consume
   the FD budget. Use the smallest seed count that estimates variance while
   keeping the campaign under the wall-time budget.
6. Freeze controls, then run the untouched frozen `T=100,N=1002` claim with up
   to eight independent particle seeds. Report Student 95% intervals when at
   least four seeds finish; otherwise classify all differences as descriptive.
7. Emit a candidate leaderboard extension row under the existing `STR-UKF`
   target. This is inclusion, not automatic admission or default promotion.

Artifact root:
`docs/benchmarks/artifacts/genut_str_ukf_leaderboard_20260722/attempt01/`.

## Default and assumption audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| `N=1002` | User minimum plus exact dimension-two GenUT replication; reviewed choice | Dense transport may still be slow | `T=10` capacity probe |
| FP32/TF32 GPU/XLA | Repository GenUT/default execution policy | FP32 structural residual or score error | unit parity and capacity probe |
| Initial observation before transition | Existing frozen target contract; required | Wrong scalar and wrong score from one extra transition | `T=1` direct likelihood parity test |
| Control grid | Prior GenUT controls used only as bounded hypotheses; retuned at the exact `T=100,N=1002` scope | Target-specific optimum may lie outside grid | residual/variance table; no default claim |
| UKF comparator | Existing same-target manual endpoint; diagnostic | Shared approximation bias or circular tuning | exclude from selection objective |
| Eight claim seeds maximum | Bounded feasibility budget | intervals may remain wide | report raw rows and uncertainty honestly |

## Skeptical plan audit

| Risk | Resolution |
|---|---|
| Wrong baseline | UKF is labeled a deterministic approximation diagnostic, not truth. No Zhao-Cui implementation exists for this Chapter 18b target. |
| Proxy promoted | FD, residuals, and UKF agreement are gates/diagnostics; none establishes scientific exactness. |
| Missing stop conditions | Memory, per-seed time, total wall time, retry count, structural identity, and device vetoes are explicit. |
| Hidden structural mismatch | Transition residual is measured before reset, and process noise is scalar. |
| Stale/default controls | The whole bounded control family is retuned for `STR-UKF,T=100,N=1002`; inherited values are hypotheses only. |
| Environment mismatch | Serious runs require memory growth, trusted GPU, XLA, FP32, and TF32; CPU is limited to focused reference tests. |
| Artifact insufficiency | Preserve plan, command, commit, hashes, controls, seeds, device/memory, raw rows, intervals, diagnostics, comparator, and decision ledgers. |
| Misleading pass | Candidate leaderboard inclusion is separated from exactness, admission, ranking, HMC, and default readiness. |

Audit verdict: `PASS_FOR_PHASED_BOUNDED_EXECUTION`.

## Attempt-1 numerical-gate correction

The first full tuning attempt was incorrectly vetoed by comparing the absolute
difference between the recursively accumulated FP32 score and a separately
reduced sum of 100 score increments to the dimensionful `5e-4` reset threshold.
For source scores as large as millions, the observed absolute difference up to
`0.25` was only about `2.5e-7` relative. Contract-E mean and OT marginal
residuals were already below `5e-5`, and the structural residual was below
`8e-6`. The repaired gate keeps reset and structural residuals absolute, but
checks score-increment accounting relatively with tolerance `2e-5`, consistent
with `O(T*u_float32)` summation error. The original control grid remains active;
the wider-grid probe is explanatory only and cannot select controls.
