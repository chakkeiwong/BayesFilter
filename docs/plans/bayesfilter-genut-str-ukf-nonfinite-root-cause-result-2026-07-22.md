# STR-UKF GenUT Non-Finite Root-Cause Diagnostic

Date: 2026-07-22
Status: `DIAGNOSIS_LOCALIZED_NO_REPAIR`

## Research question

The `N=1002,T=100` Chapter 18b structural GenUT candidate had one failing
claim seed. This diagnostic traces the first invalid operation in the frozen
`epsilon=4`, `sinkhorn_steps=4`, `ridge=1e-6`, FP32/TF32/XLA path. The consumed
claim seeds are diagnostic-only and cannot be reused for a future claim.

The diagnostic does not select new controls, repair the algorithm, admit the
leaderboard cell, or establish a scientific comparison.

## Artifacts and execution

- Increment reproduction: `docs/benchmarks/artifacts/genut_str_ukf_nonfinite_root_cause_20260722/increment_trace_attempt02/increment_trace.json`
- Stage trace: `docs/benchmarks/artifacts/genut_str_ukf_nonfinite_root_cause_20260722/stage_trace_attempt01/stage_trace.json`
- Stage tracer: `docs/benchmarks/diagnose_genut_str_ukf_stage.py`
- Diagnostic plan: `docs/plans/bayesfilter-genut-str-ukf-nonfinite-root-cause-plan-2026-07-22.md`

The stage trace ran on the trusted RTX 4080 SUPER GPU with TensorFlow FP32,
TF32 enabled, XLA compilation, and verified memory growth. It replayed the
failing seed `2026072296` and the finite control seed `2026072291`. The prior
eight-seed increment trace showed that only `2026072296` failed; all value and
score coordinates became non-finite at `t=65`.

## Confirmed failure chain

### 1. The transition and observation are finite through `t=64`

For the failing seed, at `t=64`:

| Quantity | Value |
|---|---:|
| transition particle maximum absolute value | `4.7118645` |
| transition tangent maximum absolute value | `63.21486` |
| structural transition residual | `2.38e-7` |
| log-likelihood range | `[-17.10483,-0.22579]` |
| log-likelihood tangent maximum absolute value | `77.70203` |
| likelihood increment | `-1.3782821` |
| normalized-weight ESS | `459.16` |
| normalized-weight tangent maximum absolute value | `0.16254` |

Thus the scalar innovation restriction and the transition formula are not the
initiating failure. The shared transition is
`m_t = rho*m_(t-1) + sigma*eta_t` and
`k_t = phi*k_(t-1) + gamma*m_t^2`, implemented in
`bayesfilter/testing/structural_ukf_neutra_target_design_tf.py:257` and
`bayesfilter/testing/structural_ukf_neutra_target_design_tf.py:278`. The
observation value and tangent at lines 353 and 372 are also finite.

### 2. Fixed-step Sinkhorn no longer satisfies the transport marginal

The candidate transport computes the kernel and performs exactly four
alternating updates in `bayesfilter/highdim/cubature_genut_filter.py:64`. It
then forms `gamma = N * coupling` and the barycentric cloud at line 126.

At `t=64` for seed `2026072296`:

| Quantity | Value |
|---|---:|
| required uniform row mass `1/N` | `9.98004e-4` |
| maximum row-marginal residual | `1.014803e-3` |
| row residual divided by required mass | `1.01683` |
| maximum column-marginal residual | `5.08688e-6` |
| source particle maximum absolute value | `4.71186` |
| barycentric particle maximum absolute value | `9.38945` |
| barycentric/source maximum ratio | `1.99273` |

The coupling is nonnegative, so an exactly row-normalized barycentric projection
would be a convex combination of source particles and could not exceed the
source-cloud coordinate range. The observed value nearly doubles that range.
This is direct evidence that the finite four-step scaling has lost the row
marginal sufficiently for `gamma` not to be row-stochastic. The finite
Sinkhorn residual is recorded by the implementation, but the generic runtime
does not enforce it before reset.

More precisely, for the finite coupling `pi` the barycentric map is
`B_i = sum_j pi_ij*x_j / sum_j pi_ij`. The implementation instead uses
`B_i = N*sum_j pi_ij*x_j` at
`bayesfilter/highdim/cubature_genut_filter.py:132`. These expressions are equal
only if `sum_j pi_ij = 1/N` exactly. They are unequal in the failed run. The
quantity actually computed is therefore wrong relative to the claimed
barycentric projection of the executed finite coupling. A row quotient is
needed even when terminal balancing is also used; terminal balancing is needed
to make the quotient-consumed plan preserve both intended marginals closely
enough for the covariance-ordering argument.

The tuning threshold is also not mass-scaled: the runner accepts an absolute
row/marginal residual up to `5e-4` in
`docs/benchmarks/run_genut_str_ukf_leaderboard.py:90`, while the required row
mass is approximately `1e-3` at `N=1002`. The selected tuning rows happened to
have residuals below the threshold, but this gate did not certify robustness to
the failed particle cloud.

The first clear precursor is at `t=63`, not `t=64`. The row residual is already
`9.16990e-5`, or 9.19% of `1/N`; the barycentric maximum (`3.57500`) slightly
exceeds the source maximum (`3.55387`), and the barycentric tangent jumps from
`0.75783` at `t=62` to `34.57741`. Contract E remains finite at `t=63`, but its
restored tangent is `34.70435`, which feeds the more difficult `t=64` cloud.
Thus `t=63` is the onset of material transport imbalance and `t=64` is the
first mathematically invalid reset.

This candidate core also omits a control and validity contract already present
in the canonical LEDH transport. The canonical implementation applies fixed
terminal-epsilon IPFP refinement in
`experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py:3159`,
binds `balance_steps`, and includes the consumed-plan `marginal_valid` flag in
reset validity. Its tests reject zero terminal balance and verify that terminal
balancing repairs both consumed marginals. The generic GenUT core exposes only
`sinkhorn_steps`; it has neither a separate terminal-balance stage nor a
consumed-plan marginal-validity veto.

### 3. Contract-E covariance-gap Cholesky fails in the forward value path

Contract E computes the weighted source covariance, the uniform transported
covariance, and their difference in
`bayesfilter/highdim/ledh_contract_e_reset_tf.py:60`, then calls an unguarded
Cholesky of `gap + ridge*I` at line 66.

At `t=64` for the failing seed:

| Quantity | Value |
|---|---:|
| minimum covariance-gap eigenvalue | `-4.1718379e-2` |
| ridge | `1e-6` |
| minimum gap-plus-ridge eigenvalue | `-4.1717380e-2` |
| target covariance minimum eigenvalue | `6.3044126e-2` |
| target Cholesky minimum diagonal | `0.2643727` |
| injected covariance minimum eigenvalue | `NaN` |
| reset forward finite flag | `false` |

The ridge is more than `4.17e4` times too small to repair this negative
eigenvalue. The first invalid output is therefore the Contract-E forward reset
at `t=64`; `gap_chol`, injected covariance, restored particles, and restored
tangents are non-finite. The next loop iteration at `t=65` receives those NaN
particles, so transition, likelihood, weights, value increment, and every score
coordinate become NaN.

The control seed `2026072291` remains finite at the same `t=64`: its row
residual is `1.00466e-7`, its barycentric maximum (`1.00502`) is below its
source maximum (`3.29624`), and its covariance-gap minimum eigenvalue is
`+0.00927544`. This same-time counterfactual supports the transport-imbalance
classification rather than a deterministic time-index or observation bug.

The Contract-E covariance-ordering argument requires a valid transport coupling
with the intended marginals. This trace shows that the implementation's finite
four-step approximation did not meet that prerequisite on the failed cloud.
The trace does not establish whether an exactly converged entropic coupling
would always make this particular gap positive; it establishes that the
current finite-step coupling does not.

### 4. The tangent is not the initiating failure

The failing seed has a large but finite tangent before reset (`63.21486` at
the transition and `381.895` for the barycentric tangent). The forward reset
already fails at the same time, and all value quantities become invalid before
the next transition. Therefore the observed NaNs are not a score-only bug or a
failure of recursive score accumulation. The tangent may amplify sensitivity,
but it is downstream of the invalid forward transport/reset in this trace.

## What is ruled out

- Wrong Chapter 18b structural wiring: ruled out by the pre-reset residuals and
  scalar-innovation adapter tests.
- Independent `k` process noise: ruled out by the transition construction and
  negative-control residual test.
- Wrong `y0` ordering: ruled out; the candidate uses the initial observation
  before transitions (`transition_before_first_observation=False`).
- Observation overflow as the initiating event: ruled out; observation values
  and tangents are finite at the first failing reset.
- A score-only or finite-difference implementation failure: ruled out as the
  initiating event; the forward value reset is already non-finite.
- GPU/XLA absence or memory-growth failure: ruled out for this diagnostic; the
  artifact records a logical GPU, XLA compilation, TF32, and verified growth.

## Separate harness problems

The current working-tree runner has an unrelated serialization defect that must
be repaired before another campaign. In
`docs/benchmarks/run_genut_str_ukf_leaderboard.py:138`, code referring to
`value_number` and `score_numbers` appears inside `_synthetic_dataset`, where
those names do not exist. `_evaluate` at line 237 also uses
`nonfinite_components` without defining it. A direct diagnostic call currently
raises `NameError: name 'value_number' is not defined`. This harness defect did
not create the preserved stage-trace failure, but it prevents a clean new claim
or tuning run and is distinct from the numerical failure.

The earlier attempt also had a route-identity/serialization failure after claim
execution. That consumed its seeds and is preserved as historical evidence; it
must not be conflated with the present Contract-E forward failure.

The original `run()` path has two additional fail-open reporting bugs. Its
`hard_pass` expression at `docs/benchmarks/run_genut_str_ukf_leaderboard.py:938`
does not require `claim["genut"]["all_finite"]`, and its engineering ledger at
line 988 hard-codes `"finite": True`. The later `run_claim_resume()` path does
include the all-finite condition and derives the ledger field correctly. Any
future runner repair must make the original and resume paths share the same
fail-closed predicate rather than maintaining divergent copies.

## Separate tuning-design problems

The selected control was only the lowest-ranked member of an inadequate grid;
the tuning code had no adequacy threshold for its primary variance objective.
For one validation DGP, the two tuning particle seeds produced source `phi`
scores `-3027.81` and `-12156.68`, while their values were `-223.834` and
`-222.740`. This single pair produced a scaled conditional-variance objective
of `416681.08`; the selected candidate's mean validation objective was
`208340.54`. Every grid arm had a validation objective above `2.08e5`. The
sorting at `docs/benchmarks/run_genut_str_ukf_leaderboard.py:409` nevertheless
selected the numerical minimum because eligibility at line 399 checked only
finiteness, residuals, and score accounting.

The difficult validation DGP has physical `phi=0.86756`, `gamma=0.99390`, and
`R=0.07448`, so it is a legitimate stress point near the persistence/nonlinear
boundaries, not a malformed target. The artifact proves severe score
instability there but does not by itself identify whether Sinkhorn JVP,
Contract-E JVP, or their recursive composition dominates it.

The same-scalar FD audit also does not implement a representative-point set.
`_tune` constructs one FD check from only `CALIBRATION_SEEDS[0]` at
`docs/benchmarks/run_genut_str_ukf_leaderboard.py:420`. It never audits the
held-out high-stress validation point or a random set of representative source
points. Passing maximum relative error `0.755%` at that one benign finite path
therefore establishes local derivative consistency only; it does not validate
score stability across the tuned parameter region. Two particle seeds are also
too few to support a statistically stable variance ranking. These weaknesses
should have yielded `NO_ADEQUATE_TUNING_CANDIDATE`, not a frozen claim control.

## Not checked by this diagnostic

- Whether more Sinkhorn iterations, a stabilized log-domain scaling scheme, or
  a different epsilon keeps the covariance gap positive on fresh untouched
  claim seeds.
- Whether the covariance-gap construction should be replaced by a
  mathematically guaranteed PSD construction rather than guarded or floored
  numerically.
- Whether the manual Contract-E JVP remains stable after a repaired forward
  transport/reset.
- Which JVP stage causes the extreme but finite validation scores near
  `phi=0.868`, `gamma=0.994`, and `R=0.0745`.
- Whether a repaired route improves value or score relative to the same-target
  structural UKF diagnostic.

## Decision and next action

| Decision | Status | Next justified action | Nonclaim |
|---|---|---|---|
| Root-cause localization | Complete | Repair the harness, then design a fresh scope-specific numerical repair test | No repaired claim |
| Current STR-UKF GenUT cell | Blocked | Keep it non-admitted until every fresh claim row is finite and reset/marginal gates pass | No leaderboard admission |
| Algorithm direction | Open | Test transport convergence and PSD safeguards under a new reviewed plan | No superiority or default change |

The immediate engineering repair trigger is to make the runtime fail closed
when transport marginal or Contract-E factor validity is violated, rather than
propagating NaNs. Any numerical stabilization or control change requires a new
tuning scope and fresh untouched claim seeds; it must not be retroactively
interpreted as a repair of this consumed-seed result.
