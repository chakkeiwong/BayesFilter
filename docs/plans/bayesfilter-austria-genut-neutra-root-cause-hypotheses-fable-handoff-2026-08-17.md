# Austria GenUT NeuTra Root-Cause Hypotheses: Revised Test Plan

Date: 2026-08-17

Status: `FABLE_AGREED_EXECUTION_IN_PROGRESS_GPU_CONFIRMATION_PENDING`

First audit:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-audit-reply-2026-08-17.md`

First-audit verdict: `REVISE`

Scope owner: Austria-SIR GenUT value/score consistency only. Other agents are
working in this repository; do not edit, revert, reformat, or review unrelated
files.

## Revision Ledger

| First-audit finding | Revision |
|---|---|
| F1: route identity confirmed | R0 is now a source-confirmed fact: the current batch target is diagonal plus affine restoration, with no pairwise stage and no dual caps. Phase 0 preserves a callable/control ledger rather than rediscovering this classification. |
| F2: H1 asymmetry confirmed | H1 distinguishes the confirmed source asymmetry from the still-untested causal sufficiency claim. The forward intervention removes exactly the start-of-iteration restandardization. |
| F3: score endpoint already fails closed | H4 and Phase 5 now regression-guard the existing NaN mask, validity-domain distinction, and invalid-row diagnostic semantics. They no longer predict finite scalar escape. |
| F4: covariance-gap symmetrization asymmetry | Added H3A and an explicit reset-validity boundary in Phase 2. It is validity-only and is not proposed as the cause of the finite particle mismatch. |
| F5: interior graph/XLA capture is intrusive | Phases 1-3 capture intermediates in eager deterministic mode only. Graph/XLA arms compare endpoint scalars and final clouds only. |
| F6: Austria local direct terms verified | H5 records that local formulas passed static derivation review while retaining independent composed-tangent tests. |
| F7: test economy | Upstream Phase 2 checks are classified as confirmatory; the tangent-only injected failure is a fail-closed regression guard. |
| Mechanical review | Removed the duplicated `H8` dependency line and corrected R0 wording about affine restoration. |

Second review is requested separately in
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-second-review-request-2026-08-17.md`.

Second-review result:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-second-review-reply-2026-08-17.md`

Second-review verdict: `AGREE`

Execution checkpoint:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-checkpoint-2026-08-18.md`

## Executive Problem Statement

The Austria NeuTra adapter exposes two endpoints:

```text
batch_value_status(theta)
    -> batch_finite_value(...)

neutra_batch_log_prob_and_grad_status(theta)
    -> batch_finite_value_score(...)
    -> value plus score
```

The intended mathematical contract is one deterministic finite scalar

\[
V(\theta),
\]

with the tangent-carrying endpoint returning

\[
\{V(\theta),\nabla V(\theta)\}.
\]

At the current Austria center, the two routes instead return different finite
filter likelihoods:

| Route | Filter likelihood at `theta=(0,0,0)`, `T=20`, `N=1008` |
|---|---:|
| tangent-free `batch_finite_value` | `-686.0074` |
| tangent-carrying `batch_finite_value_score` | `-680.3478` |
| difference | `+5.6596` |

Both routes reported `program_valid=True`. The posterior chart and prior terms
are shared, so they cannot explain this difference.

This is wrong relative to the claimed single-finite-program value/score target.
Austria must remain blocked from NeuTra until the value identity is repaired and
the derivative is independently validated.

## Confirmed Route Identity

The first Fable audit confirmed that the current batch NeuTra target is not the
promoted dual-cap GenUT algorithm.

The promoted dual-cap family is documented in
`docs/plans/bayesfilter-genut-dual-cap-monograph-ready-spec-2026-08-07.md` as:

```text
diagonal third/fourth moments
+ pairwise co-skewness/co-kurtosis
+ pairwise row-RMS radial cap 2
+ standardized coordinate cap b=0.98, p=8
+ final affine restoration
```

By contrast:

- `GenUTControls` in
  `bayesfilter/highdim/cubature_genut_neutra_targets.py` exposes the diagonal
  correction controls and optional LM/trust controls, but no pairwise-moment or
  coordinate-cap controls.
- `batch_finite_value` and `batch_finite_value_score` in
  `bayesfilter/highdim/cubature_genut_batch_tf.py` call the local
  `_higher_moment_batch_value` and `_higher_moment_batch_jvp` diagonal routines.
- Both local diagonal routines perform final affine restoration to the weighted
  source mean and covariance. That stage is present and must not be listed as a
  missing dual-cap stage.
- The pairwise and coordinate-cap implementation is instead in
  `bayesfilter/highdim/higher_moment_contract_e.py` and is used by the fuller
  route in `bayesfilter/highdim/cubature_genut_filter.py`.

The missing dual-cap stages are pairwise co-skewness/co-kurtosis correction,
the pairwise row-RMS radial cap, and the coordinate cap. The optional
`higher_moment_trust_radius` acts on diagonal displacement and is not the
dual-cap pairwise radial cap. The current Austria blocker therefore belongs to
a `batch_diagonal_candidate` with affine restoration and cannot be reported as
a failure of the promoted dual-cap algorithm.

## Relevant Code Anchors

Fable should inspect these exact functions before accepting or revising the
hypotheses:

- `bayesfilter/highdim/cubature_genut_neutra_targets.py`
  - `GenUTControls`
  - `_core_kwargs`
  - `_posterior_value_status`
  - `_posterior_value_score_status`
  - `make_genut_neutra_target`
- `bayesfilter/highdim/cubature_genut_batch_tf.py`
  - `_uniform_moments_jvp`
  - `_restore_cloud_batch_value`
  - `_restore_cloud_batch_jvp`
  - `_shape_iteration_batch_jvp`
  - `_higher_moment_batch_value`
  - `_higher_moment_batch_jvp`
  - `batch_finite_value`
  - `batch_finite_value_score`
- `bayesfilter/highdim/cubature_genut_batch_adapters.py`
  - `parameterized_austria_sir_batch_adapter`
- `bayesfilter/highdim/higher_moment_contract_e.py`
  - `higher_moment_contract_e`
  - its diagonal, pairwise, radial-cap, coordinate-cap, and affine-restoration
    value/JVP blocks
- `bayesfilter/highdim/cubature_genut_filter.py`
  - `finite_value_score`
  - the call into `higher_moment_contract_e`
- `docs/plans/bayesfilter-genut-dual-cap-monograph-ready-spec-2026-08-07.md`
- `docs/plans/bayesfilter-genut-four-model-neutra-readiness-result-2026-08-04.md`
- `docs/benchmarks/artifacts/genut_four_model_neutra_readiness_20260804/aggregate_attempt04/result.json`

## Evidence Already Collected

These are diagnostic observations, not yet a completed root-cause proof.

### E1. The mismatch is below the posterior chart

The filter likelihoods differ by `5.6596`; adding the common prior/Jacobian
preserves the same difference. This rules out the posterior chart as the
initiating cause at the tested point.

### E2. Austria model value equations are shared

Within `parameterized_austria_sir_batch_adapter`, the transition value is
computed by the same RK4 primal routine used alongside the tangent, and the
observation value is called on the primal particles in both endpoints. No
model-law difference was observed before the higher-moment correction.

This does not prove that the tangent equations are correct. It only says they
do not explain the already observed zero-tangent primal mismatch.

### E3. Sinkhorn and Contract-E reset primal outputs matched

On identical first-step Austria inputs, `_restore_cloud_batch_value` and
`_restore_cloud_batch_jvp` returned identical primal particles. A synthetic
zero-tangent test also gave exact Sinkhorn and Contract-E primal parity.

### E4. The first primal divergence is in higher-moment correction

At the first Austria step, identical source, weights, and reset particles were
passed to the two higher-moment routines. The output particle clouds differed
by up to about `19` after four correction iterations.

With all tangent tensors set to zero, the first correction iteration still
produced a maximum primal cloud difference of about `24`. Thus the mismatch
does not require a nonzero Austria tangent.

### E5. There is a source-level finite-program asymmetry

`_higher_moment_batch_jvp` standardizes the reset points before entering its
loop. Each call to `_shape_iteration_batch_jvp` then recomputes mean,
covariance, Cholesky, and standardized points before forming moments and the
shape system.

`_higher_moment_batch_value` standardizes once before its loop, forms the shape
system directly from that cloud, and restandardizes only after applying each
correction.

In exact arithmetic, standardizing an already standardized cloud is an
identity. In FP32 it is not bitwise idempotent. Therefore the two functions
currently define different finite arithmetic programs before considering any
tangent formula.

### E6. The normal-equation solve strongly amplifies the difference

For one first-step Austria coordinate, the observed unscaled normal system had
an approximate condition number of `1.2e5`. The value and JVP paths produced
slightly different normal matrices and substantially different solve outputs.
Across the batch, the maximum coefficient magnitude was approximately `863`
for the value path and `1461` for the JVP path in one probe.

This is consistent with ill-conditioning amplifying the redundant
restandardization difference. It does not prove that normal equations are the
initiating defect.

### E7. Deterministic TensorFlow did not remove the mismatch

The zero-tangent first-step mismatch was reproduced with
`TF_DETERMINISTIC_OPS=1` set before TensorFlow import and
`tf.config.experimental.enable_op_determinism()` enabled. Missing deterministic
mode is therefore not a sufficient root-cause explanation.

### E8. LM/trust controls did not establish a repair

A diagnostic scaled-LM plus smooth trust-cap probe reduced the reported scaled
system condition number but still left a material primal cloud difference.
The result must not be interpreted as an endpoint repair. A stable solver can
reduce amplification without making two different finite programs identical.

## Research Intent Ledger

| Field | Declaration |
|---|---|
| Main question | Why do Austria's tangent-free and tangent-carrying batch endpoints return different finite scalar values, and does that diagnosis apply to the promoted dual-cap route? |
| Candidate mechanism | A redundant restandardization in the JVP primal path perturbs an already standardized FP32 cloud, and an ill-conditioned normal-equation moment solve amplifies the perturbation. |
| Expected failure mode | Primal particle clouds diverge at the first higher-moment iteration even with zero tangent, then different clouds generate different future likelihood increments. |
| Primary engineering criterion | Locate the first unequal primal tensor and demonstrate a causal intervention that removes that inequality without changing the declared value program. |
| Promotion veto | Wrong route identity, unequal primal values, nonfinite output, branch disagreement, derivative mismatch, stale tuning scope, or a repair that changes the target while claiming to preserve it. |
| Continuation veto | Corrupted or non-reconstructible frozen inputs; inability to identify the actual callable; or a diagnostic whose instrumentation changes the compared arithmetic before the first observed divergence. |
| Repair trigger | One hypothesis is causally supported by a paired intervention while the target, inputs, arithmetic scope, and controls remain fixed. |
| Explanatory diagnostics | Condition numbers, coefficient magnitudes, cap activity, tangent norm growth, runtime, and approximate UKF/SGQF proximity. |
| Must not be concluded | No posterior correctness, exact Austria likelihood, exact nonlinear score, dual-cap correctness, NeuTra readiness, HMC readiness, default superiority, or solver promotion. |

## Hypotheses

### R0: Confirmed Wrong Route Identity

**Statement.** The current batch NeuTra target is diagonal-only and is not the
promoted dual-cap finite program. The earlier Austria admission failure may be
valid for that batch route but cannot establish failure of dual-cap GenUT.

**Source-confirmed classification.** The callable closure and controls used by
`make_genut_neutra_target("austria_sir")` will contain no pairwise correction,
radial cap `2`, or coordinate cap `.98/8`. It will contain the ordinary
Contract-E reset and final affine restoration shared by the diagonal route,
but those stages do not make it dual-cap.

**Phase 0 preservation test.** Produce a route-identity ledger from the actual
callable graph and control payload. Map every stage of the dual-cap algorithm
order to the called function and bound control. On the audited source, missing
pairwise and cap stages classify the route as `batch_diagonal_candidate`, not
`dual_cap`. If the callable changed after the audit, classify it
`unknown_route_identity` until the new graph is traced.

**Interpretation.** A confirmed route mismatch does not stop diagnosis of the
batch diagonal endpoint bug, but it splits the work into two lanes:

1. repair and test the diagonal batch primitive as shared infrastructure; and
2. separately implement or bind a batch-native dual-cap value/JVP target before
   claiming Austria dual-cap NeuTra readiness.

### H1: Redundant Restandardization Is The Initiating Primal Defect

**Statement.** `_shape_iteration_batch_jvp` restandardizes an already
standardized cloud at the start of each correction iteration, whereas the value
route does not. This is the first unequal primal operation.

**Predictions.**

- With `correction_steps=0`, the higher-moment primal outputs agree.
- With `correction_steps=1`, zero tangents are sufficient to produce a mismatch.
- A diagnostic JVP variant that skips the redundant input standardization and
  consumes the same standardized tensor as the value route removes the first
  mismatch.
- Conversely, a diagnostic value variant that adds the same redundant
  standardization should move toward the current JVP primal output. This reverse
  intervention changes the value target and is diagnostic only.

**Discriminating test.** Capture and compare, in order:

```text
input standardized cloud
restandardized cloud
m3, m4
direction3, direction4
J
residual
normal/system and rhs
coefficient
displacement
post-correction standardized cloud
```

Record exact equality, maximum absolute difference, scale-aware relative
difference, and ULP distance for every primal tensor. Do this with all tangent
tensors zero before testing nonzero tangents.

**Falsification.** H1 is falsified as the initiating cause if the first unequal
tensor precedes restandardization, or if removing only that redundant operation
does not make all subsequent pre-solve primal inputs agree.

### H2: Normal Equations Are A Numerical Amplifier

**Statement.** The unscaled solve of

\[
(J^T J + \lambda I)c=J^T r
\]

squares the conditioning of `J`; the fixed floor `1e-5` is insufficient for
some Austria coordinate systems. This amplifies a small primal input mismatch
into a large particle correction.

**Predictions.**

- Coordinates with the largest coefficient and cloud discrepancies have large
  condition estimates or small singular values.
- A float64 diagnostic and direct QR/SVD least-squares reference are less
  sensitive than FP32 normal equations.
- Scaled LM or a rank-revealing solve reduces amplification but does not, by
  itself, guarantee endpoint identity if H1 remains.

**Discriminating test.** On frozen `T=1` tensors, compare normal equations,
column-scaled LM, QR least squares, and SVD/pseudoinverse as diagnostic arms.
Use the same `J` and `r` tensor for every arm. Record singular values,
condition estimates, residual norm, coefficient norm, displacement norm, and
primal output difference.

**Falsification.** H2 is not an important amplifier if large output differences
occur in well-conditioned coordinates or remain unchanged when identical
`J,r` inputs are solved by a stable reference method.

### H3A: Covariance-Gap Symmetrization Is A Validity-Only Asymmetry

**Statement.** The value reset computes the covariance gap from unsymmetrized
covariances and then symmetrizes the difference, while the JVP reset receives a
source covariance already symmetrized by `_weighted_moments_jvp` before forming
the difference. This is a separate finite-program asymmetry.

**Scope.** It affects `minimum_gap_eigenvalue` and the `gap_valid` branch only;
it does not alter reset particles on a valid branch and cannot explain the
observed finite value mismatch. A threshold flip must fail closed.

**Test.** Add the gap calculation to the Phase 2 boundary list, record both
values and validity flags, and test a knife-edge synthetic covariance. If the
rounding difference flips only the score route's validity, the expected result
is a NaN value/score pair from that route, not a finite exposed scalar.

### H3: Additional Duplicated-Primal Arithmetic Remains After H1

**Statement.** Even after aligning the iteration standardization order, the
separate value and JVP functions may compute the same mathematical expressions
through different TensorFlow graph shapes or reductions and therefore still
produce unequal FP32 primal tensors. Source inspection found no second
higher-moment formula asymmetry; this remains an empirical graph/compiler
hypothesis only.

**Prediction.** A common primal helper that returns cached intermediates to both
the value and JVP code paths gives bitwise-identical values; separately
recomputed primals may not.

**Discriminating test.** After the H1 intervention, compare every remaining
primal boundary in eager deterministic mode. Then compare only endpoint scalars
and final particle clouds under `tf.function` and
`tf.function(jit_compile=True)`; do not add interior fetches in graph/XLA arms.
Finally construct a diagnostic shared-primal implementation where the JVP
consumes the exact cached primal intermediates. Do not call a second value
formula from inside the JVP helper.

**Falsification.** H3 is falsified for the tested scope if the H1 intervention
alone produces exact primal parity across all required backends and batch
layouts.

### H4: Tangent Validity Must Remain Fail-Closed

**Statement.** The score loop includes tangent finiteness and JVP validity in
`stage_valid` and `step_valid`, while the value loop does not. The current
source is expected to fail closed as a NaN value/score pair when only the
tangent route becomes invalid. The remaining risk is endpoint validity-domain
asymmetry and consumers reading unmasked diagnostics from an invalid row.

**Current evidence.** Fable confirmed by source inspection that a tangent-only
failure cannot escape as an externally finite score-bearing value at the
endpoint. It remains a secondary regression surface, not the cause at
`theta=0` where both routes reported valid.

**Discriminating test.** Record per-time value-valid, tangent-valid,
reset-valid, higher-moment-valid, selected-branch, first-invalid reason, and
the final NaN mask. Add an injected-tangent fixture that creates a tangent-only
nonfinite value without changing primal particles. The predeclared expected
result is a NaN pair and a permanently latched invalid status.

**Required semantics.** A derivative-invalid route must fail closed as a
value/score pair. It must not emit a finite score-bearing value for a recursion
that differs from the endpoint recursion. Diagnostics from invalid rows are
explanatory only and must be gated by `valid_pre_regularized_score`.

### H5: Austria Tangent Mathematics Has A Separate Error

**Statement.** Endpoint parity can be repaired while the Austria RK4 or
observation tangent remains wrong. The observed tangent norms above `1e6` late
in the horizon may be genuine sensitivity growth or an implementation error.

**Current source evidence.** Fable verified by derivation that the local direct
terms for `d kappa/d theta0`, `d nu/d theta1`, and
`d variance/d theta2 = 2 variance`, plus the stated RK4 JVP structure, are
present and locally correct. This lowers the prior probability of H5 but does
not validate the composed `T=20` derivative.

**Prediction.** If the manual tangent is correct, it agrees with an independent
TensorFlow autodiff JVP for each RK4 substep, complete transition, observation
log likelihood, and one-step likelihood increment on a fixed branch.

**Discriminating test.** After primal identity passes:

1. compare `initial_tangent` to autodiff;
2. compare each of four RK4 substeps and the complete transition;
3. compare `observation_tangent`, including the variance derivative;
4. compare the reset JVP;
5. compare one correction-step JVP;
6. run the declared `h^2` finite-difference regression at `T=1`, `T=2`, and
   `T=20` with branch identity recorded.

Do not use UKF or SGQF as a derivative authority for this test. The comparator
is the derivative of the same frozen finite program.

### H6: Precision And Compiler Mode Modify Severity But Do Not Define The Target

**Statement.** FP32, TF32, XLA, and reduction layout can change the severity of
the mismatch, but arithmetic-mode differences are modifiers rather than a
license for two endpoints to compute different programs.

**Discriminating test.** Run the same frozen lower-level boundary under:

- CPU float64 diagnostic reference;
- GPU FP32 with TF32 disabled;
- GPU FP32 with TF32 enabled, if the exact target scope declares it;
- eager GPU;
- graph GPU; and
- XLA GPU.

The current Austria batch adapter contains FP32 constants and FP32 tangent
allocations, so the existing full endpoint is not a valid float64 program.
Initially, the CPU float64 arm must cast the frozen standardized cloud, `J`,
and residual into a TensorFlow-only diagnostic reference. A full float64
endpoint may be tested only after a separate dtype-parametric TensorFlow
reference adapter is implemented and clearly labeled diagnostic; silently
feeding float64 theta into the current FP32 adapter is not an admissible test.

All GPU runs require trusted/escalated execution, memory growth before device
initialization, deterministic operations, and explicit device provenance.

**Interpretation.** If float64 substantially shrinks the gap, finite precision
is an amplifier. It does not repair the production FP32 target or establish
that the two finite programs are equal.

### H7: Batch Composition Or Direction Flattening Changes A Primal Result

**Statement.** Broadcasting or flattening the posterior batch and parameter
directions may accidentally influence a primal computation that should be
independent for each posterior row.

**Discriminating test.** Compare a fixed theta row when evaluated as:

- batch size one;
- two identical rows;
- the same row paired with a different row;
- reversed row order; and
- permuted tangent-direction order.

The row's value must be invariant. Direction permutation must permute tangent
columns only and must not alter the primal output.

### H8: Any Repair Creates A New Tuning Scope

**Statement.** Removing a redundant operation, replacing a solve, or changing
from diagonal to dual-cap changes the implementation identity and possibly the
finite target. Existing Austria tuning artifacts cannot automatically admit the
repaired route.

**Discriminating test.** Compare the repaired route identity and complete
controls against the bound tuning artifact. The claim route must reject stale,
diagonal-only, cross-horizon, wrong-TF32, or caller-stamped tuning identities.

This hypothesis does not locate the initiating numerical defect. It protects
the scientific interpretation after repair.

## Hypothesis Dependency Order

```text
R0 route identity
  |
  +-- diagonal batch bug lane: H1 -> H2 -> H3A/H3 -> H4
  |
  +-- promoted dual-cap batch lane: verify/implement actual stages
                                      -> repeat H1-H4 equivalents

Once one primal program is identical:
  H5 tangent correctness
    -> H6 arithmetic-scope replication
    -> H7 batch invariance
    -> H8 fresh tuning scope
```

H2 must not be tested as a replacement for H1. H5 must not be interpreted
until the value program is identical. H8 must not reuse a historical tuning
artifact merely because the output looks numerically favorable.

## Proposed Test Plan

### Phase 0: Route And Evidence Freeze

1. Record Git commit, dirty-file list, source hashes for the exact functions,
   TensorFlow build, CUDA/GPU, TF32, deterministic-op, XLA, dtype, observations,
   process noise, initial noise, cubature design, controls, and target signature.
2. Build the R0 stage-to-callable route ledger.
3. Classify the current target explicitly as `dual_cap`,
   `batch_diagonal_candidate`, or `unknown_route_identity`.
4. Do not proceed under a `dual_cap` label unless every required stage and
   bound control is present.

Output:
`docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/attemptNN/route_identity.json`

### Phase 1: Diagnostic Harness And Regression Fixtures

Create a diagnostic-only runner and focused tests. Proposed paths:

- `docs/benchmarks/run_genut_austria_endpoint_root_cause_20260817.py`
- `tests/highdim/test_genut_batch_primal_parity.py`

The runner must capture structured tensors or summaries without changing the
operations before the first mismatch. Interior capture is **eager-only**.
Graph and XLA modes are endpoint-only replication arms and must not expose
interior fetches or debug identities. It must support:

- `T` in `{1,2,20}`;
- correction steps in `{0,1,4}`;
- zero and real tangents;
- frozen `N=1008` claim inputs;
- eager interior-capture mode, plus graph/XLA endpoint-only modes;
- FP32/TF32 endpoint scope and a lower-level float64 TensorFlow diagnostic
  reference; and
- fresh versioned output roots.

Required first regression: `correction_steps=0` must establish the upstream
value/reset baseline before any correction hypothesis is tested.

### Phase 2: First-Unequal-Tensor Localization

Use `T=1`, zero tangent, deterministic operations, and the current exact
Austria inputs.

Steps 1-5 are confirmatory regression checks for the upstream equality already
observed and source-reviewed. Step 6 records the validity-only H3A difference;
step 7 is the discovery boundary for the particle-producing H1 asymmetry.

1. Compare initial particles.
2. Compare transition particles.
3. Compare log likelihood, increment, and normalized weights.
4. Compare Sinkhorn barycentric particles.
5. Compare Contract-E restored particles.
6. Compare the reset covariance-gap symmetrization and validity flags.
7. Compare every H1 higher-moment intermediate in execution order.

Stop at the first unequal tensor and preserve all immediately preceding equal
tensor hashes. This phase answers where divergence begins, not how to repair it.

### Phase 3: Causal H1 Interventions

Run three diagnostic implementations on the same tensors:

| Arm | Purpose | Eligible repair? |
|---|---|---|
| current value order | declared value baseline | yes, as baseline only |
| JVP skips redundant input restandardization | tests H1 while preserving value order | candidate |
| value adds JVP's redundant restandardization | reverse causal control | no; changes value program |

H1 is supported only if the forward intervention aligns JVP primal tensors
with the current value order and the reverse intervention aligns the value
route toward the current JVP order.

If H1 is causally confirmed, the candidate production repair must use one
shared-primal correction core. Every primal operation executes once; optional
tangent calculations consume that exact primal state under a Python-level
`tangents is not None` branch. Do not restore separate value and JVP primal
implementations after the diagnostic proves the mismatch.

### Phase 4: Conditioning And Solver Diagnosis

Holding identical `J,r` fixed, compare:

1. current normal equations;
2. column-scaled LM;
3. direct QR least squares; and
4. float64 SVD/pseudoinverse diagnostic reference.

This phase classifies H2 as an amplifier and identifies whether a solver repair
is required after H1. Solver choice must not be promoted from one Austria point;
it remains a candidate pending cross-model and target-specific tuning evidence.

### Phase 5: Fail-Closed Branch And Batch Invariants

Test H4 and H7 using `T=1`, `T=2`, and a small fixed parameter neighborhood.
Include injected tangent-only failure and row/direction permutations. The
predeclared H4 result is a NaN value/score pair, permanently latched invalid
status, and diagnostics that are not promotion evidence. Any finite scalar
after a tangent-only failure is a hard fail-closed defect.

### Phase 6: Austria Tangent Audit

Only after exact primal identity passes, execute H5. Use independent TensorFlow
autodiff for local blocks and the existing `h^2` regression diagnostic for the
complete finite program. Record tangent norms per time and per parameter; do
not infer correctness from finiteness or from approximate UKF/SGQF proximity.

### Phase 7: Exact-Scope And Cross-Model Regression

Repeat the supported repair on:

- Austria `T=1,2,20`, `N=1008`;
- LGSSM current exact scope;
- KSC-SV current exact scope; and
- predator-prey current exact scope.

Use at least three frozen particle designs for regression robustness, but treat
the differences descriptively; this is an engineering invariant campaign, not
a stochastic method ranking.

For the promoted target, first create or identify a batch-native implementation
with the actual dual-cap stage order. Repeat the checks on that route only after
its route ledger binds pairwise moments, pairwise radial cap, coordinate cap,
and affine restoration. A diagonal-only pass is insufficient for dual-cap
NeuTra admission.

### Phase 8: Post-Repair Boundary

If and only if the actual intended route passes value identity, local JVP
checks, complete-program derivative regression, branch invariants, and batch
invariants:

1. issue a new route identity;
2. produce fresh scope-specific tuning under the owner tuning policy;
3. rerun target construction/admission tests; and
4. propose, but do not silently launch, target-specific NeuTra training.

## Decision Rules

| Observation | Classification |
|---|---|
| Current NeuTra callable lacks pairwise/radial/coordinate stages | `CONFIRM_R0_WRONG_ROUTE_IDENTITY` |
| Steps 0 agree, steps 1+ disagree, and skipping redundant JVP standardization restores primal equality | `CONFIRM_H1_INITIATING_DEFECT` |
| Stable solve reduces sensitivity only after identical `J,r` are supplied | `CONFIRM_H2_AMPLIFIER` |
| Reset covariance-gap arithmetic differs while particles agree | `CONFIRM_H3A_VALIDITY_ONLY_ASYMMETRY` |
| Mismatch remains after H1 before the solve | `SUPPORT_H3_ADDITIONAL_PRIMAL_DUPLICATION` |
| Tangent-only invalidity returns a NaN pair and latched invalid status | `PASS_H4_FAIL_CLOSED_REGRESSION` |
| Tangent-only invalidity exposes any finite scalar | `CONFIRM_H4_FAIL_CLOSED_DEFECT` |
| Primal identity passes but local autodiff or complete FD regression fails | `CONFIRM_H5_DERIVATIVE_DEFECT` |
| Only float64 passes | `CONFIRM_H6_PRODUCTION_PRECISION_BLOCKER`, not a production repair |
| Batch composition changes a row's primal value | `CONFIRM_H7_BATCH_SEMANTICS_DEFECT` |
| Repaired identity does not match tuning scope | `CONFIRM_H8_RETUNING_REQUIRED` |
| Multiple first unequal boundaries remain | `INCONCLUSIVE_SPLIT_INSTRUMENTATION_MORE_FINELY` |

No tolerance may be invented after seeing results. Where the same cached primal
tensor is contractually shared, require exact equality. For separately executed
diagnostic references, report ULP and scale-aware errors and derive any future
admission tolerance from a reviewed forward-error argument rather than the
historical `2e-4` endpoint threshold.

## Evidence Contract

- **Question:** which operation first makes the two Austria finite values
  unequal, what mechanism amplifies it, and is the tested route dual-cap?
- **Exact baseline:** current tangent-free finite value program on identical
  frozen observations, innovations, design, controls, arithmetic mode, and
  theta.
- **Primary criterion:** causal first-boundary localization plus restoration of
  one shared primal value program.
- **Hard vetoes:** wrong route label, nonfinite tensor, changed frozen input,
  branch disagreement, instrumentation before the first mismatch, unequal
  endpoint value, or derivative failure after value repair.
- **Explanatory only:** solver condition, tangent magnitude, runtime, cap
  activity, UKF/SGQF distance, and cross-seed dispersion.
- **Not concluded:** posterior accuracy, exact filtering, statistical method
  ranking, NeuTra/HMC readiness, or default promotion.
- **Result artifact:** a fresh attempt under
  `docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/` plus
  a result note under `docs/plans/`.

## Compute And Attempt Budget

- Phase 0 source audit: no GPU.
- Phases 1-6: one GPU, bounded to three fresh attempts and 30 total GPU-minutes.
- Phase 7 cross-model exact-scope regression: separately budgeted only after a
  causal repair candidate exists; do not consume it during exploratory probes.
- Stop early when a continuation veto fires or when the first unequal boundary
  cannot be observed without changing upstream arithmetic.

GPU commands must use the `tftwogpu` TensorFlow build, trusted/escalated device
access, memory growth before initialization, and a single explicitly selected
GPU unless the plan is amended with a two-process allocation.

## Skeptical Plan Audit

The plan has been checked against the required failure modes:

- **Wrong baseline:** addressed by R0 before numerical repair. The batch route
  must not be called dual-cap without the required stages.
- **Proxy promoted to criterion:** condition numbers, LM behavior, UKF/SGQF
  proximity, and FD residual size are not endpoint-identity proof.
- **Missing stop condition:** wrong route identity splits the work; corrupted
  inputs, hidden branch changes, or intrusive instrumentation stop execution.
- **Unfair comparison:** every causal arm receives identical frozen tensors,
  controls, dtype, TF32, compiler mode, and operation ordering up to the tested
  intervention.
- **Hidden assumption:** deterministic operations were tested and did not remove
  the mismatch; precision and XLA remain explicit hypotheses rather than assumed
  causes.
- **Stale context:** the 2026-08-04 readiness artifact predates current dirty
  source edits and the later dual-cap default decision. Fresh source hashes and
  route identity are mandatory.
- **Meaningless success:** reducing the value gap with damping could pass while
  differentiating a changed scalar. The primary test is shared-primal identity,
  followed by an independent derivative check.
- **Underpowered inference:** this is deterministic engineering localization.
  Multi-design runs test generality but do not support stochastic superiority.

Skeptical audit outcome: the first Fable audit confirmed R0 and the H1
intervention design but required the revisions recorded above. Execution may
begin only after the bounded second review confirms that F1-F7 are closed. No
production source or default should change before that review.

## Second-Review Acceptance Questions

1. Does the route ledger now state precisely: diagonal plus affine restoration,
   no pairwise correction, no pairwise radial cap, and no coordinate cap?
2. Does H1 remove exactly the start-of-iteration restandardization while
   retaining the shared outer and post-correction standardizations?
3. Is H3A correctly limited to reset validity and excluded as the cause of the
   finite particle mismatch?
4. Do H4, Phase 5, and the decision rules now regression-guard the existing
   fail-closed NaN behavior, validity-domain distinction, and invalid-row
   diagnostic semantics?
5. Are interior captures restricted to eager deterministic mode and graph/XLA
   tests restricted to endpoints/final clouds?
6. Does the shared-primal candidate architecture prevent recurrence of this
   value/JVP duplication defect class?
7. Does any material finding from F1-F7 remain unaddressed or introduce a new
   confound?
8. May execution begin at Phase 0 without changing the scientific target,
   default, tuning artifact, or NeuTra admission status?

## Handoff Boundary

Fable should audit this memo and the cited code only. It should not infer that
the current diagnosis authorizes:

- editing the shared worktree;
- changing the GenUT default;
- changing the Austria scientific target;
- reusing historical tuning;
- running NeuTra or HMC;
- claiming the diagonal batch bug applies to dual-cap; or
- treating a finite or damped score as correct.

After an `AGREE` verdict, the implementing agent may begin Phase 0. A `REVISE`
verdict requires another bounded plan correction before creating the diagnostic
harness.
