# Zhao-Cui Moment Teacher for Contract E

Date: 2026-07-30  
Status: executed in phases; nonlinear promotion remains gated  
Research role: high-dimensional feasibility study for nonlinear state-space models; NAWM itself is not in scope

## Route-Specific Execution Decision, 2026-07-30

The selected execution route for this moment-teacher lane is
`zhao_cui_moment_teacher_gpu_fp32_no_tf32_xla_v1`: trusted GPU, FP32 tensors,
XLA enabled, and TensorFloat-32 execution disabled before graph execution.
This is a reviewed lane-specific exception to the repository-wide TF32 target
direction, not a global default change.

Rationale: the no-TF32 fused teacher passed the declared FP32/FP64 parity gate;
TF32 failed it deterministically. On the nearest complete score route, TF32
created systematic displacement relative to FP32-no-TF32 in four score
coordinates, and at `N=4096` the worst displacement was 0.759 reference MCSE.
Repeated `N=4096` timing found only an 11.0% median time reduction, below the
predeclared 20% threshold for a material speed-based waiver. The current HMC
wrapper does not separately recompute a higher-precision MH acceptance energy.

This decision retires TF32 parity as a continuation requirement for this lane.
It does not establish final-score correctness, HMC readiness, nonlinear
validity, or default readiness. Those gates remain on the selected no-TF32
route.

## Research Question

Can a deterministic Zhao-Cui-style squared tensor-train (TT) approximation be
used only as an independent per-step moment teacher for the higher-moment
Contract E correction, reducing particle-cloud higher-moment error without
requiring a dense (d)-dimensional grid or a full Zhao-Cui particle filter?

The candidate under test is the hybrid route documented in
`docs/chapters/ch32c_entropic_ot_sinkhorn.tex`, section
`sec:bf-eot-zhao-cui-moment-teacher`:

- the particle lane still supplies the likelihood increment and Contract E
  first-two-moment targets;
- the fixed-branch squared-TT teacher is fitted to the adjacent density
  recursion, not to empirical particle moments;
- paired-core operator contractions supply TT mean/covariance and selected
  whitened shape moments;
- only the declared TT shape targets are passed to the bounded correction;
- the corrected equal-weight cloud is carried to the next particle step while
  the TT filtering marginal is carried to the next teacher step.

This is an `extension_or_invention`.  The squared-TT density representation,
adjacent recursion, and marginal contraction are source-grounded Zhao-Cui
primitives; using their moments as Contract E targets is not claimed by the
paper or author implementation.

## Evidence Contract

### Question

Does the independent TT moment teacher have correct finite-density moment
contractions and total tangents, and is it sufficiently affordable and useful
to justify an opt-in Contract E experiment on high-dimensional nonlinear
models?

### Baseline Ladder

1. Direct dense quadrature on tiny one- and two-dimensional polynomial fixtures
   is the mechanics authority for represented TT moments.
2. Existing `SquaredTTDensity.sqrt_square_normalizer` and marginal contraction
   are the implementation baseline.
3. Existing empirical-target higher-moment Contract E is the filtering baseline.
4. The new hybrid TT-shape-target Contract E is the candidate.
5. A Gaussian/Wick fixed-target arm is a diagnostic LGSSM sanity check, not an
   oracle for nonlinear models.

### Promotion Criterion

The candidate may proceed from mechanics to feasibility only if all of the
following hold:

- mass, raw moments, selected affine-form moments, and normalized moments agree
  with dense quadrature within predeclared numerical tolerances;
- analytic core/operator JVPs agree with centered finite differences on small
  diagnostics, with finite differences used only for validation;
- the hybrid reset preserves the particle weighted mean/covariance to the
  existing Contract E tolerance;
- LGSSM fixed Gaussian moments agree with the Kalman moments on the same finite
  represented teacher fixture;
- no non-finite values, negative normalizers, branch mismatches, or missing
  tangent terms occur;
- resource measurements report peak host/device memory, TT ranks, basis size,
  fit sweeps, contraction time, and particle count.

These criteria establish finite-program mechanics and feasibility only. They do
not establish exact filtering likelihood, exact posterior score, nonlinear
superiority, HMC readiness, or a default change.

### Vetoes

- A contraction or derivative mismatch is a hard implementation veto.
- Fitting the TT teacher to particle empirical moments is a target-definition
  veto because it removes the proposed independent-teacher property.
- A partial score that omits core-fit, marginal, operator, defensive-density,
  chart, or whitening tangents is a score-validity veto.
- Adaptive runtime rank selection, stochastic fitting choices, or parameter-
  dependent retuning inside HMC is a fixed-branch veto.
- NumPy, Python sample loops, or scalar row-mapping in an XLA runtime path is an
  execution-policy veto.
- Dense all-pair (O(d^2)) or dense third/fourth-order tensor materialization
  is a scalability veto for the high-dimensional route.
- A failed nonlinear value/score comparison rejects the candidate for that
  scope; it does not invalidate the contraction mathematics.

### Explanatory Diagnostics

TT fit residuals, normalizer differences, whitening condition numbers, moment
residuals after correction, TT ranks, pair-graph density, operator-contraction
time, fit time, and memory are explanatory diagnostics unless a gate above
explicitly assigns them veto status.

### Nonclaims

Passing this plan does not prove that Zhao-Cui is source-faithful as a complete
filter, that the moment teacher is an oracle, that likelihood is preserved over
the full horizon, that score bias decreases, that ranks remain bounded as
dimension grows, or that the route is affordable inside high-dimensional HMC.

## Phases

### Phase 0: Source and Math Audit

Record the paper and author-source anchors before implementation:

- `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf`
  and `.txt`, sequential recursion and squared-TT sections around equations
  (9)--(14), Algorithm 1, Proposition 2, and the complexity discussion;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21` for
  the sequential loop and `:46`--`:136` for adjacent fitting;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25`
  for paired-core mass contraction;
- `bayesfilter/highdim/squared_tt.py` and `bayesfilter/highdim/tt.py` for the
  local representation and measure convention.

Audit the distinction between

\[
\int gh^2\,d\nu,
\qquad
\int g(h^2+\tau q_0)\,d\nu,
\qquad
\frac{\int g(h^2+\tau q_0)\,d\nu}{\int(h^2+\tau q_0)\,d\nu}.
\]

The plan passes only if the implementation computes the third object.

### Phase 1: Paired-Core Operator API

Add a TensorFlow/FP64 reference mechanics API in
`bayesfilter/highdim/squared_tt.py` or a dedicated module:

- one-dimensional operator matrices
  \(M_k[g_k]=\int \phi_k\phi_k^\top g_k\,d\nu_k\);
- paired-core transfer construction;
- scalar observable contraction;
- defensive-density observable contribution;
- normalized observable quotient;
- manifest fields binding measure, basis, observable powers, and branch.

The API must reject shape/measure mismatches and non-finite or non-positive
normalizers. It must remain opt-in and must not alter the existing normalizer
or marginal behavior.

### Phase 2: Raw and Affine-Form Moments

Implement raw first/second coordinate moments and the fixed-size polynomial
automaton for affine-form moments. Required initial observables are:

- (1), (r_i), (r_ir_j);
- (z_i^3), (z_i^4);
- selected (z_i^2z_j), (z_i^2z_j^2).

Do not build dense third/fourth-order tensors. The pair set must be an explicit
immutable input and is structural, not tuned on claim data.

The first implementation may be a setup-static TensorFlow reference with
Python-unrolled axes for mechanics parity. It is not XLA/default eligible until
the core sequence is represented in a graph-native padded/masked form.

### Phase 3: JVP and Fixed-Branch Identity

Implement the total directional derivative of the exact finite contraction:

- both core factors in every paired transfer;
- operator-matrix directions when basis/chart/observable controls depend on the
  parameter;
- defensive-density numerator and normalizer;
- quotient rule;
- Cholesky and affine-form coefficient tangents;
- TT fit-core tangent and previous-marginal tangent when the fit is part of the
  finite program.

Provide a frozen prefit diagnostic arm with zero teacher-target tangent only as
an explicitly different finite approximation. Do not call it the recursively
refitted score.

### Phase 4: Contract E Integration

Extend the higher-moment correction interface to accept explicit teacher shape
targets and their tangents. Preserve the empirical-target default and expose
the TT teacher only through a route-specific opt-in identifier. Keep particle
mean/covariance as the default hybrid targets. Record both empirical and TT
targets and all post-correction residuals.

No likelihood increment may be replaced by the TT normalizer in this phase.

### Phase 5: Mechanics and LGSSM Gates

Use tiny deterministic fixtures for:

- dense quadrature parity of mass, mean, covariance, and selected mixed moments;
- defensive-density contribution with \(\tau>0\);
- affine-chart parity;
- core/operator JVP versus finite-difference diagnostics;
- quotient/JVP and branch identity checks;
- hybrid Contract E mean/covariance preservation;
- LGSSM fixed Gaussian/Wick target parity against the Kalman moments.

The LGSSM test uses an independently declared Gaussian target. It must not use
the empirical particle cloud as a teacher target.

### Phase 6: XLA/Resource Gate

Convert the candidate contraction to TensorFlow graph-native control flow or a
static padded core tensor. Run with the project FP32/TF32/XLA target only after
the FP64/reference mechanics gates pass. Record memory growth policy, device,
TF32, XLA, peak allocator bytes, TT basis/rank, fit sweeps, contraction time,
and particle workspace separately.

The no-Python-loop and no-NumPy rules apply to the XLA path. Tiny CPU/FP64
diagnostics may remain independent reference code and cannot support default
promotion.

### Phase 7: Nonlinear Feasibility Probes

Run only after Phases 1--6 pass:

- predator-prey (d=2,T=20), one seed, (N>1000);
- Austria SIR (d=18,T=9) or the repository’s score-admissible structural
  target, one seed, (N>1000);
- any available actual SV/KSC-SV scalar diagnostic only as a lower-dimensional
  comparison, not as high-dimensional evidence.

Compare empirical-target Contract E, TT-shape hybrid, and fixed Zhao-Cui
diagnostic values/scores using the same previously used metric. Report the
Kalman oracle only for LGSSM/KSC-SV where it is actually defined. For nonlinear
models, all differences are descriptive unless multi-seed uncertainty supports
a ranking.

### Phase 8: Decision and Reset Memo

Create a result note with decision and inference-status tables. State whether a
failure invalidated the harness, math, implementation, tuning, or only the
candidate. Preserve failed attempts in unique artifact roots. Write a reset
memo containing the route identity, current default (empirical target), tested
controls, evidence gaps, and next smallest discriminating experiment.

## Skeptical Plan Audit

The plan was audited before execution against the following failure modes:

1. **Wrong target:** a TT fit to particles would only restate empirical moments.
   The plan forbids that and requires an independently defined adjacent target.
2. **Wrong likelihood claim:** replacing particle likelihood by TT normalizer
   would change the estimator. The hybrid phase explicitly forbids replacement.
3. **Partial score:** differentiating only final contractions would omit TT-fit
   and marginal dependence. The derivative phase requires the full finite
   program or labels frozen-core diagnostics separately.
4. **Hidden dense scaling:** all-pair fourth moments or dense moment tensors can
   defeat the high-dimensional objective. The plan requires selected structural
   pairs and bans dense third/fourth-order materialization.
5. **Source-faithfulness drift:** the moment-teacher composition is an
   extension. The classification and paper/source anchors are mandatory.
6. **Unfair comparison:** nonlinear scores have no universal oracle. The plan
   uses the same metric and treats one-seed differences descriptively.
7. **Backend mismatch:** FP64 mechanics cannot be described as production GPU
   evidence. The XLA/TF32 phase is separate and records its own manifest.
8. **Premature default change:** existing empirical-target Contract E remains
   unchanged and canonical until independent nonlinear evidence supports an
   optional candidate.

Audit outcome: PASS for bounded mechanics execution. Promotion to nonlinear
feasibility is conditional on all earlier hard gates.

## Pre-mortem and Stop Conditions

The run could appear successful while misleading us if the TT teacher is fitted
to the same particles, if a frozen teacher is compared against a refitted score,
if dense quadrature silently truncates tails, or if rank growth is hidden by a
small fixture. These are addressed by target hashes, branch identity, separate
fit/validation designs, independent quadrature, and rank/memory reporting.

Stop immediately for a wrong target, corrupted artifact, failed dense parity,
missing required tangent, non-finite/negative normalizer, branch mismatch, or
budget exhaustion. A nonlinear bias failure after valid mechanics is a
candidate rejection and a repair trigger, not evidence that the TT contraction
math is false.

## Commands and Artifacts

All serious artifacts use a new directory under
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_20260730/`. Commands,
environment, commit, seeds, device, memory policy, TF32/XLA settings, wall
time, and output hashes belong in the run manifest. The LaTeX source is
`docs/chapters/ch32c_entropic_ot_sinkhorn.tex`.

## Execution Record

Execution stopped at the first unmet prerequisite rather than running an
invalid nonlinear score comparison.

| Phase | Status | Evidence or blocker |
|---|---|---|
| 0, source and math audit | complete for implementation planning | Paper/author-source anchors are recorded in the LaTeX section. The composition remains explicitly extension_or_invention. |
| 1--2, finite TT contractions and moments | pass | Dense quadrature tests cover normalized raw and affine-form mixed moments, including defensive density. |
| 3, derivatives | reference pass | Manual core/operator/quotient JVPs, sequential fixed-ALS replay, carried-marginal quotient, two-step recursion, and TT shape targets pass finite-difference diagnostics. Particle/OT orchestration is not composed. |
| 4, Contract E interface | partial pass | Explicit frozen shape targets feed the correction and preserve particle weighted mean/covariance. The frozen zero target tangent is diagnostic only. |
| 5, mechanics and LGSSM | pass for stated finite fixtures | Focused tests pass; scalar \(T=1\) fitted-TT moments agree closely with the Kalman/Wick diagnostic. |
| 6, XLA/resource | selected FP32-no-TF32 route passes | The fused teacher runs on trusted GPU/XLA with memory growth, no host callbacks, finite recursion, and same-program FP32/FP64 parity. TF32 fails parity and is not selected for this lane. |
| 7, nonlinear feasibility | not run | Missing graph-native particle/OT/Contract-E composition remains the score-validity veto. |
| 8, decision/reset | complete | See the result and reset notes dated 2026-07-30. |

Commands executed:

    python -m py_compile bayesfilter/highdim/zhao_cui_moment_teacher.py bayesfilter/highdim/higher_moment_contract_e.py tests/highdim/test_zhao_cui_moment_teacher.py tests/highdim/test_higher_moment_contract_e.py
    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/highdim/test_zhao_cui_moment_teacher.py tests/highdim/test_higher_moment_contract_e.py
    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/highdim/test_squared_tt_density.py tests/highdim/test_zhao_cui_fixed_adjacent_tt_tf.py tests/highdim/test_fixed_branch_fit.py
    TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/run_zhao_cui_moment_teacher_mechanics.py --output docs/benchmarks/artifacts/zhao_cui_moment_teacher_20260730/attempt04

The CPU commands intentionally hid GPU devices and are reference/mechanics
evidence only. The GPU command used the managed-session trusted GPU route with
memory growth, TF32, and XLA.

### Default and Assumption Audit

| Choice | Provenance/status | Failure mode and early diagnostic |
|---|---|---|
| Degree 28, quadrature order 64, scalar \(T=1\) LGSSM | convenience diagnostic inherited from the fixed-TT mechanics fixture; not a default | Bounded-domain/fit truncation could mimic Gaussian accuracy; report the fit residual and mean/variance/skew/kurtosis errors separately. |
| Synthetic \(m=40\), padded rank 8, basis size 6 | resource probe only | Does not measure TT fitting, particle work, or rank growth; artifact scope states contraction primitive only. |
| Frozen teacher target tangent \(=0\) | explicitly different finite diagnostic | Wrong for a recursively refitted teacher score; route identity and tangent semantics state this directly. |
| Selected pair set | structural input, not tuned | An omitted pair is unmeasured, not a zero target; separate ordered co-skew and symmetric co-kurtosis masks enforce this. |
| Existing empirical-target Contract E | unchanged baseline/default | No default promotion follows from finite TT mechanics. |

### MathDevMCP Audit

A scoped derivation-v2 audit extracted the two normalized-observable equations
but returned inconclusive:source_label_missing at the enclosing proposition
level; it did not certify or refute the result. A separate label-to-code check
for eq:bf-eot-tt-teacher-normalized-observable returned consistent for the
required numerator, normalizer, defensive-density, and \(\tau\) terms. These
are diagnostic audits. Dense-quadrature and JVP parity tests remain the
executable evidence.

## Revised Mathematical Audit and Execution Plan

The initial audit found two specification errors before recursive implementation:

1. The square-root TT was described as fitting \(\bar q_t\), but the squared
   density requires fitting \(s_t=\exp((\log\bar q_t-c_t)/2)\). The LaTeX now
   defines the Radon--Nikodym target, square-root target, and a proposition
   proving the normalized-density relation.
2. The HMC parameter was ambiguous in the adjacent coordinate order. The LaTeX
   now holds \(\theta\) fixed and integrates only declared carried latent
   variables. Integrating \(\theta\) would define a different probability
   measure.

The skeptical audit also rejected reuse of the existing scalar score helper:
one route differentiates terminal cores with incomplete ALS design dependence,
and another uses TensorFlow autodiff. The implementation plan therefore uses a
sequential fixed-ALS forward JVP with the same value equations.

### Phase A: Fixed-ALS Reference Replay

Implement and test the setup-static reference in
`bayesfilter/highdim/zhao_cui_moment_teacher_als.py`. Each update must use the
current core/tangent state, include design/target/weight tangents, and record
both algebraic solve residuals and weighted approximation residuals. The
reference route may use Python loops over an immutable schedule; it is not an
XLA/runtime candidate.

### Phase B: Teacher Recursion Integration

Build the conditional adjacent target at fixed \(\theta\), including the
Radon--Nikodym factor and square-root scale tangent. Carry the fitted TT
marginal and its tangent to the next step. Contract normalized TT moments and
their JVPs, then feed only declared shape targets into Contract E. Do not
replace the particle likelihood increment with the TT normalizer.

### Phase C: Graph-Native Candidate

After reference parity passes, convert fixed ALS and the teacher recursion to a
padded/masked TensorFlow control-flow implementation. No Python sample loops,
NumPy, autodiff score route, or runtime rank/branch selection is allowed.

### Phase D: LGSSM Gates

Run the same finite program at \(T=2,10,50\), first with a deterministic
fixed-branch comparison and then with the standard \(N>1000\) particle route.
Compare value and analytical score with Kalman and empirical-target Contract E.
Use finite differences only as an independent score check. A solve residual,
JVP mismatch, branch mismatch, or nonfinite result is a hard implementation
veto.

### Phase E: Nonlinear Feasibility

Only after all LGSSM gates pass, run one-seed \(N>1000\) predator-prey and
score-admissible Austria SIR probes. Differences are descriptive feasibility
evidence, not statistical rankings. No default promotion follows from one seed.

### Plan Review Against the Mathematics

The plan is accepted only with these bindings:

- Phase A implements equations
  `eq:bf-eot-tt-teacher-als-update` and
  `eq:bf-eot-tt-teacher-als-jvp` sequentially, not terminal-core shortcuts.
- Phase B implements
  `eq:bf-eot-tt-teacher-chart-pullback`,
  `eq:bf-eot-tt-teacher-square-root-target`, and
  `eq:bf-eot-tt-teacher-square-root-jvp` at fixed \(\theta\).
- Phase B uses the normalized observable and quotient equations, including the
  defensive-density numerator and normalizer tangent.
- Phase D tests the exact finite value program and its score, not a frozen
  teacher target or a different Zhao--Cui route.
- Phase C is a promotion gate, not an assumption: the current reference module
  remains ineligible until its graph-native replacement is tested.

The plan audit passes after these revisions. MathDevMCP extracted the ALS JVP
equation without a mismatch but returned
`unverified:manual_formalization_required`, requesting solve-residual and
conditioning checks. Those checks are now explicit in Phase A and its tests.

### Execution Update: Reference Corrections

The exact-algorithm LaTeX audit found a second scale-semantic obligation.  The
max-log shift cancels from a pure squared-TT density, but a defensive mixture
only has the same normalized density if its physical weight is scaled with the
fit target.  The document now declares the unscaled defensive weight
\(\lambda_t\), the fitted coefficient \(\tau_t=e^{-c_t}\lambda_t\), and
\(\dot\tau_t=e^{-c_t}(\dot\lambda_t-\lambda_t\dot c_t)\).  A route that holds
\(\tau_t\) fixed is explicitly a different scale-dependent approximation.

Phase A reference implementation is complete for its stated boundary:

- `square_root_target_jvp` selects and replays the fixed scale row and includes
  the scale-shift tangent;
- `scaled_defensive_weight_jvp` implements the scale-consistent defensive
  coefficient and tangent;
- `fixed_als_value_jvp` replays current-core/current-tangent ALS updates and
  records primal/JVP solve residuals and relative residuals; and
- `squared_tt_normalized_marginal_jvp` computes the carried marginal quotient
  with both core-copy tangents, defensive contribution, and normalizer tangent.
- `squared_tt_shape_targets_jvp` differentiates TT mean/covariance, Cholesky
  whitening, diagonal shape targets, and declared ordered/symmetric pair masks,
  including the defensive-mixture tangent.
- `fixed_tt_teacher_recursion_jvp` composes conditional target rows, carried
  marginal quotient tangents, warm-start fitted cores/tangents, and shape
  targets over a setup-static reference recursion.
- `apply_tt_shape_targets_reference_jvp` is the named reference boundary that
  passes non-frozen TT shape values/tangents into the existing Contract E
  higher-moment correction without changing the particle likelihood scalar.
- `tt_particle_contract_e_step_reference_jvp` composes supplied particle
  likelihood value/tangent, Sinkhorn OT, canonical Contract E-Chol, and TT
  shape repair.  Its centered finite-difference test covers particle rows,
  weights, OT, reset, TT targets, and correction while verifying that the
  supplied particle scalar is returned unchanged.

The reference tests now pass 26 focused tests, including square-root scaling,
defensive scaling, fixed-ALS JVP, and carried-marginal JVP finite-difference
checks.  The independent squared-TT/fixed-branch regression shard remains a
separate 47-test check (73 tests total).  The reference primitives are now
composed in a reusable two-step recursion test, but Phase B is still not
complete for reference mechanics.  Model-scale integration and graph-native
execution remain separate promotion phases.

The whole-file MathDevMCP math-to-code audit was attempted after the edits but
the tool returned a recoverable AST `SyntaxError` while parsing the code path;
`py_compile` and executable tests pass.  This is recorded as an audit-tool
limitation, not as mathematical certification.  The scoped proposition audit
remains diagnostic and does not close the recursive-score gate.

The final measure audit found that a defensive marginal under Lebesgue mass
must include the domain volume of every integrated coordinate.  The shared
`SquaredTTDensity` helper now applies that factor, with a dedicated regression
test.  Reference-measure behavior is unchanged.

The isolated chapter LaTeX build passed.  A full monograph build was attempted
but stopped on a pre-existing missing SSL-LSTM trace PDF in a later chapter;
that unrelated asset failure does not invalidate this chapter’s isolated build.

## Phase C Execution Contract (2026-07-30)

The graph-native promotion phase uses a new route rather than changing the
FP64/setup-static reference.  Its first gate is exact finite-program parity on
tiny fixtures; model-scale or nonlinear claims remain forbidden until that
gate passes.

### Static representation and solve

- Cores and core tangents have shape `[axis, R, B, R]`.  A fixed Boolean mask
  declares the active left-rank, basis, and right-rank entries on each axis.
  Boundary rank one is represented by channel zero.  Inactive entries are
  multiplied by zero on input and after every update.
- Basis evaluations have shape `[axis, row, B]` and are immutable prepared
  inputs.  Basis construction, ranks, row design, sweep order, fit count,
  ridge, and masks are fixed before tracing and are not selected from the HMC
  parameter or runtime residuals.
- Every active reference core update is embedded in the common padded column
  space.  Inactive design columns are zero.  A strictly positive fixed ridge
  makes their unique solution zero and leaves the active normal equations
  unchanged.  Therefore ridge zero is outside this candidate route rather than
  being silently assigned a padded pseudoinverse convention.
- The value solve uses objective-preserving column scaling followed by a
  Cholesky solve of the scaled normal equations.  The returned coefficient is
  a solution of the same unscaled ridge normal equations as the reference; the
  scaling changes conditioning, not the objective.
- The analytical JVP differentiates the unscaled ridge normal equations.  It
  includes current left/right environment tangents, target tangents, and
  optional weight tangents.  No autodiff or finite difference participates in
  the score path.

The fixed update schedule is consumed by `tf.while_loop`.  Environment
construction and its tangent are also TensorFlow control flow.  The graph must
contain no `PyFunc`/`EagerPyFunc`, and the implementation module must not import
NumPy.

### Gates and diagnostics

Each update reports the scaled augmented-system condition estimate, unscaled
normal condition estimate, primal/JVP algebraic residuals, weighted fit/JVP
residuals, and a validity bit.  The scaled condition is
`sqrt(lambda_max/lambda_min)` of the scaled normal matrix; the derivative
condition is `lambda_max/lambda_min` of the unscaled normal matrix.  A
nonfinite value, nonpositive eigenvalue, condition-veto breach, or residual
veto poisons the candidate output with nonfinite values and sets validity
false; there is no eager or reference fallback.

### Skeptical audit

| Risk | Resolution before execution |
|---|---|
| Padding changes the fit | Zero design columns plus positive ridge give zero inactive coefficients and the same active normal equations.  Tiny variable-rank parity is mandatory. |
| Scaling changes the derivative target | The primal coefficient solves the unscaled normal equations; the JVP differentiates those equations directly. |
| A stable primal hides an invalid score solve | Scaled primal and unscaled derivative condition estimates and residuals are separately gated. |
| Runtime adaptation enters HMC | Rank, basis, masks, rows, schedule, ridge, and veto thresholds are prepared immutable inputs. |
| Graph-native label hides host callbacks | Concrete graph operation types and `While` presence are inspected; `PyFunc` and `EagerPyFunc` are hard vetoes. |
| FP32 agreement is mistaken for exactness | FP64 graph/reference parity is tested first. The selected FP32-no-TF32 route has a separate resource/parity gate with dtype-appropriate tolerances. |
| Successful ALS mechanics is mistaken for filtering evidence | LGSSM and nonlinear phases remain gated on the recursive teacher, particle scalar separation, and full score composition. |

Audit outcome: PASS for implementing and testing the bounded graph-native ALS
candidate.  The active evidence contract, promotion criteria, nonlinear
continuation vetoes, and nonclaims above are unchanged.

### Phase C execution result

The FP64 graph-native mechanics gate passes.  Nine tests cover fixed-ALS
value/JVP reference parity, centered finite differences, fail-closed condition
handling, concrete graph control flow without host callbacks, normalized
marginal parity, two-step carried recursion, complete fourth-order TT shape
targets/JVP reference parity, a complete shape-JVP finite difference, and the
fused per-time marginal/shape recursion with concrete-graph inspection.  The
runtime module has no NumPy import, no autodiff, and no Python numerical loop.
The pre-existing 73-test reference/regression shard also passes, `py_compile`
and `git diff --check` pass, and the isolated 43-page LaTeX chapter rebuilds.

Trusted GPU preflight passed on an RTX 4080 SUPER and verified TensorFlow memory
growth. Attempts 09 and 10 executed the full TF32 candidate and failed the
declared relative-parity veto. Attempt 11 disabled TF32 and passed. Subsequent
paired score and performance diagnostics selected the FP32-no-TF32 route for
this lane. Historical pre-execution failures remain preserved.

The repaired GPU harness now compares the identical fused FP32 and FP64 graph
programs.  Its fixture-mechanics vetoes are maximum absolute error \(2\times
10^{-3}\) and maximum relative error \(5\times10^{-3}\).  These are conservative
FP32 nomination thresholds based on the scale of the bounded synthetic fixture,
not transferred model defaults or scientific accuracy criteria.  A breach
blocks GPU mechanics promotion and triggers numerical diagnosis; passing does
not establish model value/score accuracy.

Implementation review also rejected a thin wrapper around
`tt_particle_contract_e_step_reference_jvp` as the graph-native particle step.
That adapter is explicitly setup-static, contains eager `.numpy()` validity
branches, and calls `_restore_cloud_jvp_core`; it does not bind the canonical
streaming Contract E route identity or per-scope tuning artifact.  Promoting it
would be wrong relative to the repository's canonical Contract E and tuning
policies.  The next particle integration must compose the fused TT targets with
the repository-issued canonical streaming value/JVP route and preserve its
identity, total derivative, chunk policy, and tuning scope.
