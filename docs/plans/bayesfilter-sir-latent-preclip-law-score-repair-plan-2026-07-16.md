# SIR Latent Pre-Clipping Law And Score Repair Plan

Date: 2026-07-16

Status: `PARTIAL_SUCCESS_REPAIR_SCOPE_CLOSED_ADMISSION_GAPS_PRESERVED`

## Research Intent Ledger

| Field | Binding intent |
| --- | --- |
| Main question | Can the Austria SIR clipped simulator be represented by a continuous latent-state filtering target whose value and total score can be validated before any leaderboard or HMC claim? |
| Mechanism under test | Filter the Gaussian pre-clipping state `z_t`; expose the simulator state as `x_0=z_0` and `x_t=C(z_t)` for `t>=1`, where `C` clips susceptible coordinates only. |
| Expected failure mode | Incorrect first-step clipping, stopped derivatives through the filtering marginal or reset/transport, dense-reference truncation, clipping-boundary nondifferentiability, or reuse of the historical raw-barycentric route as canonical. |
| Promotion criterion | Paired-noise simulator-law identity; refined reduced-model value/score reference; same-scalar total-score checks; then a genuine Contract E--Chol SIR factory with canonical repository-owned identity. |
| Promotion veto | Any law mismatch, nonfinite/refinement-invalid reference, score/FD mismatch, stopped-path negative control that does not differ, forged canonical identity, or model-specific support failure. |
| Continuation veto | The latent representation is proved not law-equivalent, required observations depend on unavailable pre-clipping information, the reference harness is invalid, or the approved compute budget is exhausted. Candidate approximation failure alone is not a continuation veto. |
| Repair trigger | A localized time-order, derivative, shape, compilation, serialization, or numerical-support failure under an otherwise valid target. |
| Explanatory diagnostics | Boundary mass, grid-refinement differences, clip frequency, score decomposition, runtime, compile time, graph topology, and CPU/GPU deltas. |
| Forbidden conclusion | No exact nonlinear likelihood, canonical Contract E admission, fixed-TTSIRT correctness, full-horizon readiness, HMC readiness, or leaderboard completeness follows from a component, FD, smoke, or short-prefix pass. |

## Target Mathematics

Let `C` clip susceptible coordinates to zero and leave infectious coordinates
unchanged.  The source simulator draws an unclipped initial state and clips only
after later process-noise draws:

\[
  x_0 = z_0,\qquad z_0\sim N(m_0,P_0),
\]
\[
  z_t=F_\theta(x_{t-1})+L_Q\epsilon_t,\qquad
  x_t=C(z_t),\quad t\geq1.
\]

The repaired continuous filtering state is therefore `z_t`, with

\[
  z_t\mid z_{t-1}\sim
  N\!\left(F_\theta(C_{t-1}(z_{t-1})),Q\right),
  \qquad
  C_s(z)=\begin{cases}z,&s=0,\\C(z),&s\geq1.\end{cases}
\]

and observation density `g_theta(y_t | C_t(z_t))`.  No Jacobian is attached to
`C`: it is part of the generative map, not a change of integration variables.
The latent target is an `extension_or_invention` representation of the source
simulator law.  The author's unclipped Gaussian-density program remains a
separately named diagnostic and is not this target.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the repaired latent target compute the clipped simulator's observed-data law and a derivative of that same finite value program? |
| Baseline/comparator | Paired-noise source simulator for law identity; independently refined TensorFlow dense quadrature for reduced `J=1`; autodiff and centered FD of the same reduced scalar; historical Gaussian-density route only as a wrong-target negative control. |
| Primary criterion | Phase-specific gates below pass without changing the target, observations, parameterization, or derivative ownership. |
| Veto diagnostics | First-step time-order mismatch; paired paths differ; reference boundary/refinement check fails; any nonfinite value/score; manual/AD/FD mismatch; canonical factory absent or identity forgeable. |
| Explanatory only | Clip frequency, raw value/score gaps between viable approximations, compile/runtime, and short-prefix behavior. |
| Artifact | Versioned tests/results under `docs/plans` and `docs/benchmarks/artifacts/sir_latent_preclip_repair_20260716/`. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| `z_t` pre-clipping state | Algebraic factorization of source `st_process.mlx` and local simulator | Preserves the simulator push while restoring a continuous integration law | Wrong time order at `t=0` | Paired-noise bitwise path test plus wrong-time-order negative control | hypothesis to prove in Phase 1 |
| Three log-scale parameters | Existing `ParameterizedZhaoCuiSIRSSM` benchmark contract | Keeps comparisons on the established SIR inference surface | Extension confused with author inference target | Identity/manifest assertions | reviewed existing benchmark extension |
| TensorFlow float64 references | Repository backend and reference policy | Needed for stable derivative/refinement diagnostics | Reference shares candidate defect | Direct density formulas and independent FD; no candidate transport calls | reference choice |
| Reduced `J=1` | Existing lower-rung fixtures | Smallest nontrivial susceptible clipping and SIR dynamics | Does not expose spatial coupling | Complete `J=1`, then add `J=2` before broad claim | diagnostic baseline |
| Quadrature order/range | To be selected by refinement and boundary-mass diagnostics, not fixed in advance | Prevents arbitrary box becoming an oracle | Truncation or resolution bias | Two-order/two-range ladder | hypothesis until diagnostics pass |
| `0.05*sqrt(p)` | Owner policy | Individual-direction same-scalar FD only | Misused as oracle agreement | Artifact labels diagnostic scope | reviewed FD-only policy |
| XLA/GPU | Repository default | Required for production-target execution | CPU smoke mislabeled as readiness | Trusted GPU probe and device artifact | later execution gate |

## Skeptical Pre-Execution Audit

The audit checked wrong baselines, proxy promotion, stop conditions, fairness,
hidden assumptions, stale context, environment mismatch, and artifact adequacy.
It found and repaired these material issues before execution:

1. The first draft implicitly clipped `z_0`.  The source simulator does not;
   the target now binds `x_0=z_0` and clips only for `t>=1`.
2. Existing SIR FD/GPU tests cover a historical raw-barycentric finite scalar or
   local complete-data terms.  They are not baselines for the repaired filtering
   law and cannot close any phase here.
3. The repository has a canonical Contract E--Chol graph only for LGSSM.  The
   experimental TP adapter and historical raw route cannot be relabeled.  A
   SIR-specific repository-owned factory is a required later artifact.
4. A fixed quadrature box would not answer the oracle question.  The plan now
   requires range and order refinement plus boundary-mass reporting.
5. Passing same-scalar FD proves derivative wiring only.  Agreement with the
   reduced filtering reference is a separate promotion criterion.
6. Fixed-TTSIRT derivative ownership is a separate comparator phase.  Its
   absence cannot be hidden by a passing Contract E candidate and does not block
   the earlier law/oracle phases.

Audit verdict: `PASS_AFTER_REVISION`.  Phases 1--3 may execute.  Later GPU and
full-row phases execute only after their inherited gates pass.

Execution amendment after Phase 3 diagnostics: the descriptive particle ladder
uses `N=8,16` in the focused unit suite.  `N=32` is retained as a separately
executed diagnostic because repeated combined pytest/subprocess runs failed to
emit a terminal result, while the isolated `N=32` command completed with finite
value, score, positive reset mass, and a valid chart.  This is a harness-output
classification, not a scientific exclusion of `N=32`.

## Phase 1: Latent Target And Law Identity

Objective: implement a separately identified TensorFlow latent SIR target with
the exact source time order.

Required artifacts:

- `bayesfilter/highdim/sir_latent_preclip_tf.py`;
- focused unit tests for projection, densities, paired-noise identity, and
  wrong-time-order negative control;
- model identity declaring `extension_or_invention` and denying canonical
  Contract E status.

Gate: paired physical paths and observations agree exactly for fixed noise at
`J=1,2,9`, including a fixture that actually clips; the deliberately wrong
`t=0` clipping route differs.

## Phase 2: Reduced Dense Value And Score Reference

Objective: compute the `J=1`, `T=1,2` latent filtering value with deterministic
TensorFlow quadrature and differentiate that same scalar.

Required checks:

- order and support-range refinement;
- finite normalized filtering masses and boundary diagnostics;
- autodiff score versus centered coordinate FD;
- an increment-sum check and a stopped-previous-marginal negative control.

Gate: the finest two references meet a tolerance justified by their observed
refinement gap; AD/FD meets numerical error diagnostics.  No arbitrary
cross-method equivalence threshold is introduced.

## Phase 3: Bounded Contract E Candidate

Objective: build a reduced latent-SIR finite-particle candidate using the actual
Contract E--Chol reset and total derivative composition.

Required checks:

- repository-owned route identity from the executed callables/settings;
- value and total score of the same finite program;
- manual JVP/VJP versus diagnostic autodiff and same-scalar FD;
- reset moment/weight and streaming-transport derivative contributions;
- comparison with the Phase 2 reference at `T=1,2`.

Gate: derivative wiring passes and reference gaps are reported separately.
Candidate approximation failure triggers feature/particle repair; it does not
invalidate the latent target.

## Phase 4: Reduced Spatial And Prefix Ladder

Objective: extend to `J=2`, then SIR `d=18` at `T=1,2,5` on CPU float64.

Gate: all model-specific support, same-scalar, fail-closed, and prefix reference
checks pass.  `J=1` evidence alone cannot promote spatial coupling.

## Phase 5: GPU/XLA And Full Row

Objective: execute the exact admitted factory on trusted GPU/XLA, then attempt
`d=18,T=20` within the campaign budget.

Compute budget: at most three GPU attempts, at most two hours total wall time,
and fresh versioned output directories for every attempt.  Localized harness or
compiler failures may be repaired and retried within this budget.

Gate: functional time loops, no Python time unrolling in the compiled factory,
finite value/score, device provenance, same-scalar checks, and CPU/GPU
comparison pass.  This is still not HMC or leaderboard readiness by itself.

## Phase 6: Fixed-TTSIRT Comparator Derivatives

Objective: add previous-marginal and fixed-fit/proposal/transport derivative
ownership to the source-route comparator.  For a frozen linear solve
`A(theta)c(theta)=b(theta)`, use

\[
  \dot c=A^{-1}(\dot b-\dot A c)
\]

under the exact fitted branch; do not rerun adaptive fitting inside each HMC
evaluation.

Gate: source anchors, fixed-branch identity, total same-scalar score, and
lower-rung Contract E comparison pass.  Otherwise retain the existing precise
blockers.

## Phase 7: Terminal Audit

Write a result with separate engineering, numerical, and scientific ledgers;
a decision table; inference-status table; run manifests; post-run red team; and
explicit remaining gaps.  Only a passing Contract E--Chol factory may seek
canonical/HMC/leaderboard admission.

## Stop And Handoff Conditions

- Stop a phase for invalid target law, invalid oracle, corrupted/missing
  artifacts, a genuine continuation veto, or exhausted campaign budget.
- Do not stop merely because a candidate approximation fails when the next
  phase or repair is designed to address that failure.
- At each phase close: run focused checks, write/update the result, refresh the
  next phase, and audit consistency and boundary safety before continuing.

## Execution Close Status

The simulator-law and same-finite-scalar score repair scope is executed and
closed.  This does not mean every admission phase passed:

| Phase | Close status | Evidence or handoff |
| --- | --- | --- |
| 1 | complete | paired-noise law identity at `J=1,2,9`, including forced clipping and the initial-time negative control |
| 2 | complete for `J=1` | manual previous-marginal score agrees with autodiff/FD; split-grid refinement is recorded in the result |
| 3 | complete for the bounded candidate | Contract E--Chol total tangent agrees with autodiff/FD and reset ablation changes the next-step scalar/score |
| 4 | partial | `d=18,N=32,T=2,5,20` finite-score checks pass; the planned `J=2` spatial oracle remains open |
| 5 | partial | pre-boundary-fix GPU/XLA and CPU/GPU parity pass; exact-source post-fix GPU certification remains open because the three-attempt budget is exhausted |
| 6 | blocked as planned | fixed-TTSIRT previous-marginal and proposal/transport derivatives remain unimplemented under their precise blocker labels |
| 7 | complete | terminal result separates engineering correctness, approximation accuracy, and admission nonclaims |

Terminal audit verdict: `PARTIAL_SUCCESS_REPAIR_SCOPE_CLOSED_ADMISSION_GAPS_PRESERVED`.
The current candidate may support continued repair work, but it may not be
promoted to canonical, HMC, Zhao--Cui-comparator, or leaderboard status.
