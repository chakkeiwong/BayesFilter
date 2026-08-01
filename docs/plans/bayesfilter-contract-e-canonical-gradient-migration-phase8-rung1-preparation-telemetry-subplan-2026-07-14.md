# Phase 8 Repair Subplan: Rung 1 Preparation And Telemetry

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `CLOSED_PREPARATION_TELEMETRY_PASSED_TARGET_NUMERICAL_DESIGN_BLOCKED`

## Phase Objective

Create the target-prefix prepared-input and telemetry infrastructure needed to
run canonical LGSSM `T=1` and `T=10` diagnostics without inheriting an
unexamined residual design, ridge, reset mask, or historical raw route. This
subplan does not execute or inspect a canonical target-prefix value or score.

## Entry Conditions

- Phases 0-7 are closed at their narrow gates.
- Rung 0A shared-core dtype repair passed; formal Phase 1 FD remains
  inconclusive.
- Rung 0B tiny-fixture canonical/Kalman harness passed as descriptive harness
  evidence only.
- The exact target observations come from dataset seed `81100`; Kalman uses the
  same transition-first likelihood as the canonical loop.
- Primary `T=50,N=10000` execution remains forbidden until the owner-frozen
  statistical amendment and formal FD gate pass.

## Preparation Design

The builder accepts observations and an ordered list of estimator seeds. Every
random tensor is generated in `tf.float64` with
`tf.random.stateless_normal(..., alg="philox")` from an exact signed-int32 key
`[root_seed, domain_tag]`, then cast once to the requested canonical dtype.
`root_seed` is the declared estimator seed and must fit signed int32. Domain
tags are convenience identity labels, not scientifically calibrated values:

```text
initial noise:       101
transition time t:   1000 + t
residual time t:     2000 + t
```

Time is zero-based and the builder requires `0 <= t < 1000`, so namespaces do
not collide. For each ordered seed, the exact draw shapes before stacking are
initial `[N,3]`, transition `[T,N,3]`, and residual raw `[T,N,3]`; stacked axes
are `[B,N,3]`, `[B,T,N,3]`, and `[B,T,N,3]`. The identity records TensorFlow
version, algorithm, key dtype, root seeds in order, tags, time convention, raw
draw dtype, final dtype, shapes, and hashes of every realized prepared tensor.

For residual raw normal draws `Z`, centering and scaling occur in float64 before
the final cast and implement the normative construction

```text
Xi = sqrt(N/(N-1)) * (Z - mean_N(Z)).
```

The realized tensors and construction identifier are hashed. Domain identifiers
are identity labels, not scientific tuning values; changing them creates a new
prepared target.

The builder has no silent reset or transport defaults. Callers must explicitly
provide fixed reset masks, epsilon, scaling, Sinkhorn steps, and chunk sizes.
Active-all may be instantiated later only as a labeled baseline hypothesis;
no-reset is an explanatory negative control and cannot replace Contract E.

The builder has no ridge generator, ladder, scale formula, or default. The
caller must provide the complete positive finite `[B,T]` prepared-ridge tensor.
The builder validates, casts, and hashes it but does not select, alter, or
interpret it. Target-specific ridge hypotheses remain blocked for a separately
reviewed pre-result execution plan with a covariance/Cholesky error analysis and
raw-ridge-bias contract.

## Telemetry Definitions

Telemetry is evaluated for every batch/time immediately after the quotient and
Contract E forward computation and before the fixed reset mask selects the next
state. The active mask is serialized alongside it. All tensors retain the
canonical callable dtype; later artifacts must record dtype, shape, values, and
a hash rather than rounding to a reporting dtype.

For quotient mass `M[b,t,i]`, record the full mass, `min_i M`, and
`max_i abs(M-1)`. For residual design `Xi`, record full column sums
`sum_i Xi_i` and absolute scales `sum_i abs(Xi_i)`.

For reset target moments `(mu_w,Sigma_w)`, output moments
`(mu_out,Sigma_out)`, injected covariance `Sigma_tilde`, affine `A`, and ridge
`lambda`, record the raw tensors and:

`(mu_w,Sigma_w)` are exactly
`ledh_contract_e_reset_tf._weighted_moments(source_particles,
normalized_weights)`: probability-weighted mean and centered covariance on
particle axis 1, with weights already normalized to sum one. `mu_out` and
`Sigma_out` are exactly
`ledh_contract_e_reset_tf._uniform_moments(output_particles)`, and
`Sigma_tilde` is exactly `_uniform_moments(injected_particles)[1]`: equal-weight
population mean/covariance on particle axis 1 with divisor `N`, centered by the
corresponding equal-weight mean. No sample-covariance `N-1` divisor or host
recomputation is allowed.

```text
mean_residual = mu_out - mu_w
R_ridged = A (Sigma_tilde + lambda I) A^T - (Sigma_w + lambda I)
S_ridged = |A| |Sigma_tilde + lambda I| |A|^T + |Sigma_w + lambda I|
R_raw = Sigma_out - Sigma_w
R_raw_predicted = lambda (I - A A^T)
R_raw_prediction_error = R_raw - R_raw_predicted
```

Record each residual/scale matrix in full plus its per-batch Frobenius norm;
record the mean-residual infinity norm. No normalized residual or threshold is
introduced here. Record all three Cholesky diagonal vectors and the existing
factor condition proxy
`||L||_F * ||triangular_solve(L,I)||_F` separately for gap, target, and injected
factors. Reductions are the TensorFlow operations already executed by the
canonical dtype core; no host recomputation defines the telemetry.

## Required Artifacts

- an owned TensorFlow prepared-input builder with explicit identity metadata;
- tests for seed/domain repeatability, seed separation, shapes, exact mask
  replay, residual centering, caller-supplied positive ridge, and absence of hidden
  defaults;
- canonical primal telemetry for per-time row mass/residual, mean restoration,
  ridged identity, raw covariance residual and prediction, Cholesky diagonals,
  condition proxies, and realized ridge;
- tests proving telemetry addition leaves the Phase 5 objective/score hex values
  unchanged and repeated calls identical;
- a preparation/telemetry close or blocker record;
- a refreshed Rung 1 execution subplan only if this infrastructure passes.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can target-prefix inputs and required diagnostics be constructed reproducibly without selecting a scientific default or changing the canonical scalar? |
| Baseline | Frozen Phase 5 v2 scalar/score hex values; Phase 1 residual/ridge mathematics; target dataset seed `81100` |
| Pass criterion | Deterministic source-bound preparation, exact required identities, complete named telemetry, and unchanged frozen scalar/score identity |
| Hard vetoes | parameter-dependent prepared randomness/ridge; adaptive ridge inside callable; silent reset/transport default; raw route import; scalar/score drift; nonfinite preparation; missing identity/hash field |
| Explanatory only | caller-supplied ridge magnitudes and prepared-noise summaries before any filter evaluation |
| Not concluded | ridge adequacy, target-prefix chart validity, Kalman equivalence, FD promotion, GPU feasibility, HMC, leaderboard, or default readiness |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Gaussian residual `Z` | Phase 1 normative admissible schedule | Canonical fixed realized residual construction | Particular draw changes finite target | Frozen PHILOX keys, shapes, dtype, hashes, and later multi-seed design | Identity choice, not promoted default |
| Center/scale `Xi` | Phase 1 normative equation | Required zero-mean residual covariance convention | Reduction error or `N=1` invalidity | centering residual and explicit `N>1` rejection | Required |
| Prepared ridge | Phase 1 requires a positive parameter-independent prepared tensor | Infrastructure must not invent a magnitude | Caller could provide an invalid or unjustified hypothesis | shape/finiteness/positivity validation and source hash | Required explicit input; adequacy blocked |
| Reset mask | No target default established | Must be part of finite program identity | Active-all silently promoted | require caller-supplied tensor | Open explicit input |
| Sinkhorn/chunks | Existing tiny settings are not target defaults | Avoid cross-shape transfer | Convenient settings fail for wrong reason | require caller-supplied values | Open explicit inputs |

## Skeptical Plan Audit

Decision: `PASS_FOR_IMPLEMENTATION_AND_TESTS_ONLY`.

- No target filter value/gradient is executed, and the builder cannot generate
  or select a ridge hypothesis.
- Kalman is not involved in preparation selection.
- The builder cannot silently promote active-all or tiny transport settings.
- Telemetry is appended to outputs but is forbidden from changing the scalar,
  derivative composition, branches, or prepared inputs.
- Formal FD and owner statistical blockers remain unchanged.

## Required Checks And Reviews

1. Implement builder and telemetry with TensorFlow only.
2. Add focused unit and frozen-hex preservation tests.
3. Run CPU-hidden tests with `CUDA_VISIBLE_DEVICES=-1`.
4. Re-run Phase 5 canonical tests and source-bound float64 certificate.
5. Run Python compilation and scoped `git diff --check`.
6. Write a close/blocker record and draft the next Rung 1 execution subplan.
7. Obtain one bounded review of the next material execution design.

## Forbidden Claims And Actions

- Do not execute or inspect target-prefix Contract E value/score output in this
  subplan.
- Do not generate or select a ridge, reset policy, epsilon, scaling, step count, or
  chunk size from results.
- Do not call the ridge ladder a default or scientific adequacy range.
- Do not import or invoke raw/compact historical LEDH routes.
- Do not run GPU, `T=50,N=10000`, HMC, nonlinear migration, leaderboard, or
  release work.
- Do not claim the telemetry thresholds are resolved; this subplan records
  quantities and identities only.

## Exact Next-Phase Handoff Conditions

Draft Rung 1 execution may begin only if preparation tests pass, telemetry is
complete for the declared fields, the frozen Phase 5 objective/score hex values
remain exact, no hidden default/raw dependency exists, and the next plan freezes
which explicit baseline hypotheses may be executed and how their outputs are
classified without result-dependent selection.

## Stop Conditions

Stop for scalar/score drift, parameter-dependent preparation, inability to
construct centered residuals, hidden historical dependency, missing telemetry
that would make a later run uninterpretable, campaign-clock exhaustion, or a
material review finding that invalidates the preparation design. Local shape,
serialization, or test-harness defects are repair triggers.
