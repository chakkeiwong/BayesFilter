# Phase 3 Subplan: Production Cloud-Level Contract E Reset

Date: 2026-07-13

Status: `REVIEWED_ACTIVE`

Master program:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-master-program-2026-07-13.md`

## Phase Objective

Implement the checked Contract E--Chol cloud reset as a BayesFilter-owned
TensorFlow module with XLA-on-by-default forward, manual JVP, and manual VJP
APIs. The primitive accepts source particles, normalized probability weights,
already transported row-quotient particles `Y+`, the realized fixed centered
residual design `Xi`, and a prepared positive parameter-independent ridge.

The phase must establish small-`N` float64 forward/JVP/VJP parity against the
independent dense benchmark reference and TensorFlow autodiff. It must expose
direct source-moment, direct probability-weight, transported-cloud,
residual-design, and ridge adjoints separately. It does not compose the
transport pullback or claim a full-filter total gradient.

## Entry Conditions Inherited From Phase 2

- Contract E--Chol ID is `contract_e_chol_v1`.
- Derivative composition ID is
  `contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1`.
- The normative equations, population covariance convention, row-vector
  Cholesky orientation, and fixed-ridge semantics are frozen in the Phase 1
  specification.
- Schema v2 binds actual callables and prepared inputs, but the public route
  registry is intentionally empty and no artifact is admitted.
- V1/raw routes remain historical and fail closed for admission.
- Current consumers are v1-only fail-closed, not v2-aware.
- Claude repository disclosure remains platform-blocked; use fresh bounded
  Codex review without retrying Claude.
- Unrelated dirty model/harness files are outside this phase.

## Exact Implementation Boundary

Add:

- `bayesfilter/highdim/ledh_contract_e_reset_tf.py` with exact symbols:
  - `contract_e_chol_cloud_forward_tf`;
  - `contract_e_chol_cloud_jvp_tf`;
  - `contract_e_chol_cloud_vjp_tf`.
- `tests/highdim/test_ledh_contract_e_cloud_reset_phase3.py`.

All three public functions use `tf.function(..., jit_compile=True)` by default.
An internal undecorated core may exist so deliberate CPU float64 reference tests
can run eagerly without being represented as production evidence. The
BayesFilter-owned module must not import or call NumPy.

The input shapes are:

```text
source_particles       [B,N,d]
normalized_weights     [B,N]
transported_particles  [B,N,d]
residual_design        [B,N,d]
ridge                  [B]
```

The canonical route fixes `rho=1`; `rho` is not an API input. The primitive
does not accept a dense `N x N` matrix and does not construct `Y+` internally.
It does not center or rescale caller noise: it accepts the realized `Xi` that is
hashed as a prepared input. It does not normalize weights, floor masses, choose
or escalate ridge, clip eigenvalues, or change branches.

The forward output includes particles and sufficient diagnostics/intermediates
to audit:

- weighted source mean and population covariance;
- uniform `Y+`, injected-cloud, and output means/covariances;
- covariance gap, all three Cholesky factors, affine map, and prepared ridge;
- residual-design column sum and scale;
- ridged covariance identity residual and componentwise scale;
- raw covariance residual and its exact prediction `lambda*(I-AA^T)`;
- mean residual, finite status, factor-diagonal status, and condition estimates.

The public manual JVP returns only the tangent of output `particles` for all
five input tangents. It must analytically differentiate weighted/uniform
moments, Cholesky factors, triangular solves, residual injection, and the final
affine map. Boolean finite/factor status, hashes, branch identities, residual
norms, and condition estimates are report-only forward diagnostics and are not
differentiated. Private test helpers may expose the differentiable numeric
intermediates `target_mean`, `target_cov`, `plus_cov`, `gap`, `residual_chol`,
`injected_particles`, `injected_mean`, `injected_cov`, `target_chol`,
`injected_chol`, `affine`, and `particles` with their tangents solely to localize
a failed defining-equation residual. `ForwardAccumulator` and `GradientTape`
are forbidden inside the owned module; they are allowed only in tests as
independent references.

The manual VJP returns exactly:

```text
source_particles       direct weighted-moment adjoint only
normalized_weights     direct probability-coordinate adjoint only
transported_particles  cloud adjoint to be composed in Phase 4
residual_design        parity/audit adjoint; fixed in the canonical route
ridge                  parity/audit adjoint; fixed in the canonical route
```

The only accepted output cotangent is `upstream_particles`; the VJP scalarizes
`sum(upstream_particles * particles)`. It does not accept or imply cotangents
for diagnostics or auxiliary intermediates. Condition estimates are explanatory
forward reports only and never a differentiated target.

The Phase 4 transport pullback is responsible for adding the transported path
to source/log-weight coordinates. Phase 3 must not label the direct source
adjoint as the full source derivative.

Because the Phase 2 production route factory requires reset, value, and
gradient symbols together, Phase 3 does not install a partial route
specification or issue an artifact. The exact reset symbols and hashes are
recorded for later full-route registration. The production route count remains
zero until the Phase 5 same-scalar value/gradient symbols exist.

## Dense Reference Comparator

The exact reference is the checked fixed-ridge path in
`docs/benchmarks/contract_e_reset_tf.py`:

- `contract_e_cholesky_ridge_reset_fixed_ridge`;
- `contract_e_cholesky_ridge_reset_fixed_ridge_vjp`.

For comparator construction only:

```text
Y+ = matrix @ source_particles
Xi = sqrt(N/(N-1)) * center(residual_noise)
rho = 1
```

The new cloud forward is then compared using exactly those realized `Y+` and
`Xi`. Dense VJP parity is checked by composing the cloud adjoints back through
those two preparation maps:

```text
dense source bar = cloud direct source bar + matrix^T(cloud Y+ bar)
dense matrix bar = (cloud Y+ bar) times source_particles^T
dense noise bar  = sqrt(N/(N-1))*center(cloud Xi bar)
```

Weights and ridge compare directly. This comparator checks the same finite
scalar while preserving the cloud primitive's separation of derivative paths.

## Skeptical Plan Audit

Decision: `PASS_FOR_IMPLEMENTATION_PARITY_WITH_PROMOTION_BLOCKERS_PRESERVED`.

| Risk | Finding and control |
| --- | --- |
| Wrong baseline | The fixed-ridge dense helper is a checked diagnostic reference; adaptive ridge and raw-barycentric routes are excluded. |
| Proxy promotion | Tiny parity establishes implementation agreement only. Covariance diagnostics cannot promote the full filter or route. |
| Hidden target change | `rho` is fixed to one, `Y+` and realized `Xi` are explicit inputs, and no weight/ridge adaptation occurs. |
| Partial derivative mislabeled total | VJP fields are named by path; direct source/weight and transported-cloud adjoints remain separate. |
| Circular gradient check | Manual JVP/VJP are checked against both the structurally different dense wrapper and TensorFlow autodiff/duality; no single comparator is sufficient alone. |
| Arbitrary tolerance | Engineering acceptance uses only the predeclared binary-exact certificate below and requires bitwise equality. Non-certificate charts are explanatory until an independently justified kernel/conditioning bound exists. No observed error, ULP count, or multiplier becomes a boundary. |
| XLA mismatch | Public wrappers are XLA-on by default; eager CPU float64 is labeled reference only. A tiny CPU-XLA smoke checks compilability, not GPU readiness. |
| Invalid chart | Nonfinite values or a nonpositive Cholesky diagonal are hard vetoes. No adaptive repair is hidden inside the primitive. |
| Dense production allocation | The new module has only `B*N*d` and `B*d*d` state and no `N*N` input/output; source audit checks this. Measured production memory belongs to Phase 4. |
| Environment mismatch | All Phase 3 executions are deliberate CPU-only reference/smoke checks with GPU hidden. Trusted GPU evidence begins in Phase 4. |
| Dirty worktree collision | New source/test files are used. The dirty benchmark helper is read-only and hash-checked before/after. |
| Unresolved adequacy budgets | They block promotion, not the smallest implementation/parity diagnostic. Phase 3 result must preserve them and may hand engineering work to Phase 4 without claiming the reset gate passed. |

## Predeclared Binary-Exact Engineering Certificate

The acceptance fixture is frozen algebraically before implementation output is
observed. It uses binary64 dyadic inputs whose exact intermediate results remain
representable. No tolerance is used.

For `N=8,d=2`, define the unscaled source rows

```text
( 1,  1), (-1,  1), ( 1, -1), (-1, -1),
( 1,  0), (-1,  0), ( 0,  1), ( 0, -1).
```

Use uniform weights `1/8`, `Y+=0`, `Xi=R`, and `ridge=1/4`, where `R` is the
displayed source matrix. The source mean is exactly zero and its population
covariance is exactly `3/4 I`; therefore

```text
gap + ridge*I       = I,
target_cov+ridge*I  = I,
tilde_cov+ridge*I   = I,
L_gap=L_target=L_tilde=I, A=I, particles=R.
```

The second declared batch scales the source to `2R`, retains `Xi=R`, and uses
`ridge=1`. Its source covariance is `3I`, its gap/target Cholesky factors are
`2I`, injection produces `2R`, its injected Cholesky factor is `2I`, and
`A=I`, `particles=2R`. The `d=1` certificate uses the first coordinate. Thus
the accepted fixtures cover two dimensions and two prepared positive ridge
values without selecting either from results.

The exact primary `d=2` input tangents, particle cotangent, all five expected JVP
arrays, all five expected VJP arrays, selected nonzero intermediate tangents,
selected nonzero internal reverse cotangents, the dense matrix tangent/adjoint,
and exact duality scalar are frozen before implementation in
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase3-exact-certificate-2026-07-13.json`.
Every entry is an integer or rational string with a power-of-two denominator.
The source and weight tangents induce off-diagonal covariance, Cholesky, solve,
and affine tangents. The transport, residual-design, and ridge tangents each
induce nonzero injected-covariance, injected-Cholesky, affine, and output
tangents. All five reverse blocks are nonzero under the frozen nonconstant
particle cotangent. Products and reductions on this small certificate are
audited to stay within the exact binary64 integer-significand range.

Certificate acceptance requires bitwise equality, including signed-zero
normalization to positive zero before byte comparison, for:

- every shared cloud/dense forward primary and numeric intermediate;
- manual cloud JVP versus TensorFlow `ForwardAccumulator` for each of the five
  input tangents;
- manual cloud VJP versus TensorFlow `GradientTape` for each of the five input
  adjoints;
- cloud VJP/JVP composed through the dense matrix/noise preparation maps versus
  the dense helper; and
- the two scalar Frobenius pairings in JVP/VJP duality.

For dense-helper composition, the matrix is exactly zero. Let the helper's
executed binary64 value be `s=sqrt(8/7)` and set its base residual noise to
`R/s`; the Phase 3 precheck must prove that the executed `s*center(R/s)` equals
`R` bitwise before the dense reference is admitted for the certificate. The
frozen residual-design tangent is `R/16`; use helper noise tangent `(R/16)/s`
and require the same executed bitwise replay. These are execution-identity
checks, not claims that `s` is dyadic. The dense noise VJP is compared to the
separately composed executed expression
`s*center(cloud_residual_design_bar)` using the same TensorFlow
dtype/operations; it is not included in the symbolic dyadic-rational claim. If
any replay or composed noise comparison is not bitwise, dense noise composition
is `INCONCLUSIVE` and cannot pass the certificate.

A second nonzero-`Xi`, non-diagonal chart is fully frozen in the certificate as
an explanatory diagnostic, including its source, weights, `Y+`, `Xi`, ridge,
all five tangents, and particle cotangent. It reports manual/autodiff/dense
differences, defining-equation residuals, condition estimates, and ULP
distances, but it has no pass/fail tolerance. A nonfinite result or invalid
Cholesky chart is a hard implementation veto; otherwise its numerical agreement status is
`INCONCLUSIVE_GENERAL_CHART_NO_JUSTIFIED_FORWARD_ERROR_BOUND`. This is stated
plainly rather than converting backward residuals into an unsupported
componentwise forward-error claim.

The exact certificate establishes a bounded algebraic implementation case. It
does not establish general-chart kernel accuracy. The unresolved kernel
backward-error, conditioning/downstream, ridge-bias, and ridge-domain
requirements from Phase 1 remain promotion blockers.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the owned cloud primitive implement the Phase 1 fixed-ridge Contract E equations and complete local derivatives without a dense transport matrix? |
| Exact comparator | Fixed-ridge dense helper at `rho=1`, composed through realized `Y+` and `Xi`; TensorFlow float64 autodiff; JVP/VJP duality. |
| Primary engineering criterion | The frozen dyadic certificate passes bitwise forward/JVP/VJP/dense-composition/duality equality for all five inputs; direct source/weight and transported paths are separately nonzero; the nontrivial chart is finite and valid but explicitly inconclusive for numerical agreement; public wrappers are XLA-on. |
| Hard vetoes | Nonfinite output/adjoint, failed Cholesky chart, wrong population covariance, hidden ridge adaptation, `rho` input, dense `N*N` input/allocation, omitted input adjoint, direct path labeled total, or reference/helper mutation. |
| Promotion blockers | Unjustified residual-centering, mean-restoration, ridged-identity kernel, raw ridge-bias, conditioning/downstream-error, or ridge-domain adequacy requirement. |
| Repair triggers | Forward/JVP/VJP mismatch, failed duality, missing diagnostic, XLA compile failure, dtype/shape drift, or review finding. |
| Continuation vetoes | Normative equations are inconsistent; valid test charts cannot be constructed without adaptive ridge; implementation requires a target change; concurrent in-scope edit; five material review rounds do not converge; or campaign budget exhaustion. |
| Explanatory only | Exact residual magnitudes, condition estimates, runtimes, and ULP differences. |
| Not concluded | Full transport/filter total gradient, production numerical adequacy, GPU feasibility, same-scalar FD, Kalman equivalence, nonlinear validity, HMC readiness, or admission. |

## Required Artifacts

- Owned TensorFlow module and focused tests.
- Frozen preimplementation exact certificate:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase3-exact-certificate-2026-07-13.json`.
- Machine-readable Phase 3 parity/diagnostic artifact containing exact fixture,
  input/output hashes, per-field parity values, path decomposition, finite/chart
  status, XLA status, and all preserved promotion blockers.
- Phase 3 run manifest and focused log under
  `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase3/`.
- Phase 3 result or blocker result.
- Phase 4 streaming-composition subplan.
- Updated master status, ledger, and stop handoff.

## Required Checks, Tests, And Reviews

1. Recheck status/hash of the dense helper and normative spec before editing.
2. Test batched and unbatched-shaped fixtures with `N>1`, multiple `d`, and at
   least two valid prepared ridge values declared before evaluating outputs.
3. Parse, validate, and hash the frozen certificate; construct exactly its
   `N=8`, `d=1/2`, two-ridge fixtures; audit exact-representability of every
   symbolic certificate operation; require bitwise equality for all shared
   forward fields and exact defining identities.
4. Compare manual VJP with TensorFlow autodiff for each input independently and
   compare the composed cloud VJP with every dense-helper VJP input.
5. Compare manual JVP with `ForwardAccumulator`, composed dense-helper JVP, and
   JVP/VJP Frobenius duality bitwise on the certificate. Run the predeclared
   nonzero-`Xi` chart only as a finite/valid explanatory diagnostic with no
   agreement threshold.
6. Prove direct source/weight and transported-cloud paths are separately
   nonzero on the declared discriminating fixture.
7. Source-audit absence of NumPy, autodiff in the owned module, adaptive ridge,
   `rho`, dense matrix input, `N*N` construction, inverse, and raw-route calls.
8. Check public wrapper `jit_compile=True`; execute one deliberate CPU-XLA tiny
   smoke and separately run eager CPU float64 reference tests.
9. Re-run Phase 1 math and Phase 2 schema/factory compatibility suites.
10. Run Python compile, JSON validation, source/hash checks, and scoped
    `git diff --check`.
11. Obtain bounded fresh-Codex review of mathematical fidelity, derivative path
    separation, XLA/default boundary, tests, result, and Phase 4 handoff. Repair
    and repeat for material findings, at most five rounds per blocker.

## Forbidden Claims And Actions

- Do not import or call the benchmark helper from the owned production module.
- Do not add NumPy, PyTorch, JAX, eigendecomposition, explicit inverse,
  `GradientTape`, `ForwardAccumulator`, adaptive ridge, clipping, or a dense
  transport matrix to the owned module.
- Do not reconstruct or regenerate `Xi` inside the primitive.
- Do not silently renormalize weights or add denominator/ridge floors.
- Do not merge direct probability-weight and transport log-weight coordinates.
- Do not register a placeholder value/gradient route or issue a v2 artifact.
- Do not infer an adequacy threshold from observed residuals or parity errors.
- Do not edit the dirty dense helper or model-specific harnesses.
- Do not run GPU, full filter, HMC, nonlinear, leaderboard, or long commands.

## Exact Next-Phase Handoff Conditions

Phase 4 engineering work may begin only if:

- the cloud forward/JVP/VJP symbols exist in the owned module and public
  wrappers default to XLA;
- the forward implements exactly the Phase 1 fixed-`rho=1`, fixed-`Xi`,
  fixed-ridge program;
- all five input derivatives pass the frozen binary-exact certificate bitwise,
  and the nontrivial diagnostic chart is finite/valid while clearly labeled
  inconclusive for general numerical parity;
- direct source/weight and transported-cloud adjoints are exposed separately;
- composed cloud VJP/JVP agrees bitwise with the dense reference on the exact
  certificate; no general-chart agreement claim is made;
- no dense `N*N`, autodiff, adaptive-ridge, raw-route, or NumPy production path
  exists;
- v1 revocation and Phase 2 factory isolation remain intact;
- the production route registry remains inert;
- the result distinguishes engineering parity from blocked promotion and
  preserves every unresolved numerical adequacy requirement;
- artifacts, hashes, run manifest, result, and review trail are complete; and
- the Phase 4 subplan is reviewed and names the exact streaming numerator/mass
  quotient APIs and total-coordinate composition.

The Phase 3 production-reset evidence gate may be marked passed only if the six
promotion blockers have independently justified pre-result requirements and
all pass. Otherwise the result must use
`EXACT_ENGINEERING_CERTIFICATE_PASSED_GENERAL_PARITY_AND_PROMOTION_BLOCKED`
while allowing Phase 4's planned engineering repair/composition work to
continue.

## Stop Conditions

Stop and write a blocker result if the normative target cannot be implemented
without changing semantics, no fixed ridge yields a valid predeclared tiny
chart, a required derivative path cannot be reconciled with independent
autodiff/dense reference, a target change or new scientific threshold requires
owner authority, an in-scope concurrent edit appears, five material review
rounds fail to converge, or the eight-hour campaign budget expires. Ordinary
parity/test failures are repair triggers first.

## Phase-End Protocol

1. Run focused CPU-hidden eager and CPU-XLA checks.
2. Write the parity artifact, run manifest, and result/blocker with decision and
   inference-status tables.
3. Draft the Phase 4 streaming-composition and feasibility subplan.
4. Review the Phase 3 result and Phase 4 handoff for mathematics, path
   separation, feasibility, artifact coverage, and boundary safety.
5. Patch visible material findings and rerun focused checks, up to five rounds.
6. Update master status, ledger, and stop handoff.
7. Advance on engineering handoff only; do not relabel blocked promotion as a
   pass.
