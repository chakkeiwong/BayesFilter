# Phase 4 Subplan: Streaming Quotient And Contract E Composition

Date: 2026-07-13

Status: `REVIEWED_ACTIVE`

Master program:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-master-program-2026-07-13.md`

## Phase Objective

Repair the streaming finite transport so the canonical cloud is the row
quotient `Y=Q/M`, propagate numerator and mass through its analytic JVP/VJP,
and compose the resulting transported-cloud path with the Phase 3 Contract
E-Chol direct source/weight paths. Establish small-fixture dense/stream
engineering parity and then assess production feasibility without allocating or
retaining an `N x N` transport tensor.

Phase 4 does not register a route or claim full-filter, Kalman, HMC, nonlinear,
leaderboard, or production-reset correctness. The Phase 5 one-graph callable
still owns corrected-logit normalization, candidate-dependent initialization,
likelihood increments, and time composition.

## Entry Conditions Inherited From Phase 3

- Contract E-Chol is the only reset eligible to seek canonical admission.
- The Phase 1 normative equations and weight-coordinate rules are binding.
- The Phase 3 cloud forward/JVP/VJP pass bounded exact engineering
  certificates, including noncommuting and transported-covariance branches.
- Phase 3 status is
  `EXACT_ENGINEERING_CERTIFICATE_PASSED_GENERAL_PARITY_AND_PROMOTION_BLOCKED`.
- Direct source-particle and probability-weight adjoints remain separate from
  the transported-cloud adjoint.
- The six Phase 3 promotion blockers remain unresolved and cannot be discharged
  by streaming parity.
- V1/raw routes remain historical; the production factory remains empty.
- The platform-blocked Claude route is not retried; fresh bounded Codex review
  is the substitute.
- `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
  was clean with SHA256
  `9f200ca8f85e5aeb97a4dca1b46663b60f626837ec29e940d83820bf40b16abe`
  at plan drafting. Recheck immediately before any edit or import-based hash
  claim.
- `tests/test_ledh_compact_transport_jvp.py` is dirty from another lane. Do not
  edit or rely on its unreviewed changes; add Phase 4 tests under
  `tests/highdim/`.

## Root Cause And Minimal Repair Design

The current streaming forward accumulates

```text
Q_i = sum_j P_ij X_j
M_i = sum_j P_ij
```

but returns `Q` as `transported` and only a scalar residual derived from `M`.
Its JVP accumulates `dQ` but not `dM`; its VJP accepts only `barQ`. Therefore the
existing API cannot compute the canonical quotient derivative.

The minimal no-dense repair does not duplicate the streaming kernel. In a new
owned integration module, append a constant-one feature to the streamed
particle payload:

```text
augmented_particles = concat([particles, ones], axis=-1)
```

One call to the existing streaming helper then returns
`augmented_numerator=[Q,M]`. For JVP, append a zero tangent feature so the same
analytic streaming JVP returns `[dQ,dM]`. For VJP, convert the quotient output
cotangent to `[barQ,barM]` and pass that augmented cotangent through the same
analytic streaming VJP. This preserves streaming `O(N*d + N*chunk)` live state
and avoids an `N x N` output or a second transport pass.

The existing VJP is not yet generic in payload width: it derives `state_dim`
from the geometry `scaled_x` and incorrectly reuses that width for the particle-
payload adjoint allocation and reshape. Before augmented composition, make one
narrow backward-compatible repair in the clean streaming module:

```text
geometry_dim = shape(scaled_x)[2]
payload_dim = shape(particles)[2]
```

Keep `geometry_dim` for query/key cost cotangents and their reshapes. Use
`payload_dim` only for the particle-payload cotangent allocation and reshape.
The quotient wrapper discards the returned adjoint of the appended constant-one
payload after checking its shape/finite status; it is not an input derivative.
Add an unequal-width `d` versus `d+1` regression comparing payload adjoint shape,
geometry cotangents, and log-weight cotangents with TensorFlow autodiff. Recheck
and record the clean source hash immediately before this edit.

The exact quotient equations are binding:

```text
Y_i = Q_i / M_i
dY_i = (dQ_i - Y_i dM_i) / M_i
barQ_i = barY_i / M_i
barM_i = -sum_k(barY_ik Y_ik) / M_i
```

Every `M_i` must be finite and strictly positive. There is no floor, clipping,
or stopped mass. The forward returns `Q`, `M`, `Y`, row residuals, minimum mass,
and finite/positive status. A failed chart is a hard veto.

## Exact Proposed Symbols

Create `bayesfilter/highdim/ledh_contract_e_streaming_tf.py` with XLA-on public
wrappers and private cores:

- `_streaming_row_quotient_forward_core`;
- `_streaming_row_quotient_jvp_core`;
- `_streaming_row_quotient_vjp_core`;
- `_contract_e_streaming_forward_core`;
- `_contract_e_streaming_jvp_core`;
- `_contract_e_streaming_vjp_core`;
- `contract_e_streaming_forward_tf`;
- `contract_e_streaming_jvp_tf`;
- `contract_e_streaming_vjp_tf`.

The quotient cores accept the exact finite transport inputs used by
`_filterflow_manual_streaming_finite_transport_value_total_vjp`, plus tangents
or `barY` as appropriate. They may call the current clean streaming value/JVP/
pullback helpers, but must not call a historical stopped-scale/key route.

The Contract E composition cores additionally accept normalized probabilities,
fixed residual design, prepared ridge, their declared tangents where applicable,
and an output-particle cotangent for VJP. They compose only the already-defined
streaming finite transport and cloud reset; they do not own the outer likelihood
or filter time loop.

If import layering from `bayesfilter/` to the existing experiment module is
judged architecturally unacceptable, the alternative is a narrow additive
extension of the clean streaming module that exposes an augmented-payload
generic API. Do not copy the whole kernel or change existing historical output
semantics silently. Record and review any deviation before implementation.

The generic-payload VJP dimension split above is required regardless of which
integration placement is selected; it is a known compatibility repair, not an
optional fallback.

## Coordinate-Safe Total Composition

For Phase 4 VJP, let the cloud reset return

```text
G_X_direct
G_w_probability
G_Y
```

The streaming quotient pullback of `G_Y` returns

```text
G_X_transport
G_logw_transport
G_scaled_x_transport
G_epsilon0_transport
```

Then expose separately:

```text
G_X_total_at_composition = G_X_direct + G_X_transport
G_logw_mom = normalized_weights * G_w_probability
G_logw_total = G_logw_transport + G_logw_mom
```

`G_logw_total` is a cotangent of exact normalized log weights. It is not yet the
cotangent of pre-normalization corrected logits. Phase 5 must apply the unique
normalization pullback once:

```text
G_a = G_delta * w + G_logw_total - w * sum(G_logw_total)
```

Equivalently, the direct-moment-only corrected-logit contribution is
`w*(G_w_probability-sum(w*G_w_probability))`. A probability-coordinate
cotangent must never be added directly to a log-weight cotangent, and the
simplex projection must not be applied twice.

The Phase 4 JVP consumes normalized-weight and normalized-log-weight tangents as
separate inputs. The caller must ensure `dw=w*dlogw` on tangent vectors that are
valid for the declared normalized-log-weight chart. Phase 4 tests include both
independent local-coordinate checks and consistent composed-coordinate checks.

## Dense Comparator

On small `N`, construct the same finite dense Sinkhorn transport matrix `P` from
the same potentials, schedule, weights, particles, dtype, and fixed inputs.
Compute `Q=P@X`, `M=P@1`, `Y=Q/M`, then apply the Phase 3 cloud reset.

Compare:

- streaming versus dense `Q`, `M`, `Y`, and Contract E output;
- quotient JVP against TensorFlow autodiff and a direct dense JVP;
- quotient VJP against TensorFlow autodiff and a direct dense VJP;
- full local composition JVP/VJP against autodiff of the same finite scalar;
- JVP/VJP duality;
- direct source, probability-weight, transported source, normalized-log-weight,
  residual-design, ridge, and epsilon-initialization paths separately; and
- chunk tilings with identical inputs.

No observed error becomes a threshold. Freeze an exact or independently
justified small-chart acceptance rule before evaluating outputs. If a binary
exact chart cannot cover the finite Sinkhorn exponentials, use executed-reference
identity checks for the quotient itself and leave general dense/stream agreement
descriptive until a defensible kernel error bound exists.

## Skeptical Plan Audit

Decision: `PASS_FOR_REVIEWED_LOCAL_IMPLEMENTATION; PRODUCTION_PROMOTION_BLOCKED`.

| Risk | Finding and control |
| --- | --- |
| Wrong baseline | Dense and streaming comparators execute the same finite Sinkhorn schedule and quotient; Kalman and raw routes are not Phase 4 baselines. |
| Proxy promotion | Dense/stream parity proves composition engineering only; it cannot discharge ridge bias, Sinkhorn adequacy, or downstream scientific error. |
| Hidden target change | The quotient has no floor and uses the already-computed finite plan. Constant feature augmentation is algebraically the same `P@1`. |
| Missing derivative | Augmented JVP/VJP carries both mass terms. Tests require nonzero `dM` and `barM` on a discriminating fixture. |
| Geometry/payload width conflation | Narrowly split geometry and payload dimensions in the existing clean VJP; retain geometry width for cost paths and payload width for particle-adjoint paths; test `d` versus `d+1`. |
| Coordinate mismatch | Probability, normalized-log-weight, and corrected-logit cotangents stay named and separate; projection occurs once in Phase 5. |
| Dense leakage | Dense `P` exists only in tiny tests; source/graph audit forbids it in the owned integration path. |
| Nominal streaming | Source complexity, graph intermediates, chunk behavior, and measured trusted-GPU peak memory are all required before a feasibility statement. |
| Arbitrary thresholds | Unjustified row/Sinkhorn/chunk/ridge requirements remain blockers; observed values do not create gates. |
| Environment mismatch | CPU float64 is reference only. Production-shape evidence requires trusted GPU, XLA, float32/TF32 provenance. |
| Dirty overlap | Existing dirty compact JVP tests are read-only. Recheck the clean streaming source hash before any edit. |
| Candidate versus direction failure | A local quotient/composition defect triggers repair. It rejects the direction only if the same finite target cannot be implemented streaming without dense state. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the finite streaming route compute `Y=Q/M` and the complete local Contract E direct-plus-transport derivative without dense production state? |
| Exact comparator | Same finite dense Sinkhorn plan and quotient on small fixtures; TensorFlow autodiff; JVP/VJP duality. |
| Primary engineering criterion | Frozen quotient identity checks and small-fixture forward/JVP/VJP composition checks pass; both mass derivative paths are nonzero; no dense production path exists. |
| Hard vetoes | Any nonfinite/nonpositive mass, floor/clipping, omitted `dM`/`barM`, probability/log-weight coordinate mixing, dense production allocation, wrong scalar, XLA failure, or source drift. |
| Promotion blockers | Row-mass adequacy, row/column finite-Sinkhorn convergence, chunk accumulation error, the six Phase 3 blockers, and any missing production feasibility evidence. |
| Repair triggers | Local parity, duality, chunk, XLA, memory, or review failure with a valid target/harness. |
| Explanatory only | Residual magnitudes, condition proxies, runtimes, memory, compile time, and descriptive chunk differences absent a justified error budget. |
| Not concluded | Full-filter same-scalar gradient, Kalman equivalence, nonlinear validity, HMC readiness, route admission, leaderboard or release readiness. |

## Required Artifacts

- Owned streaming quotient/composition module and focused tests.
- Frozen small-fixture acceptance/certificate artifact before output evaluation.
- Machine-readable forward/JVP/VJP, path-decomposition, mass, chunk, and
  feasibility artifact.
- Analytic complexity and graph/intermediate audit.
- Trusted GPU manifest/log for any feasibility run.
- Phase 4 result or blocker result with decision and inference-status tables.
- Phase 5 canonical one-graph subplan.
- Updated master, ledger, and stop handoff.

## Required Checks, Tests, And Reviews

1. Recheck source/status/hash of the clean streaming module and all imported
   symbols immediately before implementation.
2. Freeze a discriminating small fixture with nonunit row masses, nonzero mass
   tangent and adjoint, multiple dimensions, `B>1`, and at least two chunk
   tilings before reading repair outputs.
3. Test standalone quotient formulas exactly, including duality and no-floor
   invalid-mass failure.
4. Compare augmented-stream `Q,M,Y` with a tiny dense plan and TensorFlow
   autodiff for all transport inputs.
5. Exercise the repaired generic-payload VJP with geometry width `d` and payload
   width `d+1`; compare payload-adjoint shape plus geometry, potential, and
   log-weight cotangents with autodiff, and slice/discard only the appended
   constant-feature adjoint at the quotient boundary.
6. Compare composed Contract E forward/JVP/VJP against the same tiny dense
   finite program and autodiff, keeping direct and transport paths separate.
7. Test probability-to-normalized-log-weight conversion, additive-constant
   invariance after the Phase 5 projection formula, and prevention of direct
   probability/log-weight addition.
8. Source-audit no NumPy, tape/ForwardAccumulator in production code, dense
   matrix construction, denominator floor, stopped mass, raw reset, or
   historical stopped-scale/key route.
9. Run deliberate CPU-XLA tiny wrappers and Phase 0-3 compatibility suites.
10. Inspect XLA graphs/HLO or compiler artifacts for absence of retained
   `N x N` state; record analytic live-state complexity.
11. Only after local checks pass, run escalated/trusted GPU probes and a bounded
    production-shape preflight with at most two attempts, explicit timeout,
    structured artifact, XLA/TF32/device provenance, peak memory, compile time,
    runtime, and chunk tilings.
12. Run Python compilation, JSON/hash checks, and scoped `git diff --check`.
13. Obtain bounded fresh-Codex review after local parity and again before a
    production-feasibility conclusion when material. Repair up to five rounds
    per blocker.

## Forbidden Claims And Actions

- Do not call `Q` or its derivative the quotient cloud.
- Do not omit, stop, floor, clip, or reconstruct `M`, `dM`, or `barM`.
- Do not add probability-coordinate and log-weight-coordinate adjoints directly.
- Do not apply the normalized-log-weight projection twice.
- Do not silently change existing historical helper return semantics or callers.
- Do not allocate `N x N` in production code or use dense feasibility as a
  production proxy.
- Do not infer Sinkhorn, chunk, row-mass, ridge, conditioning, or memory
  thresholds from observed runs.
- Do not edit dirty unrelated tests/model harnesses or the dirty dense Contract
  E helper.
- Do not register a route, issue an admitted artifact, run HMC/full-filter/
  nonlinear/leaderboard work, or claim production readiness.

## Exact Next-Phase Handoff Conditions

Phase 5 may begin engineering work only if:

- `Q,M,Y` and `dQ,dM,dY` are explicit, finite, and unfloored on valid charts;
- VJP uses both `barQ` and nonzero `barM` and passes duality/autodiff checks;
- small dense/stream Contract E forward/JVP/VJP composition passes its frozen
  engineering certificate or is plainly classified inconclusive where no
  justified general bound exists;
- direct source/probability-weight and streaming source/normalized-log-weight
  paths remain separately observable and are combined only after the stated
  coordinate conversion;
- no dense production allocation, raw route, stopped mass, or duplicate
  normalization projection exists;
- CPU-XLA local checks and Phase 0-3 compatibility pass;
- production feasibility is either supported by the exact required trusted-GPU
  artifacts or explicitly blocked without preventing Phase 5 local graph work;
- every unresolved numerical/scientific blocker is preserved; and
- the Phase 4 result and Phase 5 subplan pass bounded handoff review.

The Phase 4 production feasibility/promotion gate cannot pass while row-mass,
finite-Sinkhorn, chunk-accumulation, or inherited Phase 3 adequacy requirements
lack justified pre-result criteria. Engineering continuation may proceed under
an explicitly blocked promotion status.

## Stop Conditions

Stop and write a blocker result if the exact finite target requires dense
production state, the quotient cannot be differentiated without changing the
target, mass is invalid on every predeclared fixture, a concurrent in-scope edit
appears, a new scientific threshold or target decision requires owner authority,
five material repair rounds fail for the same blocker, trusted GPU execution
cannot be interpreted under policy, or the campaign budget expires. Local
parity, compile, and feasibility failures are repair triggers first.

## Phase-End Protocol

1. Run local CPU-hidden quotient and composition checks.
2. Write structured parity/diagnostic evidence and a same-phase repair note when
   needed.
3. Review the local implementation before any production-shape run.
4. Run only the predeclared trusted-GPU preflight when local gates pass.
5. Write the result/blocker, manifest, logs, and post-run red team.
6. Draft Phase 5 and review the result/handoff.
7. Update master, ledger, and stop handoff.
8. Advance on engineering evidence only; do not relabel blocked promotion.
