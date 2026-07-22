# P6 Subplan: Parameterized Spatial SIR

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `DRAFT_REQUIRES_P5_REFRESH`

## Phase Objective

Close the observed-data posterior gaps for `SIR-SGQF`, `SIR-UKF`, and `SIR-ZC`
before any NeuTra training. Build or admit graph-native batched value/score
routes for the frozen parameterized spatial SIR target, distinguish full
observed-data filtering from local complete-data and scout diagnostics, then run
the standard same-target plain-HMC, target-specific training, and NeuTra
confirmation ladder independently for every admitted cell.

## Inherited Entry Conditions

- P0/P1 registry and harness are admitted; P5 result is closed.
- P0 freezes SIR observations/data version, reporting/observation model,
  transition convention, parameter chart/prior/support, initial law, population
  invariants, three filter definitions, signatures, comparator estimands,
  uncertainty/equivalence rules, and route ownership.
- `ParameterizedZhaoCuiSIRSSM` and its local complete-data density are inventory
  anchors, not observed-data posterior admission.
- Existing UKF is scout evidence only; no SGQF parameter posterior is presumed.
- Zhao-Cui work requires paper/math and author-source anchors and the fixed
  production-admissible source-route direction, not the generic retained grid.

## Target And Cell Scope

- `SIR-SGQF`: fixed SGQF observed-data filter posterior to be constructed and
  admitted.
- `SIR-UKF`: structural UKF observed-data filter posterior to be constructed and
  admitted; current scout evidence is explanatory.
- `SIR-ZC`: fixed Zhao-Cui TTSIRT observed-data filter posterior; local
  complete-data density may support internal checks but cannot define the final
  likelihood.

## Required Artifacts

- A derivation/target note separating transition/complete-data terms from
  integrated observed-data likelihood for all three cells.
- Graph-native batched TF/XLA posterior adapters with prior/chart/data/filter
  identity and status telemetry.
- SIR support/population/conservation, value/score, dense/focused reference,
  batch, branch, deterministic replay, and score-check artifacts.
- Zhao-Cui paper/author-source anchors and operation classification.
- Per admitted cell: tuned same-target plain-HMC, training/default audit, recipe
  screen, selected fresh 5,000-step GPU/XLA training, frozen transport, NeuTra
  confirmation, archives, agreement, repairs, and manifests.
- P6 result, cell/budget/hash ledgers, and refreshed P7 subplan.

## Required Checks And Reviews

1. Write the claimed posterior and quantity actually computed for every route.
   Prove equality or state the exact difference. Local complete-data density is
   wrong relative to a claim of full observed-data posterior if latent states
   have not been integrated under the frozen filter.
2. Bind prior/Jacobian, observations, initial law, transition, reporting model,
   filter update, and status into one registered scalar/value-score program.
3. Test epidemiological support and population invariants over the full frozen
   parameter region, batch, time, and filter branches.
4. Use dense/enumerated/focused reference ladders at feasible scale and
   same-branch score checks; freeze extrapolation/nonclaims for larger scope.
5. For `SIR-ZC`, cite paper and author source, classify fixed-HMC adaptations,
   and reject generic retained-grid or unanchored faithful claims.
6. Conduct a material bounded review of the observed-data derivation and route
   classifications before R2. A purely procedural reviewer issue does not
   block; a target/math/source contradiction does.
7. Pass R1B independently per cell, recomposing prior, full observed-data
   filter likelihood, complete chart Jacobian, and total unconstrained
   value/score; local complete-data and scout substitutions must fail.
8. Execute R2-R4 only for cells reaching `POSTERIOR_IDENTITY_ADMITTED`. Preserve
   tried/selected/rejected/untried candidate-family ledgers. Do not train a
   transport for a local-density/scout target while retaining the full-posterior
   cell ID.

## Evidence Contract

| Field | P6 contract |
| --- | --- |
| Question | Can full observed-data SIR filter posteriors be admitted and, if so, can target-specific NeuTra sample each consistently with identical-target plain HMC? |
| Comparator | Separate tuned plain HMC on each admitted full observed-data filter posterior |
| Filter reference | Dense/focused observed-data likelihood/value/score and SIR invariants; local complete-data density is an internal reference only |
| Primary pass | Equality of claimed/computed posterior target, R1 filter admission, final training validity, modern HMC diagnostics/health, and simultaneous comparator agreement per cell |
| Hard vetoes | Local/complete-data or scout route mislabeled full posterior; support/population violation; missing source anchor; generic retained-grid promotion; value/score/target mismatch; HMC health/convergence/agreement failure |
| Explanatory only | Scout results, local-density checks, loss, acceptance, runtime, truth distance, cross-filter gaps |
| Not concluded | Epidemiological calibration, filter exactness/ranking, forecasting validity, broad SIR robustness, production readiness |

## Default And Assumption Audit

Audit observed/reporting data, missingness, population scaling, time step,
initial infection law, process/observation model, inferred/fixed parameters,
support transforms, priors, transition discretization, filter ranks/design,
quadrature/sigma points, fixed randomness, parameter region, affine chart,
topology/optimizer grid, heldout construction, HMC grid, seeds, and margins.
Austria/test-fixture constants and Zhao-Cui training settings are provenance,
not target-specific defaults until checked.

## Repair Triggers

- Missing observed-data likelihood: implement/derive the smallest full filter
  route and replay R0/R1; do not bypass with complete-data sampling.
- Support/invariant/value-score failure: repair the owning model/filter route.
- Zhao-Cui anchor/route mismatch: correct/classify or block `SIR-ZC` only.
- Valid filter fails approximation gate: reject that filter candidate or execute
  a predeclared filter repair; do not use NeuTra as compensation.
- Failed training recipe: record `RECIPE_REJECTED`; reject the cell only after
  complete candidate-family accounting under the runbook rule.
- Standard infrastructure, training, sampler, and evidence repairs use fresh
  attempts and stay cell-local.

## Forbidden Claims And Actions

- No local complete-data density or UKF scout relabeled observed-data posterior.
- No training before `COMPARATOR_ADMITTED` and no plain-HMC comparator before
  `POSTERIOR_IDENTITY_ADMITTED`.
- No generic retained-grid Zhao-Cui production route or source self-attestation.
- No post-result target/filter/settings/margin changes, active NumPy/host
  callback/sample loop, CPU serious training, archive pooling, or overwrite.

## Handoff Conditions

P7 begins when every SIR cell has an honest terminal state, all admitted routes
have full observed-data target evidence, P6 result/manifest/repairs/ledgers are
complete, and P7 is refreshed with the actual entire-program cell table,
artifact inventory, missing evidence, and claims. A blocked SIR cell does not
prevent P7 synthesis.

## Stop Conditions

Block a cell for an unresolved observed-data target, invalid support/reference,
missing source, absent comparator, three identical failed repairs, or exhausted
cell budget. Stop P6 program-wide only if the shared target/harness is invalid,
common SIR data are corrupted, the required scientific target changes, or the
54 GPU-hour phase ceiling is exhausted before honest classification.

## Compute And Attempt Budget

Aggregate ceiling: 120 trusted GPU wall-hours plus 32 CPU development/reference
hours. NeuTra budget is consumed only after filter admission. Each admitted
cell reserves two 15-GPU-hour family arms: plain dense IAF and one P0-frozen
target-specific enhanced family. Each arm permits one screen, one selected fresh
5,000-step training, one NeuTra confirmation, and arm-local retries. A separate
6-hour bucket funds plain HMC and comparator retries; 4 hours fund trusted
R0/R1/R1B cell admission, cell-specific adapter/artifact emission, and their
repairs. Common harness/schema/reporting defects reopen and charge P1 only.
Three localized repairs apply per identical failure within the owning bucket.

## Skeptical Pre-Execution Audit

The current evidence does not establish the three full observed-data
posteriors. This phase makes target derivation and value/score admission the
first result, with training conditional on success. It therefore cannot report
a false NeuTra success by sampling a convenient local or scout scalar.
