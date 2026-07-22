# P5 Subplan: Chapter 18b Structural Quadratic Model

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `P4_CLOSED_TARGET_DESIGN_REVIEW_IN_PROGRESS`

P4 closed with `PP-UKF` and `PP-SGQF` confirmed only at the six-physical-mean
level and `PP-ZC` source-route blocked. No P4 target or transport is transferable
to P5. The original inherited-entry statement below is stale: P0 did not freeze the
structural dataset, inferred parameter subset, prior/chart, posterior target, or
agreement margin. It explicitly recorded those items as blockers. The first P5
rung is therefore the dedicated target-design subplan
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p5-structural-target-design-subplan-2026-07-16.md`.
No P5 HMC or training may begin from this parent draft.

## Phase Objective

Promote the Chapter 18b worked structural model from a NumPy reference fixture
to a graph-native TensorFlow parameter posterior, admit `STR-UKF` under the
deterministic structural identity, construct a deliberately naive artificial-
noise full-state UKF negative control, classify and admit `STR-ZC` only as an
`extension_or_invention`, and then test target-specific NeuTra independently for
the two eligible filter posteriors.

## Inherited Entry Conditions

- P0/P1 registry and harness are admitted; P4 result is closed.
- Chapter 18b equations and structural interpretation are the mathematical
  target. The current fixture/test are reference evidence only.
- P0 records data, prior, chart, inferred subset, target identity, and margins
  as unresolved blockers. P5 must close them prospectively before target
  issuance or serious sampling.
- The intended UKF represents new stochastic input only in `epsilon_t`; it does
  not inject process noise into deterministic `k_t`.

## Target And Cell Scope

```text
m_t = rho m_(t-1) + sigma epsilon_t
k_t = phi k_(t-1) + gamma m_t^2
y_t = m_t + k_t + e_t
```

- `STR-UKF`: structural UKF posterior respecting the deterministic coordinate.
- `STR-ZC`: Zhao-Cui fixed-route application explicitly classified
  `extension_or_invention`; Zhao-Cui literature can motivate machinery but
  cannot make this a source-faithful reproduction.
- `STR-NAIVE-UKF-NEGATIVE`: diagnostic-only full-state UKF with artificial
  noise in `k_t`; permanently ineligible for posterior/NeuTra admission.

## Required Artifacts

- Mathematical target/parameterization note and graph-native TensorFlow model,
  simulator/data record, prior, posterior adapter, and target signatures.
- Dense/reference likelihood/value/score at feasible sizes.
- Structural UKF value/score and identity artifacts over points, times, batches,
  and parameter regions.
- Naive negative-control implementation/artifact with a distinct diagnostic
  signature proving the identity/noise veto detects it.
- Zhao-Cui extension design/classification, value/score, reference, branch, and
  status artifacts.
- Per eligible cell: plain-HMC, target-specific training screen, selected fresh
  5,000-step training, frozen transport, NeuTra confirmation, separate archives,
  agreement, repairs, and manifests.
- P5 result, ledgers, and refreshed P6 subplan.

## Required Checks And Reviews

1. Derive the graph-native transition and posterior in project notation; show
   the computed likelihood target and its relation to the Chapter 18b equations.
2. For every structural point and time, require within P0-frozen dtype/scale
   tolerance:

   ```text
   k_t - phi k_(t-1) - gamma m_t^2 = 0.
   ```

3. Prove no covariance/noise parameter is introduced for `k_t` in `STR-UKF`.
   The naive negative control must fail this intended-route gate and cannot
   share its signature.
4. Compare TensorFlow simulator/value/score to the current NumPy worked fixture
   and independent closed-form/dense reference in their feasible scope. NumPy
   remains reference-only.
5. Run batch, permutation, replay, score, support, SPD/branch/status, and trusted
   GPU/XLA checks.
6. Review the structural mathematics and negative-control design materially
   before serious sampling. Review cannot authorize crossing a scientific
   target boundary.
7. Pass R1B independently per eligible cell, recomposing prior, filter
   likelihood, complete chart Jacobian, and total unconstrained value/score;
   negative substitutions include the naive-control signature/noise route.
8. Execute R2-R4 separately for `STR-UKF` and `STR-ZC` only after R1B admission,
   with tried/selected/rejected/untried candidate-family ledgers.

## Evidence Contract

| Field | P5 contract |
| --- | --- |
| Question | Can NeuTra sample the intended structural UKF posterior and the separately labeled Zhao-Cui extension posterior while preserving the deterministic state geometry? |
| Comparator | Separate same-target tuned plain HMC for each eligible signature |
| Structural reference | Chapter 18b equations, independent dense/closed-form feasible checks, and exact pointwise identity |
| Primary pass | Identity/no-artificial-noise gates, filter admission, training validity, modern HMC diagnostics/health, and simultaneous comparator agreement all pass per cell |
| Hard vetoes | Any intended-route identity violation; artificial `k_t` noise; naive-control signature collision/admission; NumPy in active graph; extension mislabeled source-faithful; value/score/support/health/convergence/agreement failure |
| Explanatory only | Negative-control numerical differences, loss, acceptance, runtime, truth distance, cross-filter gaps |
| Not concluded | Zhao-Cui source reproduction, filter exactness/ranking, broad structural validity/calibration, production readiness |

## Default And Assumption Audit

Audit inferred/fixed parameters, transforms and support, prior, data seed/length,
initial distribution, noise scales, structural partition, sigma-point weights,
square-root/jitter, identity tolerance, negative-control noise, dense reference
size, Zhao-Cui extension ranks/design, affine chart, topology/optimizer grid,
batch/steps, heldout, HMC tuning, seeds, and equivalence rules. Chapter example
constants are fixture provenance, not automatic inference defaults.

## Repair Triggers

Graph/reference or posterior-recomposition mismatch re-enters model
construction/R1B. Identity or artificial-noise failure blocks filter admission
until repaired; it cannot be waived by good likelihood or HMC metrics. Negative
control failing to be detected invalidates the harness. Zhao-Cui route/design
failure affects `STR-ZC` only. Failed recipes are `RECIPE_REJECTED`, not cell
rejection without complete candidate-family accounting. Standard sampler/report
repairs remain cell-local.

## Forbidden Claims And Actions

- No artificial process noise in the intended deterministic coordinate.
- No naive-control posterior admission or target-signature reuse.
- No labeling `STR-ZC` source-faithful Zhao-Cui.
- No active graph NumPy/host callback/sample loop, CPU serious training,
  cross-cell transport, post-result thresholds, warm-up pooling, or overwrite.

## Handoff Conditions

P6 begins when the graph-native model and negative-control detector are admitted,
both eligible cells have honest terminal states, P5 artifacts/result/manifest
and repair history are complete, and P6 is refreshed with explicit SIR
observed-data posterior gaps and construction commands. A blocked structural
cell does not block SIR unless shared structural/filter code is invalidated.

## Stop Conditions

Stop serious P5 sampling for unresolved model/identity/reference ambiguity,
failure to detect naive noise, missing same-target comparator, unavailable GPU,
three identical repairs, or 36 GPU-hour exhaustion. Record cell-local extension
failure and continue. Stop program-wide only for shared harness contamination or
a required material target redefinition.

## Compute And Attempt Budget

Ceiling: 80 trusted GPU wall-hours plus 24 CPU reference/development hours. Each
eligible cell reserves two 15-GPU-hour family arms: plain dense IAF and one
P0-frozen target-specific enhanced family. Each arm permits one screen, one
selected fresh 5,000-step training, one NeuTra confirmation, and arm-local
retries. A separate 6-hour bucket funds plain HMC and comparator retries; 4
hours fund trusted R0/R1/R1B cell admission, cell-specific adapter/artifact
emission, and their repairs. Common harness/schema/reporting defects reopen and
charge P1 only. Three localized repairs apply per identical failure within the
owning bucket. Model and negative-control construction precedes that budget
ladder.

## Skeptical Pre-Execution Audit

The worked fixture alone cannot establish a parameter posterior, and a naive
full-state UKF could look numerically smooth while computing the wrong structural
target. Graph-native construction, pointwise identity, noise exclusion, and a
deliberate negative control answer those risks before NeuTra is allowed.
