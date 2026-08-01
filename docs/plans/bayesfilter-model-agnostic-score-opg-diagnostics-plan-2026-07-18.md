# Model-Agnostic Score OPG Diagnostics Plan

Date: 2026-07-18

Status: `EXECUTED_COMPLETE_DIAGNOSTIC_ONLY`

Result: `docs/plans/bayesfilter-model-agnostic-score-opg-diagnostics-result-2026-07-18.md`

## Objective

Replace componentwise realized-score relative error as the sole score
comparison with two additional model-agnostic diagnostics: global score-vector
norm errors and a scale-aware regularized average predictive-score OPG metric.
Implement the mathematics in
`docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`, test it independently,
and add it to the current LGSSM aggregation as a diagnostic witness without
changing the historical campaign screen.

## Research Intent Ledger

| Item | Contract |
| --- | --- |
| Main question | Can score disagreement be normalized without an analytic Fisher matrix and without division by an accidentally small realized score component? |
| Mechanism | Use the norm of the full score error and an average reference predictive-score OPG with diagonal shrinkage and a scale-aware `epsilon/T` ridge. |
| Baseline | Existing componentwise realized-reference relative errors, retained only for historical reproducibility. |
| Promotion criterion | Engineering only: formulas, shapes, invariants, serialization, and the LGSSM witness agree with direct calculations. No scientific promotion threshold is selected here. |
| Promotion veto | Wrong coordinate map; candidate-derived or particle-seed-derived OPG mislabeled as reference information; non-positive ridge scale; formula mismatch; changed historical arm classification. |
| Continuation veto | The reference cannot provide time-local score increments, the implemented matrix is not positive definite under its stated assumptions, or tests show the total score is not the sum of increments. |
| Repair trigger | Shape, serialization, TensorFlow graph, or reporting defect that leaves the mathematical target unchanged. |
| Explanatory diagnostics | Raw score norm, increment energy, average-OPG eigenvalues, shrinkage, realized ridge, floor activity, condition proxy, and LGSSM descriptive values. |
| Forbidden conclusion | No Kalman equivalence, nonlinear score correctness, HMC readiness, leaderboard admission, optimal shrinkage/ridge, or statistically supported method ranking. |

## Evidence Contract

The claimed target is the error between a candidate total score and a declared
reference total score in one frozen coordinate system. The normalization uses
only the reference predictive-score increments. Particle-estimator seeds remain
the analysis units for uncertainty intervals; their covariance is not Fisher
information and is never inserted into the metric.

The implementation must emit:

- absolute score-error norm;
- relative error against the total reference-score norm, undefined at zero;
- relative error against predictive-increment energy, undefined at zero;
- average OPG, shrunk average OPG, realized ridge, total metric, and eigenvalues;
- RMS Mahalanobis error under the total metric;
- maximum diagonally standardized coordinate error;
- coordinate label, reference source label, and explicit diagnostic-only status
  in the LGSSM artifact.

The old LGSSM `screen_pass/screen_fail/inconclusive` calculation and margins
remain byte-for-byte semantically unchanged. New metrics are descriptive in
this plan because their hyperparameters and scientific acceptance regions were
not frozen before the existing candidate scores were observed.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| HMC coordinates | Existing LGSSM comparison coordinates; reviewed baseline | Candidate and oracle already report these coordinates. | A physical/HMC mismatch changes all norms; verify the increment sum against the existing HMC score. |
| Average rather than summed OPG | User-approved mathematical amendment | Makes `epsilon/T` a relative order-`1/T` regularizer. | Applying it to the sum yields order `1/T^2`; algebraic unit test checks `T*M`. |
| `lambda` has no library default | Deliberate non-default | A fixed shrinkage is model- and coordinate-dependent. | Silent arbitrary tuning; require every caller to supply it and serialize it. |
| `epsilon_0` has no library default | Deliberate non-default | Ridge strength is a hypothesis, not a universal constant. | Candidate-driven threshold selection; require caller input and diagnostic-only witness values. |
| `epsilon_min=0` | Mathematical baseline, caller-controlled | Preserves the user-approved vanishing ridge. | Poor conditioning at long horizons; emit eigenvalues and allow an explicitly supplied floor. |
| Positive diagonal `D` | Minimal scale-aware design | Guarantees finite-horizon positive definiteness without analytic Fisher information. | Bad units or hidden coordinate dependence; require positivity and explicit serialization. |
| LGSSM witness `lambda=0`, `epsilon_0=1`, `D=I` | Diagnostic convenience choices, not reviewed defaults | Isolates average OPG plus ridge and demonstrates `T<p` behavior. | Values could look favorable by construction; no pass criterion or promotion is attached. |

## Skeptical Plan Audit

Audit result: `PASS_AFTER_REVISION`.

The first draft risked treating the OPG metric as Fisher-consistent for fixed
nonzero `lambda`; the chapter now states that fixed shrinkage persists and that
unshrunk OPG recovery requires `lambda_T -> 0`. It also risked applying
`epsilon/T` to the summed OPG; the implementation is instead anchored to the
average OPG. The following rejected alternatives are explicit:

- `g g^T` is rank one and is not the predictive-increment OPG;
- outer products across particle seeds estimate Monte Carlo variation, not
  observed-data information;
- a successful LGSSM number is not a promotion criterion for nonlinear models;
- a regularized inverse at `T<p` is computable but does not establish that the
  missing directions were identified by data;
- no threshold is selected after observing the existing candidate results.

The commands below answer formula and integration questions and do not launch a
long experiment. The plan has a bounded routine-work budget of one focused CPU
test run plus one LaTeX build, with one repair retry for each.

## Phase 1: Reusable TensorFlow Diagnostics

Implement a TensorFlow-only module accepting candidate total scores, one
reference total score, reference predictive-score increments, diagonal
shrinkage, base ridge, optional ridge floor, and a positive diagonal ridge
scale. Do not add NumPy algorithmic code or a scientific default.

Required checks:

- direct small-matrix formula agreement;
- positive definiteness for `T<p`;
- `T*M = T*shrunk_average + epsilon_0*D` when the floor is inactive;
- undefined relative metrics at zero denominators;
- batched candidate support;
- rejection of invalid shapes and hyperparameters;
- `tf.function(jit_compile=True)` CPU-hidden smoke if the local XLA CPU runtime
  supports the required linear algebra, otherwise a recorded non-JIT reference
  check and no XLA claim.

## Phase 2: LGSSM Diagnostic Witness

Obtain exact Kalman predictive-score increments by differentiating the same
Kalman prefix likelihood used for the total oracle. Verify that the increments
sum to the existing total HMC score. Add the generic metrics to a fresh
versioned aggregate artifact or a focused witness artifact; do not overwrite
the historical aggregate.

Required checks:

- same observations, parameter coordinates, and Kalman implementation;
- reference increment sum equals the total score to float64 tolerance;
- unregularized OPG rank is at most two for `T=2,p=5`;
- no particle-seed covariance enters the OPG;
- old screen logic and existing tests are unchanged.

## Phase 3: Documentation And Closeout

Run focused tests, compile the LaTeX document, write a result record with the
actual commands and outputs, and update the active failure ledger only to note
that the misleading score-normalization diagnostic has a repaired optional
replacement. Preserve all scientific gaps.

## Stop Conditions

Stop rather than claim completion if the formulas disagree, the metric is not
positive definite under the documented assumptions, Kalman increments do not
sum to the oracle score, the historical screen changes, or the LaTeX build
fails from this amendment. A candidate numerical result outside a future
acceptance region is not a reason to reject the metric or alter its parameters.

## Execution Close

All engineering promotion criteria passed on 2026-07-18.  The fresh witness is
diagnostic only, the historical LGSSM screen remains unchanged, and all
scientific nonclaims remain active.  See the linked result record for commands,
checks, numerical outputs, and the terminal red-team assessment.
