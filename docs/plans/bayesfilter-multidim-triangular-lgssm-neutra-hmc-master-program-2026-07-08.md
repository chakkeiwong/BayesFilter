# BayesFilter Multidimensional Triangular LGSSM NeuTra-HMC Master Program

Date: 2026-07-08

## Status

`LAUNCH_REVIEW_CODEX_SUBSTITUTE_AGREE_PHASE0_READY`

## Objective

Build and validate a serious multidimensional LGSSM parameter-estimation test
for BayesFilter NeuTra-HMC using a stationary lower-triangular or
block-lower-triangular transition matrix.

This program is separate from the completed static QR LGSSM Phase 17-21
runbook. That earlier runbook established a fixture-local mechanics/reference
gate, not serious multidimensional parameter estimation.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can BayesFilter perform serious HMC estimation for an identifiable, stationary, multidimensional LGSSM with NeuTra training on GPU and sampling on multicore CPU? |
| Candidate/mechanism | Lower/block-lower triangular LGSSM with fixed `H=I`, diagonal positive `Q/R`, stationary initial covariance, exact Kalman likelihood, manual score path, GPU-trained frozen NeuTra transport, and CPU-hidden `jit_compile=True` HMC. |
| Expected failure modes | Stationarity/identifiability contract flaw, stationary covariance derivative mismatch, score mismatch, XLA compile failure, weak synthetic recoverability, NeuTra training instability, HMC tuning failure, R-hat/ESS failure, reference mismatch. |
| Promotion criterion | Serious pass requires all implementation gates plus clean HMC diagnostics and reference/truth checks under the predeclared Phase 9/10 criteria. |
| Promotion veto | `jit_compile=false`, runtime `GradientTape` in admitted route, posthoc criteria changes, GPU sample generation, nonstationary posterior draws, malformed artifacts, unsupported broad/default/scientific claims. |
| Continuation veto | Source/design review failure, impossible identifiability contract, missing stationary initial law, no valid score path, or artifact corruption. |
| Repair trigger | Any fixable implementation/test/review issue that does not invalidate the model contract. |
| Explanatory diagnostics | Eigenvalues, Lyapunov residuals, moment checks, score residuals, compile timing/size, acceptance, divergences/errors, per-parameter R-hat/ESS, truth z-scores. |
| Must not conclude | Production readiness, default readiness, broad LGSSM validity, nonlinear SSM validity, DSGE/c603 validity, sampler superiority, or scientific validity beyond the named synthetic fixture. |

## Canonical Anchors And Local Code

- MARSS constrained linear state-space model form: `x_t = B x_{t-1} + u + w_t`,
  `y_t = Z x_t + a + v_t`, used here only as a constrained-structure anchor.
  Source: https://arxiv.org/abs/1302.3919
- Statsmodels stationary multivariate transform / Ansley-Kohn lineage, used
  as a stationarity reference and foil. The first serious test intentionally
  starts from triangular structure because it is easier to audit.
  Source: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.constrain_stationary_multivariate.html
- Stationary VAR parameterization for HMC context.
  Source: https://arxiv.org/abs/2004.09455
- Local BayesFilter stationary/Lyapunov derivatives:
  `bayesfilter/linear/stationary_lgssm_derivatives_tf.py`.

These anchors do not by themselves prove identifiability for the proposed
target. Phase 0 must explicitly classify what each source and local module
does and does not support.

## Program Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter build a source-anchored, stationary, identifiable-enough multidimensional LGSSM synthetic estimation benchmark and pass serious NeuTra-HMC diagnostics? |
| Baseline/comparator | MARSS constrained-state-space structure, triangular/block-lower stationarity contract, local stationary/Lyapunov code, exact Kalman likelihood, synthetic truth, and reference posterior diagnostics. |
| Primary pass criterion | Each phase writes a pass/blocker artifact, preserves boundary policies, and advances only after local checks plus review. |
| Veto diagnostics | Unsupported identifiability claim, missing stationary initial covariance, nonstationary dynamics/draws, score mismatch, `GradientTape` runtime path, `jit_compile=false`, GPU sample generation, hidden training/sampling, malformed artifacts, unsupported scientific/product/default claims. |
| Explanatory diagnostics | Source anchors, eigenvalues, Lyapunov residuals, moment estimates, score residuals, compile timing/size, training loss, HMC acceptance/divergence/error diagnostics, per-parameter R-hat/ESS. |
| Not concluded | Broad LGSSM readiness, product/default readiness, sampler superiority, nonlinear SSM validity, DSGE/c603 validity, or scientific validity outside this synthetic fixture. |
| Artifacts | Master program, runbook, ledger, subplans/results, review bundles/results, code/tests, data JSON, training payloads, HMC summaries/logs. |

## Phase Index

| Phase | Name | Objective | Subplan | Result |
| --- | --- | --- | --- | --- |
| 0 | Source And Identifiability Inventory | Source-anchor the target design, inventory local stationary code, and decide the first admitted triangular/block-lower contract. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase0-source-identifiability-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase0-source-identifiability-result-2026-07-08.md` |
| 1 | Model Contract | Formalize parameter names, transforms, stationarity, initial law, nonclaims, and signatures. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase1-model-contract-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase1-model-contract-result-2026-07-08.md` |
| 2 | Synthetic Data Fixture | Generate/review fixed-truth identifiable data and moment diagnostics. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase2-synthetic-data-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase2-synthetic-data-result-2026-07-08.md` |
| 3 | Stationary/Lyapunov Implementation | Implement or adapt stationary covariance/derivative helpers and tests. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase3-stationary-implementation-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase3-stationary-implementation-result-2026-07-08.md` |
| 4 | Target Score And XLA Compile | Build exact Kalman value/score adapter and `jit_compile=True` compile gate. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase4-target-score-compile-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase4-target-score-compile-result-2026-07-08.md` |
| 5 | Reference Posterior | Create a serious reference posterior route or blocker for the parameter target. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase5-reference-posterior-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase5-reference-posterior-result-2026-07-08.md` |
| 6 | GPU NeuTra Training | Train NeuTra on GPU only with `jit_compile=True`; no CPU training. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase6-gpu-neutra-training-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase6-gpu-neutra-training-result-2026-07-08.md` |
| 7 | Frozen Transport Packaging | Freeze and validate the trained transport payload. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase7-frozen-transport-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase7-frozen-transport-result-2026-07-08.md` |
| 8 | CPU HMC Pilot | Run a small CPU-hidden `jit_compile=True` HMC pilot with no promotion. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase8-cpu-hmc-pilot-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase8-cpu-hmc-pilot-result-2026-07-08.md` |
| 9 | Serious CPU HMC Estimation | Run serious multicore HMC and record per-parameter R-hat/ESS/truth/reference diagnostics. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase9-serious-hmc-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase9-serious-hmc-result-2026-07-08.md` |
| 10 | Readiness Decision | Classify the multidimensional LGSSM estimation evidence. | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase10-readiness-decision-subplan-2026-07-08.md` | `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase10-readiness-decision-result-2026-07-08.md` |

## Required Subplan Fields

Each subplan must state:

- phase objective;
- entry conditions inherited from the previous phase;
- required artifacts;
- required checks/tests/reviews;
- evidence contract;
- forbidden claims/actions;
- exact next-phase handoff conditions;
- stop conditions.

At the end of each phase, Codex must run required local checks, write a phase
result or blocker, draft/refresh the next subplan, and review the next subplan
for consistency, correctness, feasibility, artifact coverage, and boundary
safety.

## Approval Boundaries

Approved by this prompt only after launch review converges:

- creating planning/review/runbook artifacts under `docs/plans` and
  `docs/reviews`;
- local read-only/code-inventory commands;
- local CPU-hidden smoke/import/compile checks that do not run serious HMC.

Anticipated explicit approvals needed later:

- trusted/escalated Claude review-gate use;
- trusted/escalated GPU probes and GPU NeuTra training;
- CPU-hidden multicore HMC sampling runs longer than smoke/pilot;
- serious HMC result interpretation;
- package/environment changes, network fetches, or git commit/push.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Phase 0 must source-anchor and classify local code before implementation. |
| Proxy promotion | Compile success, training loss, acceptance, and moment sanity checks cannot promote readiness without serious HMC diagnostics. |
| Missing stop conditions | Every phase has explicit blockers and no phase may advance on silent assumptions. |
| Hidden assumption | Stationarity, coordinate identification, synthetic recoverability, and HMC diagnostics are separate ledgers. |
| Environment mismatch | GPU training and CPU sample generation are separated by policy and artifact fields. |
| Artifact mismatch | All generated data/training/HMC artifacts require hashes, signatures, seeds, and nonclaims. |

Audit status: launch reviewed by same-foreground Codex substitute after the
Claude review gate was denied for external-disclosure risk. Phase 0 may begin;
no implementation, training, or HMC boundary is authorized by this status.
