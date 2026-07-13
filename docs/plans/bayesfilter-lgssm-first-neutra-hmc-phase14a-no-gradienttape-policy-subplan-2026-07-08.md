# Phase 14A Subplan: LGSSM No-GradientTape Target Policy

Date: 2026-07-08

## Phase Objective

Replace the admitted LGSSM generic target score route used by the
LGSSM-first NeuTra/HMC program with the existing analytical QR Kalman score
route, and demote `GradientTape` use to diagnostic/test-only scope.

This phase supersedes any Phase 14 interpretation that tries to make a
`GradientTape` LGSSM value/gradient path XLA-ready. The repair target is the
BayesFilter target-adapter boundary, not TensorFlow autodiff mechanics.

## Entry Conditions Inherited From Previous Phase

- Phase 13 showed that the taped value/gradient diagnostic still blocks under
  trusted GPU/XLA with a TensorList crossing-boundary error.
- The user clarified that BayesFilter should not use `GradientTape` except for
  diagnostic/reference purposes.
- The repository already contains an analytical QR derivative route through
  `QRStaticLGSSMTarget.analytic_score_hessian`.
- Existing Phase 10/11 NeuTra artifacts may remain historical diagnostics if
  the target or adapter signature changes.

## Required Artifacts

- Patched `bayesfilter/testing/lgssm_generic_target_adapter_tf.py` using the
  analytical QR score route for admitted LGSSM generic target scores.
- Focused source guard tests proving the admitted LGSSM adapter/helper source
  does not open `GradientTape`.
- A Phase 14A result note recording signature impact, checks, and nonclaims.
- A refreshed Phase 14 handoff clarifying that future XLA work starts from the
  manual/analytical score route.

## Required Checks/Tests/Reviews

- `python -m py_compile` for changed code/tests.
- CPU-hidden focused pytest for LGSSM generic target adapter and affected
  NeuTra mechanics tests.
- Source scan for `GradientTape` in the admitted LGSSM adapter path.
- `git diff --check`.
- Bounded read-only review may inspect the result/subplan before any later HMC
  sampling, training rerun, or XLA promotion.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the LGSSM generic target adapter stop using `GradientTape` for its admitted value/score route while preserving finite value/score behavior and analytical parity? |
| Baseline/comparator | Existing `QRStaticLGSSMTarget.analytic_score_hessian`, finite-difference tests, and old taped fixture tests as diagnostic history only. |
| Primary criterion | Focused tests pass and source guard verifies no `GradientTape` in the admitted LGSSM adapter/helper route. |
| Veto diagnostics | Nonfinite values/scores, finite-difference mismatch, analytical QR mismatch, hidden `GradientTape` in admitted route, silent promotion of old taped artifacts, HMC/training/sample execution. |
| Explanatory diagnostics | Target signature, adapter signature, source scan, finite parity residuals. |
| Not concluded | HMC convergence, posterior correctness, transport quality, XLA readiness, production readiness, sampler quality, or scientific validity. |
| Artifact | Phase 14A result note and focused tests. |

## Forbidden Claims/Actions

- Do not run NeuTra training.
- Do not run HMC sampling or tuning.
- Do not generate external samples.
- Do not claim XLA readiness from CPU-only tests.
- Do not promote old Phase 10/11 artifacts if signatures change.
- Do not rewrite unrelated LEDH or historical diagnostic files.

## Exact Next-Phase Handoff Conditions

The next phase may begin only if:

- the LGSSM generic target adapter uses analytical/manual score authority;
- focused tests and source guard pass;
- any signature change is recorded as invalidating old training/payload
  artifacts for promotion;
- no training, HMC sampling/tuning, or external sample generation occurred;
- the next subplan starts from the analytical/manual LGSSM score boundary.

## Stop Conditions

Stop if:

- analytical score parity fails;
- adapter signature behavior is ambiguous;
- preserving old signatures would require mislabeling the score authority;
- the repair would require training, HMC sampling/tuning, or sample generation;
- unrelated dirty worktree files block a scoped patch.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Use the existing analytical QR score route as comparator, not a taped score. |
| Proxy promotion | Passing source guards and parity tests does not imply HMC or XLA readiness. |
| Hidden assumption | Signature changes are allowed and must demote old artifacts to history. |
| Environment mismatch | CPU-hidden tests are support checks only. |
| Artifact mismatch | Result note must record whether old Phase 10/11 artifacts remain promotion-compatible. |

Audit status: proceed with scoped implementation.
