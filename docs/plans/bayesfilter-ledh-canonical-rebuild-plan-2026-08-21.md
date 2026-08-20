# LEDH Canonical Rebuild Plan: One Algorithm, Two Lanes, Zero Forks

Date: 2026-08-21
Authority: owner directive 2026-08-21 (three-part: invalidate historical
results; rebuild to the LaTeX-documented algorithm only; regression-proof).
Companion documents:
`bayesfilter-ledh-results-invalidation-notice-2026-08-21.md` (item 1),
`bayesfilter-ledh-conformance-test-plan-2026-08-21.md` (item 3).

## The Single Canonical Algorithm

The only version that may exist at completion, per the LaTeX contract
(`docs/chapters/ch19c_dpf_implementation_literature.tex:230-340`, Li 2017
Algorithm 1 + BayesFilter's reviewed extensions):

1. **UKF per-particle covariance lifecycle**: each particle carries
   `(x_{k-1}^i, P_{k-1}^i)`; unscented prediction through the nonlinear
   dynamics gives `(m^i, P^i)`; the flow consumes the PREDICTED per-particle
   covariance; unscented update produces `P_k^i`; resampling/ancestry moves
   the TRIPLE `{x, P, w}`.
2. **LEDH invertible flow (Li 17)**: dual-state integration (zero-noise
   auxiliary anchor drives linearization; actual particle moved by the same
   affine steps), per-particle `A_j^i, b_j^i` from the predicted `P^i`,
   forward log-determinant accumulated per pseudo-time step.
3. **PF-PF weight**: `w propto p(x|anc) p(z|x) theta / q(eta0|anc) * w_prev`
   with the transition density at the POST-flow particle and the proposal
   density at the PRE-flow sample — exactly the documented ratio.
4. **OT reset (Corenflos-style Sinkhorn + Contract E-Chol)** under the
   canonical reset/chunk policies already in AGENTS.md.
5. **Dual-cap trust-region GenUT correction** (diagonal + pairwise moments,
   LM damping, trust radius, radial step cap, coordinate clamp) — the
   general implementation surface, no reduced variants.
6. **Analytical recursive gradient**: the parameter score computed by the
   documented recursive/backward analytical identity. NO autodiff on any
   claim-bearing path. Autodiff and finite differences are demoted to
   parity/diagnostic oracles that gate the analytical implementation.

Two lanes, one algorithm: a single-cloud lane (rank-2, reference semantics,
float64-friendly) and a batch lane (leading batch dimension, XLA-compatible,
FP32/TF32 per repo policy). The single-cloud lane is the semantic authority;
the batch lane must pass batch-size-1 parity against it at every phase gate.
No third lane. The existing NeuTra bootstrap lane and all `*_diagonal_*`
scaffolds are deleted at completion (P7), not preserved as options.

## Existing Verified Parts (assembly inventory)

| Part | Location | Status |
|---|---|---|
| UKF predict/update (real sigma points) | `experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_alg1_ukf_tf.py` (`ukf_predict_additive_tf`, `ukf_update_additive_tf`; sigma machinery in `bayesfilter/nonlinear/sigma_points_tf.py`) | Exists; fixture-scoped; June 2026 Alg1-UKF campaign exercised it. Needs porting into `bayesfilter/` production namespace + per-particle batching. |
| LEDH flow core | `experiments/.../experimental_batched_ledh_pfpf_ot_tf.py::batched_ledh_flow_core_tf` | Exists; takes shared covariance — must be extended to per-particle `P^i` (the UKF handoff). Zero-noise anchor discipline must be verified against ch19c (dual-state integration). |
| PF-PF weight structure | `ledh_pfpf_genut_initial_rqmc_tf.py` | Exists (proposal-log, forward-log-det in weights); reuse after covariance rewiring. |
| OT reset + Contract E | `ledh_contract_e_reset_tf.py` + policies | Canonical already. |
| Dual-cap trust-region general surface | `higher_moment_contract_e.higher_moment_shape_jvp` (single-cloud, full); batch value-side port with parity oracle (2026-08-20) | Single-cloud complete; batch score-side port pending. |
| Analytical score | all-parent backward marks machinery (`standard_pairwise_backward_marks`, model score callbacks) | Exists for current models; must be extended through the UKF/flow stages — the hard new derivation (P4). |
| Model callbacks | `ledh_pfpf_genut_model_callbacks_tf.py` | Austria: replace all three identity placeholders (see D2 note below). |

## Phases

Each phase = plan -> skeptical audit -> implement -> conformance gates ->
result note. Phase gates are the tests of the companion test plan; a phase
without green gates does not close.

- **P0 — Contract extraction (doc-to-code binding).** Convert ch19c's
  Algorithm 1 into a machine-checkable step registry
  (`bayesfilter/highdim/ledh_alg1_contract.py`): named steps, required
  inputs/outputs, forbidden shortcuts (identity covariance, shared-P flow,
  state-only resampling, autodiff score). This registry is what conformance
  tests import — the spec stops being prose. Also: conformance matrix v0
  documenting current ABSENT cells as the baseline.
- **P1 — UKF lifecycle, single-cloud.** Port UKF predict/update into
  `bayesfilter/highdim/`; per-particle `P^i` storage; triple-carrying
  ancestry (states, covariances, weights move together). Gate: unit tests
  vs closed-form linear-Gaussian UKF (where UKF == Kalman exactly), sigma
  reconstruction identities, lifecycle-order tests.
- **P2 — Flow on per-particle covariances, single-cloud.** Extend the flow
  core: `P^i`-indexed prior precision; verify dual-state (anchor vs actual)
  integration and theta-product against ch19c equation by equation. Gate:
  linear-Gaussian equivalence (flow with exact covariances reproduces the
  Kalman posterior as substeps -> infinity within declared tolerance),
  invertibility check (log-det vs numerical Jacobian determinant on small
  fixtures), weight-ratio identity test.
- **P3 — Full single-cloud assembly.** UKF-predict -> flow -> PF-PF weight
  -> UKF-update -> OT reset -> dual-cap trust-region correction ->
  triple resampling. Replace Austria placeholder callbacks with real
  quantities: RK4 Jacobian or sigma-point transition covariance (P0 decides
  which per model class, recorded per model). Gate: per-step ESS
  instrumentation (Class A, mandatory artifact field), LGSSM exact-oracle
  match, Austria smoke with ESS profile compared against the identity-
  covariance historical baseline (descriptive).
- **P4 — Analytical recursive gradient, single-cloud.** Derive and implement
  the score recursion THROUGH the new stages (UKF moments, flow map,
  log-det, reset, correction). This is the scientifically hardest phase:
  each stage's parameter derivative is a documented derivation in a
  companion note (math discipline policy applies), implemented analytically.
  Gate: parity vs forward-mode autodiff oracle (autodiff as JUDGE, not
  implementation) at tight tolerance on small fixtures; FD regression as
  secondary explanatory check; the June Alg1-UKF campaign's derivative
  methodology reused where applicable.
- **P5 — Batch lane port.** Batch-native (leading batch dim preserved
  end-to-end per the NeuTra batching rule) port of P1-P4, including the
  batch score-side dual-cap port (closes registry item A5/D1). Gate:
  batch-size-1 bitwise-or-declared-tolerance parity vs single-cloud lane on
  every fixture; XLA/graph/eager mode identity gates (the compiler lessons:
  meta-off arm mandatory; TF32 arms per C-policy); capability-surface
  signature guards on BOTH value and score entry points.
- **P6 — Class C calibration + confirmation ladder.** Execute the R6
  calibration protocol (trust radius model-trust curve, LM damping bias
  curve, relative ridge derivation, dual-cap constants with owner rationale)
  on the CANONICAL lane; then the five-arm confirmation ladder (CPU, eager,
  graph-meta-off, XLA, TF32 on/off) on frozen scopes; fresh per-scope tuning
  under the LEDH tuning policy.
- **P7 — Deletion and de-confusion.** Delete the NeuTra bootstrap lane, the
  diagonal-only batch JVP, and every scaffold; the NeuTra target factory
  rebinds to the canonical batch lane; repo-wide grep proves no
  claim-bearing import of removed modules; AGENTS.md scaffold rule
  discharged. Historical artifacts remain on disk under the invalidation
  notice.

## Design Rulings Encoded (so agents cannot re-litigate)

1. Autodiff is an ORACLE, never a shipping path. This resolves the
  2026-08 value/score program-identity problem class at the root: the
  analytical score has one program by construction, and compiler rewrites
  of a value-only graph cannot split it from a JVP twin it does not have.
2. One semantic authority (single-cloud), one derived lane (batch), parity-
  gated. Lane forks are prohibited by AGENTS.md; the conformance matrix is
  regenerated in CI so an ABSENT cell is loud.
3. Per-particle covariance is non-negotiable (ch19c contract). Any
  shared-covariance approximation is a named, non-claim-bearing diagnostic.
4. D2 comparability: replacing the placeholder covariances changes the
  realized proposal distribution (identity was also the pre-flow sampling
  covariance — `ledh_pfpf_genut_initial_rqmc_tf.py:738-741`). ALL
  historical comparisons are severed by design; this is intended, per the
  invalidation notice, not a side effect to mitigate.
5. Budget realism: P4 is a research derivation, not a port; its subplan
  gets its own review before implementation. P1-P3 and P5 are engineering
  with existing parts. No phase may borrow against P4's uncertainty.

## Nonclaims

This plan promotes nothing by itself. Completion does not establish
posterior correctness, HMC readiness, or statistical superiority — those
remain gated by their own campaigns on top of the canonical lane.
