# Backward guidance in the existing three-proposal mixture

## Question and scope

Can frozen controlled-SMC quadratic messages and frozen KDM-style Gaussian
mixtures guide the existing identity/global/local proposal without changing
its target weights or losing analytical total derivatives? This stage tests
implementation correctness. It does not select a guide or compare model-score
accuracy. Work only in `ledh-refactor-with-policy-fix`, HEAD
`ea1018de1da55ed02a29a8d60f08a6827fce8a42`, in the LEDH execution worktree.

The owner correction in
`artifacts/ledh-ot-balance-weight-20261016/mixture-route-correction.md`
establishes the baseline: `covariance_proposal_tf.make_filter`, already shared
with main. The previous single-flow consumer is not this baseline. This
optional mixture candidate has its own moment-restoring reset; this work does
not confer canonical LEDH/Contract-E admission on it.

## Composition and evidence contract

- Preserve the identity branch `q0=p_N^-`. Keep branch and ancestor draws,
  full mixture density weights, OT and the existing moment reset in the SAME
  runtime implementation.
- Global: multiply the current-observation quadrature Gaussian by the future
  quadratic evaluated at its mean, then construct the existing global affine
  recolouring from predictive moments to those guided moments.
- Local: add future quadratic precision/natural vector, evaluated at the
  deterministic conditional transition mean, to the current-observation
  information in the existing conditional-Q bridge.
- Use the existing frozen guide evaluators. A mixture guide's responsibilities
  depend on moving anchors; retain their analytical tangents. Coefficients and
  beta are fixed during parameter evaluation. Current likelihood appears once
  in the importance numerator, and future information appears only in maps.
- Primary pass criterion: independent scalar Gaussian moment/density checks,
  zero-guide parity, defensive-ratio bound, and end-to-end central-difference
  checks of the returned value and particles for every model parameter in a
  small multidimensional Gaussian fixture. Verify actual consumer wiring.
- Hard continuation vetoes: wrong numerator/density, missing derivative term,
  nonfinite accepted output, rank/cap/moment violation, or budget exhaustion.
  A malformed guide must fail closed. OT marginal residual is reported, not
  automatically interpreted as model-score harm.
- Explanatory diagnostics: ESS, mass residual, runtime and derivative errors.
  None is a promotion criterion for statistical accuracy.
- Artifacts: `docs/plans/artifacts/ledh-mixture-guide-20261016/`, versioned run
  subdirectories, complete logs, manifest, result and concise checkpoint.

## Default and assumption audit

| Choice | Provenance and reason | Failure / earliest check | Status |
|---|---|---|---|
| epsilon=1, Sinkhorn=5 | Owner's requested fixed baseline | Reset validity and finite-score parity | Baseline; not tuned |
| flow_steps=16 | Existing mixture finite program | Wrong endpoint assumptions; compare actual affine density, allow Euler error only in endpoint check | Inherited baseline |
| beta=(0.2,0.3,0.5) | Interior fixture exercising every branch | Incorrect support/branch density; exact ratio bound | Test fixture, not selected beta |
| Frozen future guide | Differentiable proposal design already derived | Missing moving-anchor tangent; two-component FD check | Method hypothesis |
| PSD future precision, no added ridge/damping | Gaussian multiplication is integrable for SPD prior plus PSD information | Invalid precision/covariance must reject; relative Cholesky margins | Mathematical domain, no silent numerical repair |
| Correction 0 and 2 | Existing optional correction controls | Incorrect composition/cap; full-consumer tests | Mechanics fixtures, not tuned settings |
| CPU FP64 graph / GPU FP64 XLA | Cheap independent diagnostic, then compiled consumer | Environment ABI or compilation fork; matching tftwogpu env and same callable | Explicit reference exceptions |

Guide coefficients are synthetic fixed messages for integration tests. They do
not demonstrate controlled-SMC fit quality or a replicated KDM learner. A
one-component Gaussian representation is an algebraic parity case only.

## Skeptical audit before execution

PASS for this bounded engineering stage. The corrected comparator is the
existing three-branch mixture, not the other consumer's prior failures. Its
reset/admission distinction is explicit. No finite-program derivative check
will be presented as model-score accuracy. This engineering check uses fixed
synthetic coefficients without fitting or tuning. A later backward-message
pilot may condition on the evaluated sequence's future observations, using
independent pilot randomness and coefficients frozen during evaluation;
settings selection remains separate from final comparisons. Tests cover both
guide representations and the
moving-anchor dependence, and preserve the no-guide caller. Environment is
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; the previous tf-gpu ABI
failure is not a reason to mutate environments.

For the subsequent score experiment, construct identity/bootstrap (robust
coverage), defensive-global (shared Gaussian geometry), defensive-local
(ancestor-specific current information), and equal-mixture (no beta fitting)
adversaries. Compare conditionally on weak/persistent transitions and mild/
surprising observations, with exact Kalman scores. Separate fitting, beta
calibration, validation and untouched data; freeze everything before scoring.
Paired uncertainty and heuristic non-inferiority are required for promotion.
That research experiment is NOT executed by this mechanics plan.

## Execution and budget

1. Extend `build_maps` / `make_filter` with an optional frozen guide; add the
   proposition/proof and pseudocode to the existing guide manuscript.
2. Run focused CPU tests (GPU deliberately hidden), preserving full output:
   `CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_covariance_proposal_guide.py tests/highdim/test_covariance_proposal.py`.
   Select bounded regressions if the remaining budget requires it.
3. Run the small shared-consumer FP64 GPU/XLA check with escalated GPU access,
   memory growth verified before initialization. No new backend or scalar fork.
4. Audit algebra with MathDev, compile and inspect changed PDF pages, then
   record measured results and update the master/checkpoint.

Remaining inherited budget is **80 CPU / 400 GPU wall seconds** for numerical
commands, including retries and GPU preflight. Coding and manuscript builds
are not numerical experiment time. Use at most two CPU attempts and two GPU
attempts, reducing scope if needed; do not reuse or overwrite a run directory.
Do not begin the score campaign within this budget. New input may authorize a
separate bounded phase after this engineering result, but elapsed time alone
does not do so.

Reader contract: the mathematical reader knows Gaussian conditioning and
importance sampling and should be able to reconstruct where the guide enters,
why the numerator stays unchanged, and what the score differentiates. Extend
the existing argument; preserve its earlier proofs and mark their consumer
scope. The writing remains provisional pending reader feedback.

## Completion, 16 October 2026

The implementation, bounded tests, mathematical expansion and terminal
consistency audit are complete; see
`artifacts/ledh-mixture-guide-20261016/result.md`. Nineteen unique CPU tests
pass. The four-coordinate GPU/XLA check passes derivative and CPU parity
criteria. MathDev verifies two scalar identities; its automatic proposition
audit remains inconclusive, not a proof certificate. The document distinguishes
the earlier single-flow findings from this mixture's separate reset.

Numerical use is35.26 CPU/18.30 GPU process seconds plus one GPU second for
preflight; remaining44.74 CPU/380.70 GPU seconds. The next task is a separately
planned model-score comparison with fitted-guide and guided-beta wiring.
The mechanics stage does not establish a statistical ranking or authorize
default promotion.
