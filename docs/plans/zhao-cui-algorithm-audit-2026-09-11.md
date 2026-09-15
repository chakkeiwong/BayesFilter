# Zhao-Cui Algorithm Audit, 2026-09-11

Status: BOUNDED_DERIVATION_AND_CALL_CHAIN_AUDIT_COMPLETE_REVISE.
Target checkout: `/home/chakwong/BayesFilterZhaoCui`, branch
`zhao-cui-tt-regression-20260908`, HEAD
`47176bce7cdaa91ddd4466c39f90a11ad005c800`, including the preserved uncommitted
September 10 implementation. The owner requested continuation of the algorithm
audit. Audit findings, focused executable checks, and reconciliation of the
master are in scope; a scientific comparison is not the immediate action.

Continuation authorized by the owner's subsequent request to finish checking
the derivation and code. The active manuscript target is
`docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`,
section `sec:algthree-active` (lines 2549-3207), including its earlier Hermite,
importance-sampling, and local C2-score dependencies. Historical campaign
tables are not new Algorithm 3 evidence. The current task is an audit, so
implementation repairs and scientific comparisons will not be conflated with
the verdict. The bounded continuation is complete and recorded in
[stage-result.md](artifacts/zhao-cui-audit-20260911-03/stage-result.md).
The current implementation requires repair: rank activation and finite-output
guards fail, the manuscript conflates numerical and smooth proposal densities,
and a non-test consumer is absent. Broader source parity, GPU/XLA validation,
and filtering-quality evaluation remain unestablished.

## Continuation Checks, 2026-09-11

Skeptical preflight: the earlier tests used one generated Gaussian coordinate
and did not establish a multidimensional proposal law, the real C2 consumer,
the meaning of numerical CDF interpolation, or production graph coverage.
The new checks compare directly with analytic polynomial/Gaussian identities,
cell slopes, and the existing real-model score formulas. Finite differences
are only a same-scalar derivative check. No proxy metric will establish
filtering accuracy, source-code equivalence, or scientific promotion.

The research question for this continuation is correctness of the declared
finite algorithm and its derivation. Candidate: current uncommitted code at the
recorded checkout. Expected failure: implementation or scope mismatch.
Primary criterion: checked mathematical identities and consumer-level numerical
agreement with independent reference calculations. Veto: wrong density,
derivative, measure, accepted nonfinite output, or absent required call chain.
Repair trigger: any localized counterexample. Continuation stops only for an
unavailable required source/environment or exhausted diagnostic budget, not
merely because a candidate defect is found. Runtime and fit RMS are explanatory.

Use at most three additional CPU/reference diagnostic invocations, each bounded
by 180 seconds, and one offline Lean check bounded by 90 seconds. GPU devices
are intentionally hidden before numerical imports. Float64, low-degree exact
polynomials, interior uniform points, and explicit nonsingular charts are
mechanics fixtures, not recommended defaults. Compare multidimensional
contractions with independent quadrature/algebra, and test numerical CDF cell
slopes separately from the original smooth TT density. Preserve new code,
commands, source hashes, and results under
`docs/plans/artifacts/zhao-cui-audit-20260911-02/`. Never overwrite attempt 01.

No GPU experiment, HMC run, tuning, environment mutation, or stochastic ranking
is part of these checks. An installed Lean toolchain may be used directly;
failure of the toolchain manager to discover/download another version does
not justify a package install. Current publisher/erratum search returned
HTTP 503; this remains an explicit literature-metadata gap while local
technical source inspection continues.

## Audit Contract And Skeptical Preflight

Question: does the master describe a mathematically sound, implemented
Zhao-Cui filtering program with an explicitly identified analytical derivative?
Compare the active LaTeX propositions with Zhao-Cui Algorithms 2 and 3,
equations (13)-(23), the author conditional-transport implementation, and every
consumer path exposed by the new endpoint. Distinguish a correct identity, a
source adaptation, a working reference implementation, and validated filtering
accuracy. None implies the others.

Pass criterion: each claimed identity has a checked derivation/source anchor;
each implementation claim has a matching call chain and an appropriate
executable check. Wrong proposal laws, derivatives of another scalar, silent
coordinate/measure changes, or accepted nonfinite results veto the affected
claim and any experiment relying on it. Candidate approximation error and
weak guide response trigger diagnosis, not rejection of the whole direction.
Runtime, fit residuals, ESS, and existing test counts are explanatory only.

Audit the settings themselves: Gaussian reference/tails, defensive mass,
fixed reference parameter, sample paths held constant, TT rank/degree/rows,
ridge, finite CDF inversion and its reported density, covariance guards,
initial observation convention, backend, and the frozen-versus-adaptive
derivative distinction. The current 28-test pass is not a whole-program
certificate; it omits the guide and some consumer paths.

Skeptical preflight: the original master mixes joint parameter/state learning
with a common-parameter frozen likelihood evaluator, and its ESS roles conflict.
Audit these as distinct targets. Check both Gaussian and bounded-grid proposal
paths, rather than extrapolating conditional-sampler tests to initialization.
Use deterministic fixtures for mechanics and label finite-sample/reference
comparisons explicitly. No stochastic ranking or default change is attempted.
This bounded audit can proceed; a claim-bearing C2 comparison cannot yet.

## Bounded Diagnostics

CPU/reference only: `CUDA_VISIBLE_DEVICES=-1`,
`BAYESFILTER_TEST_DEVICE_SCOPE=cpu`, `BAYESFILTER_PRELOAD_CUSTOM_OP=0`,
`PYTHONDONTWRITEBYTECODE=1`. Use the existing `tf-gpu` Python environment.
Algorithm kernels remain unchanged. NumPy/SciPy may be used only in these
independent diagnostic checks. No pfor, GPU run, HMC run, package install, or
scientific tuning. At most four focused invocations, 180 seconds each, including
local diagnostic repairs. Save each output beneath
`docs/plans/artifacts/zhao-cui-audit-20260911-01/` in the BayesFilter workspace.

Checks will cover: rank activation from the inherited constant TT initializer,
source-direction conditional contractions, Gram integrals,
the actual initialization and conditional proposal laws, full preparation to
particle-score wiring, deterministic path-sum/finite-difference score parity,
guide moments and observation response, and fail-closed numerical behavior.
Exact diagnostic commands and results will be recorded below. Any failure
becomes a reproducible finding before implementation changes are considered.

## Source Progress

- Checked local Zhao-Cui JMLR 2024 text: lines 450-650 (defense, Lemma 1,
  Proposition 2), 670-924 (upper KR, Algorithms 2-3, weights). The paper's
  joint parameter/state sampler and the local fixed-parameter program are
  distinct. Positive Gaussian defense alone does not establish the paper's
  bounded density-ratio condition or its error-bound condition on tau.
- Checked author `@TTSIRT/marginalise.m:25-85`,
  `@TTSIRT/eval_cirt_reference.m:43-153`, and `AbstractIRT.m:217-270`, under
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/`.
  The returned author quantity is negative log density despite the shorthand
  `pdf` in its comment; line 270 adds the physical Jacobian to that potential.
- Checked the active LaTeX frozen-score proof at lines 3055-3129. Its centered
  normalized-weight derivative is algebraically consistent with a frozen
  finite importance likelihood. It does not differentiate an adaptive TT fit.
- Current publisher/erratum lookup through the web tool returned HTTP 503.
  Current publication corrections and forward citation coverage are not yet
  verified. The local full paper and author code remain available.

## Findings And Results

The first bounded diagnostic is saved at
`docs/plans/artifacts/zhao-cui-audit-20260911-01/diagnostic-result-01.json`.
It is an independent CPU/reference check with CUDA intentionally hidden and
JIT disabled; it is not a promotion or scientific-ranking run.

### Reproduced blockers

1. **Rank activation defect.** For the positive rank-two target
   `h(x,y)=1+0.25xy`, the inherited `_initial_tt_cores` initialization produces
   an effective rank-one coefficient matrix and the fixed ALS fit remains at
   weighted RMS error `0.234375`. The same solver reaches RMS
   `1.03e-10` from an explicitly active rank-two start. This is an
   initialization/optimization defect, not evidence that the requested TT rank
   cannot represent the target. Any preparation path using the inherited
   initialization cannot support a rank-two approximation claim until repaired
   and regression-tested.
2. **Nonfinite aggregate accepted.** Evaluating a finite-input fixture at a
   very large parameter returns `valid=true` and `log_likelihood=-inf`; the
   recorded particle weight sums are `2.0` rather than normalized. The current
   finite checks cover intermediate factors and normalized log weights but do
   not reject a nonfinite accumulated likelihood or invalid normalized weights.
   This vetoes affected value, score, and downstream experiment claims until a
   fail-closed guard and healthy-case no-fire regression are added.

### Checks that passed their stated fixtures

- Preparation target versus an independent density calculation: maximum log
  error `1.78e-15`.
- Identity ancestry call-chain wiring: passed.
- Frozen path-sum value: error `0`.
- Analytical score versus direct path-sum score: maximum error
  `4.16e-17`; versus central finite differences of the same frozen scalar,
  `1.84e-12`.
- Hermite incomplete-Gram quadrature: maximum error `2.66e-15`; minimum
  eigenvalue `2.19e-10` on the diagnostic fixtures.
- Bounded initial proposal density versus numerical Jacobian: maximum log error
  `1.64e-10`.
- Sigma-point guide mechanics and invalid-covariance rejection: passed. The
  guide remains a positive cubature projection extension, not an exact Gaussian
  posterior update.

These passes establish only the checked mechanics for the frozen finite
program. They do not establish adaptive-TT differentiation, exact-model
likelihood gradients, paper-scale filtering accuracy, source-equivalence of
the local proposal extension, GPU/XLA behavior, C2 quality, HMC readiness, or
default/production status. The bounded continuation additionally checked the
coupled multidimensional conditional law and real C2 frozen score, and located
the two defects in their consumer call chains. Its results, further manuscript
and wiring findings, execution limitations, and next repairs are in the linked
stage result. No runtime repair has been applied in this continuation.
