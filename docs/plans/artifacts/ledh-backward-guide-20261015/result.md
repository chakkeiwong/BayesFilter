# Frozen backward guidance: implementation and consistency result

## Checkout correction, 2026-10-15

This work was performed in the wrong checkout. The user's intended target is
`ledh-refactor-with-policy-fix` in
`/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`,
verified at `b28c25a6c749a08d191d05bd30a1ca5dfc0def69` with a clean worktree.
The completion statements and checks below describe `surrogate-hmc` only;
they do not establish completion or consistency on the intended LEDH branch.
Its executor and manuscript differ materially. Preserve the original evidence,
reconcile changes rather than replacing whole files, and revalidate in the
intended worktree before claiming completion. No transfer or merge has occurred.

Date: 2026-10-15. Checkout: `surrogate-hmc` at
`feddae4001fb85bb1be932d75179cb91ff4f0d77`, with uncommitted changes.
Scope: the bounded engineering plan in
[`ledh-backward-guide-integration-20261015.md`](../../ledh-backward-guide-integration-20261015.md).

The scoped implementation is complete. The manuscript contains six
propositions with proofs and pseudocode; the optional guide executes through
the shared LEDH recursion; all ten focused tests pass. The analytical
derivative matches central differences of the same finite program on the
tested fixture. This does not establish equality to the exact model score or
an improvement in score estimation.

## Mathematical and executable correspondence

The claimed computational target is the finite log-normalizer defined by the
existing recursion, with its proposal draw and proposal-density denominator
replaced together. The code computes its analytical total derivative with
frozen guide coefficients, including moving mixture responsibilities,
full-covariance proposal tangents, physical importance-density terms and the
existing OT/Contract E/reset recursion. Differentiation of a fitted policy is
outside this definition.

| Manuscript proposition | Implementation or derivation | Evidence and limit |
| --- | --- | --- |
| 19, `prop:bg-backward` | Backward-information recursion and smoothing transition derived in the manuscript | Mathematical specification; no pilot fitter or ancestor smoother is implemented here |
| 20, `prop:bg-gaussian` | `guided_gaussian_draw`, `FrozenQuadraticBackwardGuide.from_controlled_smc` | Normalized density, full covariance draw and moving-density tangent tests |
| 21, `prop:bg-mixture` | `FrozenMixtureBackwardGuide.evaluate` | Responsibility derivatives and spatial gradient checks; positive curvature is a local proposal choice, not the exact log-mixture Hessian |
| 22, `prop:bg-correction` | Shared executor evaluates physical transition and observation densities and subtracts the guided proposal density | Wiring and denominator tests; exact importance sampling additionally requires the actual bijective flow Jacobian |
| 23, `prop:bg-total` | Shared analytical log-normalizer and complete reset recursion | Per-parameter and initial-state/covariance directional finite differences; one small interior fixture |
| 24, `prop:bg-no-ranking` | Explicit finite-state counterexample in the proof | Establishes that weight/ESS optimization alone does not guarantee score-variance improvement; no empirical ranking is attempted |

The consumer is `build_backward_guided_ledh_kernel` in
`bayesfilter/highdim/ledh_backward_guide_tf.py`. It calls the existing executor
in `bayesfilter/highdim/ledh_canonical_score_tf.py`; the UKF, flow, weighting,
OT, Contract E and GenUT correction implementations are shared. The focused
test exercises that call chain rather than merely checking that primitives
exist. This is an optional single-cloud candidate, without a new canonical
route identity, fitted controlled-SMC/KDM workflow or batch-lane claim.

## Validation and repairs

The first test attempt in the existing `tf-gpu` environment failed during
collection because the installed symmetric-Sylvester custom operation had an
unresolved TensorFlow/Abseil symbol. This was an environment ABI failure, not a
numerical result. No packages or environments were modified. The second
attempt used the existing compatible `tftwogpu` environment and passed:
**10 tests, two TFP deprecation warnings, 11.56 seconds**. See
[`pytest-attempt-1.log`](pytest-attempt-1.log),
[`pytest-attempt-2.log`](pytest-attempt-2.log) and
[`environment-probe.log`](environment-probe.log).

Both attempts deliberately set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow
import. These are CPU-only, non-JIT debug/reference checks, not GPU/XLA
readiness evidence. The environment reports TensorFlow
`2.20.0-dev0+selfbuilt`. The candidate factory defaults to XLA, but that
execution mode was not validated by these checks.

The successful command, from `/home/chakwong/BayesFilter`, was:

```sh
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_ledh_backward_guide_tf.py > docs/plans/artifacts/ledh-backward-guide-20261015/pytest-attempt-2.log 2>&1
```

`latexmk` was unavailable. The fallback `pdflatex`, `bibtex`, `pdflatex`,
`pdflatex` sequence completed successfully, with no unresolved references or
citations on the final pass. The resulting
[`ledh_younis_kdm_score.pdf`](../../../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.pdf)
has 63 pages. All five new section pages, 54–58, were rendered and visually
inspected; equations and pseudocode are readable without visible overflow.
The build logs and `rendered-54.png` through `rendered-58.png` preserve this
check. Exact commands, source hashes and settings are in
[`run-manifest.json`](run-manifest.json).

## MathDevMCP audits and their limits

MathDevMCP was used before implementation and again after the proofs and code
were complete. The initial tree audit failed internally; its bounded
audit-and-fix fallback completed four selected propositions. The final audit
completed all six selected propositions without skipping one. Reports are
[`audit-before.md`](audit-before.md),
[`audit-before-fallback.md`](audit-before-fallback.md) and
[`audit-after.md`](audit-after.md), with structured reports in the sibling
`audit-before/agent-reports` and `audit-after/agent-reports` directories.

The audit status is **diagnostic only**, with unresolved formalization and
human-review obligations. The final report has 28 such items; it does not
certify the proofs. Validation was disabled in the reported algebraic
backend, and no proof detail was validated. Empty veto and counterexample
lists therefore do not establish mathematical correctness.

The three document/code comparison reports return **mismatch** because some
required prose/LaTeX terms are absent from the code text. These are lexical
comparisons, not semantic equivalence tests. They remain preserved as
`compare-math-guide.json`, `compare-math-executor.json` and
`compare-plan-guide.json`; they must not be reported as passing. The equation
mapping above and the executable density, derivative and consumer-wiring
checks provide the bounded consistency evidence instead. Formal proof
closure remains open.

## Defaults and assumptions

Epsilon 1 and five Sinkhorn iterations are the owner's untuned development
comparator. The synthetic seed, horizon, particle count, flow discretization,
ridge, balancing, damping, trust and correction caps come from the diagnostic
fixture or explicitly selected diagnostic options. They are not calibrated
scientific defaults. Their exact values are recorded in the manifest.

Zero-guide parity, invalid precision rejection, unsupported transition/reset
rejection and directional derivatives are early implementation checks. They
do not justify the fixture's numerics as safe across models or support a
claim-bearing run without scope-specific tuning. Mixture coefficients are
frozen; Gaussian transitions and valid covariance/precision domains are
implementation assumptions.

## Decision and inference status

| Decision field | Result |
| --- | --- |
| Decision | Keep the optional implementation for further validation; scoped engineering work complete |
| Primary criterion | Same-finite-program directional checks, zero-guide parity and shared-call-chain checks pass on the fixture |
| Veto status | No implementation veto fired on the accepted fixture; invalid/unsupported inputs are rejected or flagged in focused tests |
| Main uncertainty | Actual localized-flow Jacobian, untested models/devices and guide-fitting quality |
| Next justified action | Check the actual flow Jacobian before an exact importance-correction claim; then design a separately bounded, target-specific heldout score experiment |
| Not concluded | Exact model score, unbiased complete reset filter, score improvement, scientific/default/canonical/HMC or GPU/XLA readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Focused valid fixtures pass; collection failure was classified and repaired by selecting a compatible existing environment |
| Statistically supported ranking | None; no stochastic method comparison was conducted |
| Descriptive-only differences | Test runtime and any fixture score differences cannot establish superiority |
| Default-readiness | Not evaluated; owner `(1,5)` remains an untuned comparator |
| Next evidence needed | Actual-flow-Jacobian validation, target-specific guide training/tuning, conditional heuristic comparisons and paired score-error uncertainty against an adequate reference |

The strongest alternative explanation for apparent physical correctness is
that the existing localized-flow log determinant differs from the Jacobian
of the actual flow map. The manuscript explicitly makes the importance
identity conditional on that Jacobian; these tests do not close that question.
A same-program finite-difference mismatch or a consumer that bypasses the
shared reset would overturn the corresponding engineering verdict. The
weakest evidence is coverage of only one small interior recursion fixture.
Neither a good guide fit nor improved weight ESS would establish improved
score accuracy without the downstream comparison specified in the plan.
