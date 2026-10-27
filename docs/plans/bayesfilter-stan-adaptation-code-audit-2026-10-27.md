# Stan adaptation code audit, October 27, 2026

The owner asks whether the Python transplant is reusable and whether it is
correct and complete. This audit covers the existing standalone exception in
`docs/benchmarks/stan_nuts_warmup.py`, its caller, tests and guide. It does not
restore retired public NUTS routes or change a production default.

## Question and evidence contract

The target is the diagonal/dense Euclidean warm-up in the installed, pinned
CmdStan 2.37.0 release: reasonable-step search, dual averaging, window-local
Welford estimates, regularization, restart order and final freezing. Actual
unmodified Stan headers are the comparator, with controlled inputs where RNG
and tree differences would otherwise prevent an arithmetic comparison.
The candidate remains Stan adaptation around TFP's own NUTS transition.

Engineering acceptance requires source-by-source coverage, deterministic
agreement with the native oracle, valid metric/coordinate handoff, honest
failure propagation and complete enclosing CPU XLA compilation. A small GPU
XLA parity smoke checks changed numerical code if GPU access is available.
Unsupported options must be rejected or explicitly outside the interface.
Invalid retained states/caches, incorrect adaptation arithmetic, malformed
inputs accepted as valid and missing required oracle evidence veto acceptance.
Rejected invalid proposals contribute zero acceptance; native divergence flags
are explanatory, not adaptation rejection criteria.

The main expected failure mode is incomplete edge-case behavior hidden by
Gaussian integration tests. Targeted failures trigger local repair and a fresh
test attempt. Broken oracle provenance, a target mismatch or an exhausted
budget stops the affected validation; a failing implementation alone triggers
the planned repair. No convergence, posterior quality, runtime ranking,
production readiness, or identity of the complete Stan/TFP sampler follows
from this audit. Prior benchmark outputs remain preserved and version-specific.

## Audit and execution

1. Inspect every adaptation routine and its TFP acceptance/cache interface
   against installed source. Classify matches, deliberate limits and defects.
2. Check configuration/target contracts, singleton covariance windows,
   reasonable-step search stopping and errors, all metric boundaries, streaming
   estimator resets, final smoothing, and frozen sampling/resume.
3. Repair confirmed defects; expose/document safe standalone reuse without
   changing the public production API. Keep numerical kernels TensorFlow/XLA.
4. Expand the actual-header C++ oracle and focused tests. Require the oracle
   in the audit run instead of accepting skipped comparison tests. Verify
   CPU XLA, repeatability, batch isolation and a small GPU parity smoke.
5. Update the guide and record the coverage matrix, findings, tests, limits
   and terminal self-review. Preserve source hashes and exact commands.

Evidence root: `docs/benchmarks/artifacts/stan-adaptation-audit-20261027/`,
with a fresh numbered directory for every test launch. A maximum of six test
launches and 20 minutes total worker wall time, plus five minutes for compiling
the small oracle, is a convenience-chosen bound for focused engineering checks,
not a statistical budget. Each launch has a watchdog; no new MCMC benchmark
campaign is included. Record command, environment, git SHA and changed-source
hashes, seeds, CPU/GPU placement, memory-growth policy, wall time, failures,
repairs and remaining budget in the run manifest. Result note:
`docs/plans/bayesfilter-stan-adaptation-code-audit-result-2026-10-27.md`.

## Defaults and skeptical review before implementation

| Choice | Provenance/status | Risk and early diagnostic |
| --- | --- | --- |
| Stan 2.37.0 | Existing pinned native baseline | Avoid silently checking a different release; hash inspected headers |
| TF 2.20 / TFP 0.25, FP64 | Installed implementation baseline | Check installed acceptance semantics, dtype and enclosing HLO |
| Stan adaptation constants | Source-inherited defaults | Oracle exercises defaults and nondefault parameters; no adequacy claim |
| Tiny deterministic test targets and seeds | Convenience fixtures | Tests establish mechanics only; scripted oracle tests distinguish RNG/path differences |
| Arithmetic tolerance | FP64 rounding allowance, initially 2e-12 relative | Check actual residuals; never relax to conceal an ordering/sign error |
| Fitting positive windows, identity initial metric | Existing supported scope | Inspect singleton/short-budget behavior; document unsupported Stan options |
| Host validation around XLA | Interface hypothesis | XLA assertions may be dropped; require explicit checked status at host boundary |

Skeptical audit: the former tests leave the epsilon search, actual streaming
Welford update, small-count windows and invalid handoffs untested. They cannot
establish a complete transplant. The revised plan closes those gaps with
actual-source oracles and adversarial tests. Posterior diagnostics and timing
are not correctness proxies. The scope distinguishes generic target reuse from
a drop-in implementation of every Stan option. The plan is suitable for this
bounded audit after these corrections; this is self-review, not independent
review.
