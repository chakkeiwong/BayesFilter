# C2 source compatibility and diagnostic reference repair

28 October 2026. The clean BayesFilter candidate
`f9291ee4437aa8c00dabbe331bffc6871b574c5c` passes the installed consumer's
33 integration checks and 36 selected BayesFilter core regressions. This
establishes compatibility under an explicit process-local source binding.
Official MacroFinance admission and posterior execution remain open.

## What failed and what changed

The unchanged 31-test preview passed 28 checks. Two tests failed when an eager
reverse-mode tape differentiated an XLA covariance Kalman function: TensorFlow
could not export its internal TensorLists. The third inspected only the public
wrapper after the analytic implementation had moved to a shared helper.
Finite-difference score agreement and all ten mixed-row/reference checks
passed; there was no observed analytic-score discrepancy.

Compiling the whole reverse-mode oracle was tested first and rejected: the
matrix-exponential reverse pass requires an unbounded TensorList. That failed
attempt remains preserved. The successful repair traces the independent
covariance oracle in a fixed-signature non-XLA graph, using the existing
BayesFilter filter's undecorated body. It copies no filter equations and keeps
strict support checks before graph execution. A bounded eight-shape cache is
an explicit memory convenience, not a scientific setting. The oracle records
its non-XLA exception and is never used as the active HMC score.

The structural test now inspects the wrapper and delegated implementation.
Two added checks preserve strict support rejection and graph reuse. The exact
two-file patch passed 33 checks in a temporary overlay, was installed in
MacroFinance with platform permission, and passed all 33 again. Existing dirty
consumer edits were preserved. No source pin, source root, model, support
threshold, numerical tolerance or analytic target function changed. The
consumer changes remain uncommitted there; their standalone patch is preserved
in the committed BayesFilter evidence package.

The clean candidate's core test conftest initially tried to preload an optional
Sylvester binary absent from the isolated tree. The selected C1/Phase 9 route
does not use that op. Disabling only that test preload allowed all 36 existing
stationary-derivative, stacked-QR, QR-score and candidate-artifact regressions
to pass. This provides no evidence about that optional binary.

## Evidence and remaining integration work

All 1,063 original frozen-source hashes were rechecked; all 70 C1-loaded
BayesFilter modules match the clean candidate. Every BayesFilter module
actually imported by the integration preview also matches C1. Real clean-tree
checks passed before and after each preview. The test-only path and commit
injection is explicit; the loader and source audit were never mocked.

Changing a pin alone would still be incomplete. The public consumer fixes its
runtime root to the concurrently edited shared checkout, while its factor-native
score provenance explicitly emits `source_commit=None` and
`pending_successor_source_admission`. Its previous validator expects the old
covariance-route schema. A future scoped binding must handle actual imports,
the exact clean commit and factor-native provenance together, with negative
tests for stale/mismatched/dirty sources. Do not rewrite the global pin as a
substitute or replace the shared checkout underneath other agents.

The external one-country master's read-only recovery/next commands also report
a missing historical P19 `control01/result.md`. No consumer master dispatch
was attempted. This is distinct from numerical compatibility. It must be
restored from authentic evidence or explicitly resolved by that program; never
fabricate a passing result to satisfy the check. Independent C4 saved-evidence
diagnosis does not depend on consumer deployment and can proceed.

Compact reproducibility package:
`docs/validation/modern-hmc-20261028-c2/` (about 0.4 MB). It includes the exact
BayesFilter candidate patch, consumer patch, per-attempt manifests, results,
JUnit reports and hashes. Original logs/test copies remain in
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/c2-source-admission-20261028-01/`.
All six attempts are terminal. Their measured inclusive worker cost is
277.012249805 seconds, including failures. The remaining diagnostic allocation
is at most 235750.796414879 seconds before unmetered preparatory overhead;
the earlier 84.50-second suite is pytest-reported rather than wrapper time.
This is not represented as fully settled accounting. Separate SIR remains
7363.064379415999 seconds. No new allowance is issued.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Install reference repair | 33 integration checks pass before and after installation | Source and support checks pass | Limited synthetic target coverage | Preserve patch and evidence | Posterior validity |
| Retain clean C1 candidate | 70 loaded-module identities and 36 core tests pass | Optional binary untested | Consumer deployment and provenance | Implement scoped binding with negative tests | Official admission or release |
| Continue independent C4 work | Saved posterior evidence available | No resampling needed | Variance-estimation uncertainty and tail scope | Diagnose existing full denominators first | New estimator/default promotion |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Original oracle/bootstrap failures preserved; repaired checks pass |
| Statistically supported ranking | None; these are correctness/integration checks |
| Descriptive-only differences | Runtime and pass counts |
| Default-readiness | Not established |
| Next evidence needed | Actual source/provenance binding, posterior precision/reference and tail qualification |

Post-run red-team: the strongest overstatement would be treating a process-local
preview as official consumer admission, or test parity as posterior convergence.
Neither is claimed. The independent covariance/finite-difference references
and unchanged analytic source reduce the risk of a circular oracle repair.
New target points can still expose numerical defects not covered here.
