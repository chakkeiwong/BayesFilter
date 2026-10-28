# C2 explicit consumer binding implemented and checked

28 October 2026. The [reviewed repair](bayesfilter-modern-hmc-c2-scoped-binding-2026-10-28.md)
is installed in MacroFinance. Phase 9's public builder accepts an explicit
Phase9SourceBinding for a clean linked checkout and exact commit. The actual
factor-native callable supplies its source identity; the caller's expected
binding controls validation. Old calls retain the historical pin and reject
the successor schema unless they explicitly supply the new binding.

The numerical implementation remains BayesFilter-owned. The tested checkout
is f9291ee4437aa8c00dabbe331bffc6871b574c5c, with the same 70 C1-loaded module
hashes and unchanged frozen source checks. All newly imported BayesFilter
modules also match C1. The direct public-builder test modifies neither global
root nor global commit. Its numerical target signature equals the same
diagnostic runtime's signature. Source provenance is separate from GPU policy;
posterior and GPU readiness remain false.

## Tests and repairs

- 21 source-binding tests pass in the overlay and installed consumer. These
  exercise real Git fixtures, linked checkouts, wrong commits, dirty tracked
  and untracked sources, foreign/ignored imports, wrong callables, changed
  consumer source, mismatched provenance and unchanged legacy rejection.
- All 33 prior Phase 9 numerical and mixed-row regressions pass in the overlay.
- The direct public-builder test passes in the overlay and installed consumer,
  with independent covariance reverse-mode parity for two parameter rows and
  unchanged tolerances. The final installed test also checks target-signature
  invariance, source revalidation and CPU-only device visibility.
- The first two attempts failed because the overlay omitted the unchanged
  Phase 8 synthetic-data helper. Preserve those attempts. The fixture source
  was copied and hash-bound before fresh attempts.
- Review caught provenance unnecessarily entering the GPU-policy dictionary.
  The correction and signature regression pass; earlier attempts are preserved
  rather than relabeled as testing the final bytes.

The exact five-file installation delta, per-attempt manifests/results, source
hashes, JUnit reports and costs are in [compact evidence](../validation/modern-hmc-20261028-c2-binding/README.md).
External installation used scoped platform-approved git apply. Pre-existing
MacroFinance edits were preserved and not committed wholesale. No sampler
transitions, GPU experiment or consumer-master dispatch occurred.

## Decision and evidence limits

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Accept explicit binding implementation | Installed public call and parity pass | Source/import/identity checks pass | Other consumers have not opted in | Reconcile C3 adapter/member identity | Full consumer admission |
| Preserve legacy gates | Old validator rejects unbound new schema | Global pin unchanged | Historical P19 evidence is missing | Repair missing evidence in its own program | Permission to dispatch that master |
| Continue modern-HMC work | Tested source integration available | Criteria unchanged | Posterior precision/reference remain open | Declare C3 contract before pricing | Posterior or release readiness |

| Inference class | Status |
|---|---|
| Hard veto screen | Engineering checks pass; missing historical evidence still blocks consumer master |
| Statistically supported ranking | None; deterministic integration tests |
| Descriptive-only differences | Test durations |
| Default-readiness | No default changed or promoted |
| Next evidence needed | Downstream opt-in and source-matched posterior/reference qualification |

Measured attempts plus pytest-reported unit time total 236.518482500 seconds,
leaving at most 235506.893657597 seconds before unmetered coding, preparation
and packaging. Concurrent attempts are charged additively. This is an upper
bound, not a fully settled account; separate SIR's 7363.064379416 seconds remain.
All phase processes are terminal. The public test is skipped outside an
explicitly configured clean checkout rather than running on the dirty tree.

Red-team review: binding does not prove geometry, sampler, reference or posterior
validity. Downstream callers must opt in with the same expected binding. The
candidate under /tmp must remain present and unchanged for bound execution.
C3 must inspect existing member identities and qualification flags; it cannot
relabel diagnostic C1 members as posterior-qualified or bypass GPU readiness.
