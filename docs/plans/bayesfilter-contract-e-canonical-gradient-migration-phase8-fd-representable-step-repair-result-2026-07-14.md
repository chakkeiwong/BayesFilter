# Phase 8 Result: Representable Symmetric FD Steps

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `SEVEN_STEP_FD_HEURISTIC_SCREEN_PASSED_FORMAL_CERTIFICATE_UNSUPPORTED`

## Outcome

The nearest-dyadic step repair resolved the prior endpoint-representation
inconclusiveness. The predeclared base was `2^-17`, the nearest power-of-two in
log scale to `cbrt(2^-52)`, with seven steps `2^-14` through `2^-20`.

All 35 endpoint pairs reproduced their nominal step bitwise in both directions,
matched each other, remained finite, branch/chart valid, and denominator-
eligible. All 35 relative errors passed the owner-directed FD-only heuristic
`0.05*sqrt(5) = 0.1118033988749895`.

The rigorous callable-error-bound FD certificate remains unconditionally
`unsupported`. This result is a heuristic implementation screen only.

## Evidence

- exact nominal plus-step matches: `35/35`;
- exact nominal minus-step matches: `35/35`;
- endpoint-valid and denominator-eligible pairs: `35/35`;
- FD-only heuristic passes: `35/35`;
- maximum relative error: `1.9172384855228604e-8`;
- center objective/score/branch, fixture, prepared inputs, source closure, one
  concrete XLA callable, and CPU-hidden provenance all pass;
- preflight tests: `6 passed, 2 warnings in 3.17s`.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close FD heuristic screen | all 35 exact endpoints and errors pass | Passed | tiny fixture only | Remove FD heuristic blocker | Rigorous derivative proof |
| Rigorous FD certificate | absolute callable error bounds | `unsupported` | TensorFlow/XLA kernel error bounds absent | Preserve nonclaim | Formal error interval |
| Advance target numerical arm | owner budgets required | Blocked independently | domain/ridge/transport requirements | Obtain owner amendment | Kalman/HMC readiness |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | All source, endpoint, branch, chart, finiteness, and artifact checks passed |
| Statistically supported ranking | None |
| Descriptive-only differences | All per-step relative errors and Richardson diagnostics |
| Default-readiness | Not established |
| Next evidence needed | Owner target numerical requirements and complete primary-shape statistical design |

## Artifacts

- Plan: `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-fd-representable-step-repair-subplan-2026-07-14.md`
- Result JSON: `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/fd-representable-repair-attempt1/result.json`
- SHA-256: `0526937fe43865dcce879fa52aa13813a0ad1a6a4d48ed1ba9faca62b30b16a2`

## Nonclaims

This pass does not prove the derivative, supply a confidence interval, or
establish target-shape FD, Kalman equivalence, ridge/transport adequacy, GPU or
HMC readiness, admission, leaderboard completeness, release, or integrity.
