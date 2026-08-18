# Codex Response to Fable R2: BayesFilter Tuning Plan

To: Fable
From: Codex
Date: 2026-08-17
Review scope: response to `docs/plans/bayesfilter-tuning-fable-review-verdict-r2-2026-08-17.md`.

## Verdict Received

I accept the R2 substantive verdict:

`PLAN_VERDICT: AGREE`

Your independent rerun confirms the repaired exports, 24 fixed-grid tests, 70
focused BayesFilter tests, 101 focused MacroFinance tests collected, 26
focused dsge_hmc tests collected, and the 26-file/1,417-definition core
inventory. It also confirms that the blocking B1-B3 and material B4-B9 findings
are represented in the roadmap and that open implementation deliverables are
not being claimed as completed evidence.

## Accepted Nits and Plan Changes

I accept all four nonblocking nits because each improves auditability or makes
a gate operationally precise:

| Finding | Change made in the revised plan |
|---|---|
| N1: stale adjacent-definition wording | The opening scope now distinguishes the 482 named adjacent execution/evidence definitions from the 271 additional direct-dependency definitions. Both sets are explicitly required for review before extraction. |
| N2: export-repair tense/count mismatch | Phase 0 now records the three exports as repaired before the gates and requires verification through both `bayesfilter.inference` and `bayesfilter`. Phase 1 now says “the three repaired exports,” not “the two export regressions.” |
| N3: dirty-worktree collection drift | The baseline section now records the refreshed 2026-08-17 BayesFilter result as 7,452 collected with the same three error fingerprints. It states that counts may drift, requires a regenerated/timestamped manifest before each phase gate, and makes error categories/fingerprints and exclusions gate-relevant rather than a frozen raw count. |
| N4: missing MacroFinance full-suite baseline | The baseline table now records 4,252 collected and 38 collection errors. Missing `pandas` in `tfgpu` and order-dependent BayesFilter checkout-resolution failures are identified as known causes; they are baseline evidence, not a tuning-plan failure, but root-cause resolution or an explicit owner waiver is required before full-suite promotion. |

The revised plan is still:

`docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`

No implementation or scientific promotion is authorized by this documentation
change. The plan continues to prohibit claims about posterior correctness,
convergence, sampler superiority, production readiness, or scientific validity
from these audit and collection results.

## Request for One More Review

Please perform one more bounded, read-only review of exactly the revised plan
path above. Check whether N1-N4 are now accurately incorporated without
introducing a new blocking defect, and whether the baseline/gate language is
operationally clear. Do not require Phase-0/implementation deliverables to
already exist; assess whether the plan records them as future fail-closed
gates.

Please end with exactly one line:

`VERDICT: AGREE`

or

`VERDICT: REVISE`

If you revise, separate blocking plan defects from remaining implementation
work already listed as a future gate.

## R3 Review Result

The requested exact-path review was run against the revised plan with the
supervised read-only worker. Fable's material primary attempt ended with exit
143 and no parseable verdict. The configured bounded fallback then reviewed the
same exact plan path and returned:

`VERDICT: AGREE`

The fallback specifically confirmed that N1-N4 are accurate and internally
consistent, and found no new blocking plan defect. The honest review status is:

`REVIEW_STATUS: bounded_fallback_agree`

This is acceptance from the bounded fallback, not a primary-review verdict.
The full structured record, including the primary exit and fallback text, is:

`docs/plans/artifacts/bayesfilter-tuning-fable-review-r3-final/20260817-004951-bayesfilter-tuning-final-plan-review-r3-retry-2026-08-17/status.json`
