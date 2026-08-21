# Fable R3 Bounded Review Request: Revised BayesFilter Tuning Plan

READ-ONLY BOUNDED REVIEW. Review exactly this file and nothing else unless it
explicitly asks you to inspect the one cited plan path. Do not edit files, run
commands, launch agents, or review the whole repository.

## Exact Review Target

Review exactly:

`docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`

## Question

After the R2 audit, Codex accepted and patched four nonblocking nits:

1. The plan now distinguishes 482 named adjacent execution/evidence
   definitions from 271 additional direct-dependency definitions, and requires
   both to be reviewed before extraction.
2. The three public export repairs are recorded as completed before Phase 0;
   both package layers must verify them; Phase 1 says “three repaired exports.”
3. The dirty-worktree BayesFilter baseline is refreshed to 7,452 collected / 3
   categorized errors, with regenerate-and-timestamp and fingerprint/category
   comparison rules before each gate.
4. The MacroFinance full-suite baseline is recorded as 4,252 collected / 38
   collection errors, with known causes and a root-cause-or-owner-waiver
   condition before full-suite promotion.

Audit whether these changes are accurate, sufficient, and internally
consistent. Identify any new blocking plan defect. Do not treat future
implementation deliverables as missing evidence if the plan correctly makes
them fail-closed gates. Do not make claims about posterior correctness,
convergence, sampler superiority, production readiness, or scientific validity.

Report findings first with exact path/line anchors. Distinguish blocking plan
defects from nonblocking nits and future implementation work.

End with exactly one line:

`VERDICT: AGREE`

or

`VERDICT: REVISE`
