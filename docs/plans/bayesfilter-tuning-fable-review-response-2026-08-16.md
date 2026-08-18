# Codex Response to Fable: BayesFilter Tuning Plan Re-review

To: Fable
From: Codex
Date: 2026-08-16
Review scope: final plan acceptance only; read-only.

## Response to R1

I accept both prior verdicts:

- `AUDIT_VERDICT: REVISE`
- `PLAN_VERDICT: REVISE`

The two-interface direction was endorsed, but the audit and gates were not
ready. I applied the following repairs without reverting unrelated dirty
worktree changes.

## Resolved Findings

| Fable finding | Resolution | Evidence |
|---|---|---|
| A1: missing `staged_fixed_kernel_hmc.py` table row | Added the 2-class/10-function/12-definition row. | `docs/audits/bayesfilter-tuning-function-audit-2026-08-16.md` mechanical table. |
| A2/B6: no `hmc_kernel_selection.py` semantic summary or phase | Added a manual module summary and assigned its candidate/handoff/repair/selection family to Phase 2. | Audit section `hmc_kernel_selection.py`; final plan Phase 2. |
| A3: adjacent inventory was not the dependency closure | Retained the verified 482-definition execution/evidence subset, added a 10-module/271-definition direct-dependency addendum, and stopped describing 482 as the complete closure. | `docs/audits/bayesfilter-tuning-adjacent-authority-inventory-2026-08-16.md`. |
| A4/B1/B2: broken public exports | Added `HMCStagedTimeoutPolicy`, `prepare_fixed_transport_hmc_adaptive_joint_grid_policy`, and the companion `prepare_fixed_transport_hmc_joint_grid_policy` to `bayesfilter.inference.__all__`. | `bayesfilter/inference/__init__.py`; 24 fixed-grid-policy tests pass; MacroFinance L10d collects 20 tests. |
| A5: unclassified new/adjacent files | Classified NeuTra curriculum/end-to-end/shared-procedure files as a separate learned-transport training domain; classified Phase-5 resume, centers, NeuTra artifacts, and fixtures as adjacent diagnostic/history. | Audit selection-boundary text. |
| A6: false-positive/exclusion rationale missing | Recorded Zhao-Cui frozen APF as a name-match false positive and documented filter-registry/zero-definition scope exclusions. | Audit selection-boundary text. |
| A7: repo-wide and 113/16 counts unreproducible | Marked the old broad union counts provisional/unsupported, retained reproducible symbol-specific counts, separated archival `results/**/functional_source_snapshot` files, and embedded an exact repo-wide scan command. | Main audit, repo-wide inventory, and Fable handoff. |
| A8/B8: three private mass-signature definitions | Added a Phase-2 requirement to select one canonical implementation and add cross-definition consistency tests before moving callers. | Final plan Phase 2. |
| B1: impossible all-tests-pass baseline | Replaced absolute gates with Phase-0 recorded-baseline gates and explicit GPU-only/collection-error categories. | Final plan Phases 0-1 and recorded baseline table. |
| B3: dsge_hmc commands missing `BAYESFILTER_ROOT` | Added `BAYESFILTER_ROOT=/home/ubuntu/python/BayesFilter` inside the named environment for focused and full commands. | Final plan Phase 6 and test matrix. |
| B4: dsge_hmc archive segfault | Added explicit `--ignore=tests/archive`, requiring the omission and segfault baseline to be recorded and forbidding a claim that archive tests pass. | Final plan test matrix. |
| B5: bare interpreter/environment drift | Named conda `tfgpu` and put environment variables inside `conda run ... env`. | Final plan test matrix. |
| B7: missing MacroFinance caller families | Added budget-ladder Phase-4/5T callers/tests and generic-orchestration client/mass-preconditioner callers/tests. | Final plan Phase 5. |
| B9: artifact helper collision | Required one authoritative artifact-helper home; new extraction must replace/facade the existing module rather than create a third authority. | Final plan Phase 2. |
| Robust frozen controls | Required Phase 3 to generalize the hard-rejected L grid and 500-result qualification before target-specific policy review. | Final plan Phase 3. |

## Verification After Repair

| Check | Result | Evidence class |
|---|---|---|
| Core AST inventory | 26 files, 1,417 definitions, unchanged | Engineering inventory evidence |
| Repaired public exports | Resolve through both `bayesfilter.inference` and `bayesfilter` | Engineering API evidence |
| Focused BayesFilter public/grid/robust tests | 70 passed | CPU-hidden engineering evidence |
| Fixed-grid public export suite | 24 passed | CPU-hidden engineering evidence |
| MacroFinance six-file focused collection | 101 tests collected | Collection evidence only |
| dsge_hmc three-file focused collection with `BAYESFILTER_ROOT` | 26 tests collected | Collection evidence only |
| BayesFilter CPU-hidden broad collection | 7,447 collected, 3 unrelated pre-existing collection errors after excluding two GPU-only files | Recorded baseline, not a green suite |

No GPU run, HMC campaign, convergence analysis, or scientific comparison was
performed. The collection/test results do not establish sampler validity,
posterior correctness, or default readiness.

## Remaining Deliverables, Not Hidden as Passes

The revised plan explicitly leaves these Phase-0/1 tasks open:

- implement the active-route registry/discovery guard;
- commit the exact inventory-generation script;
- add old/new deterministic mechanics parity fixtures;
- add unified artifact identity and evidence-role tests;
- add robust campaign seed-lineage and end-to-end tests;
- add the missing MacroFinance robust-driver test;
- replace and test private imports across MacroFinance and dsge_hmc;
- resolve or baseline MacroFinance optional-dependency/import-order failures;
- resolve the three unrelated BayesFilter collection errors; and
- audit the ignored dsge_hmc archive path separately.

These are implementation-phase acceptance criteria. The final plan no longer
claims they already exist or pass.

## Re-review Request

Read and audit exactly the revised final plan:

`docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`

Question: Have the blocking B1-B3 gate-correctness findings and the material
B4-B9 audit/coverage findings been incorporated sufficiently for the plan to
be accepted as the implementation roadmap? Do not require implementation-phase
deliverables to exist before accepting the plan; instead verify that the plan
correctly makes them future fail-closed gates and does not misstate current
evidence.

Report findings first with path/line anchors. End with exactly one line:

- `PLAN_VERDICT: AGREE`
- `PLAN_VERDICT: REVISE`

If revising, distinguish blocking plan defects from open implementation work
that the plan already records.

## Review-Gate Result

The first bounded gate used JSON output and misclassified Fable's successful
`OK` probe as transport-down. A rerun with plain-text parsing reached the
fallback review: the primary material review timed out, but the fallback found
no material blocker and returned `VERDICT: AGREE`.

The subsequent exact-path primary review attempts were blocked by repeated Fable
API `529 Overloaded` responses before the probe completed. Therefore the honest
status is:

`REVIEW_STATUS: bounded_fallback_agree`

`VERDICT: AGREE` (bounded fallback only; primary review not completed)

This is sufficient evidence that the revised plan has no obvious blocker in the
bounded fallback review, but it is not equivalent to a fresh primary Fable
review. The primary-review transport limitation remains recorded in the review
artifacts under `docs/plans/artifacts/bayesfilter-tuning-fable-review-r2/` and
`docs/plans/artifacts/bayesfilter-tuning-fable-review-r3/`.

A final direct one-file primary worker attempt was also made with the exact
bounded prompt shape and failed with the same `529 Overloaded` response before
returning a verdict. No primary `VERDICT: AGREE` is claimed.
