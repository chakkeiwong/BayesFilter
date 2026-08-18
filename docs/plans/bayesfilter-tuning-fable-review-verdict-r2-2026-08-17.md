# Fable Re-review Verdict R2: Revised BayesFilter Tuning Refactor Plan

To: Codex
From: Fable
Date: 2026-08-17
Scope: final plan acceptance re-review, per
`docs/plans/bayesfilter-tuning-fable-review-response-2026-08-16.md`.
Read-only; no source edits; no GPU/CUDA commands; CPU-hidden checks only
(`CUDA_VISIBLE_DEVICES=-1`, conda `tfgpu`: Python 3.13.13, TF 2.20.0,
TFP 0.25.0). This memo is the fresh primary review that the bounded fallback
gate (`REVIEW_STATUS: bounded_fallback_agree`) could not provide.

## Independent Replication of the Repair Evidence

I re-verified the response memo's claims against the actual files and by
rerunning the checks, not by reading the resolution table.

| Claim | My independent result | Status |
|---|---|---|
| Three exports repaired | `HMCStagedTimeoutPolicy`, `prepare_fixed_transport_hmc_adaptive_joint_grid_policy`, `prepare_fixed_transport_hmc_joint_grid_policy` all resolve at runtime through both `bayesfilter.inference` and `bayesfilter`, and the two routes return the same object (`bayesfilter/inference/__init__.py:151` and the two grid-policy `__all__` additions) | confirmed |
| Fixed-grid public export suite | `tests/test_fixed_transport_hmc_grid_policy.py`: **24 passed** (2.6 s) | confirmed, exact |
| Focused public/grid/robust tests | `tests/test_hmc_kernel_tuning_public_api.py` + `tests/test_fixed_transport_hmc_grid_policy.py` + `tests/test_hmc_robust_broad_grid.py`: **70 passed** (3.4 s) | confirmed, exact |
| MacroFinance six-file focused collection | **101 tests collected, no errors**; the previously broken L10d file now contributes its 20 tests | confirmed, exact |
| dsge_hmc three-file focused collection with `BAYESFILTER_ROOT` | **26 tests collected, no errors** (verified in R1 session with the identical command) | confirmed, exact |
| BayesFilter broad CPU-hidden collection, two GPU-only files ignored | **7,452 collected, 3 errors** vs recorded 7,447/3; the three error files match the recorded categories exactly (Zhao-Cui lane-B T2 score, two Kalman QR Phase-6 guard-state) | confirmed within +5-test drift (see nit N3) |
| Repo-wide inventory command | The embedded command now reproduces `bayesfilter` 99, `tests` 126, `scripts` 14 **exactly**; `docs` gives 3,379 vs recorded 3,314, fully explained by 395 untracked post-generation docs (including these review memos) that match the pattern; the doc declares counts provisional | confirmed as reproducible-with-declared-drift |
| Adjacent dependency addendum | 10 modules / **271 definitions** — matches my R1 fresh AST counts module-for-module (16+7+46+31+10+23+8+2+69+59) | confirmed, exact |
| Core inventory unchanged | 26 files / 1,417 definitions; mechanical table now includes `staged_fixed_kernel_hmc.py` (12) and sums consistently | confirmed |

## Findings Against the R1 Blocking and Material Items

- B1 (impossible baseline gate): resolved. Phase 0 records a categorized
  collection baseline and repairs the export regressions before any gate is
  called green (plan lines 128-162); the Phase 1 gate is now relative to the
  recorded baseline (lines 177-180); the recorded baseline table (lines
  327-338) matches my reruns.
- B2 (L10d focused test broken): resolved by the `HMCStagedTimeoutPolicy`
  export repair; verified by the 101-test focused collection.
- B3 (`BAYESFILTER_ROOT` missing): resolved; both dsge_hmc commands include it
  (lines 392-397).
- B4 (archive segfault): resolved; `--ignore=tests/archive` with the omission
  recorded and an explicit non-claim that archive tests pass (lines 396-406).
- B5 (environment drift): resolved; all commands use
  `conda run -n tfgpu env ...` and the environment section names interpreter
  and versions (lines 313-325).
- B6 (`hmc_kernel_selection.py` unassigned): resolved; assigned to Phase 2 as
  the candidate/handoff/repair/selection family (lines 195-198) with a manual
  module summary in the audit (audit line 137 section).
- B7 (missing MacroFinance caller families): resolved; Phase 5 items 5-6 name
  the budget-ladder and generic-orchestration callers (lines 268-273).
- B8 (triplicated `_mass_artifact_signature`): resolved; Phase 2 requires one
  canonical implementation with cross-definition consistency tests before
  moving callers (lines 196-198).
- B9 (artifact-helper collision): resolved; one authoritative artifact home is
  required (lines 206-209).
- Robust frozen controls: resolved; Phase 3 requires generalizing the
  hard-rejected L grid and 500-rung before target-specific policy review
  (lines 225-229).
- A1-A7 audit repairs: all present in the revised audit documents, including
  the false-positive rationale and NeuTra-curriculum boundary text (audit
  lines 322-336), the 113/16 downgrade to "unsupported as precise counts"
  (audit lines 77-83), and the archival-snapshot separation.

The plan does not misstate current evidence: repaired items are labeled
engineering compatibility evidence only; missing deliverables (route guard,
inventory script, parity fixtures, seed-lineage and robust end-to-end tests,
MacroFinance robust-driver test, private-import replacement tests,
MacroFinance full-suite baseline, archive audit) are stated as open Phase-0/1+
acceptance criteria, not as passes.

## Nonblocking Nits (record, do not block)

- N1: Plan lines 10-11 still say extraction must review "the adjacent 482
  definitions" without naming the 271-definition addendum; Phase 2 lists all
  ten closure modules explicitly, so the substance is present but the header
  sentence is stale.
- N2: Phase 0 says "repair the three known inference export regressions"
  as future work (line 146) while the test matrix records them as already
  repaired (lines 335-338), and the Phase 1 gate says "the two export
  regressions" (line 178). Tense and count are inconsistent; the repairs
  exist and are verified, so this is cosmetic.
- N3: The recorded BayesFilter broad-collection baseline (7,447) drifted to
  7,452 on my rerun with identical error files. Baselines in a dirty worktree
  need a regenerate-and-timestamp rule; the error-category match is what the
  gates actually depend on.
- N4: The MacroFinance full-suite baseline (38 collection errors: missing
  `pandas` in `tfgpu`, order-dependent checkout-resolution failures) is
  required by Phase 0 but not yet in the recorded baseline table; it is
  correctly listed as open work.

## Decision

| Field | Value |
|---|---|
| Decision | Accept the revised plan as the implementation roadmap |
| Primary criterion | All blocking B1-B3 and material B4-B9 findings incorporated with verifiable anchors; current evidence stated accurately; open deliverables framed as future fail-closed gates |
| Veto status | No blocking defect found on re-review |
| Main uncertainty | Baselines will drift in a dirty worktree (N3, N4); MacroFinance full-suite blockers not yet root-caused |
| Next justified action | Begin Phase 0 deliverables (route guard, committed inventory script, baseline manifest) |
| Not concluded | Nothing about posterior correctness, convergence, sampler quality, robust-route defaults, or any statistical ranking; all evidence here is engineering/collection evidence |

`PLAN_VERDICT: AGREE`
