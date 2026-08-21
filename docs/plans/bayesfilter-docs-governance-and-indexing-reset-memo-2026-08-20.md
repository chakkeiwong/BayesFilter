# Reset Memo: Docs Governance Cleanup, Plans Indexing, And Tracking Boundary

Date: 2026-08-20
Status: `SESSION_CLOSE_DOCS_GOVERNANCE_AND_INDEXING_COMPLETE`

## Context

Three work streams closed at this restart boundary, all documentation and
repository hygiene — no production numerical code was edited by this lane:

1. The 2026-08-19 docs/governance cleanup commissioned by
   `docs/plans/bayesfilter-docs-governance-cleanup-handoff-2026-08-19.md`.
2. Tier-1 indexing infrastructure for `docs/plans/` (6,200+ documents).
3. A tracked-or-ignored boundary for the whole worktree, committed to main.

Separately and concurrently, another session ran the Austria GenUT XLA NaN
localization campaign (plan 2026-08-19, result 2026-08-20) and the
score-recursion while-loop audit/repair plan. Their documents and receipts
are included in this commit as evidence; their scientific content is NOT
re-interpreted here — see their own result notes.

## Decision / policy

Future sessions should assume and not re-litigate:

- **Governance dialect.** The migration note
  `docs/plans/bayesfilter-governance-migration-note-2026-08-19.md` is the
  authority on retired ceremony (approval tokens, one-use authority,
  hash-bound approval statements, launch claims). Historical gates using
  that dialect are preserved evidence and must not be finished, regenerated,
  or satisfied.
- **Red-team ledger.** `docs/plans/bayesfilter-docs-redteam-ledger-2026-08-19.md`
  records the attempt-numbering errata (proposed-but-never-created Codex
  attempts 07–09 vs the real 2026-08-18/19 artifacts), the corrected-claim
  sweep, the standing known issues (Sylvester `.so` ABI failure; stale
  `repair_validation_attempt06`; `T=1` controls too weak, need `T>=3`), and
  its own coverage boundary. Findings marked `not checked` there remain open.
- **Conventions.** New `docs/plans/` documents follow
  `docs/plans/CONVENTIONS.md`: kebab-case filename with role suffix and
  date, `Date:`/`Status:` header, append-only supersession/correction
  banners, no inlined bulk data, regenerate the index after changes. The
  three templates under `docs/plans/templates/` carry the supersession
  checklist.
- **Indexes are generated, not evidence.** `docs/plans/INDEX.md` and
  `INDEX-FULL.md` are gitignored navigation aids; rebuild with
  `python docs/plans/generate_plans_index.py` (~2 s, deterministic, writes
  only those two files). Never hand-edit them; never cite them as evidence.
- **Tracking boundary.** Every file in the worktree is now either tracked or
  gitignored. Generated files are ignored EXCEPT promotion/claim/veto-
  supporting receipts, which are re-included by narrow `!` rules — the
  2026-08-19/20 additions retain the graph-bisect (3), XLA hard-veto
  endpoint (1), and XLA-NaN campaign (5) JSON receipts under
  `docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/`.

## What changed

- File: `.gitignore`
  - Ignored `docs/plans/INDEX.md`, `docs/plans/INDEX-FULL.md` (regenerable).
  - Re-included the nine claim/veto receipts of the graph-mode and XLA-NaN
    campaigns (narrow per-file `!` rules, matching the existing pattern).
- File: `docs/plans/generate_plans_index.py` (new)
  - Stdlib-only index generator: lineage grouping (date/role-suffix
    stripping), family rollup (325+ families), `Status:` extraction,
    supersession/correction flags.
- File: `docs/plans/CONVENTIONS.md` (new) — see policy above.
- Files: `docs/plans/templates/{experiment-plan,experiment-result,reset-memo}-template.md`
  - `Date:`/`Status:` header; supersession checklists on result and memo.
- Files: cleanup edits from 2026-08-19 (all additive banners/corrections):
  - checkpoint (stale-gate correction + falsified-localization correction),
  - execution result (banner was already present at origin),
  - five silently-superseded 2026-08-03..05 Austria/GenUT docs (banners),
  - new: migration note, red-team ledger, cleanup handoff (commissioning
    document, preserved).
- Files from the concurrent campaigns (committed as-is, not authored here):
  - `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py` (while-loop
    repair of the O(N^2) backward-marks block loop),
  - `docs/benchmarks/run_genut_austria_xla_nan_localization_20260819.py`,
  - XLA-NaN plan/result, graph-mode localization result, while-loop repair
    plan, SQMC streaming memo/result updates, and the nine receipts.

## Bugs / blockers resolved

- Symptom: harness permission-classifier outages (repeated
  "claude-sonnet-5 temporarily unavailable" Bash rejections) during index
  generation on 2026-08-19/20.
- Root cause: transient harness-side classifier failure, not repository
  state.
- Resolution: retried; no repository impact. Recorded because the XLA-NaN
  campaign log also mentions ~14 rejected launches from the same outage —
  treat such rejections as infrastructure, never as scientific attempts.

## Verification already run

```bash
python docs/plans/generate_plans_index.py   # 6,204 files indexed cleanly
git check-ignore docs/plans/INDEX.md docs/plans/INDEX-FULL.md   # ignored
git ls-files --others --exclude-standard    # empty after commit
git status --short                          # clean after commit
```

Observed:
- Untracked-unignored set reduced to exactly the intended tracked additions
  before commit; empty after.
- Ignored-but-retained receipts resolve to the nine named JSON files only.

## Current policy

- Repository state: committed to `main` and pushed at this boundary; the
  worktree is clean. The multi-agent dirty-worktree caveat in the 2026-08-18
  reset memo is discharged — future sessions start from committed state.
- The Austria campaign's scientific status is owned by
  `docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-checkpoint-2026-08-18.md`
  (top status line) and the XLA-NaN result. As of this memo the checkpoint
  reads `XLA_T20_NAN_LOCALIZED_TF32_SEEDED_STAGE_D_BLOWUP_GUARD_CORRECT_TF32OFF_XLA_FINITE`.

## Known limitations / cautions

- The index generator's lineage grouping is heuristic (~2,500 singletons);
  INDEX headers say so. Only ~38% of historical files expose a parseable
  `Status:` line; that ratio improves only as new files follow CONVENTIONS.
- CONVENTIONS is convention, not enforcement. A lint mode
  (`--check` on the generator or a pre-commit hook) was deliberately NOT
  added — that gating decision is the owner's.
- The red-team ledger's `not checked` rows (notably whether the 2026-07-30
  scalar-route score audit and the 2026-08-18 batch-route "recursive score
  wrong" finding concern the same program) remain open.
- The `.gitignore` receipt re-inclusion list is per-file by design; new
  campaign receipts need their own `!` rules or they stay ignored.

## Suggested next steps

1. Owner decision owed (from the XLA-NaN result): choose among Stage D
   numerical hardening / TF32-off for claim-bearing Austria XLA / scope
   exclusion. Option 1 is the only mechanism-addressing one and requires a
   reviewed production-source plan.
2. Owner decision owed (from the graph-mode result): whether non-XLA graph
   mode remains a claim-bearing confirmation arm given eager passes and
   meta-optimizer-off restores bitwise identity.
3. Optional docs follow-ups: status banners for the four protected
   `hypotheses-fable-*-2026-08-17` files (proposal in the red-team ledger,
   F2); index `--check` lint mode if enforcement is wanted.

## Supersession checklist (see docs/plans/CONVENTIONS.md)

- [x] Documents this memo materially supersedes are listed here by exact path:
  - none — this memo records a restart boundary; the 2026-08-19 cleanup's
    banners were applied in that session and are listed in
    `docs/plans/bayesfilter-docs-redteam-ledger-2026-08-19.md`.
- [x] Each listed document received a dated supersession banner (n/a).
- [x] `python docs/plans/generate_plans_index.py` was rerun.
