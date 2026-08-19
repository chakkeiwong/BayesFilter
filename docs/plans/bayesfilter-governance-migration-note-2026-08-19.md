# BayesFilter Governance Migration Note

Date: 2026-08-19

Commissioned by:
`docs/plans/bayesfilter-docs-governance-cleanup-handoff-2026-08-19.md` (D1).

This note discharges the migration-note obligation in the "Legacy Governance
Migration" section of the current user-level policy. Per that policy, this
simplification does not itself pass through any retired ceremony.

## (a) Which policy generation governs active work

Active work in this repository is governed by, newest first:

1. The user-level global CLAUDE.md policy ("Academic Research Governance And
   Proportionality", "Execution Tiers", "Campaign Repair And Retry", "Review
   Proportionality", "Legacy Governance Migration").
2. The repository `AGENTS.md` ("Academic Research Governance Profile", owner
   directive 2026-07-13), which already states that it supersedes older
   BayesFilter plans, runbooks, and notes where they require stricter
   procedural controls.
3. The repository `CLAUDE.md` (backend, execution-target, chunk, GPU-memory,
   and NeuTra rules). These are scientific/engineering rules, not launch
   ceremony, and are unaffected by this migration.

Earlier lane-scoped migration artifacts remain valid history but are narrower
than this note:

- `docs/plans/bayesfilter-academic-research-governance-simplification-2026-07-13.md`
- `docs/plans/bayesfilter-academic-risk-tier-governance-reset-2026-07-13.md`
- `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-governance-migration-note-2026-07-14.md`
  (Contract E campaign only)

## (b) Which ceremony classes are retired

Retired absent a documented concrete adversarial risk or an explicit user
request:

- hash-bound natural-language approval statements;
- one-use authority, approval-token, and permanent launch-claim files;
- inode/descriptor/hard-link/immutable-empty-file/crash-durable output
  reservation protocols;
- custom cryptographic approval schemas where Git provenance and ordinary
  SHA-256 artifact checksums answer the integrity question;
- separate human approval for each local retry under an unchanged scientific
  contract and campaign budget;
- mandatory review of every proposal, manifest, subplan, result, and handoff.

NOT retired (still binding):

- evidence contracts, vetoes, nonclaims, run manifests, versioned
  never-overwritten output directories, bounded compute/attempt budgets;
- ordinary hash-binding of *artifacts and inputs* (frozen tensors, tuning
  archives, chain prefixes). Most historical "hash-bound" wording in this
  corpus is this legitimate integrity use, not approval ceremony;
- explicit human approval at real boundaries: publication, credentials,
  destructive operations, environment mutation, paid or materially expanded
  compute (e.g. the 544-hour SSL-LSTM ladder), privacy changes, and material
  direction changes.

## (c) Status of historical artifacts using retired ceremony

Historical documents that operate the retired ceremony remain preserved
evidence. Their approval-token, one-use-authority, launch-claim, and
hash-bound-approval gates must not be finished, regenerated, or satisfied.
Reaching one of those gates in an old runbook is not a blocker for new work;
the current execution-tier rules apply instead. Do not delete or rewrite the
historical text; supersession is by banner or by this note.

## (d) Documents identified as carrying retired-ceremony language (searched)

Search: case-insensitive grep over `docs/plans/*.md` for
`approval token|approval-token|approval statement|launch claim|launch token|launch authority|one-use authorit*|authority file|hash-bound`
(2026-08-19). Hits where "hash-bound" binds artifacts rather than approvals
were excluded as legitimate integrity use (examples: the weighted-forward-kl,
pp-ukf continuation, ssl-lstm-q20 seed-b, svx-zc, and ksc-ukf plans, and the
`detached-overnight-supervisor-plan` overlay hashes).

### Operate retired ceremony (gates must not be satisfied or regenerated)

HMC semantic-identity migration lane (the lane whose ceremony prompted the
2026-07-13 simplification; its own Phase 7 academic-campaign subplan/result
performed the lane-local retirement):

- `bayesfilter-hmc-semantic-identity-migration-master-program-2026-07-11.md`
- `bayesfilter-hmc-semantic-identity-migration-phase6-smoke-subplan-2026-07-11.md`
- `bayesfilter-hmc-semantic-identity-migration-phase6-smoke-result-2026-07-11.md`
- `bayesfilter-hmc-semantic-identity-migration-phase7-serious-subplan-2026-07-11.md`
- `bayesfilter-hmc-semantic-identity-migration-phase7-serious-result-2026-07-11.md`
- `bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt1-infrastructure-result-2026-07-13.md`
- `bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt2-preruntime-result-2026-07-13.md`
- `bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt2-runtime-subplan-2026-07-13.md`
- `bayesfilter-hmc-semantic-identity-migration-phase8-closeout-subplan-2026-07-11.md`
- `bayesfilter-hmc-semantic-identity-migration-visible-ledger-2026-07-11.md`

Overnight gated runbook/ledger generation (launch tokens/authority):

- `bayesfilter-highdim-zhao-cui-p30-overnight-gated-self-recovery-runbook-2026-06-05.md`
- `bayesfilter-highdim-zhao-cui-p30-overnight-gated-self-recovery-runbook-claude-review-ledger-2026-06-05.md`
- `bayesfilter-highdim-zhao-cui-p44-overnight-gated-self-recovery-claude-review-ledger-2026-06-07.md`
- `bayesfilter-highdim-zhao-cui-p45-phase0-target-governance-subplan-2026-06-08.md`
- `bayesfilter-highdim-zhao-cui-p58-m9-source-route-pipeline-blocker-audit-repair-plan-2026-06-11.md`
- `bayesfilter-highdim-zhao-cui-p60-visible-gated-execution-runbook-2026-06-12.md`
  (contains an "Initial Launch Token" section)

Complete-highdim-leaderboard generation (hash-bound approval/decision gates,
launch authority):

- `bayesfilter-complete-highdim-leaderboard-visible-gated-execution-runbook-2026-07-11.md`
- `bayesfilter-complete-highdim-leaderboard-visible-execution-ledger-2026-07-11.md`
- `bayesfilter-complete-highdim-leaderboard-phase0-authority-repair-result-2026-07-12.md`
- `bayesfilter-complete-highdim-leaderboard-run-risk-acceptance-amendment-2026-07-12.md`
- `bayesfilter-complete-highdim-leaderboard-local-only-continuation-runbook-2026-07-12.md`
- `bayesfilter-complete-highdim-leaderboard-phase0-boundary-freeze-subplan-2026-07-11.md`

Launch-authority phase-0 pattern and visible ledgers:

- `bayesfilter-actual-sv-single-target-visible-execution-ledger-2026-06-29.md`
- `bayesfilter-generalized-sv-phase0-launch-boundary-freeze-result-2026-06-29.md`
- `bayesfilter-generic-nonlinear-ssm-likelihood-gradient-phase0-launch-result-2026-07-01.md`
- `bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-result-2026-07-11.md`
  (conditions A1 entry on a "fresh hash-bound review" of the exact final
  result — a review-token gate)

The Austria GenUT reset memo carries approval-boundary/blocked-attempt
dialect from the Codex approval-reviewer era (404/502 accounting, per-retry
approval language) alongside still-valid scientific contracts:

- `bayesfilter-austria-genut-neutra-root-cause-reset-memo-2026-08-18.md`
  (already superseded by banner for status; its approval-boundary mechanics
  are historical, its evidence contract and frozen-scope tables remain valid)

### Reference the retired ceremony without operating it (no action needed)

These mention tokens/authority only to disclaim or retire them, or use the
old dialect for a boundary the new policy still requires (material compute):

- `bayesfilter-academic-research-governance-simplification-2026-07-13.md`
- `bayesfilter-academic-risk-tier-governance-reset-2026-07-13.md`
- `bayesfilter-contract-e-canonical-gradient-migration-governance-migration-note-2026-07-14.md`
- `bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-subplan-2026-07-13.md`
- `bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-result-2026-07-13.md`
- `bayesfilter-contract-e-tp-all-model-clean-xla-validation-visible-stop-handoff-2026-07-15.md`
- `bayesfilter-multimodel-neutra-filter-posterior-execution-runbook-2026-07-15.md`
- `bayesfilter-multimodel-neutra-filter-posterior-p0-target-route-freeze-subplan-2026-07-15.md`
- `bayesfilter-neutra-batch-native-training-knowledge-transfer-master-program-2026-07-14.md`
- `bayesfilter-neutra-batch-native-training-knowledge-transfer-master-program-result-2026-07-14.md`
- `bayesfilter-neutra-remaining-models-broad-grid-continuation-plan-2026-07-30.md`
- `bayesfilter-zhao-cui-austria-sir-score-completion-fable-review-request-2026-08-02.md`
- `bayesfilter-actual-sv-overcomplete-analytic-chart-repair-plan-2026-07-17.md`
- `bayesfilter-sgqf-whole-highdim-leaderboard-repair-master-program-2026-07-22.md`
- `bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md` and
  `...-result-2026-07-19.md` ("explicit launch authority" here guards a
  544-hour material-compute run — a boundary the current policy retains; only
  the wording is old dialect)
- `bayesfilter-highdim-nonlinear-filtering-paper-first-scholarship-p32-substantial-scholarly-remediation-plan-2026-06-04.md`
  ("approval statement" here is a scholarly claim target, not launch ceremony)
- `bayesfilter-ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-reset-memo-2026-08-09.md`
  ("Authority files" is a heading listing authoritative documents, not
  ceremony)

### Coverage boundary

The search covered `docs/plans/*.md` filenames matching `bayesfilter*` and
`batched*` (6,109 + 1 files) by pattern only; files whose ceremony language
uses other vocabulary would be missed. `docs/chapters/`, `docs/benchmarks/`,
and non-BayesFilter plans were not searched. Silence about a file is not a
clean bill.

## Concise active campaign statement

The active campaign is the Austria GenUT batch value/score compiler-mode
investigation. Authoritative status is the top status line of
`bayesfilter-austria-genut-neutra-root-cause-execution-checkpoint-2026-08-18.md`;
the authoritative doc chain (newest first) is recorded in
`bayesfilter-docs-governance-cleanup-handoff-2026-08-19.md`. User priority
ruling 2026-08-19: XLA `T=20` nonfiniteness is problem #1; graph value/score
program split is problem #2 (two verified-clean workarounds: eager,
meta-optimizer off). A plain-language user request to execute or resume that
campaign is sufficient authorization; no retired ceremony applies.
