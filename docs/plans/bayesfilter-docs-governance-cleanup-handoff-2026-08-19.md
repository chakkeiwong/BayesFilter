# BayesFilter Documentation And Governance Cleanup Handoff

Date: 2026-08-19

Audience: a Fable-class Claude Code agent assigned to documentation and
governance hygiene for `/home/chakwong/BayesFilter`.

Origin: on 2026-08-19 the user reviewed an audit of this repository's
agent-governance corpus and commissioned this cleanup. The audit findings are
reproduced in full below so no context from that conversation is lost.

## Hard Scope Boundary — Read First

This handoff is **documentation and governance hygiene only**. A separate,
active debugging campaign (Austria GenUT XLA nonfiniteness and graph-mode
value/score divergence) is running in another session. You must NOT:

- run any GPU, TensorFlow, benchmark, or endpoint command;
- edit any production source under `bayesfilter/`;
- edit any diagnostic runner under `docs/benchmarks/`;
- create, modify, or delete anything under `docs/benchmarks/artifacts/`;
- delete or rewrite historical text in any plan/result/checkpoint document —
  supersession is by appended banner or errata entry, never by removal;
- reset, stash, checkout, or reformat the worktree. It is dirty with edits
  from multiple agents; that is deliberate. Commit nothing unless the user
  approves.
- touch these actively-evolving campaign files except where a defect below
  explicitly names them:
  `docs/plans/bayesfilter-austria-genut-neutra-root-cause-*`,
  `docs/plans/bayesfilter-austria-genut-graph-mode-divergence-*`.

If a cleanup action would require crossing any of these lines, record it in
your report as a proposal instead of doing it.

## Authoritative State Snapshot (2026-08-19)

Preserve this; future readers may reach this memo before newer notes.

- Git commit at handoff: `dae37183bf4421682b2ad991e2dc0d0f3c53f260`
  (worktree dirty, multi-agent).
- Active campaign: Austria GenUT batch value/score compiler-mode
  confirmation. Current authoritative status (recorded at top of the
  execution checkpoint):
  `GRAPH_INEQUALITY_GRAPPLER_CONSISTENT_METAOFF_BITWISE_RESTORED_XLA_T20_NONFINITE_INVALID_HARD_VETO`
- Established, per-mode, frozen Austria scope (`T=20`, `N=1008`, FP32, TF32,
  RTX 5080, source `cubature_genut_batch_tf.py` = `ae8cbfb...a976e`):
  - CPU: value and score exactly equal to independent forward autodiff
    (`attempt04/derivative_cpu.json`).
  - GPU eager: exact within-mode value identity
    (`repair_validation_attempt07/endpoint_gpu0_eager.json`).
  - GPU graph (default grappler): within-mode identity FAILS; gap `0.562`,
    score sign-flipped vs eager
    (`repair_validation_attempt08/endpoint_gpu0_graph.json`); reproduces down
    to `T=2,steps=1` and `T=3,steps=0`; bitwise-restored in 4/4 probed cases
    by `disable_meta_optimizer=True` (`graph_bisect_attempt01..03/`).
  - GPU XLA: `T=20` nonfinite value AND score, `program_valid=false`, hard
    veto (`repair_validation_attempt09/endpoint_gpu0_xla.json`).
- User priority ruling (2026-08-19): the XLA NaN is problem #1 (validity
  failure of the repository default target, no workaround); the graph
  value/score program split is problem #2 (has two verified-clean
  workarounds: eager, meta-off). Do not re-litigate this ranking.
- Authoritative doc chain, newest first:
  1. `docs/plans/bayesfilter-austria-genut-graph-mode-divergence-localization-result-2026-08-19.md`
  2. `docs/plans/bayesfilter-austria-genut-graph-mode-divergence-localization-plan-2026-08-18.md`
  3. `docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-checkpoint-2026-08-18.md`
     (status line current; interior sections are layered history)
  4. `docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-result-2026-08-18.md`
     (contains a 2026-08-19 correction banner)
  5. `docs/plans/bayesfilter-austria-genut-neutra-root-cause-reset-memo-2026-08-18.md`
     (superseded; carries a supersession banner)
- Everything remains blocked downstream: NeuTra training, HMC, tuning,
  cross-model regression, dual-cap, default promotion.

## Defects To Clean Up

Work these in order. For each, the deliverable is listed; the collective
acceptance criteria are in the final section.

### D1. Governance dialect contradiction (migration note owed)

The newest global policy (user-level CLAUDE.md, "Academic Research
Governance And Proportionality" and "Legacy Governance Migration") retires
approval-token ceremony: hash-bound approval statements, one-use authority
files, per-retry human approval under an unchanged contract, launch-claim
mechanics. Older project artifacts still speak that dialect — e.g. the reset
memo's blocked-attempt accounting and approval-boundary language, and any
older plans that condition execution on approval tokens or launch claims.

The policy itself prescribes the remedy: "Write a short migration note and a
concise active campaign plan. Do not make the simplification itself pass
through the retired ceremony." That migration note has never been written.

Deliverable: `docs/plans/bayesfilter-governance-migration-note-2026-08-19.md`
stating (a) which policy generation governs active work, (b) which ceremony
classes are retired, (c) that historical artifacts using retired ceremony
remain preserved evidence but their gates must not be finished, regenerated,
or satisfied, and (d) a list of the specific documents you identified as
carrying retired-ceremony language (search, do not guess).

### D2. Inconsistent staleness marking across docs/plans generations

Only the newest documents received supersession banners (the reset memo and
the execution result). Older generations in the same lineage — earlier reset
memos, result notes, and checkpoints from prior phases of this campaign and
adjacent ones — were superseded silently. A reader sampling the directory
cannot tell current from stale without reading everything.

Deliverable: sweep `docs/plans/` (there are on the order of dozens of
BayesFilter plan/result/memo files). For each file in a lineage where a newer
document supersedes it materially, add a short banner at the top:
date, "superseded by <path> for <which claims>", nothing else changed.
Ambiguous cases go in the report, not into banners.

### D3. Attempt-numbering collision (errata note owed)

Historical prose from Codex sessions refers to "proposed attempts 07 through
09" that were rejected at the approval boundary — no directories or JSON were
ever created. On 2026-08-18/19 the real directories
`repair_validation_attempt07/` (eager PASS), `attempt08/` (graph FAIL), and
`attempt09/` (XLA NaN) were then actually created with entirely different
content. Any future reader cross-referencing old prose against the artifact
tree will mis-associate rejected launch proposals with completed runs.

Deliverable: a short errata section (inside the D5 ledger or a standalone
note) stating the collision explicitly: which prose mentions refer to
never-created Codex proposals, which directories now hold real evidence, and
the rule already recorded in the reset memo ("attempt numbering is an
artifact-path convention, not evidence that a process ran").

### D4. Corrected-claim propagation

Two claims are now known wrong and were corrected at their origin, but may be
repeated elsewhere:

1. "The graph divergence localizes to the four-step higher-moment
   correction" — falsified by `T=3, steps=0` reproduction (correction banner
   already present in the execution result; the checkpoint's 2026-08-19
   section records it).
2. "The only remaining gate is a trusted RTX 5080 endpoint run" and
   equivalents — true when written, now stale in mid-file prose of the
   checkpoint and possibly other documents.

Also verify no document upgraded the stale `repair_validation_attempt06`
three-mode artifact into current evidence, and none treats the historical
finite XLA value (`-682.3775024`, old source) as evidence against the current
NaN observation.

Deliverable: grep-driven sweep (suggested patterns: "higher-moment
correction", "localizes", "only remaining gate", "attempt06", "682.37",
"cross-mode") plus corrections — banner or inline dated correction referencing
the localization result note. List every hit and disposition in the report.

### D5. Red-team contradictions ledger (policy-mandated, never produced)

The standing policy requires periodic red-teaming of AI-generated docs for
unsupported claims, stale assumptions, and mismatch with current code. It has
not been done for this corpus.

Deliverable:
`docs/plans/bayesfilter-docs-redteam-ledger-2026-08-19.md` with one row per
finding: file, claim, classification (`unsupported` / `stale` /
`contradicts <file>` / `mismatch with code`), evidence, and disposition
(banner added / correction added / proposal-only because it crosses the scope
boundary). Include the D3 errata and D4 results. Cap the sweep at the
BayesFilter documents in `docs/plans/`; note explicitly what you did NOT
review (e.g. `docs/chapters/`, other projects' plans) so silence is not read
as coverage.

### D6. Known-issues capture (do not lose these)

Ensure the ledger records these standing items so they are not rediscovered:

- `bayesfilter/ops/_symmetric_sylvester_ops.so` has an undefined
  TensorFlow/Abseil ABI symbol; target-factory tests fail on Austria
  construction for environment reasons unrelated to the GenUT campaign. No
  rebuild is authorized.
- The stale `repair_validation_attempt06` GPU artifact is the only complete
  three-mode run and is ineligible for the current source (hash mismatch,
  historical cross-mode drift).
- `T=1` zero-correction controls are too weak to detect recursion-lane
  compiler divergence (needs `T>=3` at FP32 visibility) — a diagnostic-design
  lesson recorded in the localization result; worth surfacing wherever the
  endpoint-control design is described.

## Method Notes

- Read before editing; every banner must name its superseding document
  precisely.
- Prefer many small, single-purpose edits over rewrites. `git diff` of your
  session should show only added banners/sections and the two new files.
- When old text and new evidence conflict and the resolution is not obvious
  from the documents alone, classify as `not checked` in the ledger rather
  than guessing.
- Plain-language discipline applies: use `stale`, `wrong relative to that
  claim`, `unsupported`, `not checked` — no softening.

## Acceptance Criteria

1. Migration note exists and lists actual (searched) retired-ceremony
   documents.
2. Every materially superseded BayesFilter plan/memo/result in `docs/plans/`
   carries a dated supersession banner or appears in the ledger with a reason
   it was left unmarked.
3. The attempt-numbering errata exists.
4. All D4 grep hits are listed with dispositions; no known-wrong claim
   remains unmarked at any occurrence.
5. The red-team ledger exists, states its coverage boundary, and contains the
   D6 items.
6. `git status` shows changes ONLY to `docs/plans/*.md` (banners/corrections)
   plus the two new files. Anything else is a scope violation.
7. A final report to the user: what was marked, what was proposed-only, what
   remains ambiguous.
