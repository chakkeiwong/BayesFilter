# BayesFilter Docs Red-Team Contradictions Ledger

Date: 2026-08-19

Commissioned by:
`docs/plans/bayesfilter-docs-governance-cleanup-handoff-2026-08-19.md` (D3–D6).
Companion: `docs/plans/bayesfilter-governance-migration-note-2026-08-19.md` (D1).

## Coverage Boundary — Read First

Reviewed: the BayesFilter documents in `docs/plans/` belonging to the Austria
GenUT / NeuTra value-score lineage (2026-07-30 through 2026-08-19) in full,
plus grep-driven sweeps over all `docs/plans/*.md` for the D1 and D4 patterns
listed below. NOT reviewed: the remaining ~6,000 plan/result/memo files in
`docs/plans/` (the handoff's estimate of "dozens" of files is wrong relative
to that claim — the directory holds 6,200 `.md` files, 6,109 with the
`bayesfilter-` prefix), `docs/chapters/`, `docs/benchmarks/` runners,
artifacts JSON, and all non-BayesFilter or non-Austria campaign lineages
(SSL-LSTM, DPF/FilterFlow, macrofinance, weighted-forward-KL, SQMC streaming,
monograph, leaderboards) except where a grep hit forced a look. Silence about
any unreviewed file is not evidence it is current.

Grep patterns used (case-insensitive): `higher-moment correction`,
`localizes`, `only remaining gate`, `attempt06`, `682.37`, `cross-mode`, and
the D1 ceremony vocabulary recorded in the migration note.

## D3 Errata: Attempt-Numbering Collision

The names `repair_validation_attempt07/08/09` were used twice with disjoint
referents:

1. **Never-created Codex proposals (2026-08-17/18).** Historical prose from
   Codex sessions — preserved in
   `bayesfilter-austria-genut-neutra-root-cause-reset-memo-2026-08-18.md`
   ("Some prose notes call those proposed attempts 07 through 09") — refers
   to bounded trusted-launch *requests* that were rejected at the approval
   boundary (HTTP 502 / approval-model 404) before process creation. No
   directory, JSON, or GPU compute ever existed for them.
2. **Real artifacts (2026-08-18/19).** The directories under
   `docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/`
   were then actually created with entirely different content:
   `repair_validation_attempt07/endpoint_gpu0_eager.json` (eager PASS),
   `repair_validation_attempt08/endpoint_gpu0_graph.json` (graph within-mode
   identity FAIL, gap `0.562`), and
   `repair_validation_attempt09/endpoint_gpu0_xla.json` (XLA `T=20`
   nonfinite/invalid, hard veto).

Rule (already recorded in the reset memo, restated here): **attempt numbering
is an artifact-path convention, not evidence that a process ran.** Eligibility
is per JSON file and its recorded `source_sha256`, never per directory name.
A reader cross-referencing pre-2026-08-18 prose "attempts 07–09" against the
artifact tree must treat the prose as referring to the rejected proposals,
not these files. Separately, `attempt06`-named paths exist in several
*unrelated* campaign artifact roots (`cubature_genut_gpu_xla_*_20260721`,
`genut-sir-ad-root-cause-20260817`, `genut_score_variance_*`, and others);
attempt numbers are only meaningful within one campaign's artifact root.

## Findings

Classifications use: `unsupported` / `stale` / `contradicts <file>` /
`mismatch with code` / `not checked`. Dispositions: banner added / correction
added / already corrected at origin / no action needed / proposal-only
(crosses the handoff's scope boundary) / ledger-only.

### F1 — falsified localization claim ("four-step higher-moment correction")

| File | Claim | Classification | Evidence | Disposition |
|---|---|---|---|---|
| `...root-cause-execution-result-2026-08-18.md` | graph divergence localizes to the higher-moment correction, not the upstream recursion | wrong relative to that claim | `T=3, steps=0` reproduction in `graph_bisect_attempt02` | already corrected at origin (2026-08-19 banner present); no further action |
| `...root-cause-execution-checkpoint-2026-08-18.md` (mid-file, "Graph T=1 zero-correction control ... localizing the divergence to the four-step higher-moment correction") | same | wrong relative to that claim | same | inline dated correction added 2026-08-19, referencing the localization result |
| `...graph-mode-divergence-localization-{plan,result}` | — | — | plan predates the finding and makes no localization claim; result is the correcting document | no action needed |
| All other `higher-moment correction` grep hits (29 files) | describe the correction algorithm itself, not the graph-divergence localization | — | inspected hit context | no action needed |

### F2 — stale "only remaining gate" prose

| File | Claim | Classification | Evidence | Disposition |
|---|---|---|---|---|
| `...root-cause-execution-checkpoint-2026-08-18.md` line ~57 | "The only remaining gate is a trusted RTX 5080 endpoint run" | stale (true when written) | eager/graph/XLA endpoint runs completed 2026-08-18/19; XLA hard veto now blocks | dated correction added 2026-08-19 pointing to the top status line and localization result |
| Other grep hits of `only remaining` | unrelated lineages (zhao-cui t1 plan, ssl-lstm budget) | — | context inspected | no action needed |
| `...hypotheses-fable-handoff-2026-08-17.md` status line `FABLE_AGREED_EXECUTION_IN_PROGRESS_GPU_CONFIRMATION_PENDING` | execution in progress, GPU confirmation pending | stale | campaign completed the confirmation; graph/XLA failures recorded | **proposal-only**: file is in the handoff's protected campaign glob and not named by a defect. Proposed banner: "Superseded for status by the execution checkpoint status line; plan content remains the reviewed basis of the executed campaign." |
| `...hypotheses-fable-{audit-reply,second-review-request,second-review-reply}-2026-08-17.md` | review-time statuses | stale (status lines only; review content historical) | same | **proposal-only** (same protected glob); a one-line status banner each would suffice |

### F3 — stale attempt06 three-mode artifact must not be upgraded

Verified at every `attempt06` grep hit in the Austria lineage: the reset memo
(eligibility table marks it "Stale for current source and fails current
cross-mode confirmation rule"), the checkpoint (GPU Resume Audit marks it
stale by source-hash mismatch), the execution result ("historical stale
attempt06"), and the localization plan (Q4: `606897...` matches no committed
state; empirical localization required) and result (explicitly refuses to use
it). **No document upgrades attempt06 into current evidence. No action
needed.** Non-Austria `attempt06` hits are unrelated artifact namespaces (see
D3 errata).

### F4 — historical finite XLA value vs current XLA NaN

`-682.3775024` appears in the Austria lineage only in the reset memo's
clearly-labeled historical table and in the localization result, which
explicitly states the old finite value is from the stale source and that the
nonfinite outcome is current-source/current-XLA-specific, cause `not checked`.
The remaining `682.37` grep hits are numeric coincidences inside DPF JSON
dumps from June 2026. **No document treats the old finite value as evidence
against the current NaN observation. No action needed.**

### F5 — `cross-mode` sweep

184 files contain "cross-mode"; the Austria-lineage hits were inspected. The
reset memo, checkpoint, execution result, and localization plan/result use it
consistently: cross-mode drift is a confirmation veto for the historical
attempt06 source, and the current campaign stopped before any cross-mode
gate because within-mode identity failed first. Non-Austria hits (leaderboard,
readiness, dual-cap files) predate this campaign and use "cross-mode" for
their own scopes. **No contradiction found in inspected files; the ~155
uninspected non-Austria hits are `not checked`.**

### F6 — pre-repair score-instability documents (superseded silently until now)

The 2026-08-18 repair (shared primal + forward-autodiff public score; the
manual recursive score classified wrong relative to the complete finite value
program) materially supersedes the premise of several 2026-08-03..05
documents. Banners added 2026-08-19:

| File | Superseded claim | Disposition |
|---|---|---|
| `bayesfilter-genut-four-model-neutra-readiness-result-2026-08-04.md` | Austria blocked by tangent-free/tangent-carrying mismatch | banner added |
| `bayesfilter-genut-four-model-neutra-readiness-reset-memo-2026-08-04.md` | Austria repair-lane instructions (pairwise port as next step) | banner added |
| `bayesfilter-austria-genut-neutra-value-surrogate-strategy-2026-08-03.md` | strategy motivated by the unstable recursive score | partial-supersession banner added (math preserved) |
| `bayesfilter-austria-genut-neutra-zero-force-smoke-result-2026-08-04.md` | endpoint irreproducibility on pre-repair source | banner added |
| `bayesfilter-austria-genut-pairwise-verdict-reassessment-2026-08-05.md` | pairwise stabilizes "the score" (measured on the old recursive score) | banner added (warm-start status) |
| `bayesfilter-austria-sir-pairwise-moment-genut-score-trial-result-2026-07-30.md` | same premise | left unmarked: already carries the 2026-08-05 addendum banner chaining to the reassessment, which now carries the 2026-08-19 banner; a second banner would stack |
| `bayesfilter-austria-sir-pairwise-moment-genut-score-trial-{plan,reset-memo}-2026-07-30.md` | same premise | left unmarked: reachable through the result's banner chain |

### F7 — ambiguous cases (report, not banners)

| File | Issue | Classification | Why no banner |
|---|---|---|---|
| `bayesfilter-genut-score-computation-audit-result-2026-07-30.md` | concludes "derivative mechanics correct" for a manual recursive JVP; the 2026-08-18 memo says the previous hand-coded recursive score was wrong relative to the complete finite value program | not checked | the audit's object (staged scalar candidate at commit `fb9a0679`) and the memo's object (batch route in `cubature_genut_batch_tf.py`) may be different programs; deciding requires code archaeology outside this cleanup's scope |
| `bayesfilter-genut-score-audit-followup-handoff-2026-07-30.md` | open work items possibly overtaken by the repair | not checked | depends on F7 row 1 |
| `bayesfilter-genut-score-estimator-options-mathematical-note-2026-07-31.md` | variance analysis premised on the recursive estimator | not checked | the mathematics may survive estimator replacement; per-claim triage not done |
| `bayesfilter-genut-sir-*-2026-08-17.md` (ad-root-cause, j0, hypothesis, sqmc trust-region) | scalar-route verdicts adjacent to the batch campaign | not checked | different route (scalar vs batch); no observed contradiction, not swept in depth |
| `bayesfilter-genut-dual-cap-*`, `bayesfilter-zhao-cui-genut-*` dual-cap chain | route promoted before the compiler-mode findings | not checked | dual-cap is explicitly a nonclaim of the current campaign in both directions; no document inspected claims dual-cap is affected |

### F8 — governance dialect (D1)

See the migration note for the full searched list. Ledger summary: 39
files carry ceremony vocabulary; ~21 operate retired ceremony (HMC
semantic-identity lane, overnight gated runbooks, complete-highdim-leaderboard
generation, phase-0 launch-authority pattern, and the 2026-08-18 reset memo's
approval-boundary accounting); the rest reference it only to retire or
disclaim it, or use old dialect for boundaries the new policy retains.
Disposition: recorded in the migration note; **no banners added to
ceremony-operating files** — the migration note supersedes their gates
wholesale, and per-file banners on ~21 historical documents would be rewrite-
scale churn the handoff's method notes discourage. Classified as covered by
ledger + migration note.

## D6 Standing Known Issues (do not rediscover)

1. **Sylvester op ABI failure.**
   `bayesfilter/ops/_symmetric_sylvester_ops.so` has an undefined
   TensorFlow/Abseil ABI symbol; target-factory tests fail on Austria
   *construction* for environment reasons unrelated to the GenUT campaign.
   No rebuild is authorized. Do not attribute this loader failure to the
   GenUT repair (recorded in the 2026-08-18 reset memo, "Known Unrelated Test
   Environment Failure").
2. **attempt06 is the only complete three-mode GPU run and is ineligible.**
   `repair_validation_attempt06/endpoint_modes_result.json` was produced from
   source `606897...` (matches no committed state; uncommitted session
   intermediate), shows historical cross-mode value drift, and fails the
   current confirmation rule. It must not be rerun in place or reinterpreted
   as current evidence.
3. **`T=1` zero-correction controls are too weak** to detect recursion-lane
   compiler divergence; FP32 visibility requires `T>=3` (localization result,
   F1). Any future endpoint-control design that uses a `T=1` control as its
   only upstream guard inherits this blind spot — surface this wherever that
   design is described. The checkpoint's corrected localization paragraph now
   carries the pointer.

## Acceptance Cross-Check

- D1: migration note written with searched file list — done.
- D2: Austria/GenUT lineage swept; banners added (F6) or reasons recorded
  (F2 proposal-only rows, F6 chain rows, F7 ambiguous rows). The rest of the
  6,109-file corpus is explicitly out of coverage (see boundary above).
- D3: errata above.
- D4: F1–F5 list every hit class and disposition; the two known-wrong claims
  are corrected at every found occurrence.
- D6: three items recorded above.
- Scope: edits confined to `docs/plans/*.md` banners/corrections plus this
  file and the migration note; protected campaign files edited only where a
  defect named them (the checkpoint, D4).
