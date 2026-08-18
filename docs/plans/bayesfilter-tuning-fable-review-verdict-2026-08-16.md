# Fable Review Verdict: BayesFilter Tuning Audit and Refactor Plan

To: Codex
From: Fable
Date: 2026-08-16
Review mode honored: read-only audit. No source files were edited, no
experiments or GPU/CUDA commands were launched, and all pre-existing worktree
changes were preserved. All verification scripts were written under
`/tmp/fable_review/` only.

Reviewed targets:

1. `docs/audits/bayesfilter-tuning-function-audit-2026-08-16.md`
2. `docs/audits/bayesfilter-tuning-definition-inventory-2026-08-16.md`
3. `docs/audits/bayesfilter-tuning-adjacent-authority-inventory-2026-08-16.md`
4. `docs/audits/bayesfilter-tuning-repo-wide-reference-inventory-2026-08-16.md`
5. `docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`
6. `bayesfilter/inference/hmc_robust_broad_grid.py`

## Run Manifest

| Field | Value |
|---|---|
| BayesFilter commit | `3030d86df9cb00346df82c7c19f015c09c7c6e1f` (worktree intentionally dirty; preserved) |
| Environment | conda `tfgpu`: Python 3.13.13, TF 2.20.0, TFP 0.25.0; bare `python` resolves to base anaconda 3.11.7 **without TensorFlow** |
| CPU/GPU | All pytest/import commands run with `CUDA_VISIBLE_DEVICES=-1` (GPU intentionally hidden); no GPU framework initialization by this review |
| Commands | `python -m compileall -q bayesfilter tests` (clean, 2.6 s); `pytest --collect-only -q` in BayesFilter, MacroFinance, dsge_hmc; fresh `ast` inventory and import-closure scripts; `rg` symbol scans with and without `--no-ignore` |
| Data version | N/A (no data runs) |
| Seeds | N/A (no stochastic runs) |
| Artifacts | This memo; scratch scripts in `/tmp/fable_review/` (not repo artifacts) |
| Plan file | `docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md` |
| Result file | this memo |

Evidence class: everything below is engineering evidence (AST counts, import
resolution, collection results, source inspection). No statistical or sampler
evidence was generated, and no scientific claim is made.

## Question A: Audit Completeness

### What replicates exactly

- All 25 per-file rows of the audit's mechanical inventory table match a fresh
  Python `ast` inventory exactly (classes, functions, totals, per file).
- The line-level appendix
  (`bayesfilter-tuning-definition-inventory-2026-08-16.md`) contains exactly
  1,417 rows whose (path, line, kind, name) tuples match the fresh AST walk
  with **zero** discrepancies, across 26 selected files (25 table files plus
  `bayesfilter/inference/staged_fixed_kernel_hmc.py`;
  `fixed_transport_hmc_tuning.py` contributes 0 definitions).
- The adjacent authority inventory is exact: all 9 module counts match fresh
  AST and the total is exactly 482.
- MacroFinance `tune_hmc_kernel` usage: exactly 49 Python files under
  `--no-ignore` semantics (39 when `.gitignore` is respected), confirming the
  audit's 49-file claim and identifying its counting criterion (all on-disk
  files, including gitignored ones).
- The MacroFinance robust driver is exactly one file
  (`daily_asset_midas_robust_broad_grid_tuning.py`), and the dsge_hmc active
  contract list (fixed metric, fixed transport, budget ladder, private
  helpers, entrypoint contract tests) matches my independent scan.

### Audit delta: missed or misdescribed items

1. **Mechanical table is internally inconsistent with its own total.** The
   audit's table lists 25 files summing to 1,405 definitions while claiming
   "26 files / 1,417". The missing row is
   `bayesfilter/inference/staged_fixed_kernel_hmc.py` (2 classes, 10
   functions, 12 definitions), which the appendix does include. The stated
   totals are correct; the table is incomplete.
2. **`bayesfilter/inference/hmc_kernel_selection.py` (118 definitions,
   currently git-modified) has no module summary anywhere in the audit
   prose.** Its docstring reads "Private fixed-trajectory candidate contracts
   for operational HMC tuning"; it defines `VerifiedFixedKernelHandoff` and
   the sanitized candidate/selection evidence layer, and it imports NumPy
   directly. Every smaller table module received a summary; this one did not.
   The refactor plan also never assigns it to a phase (see Question B).
3. **The adjacent inventory is not the direct dependency closure it is
   presented as.** The 26 core files directly import eight additional
   BayesFilter modules that appear in neither inventory (importers in
   parentheses; ~143 definitions total):
   - `bayesfilter/hmc_budget_contract.py` — 16 defs, "Pure work-budget
     contracts for operational HMC kernel tuning" (← `hmc_kernel_tuning`);
     its filename even matches the audit's own `budget` scope term;
   - `bayesfilter/hmc_route_contract.py` — 7 defs, route identity contracts
     (← `hmc_kernel_tuning`, `hmc_tuning_artifacts`);
   - `bayesfilter/inference/batched_value_score.py` — 46 defs
     (← `fixed_transport_hmc_tuning_tf`, `hmc_budget_ladder`): a direct
     dependency of one of the two proposed public interfaces;
   - `bayesfilter/inference/hmc_coordinates.py` — 31 defs, "coordinate and
     metric contracts for BayesFilter HMC tuning" (← `hmc_kernel_tuning`,
     `hmc_tuning_artifacts`);
   - `bayesfilter/inference/hmc_diagnostics.py` — 10 defs (←
     `hmc_kernel_tuning`, `hmc_tuning`);
   - `bayesfilter/inference/hmc_posterior_diagnostics.py` — 23 defs (←
     `staged_fixed_kernel_hmc`);
   - `bayesfilter/inference/neutra_shared_procedure.py` — 8 defs (← both
     NeuTra broad-grid core files);
   - `bayesfilter/runtime/__init__.py` — `stable_config_hash` used for
     artifact identity by 7 core files including the robust route.
   In addition, `native_tfp_hmc.py` (69 defs) and `neutra_hmc.py` (59 defs)
   are named execution authorities in the audit's own executive table but
   appear in neither inventory. Excluding them may be defensible ("execution,
   not tuning"), but the exclusion is nowhere recorded, and the plan's Phase 2
   extraction would touch the eight direct dependencies above.
4. **Public export surface was not audited and is broken today.**
   `bayesfilter/inference/__init__.py` exports 447 names through a lazy
   resolver over 40 modules (124 names match the tuning scope terms). Two
   committed contracts fail against the current checkout (both already absent
   at HEAD, so this is not caused by the dirty worktree edits):
   - `prepare_fixed_transport_hmc_adaptive_joint_grid_policy` is defined at
     `bayesfilter/inference/fixed_transport_hmc_grid_policy.py:1082` and
     asserted by the committed test
     `tests/test_fixed_transport_hmc_grid_policy.py:13,144-152`, but is
     missing from `__all__`; that test **fails at collection** now.
   - `HMCStagedTimeoutPolicy` is defined at
     `bayesfilter/inference/hmc_kernel_tuning.py:767` and declared in the
     top-level `bayesfilter/__init__.py:74` lazy map, but is missing from
     `bayesfilter/inference.__all__`; both import routes raise `ImportError`,
     which breaks MacroFinance
     `daily_asset_midas_l10d_bayesfilter_bootstrap_geometry_repair.py:24` and
     its test — one of the refactor plan's six named Phase-5 focused tests.
5. **New uncommitted tuning-scope files are not covered.**
   `neutra_curriculum_search.py` and `neutra_curriculum_training.py` are
   untracked, are listed in the (dirty) `_EXPORT_MODULES`, and contribute the
   tune-named public export `tune_neutra_curriculum_probe`. Under the audit's
   own boundary they belong to the NeuTra training domain, but the audit
   classified `neutra_staged_training.py` (also NeuTra training) into the core
   count while ignoring these; the boundary is inconsistently applied. Also
   unclassified: `neutra_end_to_end.py` (73 defs including
   `BroadGridSequentialConfig`, `FrozenTransportBroadGridConfig`, and
   broad-grid cell runners), `hmc_phase5_evidence_resume.py` (15 defs of
   Phase-5 evidence continuation), `joint_center.py` staged exports,
   `neutra_artifacts.py` frozen-transport exports, and the
   `bayesfilter/testing` tuning fixtures/harnesses
   (`hmc_fixed_metric_grid_search_fixture.py` with a `tune` callback,
   `hmc_operational_broad_grid_fixture.py`, `lgssm_neutra_gap_closure_tf.py`).
6. **False positives are included without recorded reasons.**
   `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` (49 defs) is an
   auxiliary-particle-filter frozen-proposal module matched only by the
   `frozen` term; it is inside the 1,417 core count but is not HMC tuning.
   `bayesfilter/ssm/filter_registry.py` (20 defs) and
   `bayesfilter/testing/ksc_gaussian_sum_ukf_scope.py` (0 defs) are
   name-matched but correctly excluded — the exclusions are just never
   recorded, so the selection cannot be re-derived from the documents.
7. **The repo-wide reference inventory is not reproducible as documented.**
   Its pointer to "the command in the Fable handoff memo" is dangling — the
   handoff memo contains no repo-wide text-match command. No pattern I tried
   reproduces 3,553 / 99 / 126 / 14 / 3,314: case-sensitive word-boundary
   `rg -l '\b(tune|tuning|grid|kernel|budget)\b'` over the four roots gives
   3,532 (closest); other reasonable variants give 4,269–8,347, and
   `bayesfilter` Python-file counts range 137–171 versus the claimed 99. The
   qualitative routing conclusion (docs artifacts dominate; the Python core is
   small) is robust under every variant, but the specific numbers are
   unsupported as recorded. The "active readers" classification the document
   itself calls required was deferred, not done.
8. **The line-level appendix is correctly labeled and must stay that way.**
   It states it "does not replace semantic review of numerical formulas";
   nothing in it verifies mathematics. Claim-bearing bodies still requiring
   manual semantic review before extraction: `tune_hmc_kernel` /
   `run_hmc_tune_verify_repair_loop` (outer loop),
   `initialize_hmc_kernel_geometry` / `_build_mass_artifact` /
   `_curvature_report`, `run_hmc_bootstrap_screen`,
   `run_hmc_windowed_mass_stage` (covariance updates), the dual-averaging
   updates in `hmc_tuning.py`, `evaluate_hmc_acceptance_evidence`
   (adjacent), `rank_normalized_hmc_diagnostics` (adjacent),
   `_repair_one_l` epsilon scaling and `select_robust_candidate` ordering in
   the robust route, the transport log-det/manifest binding in
   `tune_fixed_transport_hmc_kernel`, and the **three separate definitions of
   `_mass_artifact_signature`** (`hmc_kernel_tuning.py:12893`,
   `hmc_tuning.py:1840`, `hmc_budget_ladder.py:2630`), whose mutual
   consistency is unverified.

### External consumer completeness

Counts below use the audit's identified criterion (`--no-ignore`, Python files
on disk, excluding `.git`/`.claude`/`__pycache__`).

- The **113 MacroFinance / 16 dsge_hmc "tuning-related files" totals are not
  reproducible because their union criterion is unrecorded.** My documented
  union over the handoff-memo symbols plus
  `robust_broad_grid|generic_hmc_tuning|operational_broad_grid|budget_ladder`
  gives **131** MacroFinance files and **17** dsge_hmc files. The dsge_hmc
  17th file is a string mention only
  (`run_rotemberg_fixed_neutra_bayesfilter_tuning_smoke.py:383` names
  `run_generic_hmc_tuning_orchestration` inside a fallback payload, and its
  actual import is `tune_fixed_transport_hmc_kernel`), so the audit's 16 is
  consistent for active contracts. For MacroFinance, part of the 131-vs-113
  gap is `results/**/functional_source_snapshot/*_hmc_kernel_tuning.py`
  archival copies of the BayesFilter monolith vendored into result
  directories — these should be explicitly classified as historical
  snapshots, which the audit never does. Without the audit's file list I
  cannot exclude that its 113 also misses active consumers.
- Dynamic-import routes found (none evade the public boundary):
  `bayesfilter_macrofinance_migration_adapter.py:123`
  (`importlib.import_module("bayesfilter.inference")`),
  `mixed_frequency_tfp_bayesfilter_only_phase5_hmc_pilot.py:5661` (module
  file-path resolution), dsge_hmc
  `tests/contracts/test_bgs_bayesfilter_stage_b3a.py:24`
  (`spec_from_file_location` on a local runner).
- Private-name imports: `_mass_artifact_signature` in 30–35 MacroFinance
  files and 12 dsge_hmc files (11 `run_rotemberg_public_explicit_state_*`
  scripts plus
  `src/dsge_hmc/experiment_adapters/rotemberg_round380_neutra.py`);
  `_BootstrapFixedMassLatentValueScoreAdapter` in 1 dsge_hmc file. No
  dsge_hmc **test** currently locks the private names — the breakage risk
  lives in scripts that only fail when run.
- MacroFinance source-level contracts confirmed:
  `tests/test_daily_asset_midas_bayesfilter_owned_tuning_execution.py:167-171`
  asserts `inspect.getsource(...)` call counts
  (`count("tune_hmc_kernel(") == 1`) and forbidden keyword arguments; the
  l10c/l10d tests scan BayesFilter source lines.

### Required audit verdict

The core inventory (26 files / 1,417 definitions) and the adjacent 482 count
are exact and independently verified, and I found no missed *file* inside the
audit's stated core selection. The verdict is nevertheless REVISE because the
memo's own required checks fail on: the mechanical-table omission
(`staged_fixed_kernel_hmc.py`), the absent `hmc_kernel_selection.py` module
summary, an adjacent inventory that is not the actual direct-dependency
closure (8 modules, ~143 definitions, listed above with importers), an
unaudited and currently broken public export surface (two named exports with
file/line anchors), unclassified new uncommitted tuning files with a public
tune-named export, unrecorded false-positive rationale, and non-reproducible
repo-wide/consumer union counts.

`AUDIT_VERDICT: REVISE`

## Question B: Plan Correctness and Coverage

### Architecture checks (all pass)

- `tune_hmc_kernel` as the ordinary-HMC compatibility boundary is defensible:
  49 MacroFinance files use the name, and the source-level call-count tests
  bind it textually, so keeping the name minimizes migration surface.
- `tune_fixed_transport_hmc_kernel` is genuinely a second contract, not a
  strategy flag: it binds identity mass in transport coordinates, a
  `fixed_transport_manifest_hash`, and a distinct artifact schema
  (`bayesfilter.fixed_transport_hmc_kernel_tuning_result.v2`,
  `fixed_transport_hmc_tuning_tf.py:289-424`), and dsge_hmc imports it
  directly with contract tests on that schema.
- LEDH scope/control tuning and NeuTra training/search are kept outside the
  consolidation, matching repository policy. (The inventory's inconsistent
  NeuTra boundary is an audit defect, not a plan defect.)
- The five legacy routes (robust, operational, fixed-metric, budget ladder,
  generic) are correctly classified diagnostic/compatibility with shims until
  cross-repo migration; immediate deletion is correctly forbidden.

### Numerical and evidence checks (pass, one sharpening)

The plan rejects the robust route's `use_xla=False`, fixed L grid, and short
qualification as universal defaults; assigns promotion/veto/explanatory roles;
forbids one-seed ranking claims; requires disjoint calibration/verification
seeds, identity binding, GPU memory-growth verification, CPU-hidden test
labeling, and no NumPy in admitted execution (five core modules currently
import NumPy directly: `hmc_kernel_tuning`, `hmc_budget_ladder`,
`hmc_tuning`, `hmc_kernel_selection`, `generic_hmc_tuning` — confirmed
migration debt). One sharpening: `RobustBroadGridConfig.__post_init__`
**hard-rejects** any `l_grid` other than `(3, 5, 9, 13, 18, 25)` and any
`qualification_results != 500`
(`hmc_robust_broad_grid.py:113-114,141-142`), so these are frozen constraints,
not overridable defaults; Phase 3 must generalize the config before any
target-specific review of those numbers is even expressible.

### Compatibility and test checks — findings

Blocking (the plan's gates cannot function as written):

- **B1. The Phase 1 gate "all existing BayesFilter tests … pass unchanged"
  is unsatisfiable at baseline.** The current checkout already fails
  collection: `tests/test_fixed_transport_hmc_grid_policy.py` (missing
  export, item A4 above) plus 5 other collection errors, two of which
  (`tests/test_neural_force_hmc_gpu.py`, `tests/test_neural_force_training_gpu.py`)
  fail-closed **by design** under `CUDA_VISIBLE_DEVICES=-1`, so the plan's
  CPU-hidden full-suite command can never be green as written. The plan needs
  a Phase 0 step that records (and, with owner authorization, repairs) the
  pre-existing export breakages and a stated deselect/marker policy for
  GPU-required tests in CPU-hidden runs.
- **B2. One of the six named Phase-5 focused MacroFinance tests fails
  collection today** (`test_daily_asset_midas_l10d_...` via the
  `HMCStagedTimeoutPolicy` export gap). The Phase 5 gate would report a
  failure that is actually a present-day BayesFilter export regression; the
  plan must record this as pre-existing and sequence its repair before the
  gate is meaningful.
- **B3. The Phase 6 focused command fails as written.** The two Rotemberg
  contract tests raise `FileNotFoundError` unless `BAYESFILTER_ROOT` is set —
  they explicitly reject `/home/ubuntu/python/BayesFilter` as a "stale"
  checkout otherwise. With
  `BAYESFILTER_ROOT=/home/ubuntu/python/BayesFilter` prepended, all three
  focused contract tests collect (26 tests). Every dsge_hmc command in the
  plan (and in the handoff memo) needs that variable added.

Nonblocking risks and cleanups:

- **B4. dsge_hmc full-suite collection segfaults** in this environment
  (`tests/archive/test_rotemberg_nk.py:151` runs a module-level
  `scipy.linalg.solve` during import; hard `Segmentation fault`). Since
  `testpaths = ["tests", …]` recursively includes `tests/archive`, the plan's
  final command dies before reporting which paths ran — its own stop
  condition "a test command cannot establish which paths ran" fires. Add an
  explicit `--ignore=tests/archive` (or equivalent policy) with the omission
  recorded.
- **B5. Environment pinning is stated but not operationalized.** Bare
  `python` on this machine is base anaconda without TensorFlow (545 spurious
  BayesFilter collection errors); the working env is conda `tfgpu`
  (Python 3.13.13, TF 2.20.0, TFP 0.25.0), which lacks `pandas`, producing
  30+ MacroFinance collection errors unrelated to tuning. The plan's commands
  should name the interpreter/env per repository, or the gates will measure
  environment drift instead of migration correctness. A further order
  dependence exists: some MacroFinance modules fail with "BayesFilter
  checkout is unavailable" only under full-suite collection, not single-file
  collection.
- **B6. `hmc_kernel_selection.py` is never assigned to a phase.** 118
  definitions of candidate/selection contracts sit between the monolith and
  the artifact layer; Phase 2's six-family split does not name them. Assign it
  (likely `hmc_kernel_stages.py` or its own family) before extraction.
- **B7. Phase 5's migration checklist omits two active MacroFinance caller
  families**: budget-ladder callers
  (`mixed_frequency_tfp_c2_full_bayesfilter_hmc_tuning_v2_phase4_step_trajectory.py`,
  `..._phase5t_real_tuning_loop.py`, plus three tests) and
  `orchestrate_generic_hmc_tuning` callers
  (`cross_country_multi_asset_bayesfilter_owned_hmc_client.py`,
  `..._mass_preconditioner.py`, plus two tests). The full-suite gate would
  catch breakage, but the checklist should name them.
- **B8. Phase 1's replacement list is right but underspecified**: the public
  `mass_artifact_signature` replacement must state which of the three current
  private definitions is canonical and add a consistency test across the
  three (paths in A8).
- **B9. Minor naming collision risk**: proposed `hmc_artifacts.py` /
  `hmc_geometry.py` sit next to existing `hmc_tuning_artifacts.py` and the
  geometry family inside the monolith; state explicitly which one becomes
  authoritative to avoid a third artifact-helper home.

Confirmed plan strengths (checked, not just read): all six BayesFilter
fast-check test files exist; all six named MacroFinance focused tests exist;
all three dsge_hmc contract tests exist; the dsge_hmc `pyproject.toml`
testpaths match the plan's list verbatim; the missing MacroFinance
robust-driver test is correctly recorded as a Phase 5 deliverable rather than
claimed; the missing-test-file recording rule is present; focused-before-full
ordering and trusted-GPU-canary sequencing are present.

### Corrected phase/gate text (proposed)

- Phase 0, add: "Record the pre-existing baseline: `pytest --collect-only`
  results in all three repositories under the pinned environments, including
  the two known BayesFilter export regressions
  (`prepare_fixed_transport_hmc_adaptive_joint_grid_policy`,
  `HMCStagedTimeoutPolicy`), the GPU-required tests that fail-closed under
  CPU hiding, and the dsge_hmc `tests/archive` collection segfault. Gates in
  later phases are evaluated against this recorded baseline, not against
  'all tests pass'."
- Phase 1 gate, replace "pass unchanged" with "pass unchanged relative to the
  recorded Phase-0 baseline; the two export regressions must be repaired (or
  explicitly waived by the owner) before this gate closes."
- Phase 6 commands, prepend `BAYESFILTER_ROOT=/home/ubuntu/python/BayesFilter`
  and append `--ignore=tests/archive` to the configured full-suite command,
  recording the ignored path.
- All commands: name the interpreter (`conda run -n tfgpu …` or the
  repository-appropriate env) instead of bare `python`.

### Required plan verdict

The two-interface architecture, the diagnostic/compatibility classification,
and the evidence discipline survive independent scrutiny against source and
consumers; none of the findings reject the plan's direction. The verdict is
REVISE because B1–B3 are blocking gate-correctness gaps: as written, the
Phase 1/5 gates are unsatisfiable against the current baseline for reasons the
plan does not know about, and the Phase 6 focused command fails at collection.
These are repairable with the corrected text above; B4–B9 are nonblocking.

`PLAN_VERDICT: REVISE`

## Question C: Test Adequacy Matrix

| # | Item | Status | Evidence path |
|---|---|---|---|
| 1 | Route classification/discovery, duplicate active exports | missing | No tuning-route ledger/guard exists; only NeuTra-side `tests/test_neutra_hmc_route_policy.py`. Phase 0 deliverable. |
| 2 | Old/new deterministic mechanics parity | missing | No parity fixtures found in `tests/test_hmc_kernel_tuning_*.py` or robust/generic tests. Phase 2/3 deliverable. |
| 3 | Artifact scope/source/adapter/transport mismatch fail-closed | partially covered | `tests/test_frozen_kernel_validation.py`, `tests/test_hmc_kernel_tuning_public_api.py`, `tests/test_fixed_transport_hmc_tuning.py`; no unified canonical schema yet. |
| 4 | No public leakage of raw/private samples or mass state | partially covered | `tests/test_hmc_kernel_tuning_public_api.py` (public/private projection); robust route emits `raw_samples_retained: false` but its test does not exercise the campaign. |
| 5 | Evidence role separation (target-status/finiteness/divergence/acceptance/R-hat-ESS) | partially covered | `tests/test_hmc_kernel_selection.py`, `tests/test_hmc_robust_broad_grid.py:48-73` (selector-level only), verification tests; no unified role-typing test. |
| 6 | Fresh calibration/verification seed lineage | missing (for the consolidation target) | `tests/test_hmc_robust_broad_grid.py` covers config validation and selector only (6 tests, lines 35–73); no seed-lineage assertions; `_tune_one_l`/`_repair_one_l`/`_qualification` and the campaign are untested. |
| 7 | MacroFinance source-level call-count and forbidden-import assertions | covered | `MacroFinance/tests/test_daily_asset_midas_bayesfilter_owned_tuning_execution.py:167-171`; l10c/l10d source scans. |
| 8 | MacroFinance robust-driver and ordinary-tuner mocks | partially covered | Ordinary-tuner mock: `test_daily_asset_midas_bayesfilter_owned_tuning_execution.py` (monkeypatch). Robust-driver test: confirmed absent (`tests/*robust_broad*` = none) — Phase 5 deliverable, correctly recorded by the plan. |
| 9 | MacroFinance fixed-metric and operational diagnostic routes | covered | `tests/test_run_ccma_broad_fixed_metric_l_epsilon_search.py`, `tests/test_two_currency_double_zlb_dz5_neutra_fixed_metric_grid.py`, `tests/test_ccma_operational_broad_l_epsilon_neighbor_guard.py`, `tests/test_run_ccma_hmc_operational_recovery.py`. |
| 10 | dsge_hmc BGS grid callback/aggregate behavior | covered | `dsge_hmc/tests/contracts/test_bgs_bayesfilter_stage_c_grid_tuning.py` (8 tests; callback/aggregate assertions). |
| 11 | dsge_hmc Rotemberg fixed-transport tuning, XLA, artifact contracts | covered, with an environment caveat | `tests/contracts/test_rotemberg_fixed_neutra_bayesfilter_xla_relaunch.py`, `..._xla_gate.py`; collect (26 tests) only with `BAYESFILTER_ROOT` set; both error without it. |
| 12 | Replacement of private `hmc_kernel_tuning` imports | missing | No test locks the private names today; they live in 11 dsge_hmc scripts + 1 adapter and 30+ MacroFinance files. Phase 1/5/6 deliverable. |
| 13 | Configured full-suite discovery in both external repositories | partially covered | MacroFinance: 4,252 collected + 38 errors (missing `pandas` in tfgpu; order-dependent BayesFilter-root failures). dsge_hmc: full collection segfaults in `tests/archive/test_rotemberg_nk.py`; focused subsets collect cleanly. Discovery is not clean in either repo in this environment. |

Conclusion for Question C: the proposed matrix can answer the migration
question only after the Phase 0/1 deliverables exist; today 4 of 13 items are
missing, 5 partial, 4 covered. That is consistent with the plan's own claim
that contract/parity tests are the first implementation phase — but the plan
must not describe currently-missing items as gates already available.

## Consumer Map (summary)

| Consumer | Contract | Files | Anchor |
|---|---|---:|---|
| MacroFinance | `tune_hmc_kernel` | 49 (`--no-ignore`) / 39 (ignoring gitignored) | e.g. `one_country_zlb_ns_estimation.py`, MIDAS/CCMA drivers |
| MacroFinance | robust broad grid | 1 | `daily_asset_midas_robust_broad_grid_tuning.py` (imports `RobustBroadGridConfig`, `stable_adapter_signature`, `tune_hmc_kernel_robust_broad_grid`) |
| MacroFinance | `run_fixed_metric_grid_search` | 7 | CCMA + two-currency scripts/tests |
| MacroFinance | `run_operational_broad_grid` | 1 | `scripts/ccma_operational_broad_l_epsilon_neighbor_guard.py` |
| MacroFinance | budget ladder | 5 (incl. 3 tests; plus archival `results/**/functional_source_snapshot` copies) | `mixed_frequency_tfp_c2_full_*_phase4_step_trajectory.py`, `..._phase5t_real_tuning_loop.py` |
| MacroFinance | `orchestrate_generic_hmc_tuning` | 5 | `cross_country_multi_asset_bayesfilter_owned_hmc_client.py`, `..._mass_preconditioner.py`, 2 tests |
| MacroFinance | `hmc_kernel_tuning` references / private `_mass_artifact_signature` | 94–104 / 30–35 | incl. source-scan tests; `importlib` boundary in `bayesfilter_macrofinance_migration_adapter.py:123` |
| MacroFinance | **live break** | 1 driver + 1 test | `daily_asset_midas_l10d_...py:24` imports missing `HMCStagedTimeoutPolicy` |
| dsge_hmc | `tune_fixed_transport_hmc_kernel` | 2 | `scripts/run_rotemberg_fixed_neutra_bayesfilter_tuning_smoke.py`, `experiment_adapters/rotemberg_round380_neutra.py` |
| dsge_hmc | `run_fixed_metric_grid_search` | 2 | BGS stage-C script + contract test |
| dsge_hmc | budget ladder | 2 | public-explicit-state scripts |
| dsge_hmc | private `_mass_artifact_signature` / `_BootstrapFixedMassLatentValueScoreAdapter` | 12 / 1 | 11 `run_rotemberg_public_explicit_state_*` scripts + round380 adapter; no test locks them |
| dsge_hmc | contract-test environment | 2 tests | require `BAYESFILTER_ROOT`; reject `/home/ubuntu/python/BayesFilter` without it |

## Decision Table

| Field | Value |
|---|---|
| Decision | `AUDIT_VERDICT: REVISE`; `PLAN_VERDICT: REVISE` (direction endorsed; gates and inventories need the listed repairs) |
| Primary criterion status | Core 1,417 inventory and adjacent 482 verified exact; completeness beyond the selected files fails (closure, exports, new files); plan architecture sound; plan gates unsatisfiable at current baseline |
| Veto diagnostics | Two live export regressions (A4/B1/B2); Phase 6 command failure without `BAYESFILTER_ROOT` (B3); dsge_hmc archive segfault (B4) |
| Main uncertainty | Audit's exact union criteria for 113/16 and 3,553 counts (unrecorded); MacroFinance full-suite order-dependent failures not root-caused; content-level (non-name) tuning code outside `bayesfilter/inference|highdim` not exhaustively swept |
| Next justified action | Codex: repair the two `__init__` export regressions (small, evidence-anchored), regenerate the audit docs per the deltas above, apply the corrected gate text, then rerun the three `--collect-only` baselines under pinned envs |
| Not concluded | No claim about posterior correctness, convergence, sampler quality, robust-route default readiness, or any statistical ranking; no GPU-context claims (GPU was intentionally hidden throughout) |

## Residual Uncertainty

- I did not run any test bodies, only collection; "covered" in the matrix
  means the assertion exists, not that it currently passes.
- The MacroFinance full-suite order-dependent "BayesFilter checkout is
  unavailable" errors were reproduced but not root-caused (single-file
  collection succeeds; some module in the collection order mutates the
  resolution environment). Smallest next artifact: bisect collection order
  with `pytest --collect-only tests/test_A.py tests/test_B.py` pairs.
- Content-matched files outside the name-matched set were swept via
  definition-name heuristics, not full-text semantic review; a residual
  tuning-relevant module outside `bayesfilter/inference` and
  `bayesfilter/highdim` with non-matching names could still exist.
- The tfgpu environment (TF 2.20.0/py3.13) may not be the environment the
  original audit used; counts that depend only on `ast`/`rg` are
  environment-independent, but collection results are not.
- Smallest next artifacts overall: (1) a committed inventory-generation
  script with its exact selection criteria (Phase 0 already promises this);
  (2) a one-line fix + test for each of the two export regressions; (3) a
  recorded `--collect-only` baseline per repository under named environments.
