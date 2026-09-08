# BayesFilter Tuning Streamline: Step D–H Execution Result

Date: 2026-08-21
Executor: Claude Code (Fable 5), interactive session supervised by user
Governing documents:
- `docs/plans/bayesfilter-tuning-streamline-claude-code-handoff-2026-08-19.md`
- `docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md` (Phases 5–7)
- `docs/plans/bayesfilter-tuning-streamline-step-d-adjudication-2026-08-20.md`
  (adjudication, owner decisions, family notes, contract-gap finding §6a)
- `docs/plans/bayesfilter-tuning-streamline-consumer-repair-result-2026-08-20.md`
  (Steps A–C result)

## 1. Findings First

1. **Contract gap (owner review required).** The two remaining Group-A
   migrations — C2 `phase4_step_trajectory` + `phase5t` and the CCMA
   owned-HMC client — are BLOCKED by a genuinely different numerical
   contract: a caller-owned candidate campaign under an externally frozen
   precomputed mass artifact, with BayesFilter as runtime/packaging
   authority. `tune_hmc_kernel` cannot express it (`mass_policy` admits only
   `windowed_adaptive`/`fixed_identity`; no caller mass artifact, candidate
   grid, leapfrog count, or budget schedule). Both are completed frozen
   June-2026 campaigns; they are reclassified frozen-campaign historical
   drivers. Details and the two candidate redesign paths: adjudication §6a.
   This discharges handoff §4's "stop and document it for review" rule
   without deleting or altering either chain.
2. **Step F full-suite run is blocked, not waived**: missing optional
   dependencies (`pandas`, `torch`) cannot be installed under the handoff's
   no-install rule; order-dependent BayesFilter checkout-resolution
   collection errors are pre-existing baseline debt outside tuning scope;
   suite scale (9,499 tests incl. 180 s-timeout integration tests) exceeds
   the session's command ceiling. Collection manifests were regenerated and
   categorized instead (Section 5).
3. Everything else in Step D/E/G that was expressible completed green. No
   handoff stop condition fired beyond the documented contract gap.

## 2. Files Changed (this continuation, beyond the A–C memo)

BayesFilter:
- `bayesfilter/inference/hmc_bootstrap.py` — public
  `build_bootstrap_fixed_mass_adapter` alias (+8 lines; same function
  object, adapter-signature identity preserved by construction).
- `bayesfilter/inference/__init__.py` — `build_bootstrap_fixed_mass_adapter`
  added to `__all__` (lazy export resolves via `hmc_bootstrap`).

dsge_hmc (import swaps, local alias preserved):
- `src/dsge_hmc/experiment_adapters/rotemberg_round380_neutra.py:529`
- `scripts/run_rotemberg_public_explicit_state_mass_artifact_repair.py:78`
- `scripts/run_rotemberg_public_explicit_state_bayesfilter_candidate_campaign.py:213`
- `scripts/run_rotemberg_public_explicit_state_bayesfilter_candidate_refinement.py:176`

MacroFinance:
- 13 CCMA `phase5*/phase10*` diagnostics — `hkt._mass_artifact_signature` →
  `hmc_artifacts.mass_artifact_signature`, `hkt._build_bootstrap_fixed_mass_adapter`
  → `hmc_bootstrap.build_bootstrap_fixed_mass_adapter` (15 call sites; two
  facade imports added beside the retained `hkt` import in each file).
- `daily_asset_midas_eom_bootstrap_hmc_diagnostic.py:40` — direct-import swap.
- `daily_asset_midas_bayesfilter_retained_synthetic_hmc.py:109` — swap with
  underscore aliases so `_require_runtime()` `locals()` keys are unchanged.
- `daily_asset_midas_robust_broad_grid_tuning.py` — migrated to canonical
  `tune_hmc_kernel` (config, call, result processing, schemas
  `canonical_kernel_tuning_*.v1`, new dated artifact scope 2026-08-21,
  frozen-lineage docstring; `MASS_PREPARATION_SEED` and L-grid removed —
  canonical route owns mass windows and candidate grids).
- `tests/test_daily_asset_midas_robust_broad_grid_tuning.py` — NEW (Step E):
  6 AST/source-anchor contract tests covering the five handoff-required
  points (driver has import-time GPU guards, so no CPU-hidden import).

Untouched by owner decision: dsge_hmc `retained_validation` and
`mass_injection_audit` (frozen debt), MacroFinance DZ5 canary and
`daily_asset_midas_phase10_synthetic_hmc_validation.py` (grandfathered),
five dsge_hmc historical one-shots, and the two contract-gap families.

## 3. Commands and Environment

All test runs: conda `tfgpu` (Python 3.13, TF 2.20.0), `-p no:cacheprovider`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, `PYTHONPATH=/home/ubuntu/python/BayesFilter`
(+`BAYESFILTER_ROOT` for dsge_hmc). Focused commands are the exact handoff
§6 commands (MacroFinance list extended by the new Step E test file);
CPU-hidden runs used `CUDA_VISIBLE_DEVICES=-1`.

## 4. Results

| Gate | Result | Wall time |
| --- | --- | --- |
| Route inventory `--check` after facade | clean: 10 discovered/registered, 0 unclassified, 0 stale | — |
| Facade identity checks (tfgpu) | public alias `is` private function/class; lazy export resolves | — |
| dsge_hmc focused after swaps (CPU-hidden) | **50 passed** | 46.8 s |
| MacroFinance focused after swaps (CPU-hidden) | **64 passed** | 6.3 s |
| Step E test | **6 passed** | 0.2 s |
| MacroFinance focused incl. Step E (CPU-hidden) | **70 passed** | 6.2 s |
| BayesFilter GPU-visible canary suite (Step G) | **108 passed** (handoff baseline 106; +2 worktree drift) | 195.0 s |

## 5. Step F Collection Manifests (CPU-hidden, 2026-08-21)

| Repository | Collected | Errors | Categories |
| --- | --- | --- | --- |
| MacroFinance `tests/` | 4,553 | 33 | (a) missing optional `pandas` (MIDAS replication family) and `torch` (16 `portfolio_rl` files) in `tfgpu`; (b) order-dependent BayesFilter checkout-resolution RuntimeErrors — probe: `test_daily_asset_midas_bayesfilter_map_tuning_execution.py` errors in full collection, collects 3 tests cleanly alone. Baseline was 4,252/38 — error count down, no new categories. |
| dsge_hmc configured testpaths, `tests/archive` ignored | 4,946 | 2 | `tests/contracts/test_threading_env.py` (TF threading init, order-dependent) and `tests/numerics/test_bgs_descriptor_coefficients.py` (OSError in reader; belongs to unrelated dirty BGS work). |

Full-suite run: NOT PERFORMED in `tfgpu` — blockers recorded in Section 1.
No partial run is reported as a full-suite pass.

### 5a. Owner-authorized environment clone (2026-08-21, supersedes the
dependency blocker for full-suite execution)

The owner authorized a disposable clone for full-suite testing; the canonical
`tfgpu` environment is untouched and remains the provenance baseline for all
focused-gate evidence above.

- Environment: `tfgpu-full` = conda clone of `tfgpu` plus pip installs
  `pandas 3.0.5`, `torch 2.13.0+cpu` (CPU wheel, `cuda_available: False`),
  `gymnasium 1.3.0`. Versions are pip-resolved convenience choices recorded
  here; no claim-bearing artifact may cite `tfgpu-full` as its environment
  without owner promotion.
- Collection manifest in `tfgpu-full` (CPU-hidden,
  `--continue-on-collection-errors`): **4,655 collected, 12 errors** —
  down from 4,553/33 in `tfgpu`. The +102 tests are the unlocked
  pandas/torch families. Residual categories: 7 order-dependent BayesFilter
  checkout-resolution RuntimeErrors (probe: `..._final_gate_closure.py`
  collects cleanly alone) and 5 portfolio_rl files that needed `gymnasium`
  (installed after the manifest; expected to clear on the full run).
- Full-suite execution: chunked runner (20 files/chunk, CPU-hidden,
  `tests/archive` excluded for dsge_hmc), per-chunk JSON records. The run
  COMPLETED (31 MacroFinance chunks + 32 dsge_hmc chunks, ~8 h wall). The
  session restart deleted the scratchpad chunk logs; the totals below are
  reconstructed from the monitor event stream (complete per-chunk counts),
  but per-test failure lists from failing chunks were lost and can only be
  re-derived by re-running those chunks.

### 5b. Chunked full-suite aggregate (tfgpu-full, CPU-hidden, 2026-08-21/22)

| Repository | Chunks | Passed | Failed | Errors | Skipped | Segfaulted chunks |
| --- | --- | --- | --- | --- | --- | --- |
| MacroFinance | 31/31 ran | 4,259 | 458 | 6 | 6 (+1 xfail) | 0 |
| dsge_hmc | 32/32 ran, 26 completed | 3,591 | 428 | 8 | 7 | 6 (rc=139: DS_002/003/024/025/026/027, ~120 files unreported) |

This is a chunked engineering census of two dirty worktrees, not a
full-suite pass, and is not claim-bearing evidence (environment is the
unpromoted `tfgpu-full` clone).

Failure classification (from live triage before log loss):

1. **No migration-touched file failed.** All campaign-migrated/swapped files
   are covered by the focused gates, which pass (70/70, 50/50). Spot-checked
   suspicious names: `test_bayesfilter_runtime_authority_policy.py::
   test_active_c2_imports_do_not_load_historical_runtime` passes in
   isolation (order-dependent `sys.modules` pollution);
   `test_ccma_hmc_estimation_recovery_phase1.py::test_phase1_binds_current_
   bayesfilter_origins_and_sources` fails on a stale pinned source-hash
   table (`CURRENT_SOURCE_HASHES`) — `bayesfilter/inference/hmc.py` is
   git-clean yet hashes differently from the pin, so the pin predates the
   current BayesFilter commit; additionally `hmc_kernel_tuning.py` carries
   pre-existing unrelated dirty edits and two files carry this campaign's
   facade edits. Stale-pin family, adjudicated not-a-regression, but the
   pins will need refresh whenever the owner rebaselines.
2. `test_bayesfilter_macrofinance_migration_adapter.py` freshness failures
   reproduce identically in the canonical `tfgpu` env — pre-existing
   dirty-worktree artifact drift (matched-DGP initialization artifact no
   longer rebuilds), independent of this campaign.
3. Large failure clusters (MF_014: 129, MF_021: 54, DS_012/013: 52/55,
   DS_009/016/017: 46/41/46) sit in files never exercised by this
   campaign's focused gates; with the per-test lists lost, they are
   recorded as uncharacterized pre-existing full-suite debt pending
   re-run if the owner wants a census with failure identities.
4. dsge_hmc segfaults: RESOLVED TO FILE LEVEL by per-file bisection
   (2026-08-22, probe log
   `/tmp/bf-release-census/ds_segfault_bisect.txt`). Six files
   segfault (rc=139) when run alone, execution-time not import-time:
   - `tests/contracts/test_bgs_public_integration.py`
   - `tests/contracts/test_bgs_synthetic_generator.py`
   - `tests/extended/test_svd_lgssm_hmc_recovery.py`
   - `tests/extended/test_svd_nonlinear_ssm_hmc_recovery.py`
   - `tests/integration/test_bgs_d296_likelihood_gradient.py`
   - `tests/integration/test_bgs_d296_state_space.py`
   Two families: BGS synthetic/public-integration + d296 likelihood/state
   space, and the SVD-SSM HMC-recovery pair. Six additional files exceeded
   the 300 s single-file probe budget (rc=124) and are long-runners, not
   crashers: `test_convergence_medium.py`, `test_neural_solver_overnight.py`,
   `test_nk_convergence_audit.py`, `test_nk_typical_hessian_variation.py`,
   `test_overnight_diagnostics.py` (all `tests/extended`), and
   `tests/integration/test_bgs_bayesfilter_posterior.py`. The remaining
   ~108 files in the six chunk ranges pass or fail normally without
   crashing. Release action: extend the exclusion policy to name these six
   segfault files explicitly (mirroring the `tests/archive` precedent) and
   mark the six long-runners with the repo's `slow`/`overnight` markers so
   default runs skip them.

## 6. GPU Provenance (Step G)

Trusted GPU context, 2026-08-21: two physical GPUs visible to TF 2.20.0
(NVIDIA GeForce RTX 4080 SUPER ×2, compute capability 8.9, ~30 GB each);
`set_memory_growth(True)` applied and `get_memory_growth` verified `True` on
both **before** runtime initialization; `TF_FORCE_GPU_ALLOW_GROWTH=true`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. The 108-test focused tuning suite
passed in this GPU-visible process (194.95 s). This establishes device and
allocator provenance for that process only — not per-test GPU utilization,
performance ranking, or GPU default readiness for any consumer.

## 7. Git State

- BayesFilter `5699dafe`, MacroFinance `5e310d7`, dsge_hmc `ba251192` — all
  unchanged commits; all edits uncommitted; nothing pushed.
- `git diff --check` clean in all three repositories.
- Unrelated dirty work preserved untouched: MacroFinance phase-14 HMC
  files/docs, dsge_hmc Rotemberg/BGS files (incl. its unrelated collection
  error), BayesFilter NeuTra/SSL-LSTM work.

## 8. Decision Table

| Item | Status |
| --- | --- |
| Decision | Step D closed for all expressible families; contract-gap families documented and frozen; Step E delivered; Step F manifests recorded with blockers; Step G canary green; Step H deferred |
| Primary criterion | Focused gates green post-migration (50/50, 70/70) with no signature drift — met |
| Veto diagnostics | None fired. Import swaps proven behavior-identical by source inspection (both private signature homes are one-line delegates; facade is the same function object) |
| Main uncertainty | Contract-gap redesign path (fixed-transport reformulation is a `not checked` hypothesis); full-suite health beyond collection |
| Next justified action | Owner: choose a redesign path for the frozen-mass contract (§6a) and rule on Step F blockers (installs/waivers); then Step H quarantine can be re-evaluated |
| Explicit nonclaims | No posterior correctness, convergence, sampler validity, statistical ranking, GPU readiness beyond process provenance, or full-suite pass |

## 9. Step H Status

Deferred with cause, independent of green-run counting: historical routes
retain sanctioned consumers — the frozen-campaign drivers (C2 phase4/5T,
CCMA owned client) and grandfathered files still import
`run_fixed_mass_hmc_tuning_budget_ladder`, `orchestrate_generic_hmc_tuning`,
and `run_fixed_metric_grid_search` by design. Removing historical routes
from default exports now would break files the owner decided to freeze.
Quarantine becomes actionable only after the §6a redesign decision retires
or re-homes those consumers.

## 10. Evidence Ledger

| Ledger | Evidence added this continuation | Supports | Does not support |
| --- | --- | --- | --- |
| Engineering correctness | Focused gates green post-swap/migration; route inventory clean; compile checks; diff hygiene | Swap/migration mechanics; facade identity | Full-suite health; scientific validity |
| Numerical validity | Source-inspection proof that signature swap and facade are behavior-identical; frozen chains untouched | `correct` verdict for the swaps | Anything about tuning quality; the un-run canonical MIDAS campaign |
| GPU execution | Verified pre-init memory growth on both GPUs; 108-test GPU-visible pass | Process-level device/allocator provenance | Per-test GPU use; performance; consumer GPU readiness |
| Statistical ranking | none added | — | Any method ranking |
| Default readiness | Robust driver now on the canonical interface | Interface-direction progress | Kernel quality of any future campaign |
