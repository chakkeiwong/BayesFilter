# BayesFilter Tuning Streamline: Real Consumer Repair Result (Steps A–C)

Date: 2026-08-20 (execution on 2026-08-19, memo written after session restart)
Executor: Claude Code (Fable 5), interactive session supervised by user
Governing documents:
- `docs/plans/bayesfilter-tuning-streamline-claude-code-handoff-2026-08-19.md` (Sections 6–8, 10)
- `docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md` (Phases 5–6 gates)

## 1. Findings First

- No unresolved failures. Both real-worktree focused gates are green after the
  audited repair bundles were applied.
- No new failure fingerprints appeared; all four MacroFinance and three
  dsge_hmc baseline failures resolved exactly as adjudicated in the handoff.
- No stop condition from handoff Section 11 fired.
- Baseline-document discrepancy (recorded, not blocking): the primary plan's
  Phase-5 gate text references the 2026-08-16/17 baseline (6 focused files,
  101 collected, 90 passed / 11 failed). The handoff's 2026-08-19 baseline
  (7 files, 64 collected, 60/4) supersedes it per the handoff's own stale-
  authority rule. The 11-failure language in the primary plan is stale; the
  four-failure adjudication ledger in handoff Section 6 is the live contract
  and is now fully discharged.
- Residual risk: the focused gates cover contract drift only. Consumer
  call-site debt (handoff Section 9) is untouched; full suites (Step F) not
  run; only one green cross-repo focused run exists, so Phase 7 cleanup
  remains forbidden.

## 2. Files Changed

Patch application was done by manual edit replicating the reviewed
`apply_patch` bundles exactly (no `apply_patch` binary on PATH; bundle
checksums verified against handoff Section 7 before application):

```text
6408f92adba53e3b05dca124d0bbae04699d44ba49b61e44cc31f47a5330a824  macrofinance-tuning-consumer-repair-2026-08-19.patch
b8b6d4f692fdb59f4bbf0c5fa9f089eed3c4ccc889bb2974d99e930b3c5672e7  dsge-hmc-tuning-consumer-repair-2026-08-19.patch
```

MacroFinance (2 files, +6/-4 lines):

- `tests/test_daily_asset_midas_l10d_bayesfilter_bootstrap_geometry_repair.py`
  - lines 251, 549: staged-timeout policy id
    `ccma_phase4y_stage_budget_v1` -> `bayesfilter_hmc_emergency_stage_caps_v2`
  - lines 416-418: brittle textual `"0.25" not in public_text` assertion
    replaced by structural checks
    (`public_summary_contains_mass_matrices is False`,
    `external_geometry_hints.raw_matrix_values_publicized is False`)
- `bayesfilter_macrofinance_fixed_kernel_tuning_screen.py`
  - line 129 area: `checks.pop("chain_execution_mode", None)` with comment;
    execution mode is not a tuning control in the historical screen

dsge_hmc (2 files, +72/-3 lines):

- `tests/contracts/test_rotemberg_fixed_neutra_bayesfilter_xla_relaunch.py`
  - line 188: selection rule ->
    `eligible_trajectory_acceptance_in_band_then_rhat_convergence_then_ess`
  - line 198: `candidate_count` 49 -> 63; line 199 `candidate_workers == 49`
    preserved unchanged
  - line 226: `full_grid_candidate_count` 49 -> 63
- `src/dsge_hmc/experiment_adapters/fixed_neutra_hmc.py`
  - after line 199: four new `FlowFixedTransportAdapter` methods —
    `pullback_score`, `pullback_score_batch`, `log_abs_det_jacobian_score`,
    `log_abs_det_jacobian_score_batch`. Each delegates to a flow-native
    method when available, otherwise uses `tf.GradientTape`; score pullback
    fails closed (`ValueError`) on a disconnected transport; log-Jacobian
    score returns zeros for a legitimately constant log-Jacobian.

Post-edit `git diff` was compared hunk-by-hunk against the patch text; only
intended hunks present. `git diff --check` clean in both repositories.

## 3. Commands and Environment

Environment: conda `tfgpu` (Python 3.13, TensorFlow 2.20.0, TFP 0.25.x),
CPU-hidden (`CUDA_VISIBLE_DEVICES=-1`, GPU devices intentionally hidden),
`TF_FORCE_GPU_ALLOW_GROWTH=true`, `-p no:cacheprovider`.

MacroFinance (from `/home/ubuntu/python/MacroFinance`):

```bash
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  PYTHONPATH=/home/ubuntu/python/BayesFilter \
  python -m pytest -q -p no:cacheprovider \
  tests/test_daily_asset_midas_bayesfilter_map_tuning_execution.py \
  tests/test_daily_asset_midas_bayesfilter_owned_tuning_execution.py \
  tests/test_daily_asset_midas_l10c_bayesfilter_tuning_repair.py \
  tests/test_daily_asset_midas_l10d_bayesfilter_bootstrap_geometry_repair.py \
  tests/test_ccma_operational_broad_l_epsilon_neighbor_guard.py \
  tests/test_bayesfilter_macrofinance_fixed_kernel_tuning_screen.py \
  tests/test_mixed_frequency_tfp_c2_full_bayesfilter_hmc_tuning_v2_phase5t_real_tuning_loop.py
```

dsge_hmc (from `/home/ubuntu/python/dsge_hmc`, `tests/archive` excluded by
never listing it; known import-time segfault policy preserved):

```bash
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  BAYESFILTER_ROOT=/home/ubuntu/python/BayesFilter \
  PYTHONPATH=/home/ubuntu/python/BayesFilter \
  python -m pytest -q -p no:cacheprovider \
  tests/contracts/test_bgs_bayesfilter_stage_c.py \
  tests/contracts/test_bgs_bayesfilter_stage_c_grid_tuning.py \
  tests/contracts/test_bgs_mass_geometry_tf.py \
  tests/contracts/test_rotemberg_fixed_neutra_bayesfilter_xla_relaunch.py \
  tests/contracts/test_rotemberg_fixed_neutra_hmc_bridge.py \
  tests/contracts/test_rotemberg_fixed_neutra_xla_gate.py \
  tests/contracts/test_rotemberg_round380_neutra_fixed_hmc_reference.py \
  tests/regressions/test_mass_matrix.py
```

## 4. Results

| Repository | Baseline (pre-patch, handoff §6) | After repair | Wall time |
| --- | --- | --- | --- |
| MacroFinance focused | 60 passed / 4 failed of 64 | **64 passed**, 2 warnings | 6.26 s |
| dsge_hmc focused | 47 passed / 3 failed of 50 | **50 passed**, 12,136 warnings | 38.90 s |

Warnings are pre-existing TFP `distutils` and `gast`/autograph deprecation
noise; no new warning categories attributable to the repair.

Counts match the isolated-copy validation (64/64, 50/50) exactly.

## 5. Git State

- BayesFilter: `5699dafe` main, unchanged by this step (memo/doc additions
  only). Unrelated NeuTra/SSL-LSTM dirty files preserved.
- MacroFinance: `5e310d7` main. Only the two targeted files modified by this
  step; unrelated phase-14 diagnostic doc changes preserved untouched.
- dsge_hmc: `ba251192` main. Only the two targeted files modified; unrelated
  Rotemberg/BGS dirty files preserved untouched.
- Nothing committed, nothing pushed, per handoff instruction.

## 6. GPU Provenance

None claimed. This step's runs were deliberately CPU-hidden engineering
evidence (`CUDA_VISIBLE_DEVICES=-1`). No GPU-readiness statement is made or
implied. Step G canary remains open.

## 7. Decision Table

| Item | Status |
| --- | --- |
| Decision | Real consumer focused migration gates CLOSED (green) |
| Primary criterion | Focused suites: MacroFinance 64/64, dsge_hmc 50/50 — met |
| Veto diagnostics | None fired: no new failure fingerprints, `git diff --check` clean, only intended hunks, `candidate_workers=49` preserved, unrelated dirty files intact |
| Main uncertainty | Full-suite behavior (known MacroFinance collection blockers), unmigrated call-site debt, GPU path unexercised |
| Next justified action | Step D call-site classification (in progress), then Step E robust-driver test, then Step F full suites |
| Explicit nonclaims | No posterior correctness, no convergence, no sampler validity on real targets, no statistical ranking, no GPU/production readiness, no full-suite pass |

## 8. Evidence Classification

This step validates: patch mechanics against the real worktrees, and that the
previously adjudicated drift explanations were correct (each failure resolved
by exactly the predicted repair). It shows the consumer contracts and the
BayesFilter interfaces are consistent on the focused surface.

It does not validate: the harness beyond the focused surface, the scientific
targets, data, or math. It is one green cross-repository focused run; the
Phase-7 precondition of two green runs is not met, and the second run is
expected to be the post-Step-F full/focused rerun.
