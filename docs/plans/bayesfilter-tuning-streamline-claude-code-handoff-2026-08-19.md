# BayesFilter Tuning Streamline: Claude Code Handoff

Date: 2026-08-19
Prepared for: Claude Code continuation worker
Repository authority: `/home/ubuntu/python/BayesFilter`
Primary plan: `docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`

## 1. Mission

Continue the reviewed BayesFilter tuning-streamline refactor through the real
MacroFinance and dsge_hmc consumer worktrees. The immediate objective is to
apply the two already-audited consumer repair bundles, rerun the real focused
gates, and then proceed only as far as the evidence supports.

The intended architecture has exactly two active, artifact-authoritative HMC
tuning interfaces:

1. `bayesfilter.inference.tune_hmc_kernel`
2. `bayesfilter.inference.tune_fixed_transport_hmc_kernel`

Every other tuning/orchestration route is historical or diagnostic during the
migration window. Historical code and artifacts must remain readable, but they
must not issue canonical admission artifacts or support new claim-bearing
runs.

This memo is an execution handoff, not authorization to delete historical
routes, alter scientific targets, change default numerical policy, install
packages, commit, push, or publish anything.

## 2. Required Reading Before Edits

Read these files completely before modifying a consumer repository:

- `/home/ubuntu/python/BayesFilter/AGENTS.md`
- `/home/ubuntu/python/BayesFilter/CLAUDE.md`
- `/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`
- `/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-tuning-streamline-refactor-execution-result-2026-08-18.md`
- `/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-tuning-streamline-refactor-continuation-result-2026-08-19.md`
- `/home/ubuntu/python/MacroFinance/AGENTS.md`
- `/home/ubuntu/python/MacroFinance/CLAUDE.md`
- `/home/ubuntu/python/dsge_hmc/AGENTS.md`
- `/home/ubuntu/python/dsge_hmc/CLAUDE.md`

The BayesFilter governance profile is the controlling scientific policy. In
particular:

- TensorFlow/TFP is the algorithmic backend.
- NumPy is diagnostic/reference-only in BayesFilter runtime paths.
- GPU is the default execution target, but CPU-hidden runs are valid
  engineering/reference checks when explicitly labeled.
- GPU TensorFlow processes must set and verify
  `TF_FORCE_GPU_ALLOW_GROWTH=true` before device initialization.
- No tuning-only result establishes posterior correctness, convergence,
  sampler superiority, production readiness, or scientific validity.
- Mass, epsilon, and trajectory length (`L`) tuning is staged and conditional;
  a mass change invalidates the previous epsilon/`L` context and requires fresh
  retuning.

## 3. Current Repository State

### BayesFilter

Current commit at handoff:

```text
5699dafe (HEAD -> main, origin/main, origin/HEAD) Merge remote main into clean restart boundary
5699dafec23de9549a8092bec638997e7973593c
```

The BayesFilter checkout is dirty from unrelated and ongoing work, including
NeuTra/SSL-LSTM work. Preserve all unrelated changes. Do not use
`git reset --hard`, `git checkout --`, broad formatting, or cleanup commands.

Tuning-related implementation and test changes are already present in the
current checkout/commit history. The handoff adds only documentation and the
two patch bundles listed below. At the time of this memo, the relevant handoff
files are:

- `docs/plans/macrofinance-tuning-consumer-repair-2026-08-19.patch`
- `docs/plans/dsge-hmc-tuning-consumer-repair-2026-08-19.patch`
- `docs/plans/bayesfilter-tuning-streamline-claude-code-handoff-2026-08-19.md`

### MacroFinance

Current commit:

```text
5e310d7 (HEAD -> main, origin/main, origin/HEAD) Prepare clean repository restart snapshot
5e310d71c36fe124998e4b1cbacc1fd1d42bd660
```

The real MacroFinance worktree is dirty with unrelated phase-14 diagnostic
documents. The tuning consumer files targeted by the repair bundle were
unchanged at handoff. Preserve all unrelated dirty files.

### dsge_hmc

Current commit:

```text
ba251192 (HEAD -> main, origin/main, origin/HEAD) Define generated artifact boundary and restart state
ba251192ffcedc8606fb8eb5c083d08d66764e0d
```

The real dsge_hmc worktree is dirty with unrelated Rotemberg/BGS work. The
tuning consumer files targeted by the repair bundle were unchanged at handoff.
Preserve all unrelated dirty files.

## 4. What Is Already Implemented in BayesFilter

Do not reimplement or undo these pieces. Review them only if a consumer test
reveals a real contract mismatch.

### Canonical contract and route registry

`bayesfilter/inference/tuning_contract.py` provides:

- `HMCTuningScope`, binding target scope, adapter/coordinate/transport
  signatures, dimension, backend, dtype, XLA, and chain execution mode;
- `HMCTuningRouteRecord`; and
- `HMC_TUNING_ROUTE_REGISTRY`.

The registry currently contains exactly two `active` routes:

- `tune_hmc_kernel`
- `tune_fixed_transport_hmc_kernel`

The registry explicitly classifies robust broad-grid, fixed-metric,
operational-grid, fixed-mass ladder, generic orchestration, and tiny fixed-
trajectory routes as `diagnostic` or `historical`, with replacement
`tune_hmc_kernel` and nonclaims.

### AST inventory guard

`scripts/inventory_hmc_tuning_routes.py` discovers top-level tuning and
orchestration definitions and checks them against the registry. The latest
check reported:

- 10 discovered/registered entries;
- 0 unclassified entries;
- 0 stale registry entries.

Run:

```bash
cd /home/ubuntu/python/BayesFilter
python scripts/inventory_hmc_tuning_routes.py --check
```

Do not add a third active route to make a consumer test easier. If a truly
different numerical contract is discovered, stop and document it for review.

### Compatibility facades and public helpers

The extraction/compatibility surface includes:

- `hmc_artifacts.py`
- `hmc_artifact_identity.py`
- `hmc_bootstrap.py`
- `hmc_budget_policy.py`
- `hmc_geometry.py`
- `hmc_kernel_stages.py`
- `hmc_mass_adaptation.py`

`hmc_artifacts.mass_artifact_signature` is the public canonical mass-signature
home. Consumer code must not keep importing private duplicate helpers from
`hmc_kernel_tuning`.

### Mass-matrix behavior

The ordinary tuner supports the reviewed geometry/mass policies, including
`windowed_adaptive` and `fixed_identity`. The fixed-identity operational path
now preserves the original artifact identity when mass updates are forbidden.
Every actual mass update invalidates the prior epsilon/trajectory context and
requires fresh tuning evidence.

The mass contract distinguishes:

- position-coordinate covariance/preconditioner `Sigma_theta`;
- factor orientation and reconstruction;
- latent-coordinate transform; and
- downstream TFP momentum covariance/precision.

Do not call a warmup covariance estimate the posterior covariance merely
because it came from draws.

### Robust broad-grid behavior

`bayesfilter/inference/hmc_robust_broad_grid.py` is retained as a diagnostic or
compatibility strategy. It is not an independent canonical artifact authority.
Its target-specific `L` grid and qualification controls now carry explicit
configuration/provenance rather than silently forcing one universal grid.

Operational acceptance-screen budgets are floored high enough to support the
declared evidence; public mechanics presets remain small where they are
explicitly diagnostic.

### Posterior oracle

`tests/test_hmc_tuning_posterior_oracle.py` contains:

- an exact shifted, correlated 2-D Gaussian value/score oracle;
- TensorFlow gradient and batched/scalar adapter checks;
- wrong mean/covariance/score negative controls;
- a nontrivial affine fixed transport with Jacobian and score pullback;
- ordinary tuner calibration followed by independent holdout;
- fixed-transport calibration followed by affine holdout;
- four mass arms: identity, exact covariance, adapted covariance, and
  precision mistakenly supplied in the covariance role;
- fresh mass-specific dual averaging and `L` retuning; and
- finite-state, target-status, divergence, R-hat/ESS, analytic-moment, and
  MCSE-aware checks.

The oracle is an engineering/numerical validity gate on a known target. It is
not evidence of superiority or universal correctness.

### Audit and review provenance

The plan was built from the committed audit trail. Claude should use these as
source evidence when a route/function classification is unclear:

- `docs/audits/bayesfilter-tuning-function-audit-2026-08-16.md`
- `docs/audits/bayesfilter-tuning-definition-inventory-2026-08-16.md`
- `docs/audits/bayesfilter-tuning-adjacent-authority-inventory-2026-08-16.md`
- `docs/audits/bayesfilter-tuning-repo-wide-reference-inventory-2026-08-16.md`
- `docs/plans/bayesfilter-tuning-fable-review-verdict-2026-08-16.md`
- `docs/plans/bayesfilter-tuning-fable-review-verdict-r2-2026-08-17.md`
- `docs/plans/bayesfilter-tuning-fable-coverage-audit-verdict-2026-08-17.md`
- `docs/plans/bayesfilter-tuning-fable-coverage-audit-verdict-r2-2026-08-17.md`
- `docs/plans/bayesfilter-tuning-fable-coverage-audit-response-2026-08-17.md`

Do not treat an old plan or historical result as live authority when it
conflicts with current source, current governance, or the revised streamline
plan.

## 5. BayesFilter Verification Already Obtained

All commands below used the `tfgpu` environment. The first group was CPU-hidden
engineering evidence:

```bash
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  python -m pytest -q \
  tests/test_hmc_tuning_contract.py \
  tests/test_hmc_mass_matrix.py \
  tests/test_hmc_windowed_mass_adaptation.py \
  tests/test_hmc_robust_broad_grid.py \
  tests/test_fixed_transport_hmc_tuning.py \
  tests/test_hmc_kernel_tuning_public_api.py \
  tests/test_hmc_tuning_posterior_oracle.py
```

Result: **106 passed** in approximately 194 seconds.

Additional BayesFilter checks:

- compileall: passed;
- mass/geometry/windowed regression subset: **79 passed, 1 skipped, 203
  deselected**;
- fixed-grid/route-selection/handoff/robust-grid suite: **159 passed**;
- `git diff --check`: passed; and
- route inventory `--check`: passed.

The same 106-test command was also run in a trusted GPU-visible context with:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

TensorFlow 2.20.0 saw both NVIDIA GeForce RTX 4080 SUPER devices and memory
growth was verified `True` for both. The suite again passed **106 tests** in
approximately 209 seconds. This establishes device visibility and allocator
provenance for that process; it is not a claim that every test used the GPU or
that the tuner is faster or better.

## 6. Real Consumer Baseline and Exact Failures

These are results from the real repositories before the repair bundles were
applied.

### MacroFinance real baseline

Live focused command:

```bash
cd /home/ubuntu/python/MacroFinance
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

Result: **60 passed, 4 failed** out of 64 collected.

Failures and adjudication:

1. `test_config_for_preset_can_enable_bayesfilter_staged_timeout_policy`
   expected stale `ccma_phase4y_stage_budget_v1`. The live repository-owned
   default is `bayesfilter_hmc_emergency_stage_caps_v2`.
2. `test_evaluate_can_pass_default_staged_timeout_policy` had the same stale
   policy-identifier expectation.
3. `test_evaluate_appends_external_covariance_arm` rejected every textual
   occurrence of `0.25`; that value also appears in legitimate budget metadata.
   The correct structural checks are absence of public mass matrices and
   `raw_matrix_values_publicized == False`.
4. `test_trial_config_preserves_validation_defaults_except_step_size` treated
   `chain_execution_mode` as a tuning control. The historical screen
   reconstructs the same fixed kernel through the public runner; this execution
   detail is not the varied tuning parameter.

### dsge_hmc real baseline

Live focused command:

```bash
cd /home/ubuntu/python/dsge_hmc
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

Result: **47 passed, 3 failed** out of 50 collected.

Failures and adjudication:

1. The relaunch test expected old selection rule
   `shortest_leapfrog_acceptance_in_band_then_diagnostics`; the committed rule
   is `eligible_trajectory_acceptance_in_band_then_rhat_convergence_then_ess`.
2. The same test expected 49 policy candidates; current policy generates 63.
   The operational worker cap remains 49 and must not be changed to 63.
3. The Rotemberg bridge failed because `FlowFixedTransportAdapter` lacked
   `pullback_score`, `pullback_score_batch`,
   `log_abs_det_jacobian_score`, and
   `log_abs_det_jacobian_score_batch`.

The known `tests/archive` import-time segfault remains excluded by policy.

## 7. Ready-to-Apply Repair Bundles

The bundles are in BayesFilter and were validated in isolated copies of the
current consumer source trees:

- `docs/plans/macrofinance-tuning-consumer-repair-2026-08-19.patch`
- `docs/plans/dsge-hmc-tuning-consumer-repair-2026-08-19.patch`

Expected SHA-256 checksums at handoff:

```text
6408f92adba53e3b05dca124d0bbae04699d44ba49b61e44cc31f47a5330a824  macrofinance-tuning-consumer-repair-2026-08-19.patch
b8b6d4f692fdb59f4bbf0c5fa9f089eed3c4ccc889bb2974d99e930b3c5672e7  dsge-hmc-tuning-consumer-repair-2026-08-19.patch
```

Verify them before application:

```bash
cd /home/ubuntu/python/BayesFilter
sha256sum \
  docs/plans/macrofinance-tuning-consumer-repair-2026-08-19.patch \
  docs/plans/dsge-hmc-tuning-consumer-repair-2026-08-19.patch
```

They are ordinary `apply_patch` input, with paths relative to each consumer
root:

```bash
cd /home/ubuntu/python/MacroFinance
apply_patch < /home/ubuntu/python/BayesFilter/docs/plans/macrofinance-tuning-consumer-repair-2026-08-19.patch

cd /home/ubuntu/python/dsge_hmc
apply_patch < /home/ubuntu/python/BayesFilter/docs/plans/dsge-hmc-tuning-consumer-repair-2026-08-19.patch
```

The MacroFinance bundle changes only the two timeout-policy expectations, the
brittle redaction assertion, and the historical execution-mode invariant.

The dsge_hmc bundle changes only the selection rule and candidate-count
expectations plus the four public frozen-transport score/Jacobian methods. It
delegates to flow-native methods when available, otherwise uses TensorFlow
`GradientTape`, fails closed on a disconnected transport pullback, and returns
zero for a legitimately constant log-Jacobian.

Do not alter `candidate_workers=49`: it is a CPU worker cap, not the policy
candidate count.

## 8. Isolated Patch Validation Evidence

Because the cross-root mutation gateway was unavailable, temporary copies were
created under `/tmp` from the live source trees. Large result/model archives
were omitted initially; the dsge model source tree and Git metadata were then
added because its tests require them. No real consumer files were changed.

MacroFinance after patch: **64 focused tests passed**.

dsge_hmc after patch: **50 focused tests passed**. Independent subsets also
passed: Rotemberg bridge **3 passed** and BGS grid contract **8 passed**.

These isolated passes validate the repair logic. They are not a real-worktree
migration pass until the bundles are applied to the actual repositories.

The temporary validation roots were:

- `/tmp/macrofinance-tuning-repair-20260819`
- `/tmp/dsge-hmc-tuning-repair-20260819`

They are disposable validation copies, not source authority. Do not copy whole
trees or artifacts back from `/tmp`; apply the reviewed patch bundles to the
real repositories instead.

### Prior cross-root gateway failure history

Codex could read both consumer repositories but its managed workspace allowed
writes only under BayesFilter and `/tmp`. Attempts to apply the same bounded
consumer changes through the permission gateway failed before process/file
mutation with these infrastructure responses over successive sessions:

- repeated HTTP 502 responses from the automatic patch reviewer;
- one approval timeout followed by HTTP 429 retry exhaustion; and
- after VS Code restart, repeated HTTP 404 responses stating
  `model is not available: gpt-5.6-luna`.

The user explicitly approved the cross-repository edits. Verification after
each failure showed the real targeted files unchanged. This is infrastructure
history, not a scientific or implementation veto. Claude Code should still
inspect `git status` and the exact target hunks before applying anything.

## 9. Current Consumer Call-Site Debt After the Repair

Applying the bundles repairs only the focused contract drift. It does not
complete Phase 5 or Phase 6 migration. Perform a fresh source scan after
applying them.

### MacroFinance notable uses

Claim-bearing or potentially active paths still requiring review include:

- `daily_asset_midas_robust_broad_grid_tuning.py`, which imports/calls
  `tune_hmc_kernel_robust_broad_grid`;
- `mixed_frequency_tfp_c2_full_bayesfilter_hmc_tuning_v2_phase4_step_trajectory.py`,
  which calls `run_fixed_mass_hmc_tuning_budget_ladder` and
  `orchestrate_generic_hmc_tuning`;
- `mixed_frequency_tfp_c2_full_bayesfilter_hmc_tuning_v2_phase5t_real_tuning_loop.py`,
  which refers to the fixed-mass ladder;
- `cross_country_multi_asset_bayesfilter_owned_hmc_client.py` and
  `cross_country_multi_asset_bayesfilter_mass_preconditioner.py`, which refer
  to generic orchestration;
- many diagnostic/retained scripts using private `_mass_artifact_signature`;
  and
- retained synthetic scripts importing private fixed-mass/bootstrap helpers.

The many phase-5/10 retained-memory, shape-validation, and diagnostic files
must be classified individually. A diagnostic file may remain on a historical
route only when its source/result explicitly says diagnostic or historical and
it cannot issue canonical admission evidence. A claim-bearing caller must
migrate to `tune_hmc_kernel` or the fixed-transport interface.

Replace allowed private mass-signature imports with
`bayesfilter.inference.hmc_artifacts.mass_artifact_signature`. Replace private
fixed-mass/bootstrap imports with the public compatibility exports. Do not
change numerical behavior while moving imports.

### dsge_hmc notable uses

Review at least:

- `scripts/run_bgs_bayesfilter_phase08_stage_c_grid_tuning.py` and its tests;
- `scripts/run_rotemberg_fixed_neutra_bayesfilter_tuning_smoke.py`;
- public explicit-state fixed-mass/tuning diagnostics;
- `src/dsge_hmc/experiment_adapters/rotemberg_round380_neutra.py` private
  mass-signature usage; and
- remaining private mass/bootstrap imports discovered by `rg`.

The BGS Stage-C grid may remain a documented diagnostic/candidate-policy route
if its aggregate policy is preserved and it cannot issue canonical admission
artifacts. A claim-bearing Rotemberg fixed-NeuTra tuner should use
`tune_fixed_transport_hmc_kernel`.

### Required inventory command

From each consumer root, run and preserve the output of:

```bash
rg -n "tune_hmc_kernel_robust_broad_grid|run_fixed_mass_hmc_tuning_budget_ladder|run_fixed_metric_grid_search|run_operational_broad_grid|orchestrate_generic_hmc_tuning|run_generic_hmc_tuning_orchestration|_mass_artifact_signature|_build_fixed_mass_hmc_adapter|_build_bootstrap_fixed_mass_adapter|_BootstrapFixedMassLatentValueScoreAdapter|tune_fixed_transport_hmc_kernel" --glob '*.py'
```

Classify each result as active claim-bearing, diagnostic, historical, test, or
documentation-only. Inspect the producer, consumer, artifact role, and
nonclaims; do not classify from the name alone.

## 10. Required Next Sequence

### Step A: Preserve and inspect

From each real consumer root:

```bash
git status --short --untracked-files=all
git diff --check
git log -1 --oneline --decorate
```

Do not revert unrelated changes. Confirm targeted files are still unchanged
before applying each bundle.

### Step B: Apply and verify bundles

Apply the commands in Section 7. Immediately run:

```bash
git diff --check
git diff -- <each targeted file>
```

Confirm the diff contains only the intended hunks. Do not commit or push unless
the user separately requests it.

### Step C: Run real focused matrices

Use the exact live-path commands in Section 6. The first pass is deliberately
CPU-hidden engineering evidence. Record:

- command exactly as run;
- environment and package versions;
- collected/pass/fail/skip counts;
- failure fingerprints;
- repository commit/status;
- elapsed wall time; and
- whether `tests/archive` was excluded.

Expected after the bundles: MacroFinance 64 passing focused tests and dsge_hmc
50 passing focused tests, subject to unrelated worktree drift. Classify any new
failure before changing code.

### Step D: Run source/import audits and migrate callers

Run Section 9's inventory and adjudicate each live caller. Migrate one consumer
family at a time with focused tests. Preserve explicitly historical/diagnostic
callers and their nonclaims; do not turn them into canonical evidence by
renaming a field.

### Step E: Add missing MacroFinance robust-driver coverage

The live MacroFinance tree has no
`tests/test_daily_asset_midas_robust_broad_grid_tuning.py`. This remains a
migration gap. Add it only after reading the driver and its current contract.
Cover:

- import from the intended public path;
- configuration/provenance handoff;
- progress and artifact boundaries;
- canonical-interface selection or explicit diagnostic classification; and
- absence of MacroFinance-local HMC mechanics.

Do not report this test as run before it exists.

### Step F: Run configured full suites

After focused suites and source scans are green, run each configured full suite
from its repository root. Preserve `tests/archive` exclusion for dsge_hmc due
to the known import-time segfault. Record collection errors by category rather
than hiding them.

MacroFinance historically had missing optional `pandas` and order-dependent
BayesFilter checkout-resolution collection errors in `tfgpu`. Resolve or
explicitly record each blocker; never call a partial suite a full-suite pass.

### Step G: GPU/XLA canary only after CPU contracts

For target-specific GPU evidence, use trusted/elevated execution and configure
memory growth before TensorFlow import:

```bash
conda run -n tfgpu env \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python <target-canary>
```

Use the repository memory-policy helper where the target owns TensorFlow
initialization. Record both physical devices, growth values, TF32, XLA/JIT,
device placement, command, seed, wall time, and artifact path. A CPU-hidden
test is not GPU-readiness evidence.

### Step H: Require two green runs before cleanup

Only after two green cross-repository focused/full migration runs may Claude:

- quarantine historical routes from default exports;
- add the new claim-bearing-source import guard; or
- consider deleting dead compatibility code.

Never delete historical artifacts or readers in this task.

## 11. Stop Conditions

Stop and write a result/reset note without deleting code if any of these occur:

- a real consumer imports an unclassified tuning route;
- a focused test fails for a reason not isolated to audited drift repair;
- old/new deterministic fixture outputs disagree beyond declared tolerance;
- an artifact accepts missing, stale, mismatched, or caller-stamped scope;
- a private helper is needed by a claim-bearing caller and no public semantic
  replacement is proven;
- GPU memory growth cannot be verified before initialization;
- a command cannot establish which test paths ran;
- a full-suite collection/import error is silently hidden; or
- a proposed repair changes target, data, hardware class, privacy boundary,
  evidence criteria, or campaign budget.

A failed candidate or tuning screen is a repair trigger, not evidence against
the HMC research direction. Separate implementation failure, tuning failure,
diagnostic failure, and evidence against the scientific idea.

## 12. Expected Reporting Format

At the end of continuation, write a result memo with:

1. Findings first: unresolved failures, blockers, or contract risks.
2. Exact files changed, with line references.
3. Commands and environments actually run.
4. Pass/fail/skip/collection counts for each repository.
5. Git commits/status and confirmation that unrelated dirty files remain.
6. GPU provenance, if any, including memory-growth verification.
7. A decision table: primary criterion, veto status, uncertainty, next action,
   and explicit nonclaims.
8. Whether evidence validates the harness, implementation, target, data, math,
   or merely shows a candidate/consumer contract failed.

Do not report isolated-copy results as real-repository passes.

## 13. Bounded Claude Worker Prompts

Use separate non-interactive workers so each has a small write set. Workers
must not commit or push.

### MacroFinance worker

```text
READ-WRITE BOUNDED MIGRATION TASK.

Read completely:
  /home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-tuning-streamline-claude-code-handoff-2026-08-19.md
  /home/ubuntu/python/MacroFinance/AGENTS.md
  /home/ubuntu/python/MacroFinance/CLAUDE.md

Allowed write set for the first invocation only:
  tests/test_daily_asset_midas_l10d_bayesfilter_bootstrap_geometry_repair.py
  bayesfilter_macrofinance_fixed_kernel_tuning_screen.py

Apply only:
  /home/ubuntu/python/BayesFilter/docs/plans/macrofinance-tuning-consumer-repair-2026-08-19.patch

Verify the exact diff and run the focused MacroFinance command from the handoff.
Do not touch unrelated dirty files, edit BayesFilter, install packages, commit,
or push. Report findings first, then changed files, exact commands, test counts,
residual risks, and whether the real-worktree focused gate passed.
```

Launch pattern:

```bash
bash /home/ubuntu/python/claudecodex/scripts/claude_worker.sh \
  --cwd /home/ubuntu/python/MacroFinance \
  --name bayesfilter-tuning-macrofinance-repair \
  "<prompt above>"
```

### dsge_hmc worker

```text
READ-WRITE BOUNDED MIGRATION TASK.

Read completely:
  /home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-tuning-streamline-claude-code-handoff-2026-08-19.md
  /home/ubuntu/python/dsge_hmc/AGENTS.md
  /home/ubuntu/python/dsge_hmc/CLAUDE.md

Allowed write set for the first invocation only:
  tests/contracts/test_rotemberg_fixed_neutra_bayesfilter_xla_relaunch.py
  src/dsge_hmc/experiment_adapters/fixed_neutra_hmc.py

Apply only:
  /home/ubuntu/python/BayesFilter/docs/plans/dsge-hmc-tuning-consumer-repair-2026-08-19.patch

Preserve candidate_workers=49. Verify the exact diff and run the focused
dsge_hmc command from the handoff with tests/archive excluded. Do not touch
unrelated dirty files, edit BayesFilter, install packages, commit, or push.
Report findings first, then changed files, exact commands, test counts, residual
risks, and whether the real-worktree focused gate passed.
```

Launch pattern:

```bash
bash /home/ubuntu/python/claudecodex/scripts/claude_worker.sh \
  --cwd /home/ubuntu/python/dsge_hmc \
  --name bayesfilter-tuning-dsge-repair \
  "<prompt above>"
```

Claude workers must be launched through the narrow wrapper and with
trusted/elevated execution when supervised by Codex.

## 14. Decision State at Handoff

| Area | State | Meaning |
| --- | --- | --- |
| BayesFilter canonical interfaces | GREEN | Two active routes; route ledger, contracts, oracle, and mass tests pass |
| BayesFilter GPU visibility | GREEN | Two GPUs visible; memory growth verified on both in trusted context |
| MacroFinance repair logic | GREEN IN ISOLATED COPY | 64 focused tests pass after bundle; real worktree unpatched |
| dsge_hmc repair logic | GREEN IN ISOLATED COPY | 50 focused tests pass after bundle; real worktree unpatched |
| Real MacroFinance migration gate | OPEN | Apply bundle and rerun real focused suite |
| Real dsge_hmc migration gate | OPEN | Apply bundle and rerun real focused suite |
| Full consumer suites | NOT RUN AFTER REPAIR | Must follow real focused green result |
| Phase 7 quarantine/cleanup | FORBIDDEN YET | Requires two green cross-repository runs |

The safest immediate action is to apply the two bundles to the real consumer
roots, verify their diffs, and rerun the exact focused commands. Everything
needed for that bounded first step is preserved here and in the patch files.

## 15. Evidence Ledger and Nonclaims

| Ledger | Evidence currently available | What it supports | What it does not support |
| --- | --- | --- | --- |
| Engineering correctness | BayesFilter focused suites green; route inventory clean; isolated consumer matrices green | Contract implementation, route classification, patch mechanics | Scientific validity or production readiness |
| Numerical/sampler validity | Exact Gaussian and affine-transport oracle, mass arms, target-status/divergence/R-hat/ESS/moment checks | Known-target tuner adequacy and mass/epsilon/L integration on the fixture | Adequacy for every nonlinear/domain target |
| Consumer compatibility | Real pre-patch failure fingerprints plus isolated post-patch 64/64 and 50/50 | Strong evidence the bounded repairs resolve known drift | Real-worktree pass until patches are applied there |
| GPU execution | Two GPUs visible to TensorFlow; memory growth true on both; 106-test suite passes in GPU-visible process | Trusted device/allocator provenance for that process | Per-test GPU utilization, performance ranking, or GPU default readiness for every consumer |
| Statistical ranking | No predeclared uncertainty-supported method ranking | Nothing beyond viability screens | No claim that one mass arm, tuner, grid, or sampler is best/superior |
| Default readiness | Two canonical interfaces selected by reviewed architecture | Interface direction and migration target | No numeric robust-grid default promotion without target-specific stress evidence |

Hard-veto evidence takes precedence over runtime, acceptance, ESS, and other
descriptive metrics. Passing a screen means the candidate remains viable under
that screen; it does not establish superiority. A failed mass arm or kernel is
a repair trigger unless it invalidates the target, harness, math, or artifact.
