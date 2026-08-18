# Fable Handoff Memo: BayesFilter Tuning Audit and Refactor Plan

To: Fable
From: Codex
Date: 2026-08-16
Review mode: read-only audit first; do not edit source or launch experiments
unless a follow-up task explicitly authorizes it.

## Review Targets

1. `docs/audits/bayesfilter-tuning-function-audit-2026-08-16.md`
2. `docs/audits/bayesfilter-tuning-definition-inventory-2026-08-16.md`
3. `docs/audits/bayesfilter-tuning-adjacent-authority-inventory-2026-08-16.md`
4. `docs/audits/bayesfilter-tuning-repo-wide-reference-inventory-2026-08-16.md`
5. `docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`
6. `/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_robust_broad_grid.py`

The worktree is intentionally dirty with pre-existing user changes. Preserve
all unrelated changes.

## Codex Response to Verdict

Fable's verdict is accepted as `AUDIT_VERDICT: REVISE` and
`PLAN_VERDICT: REVISE`. The following repairs were applied after review:

- Added the missing inference-package exports for
  `HMCStagedTimeoutPolicy`, `prepare_fixed_transport_hmc_adaptive_joint_grid_policy`,
  and `prepare_fixed_transport_hmc_joint_grid_policy`.
- Added the omitted `staged_fixed_kernel_hmc.py` inventory row and the missing
  `hmc_kernel_selection.py` semantic summary.
- Distinguished the 482-definition adjacent execution/evidence subset from
  the 271-definition direct dependency addendum; the 1,417 count remains the
  core tuning surface only.
- Marked the original broad 113/16 consumer totals and 3,553 repo-wide
  reference count provisional because their historical union pattern was not
  recorded reproducibly.
- Assigned `hmc_kernel_selection.py`, direct dependencies, and the single
  artifact-helper authority explicitly to Phase 2.
- Generalized the Phase-3 requirement to remove the robust route's frozen
  L-grid/500-qualification constraints before target-specific policy review.
- Changed test gates to baseline-relative gates, named `tfgpu`, set
  `BAYESFILTER_ROOT` for dsge_hmc, and explicitly recorded the CPU-hidden GPU
  exclusions and dsge_hmc `tests/archive` omission.

Verification after the export repair:

- `tests/test_fixed_transport_hmc_grid_policy.py`: 24 passed.
- MacroFinance L10d focused file: 20 tests collected.
- MacroFinance six-file focused tuning collection: 101 tests collected.
- dsge_hmc three-file focused contract collection with `BAYESFILTER_ROOT`:
  26 tests collected.
- BayesFilter CPU-hidden collection with two GPU-only files ignored: 7,447
  tests collected and 3 unrelated pre-existing collection errors remain.

These are engineering compatibility/collection results only; no GPU or HMC
campaign was run and no scientific or convergence claim follows.

## Question A: Audit Completeness

Verify that the audit found every current tuning-related function/class and did
not silently omit a route.

### Required checks

Run a fresh AST inventory over BayesFilter files whose names or contents match
the audit scope (`tune`, `grid`, `kernel`, `budget`, `capacity`, `broad`,
`scope`, `registry`, `frozen`, `trajectory`, `staged`) and compare it with the
audit's 26-file/1,417-definition inventory. Report:

- files present in source but absent from the audit;
- definitions present in source but absent from the module summaries;
- definitions classified as active by code/import evidence but called
  historical/diagnostic in the audit;
- public exports in `bayesfilter/inference/__init__.py` missing from the audit;
- new uncommitted tuning files not covered by the 26-file selection; and
- any false positives that should be excluded, with a reason.

Then inspect the adjacent authority inventory (`hmc.py`, verification,
convergence, warmup, mass, fixed-transport mechanics/candidate discovery, and
posterior adapter). Confirm that its 482 definitions are the relevant dependency
closure or identify missing adjacent modules. Do not treat the 1,417 core count
as the whole HMC tuning universe.

Also inspect the repo-wide reference inventory. It currently finds 3,553 files
matching tuning/grid/kernel/budget terms, including 3,314 documentation or
structured-artifact paths. Confirm that active readers are classified and that
historical documents are not accidentally promoted to executable requirements.

The prior 3,553/3,314 counts are now provisional: the original exact matching
command was not recorded and Fable could not reproduce it. Recompute them with
an explicit command, record the pattern/roots/exclusions, and correct the audit
instead of preserving unsupported numbers. Likewise, treat the broad
MacroFinance 113 and dsge_hmc 16 totals as unsupported union counts; retain the
reproducible symbol-specific counts and classify archival
`results/**/functional_source_snapshot` copies separately.

The line-level appendix is intended to make the phrase "each function" auditable:
it lists every class/function path and line and gives either its first docstring
line or an AST behavioral fingerprint (calls, returns, raises, and loops).
Check that this generated description is not mistaken for a mathematical
correctness proof; flag any public or claim-bearing function whose body needs a
manual semantic review.

Use `rg` to find all definitions and all external references, not only filename
matches. Search both direct imports and string/source checks.

### External consumer completeness

Independently inventory `/home/ubuntu/python/MacroFinance` and
`/home/ubuntu/python/dsge_hmc`, excluding `.git`, `.claude`, and caches. Check
for all uses of:

- `tune_hmc_kernel`;
- `tune_hmc_kernel_robust_broad_grid`;
- `tune_fixed_transport_hmc_kernel`;
- `run_fixed_metric_grid_search`;
- `run_operational_broad_grid`;
- `run_fixed_mass_hmc_tuning_budget_ladder`;
- `run_generic_hmc_tuning_orchestration` and
  `orchestrate_generic_hmc_tuning`;
- imports of `bayesfilter.inference.hmc_kernel_tuning` and private names such
  as `_mass_artifact_signature` or `_BootstrapFixedMassLatentValueScoreAdapter`;
- source-level checks for BayesFilter entrypoint strings and artifact schemas;
  and
- dynamic `__import__`, importlib, or string locator routes that evade normal
  import scanning.

The audit currently reports 113 MacroFinance tuning-related files and 16
dsge_hmc files. Confirm or correct these counts and list every discrepancy.

### Required audit verdict

End this section with exactly one of:

- `AUDIT_VERDICT: COMPLETE`
- `AUDIT_VERDICT: REVISE`

If `REVISE`, list each missed file/function/call site with a path and line.

## Question B: Plan Correctness and Coverage

Audit the refactor plan against the actual source and consumers. Check each
phase, not only the headline recommendation.

### Architecture checks

- Is `tune_hmc_kernel` a defensible ordinary-HMC compatibility boundary given
  current MacroFinance use?
- Is `tune_fixed_transport_hmc_kernel` genuinely a separate contract given
  dsge_hmc's transformed-coordinate and transport-manifest requirements?
- Does the plan keep LEDH scope/control tuning separate from HMC epsilon/L
  tuning, as required by the repository policy?
- Does the plan preserve NeuTra training/search as a separate batch-native
  GPU/XLA concern rather than conflating it with kernel tuning?
- Are operational broad-grid, fixed-metric, budget-ladder, generic, and old
  phase routes correctly classified as diagnostic/compatibility rather than
  deleted immediately?

### Numerical and evidence checks

- Does the plan reject the robust route's `use_xla=False`, fixed L grid, and
  short 500-draw qualification as universal defaults until reviewed?
- Are acceptance, ESS, R-hat, runtime, and repair counts assigned the correct
  promotion/veto/explanatory roles?
- Does the plan prevent a one-seed or short-chain ranking from being called
  superiority?
- Are calibration/verification data and fresh repair seeds disjoint?
- Are scope, adapter, transport, source dependency, dtype/backend, TF32/XLA,
  and chunk-policy identities bound in artifacts?
- Are GPU memory-growth and CPU-hidden test requirements explicit?
- Does the plan forbid NumPy numerical paths in admitted TensorFlow/TFP tuning?

### Compatibility and test checks

- Does Phase 1 provide stable replacements before private dsge_hmc imports are
  removed?
- Does Phase 5 cover all MacroFinance ordinary, robust, fixed-metric,
  operational, budget-ladder, source-scan, and test consumers?
- Does Phase 6 cover BGS stage-C grid, Rotemberg fixed-transport smoke, private
  mass/bootstrap imports, artifact schema tests, XLA tests, and no-local-sampler
  contracts?
- Do the commands run tests from the correct repository with BayesFilter on
  `PYTHONPATH`?
- Does the plan detect missing test files instead of claiming they ran?
- Does the final dsge_hmc command actually honor its `pyproject.toml` testpaths?
- Are focused tests run before full suites, and are GPU tests reserved for a
  later trusted canary?

### Required plan verdict

End this section with exactly one of:

- `PLAN_VERDICT: AGREE`
- `PLAN_VERDICT: REVISE`

If `REVISE`, distinguish a blocking correctness gap from a nonblocking cleanup
recommendation. Do not approve the plan merely because it is internally tidy.

## Question C: Test Adequacy

Review whether the proposed tests can answer the stated migration question.
Mark each item `covered`, `partially covered`, or `missing`, with the test path
that establishes the status.

1. Route classification/discovery and duplicate active exports.
2. Old/new deterministic mechanics parity.
3. Artifact scope/source/adapter/transport mismatch fail-closed behavior.
4. No public leakage of raw/private samples or mass state.
5. Target-status, finiteness, divergence, acceptance, and R-hat/ESS evidence
   role separation.
6. Fresh calibration/verification seed lineage.
7. MacroFinance source-level call-count and forbidden-import assertions.
8. MacroFinance robust-driver and ordinary-tuner mocks. Note: no dedicated
   robust-driver test currently exists; this is a Phase 5 deliverable, not a
   passing test today.
9. MacroFinance fixed-metric and operational diagnostic routes.
10. dsge_hmc BGS grid callback/aggregate behavior.
11. dsge_hmc Rotemberg fixed-transport tuning, XLA, and artifact contracts.
12. Replacement of private `hmc_kernel_tuning` imports.
13. Configured full-suite discovery in both external repositories.

## Bounded Commands

These are read-only or CPU-hidden checks. Do not run GPU/CUDA commands in this
handoff review.

```bash
cd /home/ubuntu/python/BayesFilter
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 python -m compileall -q bayesfilter tests
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 python -m pytest --collect-only -q tests \
  --ignore=tests/test_neural_force_hmc_gpu.py \
  --ignore=tests/test_neural_force_training_gpu.py
rg -n '^(async )?(def|class) ' bayesfilter/inference bayesfilter/highdim
rg -n 'tune_hmc_kernel|tune_fixed_transport_hmc_kernel|run_fixed_metric_grid_search|run_operational_broad_grid|run_fixed_mass_hmc_tuning_budget_ladder|hmc_kernel_tuning' /home/ubuntu/python/MacroFinance /home/ubuntu/python/dsge_hmc --glob '*.py' --glob '!**/.git/**' --glob '!**/.claude/**'

cd /home/ubuntu/python/MacroFinance
conda run -n tfgpu env PYTHONPATH=/home/ubuntu/python/BayesFilter:$PWD CUDA_VISIBLE_DEVICES=-1 python -m pytest --collect-only -q

cd /home/ubuntu/python/dsge_hmc
conda run -n tfgpu env PYTHONPATH=/home/ubuntu/python/BayesFilter:$PWD BAYESFILTER_ROOT=/home/ubuntu/python/BayesFilter CUDA_VISIBLE_DEVICES=-1 python -m pytest --collect-only -q --ignore=tests/archive
```

If a command is too broad for the review budget, narrow it by path and record
the omitted path. Do not launch full HMC campaigns.

## Findings to Challenge Explicitly

The audit's strongest claims that require independent scrutiny are:

- the exact 1,417-definition count;
- the claim that MacroFinance's active ordinary interface can remain named
  `tune_hmc_kernel`;
- the claim that fixed transport warrants the second interface rather than a
  strategy parameter;
- the claim that the robust broad grid is the best current ordinary-HMC
  candidate despite `use_xla=False` and short qualification;
- the boundary between historical/diagnostic fixed-metric routes and active
  consumer contracts; and
- whether any LEDH or NeuTra tuner was incorrectly excluded from the HMC audit
  or incorrectly included in the two-interface consolidation.
- whether the proposed test paths exist. In the current checkout the correct
  BayesFilter fixed-transport test is
  `tests/test_fixed_transport_hmc_tuning.py`, and MacroFinance lacks a
  dedicated robust-broad-grid test.

## Handoff Output

Write a concise review result in your response or a new dated memo containing:

| Section | Required output |
|---|---|
| Audit delta | Missed/extra files, functions, exports, and consumer call sites. |
| Consumer map | MacroFinance and dsge_hmc usage, including private/source-level contracts. |
| Plan findings | Blocking flaws, nonblocking risks, and corrected phase/gate text. |
| Test matrix | 13-item coverage status with paths. |
| Verdicts | `AUDIT_VERDICT` and `PLAN_VERDICT`. |
| Residual uncertainty | What was not checked and the smallest next artifact. |

Do not emit `AGREE` if the source anchors or consumer tests were not actually
inspected. A plan review that only checks internal prose is insufficient.
