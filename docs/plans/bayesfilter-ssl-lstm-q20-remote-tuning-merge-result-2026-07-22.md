# q=20 Tuning Remote-Merge Audit Result

Date: 2026-07-22  
Tier: 2 material sampler-engineering integration check  
Status: `MERGE_REPAIRED_FOCUSED_CHECKS_PASS`

## Question

Does the remote tuning/HMC change affect the q=20 SSL-LSTM NeuTra lane, and is
the merged source executable without reusing stale tuning evidence?

## Source State

- Remote was fetched and fast-forward merged through `2ace1bb` (`Correct PP-UKF
  statistical tuning classification`), with `HEAD == origin/main` before the
  local repair edits.
- Existing unrelated dirty worktree edits were preserved.
- The in-flight q=20 run was stopped after the merge because its source
  signature preceded the merge. Its checkpoint and timing are retained as
  debugging/resource evidence only; they are not current-source tuning or HMC
  evidence.

## Impact Found

The remote HMC changes materially affect this lane. Shared HMC and fixed
transport tuning now require `bayesfilter.inference.hmc_convergence` for modern
rank-normalized/folded split R-hat verification. That module was absent from the
merged tree, so affected tests failed at collection with
`ModuleNotFoundError`.

The merged source also exposed an accounting defect in retained target-status
telemetry: the verifier stopped after the first failing logical draw instead of
counting all failures in the retained chunk.

## Repairs

1. Added [hmc_convergence.py](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_convergence.py),
   delegating rank-normalization and ESS calculations to the existing
   `hmc_posterior_diagnostics` implementation. It provides the compact summary
   required by fixed-kernel verification and the full per-parameter report used
   by deterministic LGSSM recovery.
2. Added the convergence symbols to the lazy
   `bayesfilter.inference` export surface.
3. Fixed retained telemetry accounting to continue through the logical draws
   and count every status failure. The existing shared-invalidity and callback
   error stop behavior remains unchanged.
4. Allowed the retained-sample summary to handle an odd valid-draw count by
   dropping only the unmatched final draw for split-R-hat computation. The
   ordinary diagnostic API still requires an even draw count.

## Checks

Environment: `tfgpu`; focused checks deliberately used
`CUDA_VISIBLE_DEVICES=-1` with `TF_FORCE_GPU_ALLOW_GROWTH=true` because these
were CPU contract tests, not GPU evidence.

Command:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_fixed_transport_hmc_tuning.py \
  tests/test_hmc_fixed_size_chunk_runner.py \
  tests/test_hmc_operational_broad_grid.py \
  tests/test_hmc_posterior_diagnostics.py \
  tests/test_tensorflow_gpu_memory_policy.py \
  tests/test_ssl_lstm_neutra_complexity_hmc_tuning.py \
  tests/test_ssl_lstm_complexity_hmc_budget_rate.py
```

Result: `89 passed`.

Additional direct smoke: `rank_normalized_hmc_diagnostics` on a seeded
`[1000, 4, 3]` IID fixture returned finite diagnostics with
`max_rhat=1.0012264405351763`, `min_bulk_ess=3778.4543171160817`, and
`min_tail_ess=3138.0810575884266`.

A broader deterministic-LGSSM test invocation had one unrelated repository
fixture failure: `scripts.run_hmc_phase6_typed_identity_smoke` is absent while
`tests/test_deterministic_lgssm_hmc_tuning_driver.py` imports it. The remaining
35 tests in that invocation passed; this missing launcher was not introduced by
the remote tuning merge and was not changed here.

## Decision And Nonclaims

| Item | Status |
| --- | --- |
| Remote merge affects q=20 lane | Yes; shared HMC verification API changed |
| Merged source integration | Repaired; focused checks pass |
| Old q=20 tuning run | Source-stale; do not resume or promote |
| Fresh q=20 timing canary | Still required under repaired source |
| Fresh q=20 tuning | Not run in this audit |
| HMC convergence/posterior correctness | Not assessed |
| GPU performance or memory claim | Not assessed by CPU contract checks |

## Next Handoff

Refresh the q=20 current-source GPU/XLA timing canary with TensorFlow memory
growth verified before import. Then run the existing bounded preflight and
tuning plan under a fresh versioned output directory. Do not launch retained HMC
until the fresh source-bound tuning and fixed-kernel verification gates pass.
