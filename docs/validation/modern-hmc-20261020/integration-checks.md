# Integration checks, 20 October 2026

The scoped controller, pricing, report, recovery and settlement regressions
passed: **185 tests**, with two TensorFlow Probability dependency deprecation
warnings. This is engineering verification, not a new sampler experiment.

The isolated checkout initially lacked the ignored compiled custom operations.
That run produced 170 passes and 15 setup errors. Pointing the checkout at the
existing operations directory resolved all 15 errors without a package change:

```sh
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  BAYESFILTER_OP_LIB_DIR=/home/ubuntu/python/BayesFilter/bayesfilter/ops \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_modern_hmc_frozen_cohort.py \
  tests/test_modern_hmc_price_recovery.py \
  tests/test_modern_hmc_price_report.py \
  tests/test_modern_hmc_frozen_settlement.py \
  tests/test_modern_hmc_comparison_prices.py \
  tests/test_modern_hmc_comparison_scope.py \
  tests/test_modern_hmc_price_monitor.py
```

The successful run took 5.42 seconds. GPU devices were intentionally hidden.
Logs: `/tmp/bayesfilter-sir-integration-tests-20261020.log` and
`/tmp/bayesfilter-sir-integration-tests-20261020-r2.log`.

## Remote integration

Merged remote main `023247142` into campaign commit `0318dc6f2`. Conflicts
retained the modern transition profiles, v8 acceptance rule, duration-preserving
repairs, supervised timing and progress reporting, and NUTS exclusion. Remote
chain-execution exports and its independent numerical/manuscript changes remain.
The bibliography preserves remote corrections and all modern-HMC citation keys.

Post-merge checks expanded the seven files above with acceptance protocol,
statistics and independent trials; duration-preserving repair; modern contracts;
route and documentation contracts; and timeout-policy tests. All **356 passed**
in 80.64 seconds. The 2,292 warnings are dependency deprecations. GPU devices
were intentionally hidden, using the same environment and native-op directory.
The log is `/tmp/bayesfilter-merged-regressions-20261020.log`.

The partial clone retrieved missing historical objects during the merge and
reported stale reflog/automatic-GC warnings. No history was reset, no raw data
was pruned by this task, and no force push is used. The shared dirty checkout
is preserved; integration occurs in its isolated worktree.
