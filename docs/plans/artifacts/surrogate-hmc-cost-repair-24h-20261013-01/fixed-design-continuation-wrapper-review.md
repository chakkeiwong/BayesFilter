# Corrected-source continuation wiring review

The corrected inverse-CDF ancestor selection invalidates the former tuning scope.
Both follow-up controllers now receive explicit corrected advance, execution-plan,
and paired-result paths. Defaults remain compatible with archived invocations;
the active commands below bind the corrected paths. Nine focused CPU-only tests
pass, including stale-terminal rejection, fresh-selection wiring, continued work
after a model failure, single-worker checks, and budget accounting.

The skeptical audit found no change to the scientific comparator or numerical
kernel in this repair. The cost baseline and candidate both consume the corrected
ancestor selector; the selected precision comparison uses the same finite program
in FP64. Neither comparison establishes accuracy relative to the model likelihood,
posterior convergence, or canonical Contract-E admission. These checks implement
the existing consumer-paired-contract.md and selected-consumer-precision-contract.md.

| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Run the existing follow-ups on corrected scopes | Fresh-path orchestration checks: 9 passed | Stale selection cannot substitute; overlapping GPU worker prevents launch | Actual selected consumer precision and cost remain to be measured | Serial paired cost then selected FP64 comparison | Numerical/posterior promotion |

Commands use the tftwogpu Python executable and escalated GPU access. Each controller
waits for its named predecessor to terminate and for the campaign ledger to be free.
All GPU worker time is charged to the same 86400-second extension, with a shared
120-attempt maximum. Per-worker caps are 2400 seconds for cost and 3600 for precision.
A failed model does not stop independent model checks. Existing artifacts are preserved.

```text
python scripts/run_surrogate_hmc_consumer_followups.py --campaign docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01 --output docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-consumer-paired-01 --advance-dir docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-scope-advance-01 --execution-plan docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-scope-execution.json --wait-seconds 64000 --per-model-seconds 2400
python scripts/run_surrogate_hmc_selected_precision.py execute --campaign docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01 --output docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-selected-precision-01 --advance-dir docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-scope-advance-01 --paired-dir docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-consumer-paired-01 --execution-plan docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-scope-execution.json --wait-seconds 64000 --per-model-seconds 3600
```
