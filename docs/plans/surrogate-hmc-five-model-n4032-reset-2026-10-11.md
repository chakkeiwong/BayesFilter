# Active surrogate-HMC reset memo

Updated 2026-10-21 for optimization implementation closeout.
Checkout: /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917.
Branch: ledh-graceful-failure-execution-20260917.
Optimization baseline: d189c429542eb3a682677ce7fc463c3b796b69ea.

## Current checkpoint

The requested four-optimization implementation, review and equivalence plan is
complete. Final GPU/XLA N4032 comparisons retain ordinary FP32 with TF32 off:
all saved tensors for LGSSM, KSC SV, predator-prey and SIR are bitwise identical
to baseline. Range bearing has small recorded FP32 differences, with observed
runtime about 28% lower and allocator peak about 32% lower on the paired fixture.
The triangular OT cache was rejected for slowdown and the original dense cache
restored; the invariant feature hoist remains. The 130 distinct CPU tests all
pass in their latest execution, including the repaired independent polar
finite-difference check.

The SIR matched target/trainer adapter, complete-update sharding and frozen-map
HMC connection are implemented. A real N4032/T50 two-update integration pilot
passes gradient/Adam equivalence and bitwise checkpoint continuation; its HMC
mechanics rows are finite. This is an integration-only map, not a
coverage-trained posterior map. Parent Phase 1's adapter/update gap is closed;
target-specific training/search and subsequent adaptive SIR sampling remain.
The five-model posterior objective is still incomplete.

Plan: [optimization implementation](surrogate-hmc-optimization-implementation-2026-10-21.md).
Result: [optimization results](surrogate-hmc-optimization-results-2026-10-21.md).
Evidence: `artifacts/surrogate-hmc-optimization-20261021-01/`, including
`comparison.json`, `saved-tensor-equivalence.json`, `test-summary.json` and
`training-sir_d18-candidate-03/`. Master: `scripts/run_surrogate_hmc_optimization.py`.
All final numerical source hashes match the executed manifest. Accounted worker
time is 2,758.012 seconds of six hours; 18,841.988 seconds remain unused and no
jobs are pending. Code and these records are included in the owner-requested
optimization commit on this branch; this task does not repeat the earlier
merge/push integration.

Next research action: add the parent SIR training/search and adaptive-HMC phases
to the five-model master, then execute their measured, bounded protocol. The
numerical training machinery is ready, but the five-model master still has no
NeuTra phase. The refreshed parent plan records that distinction and the pilot
update costs. This is beyond the completed implementation/equivalence plan,
not an unfinished job in the optimization master. No further GPU job was
launched at closeout.

## Earlier campaign and Git closeout (historical status)

The following records the state before the optimization plan above. Its pending
adapter/update statements are superseded by the current checkpoint.
Reviewed batching commit: 65a379b2c; closeout commits follow it.

Goal: test surrogate HMC on LGSSM, KSC SV, predator–prey and SIR at T50,
and the saved range-bearing fixture at T20. Requested N4032, ordinary FP32,
TF32 off, GPU/XLA and literature priors. Historical per-run precision
exceptions remain documented in their manifests.

Execution is finished: four models passed the sampling rule; SIR ended
BUDGET_EXHAUSTED with 5,500 warm-up transitions per chain and no retained draws.
The fresh range-bearing 70%-target rerun passed with 1,000 retained draws per
chain and maximum modern R-hat 1.003620. No new sampling or training run is
launched by this closeout. The five-model sampling objective remains incomplete.

Native batching and bounded capacity testing are complete: 119 isolated tests
passed; logical B128/native S4 completed within 0.761 GiB. No SIR NeuTra map has
been trained. Parent Phase 1 target-to-trainer adapter and full-update checks
remain pending. The fixed 1% scientific blocker is retired; compare against the
Zhao–Cui adapted reference with estimation uncertainty. The nominal old/new
batching change is accepted as immaterial for continuation; broader equivalence
and independent posterior accuracy remain unestablished.

Final master summary: surrogate-hmc-master-final-summary-2026-10-21.md.
Materiality policy: sir-batching-materiality-policy-2026-10-21.md.
Parent integration plan: surrogate-hmc-sir-neutra-training-audit-plan-2026-10-20.md.
Evidence: artifacts/sir-native-batching-20261021-01/summary-materiality.json,
artifacts/surrogate-hmc-48h-extension-20261018-01/run/result.md, and
artifacts/surrogate-hmc-range-bearing-accept70-20261019-01/result.md.

Campaign documentation, code, and evidence are committed in 091ee89bc;
main incorporated the branch in a32f00ccb. Remote main at 3eb9462d3 has been
integrated and its focused checks passed. The integration record is
surrogate-hmc-git-integration-2026-10-21.md, with preserved check logs under
artifacts/surrogate-hmc-closeout-20261021-01/integration/.

Repository integration is complete. Remote merge e0ded6482 passed all three
oracle hook tests (138.16 seconds), was pushed normally to origin/main, and was
fast-forwarded into this branch. Direct remote lookup and local checks confirmed
main, origin/main and this branch identical, with both worktrees clean. The
following documentation and evidence commit records that observed outcome and
explicitly includes scientific records otherwise omitted by JSON/log ignore
rules. Its exact file inventory is preserved in the integration artifact.

No compute task is active or launched by this closeout. The next research work
is the existing SIR NeuTra Phase 1 adapter/update plan under an explicit bounded
follow-on execution plan; the five-model objective remains incomplete. Do not
repeat the finished Git integration or silently start another GPU allocation.
