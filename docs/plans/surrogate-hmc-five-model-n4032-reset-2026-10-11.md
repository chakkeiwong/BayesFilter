# Active surrogate-HMC reset memo

Updated 2026-10-21 for campaign closeout and repository integration.
Checkout: /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917.
Branch: ledh-graceful-failure-execution-20260917.
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
