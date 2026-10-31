# Active surrogate-HMC reset memo

Updated 2026-10-23 for the resumed SIR NeuTra training/HMC campaign.
Checkout: /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917.
Branch: ledh-graceful-failure-execution-20260917.
Optimization baseline: d189c429542eb3a682677ce7fc463c3b796b69ea.

## Current checkpoint

The full five-model sampling objective is complete: LGSSM, KSC SV,
predator–prey, range bearing and SIR all passed cumulative modern R-hat <=1.01.
See the [current overall summary](surrogate-hmc-master-final-summary-2026-10-23.md)
for the checked per-model results. No computation phase remains. All 17 truths
are inside empirical 95% posterior intervals and all physical-parameter R-hat
values are below 1.01. Four models meet 65–75% chain acceptance; SIR is above
that band at 75.9–79.3%. The owner requested Git integration, tracked in the
[current closeout plan](surrogate-hmc-posterior-closeout-2026-10-23.md); the
historical Git closeout below predates this work.

The SIR NeuTra surrogate-HMC run finished normally at 03:57:44 Hong Kong on
2026-10-23. The owner requested one working method and removed the dense-affine
comparison. Tuning, training, freeze, adaptive warm-up and retained sampling
have completed. No worker remains active.

Scope: SIR T50, N4032, FP32 filter, TF32 off, GPU/XLA; learned map and HMC
coordinates FP64. Four chains each have 2,000 warm-up and 2,000 retained draws.
Warm-up is excluded; all 8,000 retained draws are inference draws. The frozen
canonical IAF is width 6, initial LR .01, scale .35, seed 201. Three training
replicas completed 512 accepted updates each; this count alone does not prove
training convergence.

The cumulative R-hat stopping rule passed: max 1.009824 in latent z and
1.009603 in the SIR parameter chart. Bulk ESS is [412.28, 595.13, 475.48],
tail ESS [199.29, 1129.09, 769.04], and mean MCSE/SD
[0.05317, 0.04134, 0.04599]. The owner clarified on 2026-10-23 that ESS is
diagnostic, not a convergence criterion. The corrected master reports
`NEUTRA_POSTERIOR_CRITERIA_MET`, matching sampler `VERIFIED`. ESS and MCSE
remain reported without a completion veto. No further sampling is required.

Retained acceptance is 75.90–79.30% (pooled 77.70%). There are zero recorded
divergences and force fallbacks; all numerical, endpoint and movement checks
pass. Maximum absolute energy error is 6.44571 and remains explanatory only.
These diagnostics describe the fixed bootstrap 4x4032 approximate posterior,
not established agreement with the exact SIR posterior. Default readiness
remains a separate unevaluated claim.

Master: `scripts/run_sir_neutra_training.py`.
Output: `artifacts/sir-neutra-training-20261022-neutra-only01/`.
Plan: [execution plan](sir-neutra-training-execution-2026-10-21.md).
Result: [terminal interpretation](artifacts/sir-neutra-training-20261022-neutra-only01/final-result-20261023.md).
All prior attempts remain charged, including the stopped dense-affine attempt.
Remaining allowances: 8.4580 of 24 HMC hours and 8.5678 of 16 training hours.
The corrected master passes 35 focused CPU tests in 15.05 seconds. Only
post-run assessment/reporting changed; the original draws, runtime provenance
and budget charges are preserved in the
[reassessment record](artifacts/sir-neutra-training-20261022-neutra-only01/reassessments/001.json).

Recovery action: report successful completion under the declared R-hat rule,
with ESS and MCSE as precision diagnostics. There is no pending automatic
master phase or required precision extension. Do not restart training or add a
comparator or validation phase. The earlier
[whitening observation](artifacts/sir-neutra-training-20261022-neutra-only01/observations-20261023-01/result.md)
remains provisional and is superseded by the terminal diagnostics for sampling
status. Detailed historical work follows below.

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
