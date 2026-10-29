# Active surrogate-HMC checkpoint — 2026-09-30 09:21 UTC

Question: testing surrogate HMC on frozen M13/data1704/T20.
LG1 accepted; M13 and all-model campaign NOT complete.
Master48h-03 STOPPED2026-09-30T06:37:46Z, M13_PUBLIC_TUNING_BUDGET_EXHAUSTED.
Supervisor2224426 and worker2236101 exited; no active sampling.
This is a budget-limited inconclusive result, not rejection of the method.
Two full scientific attempts used; no third attempt under the current plan.

Warmup passed at9000 per chain. Latest1000 modernRhat maxima1.029793/1.036218,
both<=1.05. Post-warmup verification reached2250 per chain.
Last scheduled check at2000: modernRhat1.057747/1.025917, above1.01.
Those2250 are actual frozen-kernel MCMC draws, excluded from posterior estimates
by the current tuning protocol. All45 archived chunks pass every health check.
Terminal audit verified450 checksums, chain boundaries and unique chunk seeds.
Guard receipt COMPLETE, exit0; master source_unchanged=true.
No harness corruption, numerical veto or endpoint/force failure was found.

Budget: additional48h deadline2026-09-30T16:27:39Z (Oct1 00:27:39HKT).
CPU/GPU caps172800seconds each. Total closed charges112433.525712/
109994.262141 seconds; unused60366.474288/62805.737859, also deadline-limited.
Stopped on tuning phase time limit, which protects ten-hour retained reserve;
total compute allocation was NOT exhausted. The unqualified retained phase
cannot run. Do not report the master as running or campaign as complete.

Evidence root docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-03.
Terminal review: terminal-review-20260930-01.json.
Progress: m13-public-tuning-02/tuning/fresh-progress.json.
Run/receipt/accounting and all archives preserved.
Result: docs/plans/surrogate-hmc-m13-health-veto-repair-2026-09-29.md.
Plan: docs/plans/surrogate-hmc-m13-48h-continuation-2026-09-29.md.

NEXT: terminal review is complete; current master has no automatic next run.
If execution is continued, first revise the bounded phase plan and establish
resume feasibility for the SAME frozen kernel and saved verification history.
Do not reset warmup, source/target identity, unique seeds, cumulative charges,
attempt count or deadline. Do not promote verification draws to posterior.
Preserve Rhat/ESS/reference criteria. This resume is not an implemented or
authorized third fresh attempt. No time extension has been granted.

Frozen route: force FP32/TF32 canonical analytical DCTR; endpoint FP64 frozen
full-support4096x8 trajectory-importance target plus prior. Approximate
likelihood, not exact-likelihood posterior. Data3701 untouched.
Freeze reference/assessment and calibrated six-look sources.
Claude/subagents deferred. GPU commands escalated; CPU diagnostics hide CUDA.
Branch ledh-graceful-failure-execution-20260917; HEAD8a4cffa2fd085f43ab5710c24d6e7b48770aea0d.
Preserve unrelated dirty work.

Guide clarification: committed ch21b_hmc_tuning_interfaces.tex:93 requires
fresh verification on draws not used for kernel selection; :129 specifies
ordinary tuning Rhat<=1.01 and separate posterior ESS assessment. This predates
the campaign. ch26b_neutra_transport_hmc.tex:334 instead describes sequential
warmup followed by cumulative retained sampling. The campaign TF wrapper
(surrogate-hmc-tf-tuner-repair-2026-09-19.md:16) implements the ordinary tuner
requirement with separately archived warmup and discarded verification, followed
by separately counted retained sampling. Verification continues the SAME warmup
chains with the SAME frozen kernel; it is not an independent replication.
This clarification changes neither the frozen protocol nor the budget.

Stopping-rule clarification: saved tuning/manifest.json specifies verification
minimum1000, maximum10000 per chain, checks every1000 and health chunks250.
The failed check at2000 did NOT stop sampling: a further250 were archived.
The active worker had wall allowance85589.26395s and finished at84785.202148s;
the between-chunk resource guard stopped before its deadline-derived cap,
which preserved the ten-hour retained reserve. It was not a fixed2000-draw
failure, a10000-draw maximum, or exhaustion of the total48h compute allowance.
Any continuation repair belongs to the existing master and must preserve the
same kernel, cumulative samples and budget history; no new run was launched.
