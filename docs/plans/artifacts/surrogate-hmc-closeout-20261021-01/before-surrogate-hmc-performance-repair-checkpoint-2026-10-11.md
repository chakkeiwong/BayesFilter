# Active surrogate-HMC checkpoint

Updated 20 October 2026 after SIR stopped at 07:37:54 UTC (15:37:54 Hong Kong).
The goal remains testing surrogate HMC on five models. Execution has finished:
four models passed their sampling rule; SIR ended BUDGET_EXHAUSTED without
finishing warm-up. No campaign processes remain active. This is an incomplete
five-model outcome, not an all-model success.

Checkout: /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917.
Branch: ledh-graceful-failure-execution-20260917.
HEAD: 10884528ca8b1a7463895256d30b9655d69c0955. Preserve unrelated dirty changes.
Python: /home/chakwong/anaconda3/envs/tftwogpu/bin/python.
GPU commands require escalation and verified memory growth.

## Current requested rerun

Plan: docs/plans/surrogate-hmc-range-bearing-accept70-rerun-2026-10-19.md.
A: docs/plans/artifacts/surrogate-hmc-range-bearing-accept70-20261019-01.
Owner authorization: "redo the run again", after specifying the 70% target.
This separate requested rerun has an eight-GPU-worker-hour cap and one planned
attempt; it does not enlarge SIR's budget or alter the old campaign ledger.
A/budget.json charges this run. The existing master enforces its worker timeout.

Finished 2026-10-19 19:49:33 UTC (20 October 03:49 Hong Kong). Recorded worker
time 25,883.80 seconds; remaining allocation 2,916.20 seconds. Status VERIFIED /
SELECTED_MODELS_HMC_SAMPLING_COMPLETE. Four chains completed 2,000 warm-up and
1,000 retained draws each. Acceptance 70.2%, 68.0%, 68.4%, 68.7%; maximum modern
R-hat 1.003620; bulk ESS 1,111-2,221; tail ESS 1,688-1,910. Zero arithmetic
divergences; one retained force fallback and large finite energy warnings were
preserved without stopping the run. The actual config requested 0.70 and no
resume manifest. GPU/XLA, FP32, TF32 off and verified memory growth are recorded.
Evidence: A/run/range_bearing/inference/hmc.json and A/result.md. No extra
sampling or validation is needed for this completed requested rerun.

Exact next action: report the terminal five-model outcome from E/run/result.md
and the fresh range-bearing result above. No run is pending or automatically
continuing. Any SIR repair or additional compute needs a new bounded allocation;
do not silently extend this stopped run. Independent posterior agreement is
still not checked. Timing differences from the old RTX 5080 range-bearing
baseline cannot be attributed solely to tuning.

## Terminal SIR and previous results

E: docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01.
SIR ended BUDGET_EXHAUSTED with 5,500 warm-up transitions per chain, zero
retained draws, and warmup_ready=false. Its configured acceptance target was
already 70%. The last readiness check, at 5,000, had maximum modern R-hat
3.417349361; the 5,500 chunk passed numerical health checks, with readiness not
due. Finite energy warnings did not stop it. Trusted process inspection
confirmed former master PID 78222 and worker PID 78224 have exited.
Progress: E/run/sir_d18-energy-retry-01/sir_d18/inference/public-tuning/fresh-progress.json.
Terminal evidence: E/run/sir_d18-energy-retry-01/sir_d18/inference/hmc.json.
Supervisor status: INCOMPLETE_SEE_MODEL_RESULTS; pending_models is empty.
All six attempts are terminal. The ledger charged 161,236.65 worker seconds
(44.79 hours), including 118,731.48 seconds for the last SIR attempt. It still
records 11,563.35 aggregate seconds unspent; do not describe the aggregate
ledger as zero. The worker's BUDGET_EXHAUSTED status is its stopping decision.
The old campaign cap was 172,800 worker seconds and its deadline is
20 October 08:00:41 UTC (16:00:41 Hong Kong). Its last SIR attempt is bounded
by 125,497.27 seconds from its start on 18 October 21:08:04 UTC.

The old range-bearing run completed VERIFIED at target 95%, with four chains,
2,000 warm-up and 2,000 retained draws per chain, maximum modern R-hat 1.00745857,
acceptance 93.8-94.9%, bulk ESS 672-1,523 and tail ESS 1,310-2,504.
Its result remains E/run/range_bearing-energy-resume-01/range_bearing/inference/hmc.json.
It is preserved as the comparison; it is not relabelled as a 70%-target run.
LGSSM, KSC and predator-prey previously met their sampling rules. Posterior
agreement remains not checked; do not claim full scientific verification.

## Frozen scientific settings and implemented repairs

N=4032, K=2016, FP32/TF32 off, GPU/XLA, OT epsilon 1 with five iterations,
literature model priors. T=50 except range bearing T=20 with 16 flow steps.
Four chains; warm-up at least 2,000, latest 1,000 modern R-hat <=1.05, maximum
10,000; retained cumulative modern R-hat <=1.01 or maximum 20,000 per chain.
All retained samples serve inference; there is no extra validation phase.
Finite energy errors are warnings, never stopping criteria. MH is unchanged.

Energy-only resume and empty-phase artifact repairs passed 84 focused tests
and six-chunk GPU replay. Evidence:
docs/plans/artifacts/surrogate-hmc-energy-warning-repair-20261019-01.
The fresh range command's 95% override was corrected to 70%; 24 focused tests
passed. Evidence: docs/plans/artifacts/surrogate-hmc-acceptance-target-repair-20261019-01.
Old queue source pins remain historical; the live supervisor already loaded
its source. The new launch records current source/selection hashes and exact argv.
Earlier checkpoint is preserved in A/checkpoint-before.md.
