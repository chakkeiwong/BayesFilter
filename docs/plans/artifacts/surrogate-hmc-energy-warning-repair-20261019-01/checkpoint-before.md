# Active surrogate-HMC checkpoint

Updated 2026-10-18 15:51 UTC. Continue the five-model surrogate-HMC campaign
under the owner's additional 48 hours. The goal is incomplete. Predator-prey is
running; do not duplicate or interrupt its worker. Range bearing stopped at
1,500 warmup transitions per chain with an energy-error guard failure and zero
retained draws; other final-chunk health checks passed. Its failure is unresolved.
LGSSM/KSC previously passed the recorded sampling rule; posterior agreement
remains not checked.

Checkout: /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917
Branch: ledh-graceful-failure-execution-20260917; HEAD10884528ca8b1a7463895256d30b9655d69c0955.
Preserve unrelated dirty work. Python: /home/chakwong/anaconda3/envs/tftwogpu/bin/python.
Escalate GPU launches; verified memory growth is mandatory.

Active plans: surrogate-hmc-48h-extension-2026-10-18.md and
surrogate-hmc-48h-queue-repair-2026-10-18.md.
Output E: artifacts/surrogate-hmc-48h-extension-20261018-01/.
E/run/budget.json is the single ledger: 172800 worker seconds, maximum six
attempts, deadline 2026-10-20 08:00:41 UTC (Hong Kong16:00). The repaired
supervisor PID3835358 took over at 10:32:20 UTC. It has advanced automatically
to predator-prey master PID3843697; both processes were confirmed alive using
escalated ps at this status check. Original deadline and attempt budget remain
unchanged. Recheck PIDs before acting. Handoff receipt and supervisor log:
E/run/queue-repair-01/{handoff.json,supervisor.log}.

Predator-prey's first attempt failed after5.99 seconds: saved replay events
used the older healthy force-check schema. This is a harness failure, not a
new numerical failure. The narrow schema/source compatibility repair passes
46 CPU-only tests in E/replay-repair-tests.log; rationale and source archives
are in E/resume-repair-note.md and E/source-before-replay-repair/.
Modified modules: bayesfilter/inference/hmc_tensorflow_{verified,resume}.py;
focused tests: tests/test_hmc_tensorflow_{verified,resume}.py.

Queue repair executed after skeptical audit: 20 focused queue/original-extension
tests passed in 3.19 seconds (E/queue-repair-tests.log). The repaired queue is
active range bearing (existing eight-hour allowance), predator-prey retry (up
to eight hours), then SIR (remaining budget). Predator-prey's retry output is
E/run/predator_prey-retry-01; prior evidence is preserved. No pinned original
supervisor source was changed. Queue script:
scripts/run_surrogate_hmc_48h_queue_repair.py.

Latest retained predator-prey progress: 5000 draws per chain after
3,000 warmup; maximum cumulative modern R-hat 1.011589 (target <=1.01).
All latest-chunk health checks passed; sampling continues under the same frozen
kernel. SIR follows automatically. Range-bearing is not queued for an additional
retry by this supervisor; review its terminal cause before any further repair.
Current status evidence: docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01/status-check-20261018-155156.json.

Exact next action: monitor E/run/state.json, next-phase-plan.json, budget.json,
the active model's public-tuning progress, and the repaired supervisor log.
Allow the supervisor to advance automatically; do not duplicate the active
worker. Record terminal diagnostics when each model finishes. The first
predator-prey failure was infrastructure only; its retry remains planned work.

Frozen settings: N4032/K2016, FP32/TF32off, GPU/XLA, epsilon1/five OT iterations,
literature model priors, SIR FP64 recurrence/FP32 pairs. T50 except range T20
with16 flow steps. Four chains: warmup min2000, latest1000 modern R-hat<=1.05,
max10000; retained cumulative modern R-hat<=1.01, max20000 or budget. No extra
validation chain. Range retry uses predeclared acceptance.95 and step.005;
energy-error guard1000 remains unchanged.

Prior terminal results and full evidence paths:
artifacts/surrogate-hmc-60h-continuation-20261016-01/status-check-20261018-052637.json.
LGSSM2000/1000 warmup/retained,R-hat1.00708; KSC2000/1000,1.00600;
PP3000/2750,1.02970(time); SIR2000/0(time,warmup not ready);
range2000/500,1.02058(energy1509>1000). These do not establish five-model
completion, posterior correctness, or statistically supported method ranking.
