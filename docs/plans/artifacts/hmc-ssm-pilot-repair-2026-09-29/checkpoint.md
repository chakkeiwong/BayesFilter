# State-space campaign terminal checkpoint

Checked September 30, 2026. The user asked whether execution is done.

The main queue ended normally at **23:23:42 Shanghai on September 29**.
Both `bayesfilter-hmc-ssm-priced-main-20260929-r1.service` and
`bayesfilter-hmc-ssm-main-sequence-20260929-r1.service` are inactive. A trusted
host process check found no campaign Python worker. No work is queued.

- Original main denominator: 32. Seven fits started; three K4 fits completed,
  four K0 fits timed out during tuning, one funded K4 fit missed the original
  latest-start cutoff, and 24 slots were unfunded within the wall allocation.
- K4 data-A seeds 0/1 and data-B seed 0 passed the declared posterior checks,
  precision screens and descriptive reference tolerances. Each assessed one
  predeclared member using 2,000 warmup and 1,000 retained transitions per chain.
  All six/eight/four verified candidates remain retained. Coverage records are
  unavailable for the four declared mean/quantile quantities. No calibration
  or ranking is established, and no nonlinear main fit completed.
- K0 candidate counts are 84/70/98/98, verified counts 2/5/4/4, and pending work
  counts 19/17/18/28. All four exhausted 3,352-second worker allocations including
  560.333-second contention extensions, before posterior sampling. Checkpoints
  remain available. Pilot pricing times 1.5 did not cover these main fits;
  foreign-process detection does not establish causal slowdown.
- Main cost: 18,893.276967168087 enclosing GPU seconds, charged exactly once.
  The reservation is cleared. The settled additive grant balance is
  151,610.48782264302 GPU seconds. Nested fits must never be charged again.
- Original Shanghai deadlines remain: latest start September 29 23:03:52;
  compute September 30 03:03:52; report September 30 05:03:52. An unused grant
  balance does not reset the original campaign clock.
- Main output: `docs/plans/artifacts/hmc-ssm-main-2026-09-29/r1/`. The
  `terminal-audit-2026-09-30.json` verifies every executed receipt, completed
  assessment, worker source identity and GPU/XLA/memory-growth provenance.
  `inventory-live.json` now contains all 32 terminal dispositions.
- Frozen worker source: `docs/plans/artifacts/hmc-shared-gpu-recovery-2026-09-29/prepared-r5/source`,
  identity `a8c6c6107c2da78d25a144be28a3503431352681465e338e945fca899f0a0e43`.
  Source, data, criteria and seeds were unchanged in main execution.
- Completed cumulative pilot costs: K0 2,241.074108 seconds, K2 4,841.005600,
  K4 2,533.405727, K7 7,157.783499. K7 retained all 21 verified candidates and
  both assessed members passed declared checks. Its independent reference is
  for the declared sigma-point approximation. It is development evidence only.
- Remaining pilot gaps: K1/K3 searches incomplete; K3 data-B reference workload
  differs from its pilot; K5 preparation repaired but no full repaired pilot;
  K6 preparation and its independent reference remain unresolved. C1 is complete
  and must not be rerun as part of this closeout.
- Existing verification: 56 focused main/allocation tests in
  `main-allocation-r5.xml`; earlier checkpoint/preparation checks in result.md.
  This terminal audit added no numerical code and launched no new experiment.
- Next justified work: diagnose K0 search expansion, compilation and per-fit
  funding from saved records before scheduling another campaign. Preserve all
  failures and missing slots. Do not describe queue completion as full validation.

[Detailed results and decisions](result.md). The master program and its progress
JSON have been refreshed. No subagents or commit/push in this status check;
unrelated worktree changes are preserved.
