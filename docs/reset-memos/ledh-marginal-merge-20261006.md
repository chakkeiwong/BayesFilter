# Active LEDH/Zhao–Cui checkpoint

Question: marginal versus ancestor LEDH likelihoods and scores for predator–prey
and SIR d18, T=10/20/40/50, with original author TT and bootstrap references.
Branch sqmc-development. Commits: b3f4ca646 shared correction; 8f0ebe955 campaign;
509fe726c nonlinear writer repair; 2b909c343 completed rank20 comparison.
Plan: docs/plans/ledh-zhao-horizon-comparison-20261006.md.
Root: docs/plans/artifacts/ledh-zhao-horizons-20261006-01/.
Result: docs/benchmarks/ledh-zhao-horizon-results-20261006.md.
User authorized commits, scoped permissions, comparisons and replication repairs.
No main merge/push or new agents this stage.

Complete: all 64 main GPU FP64/XLA LEDH evaluations, 32 covariance-only checks,
bootstrap N=1008/32768/131072 with four seeds, three PP rank20 fits and all
PP scores, one SIR rank20 fit and all T10/20/40/50 scores. T40/T50 missing-radius
repairs succeeded, preserving their original paths, design and radius.
Report009 has 309 cells and zero invalid LEDH evaluations. Committed-evidence-03
preserves it; completed-comparison.md puts the explicitly labeled rank20
reference beside both LEDH arms. Ranks and independent fits remain separate.

Still running: primary session15473, rank40 SIR fit2 in attempt036, capped at
115200sec; step12 of50 finished at this checkpoint. It schedules four quadratic
jobs on success. Auxiliary session4604 is waiting for later quadratic timeouts;
it retries missing predeclared radii only. Do not relaunch either supervisor or
write their attempts.json / score-repair-attempts.json ledgers. If the rank40
fit times out, recover valid completed prefixes and assess missing scores within
the remaining budget; the current primary does not score a failed full fit.

Budget172800 aggregate job-seconds, including failures and derivative checks.
09:38UTC: completed27958sec, used53809 including active elapsed, reserved143158
including full active caps, remaining29642. Auxiliary watcher also protects
9600sec of future primary score slots. Current primary retains its launch-time
code; new launches use shared accounting. No expanded campaign is authorized.

47 focused tests and three commit-contract checks pass. Nonlinear checkpoint
writer now records unavailable legacy evidence as NaN plus a flag; both return
formats execute through the actual Octave writer. Shared singleton and M13
parity passed earlier. GPU derivative diagnostic002 agrees locally for SIR:
maximum normalized errors7.44e-6/6.04e-5; exact trace value/coordinate0 parity.
Moment-safety branches can change under perturbation; global smoothness is open.
Monograph compiled614pages and new pages402–404 were visually inspected.

Finding: the marginal change does not repair this untuned SIR configuration.
T50 log likelihoods: ancestor-1924.200, mixture-1902.554, rank20TT-1669.094,
bootstrap131072-1669.111. SIR scores remain unstable across four LEDH designs.
PP value differences are small; some reference scores disagree. No supported
ranking, oracle certification, scope-specific admission, default or HMC claim.
Original paper numbers remain incompletely replicated; current-target callbacks
are explicit extensions, not relabeled paper experiments.

Next: finish rank40/prefix score checks within budget, refresh the report and
commit the final evidence. Preserve large local MAT/source trees and all prior
attempts. Exact approved wrapper: Python -B docs/benchmarks/run_ledh_zhao_horizons.py
with absolute env/script paths, actions status/report/repair_scores as applicable.
