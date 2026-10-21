# M13 single initialization repair — 2026-09-27

The research question remains testing canonical analytical DCTR-force surrogate
HMC with the separately frozen full-support MH endpoint. This is the existing
master's initialization repair, on exploratory data1704/T20. Data3701 is reserved.

Localization01 found all original starting forces finite and reported valid.
At (-log(2),-log(2)), force=(276.0,426.5), and the nominated zero-momentum
L=1 proposal moved to(-1.621,-2.020), where the filter becomes invalid. The frozen
endpoint potential increased from -17.06 to583.97. All other chains had no
fallback. This identifies a poor force/trajectory at that start, not a broken
endpoint. It does not establish force accuracy there or a global cure.

Use four dispersed starts equal to the discarded RWM metric-warmup mean plus
the Cholesky factor of C=.9 empirical covariance+.1 prior covariance times
(-1,-1),(-1,1),(1,-1),(1,1). Only the last256 warmup transitions enter the
1024-point cloud. No retained samples inform initialization. This gives starts
(-.2210,-.3858),(-.2210,.1458),(.4496,-.4729),(.4496,.0588). Localization01 found
all four forces and deterministic trial proposals valid, with lower endpoint
potential. This is a nomination diagnostic; it does not establish convergence.

Repeat public tune_hmc_kernel through the same repository-issued position-only
binding. Keep all nomination controls,500 discarded fresh warmup,1000 fresh
verification draws, energy bound, endpoint, force controls, TF32, and XLA
unchanged. Use fresh verification seed(20260927,1102); nomination seed remains
(20260926,1101). The public tuner retunes the metric/epsilon/L on its original
training stream; it does not reuse or relabel a failed verified artifact.

Primary criterion for this stage: public artifact authority and every fresh
verification check passes, including R-hat<=1.01 and no force fallback. A
failure rejects this repair candidate and preserves it. Exact endpoint or
provenance invalidity is a continuation veto. Posterior assessment belongs to
the already planned retained stage and must exclude all warmup/verification.
Acceptance, force magnitude and potential changes explain the failure; they
are not promotion evidence. No superiority or all-model conclusion is possible.

The same-endpoint RWM comparator is healthy and practically agrees with the
reference at2000/chain; formal simultaneous equivalence remains inconclusive.
It is sufficient for its comparator role; the surrogate's own equivalence,
R-hat and ESS criteria remain mandatory. No change to reference margins.

Budget allocation revision within the authorized144000CPU/126000GPU total:
public tuning60000CPU/50000GPU; repair22000CPU/22000GPU; baseline6000/6000 and
retained56000/48000 unchanged. This transfers12000CPU/12000GPU from the unused
public allocation, with no increase to total authorized compute. Recorded
repairs before focused tests: materiality1386.822697CPU/1266.589424GPU;
resume02--06 2097.327239CPU; localization113.185375CPU/52.856586GPU;
resume regression01 19.108475CPU. Allow one new public repair attempt at most
18000CPU/18000GPU and up to300CPU focused integration checks. Its launcher
sums existing repair receipts and fails before exceeding22000 on either axis.
Record every failed infrastructure/test attempt. A second tuning failure or cap
has no automatic retry. The retained supervisor can continue without another
user turn if this repair qualifies.

Skeptical/default audit: prior-scale initialization was an untested hypothesis
and caused the observed trap. Covariance-scale initialization is justified by
discarded warmup, not by choosing favorable retained samples. Shrinkage is the
same .9/.1 positive-definite protection used and checked in the comparator.
Keep the old failure as evidence of initialization sensitivity. Fresh starts
could still miss modes or visit the bad-force region, so all-chain movement,
finite values, no fallback, energy, R-hat and independent retained agreement
remain required. No clipping, changed force, target restriction, or default
algorithm substitution is introduced.

Artifacts:
`docs/plans/artifacts/surrogate-hmc-m13-initialization-repair-20260927-01`.
Run:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_m13_initialization_repair.py --output docs/plans/artifacts/surrogate-hmc-m13-initialization-repair-20260927-01
```
The already reviewed retained supervisor waits on this root and the healthy
RWM resume06 report, then executes its original bounded retained stage.

Prelaunch review completed: 70 focused tests passed in initialization-tests-03
(38.13 seconds wall,58.461570 CPU seconds). Initialization-tests-01 preserved
a stale master reservation failure; -02 passed the affected checks. The final
batch covers interrupted/uninterrupted RWM parity, warmup-only initialization,
actual consumer/public-tuner/native-kernel wiring, budget receipt discovery,
legacy-floor accounting, watchdogs, and retained handoff. The launcher now
passes its explicit five-hour watchdog ceiling. New M13 receipts are additive
to the legacy campaign floor and counted once.

Global prelaunch charges (all discovered receipts): {"cpu_worker_seconds": 77250.058277292, "gpu_seconds": 59597.89602986984}.
The repair reserves18000CPU/18000GPU and the retained followup reserves
56000CPU/48000GPU; both fit the unchanged198000CPU/226800GPU ceiling.
