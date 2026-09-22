# A09 execution record

Budget closed at 10:05:56 UTC: 21758 seconds charged, including 120 seconds
reserved for the final response. Administrative closeout exceeds the A09
six-hour ceiling by 158 seconds including that allowance; numerical work
ended before the ceiling. This overrun is explicitly charged to H11, not
hidden as GPU work. Numerical attempt wall times total 4248.447 seconds;
all failed attempts are included. H11 conservative remaining balance is
41920.892609 seconds (11.64 hours), retaining the prior A08 reservation.

2026-09-16 closeout: confirmation-02 completed at 09:53:13 UTC in
1287.928334872995 seconds. Its 24 sequences yield 12 matched scalar and
11 matched four-dimensional cases; d4-s10 failed guide/reference checks.
d4-s04 passed reference precision but the accepted guide's covariance factor
collapsed to 2--4e-16 at t8, producing severe evidence bias for every
guide-dependent proposal. Both scalar degree-4 nominees passed the exploratory
paired MSE comparison. No d4 ranking or default promotion is supported.

Final artifact audit PASS: 14880 particle-time records, 6992 TT CDF checks,
2160 reference records, 4080 distinct scalar seed slots, 19 source snapshots,
1311 frozen fit/seed checks and 54 same-target regression audits. All 54 fits
reduce H2 relative to their own initializer. The audit checked 1077122 finite
numeric values. All 27 distinct focused tests passed. Terminal review is a
documented executor self-review; no independent review is claimed. The master,
active checkpoint and sole H11 budget ledger now close A09. No jobs remain.

2026-09-16 09:33 UTC: confirmation-02 launched with the same frozen
calibration-02 controls, seed partition 2, repetition stride 1000, and an
1800-second attempt cap. Its manifest records the exact command; full output
is in command.log. All ten source snapshots match their start hashes.
The seed-schedule and reference-endpoint tests pass; the final focused file
has five passing tests (27 distinct relevant tests across this amendment).
At 09:45 UTC, 18/24 fresh sequences were complete, with zero recorded guide,
reference or consumer failures. assemble.py rejects an invalidated attempt;
audit.py checks saved seeds and actual particle/CDF records, not just labels.

2026-09-16 09:32 UTC: final call-chain review found time-shift seed reuse
between confirmation-01 sequences and repetitions, including particle
references. Its intervals cannot support the planned independent-sequence
claim. Preserve the completed result (1399.306 s), mark it descriptive only,
and rerun fresh confirmation partition 2 with disjoint ten-million seed blocks
and repetition stride 1000. Calibration-02 remains nomination only; do not
retune after the inspected confirmation. Revision 2 records the review and
repair before replacement execution. No scientific setting/criterion changed.

Plan reviewed before implementation; see plan-review.md. Source changes are
optional initial-centered L1, a diagnostic standalone driver, and one explicit
consumer dispatch alias. Existing fitter defaults remain zero-centered.

2026-09-16 05:44 UTC: 23 focused CPU-reference tests passed in 10.24 s
(`focused-tests.log`), including one-core closed form, shifted KKT, lambda-zero
multicore parity, actual standalone-to-pair-sampler call chain and missing-arm
selection exclusion. GPUs were explicitly hidden for these tests.

Trusted nvidia-smi and TensorFlow probes confirm CUDA_VISIBLE_DEVICES=1 maps
to RTX 5080, with memory growth true (`gpu-probe.log`). The tool reported
5102.164 s wall time for nvidia-smi, including its permission/tool wait; this
is not experiment runtime. Preserve elapsed accounting conservatively at
closeout rather than count it as a 5102-second GPU benchmark.

The bounded smoke additionally checks GPU/XLA versus CPU graph core fits at
degree 4 for both penalties. Serious calibration starts only after smoke
artifacts pass numerical validity, consumer and reference checks.

Smoke attempt-smoke-01 passed in 61.122 s: both d=1 and d=4 references pass,
eight methods per dimension, zero invalid consumer steps or CDF brackets.
GPU/CPU core differences <=2.23e-16. GPU allocator peak 145979136 bytes.
Source hashes unchanged during execution. After the smoke, the driver added
the same device/thread environment defaults internally to avoid a shell-env
approval wrapper; this changes launch setup only, not numerical controls.
At 07:48:38 UTC the elapsed interval since recorded start is 223.33 minutes;
tool approval waits occupy most of it. The manifest preserves actual run time.

Calibration-01 INVALIDATED: after launch, provenance checking found that root
926000 duplicated A07's calibration observations. Stopped via SIGINT, exit
130, measured wall 548.541 s. The manifest's stale RUNNING status is explained
in its sibling invalidation.json; preserve the original manifest and source/
plan snapshots. No results from this attempt select or confirm controls.

Revision-1 skeptical review: seed root 18626000 (confirmation 18636000) has
no matches in checked observation-TT plans/drivers, avoids A06/A07/A08 roots,
and retains disjoint calibration/confirmation and particle/reference offsets.
Method, criteria and budgets unchanged. Replacement calibration may execute.
The driver now also finalizes its manifest after KeyboardInterrupt.

Calibration-02 launched directly with the environment's approved Python
command, using the revised seeds and the same 5400-second cap. Initial
TensorFlow startup stderr was returned by the execution tool; subsequent
captured output is in calibration-02-captured.log. The structured per-step
artifacts and run manifest are the numerical evidence.

Additional independent checks passed: 9 tests in 5.70 s, including the
degree-four coefficient recurrence against physical Gauss-Hermite integration
and paired-inference handling of missing/duplicate candidates. These extend
the earlier 23-test set by two cases (25 unique cases checked in total).
Consumer inspection confirms invalid bracket/nonfinite/CDF residual >1e-8
raises before a particle update; the A09 report reads the actual
cdf_bracket_valid flag and also stores maximum residual and invalid-step counts.

Calibration-02 COMPLETE: 951.549 s, all six references pass, fourteen arms per
sequence (twelve degree/row/L1 arms plus two anchored arms), no failures.
All 54 same-target audits are present. Controls hash matches completed result;
source unchanged during execution. GPU allocator peak 189834752 bytes.

Frozen choices: baseline p3/n1024/L1=.001 in both dimensions; capacity
d1=p4/n4096/L1=.001, d4=p4/n1024/L1=.001; preservation shares those degree/row
choices with initial-centered L1 .001 and .00001 respectively.

Before confirmation, localized observability/provenance improvements redirect
stdout/stderr to attempt-local command.log and require COMPLETE calibration
plus its existing controls hash. Numerical routines/choices are unchanged.
Explicit exception: guide/row/feature/target host setup reuses the bounded
diagnostic orchestration; XLA refers to numerical fitting/sampling/projection
kernels, not the entire end-to-end driver. No default-runtime claim is made.
