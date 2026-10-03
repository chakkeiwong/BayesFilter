# M21 confirmation inventory and pilot review

The twelve controller pilots completed in 47.6015605708817 CPU worker-seconds.
All 11 focused engineering checks pass. The iid pilots stopped at warmup 2000
and retained 1000; rho=0.8 stopped at retained 1000 and 1500. The stationary
rho=0.98 pilots delayed warmup to 4500 and 6500, then retained 8000 and 7500.
The shifted rho=0.98 pair produced one posterior pass and one warmup cap.
The dispersed rho=0.995 and common-shift rho=0.9999 pilots all reached warmup
caps. These are development observations, not rate estimates or a ranking.

The same six cases now receive **400 fresh independent replications each**,
plus independent fixed-count comparators. The frozen machine inventory is
`m21-r1/confirmation-inventory.json`; root seed 2026092212 is disjoint from
development. No pilot observation enters confirmation, and no replication
count changes after a coverage result is seen. The slowest pilot cost including
startup was about 5.848 seconds per replication. At 400 replications this
projects 2339.2 seconds per case; a 3600-second ceiling allows about 54% overhead.
Reserve 21600 CPU worker-seconds within M21's 64800-second ceiling. At most two
single-thread numerical workers run together.

Exact Gaussian fixed-count intervals about their analytically known transient
expectation are the harness check. For each of six cases, compute a
Clopper--Pearson interval at alpha=0.05/6 and check whether it includes 0.95.
A failure triggers investigation of arithmetic, streams or the declared law
before scientific interpretation; it is not an automatic proof of a bug.

For stationary cases, assess fixed and stopped lugsail intervals separately:
the lower pointwise 95% Clopper--Pearson bound for unconditional interval
delivery and coverage must reach 0.90 to pass this diagnostic adequacy screen.
The 0.90 floor is an explicit study choice allowing a five percentage-point
deficit from nominal 0.95, not a calibrated product default. Also report
conditional coverage, interval unavailability and posterior caps. For the iid
and rho=0.8 controls, the corresponding lower bound for completed posterior
checks must reach 0.90. Stress cases may fail delivery; their failure remains
part of the answer. No success criterion is attached to a hoped-for ranking.

Review: all intended controller branches were exercised by the pilot without
altering a threshold. The fixed comparator's exact reference accounts for its
remaining transient mean; a common-shift chain at 10000 is not assumed
stationary. The denominator and screens are fixed before confirmation. The
diagnostic is affordable at measured cost, with timeouts and all missing fits
visible. It answers controller and interval questions; the separate HMC control
inventory still must exercise actual preparation, tuning and candidate replay.

Command: `run_confirmation.py` in `m21-r1`, with CUDA hidden, growth enabled,
one TensorFlow intra/inter-op thread, using the tfgpu Python. It consumes the
frozen inventory and writes `confirmation-cpu-r1`, exact worker commands,
per-replication tensors and summaries. Budget ceilings include timed-out and
failed workers. Numerical-policy changes require separate fresh evidence.
