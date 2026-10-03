# M21 public HMC controls

The AR(1) study isolates stopping and MCSE. These controls separately exercise
ordinary preparation, the broad candidate search, fresh pair verification,
retained replay and actual posterior checks on merged source. All verified
members remain in the candidate inventory; only the first candidate ID at L=3
is selected before posterior draws. If that group has no verified member,
record a missing posterior, without selecting a favorable sibling.

Four one-fit CPU-reference pilots use exact synthetic laws: unit Gaussian,
rotated Gaussian (condition 100, angle 0.6), beta-binomial (prior Beta(2,3),
5 successes in 12 trials), and LGSSM location (prior SD 2, observation SD 0.5,
state variance 1, persistence 0.6, observations [1,-0.3,0.4,0.5,1.2,-0.2]).
The fixed data are convenience fixtures, not empirical data or SBC experiments.
The Gaussian and LGSSM truth follows their existing checked analytic references;
the beta posterior is exactly Beta(7,10). Reference truth is assessor-only.

Use M20's standard ordinary preparation, finite-window metric policy, startup
20, probe16, recovery3 and bound expansion1. These optional settings remain
inherited hypotheses. The primary L grid is (3,5,9,13,18,25), evidence128,
independent verification and the unchanged typed acceptance policy. The
controller schedule matches M21. Retained mean/median MCSE tolerances are 0.05
for Gaussian/rotated, 0.005 for beta-binomial and 0.025 for LGSSM. The rotated
target deliberately repeats the costly M20 accuracy request. For the others,
the values are explicit diagnostic precision choices, approximately a few
percent of marginal SD, with cost checked in this pilot; they are not general
consumer defaults. All outcomes, including caps, remain in the evidence.

Each fixed comparator uses the same selected member, independent streams and
predeclared 2000 discarded plus 10000 retained transitions per chain. A favorable
stopped result cannot shorten or suppress the comparator. Observe the native
inventory oracle, excluded warmup, quantity-level MCSE, exact-reference interval
coverage and time. Public engineering correctness requires intact identity,
all surviving siblings, fresh streams and correct exclusion. Posterior success
requires all configured checks; a tuning pass alone is insufficient. One fit
cannot estimate coverage or rank models or settings.

Reserve at most 600 CPU seconds per pilot, 2400 total within M21. Use immutable
`m21-r1/source-r1`, tfgpu Python, hidden GPUs, one intra/inter-op thread and at
most two workers. The pilots are queued after the controller confirmation so
the two-worker limit is shared. Freeze replication counts for any subsequent
whole-fit confirmation from measured cost and the same binomial precision
calculation before examining confirmation outcomes. Root seed 2026092214 is
development-only. Preserve failures and fix infrastructure before retrying;
do not relax the model, policy or precision because a pilot fails.

Skeptical review: AR(1) evidence is not evidence for HMC tuning. These complete
ordinary fits address that gap, with exact references and raw observations.
Fixed data make the conditional question precise but do not test calibration
over datasets. The short pilot is only a mechanics, failure and affordability
check; phase completion still requires its planned replication audit. No new
default or broad robustness claim follows from a successful control.
