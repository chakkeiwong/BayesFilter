# Frozen first-study reconstruction: 1000 replicas at five dimensions

This phase belongs to the owner's 24-hour campaign. The question is whether
the two already frozen independent R QR reconstructions reproduce useful
likelihood accuracy across the published first linear-Gaussian study design.
We will characterize both even if the preceding64-pair phase rejects one.
The choice to retain both is made before the new study data are generated;
neither will be selected by its agreement with a paper table.

## Sources and precise comparison

The locally preserved GJL arXiv v2 source is
`docs/plans/artifacts/iapf-r-source-reconciliation-20260920-01/sources/arxiv-extracted/iapf_arxiv.tex`.
Lines742--822 specify the first study and tables; lines653--704 specify
diagonal Equation15 fitting, a positive floor, k=5, tau=.5 and kappa=.5;
lines475--550 specify backward fitting and the controller. Those method,
theory and evaluation sections have been inspected alongside the earlier
`source-and-math-audit.md` under the replication-gap audit. No verified original
author code or original datasets are available; the audited public third-party
R code is an independent comparator, not an author oracle.

Use m=0, initial covariance I, process and observation covariance I, observation
matrix I, transition A[i,j]=.42^(abs(i-j)+1), T=100. For each d=5,10,20,40,80,
generate one new fixed dataset with seed89800000+d. Run replica IDs3001--4000
at every dimension,1000 each. The per-method seed remains
53000000+100000*d+10*replica+index, with index1 for both QR controllers,
2 for BPF,3 for fully adapted APF,4 for SIS. No previous campaign used these
data seeds. Common seeds aid pairing but do not promise identical random draws
after controller trajectories diverge.

Both QR arms use N0=1000, diagonal log-quadratic regression, positive floor
power8, delayed doubling, k=5, tau=.5 and kappa=.5. Their stopping windows are
six and five respectively. The latter is an explicit extension. Both evaluate
the learned guide in a fresh final APF, preserving conditional unbiasedness.
QR minimizes a different objective from Equation15; matching study dimensions
and particle counts does not close that mathematical gap. Retain caps of20
learning iterations and16000 particles, with failures reported, not censored.

Baselines are BPF10000 and fully adapted APF5000, as in the first paper study,
plus SIS10000 as a constructed no-resampling adversary. Exact Kalman likelihood
and all100 prefixes are the authority. Record whole learning-plus-final cost.
Comparisons use fixed particle counts, not equal computing time.

## Evidence and interpretation

Report each dimension and method separately. The practical primary screen is
a pointwise95% mean likelihood-ratio interval wholly inside[.8,1.2], upper SD
interval <= twice the paper SD, and mean final N <=1.5 times paper mean N.

| d | Paper SD | Paper mean final N | Paper mean resampling count |
|---:|---:|---:|---:|
|5|.09|1000|6.93|
|10|.14|1000|15.11|
|20|.19|1000|27.61|
|40|.23|1033|42.41|
|80|.35|1142|71.88|

The separate literal-pattern screen requires the SD interval within[.5,2]
times the paper SD and mean resampling within[.5,2]times the paper count.
These broad descriptive tolerances are inherited from the existing comparison;
they are not a statistical equivalence test against unavailable author samples.
Report exact model/data/source agreement separately from these numerical screens.

Use4000 paired replicate bootstrap draws, fixed seed9212026, for mean, SD,
terminal squared-error differences, and cost/particle differences. Evaluate
prefix squared relative errors conditionally on ordinary and large Kalman
innovations, using the fixed chi-square90% cutoff. Compare each QR arm with
all three heuristic methods. Any observed conditional mean loss to a heuristic
is a conservative promotion veto; infer population differences only from their
predeclared paired pointwise intervals. Intervals are conditional on one dataset
per dimension, not simultaneous or population-over-datasets inference. A failure
in any dimension blocks a cross-dimension reference-quality claim; no later
dimension may be used to hide it in an average.

Source, data/oracle identity, exact seed/replica coverage, finite log values,
fit convergence, guide tail checks,100prefixes per completed method and saved
fit histories are validity checks. Underflow of an exponentiated finite log ratio
is recorded explicitly. A baseline with underflowed ratios has unresolved rare
event behavior, not zero theoretical variance. Numerical errors are retained
as failed replicas and veto the affected cell; no favorable-result reruns.

Candidate rejection does not stop independent dimensions or characterization.
Corrupt common code/data/oracle, missing mandatory evidence that cannot be
repaired, the elapsed deadline or aggregate budget are continuation vetoes.
No result proves original-author identity, Equation15 correctness, general iAPF
failure, production-default readiness, multivariate R/TF parity, or LEDH/KDM/HMC
correctness. No alpha sweep, stochastic-volatility study or MCMC study is added.

## Resources, recovery and review

Use the existing CPU-only R4.1.2 environment, two single-thread workers and the
same172800-worker-second /24-hour campaign ceilings. The full study is expected
to require roughly130000--150000worker seconds; this is an estimate, not a new
allocation. No independent old budget is transferred. Each replica is a
resumable unit with a300-second cap; localized timeout repairs use identical
seeds and only missing pairs, charged to the same budget. Interleave dimensions
by replica number so a hard deadline does not omit all high-dimensional data.
Complete the fixed1000count before interpreting a cell as a full study result;
partial counts remain explicitly incomplete. Preserve all prior attempts.

Run the campaign phase driver with `phase03.json`. An ordinary local supervisor
waits for phase2 completion, validates its evidence, runs this phase, produces
its diagnostic report and terminal decision, and updates the master checkpoint.
Failed candidates continue; common validity failures pause dependent execution.
The supervisor periodically emits heartbeat/progress, so user interaction is
not required for phase transitions. The output root is the existing new24-hour
campaign root; every phase has immutable sources and every attempt a fresh path.
Expected disk use is below10GiB against more than570GiB available at preparation.

Default/assumption audit: the paper supplies model, horizon, counts and primary
table values. QR, floor8, delayed doubling and window5 are frozen reconstruction
hypotheses with checked code, not author settings. The same hypotheses are now
tested independently by dimension; cross-model transfer is not claimed. The
1000count comes from the published study and the user's selected scope, not
observed precision. Two-worker timing is descriptive because contention is
uncontrolled. The fixed tail check is a finite-guide validity diagnostic,
not a proof that all positive-floor mixtures have infinite or finite variance.

Pre-mortem: even1000replicas may miss rare huge weights, an attractive SD may
reflect underflow, an exact mean may conceal conditional heuristic losses, and
different author data may explain table differences. Preserve log ratios, means
and uncertainty, conditional errors, and source limitations to expose these risks.
The earlier source, exact-guide and fresh-final-filter tests provide the cheap
mechanics checks before this expensive rung; the new phase1 checked resumable
coverage and reporter bootstrap parity. No algorithm changed in this campaign.

Skeptical audit: PASS. The new datasets and methods are frozen before use,
the exact comparator is available, the source discrepancy is explicit,
both baseline strength and learning cost are accounted for, dimensional and
conditional results cannot be pooled away, and resource stops are distinct
from scientific rejection. This is a replication of the published design using
two labeled reconstructions, not a claim of literal author-code reproduction.
