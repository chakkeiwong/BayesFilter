# Prospective calibration of a batch-dependence guard

The research goal remains testing surrogate HMC, including an imperfect full
DCTR force, against the declared endpoint and independent posterior reference.
The current MCSE guard vetoed454/512 IID Gaussian controls. That is a diagnostic
calibration defect. This bounded preparation experiment tests a replacement;
it does not change closed LG1/C01 results, generate additional posterior looks,
or install a new default. No HMC consumer or target changes in this stage.

## Candidate, comparator and derivation

Comparator: the actual existing equivalence code, including maximum batch-means
MCSE over40/20/10 batches per chain, sensitivity<=2, absolute largest-batch
lag<=.3, modern parameter R-hat<=1.01 and bulk/tail ESS>=400, function
R-hat<=1.01, and simultaneous Student-t36 intervals across15functions and two
looks. Preserve every check except the lag screen in the candidate comparison.

Candidate: use40 disjoint batch means per chain (length25/50 at1000/2000draws)
and a randomization test of the maximum absolute centered lag statistic across
all15functions. Center each chain's batch means. For each function,
r=sum(c[a,b]c[a,b+1])/sum(c[a,b]^2). Under a uniform permutation of m batches,
E[r]=-1/m: sum of all ordered distinct products in each centered chain equals
-sum(c^2), and each adjacent ordered pair has probability1/(m*(m-1)); summing
m-1adjacencies gives-sum(c^2)/m. Thus use T=max_j abs(r_j+1/m).

Draw1023 uniform permutations of the40batch positions, shared across chains and
functions, preserving their joint dependence at each batch. Sort independent
FP64 random keys; fail closed on any tied keys, so accepted permutations are
uniform conditional on distinct keys. Compute
p=(1+number(T_permuted>=T_observed))/(1023+1). Include numerical ties using a
relative1e-12 tolerance, making rejection conservative. With exchangeable
batch vectors this finite Monte Carlo randomization p-value is super-uniform:
the observed ordering and its randomly permuted orderings have an exchangeable
rank, with ties counted against rejection. IID observations satisfy the
exchangeability assumption exactly; dependent chains need empirical checks.
Failure to reject does not establish independence or convergence.

Allocate .02 false-veto probability across the two planned looks, hence reject
at p<=.01/comparison_families at each look. This reserve is a diagnostic design
choice below the5% tolerable IID false-veto rate, not a fitted threshold.
Use comparison_families=1 here;1023 permutations also resolve the registry's
1,2,4family thresholds. Zero variance or nonfinite data are invalid. Other
diagnostics and the existing interval calculation remain unchanged.

## Evidence contract and defaults

Generate four chains of2000draws and evaluate nested1000/2000looks with the
actual15C01 mean, second-moment and marginal-CDF functions. Known stationary
reference: three independent standard Gaussian coordinates. Gaussian AR(1)
starts in stationarity, innovation scale sqrt(1-rho^2). Use fresh seed ranges,
fixed before running: date key20260921, ranges400000(IID),410000(rho=.5),
420000(rho=-.5),430000(rho=.9),440000(rho=.99),450000(chain offsets),
460000(linear drift). Permutations use separate date key20260922 and
range+2*replication+look_index. No diagnostic result selects a threshold.

IID:512replications. Each other regime:256replications, total2048. Chain-offset
control adds(-.6,-.2,.2,.6) to coordinate1 of independent draws; drift adds
the deterministic path-.75to+.75 to coordinate1. These intentionally violate
the stationary common-target assumption and should be vetoed. The three
stationary dependence levels span ordinary, borderline and very slow mixing;
negative dependence checks asymmetric repairs. This is Gaussian calibration,
not evidence of validity on arbitrary multimodal, heavy-tailed or SV targets.

Primary IID criterion: upper Wilson95% bound on any-look, full-diagnostic
false-veto rate<=.05. Coverage criterion: in each stationary regime, upper
Wilson95% bound on any admitted interval missing any true function<=.05;
also require conditional failure upper bound<=.05 among replications with at
least one admitted look. Fewer than64admitted replications make that regime
inconclusive, never a coverage pass. The Student-t intervals already account
for15functions and two looks. Wilson bounds are per predeclared regime;
no simultaneous across-regime confidence claim is made.

For rho=.99, offsets and drift, lower Wilson95% bound on full-diagnostic veto
at both looks must>=.95. If rho=.99 admits fewer than64replications, report
coverage as inconclusive there and require the stated veto criterion; this
regime is a rejection control, not an admitted-mixing scope. Nonstationary
controls have no stationary common-target coverage claim. Retain all raw
per-look decisions, p-values, lag estimates, R-hat, ESS and interval coverage.
Old-rule comparison and all mean/MCSE/timing differences are explanatory only.

Default/assumption audit:40batches is the smallest existing batch length and a
power hypothesis; long dependence may survive or be hidden by within-chain
centering. Modern R-hat/ESS and slow/offset/drift controls discriminate this.
Maximum over functions avoids selecting a favorable coordinate; shared
permutations preserve nonlinear-function correlations.1023 is a cost/precision
choice, with finite-randomization conservatism recorded. Student-t36 and the
Gaussian function family are inherited candidates, not universal uncertainty
guarantees. Test their admitted coverage directly. No heavy-tail or all-model
promotion is possible even if every screen passes.

Heuristic adversaries for this diagnostic are constructed explicitly: naive
IID standard error (ignores serial dependence), the existing batch-means rule
(current comparator), and exact finite-length AR(1) mean variance
[n+2sum((n-k)rho^k)]/(4*n^2) (oracle). Report MCSE/oracle ratios for means in
each stationary regime. Heuristic and oracle diagnostics explain failure;
coverage and veto calibration decide this stage, not minimum interval width.

## Execution and skeptical audit

CPU-only FP64 TensorFlow/TFP, GPU intentionally hidden, fixed signatures, XLA
for generation/batch/permutation kernels. The existing tied-rank diagnostic
graph retains its documented non-XLA exception. No NumPy or pfor. Independent
small exact enumeration using Python standard-library arithmetic checks the
permutation statistic and centering identity before the experiment.

Budget:600CPU-worker seconds and600wall seconds across initial execution plus
one localized infrastructure retry; focused tests<=60CPU seconds. The shared
repair allocation and whole campaign63GPU/15CPU-hour cap remain unchanged.
After8IID replications, forecast1.5times average steady per-replication cost
over the last4times remaining replications. If this exceeds remaining budget,
stop UNDER_BUDGETED without reducing replication counts or relabeling a pilot.
Use cumulative accounting, process watchdogs and a fresh versioned root for
each launch. Missing references, parity failure, nonfinite generated data or
resource exhaustion stop the diagnostic; a failed candidate prompts a recorded
repair. Undefined diagnostic statistics must already veto admission. Preserve
them as JSON null with their exact field names, never replace them by finite
values. An undefined statistic in an admitted replication stops the diagnostic.

Pre-mortem: a guard can accept IID yet miss biased dependent intervals; evaluate
both coverage and dependence protection. It can seem safe by rejecting all
chains; require admitted counts and IID false-veto control. It can pass only
after inspecting old failures; fresh simulations and fixed thresholds prevent
that selection. The candidate is not a mathematically proven general MCMC
uncertainty procedure. Skeptical audit passes for this diagnostic question,
with inherited assumptions and promotion limits explicit.

Run `scripts/calibrate_surrogate_hmc_mcse_randomization.py --output <fresh>`
with the tftwogpu interpreter, CUDA_VISIBLE_DEVICES=-1 and2/1TF threads.
Output root `docs/plans/artifacts/surrogate-hmc-all-models-20260920-mcse-randomization-01`.
Save source snapshots, exact command/environment/seeds, incremental accounting,
per-replication JSONL, report, result note and current checkpoint. All policy
changes remain prospective and explicit; this stage supplies evidence only.

Initial execution saved22IID replications before strict JSON serialization
rejected an undefined diagnostic value. This is a reporting failure; the
statistic was not admitted. The early remaining forecast was309.00wall/
347.96CPU seconds, within budget. First attempt charged13.903043CPU/8.978wall
seconds. Repair records undefined field names and serializes those values as
null while preserving their existing veto. No criterion, count or seed changes.
One fresh retry consumes the remainder of the same600/600 allowance.
