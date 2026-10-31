# C3 independent-reference and posterior continuation

29 October 2026. The public GPU qualification is complete. Do not rerun its
1,248 trials or discard either movement failure. This continuation answers the
next question: can an independently evaluated posterior reference support
assessment of every one of the seven retained candidates on the same
48-observation, nine-coordinate MacroFinance target?

## Research intent and evidence contract

### Execution refresh: independent proposal repair

C3a passed: 28 public-target points, maximum value discrepancy
2.8422e-14 and chart discrepancy 2.6646e-15. Central differences at
1e-5 and 5e-6 satisfy the declared combined absolute/relative score
tolerance. The focused reference suite has 28 passing tests. C3b attempt01
failed before integration because diagnostic helpers were absent from the
clean sampler checkout; attempt02 used a separately frozen helper package.
Its 59.73-second ladder finished but failed the unchanged .02-SD screen:
maximum empirical uncertainty was .118814 SD. Preserve both attempts.

The inherited P18 center is synthetic generating geometry, not a posterior
mode. Its first calibration ESS was only 13.77/32768; the second was
86.31/32768. EM consequently fitted highly concentrated weighted samples.
This is a reference-proposal failure, not a sampler or likelihood failure.
Repair within the existing CPU allocation, without borrowing HMC draws:

1. Independently optimize the covariance likelihood plus the exact standardized
   prior from the original center, zero, and five frequency-displaced starts.
   Save every optimizer return, gradient residual and objective. Use a batched
   central gradient (step1e-5), BFGS max300 iterations and tolerance1e-5.
   These are diagnostic convenience choices, checked by gradient residual and
   finite-difference sensitivity, not new runtime defaults.
2. At stationary results (maximum gradient <=1e-3), compare central Hessians
   at steps1e-3 and5e-4. Reject nonpositive or materially inconsistent local
   curvature; do not silently floor eigenvalues. Relative Frobenius discrepancy
   <=.01 is a provisional numerical check, validated against an exact Gaussian.
   Retain up to three distinct modes, using unit local-Mahalanobis distance
   only to remove duplicate proposal centers; mode-search completeness is not
   claimed. Build seven local Gaussian components with widths1,2,3 times
   checked local scales, plus the unchanged prior component. Width expansion
   is a proposal-coverage hypothesis; all weights use the full mixture density.
3. Fit two discarded 2^16-point calibration rounds. Preserve a twice-expanded
   local component after each fit to reduce tail collapse. Freeze the result
   before fresh eight-scramble validation at powers14,16,18. If that fails,
   permit powers20,22 with the SAME frozen proposal and new streams, only
   when observed full-horizon pricing fits the remaining allocation. The
   unchanged screen uses the latest two refinements and all31 quantities.
4. Seeds are inventoried without overlap: pricing410000001; local search
   deterministic; calibration420000000+10000*round+[0..7]; validation
   430000000+10000*power+[0..63]. Attempt02 pricing overlapped a calibration
   component seed; final validation did not overlap. Record that limitation,
   preserve the old result, and use fully disjoint streams for the repair.

Skeptical audit: the repair changes only the independent diagnostic proposal,
not target/data/prior, HMC tuning, posterior criteria or uncertainty tolerance.
The prior component cannot prove global coverage; multistart search and broader
components are additional diagnostics, not a proof. Optimizer success is not
posterior validity. Fresh final streams avoid using the failed validation
draws as training data. Nonfinite target evaluations, source drift, failed
curvature checks or budget exhaustion stop the affected attempt and preserve
evidence. The audit passes for this bounded repair and focused regression.

The primary engineering criterion is agreement of an independent covariance
Kalman likelihood with the frozen public factor-QR target, including its prior,
chart, stationary initial law, offsets and prediction-first timing. Posterior
reference integration then uses independent scrambled importance samples;
HMC draws never fit the proposal or estimate the reference. Matching one local
value, passing tuning, or synthetic truth recovery is insufficient.

The comparator is the existing frozen public target at commit
f9291ee4437aa8c00dabbe331bffc6871b574c5c, not a new target or regenerated data.
Source and data drift, nonfinite required evidence, wrong measure/timing or a
failed numerical equivalence check veto dependent execution. Poor importance
coverage or precision rejects the reference candidate and triggers the next
predeclared refinement; it does not invalidate the HMC candidates. Budget
exhaustion stops execution with preserved partial results.

Weight concentration, between-scramble dispersion and refinement differences
are reference diagnostics. They are not proof that unseen modes are absent or
that finite-sample confidence coverage is calibrated. No method ranking,
posterior promotion, new default or release follows from this phase. All
sampler membership, acceptance policy and tuning-R-hat semantics stay fixed.

## Execution phases and refresh points

1. **C3a, reference implementation and parity.** Implement the independent
   diagnostic covariance recursion in a testing/reference module. Test against
   a dense joint-Gaussian likelihood, scalar recurrence, row permutations,
   duplicated rows, invalid shapes/inputs and finite-difference scores. Then
   compare a frozen public CPU target at initial, displaced and prior-generated
   points; this is an explicitly CPU-only reference exception. Preserve exact
   source/input identities and maximum errors. Do not alter bound runtime files.
2. **C3b, bounded reference integration.** Reuse the existing eight-component
   diagnostic importance integrator in standardized prior coordinates. Keep
   one standard-normal prior component and fit the other seven with discarded
   independent calibration draws. Freeze the proposal, then evaluate independent
   scrambles at increasing powers. Persist every replication and resume only
   unchanged source/proposal/data. Price the first full-horizon batch before
   committing the full ladder. Record actual cost and refresh the next phase.
3. **C3c, posterior preparation and execution.** After reference adequacy and
   public sequential wiring pass, load all seven unchanged verified members and
   use the shared posterior controller, fresh warm-up/retained streams, durable
   chunks and original frozen kernels. First price one actual retained chunk
   for every member; then reserve an affordable inclusive tranche. Keep all
   failures. Do not change claim_eligible or posterior-convergence authority.
4. **C4/C5/C6.** Continue estimator/tail reliability separately from arithmetic,
   price independent-fit SIR against its separate funds, and reconcile examples,
   integration tests and the official book. Refresh the master from each phase's
   result before launching the next. Conditional research is not made a new
   release prerequisite by this continuation.

## Target, measure and estimands

### C3c executable posterior design and pricing audit

Proceed only after reference03 or a preserved successor meets the unchanged
reference screen. Load all seven members together through the frozen public
adapter and grouped loader; preserve their exact kernels, verified endpoints,
source hashes, GPU identity and experimental claim restrictions. Do not copy
the 1.63GB shared evidence bundle. Price1000 fresh transitions per chain for
each member using the public retained runner, in increasing (L,epsilon) order.
Price draws are discarded and cannot become posterior samples. Reserve at most
7200s inclusive for this price stage from the existing upper balance. This cap
is an operational hypothesis: prior full64-transition trial times extrapolate
to approximately3500s for the seven1000-step blocks, plus grouped validation
and compilation. Actual prices, not that extrapolation, determine the next
posterior reservation. Any unpriced member remains explicitly unpriced.

The prospective shared sequential configuration keeps the inherited1000-step
chunks, warmup minimum2000/latest1000 window/maximum10000 with R-hat<=1.05,
and retained minimum1000/maximum10000 with R-hat<=1.01. It assesses all31
declared quantities. Retained bulk/tail ESS floors400 and mean MCSE/SD<=.02
with lugsail(r=3,c=.5), sqrt(n) batches, minimum20 batches are inherited N5
operational hypotheses (see complete-fit-replication-01/block-04-nl-acc-randomized/
fit/posterior-01/policy.json), not established universal calibration. Zero
warmup ESS floors and one successful readiness check preserve the existing
shared policy. Reference agreement uses (absolute mean difference +3*MCSE
+reference envelope)/reference SD <=.1 for every quantity. These are
diagnostic pass criteria only; C4 still owns tail/MCSE coverage reliability.

Numerical nonfinite states, target failures, chain movement and energy
log-acceptance<-1000 retain the shared numerical vetoes. Candidate-local
numerical or convergence failure stops that member, preserves its evidence,
and does not remove its tuning membership or prevent assessing the others.
Source/data/reference mismatch invalidates the shared attempt. Posterior
R-hat is separate from reporting-only tuning R-hat. No ranking is authorized.

Price roots are (20261029,30001+index). Posterior warmup/retained roots are
(20261029,1000000+200000*index) and (20261029,1100000+200000*index). Check all
derived seeds against tuning, reload, pricing and other members before work.
Serialize durable chunks and their identities; restart may replay unchanged
completed chunks, never regenerate them under altered policy. Store only
compact tensor archives and summaries. A stable TensorFlow/XLA quantity
transform must agree with the independent reference at saved parity points.

GPU selection respects the bound GPU UUID for these exact exported artifacts,
with load<50% and at least4096MiB free (inherited operational headroom, actual
allocator peaks recorded). This is a reload-identity constraint, not an idle
GPU requirement. Reassess capacity every1800s; do not kill healthy work because
its own utilization is high. Check progress every60s. A cooperative budget
boundary and outer process cap preserve partial work.

Skeptical audit passes for quantity parity, public retained pricing and
preparation: no truth-as-posterior comparison, no reused final streams, no
silently selected representative, no presumed transfer of reference authority.
The pricing stage does not establish posterior validity. Freeze the posterior
reservation only after actual all-member costs are available.

Pre-posterior orchestration audit found one engineering gap: the first driver
serialized durable chunks but had no command to resume them. Before posterior
launch, delegate cohort execution to the existing tested
`scripts/resume_modern_hmc_frozen_full_model.py::run_cohort` helper, freezing it
beside the driver because it is absent from the clean numerical checkout.
Preserve all numerical source files and the already-running price worker.
Use fresh output directories; validate and copy only an unchanged committed
checkpoint prefix, carry terminal successes and failures without resampling,
and preserve original member indices/seeds. Regression must check interrupted
versus uninterrupted equality and corrupt-prefix rejection. This is a localized
harness repair under the existing preparation reserve, not a new campaign or
a change to posterior criteria.

To avoid another manual stop between pricing and posterior assessment, a
bounded continuation supervisor may wait for the active price01 terminal
settlement and then launch the frozen, regression-tested posterior driver.
It must require all seven actual prices, unchanged price/reference/source
identities, no numerical failure in pricing, and an affordable maximum-count
quote. Reserve ceil(quoted seconds)+180s, capped by the remaining campaign
balance after the 3600s preparation reserve and all actual reference/pricing
charges. The extra180s is explicit operational allowance for supervisor and
closeout, not additional scientific work or a promise about runtime. The
quote already includes a1.5 contention margin and120s serialization allowance.
If unaffordable, preserve prices and report the resource gap; never shrink
the cohort, loosen criteria or borrow SIR funds to force a launch.

The waiting observer's overlapping wall time is not added to the active
pricing charge. Record its own process CPU overhead within preparation. The
posterior supervisor's inclusive wall is charged once, without adding worker
time. The continuation updates a compact status/result note and the active
program block at launch and terminal settlement. A candidate failing
posterior checks is preserved and the next member proceeds; shared artifact
invalidity stops the attempt. No automatic retry changes numerical policy.

Audit: this transition implements the already-reviewed C3c experiment under
existing authorization and budget. It has one numerical source, one frozen
reference, exact cohort membership, explicit worst-case count pricing,
separate warmup/retained streams and fresh directories. The cost margin is
a planning hypothesis; outer caps and checkpoints handle an underestimate.
Continuation is allowed after recovery regressions and exact-source smoke
checks pass, with no new approval-token or reviewer chain.

Execution update23:15 Asia/Shanghai: the expanded suite passed66 tests; seven
additional checks against the frozen numerical source passed, including
interrupted/uninterrupted native equality. The continuation driver/helper are
frozen under C3 posterior source-02. Observer990814 is live and waiting for
price01 to settle; price01 has completed L=3/.4 and L=5/.4 and is running
L=9/.4. Exact progress and subsequent phase refresh are written to
`N5/c3-posterior-20261029-01/continuation-01/status.json`.

The runtime chart coordinates are theta in R^9 with unnormalized prior
exp(-sum((theta/0.35)^2)/2), without a chart Jacobian. Integrate in z=theta/0.35
under the normalized standard Gaussian; the constant change of variables
cancels in posterior expectations. The existing generic importance code
requires precisely this standardized prior coordinate, so passing theta
directly would compute the wrong prior. A dedicated equivalence test is required.

Reproduce the chart from frozen Phase9 equations: K=S S' plus skew omega,
F=exp(-K/12), Q from its Cholesky chart, diagonal R from the bounded log-contrast,
P solving P=F P F'+Q, mean (0.35,-0.22), observation offset (0.12,-0.07),
identity observation matrix and zero jitter. Use a direct four-dimensional
Lyapunov linear solve and covariance updates as an independent check of the
factor-QR/doubling runtime. A dense observation-covariance Gaussian test checks
the covariance recursion's timing and normalization independently.

Report the nine chart coordinates, their standardized squares, nine smooth
upper-tail quantities sigmoid(2*(z-1)), and the model's four declared targets:
state standard deviations and lag-one autocorrelations of G and Pi. All 31
quantities remain in summaries, including failed ones. The smooth-tail function
is inherited diagnostic scope, not a literal tail-probability estimate.

## Defaults, numeric provenance and affordability

| Choice | Provenance / status | Failure mode and early check |
|---|---|---|
| Prior scale .35, chart constants and 48 observations | Exact frozen target | Wrong measure or timing; dense/public parity |
| P18 center/covariance as initial proposal | Existing warm-start hypothesis, not posterior reference | Missing modes or poor weights; retained prior component and independent refinements |
| Eight equal mixture components, two discarded fits of 2^15 points, 12 EM updates, .1 covariance shrinkage | Existing independent-reference algorithm, unpromoted for this target | Bad local fit or weight collapse; calibration ESS/concentration and failures preserved |
| Eight independent scrambles at powers 12,14,16 | Initial feasibility ladder, two lower rungs followed by the inherited 2^16 scale; not a universal error guarantee | Too few replications or unresolved tails; between-scramble error and successive-rung changes |
| New seeds derived from 20261029, with separate calibration and final ranges | Reproducibility choice | Reuse/collision; inventory disjoint seeds before running |
| Value/score parity tolerance 1e-7 absolute and 1e-8 relative | Numerical hypothesis, looser than observed public engineering roundoff to allow independent covariance arithmetic | Masked discrepancy; record raw errors, test dense oracle and step-size sensitivity |
| Reference envelope <= .02 posterior SD for every quantity | Provisional engineering screen allocating 20% of an inherited .1-SD comparison tolerance | Underestimated integration bias; at least two refinements and separate proposal diagnostics; no coverage claim |
| CPU reference cap 14400s; preparation/test reserve 3600s | Bounded next tranche from existing 186372.1157282945s upper balance | Overrun; explicit deadline, actual measured costs, no double charge |

The reference envelope is the maximum of recent successive-order mean changes
plus three between-scramble standard errors. It is an operational diagnostic,
not a rigorous integration bound or simultaneous confidence interval. A failed
screen requires honest reference-limited status, not a widened tolerance.
Posterior tolerances remain provisional until a reference is available and the
full posterior design is priced; this plan does not silently authorize weaker
posterior checks. Separate SIR funds remain 7363.064379416s and untouched.

C3b outputs are under a new N5/c3-reference-20261029-01 directory. The exact
command/environment, Git plus source hashes, seeds, elapsed time and source
input are saved before scientific execution. CPU-only reference processes set
CUDA_VISIBLE_DEVICES=-1 before imports. NumPy/SciPy are confined to independent
reference calculations and are never imported by tuning or sampling. GPU
posterior work remains TensorFlow/TFP/XLA with verified incremental allocation.

## Skeptical review before execution

The plan rejects the tempting but wrong baselines of synthetic truth as a
posterior mean, covariance-QR agreement as convergence, and old private-wrapper
qualification as current public qualification. It checks actual full-horizon
inputs, standardized-prior measure and prediction-first timing before spending
on integration. No frozen source or existing member is relabeled. A shared
posterior controller can assess an experimental retained member without
claim-eligible promotion; existing restrictions stay intact. Sequential stopping
and nonlinear-tail MCSE limitations remain explicit rather than being waived.

Pre-mortem: local importance fitting may miss another mode despite tiny
between-scramble error. The prior component, calibration diagnostics, repeated
scrambles and refinement sensitivity can expose some failures but do not prove
global coverage. A material mismatch in numerical parity stops reference use.
The audit passes for implementation, focused tests, pricing and the bounded
reference ladder. Posterior launch requires its subsequent recorded design and
affordability check, which are executable next steps rather than new approval
ceremony. No external message or public/default change is authorized here.
