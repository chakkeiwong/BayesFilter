# q20 NeuTra parameter audit: rationale, tests and failure interpretation

Date: 2026-09-15. Status: `DOCUMENTED_HYPOTHESES_CALIBRATION_NOT_EXECUTED`.

We can give every proposed choice an explicit purpose and a way to challenge it.
We cannot derive every exact value from mathematics. In particular, the proposed
network widths, learning rates, training counts, clipping threshold, plateau
tolerance and replica travel counts are not established q20 defaults. They are
testable hypotheses. The tables below distinguish those hypotheses from
identities, measured observations, resource limits and scientific requirements.

This is the mathematical companion to the [numerical parameter ledger](bayesfilter-ssl-lstm-q20-production-parameter-ledger-2026-09-15.md)
and [production design](bayesfilter-ssl-lstm-q20-production-pipeline-design-2026-09-15.md).
It covers all **152 parameter-table entries**, plus the reference, uncertainty
and cost choices introduced in the ledger's prose. A row with several controls
keeps their joint purpose visible and distinguishes their individual effects.
The source ledger remains the numerical inventory; this file records why a
choice might work, how to find out, and what a failure would tell us.
The owner's subsequent instruction makes these hypotheses the first quantities
to investigate when a run has problems. Use the
[failure investigation order](#first-checks-when-a-run-has-problems) below to
select the relevant checks; their current status remains uncalibrated.

No target evaluation, training, HMC, GPU probe or campaign was launched for this
audit. Source inspection and standard-library arithmetic are the only new
evidence. An audit check described below is **pending unless explicitly stated
otherwise**. Documenting a test is not passing it.

## Scope, evidence and the meaning of a sensible choice

The question is whether an end-to-end procedure can estimate the specified
four-parameter q20 UKF posterior to declared accuracy, and whether trained
NeuTra transports help at matched total cost. Comparators are physical-coordinate
identity HMC, properly tuned classical HMC, fully trained single-map NeuTra, and
the declared enhanced ensemble. The posterior is an approximate-likelihood,
synthetic-data target. Neither successful sampling nor a useful transport would
validate its economic interpretation or the UKF approximation to the original
latent-model posterior.

The desk audit looked for weak baselines, proxy promotion, stale source,
unpriced counts, hidden defaults, missing stop conditions, invalid derivatives,
unfair random-stream reuse and checks that cannot answer their stated question.
It found unresolved scientific calibration and integration work, not an
executable production configuration. The 31 previously inventoried source
files were unchanged when rechecked for this document. Shared HMC descriptions
refer to the inspected main-checkout interface, not automatically to the older
copy in this isolated checkout. Export of the q20 trained map to that interface
and the actual claim-bearing consumer call chain remain **not checked**.

Each table gives provenance/status, a mathematical or practical reason, an
observable check and a failure response. The following distinctions apply:

- **Derived:** an identity or conditional calculation is checked here; its
  assumptions still need to hold in the actual run.
- **Fixed:** existing target definition or owner requirement. Reproducibility
  or authorization is established; scientific adequacy is not established.
- **Inherited/proposed hypothesis:** an exact coefficient has no demonstrated
  target-specific justification. Its source is known and its test is specified.
- **Limited measurement:** an identified earlier experiment supports only the
  recorded scope. It is not new q20 production qualification.
- **Decision requirement:** an accuracy, cost or reporting choice expresses
  what would be useful. Mathematics determines its consequences, not its value
  to the application.

There is no universal numerical tolerance that closes every row. For quantities
with physical units, record those units and scale the error before comparing
thresholds. If a test needs an application accuracy margin, use the declared
quantity-specific margins in section P, or say that calibration is unresolved;
do not invent a second undocumented threshold.

### Diagnostic roles and stopping

| Role | What it does here | Consequence of failure |
| --- | --- | --- |
| Promotion criterion | Downstream numerical validity, posterior/reference agreement and requested precision; supported cost comparison for an efficiency claim | Candidate is unqualified; further repair may remain justified |
| Promotion veto | Invalid target/score/map evidence, missed required posterior checks, inadequate reference or scope mismatch | Cannot promote affected result |
| Continuation veto | Corrupt evidence, unresolved implementation/target invalidity, unavailable required reference at final assessment, exhausted declared budget | Pause affected phase for repair or report incomplete; do not turn its outputs into inference |
| Repair trigger | Ongoing learning, poor coverage, unsuitable metric, travel bottleneck, affordability mismatch | Run only the discriminating, funded development repair; preserve failed confirmation |
| Explanatory diagnostic | Acceptance, loss, curvature, clipping, occupancy, timing, round trips | Explains or nominates; by itself neither proves convergence nor rejects the research direction |

For a scientifically valid but unsuccessful candidate, ask whether the proposed
repair addresses its observed failure. If it does, candidate rejection is not a
reason to abandon the method. Numerical corruption, on the other hand, must be
repaired before interpreting sampler efficiency.

## First checks when a run has problems

Owner instruction, recorded 2026-09-15: when problems occur, examine and test
the uncalibrated quantities documented here first. This is the active
troubleshooting order for the proposed q20 pipeline. An inherited value, an
earlier nomination or a configuration called `serious` does not exempt a choice
from investigation. These instructions select tests; they do not establish a
failure's cause or mark any pending calibration as completed.

Start with the failed run's existing evidence. Verify the actual target,
checkpoint, optimizer state, source and execution path, then check numerical
validity before interpreting loss, geometry or sampling. After those checks,
follow the symptom-specific row below. Numerical health is the first priority
when it fails; for healthy runs with poor whitening, training adequacy and
posterior coverage take priority over extending expensive HMC diagnostics.
An already resolved check can be reused in its exact scope. Do not rerun every
parameter test for every failure or start an undirected grid search.

### Observed problem and first discriminating test

The row identifiers in parentheses refer to the parameter entries later in
this file. Exact values below identify current hypotheses, not new defaults.
Each check must use the row's existing error/uncertainty requirements. If its
required evidence or threshold is unresolved, record that limit instead of
manufacturing a pass/fail cutoff.

| Problem | Quantities or assumptions to examine first | First useful check or controlled contrast | Interpretation and next action |
| --- | --- | --- | --- |
| D01 — Results change after setup/resume, or serious run behaves like a canary | Actual positive-beta update count; loaded weights; Adam slots/iteration; beta/RNG state; export and public consumer binding (A01–A03, A15, A19–A20, N18, H02, H10) | Trace the failing consumer to its actual saved map and target; compare stored tensors, objective coefficients and optimizer continuation. Use a small replay/resume check before more optimization. | Missing or reconstructed training state is an implementation failure. Repair it and preserve the failed run; extra updates or sampler tuning cannot validate the wrong checkpoint. |
| D02 — Nonfinite values, covariance failures, score disagreement or backend disagreement | Dtype; active floor/jitter/repair branch; innovation floor `1e-12`; rho `1e-14`; eigensolver sweeps/residuals; parity and map tolerances (N01, N04–N19) | Replay the first failing state on the same target, preserving accepted/proposed status. Inspect normalized factor/Sylvester residuals and finite differences over the existing step ladder on a fixed branch; inspect both sides if a branch changes. | Resolved value/score mismatch blocks numerical qualification. Establish whether conditioning, implementation or branch semantics failed before evaluating a defined numerical repair; passing a relaxed tolerance is not a repair. |
| D03 — Loss explodes, oscillates, updates vanish, or training stalls after changing beta | LR `5e-4/1e-3`; Adam `.9/.999/1e-7`; clipping `10` and its `10%` alert; moment carry/reset; initialization SD `.02` (A07–A09, A16, A18–A19) | After D01/D02, inspect raw gradient, clipping factor, actual update and optimizer denominator. Compare only the implicated change: half LR, clip `10/30`, beta2 `.999/.99`, or carry/reset at a beta boundary, using the same parent and a declared paired development design. | A supported change in learning implicates that optimizer setting in this scope. Unresolved differences remain inconclusive; clipping frequency alone does not prove defective training. |
| D04 — Loss plateaus and the map has poor geometry despite healthy optimization | Widths `(16,16)/(32,32)`; two stages; tanh/order; scale cap `2`; hidden initialization (A04–A07, A17–A18) | Check saturation, scale sensitivity and independent posterior pullback geometry. Price one relevant capacity contrast: width `(64,64)` or four stages; test cap `2/3` only when the cap is implicated. | Improved heldout fit/geometry supports a capacity or parameterization explanation. A failed contrast may still be underoptimized; it does not prove the flow family cannot work. |
| D05 — Learning is noisy or expensive, or larger batches do not help | Batch `8/32/128`; gradient noise; actual batch-native graph; compilation and update cost (A11–A12, N02, N21–N23) | Estimate variability from independent batch gradients and price complete graphs. Use the existing paired batch comparison, recording both matched update work and unequal total overhead. | Separate poor gradient information from hardware cost or scalar fallback. Select by resolved learning and downstream progress per cost, not updates/second alone. |
| D06 — A training cap is reached, or a short checkpoint is described as trained | Cumulative `128/512/2048/8192` schedule; actual learning curve; promised cohort coverage; remaining budget (A13–A15, B08–B12) | Establish whether the curve is still improving with resolved uncertainty, or is flat under D03/D04. Price the next justified increment plus all remaining validation/sampling before extending. | Ongoing progress means training remains incomplete; a flat curve triggers diagnosis. Reaching a count, including `8192`, never certifies sufficient training. An unaffordable next step is under-budgeted. |
| D07 — Checkpoint decisions change with the validation bank, or a plateau is uncertain | Delta `.04` nat/draw; half-width `.02`; banks `768/3072/12288`; repeated looks; two-comparison rule (V01–V08) | Check paired-loss variance, influential rows, bank disagreement and previous selection use. Calculate required bank size under M4 before adding evaluations; freeze a pair and use fresh evidence for a nominal confidence statement. | Excess noise or selection effects can invalidate the stopping decision. Do not call an unresolved interval a plateau or treat `.04` as a posterior-bias tolerance. |
| D08 — Training succeeds only for selected seeds, or method costs vary widely | Three roots; two-of-three nomination; three final method replications; initialization and all failed attempts (A07, A13, V09, B03–B07) | Inspect every root and failure, separating within-bank error from between-root variation. Check stream independence, then price replication for the uncertainty that matters. | A lucky map or three descriptive costs cannot establish reliability or superiority. Preserve viable candidates and report the unresolved variability; do not remove failed seeds from the comparison. |
| D09 — Loss is low but whitening or sign-region exploration is poor | Reverse-KL coverage; own-map occupancy bank `4096`; training adequacy/capacity; initialization and reference coverage (A04–A19, V10–V11, T12, R01–R05) | Check inverse-map values for independently validated posterior draws, not `T^-1(T(z))`. Inspect covariance, tails, transformed scores and region mass with uncertainty, then apply the implicated D03/D04/D06 repair. | Good self-generated loss does not establish whitening. Common missing mass calls for coverage/training repair; without a usable reference, whitening remains unassessed rather than passed. |
| D10 — Very low acceptance, frequent invalid proposals or failure to verify a kernel | Frozen target/map/metric; score; epsilon seed `.1`, curvature guard `.8`; repair bounds; per-pair evidence `64/128/256` (H01–H02, H11–H20, H25) | Use D01/D02 when indicated. For valid target/score evidence, inspect signed energy and acceptance evidence in the actual coordinates; apply only the supported directional epsilon repair and freshly verify the exact child pair. | Numerical invalidity and excessively large steps are different failures. Inconclusive evidence needs its declared extension; broadening acceptance bands cannot qualify a failed kernel. |
| D11 — Acceptance is high but chains barely travel, recur or do not cross regions | Duration `L*epsilon`; L grid up to `25`; movement `.05`, repeats `.95`, return floor `1e-4`, recurrence lags `2..16` (H11–H16, H21–H23) | Inspect physical/latent displacement, recurrence and region dwell times. Price a supported new L candidate when short duration is implicated; `50/100` are conditional hypotheses and require a supported binding. | Acceptance alone can hide short or periodic trajectories. Reassess transport if stiffness persists; do not describe an L-limited search as exhausting classical HMC possibilities. |
| D12 — Tuned classical geometry is unstable or remains highly correlated | Actual metric update; mass convention and affine layers; `1000` adaptation steps; window schedule; shrinkage `.25`; information/condition screens (H04–H06, G01–G12) | Inspect the issued transform, covariance and covariance-moment information across windows/independent evidence. Determine whether it estimated trapped local geometry, insufficient data or overshrunk correlations before a priced metric-preparation contrast. | A completed schedule can still give a poor metric. Repair the identified preparation problem; legacy metadata or enough raw rows cannot establish learned geometry. |
| D13 — Extra charts/temperatures add cost without cold-chain improvement | K `2/4`; ladders `(0,.5,1)` and `(0,.25,.5,.75,1)`; half-chart restart at `.5`; uniform chart weights; one-update swap cadence (E01–E11) | Verify each chart is actually trained and each transition valid. Inspect overlap, chart diversity, within-slot movement and travel; test the single implicated change rather than changing charts, spacing and cadence together. | Separate duplicate maps, insufficient training, temperature bottleneck and swap bookkeeping. High swap acceptance alone cannot establish useful posterior exploration. |
| D14 — Crossing/travel screens fail despite otherwise promising results, or pass while mixing remains poor | Four cold crossings; sixteen aggregate round trips/orientation; one per replica identity; region mass and dependence (E12–E13, T12, P03, P15) | Compare actual dwell/travel distributions and independent region mass with event-specific MCSE. Examine dependence and whether a region is substantial; do not apply an IID/Poisson argument to dependent crossings. | These counts are uncalibrated observability screens. Substantial missed mass requires repair; a negligible region may justify an explicit scientific-contract revision, never a hidden threshold waiver. |
| D15 — R-hat looks acceptable but estimates disagree, MCSE is unstable, or retained cap is exhausted | Correct rank/folded arithmetic; original-scale versus bulk/tail ESS; warmup/window/draw counts; MCSE estimator; batch size `floor(sqrt(n))`, lugsail `3/.5`, twenty batches (P01–P15) | Recompute saved-draw diagnostics with an independent diagnostic, inspect starts/regions and compare declared MCSE estimators. Check the actual named means/events/quantiles and forecast required evidence from valid development draws. | Diagnose arithmetic, nonstationarity, missed mass, estimator uncertainty or insufficient precision separately. Do not choose the smallest MCSE, shorten warmup after the fact or relax accuracy goals to fit the cap. |
| D16 — Reference changes on refinement or disagrees with all methods | Prior-standardized box radii `4/6/8`; nodes `9/17/33`; actual integration rule; tail/normalizer error; reference independence; one-third total error allowance and equivalence scales (R01–R05, P16–P20) | Inspect total error including normalization and tails, plus shared target-code risks. Compare only when reference and method uncertainty are usable; price the next discriminating domain/refinement or independently specified reference check. | Several methods can share a missed region or wrong target. An unqualified reference cannot prove all samplers wrong; unresolved total reference error leaves promotion incomplete. |
| D17 — Claimed speed advantage disappears or three replications give contradictory results | Total cost at matched accuracy; amortization; failures/timeouts; paired log-cost model; replication and multiplicity (B03–B07, P19–P20) | Reconcile complete costs and censoring, verify every compared run met the same criteria, and inspect replicate-level differences with the declared uncertainty model. | Missing costs or invalid arms make comparison unfair. With insufficient uncertainty support, retain descriptive results and viable unranked methods. |
| D18 — GPU OOM, compilation stalls, poor throughput or lost work consumes the budget | Batch/native graph; memory growth and `4 GiB` pilot assumption; worker/thread count; serial HMC topology; eight-update pricing; `128`-update checkpoints and sampler chunks (N02–N03, N21–N23, A11–A12, A20, H26–H27, B08–B13) | Inspect actual placement, allocator/compile peaks, call duration, concurrent workloads and resume state. Reprice only implicated scopes with full overhead and explicit remaining budget before another cohort. | Infrastructure/resource estimates may be wrong while the method remains viable. Repair within the authorized scope, charge attempts, and preserve required downstream work; do not cut scientific evidence merely to fit an optimistic forecast. |

### How to execute a diagnostic repair

1. Preserve the failure, exact settings, source, random streams and actual
   checkpoint. Name the symptom and its applicable parameter IDs. Classify the
   failed quantity as numerical validity, promotion evidence, repair trigger
   or explanation under the existing contract.
2. Read existing telemetry and unchanged-scope checks first. State the two
   plausible explanations the next test will distinguish, its comparator,
   expected observations and decision rule. A test should answer why the
   failure happened, not merely produce a better-looking secondary metric.
3. Choose the smallest useful, funded check. For a numerical experiment, record
   the evidence contract, exact command/environment, output directory, attempt
   cost and stop conditions in the active campaign plan before execution.
   Diagnostic CPU/reference exceptions remain labeled; serious NeuTra training
   remains batch-native GPU work with the required memory policy and XLA.
4. In development, vary the implicated factor while preserving the parent,
   target/data and meaningful comparators. Use paired roots/rows where valid,
   preserve all seed outcomes and separate stochastic uncertainty sources.
   If an interaction is indicated, state it and test the smallest explicit
   combined contrast; do not attribute a multi-setting change to one cause.
5. Record whether the evidence supports the proposed explanation, rules it out
   in the tested scope, or is inconclusive. A repair that changes the map,
   numerical program, metric or kernel needs its applicable downstream checks
   and fresh tuning/verification; it does not inherit the parent's qualification.
   Keep retained kernels fixed. A failed confirmation is preserved, subsequent
   repair uses development data, and any new confirmation uses fresh evidence.

If no pending test is useful or affordable, report precisely what remains
unresolved and why. Do not exhaust all alternative settings merely to satisfy
this checklist. Candidate rejection and direction rejection remain different
decisions, and a diagnostic that explains a failure does not itself establish
posterior validity.

Target assumptions and accuracy requirements need explicit treatment when
implicated. Prior SD `4`, fixture/noise coefficients and the UKF rule define
the current target (T01–T13, N04); changing them is a model-sensitivity study,
not a sampler repair. The `.02` relative mean MCSE, `.01` event MCSE, `.05`-SD
quantile MCSE and equivalence margins express the intended research resolution
(P01–P03, P16–P20). Examine their usefulness and cost, but do not loosen them
after observing a failure and call the original requirement met. Owner policy,
compute authorization and platform permissions remain applicable.

### Record in the existing result note

For each executed investigation, add a compact entry to the existing campaign
result and update the corresponding parameter row here with its evidence link:

```text
Observed problem and first failing artifact/state:
Parameter IDs, actual values and why they were examined first:
Competing explanations; diagnostic role and predeclared decision rule:
Parent/comparator, changed factor, command, source, seeds and output:
Observed values, uncertainty and supported / ruled-out / inconclusive finding:
Repair or next action; affected checks and remaining unproved claims:
Attempt wall time, budget charged and remaining allocation:
```

A completed diagnostic only supports the tested scope. If it is skipped because
existing evidence resolves it or another first failure makes it irrelevant,
record that reason briefly. Missing telemetry remains missing evidence; it does
not establish that the associated setting was sensible.

Continuation note: this section implements the owner's request to examine the
uncalibrated quantities first when problems appear. The skeptical desk audit
found that an indiscriminate sweep, automatic threshold relaxation or retuning
on failed confirmation would defeat that purpose; the ordered procedure above
avoids those errors. This documentation change launches no experiments, selects
no new numerical defaults and does not change the campaign budget. The actual
training/export/reference integration and parameter calibrations remain pending.

## Mathematical checks used by the tables

### M1. Separate model definition, approximation and arithmetic

Write the target explicitly as

\[
 \pi_\beta(\theta)=Z_\beta^{-1}p(\theta)
       L_{\rm UKF}(y_{1:30}\mid\theta)^\beta,\qquad
 p(\theta)=\mathcal N(\mu_0,16I_4).
\]

Fixing data and coefficients makes this a reproducible target. It does not show
that the four-dimensional slice represents inference with uncertain nuisance
parameters, that the UKF likelihood is sufficiently accurate, or that the
fixture describes financial data. Those require separate sensitivity or model
validation. In the inspected fixture the simulated free-coordinate truth is
the prior center. Also, data generation starts at `params.initial_mean`, without
an initial covariance draw. Thus this one dataset is not a repeated-dataset
calibration study drawn from every element of the inference prior/initial-state
law. These are known design choices, not demonstrated inference defects.

For `s(r)=log(1+exp(r))+f`,
`ds/dr=sigmoid(r)` and `d(s^2)/dr=2*s*sigmoid(r)`.
The positive floor f defines the model's variance, rather than merely preventing
a computer exception. Check the ratio `f/softplus(r)` and propagate a perturbation
to the reported quantities in a separately declared model-sensitivity study.
For the current noise ranges f=0.0001 is small relative to the SD, but that
fact alone does not bound a 30-step likelihood or posterior error.

For unscented points at state dimension n,
`lambda=alpha^2*(n+kappa)-n`, displacement scale is `sqrt(n+lambda)`,
`w0_mean=lambda/(n+lambda)`, `w0_cov=w0_mean+1-alpha^2+beta`, and
the other weights are `1/[2*(n+lambda)]`. With n=60, alpha=1, beta=2,
kappa=0 this gives 121 points, displacement `sqrt(60)`, and weights
0, 2 and 1/120. Direct summation recovers the input Gaussian mean and
covariance. In one dimension the covariance correction 2 also recovers the
variance of the quadratic transform of a centered Gaussian. It does not make
all nonlinear transforms or multivariate fourth moments exact. Compare
independently estimated transformed moments before making that stronger claim.

### M2. Tolerances must match conditioning and the computed value

For an SPD square root `S^2=C`, differentiation gives the Sylvester equation

\[
 S\,dS+dS\,S=dC,\qquad
 (Q^T dS Q)_{ij}=\frac{(Q^T dC Q)_{ij}}{s_i+s_j}.
\]

The denominator involves sums of positive square-root eigenvalues, not gaps
between covariance eigenvalues. Nearly repeated positive eigenvalues therefore
do not alone make this derivative singular; small eigenvalues can. A small
factor residual does not establish an accurate derivative near singularity.
Measure both reconstruction and Sylvester residuals, divided by their operand
scales, and independently compare the full value derivative.

For a covariance perturbation E with
`eta=||C^(-1/2) E C^(-1/2)||_2 < 1`, each relative eigenvalue perturbation
lies in `[-eta,eta]`. Consequently
`abs(logdet(C+E)-logdet(C)) <= -n*log(1-eta)` and, for fixed innovation r,

\[
 |r^T[(C+E)^{-1}-C^{-1}]r|
 \leq \frac{\eta}{1-\eta}\,\|C^{-1/2}r\|^2.
\]

These follow by diagonalizing the normalized perturbation. They give a local
Gaussian log-likelihood error bound; a recursive filter also changes future
means and covariances, so its full error still needs propagation or replay.
`condition(C)*machine_epsilon` is a warning scale, not a universal forward-error
certificate. The coefficient 64 in the inherited eigensolver residual screen
is an engineering allowance; the form `n*epsilon` does not derive 64.

The inspected strict classifier sends valid C to `C+rho*I` and permitted
near-SPD C to `C+2*rho*I`, where `rho=max(requested_floor,1e-14)`.
At the branch boundary the shift changes. On a fixed branch with fixed rho,
`d(C+rho*I)=dC`; at the boundary that smooth identity does not establish a
derivative. Spectral clipping, if another active operation uses it, also needs
the derivative of that operation. A branchwise Sylvester solve must not be
described as proof of a globally smooth finite-program score. Boundary tests
need one-sided limits, branch labels and the actual scalar value, not only a
score comparison between two related implementations. Nonsmoothness weakens
smooth energy-error predictions; it does not by itself prove that a correctly
implemented Metropolis kernel has the wrong invariant law.

For a smooth scalar f, central differences obey

\[
 D_h f=f'+\frac{h^2}{6}f'''+O(h^4),\qquad
 |\hbox{evaluation-error contribution}|\leq\delta_f/h
\]

when each endpoint's absolute evaluation error is bounded by delta_f.
Balancing `A*h^2+B/h` gives `h^3=B/(2A)`. The familiar
`epsilon_machine^(1/3)*coordinate_scale` assumes compatible derivative and
value scales; it is only a seed. Richardson's `/3` estimates the leading
truncation error of `D_(h/2)` from `D_h-D_(h/2)`. Cancellation, noise or a branch
crossing can destroy that estimate. Use the whole declared five-step refinement
curve; do not select the h that agrees with the tested score.

A useful bridge from numerical error to posterior error exists only with a
strong bound: if `abs(ell_tilde-ell)<=a` uniformly over the target support,
then normalized density ratios are between `exp(-2a)` and `exp(2a)`, and total
variation is at most `tanh(a)`. One derivation bounds a likelihood ratio of mean
one at its two permitted endpoints. Therefore bounded quantities g have error
at most `(sup(g)-inf(g))*tanh(a)`. Pointwise replay tolerances do not give this
uniform bound; unbounded means and quantiles need more assumptions. This is why
`1e-10` numerical parity is not by itself posterior certification.

### M3. What reverse KL and map training establish

For invertible theta=T_phi(z), z~q0=N(0,I4), change of variables gives

\[
 D_{\rm KL}(q_\phi\Vert\pi_\beta)
 =E_{q_0}[\log q_0(z)-\log|\det J_{T_\phi}(z)|
                     -\ell_\beta(T_\phi(z))]+\log Z_\beta.
\]

Thus dropping `log q0` and `log Z_beta` does not change the gradient with respect
to phi at fixed beta. It does prevent interpreting the raw loss as an absolute
KL without the missing constants. A loss decrease of 0.04 is a KL decrease of
0.04 only for that same objective and beta; it does not imply that the final
KL is small. Changing the likelihood coefficient, rejecting training rows or
changing the Jacobian coefficient changes the objective. Clipping the gradient
changes the optimization update, rather than defining a new scalar target loss.

Even small absolute reverse KL can tolerate an unvisited low-mass region:
if q is pi conditioned on A, `KL(q||pi)=-log pi(A)`. For pi(A)=0.99 this is
about 0.01005 despite missing the other 1%. This limiting example explains
why self-generated validation cannot certify mode or event coverage. Actual
whitening means `T^(-1)(theta)` for independent theta~pi resembles N(0,I).
Inversion of one's own T(z) only tests algebra.

For independent per-row gradients with finite covariance Sigma_g, a batch
average has covariance `Sigma_g/B`. A proposed noise diagnostic is
`trace(Sigma_g)/(B*||E g||^2)`, or its directional counterpart. Estimate it
from independent batch gradients when full per-row gradients are too costly;
do not introduce a forbidden scalar/pfor training path. Near a stationary
point the denominator can vanish, so this is an explanatory scale, not a
pass threshold. Select batch size by learning achieved per measured total cost
with uncertainty as well as numerical health.

For `m_t=beta*m_(t-1)+(1-beta)*g_t`, weights decay geometrically;
`1/(1-beta)` is a memory scale and `log(1/2)/log(beta)` a half-life.
Neither determines how many updates suffice. Adam's bias correction also
changes its early behavior. In a locally quadratic objective, an ordinary
gradient step along curvature lambda is stable for `0<eta*lambda<2`.
For Adam this is at most a frozen-preconditioner diagnostic; moving moments,
clipping and noise prevent turning that formula into a certified LR rule.

### M4. Validation resolution and selection uncertainty

For a **fixed** checkpoint pair and independent base rows, define
`I=loss_old(z)-loss_new(z)`. Under a suitable CLT its mean has standard error
`s(I)/sqrt(M)`. The nominal 95% half-width is `1.96*s/sqrt(M)`, so a desired
half-width h suggests `M=ceil((1.96*s/h)^2)`. With h=.02, s=1 needs
9,604 rows; s=10 needs 960,400. The proposed cap 12,288 resolves that h only
if s is roughly at most 1.1311 under those assumptions. A factor-four bank
increase halves the standard error, not the bias or optimization uncertainty.

The proposed delta=.04 and h=.02 have **no established q20 adequacy basis**.
They set optimization resolution; calibrate whether differences on that scale
predict worthwhile downstream improvement and whether resolving them is
affordable. A common fixed validation bank is selection data after it has
influenced checkpoint, architecture or repair decisions. Even the next nominal
interval is then not generally a fixed-comparison 95% interval. Treat those
intervals as development summaries. For a confirmatory loss statement, freeze
the pair, use a fresh independent bank, and predeclare the number of looks and
comparisons (or a valid sequential procedure). Independent posterior
confirmation remains necessary. Bank replication is not training-seed replication.

### M5. Temperature spacing, chart probabilities and transport checks

Let a(theta)=log L(theta). Differentiating Z_beta, when the expectations exist,
gives `d log Z_beta/d beta=E_beta[a]` and
`d^2 log Z_beta/d beta^2=Var_beta(a)`. A Taylor expansion then gives

\[
 D_{\rm KL}(\pi_\beta\Vert\pi_{\beta+d\beta})
     =\tfrac12(d\beta)^2\operatorname{Var}_\beta(a)+O((d\beta)^3).
\]

This motivates smaller spacing where likelihood variance is larger, rather
than declaring .5 or .25 universally sensible. It does not supply a universal
target for `(d beta)^2*Var(a)`. Estimate overlap and travel on valid development
samples, price bisection, and assess cold-stream accuracy at matched total cost.
Poorly mixed samples can understate the variance and conceal a bottleneck.

For physical replica exchange at beta_i,beta_j and states x_i,x_j, the log
Metropolis ratio is `(beta_j-beta_i)*(a(x_i)-a(x_j))`; the prior cancels.
A transported swap must additionally have the coordinate and Jacobian factors
required by its actual proposal. Check that kernel's invariant-law derivation
and inverse numerically; do not substitute the physical formula blindly.
If each chart kernel leaves pi invariant, a state-independent mixture with
nonnegative gamma summing to one also does: `pi*sum gamma_k P_k=pi`.
Uniform gamma is therefore valid but need not be efficient. State-dependent
gamma needs a different derivation. A density-mixture weight alpha is a
different object and cannot stand in for this proof.

### M6. HMC scale, mass conventions and acceptance

Use the convention `p~N(0,M)`, kinetic energy `p^T M^(-1)p/2`.
For quadratic potential `U(q)=q^T H q/2`, frequencies squared are eigenvalues
of `M^(-1/2) H M^(-1/2)`. In one mode the leapfrog matrix has determinant one
and trace `2-epsilon^2*lambda`; eigenvalues stay on the unit circle when
`0<epsilon*sqrt(lambda)<2`. This derives the quadratic scale, but neither the
0.8 guard nor the .1 seed. Nonlinear curvature, branch changes and finite
arithmetic require actual trajectory checks. For a Gaussian covariance Sigma,
this convention is whitened by M=Sigma^(-1); an implementation storing an
inverse mass or using q=mu+A*z uses different labels. Inspect the actual
transform and kinetic energy before calling a covariance a mass.

Trajectory duration is `tau=L*epsilon`. Very small epsilon at fixed L can
produce excellent acceptance and very little travel; near-periodic durations
can return near the start. Approximate periods `2*pi/sqrt(lambda)` motivate
diagnosing recurrence and extending L, not an unsupported universal L grid.
The exact Metropolis ratio is `min(1,exp(-Delta H))` for the reversible,
volume-preserving integrator with the declared target. Acceptance screens test
proposal mechanics. There is no target-specific derivation here of optimal
acceptance .70 or its bands, and no claim that acceptance proves stationarity.

#### September 22: acceptance-based initial step estimates

An acceptance target supports an approximate step calculation when the energy
error scale is known. It does not determine epsilon by itself. The derivation
below is conditional mathematics, not a new q20 calibration or a configuration
change. It uses stationary starts, a smooth target, small-step leapfrog behavior
and a fixed integration duration. These assumptions have not been established
for the current four-chain, prior-started q20 tuning observations.

Let `D=H(proposal)-H(start)`. Under the stationary joint position/momentum
distribution and a volume-preserving bijection, change of variables gives
`E[exp(-D)]=1`. If `D` is approximately Gaussian with variance `v`, this identity
motivates mean `v/2`. Integrating the Metropolis probability then gives

\[
 \bar a \simeq P(D\leq0)+E[e^{-D}1_{D>0}]
 =2\Phi(-\sqrt v/2).
\]

Here Phi is the standard-normal CDF. For leapfrog over fixed duration, the
leading energy error is second order, so its variance has the local expansion
`v(epsilon) approximately C*epsilon^4` when that leading term is nonzero.
Consequently an acceptance-based proposal is

\[
 \epsilon_0\simeq
 \left\{\frac{4[\Phi^{-1}(1-a_*/2)]^2}{C}\right\}^{1/4}.
\]

At `a_*=0.70`, the numerator is approximately `0.5938874473`, calculated with
Python standard-library `statistics.NormalDist.inv_cdf`. C depends on the
target, mass and trajectory duration; a justified energy-error pilot could
estimate it as `Var(D)/epsilon_pilot^4`. Neither the constant nor the Gaussian
energy-error approximation is established by naming the transport NeuTra.

For the ideal independent standard-normal target, one leapfrog trajectory
preserves `p^2+(1-epsilon^2/4)*z^2` in each coordinate. Thus
`D=(epsilon^2/8)*(z_end^2-z_start^2)`. Substituting the leading exact trajectory
`z_end=z_start*cos(tau)+p_start*sin(tau)`, with independent unit Gaussian
position and momentum, gives per-coordinate leading variance
`epsilon^4*sin(tau)^2/16`. Hence `C=d*sin(tau)^2/16`, explaining the familiar
`d^(-1/4)` step scaling away from return phases. This is an asymptotic scale
argument; the normal approximation can be poor at d=4. At fixed L, changing
epsilon also changes `tau=L*epsilon`, so C is not generally fixed. Near a return
phase the leading term vanishes. Neither the power-law rescaling nor monotone
acceptance may be assumed for the saved q20 per-L search.

#### Explicit L dependence for an ideal Gaussian pullback

Substituting `tau=L*epsilon` into the leading Gaussian energy-error calculation
above makes the dependence explicit:

\[
 a(\epsilon,L)\simeq
 2\Phi\!\left(-\frac{\sqrt d\,\epsilon^2
                  |\sin(L\epsilon)|}{8}\right).
\]

For a desired acceptance `a_*`, this gives an implicit equation
`epsilon^2*abs(sin(L*epsilon)) = 8*Phi^(-1)(1-a_*/2)/sqrt(d)`.
Only in the additional short-trajectory limit `L*epsilon << 1` may sine be
replaced by its argument, yielding the explicit initial-scale approximation

\[
 \epsilon_0(L)\simeq
 \left\{\frac{8\Phi^{-1}(1-a_*/2)}{L\sqrt d}\right\}^{1/3}.
\]

Thus the familiar `L^(-1/3)` dependence is a short-trajectory approximation,
not a law valid at arbitrary L. For d=4 and `a_*=0.7`, the numerator after
division by `sqrt(d)` is 1.5412818656. At L=3,5,9,13,18,25 this cubic formula
proposes approximately .8009,.6755,.5553,.4913,.4408,.3950, with `L*epsilon`
approximately 2.40,3.38,5.00,6.39,7.93,9.88. Every one violates the presumed
short-trajectory limit; these numbers are algebraic illustrations, not usable
q20 initial settings. The energy-normal approximation itself is not reliable
by default at dimension four.

An exact stationary ideal-Gaussian calculation avoids that latter approximation.
For `0<epsilon<2`, let `r=sqrt(1-epsilon^2/4)`,
`theta=2*asin(epsilon/2)` and `psi=L*theta`. L leapfrog steps in each coordinate
have matrix

\[
 A_L=\begin{pmatrix}
 \cos\psi & \sin\psi/r\\
 -r\sin\psi & \cos\psi
 \end{pmatrix}.
\]

Its determinant is one and `trace(A_L^T*A_L)=2+b`, where
`b=epsilon^4*sin(psi)^2/(16*r^2)`. The eigenvalues of `A_L^T*A_L` are
`rho` and `1/rho`, with `rho=1+b/2+sqrt(b*(b+4))/2`. At stationary independent
standard-normal positions and momenta, rotational invariance gives
`D=((rho-1)*X+(1/rho-1)*Y)/2`, with independent `X,Y ~ chi_squared(d)`.
For the reversible proposal, expected acceptance is `2*P(D<0)` (with the
zero-error case handled by continuity), hence

\[
 a(\epsilon,L)=2I_{1/(1+\rho)}(d/2,d/2),
\]

where I is the regularized incomplete beta function: `D<0` is equivalent to
`X/(X+Y)<1/(1+rho)`. For d=4, write `x=1/(1+rho)`; then
`a=6*x^2-4*x^3`. Target acceptance .7 gives `x=0.3986103026924348`,
`rho=1.5087158892919876`, and `b=0.17153120601095018`. The exact d=4
ideal-Gaussian equation for epsilon at any declared L is therefore

\[
 \frac{\epsilon^2|\sin(2L\arcsin(\epsilon/2))|}
      {\sqrt{1-\epsilon^2/4}}
 =4\sqrt{0.17153120601095018}.
\]

For the current L grid, numerical inversion gives the following **smallest
positive root** in `0<epsilon<2` for each L. Selecting the smallest root is an
explicit initialization convention; it does not establish an optimal kernel.

| L | Smallest epsilon root | Exact ideal-Gaussian expected acceptance |
| --- | --- | --- |
| 3 | 1.2595670000640444 | 0.70 |
| 5 | 1.3053652851902746 | 0.70 |
| 9 | 1.3474553766269795 | 0.70 |
| 13 | 1.2069251634800130 | 0.70 |
| 18 | 1.1997862812031914 | 0.70 |
| 25 | 1.2113847601121580 | 0.70 |

The deterministic [solver](artifacts/q20-gaussian-initial-epsilon-2026-09-22/solve.py)
and [result](artifacts/q20-gaussian-initial-epsilon-2026-09-22/result.json)
preserve the exact computation. Root isolation uses
`g(epsilon)=epsilon^3*abs(U_(L-1)(1-epsilon^2/2))`, the same equation written
with a Chebyshev polynomial. Its positive zeros are
`r_k=2*sin(k*pi/(2*L))`, for `k=1,...,L-1`. Between zeros,
`d(log g)/d(epsilon)=3/epsilon+sum_k(1/(epsilon-r_k)+1/(epsilon+r_k))`
is strictly decreasing. Each interior lobe therefore has one maximum, allowing
earlier lobes to be excluded before bisecting the rising branch of the first
lobe reaching the required value. The final lobe is increasing throughout.
This avoids a global monotonicity assumption or an unresolved sampling grid.

Direct multiplication of the one-step leapfrog matrix independently checked
expected acceptance at each root. Maximum absolute acceptance discrepancy from
0.7 was `9.99e-16`; maximum equation residual was `6.44e-15`. The script records
an arithmetic tolerance of `1e-12`, its source checksum, environment, command
and elapsed time. No target evaluations or sampling occurred, and no q20
runtime setting was changed. These values are conditional Gaussian reference
seeds; the current map and prior-started chain bank have not been shown to meet
their assumptions.

The September 22 [actual-map canary](bayesfilter-q20-gaussian-epsilon-canary-result-2026-09-22.md)
subsequently rejected every Gaussian-root pair in both the original-start and
fresh map-proposal banks: 0 of 192 proposals accepted, numerical health failures
at every pair. The .0620027091/L3 controls accepted 29 of 32 and passed health
checks. The table above remains an analytic ideal-Gaussian reference; none of
its roots is qualified for the current q20 frozen map.

This equation can have multiple roots as L changes the trajectory phase; a
root is not an efficiency or exploration certificate. It assumes an exact
Gaussian target and stationary starts, neither of which is established for the
q20 map and prior-started bank. It is eligible as a conditional analytic
reference, not a replacement for measurement and fresh verification.

Standard-library deterministic checks compared the matrix formula with direct
leapfrog iteration and the energy identity at epsilon .075,.5,1.2,1.8 and
L=3,5,9,25. Maximum absolute discrepancy was `2.89e-15`; these points are
arithmetic check cases, not numerical defaults. The beta-polynomial inverse
above was evaluated by bisection. No stochastic run or q20 target call was
made. Across the actual L grid, the exact ideal-Gaussian stationary acceptance
at the previously tested epsilons .0620027091 and .0759375 would be between
approximately .99894 and .99987. The actual nonstationary q20 observations
cannot be identified with this ideal reference or used to diagnose one unique
failure cause from their difference.

There is also an existing geometry proposal in
`bayesfilter/inference/hmc_geometry.py::_initial_step_size`:

\[
 \epsilon_{\rm geom}
 =\min\{c\,d^{-1/4}/\omega_{\rm rms},\;2s/\omega_{\max}\}.
\]

The inspected code has inherited coefficients `c=0.5` and `s=0.8`. Under ideal
unit frequencies and d=4 it proposes `0.3535533906`; c is not derived from an
acceptance target of 0.70. This formula exists in the ordinary geometry
initializer, but the executed q20 fixed-transport route does not call it. Its
unit-frequency example is not a recommended replacement for the current map's
measured proposals. A current-map analogue must use geometry in the fixed
latent coordinates and preserve identity mass. Step adaptation or the current
public candidate search must measure actual acceptance, freeze the proposed
pair and independently verify it. The existing shared candidate-set guide does
not require adding ordinary mass adaptation or switching to a legacy tuner.

### M7. Posterior accuracy and independent reference error

For a stationary quantity g with variance sigma_g^2 and integrated
autocorrelation time tau_g, the Markov-chain CLT gives

\[
 \operatorname{Var}(\bar g)\simeq\frac{\sigma_g^2\tau_g}{N},\quad
 \mathrm{ESS}_g=N/\tau_g,\quad
 \frac{\mathrm{MCSE}(\bar g)}{\sigma_g}\simeq\mathrm{ESS}_g^{-1/2}.
\]

Thus relative mean MCSE .02 suggests ESS 2,500 **for the original-scale mean**.
Rank bulk ESS 400 is a different diagnostic. An indicator has variance p(1-p),
so MCSE .01 requires ESS `p*(1-p)/.01^2`, at most 2,500. Constant sampled
indicators do not establish p=0 or 1. For a regular quantile x_p with positive
density f(x_p), inversion of the empirical CDF gives

\[
 \mathrm{MCSE}(\hat x_p)\simeq
 \frac{\sqrt{p(1-p)/\mathrm{ESS}_{1\{X\leq x_p\}}}}{f(x_p)}.
\]

Density uncertainty, ties, rare tails and nonstationarity can invalidate this
approximation. For a Gaussian, .05-SD quantile MCSE suggests about 628.32
effective indicator draws at the median and 2,854.36 at .025/.975. This is an
illustration, not an assumption that q20 is Gaussian or has equal autocorrelation
for every quantity. The API's 5%/95% tail ESS does not certify 2.5%/97.5% precision.

For independent method/reference estimates, variance of their difference is
the sum of variances. A conservative equivalence check is
`abs(difference)+critical*sqrt(se_method^2+se_ref^2)+bias_bound <= margin`.
A deterministic error allowance is added as a bound, not combined as variance.
Reference error <=one third of the **permitted** sampler error increases combined
SE by at most `sqrt(1+1/9)=1.05409` when the sampler is at that limit. It may
dominate an unusually precise sampler; always use measured errors.

For J approximately normal comparisons, Bonferroni family error .05 gives
`critical=Phi^(-1)(1-.05/(2*J))`. J=17 gives 2.9738199 and J=5 gives
2.5758293. This controls a fixed family only to the extent that the individual
intervals are valid; repeated stopping and adaptive selection need separate
treatment. Start-group estimates require their own errors and dependence
accounting; pooled precision does not imply between-group equivalence.

### M8. Resource and replication calculations

For C independent chains of length n with similar long-run variance,
`ESS approximately C*n/tau`; a target of 2,500 effective draws with C=4 and
n<=10,000 can tolerate tau only up to about 16 at the cap. This is a forecast
from valid development evidence, not a license to use unreliable canary ESS.

Charge cumulative training updates once, including compilation, validation,
checkpoints and failures. For HMC use actual call timing; the interface's
`C*sum_stage(draws+discarded)*(L+1)` is a work-accounting estimate, not an exact
target-call count. Per-worker time sums across parallel processes.

For paired log costs D with approximately independent normal replications,
`mean(D) +/- t_(n-1,.975)*s_D/sqrt(n)` is a small-sample interval. At n=3,
the critical value is about 4.3027 and normality is essentially untested.
Ignoring magnitude, a two-sided sign test with n nonzero unanimous independent
pairs has p-value `2^(1-n)` under equal sign probability: six gives .03125.
Six is not a robust power calculation. The proposed normal planning formula
`n approximately [(1.96+.8416)*s_D/Delta]^2` assumes a fixed-size comparison,
adequate variance estimate, no informative timeouts, and a prespecified
worthwhile effect. Neither 95% confidence nor 80% power follows from physics
or finance; they are design conventions.

## T. Target and fixed data

| ID / parameter and value | Basis and present status | Smallest useful check and decision | Failure meaning and next action |
| --- | --- | --- | --- |
| T01 - Free parameter dimension: 4 | Fixed scientific slice, not a derived sufficient parameterization. Names/order are in the ledger. | Check packing/unpacking and score coordinates against those four names; examine sensitivity to fixed nuisance coefficients separately. | Wrong indexing invalidates implementation; sensitivity to nuisance values limits the four-coordinate inference claim. |
| T02 - Latent / hidden / augmented dimensions: 20 / 20 / 60 | Fixed complexity fixture; 60=20+2*20 is derived. No evidence that 20 is optimal capacity. | Assert actual state shapes and covariance dimension. For model adequacy, compare separately specified dimensions against the scientific data question. | Shape mismatch is a bug; dimension sensitivity is model dependence, not a reason to transfer another scope's tuning. |
| T03 - Horizon / observations per time: 30 / 1 | Fixed dataset, not sample-size adequacy evidence. | Verify every phase consumes identical 30-by-1 values; inspect information/identifiability and prior-versus-posterior uncertainty. | Data mismatch invalidates comparison; weak information means a prior-sensitive posterior, not necessarily poor HMC. |
| T04 - Prior: mu=(.35,-.08,.65,.05), SD=4, variance=16 | Fixed Gaussian modeling assumption. M1 shows meaning; center is also fixture truth. Neither center nor scale is empirically justified for finance. | Check analytic score `-(theta-mu)/16`; use prior predictive and declared alternative-scale/center sensitivity if a modeling claim is sought. | Score failure is implementation error; strong sensitivity limits inference. Do not change the prior to make the sampler pass. |
| T05 - Other model coefficients: .025*sin(.371*i) | Fixed deterministic convenience fixture; neither coefficient is mathematically selected. | Reconstruct full vector and overrides exactly; inspect transition Jacobians, saturation and generated scales across declared parameter regions. | Degenerate or explosive fixture behavior identifies a model-design limitation. Correct sampler output does not cure it. |
| T06 - Initial raw SD coordinates: -.85+.011*i, i=0..59 | Fixed convenience range; M1 yields positive SDs about .355965..597789. Data start uses initial mean only. | Verify all transformed entries and inference initial covariance; inspect sensitivity to the initial law at T=30. | Index/transform error invalidates target; persistent initial-law dependence limits inference and generative-calibration claims. |
| T07 - Process raw SD coordinates: .55+.017*i, i=0..19 | Fixed convenience range; SDs about 1.005592..1.222133. | Check noise is injected into intended latent coordinates with variance s^2; inspect process-versus-observation signal scales. | Misplaced noise is a bug; weak identifiability or overwhelming process noise is a model limitation. |
| T08 - Observation raw SD: -.2 | Fixed; softplus plus floor yields .5982388694, variance .3578897448. Exact raw value has no adequacy proof. | Reconcile simulation noise and UKF observation variance; inspect standardized innovation residuals and likelihood sensitivity. | Variance/SD confusion invalidates target; inconsistent residual behavior motivates model/UKF checks, not automatic noise inflation. |
| T09 - SD floor: 1e-4 | Fixed model transformation. Positivity is motivated; this coefficient is uncalibrated. | Use M1 derivatives and floor-to-SD ratios; inspect full target sensitivity in a separately named model comparison. | Floor dominance changes substantive noise assumptions. A small ratio alone cannot prove negligible posterior effect. |
| T10 - Synthetic data seeds: (20260719,1020), (20260719,2020) | Fixed reproducibility labels, not meaningful numerical magnitudes. | Reproduce observation bytes with pinned software and CPU construction; verify distinct process/observation/time streams. | Collision or device/version drift changes dataset. Fix identity; one path still cannot establish repeated-data calibration. |
| T11 - Parameter constraints: R^4 | Fixed unconstrained coordinates; Gaussian prior is proper. Finite numerical-domain failures are a separate issue. | Check transform/inverse and no unintended sign filtering in training or target. Count numerical invalidity over relevant regions. | Accidental truncation computes a different posterior. Numerical invalidity must not silently become a new prior support. |
| T12 - Region quantity: P(theta[2]>0) | Fixed question about observation weight; index 2 is a naming convention. Sign alone does not identify modes. | Verify named-coordinate indicator, inspect reference mass and topology in both regions; assess its own MCSE. | Missed substantial mass vetoes posterior claim; negligible mass challenges the crossing rule, not automatically the sampler. |
| T13 - Bridge: prior+beta*log L, beta in [0,1] | Fixed method; endpoint identities follow from M1, assuming finite normalizers. | Check beta-zero density/score and beta-one equality; use identical full observations and beta-independent prior. | Tempered prior or time-averaged likelihood changes target. Repair scaling before any learning comparison. |

## N. Numerical implementation, tolerances and hardware

| ID / parameter and value | Basis and present status | Smallest useful check and decision | Failure meaning and next action |
| --- | --- | --- | --- |
| N01 - Runtime arrays and derivatives: TensorFlow/TFP, float64 | Backend policy plus inherited q20 precision choice. Spacing at one is epsilon=2^-52; round-to-nearest unit roundoff is 2^-53. Float64 is not proof of accuracy. | Check actual dtypes, batch/scalar value-score parity and M2 conditioning/residuals on the same target. | Mixed or inadequate precision invalidates claimed parity; repair or explicitly evaluate another numerical scope. |
| N02 - Training and sampling device: one GPU/worker, growth, XLA | Owner/resource requirements, not scientific constants. Whole-graph execution and batching remain requirements. | Verify memory policy before initialization, stable signatures, device placement and actual complete consumer graph; compare a qualified reference. | Allocation, scalar fallback or compilation mismatch is implementation/resource failure. It says nothing about NeuTra quality. |
| N03 - TF32 flag: true | Inherited launcher setting; float64 arithmetic is unaffected by TF32 permission. | Trace actual operand dtypes; separately compare any float32 matmul subpath with TF32 off on identical inputs. | Unexpected lower-precision path invalidates a float64 claim. Flag alone cannot demonstrate performance or error. |
| N04 - UKF rule: alpha=1, beta=2, kappa=0; 121 points | M1 derives weights and moment reproduction. Exact choices are fixed approximation hypotheses, not posterior-accuracy evidence. | Verify linear/quadratic fixtures and transformed moments against independent Gaussian integration at relevant covariances. | Failed identity is a bug; nonlinear moment error is approximation error. Changing rule changes the target and needs a distinct comparison. |
| N05 - Square-root backend: tensorflow_eigh_strict | Limited earlier feasibility/parity support; cached-factor alternative has a separate scope. | Same covariance/value/score/branch bank and M2 residuals, then matched map/starts for timing. | Backend differences may be numerical or call-chain differences. Localize before attributing sampler failure to geometry. |
| N06 - Explicit placement floor / added jitter: 0 / 0 | Fixed inputs; internal shifts remain nonzero. No proof that zero extra protection is sufficient. | Record realized shifts and smallest eigenvalues; quantify perturbation versus solve/score errors with M2. | Inadequate conditioning triggers explicit numerical repair evaluation; silent jitter changes the finite target. |
| N07 - Innovation floor: 1e-12 | Inherited absolute scale, not calibrated. Under strict classification it can cause a shift even without a floor-count event. | Record actual innovation covariance, requested floor, rho and shifted factor; apply M2 likelihood bounds and full recursion parity. | Counting only clamped eigenvalues misses target alteration. If material, report unresolved numerical target accuracy. |
| N08 - Rank / spectral-gap / fixed-null / reconstruction arguments: 1e-12 / 1e-8 / 1e-10 / 1e-10 | Inherited branch-specific tolerances. Strict Sylvester derivative does not require eigenvalue-gap separation. | Identify which checks execute; normalize residuals and probe just above/below any active classification boundary. | Inactive controls provide no evidence. Threshold sensitivity signals unresolved rank/branch semantics; repair that route without borrowing another branch's certificate. |
| N09 - Strict roundoff scale rho: max(floor,1e-14) | Inherited numerical hypothesis; M2 derives shift effects, not 1e-14. Placement and innovation can have different rho. | Inspect normalized perturbation eta and value/score effects, including switch between rho and 2*rho. | Material bias or unresolved boundary derivative blocks score/target qualification; evaluate a defined repair with separate scope. |
| N10 - Permitted near-SPD repair: lambda_min>=-1e-14, max entry<=1e8 | Inherited scale-sensitive classifier; absolute negativity is not evidence of roundoff origin. | Measure backward error, covariance scale, units and upstream cancellation; test both decision boundaries without relaxing them. | A true indefinite covariance is model/implementation invalidity; a scale artifact is classifier inadequacy. Neither is cured by relabeling it roundoff. |
| N11 - Invalid target sentinel: -1e100 plus status | Coding convention; finiteness is deliberately insufficient. Exact magnitude has no posterior meaning. | Inject invalid status at accepted/proposed/integration points; assert every required consumer rejects or stops under its stated contract. | A swallowed status invalidates results. Frequent numerical rejection can impose unintended support and requires repair. |
| N12 - Eigensolver refinement: 8 sweeps, 472 rounds at n=60 | Limited measurement: 4 previously failed, 8 passed one residual check. Round count is 8*(60-1). | Reuse valid evidence only in its scope; test hard spectra and actual new batch/device matrices with independent residuals. | Failed residual is a solver issue; additional sweeps are a candidate repair, not evidence that the posterior is impossible. |
| N13 - Eigensystem residual and orthogonality limit: 64*n*eps=8.5265e-13 | Dimension/roundoff scaling is motivated; 64 is inherited slack without a universal theorem. | Report raw normalized residuals and sensitivity to conditioning; compare factor and derivative errors against M2, not just threshold passage. | A passed residual with inaccurate score exposes insufficient screening. Recalibrate using error consequence, not desired pass rate. |
| N14 - Value replay parity: 1e-10+1e-12*abs(reference) | Inherited regression rule. Absolute term handles near zero; relative term handles scale. Neither is a posterior error bound. | On fixed data compare full values and likelihood increments; inspect cancellation and error relative to acceptance-relevant log differences. | Failed replay blocks equivalence; shared bias may pass it, requiring independent reference. Do not enlarge tolerance to hide disagreement. |
| N15 - Score replay parity: 1e-9+1e-10*abs(reference) | Inherited regression rule. Coordinate units/conditioning matter. | Compare each component and directions, with independent FD error intervals from M2 and branch labels. | Pairwise agreement alone leaves common score error unresolved. Disagreement prompts derivative localization. |
| N16 - Finite-difference initial step: eps^(1/3)*max(4,abs(theta_i)) | M2 derives cube-root order under smoothness; factor 4 comes from prior SD, not a fitted optimum. | Run h/4,h/2,h,2*h,4*h on valid same-branch points; seek a resolved error envelope rather than one favorable match. | No stable scale means FD evidence unavailable, due to noise, curvature or branch change; choose an independent check. |
| N17 - Finite-difference error assessment: Richardson /3 plus value-error/h | Derived for second-order central differences with bounded value errors. Those assumptions need checking. | Confirm approximately quadratic truncation behavior before using /3; compare returned score to combined envelope. | If assumptions fail, no derivative verdict follows from nominal tolerance. If resolved mismatch persists, score claim is wrong. |
| N18 - Same-map checkpoint tensors: exact restore | Serialization identity is an engineering invariant, not an approximation. | Compare dtype, shape and all stored values, optimizer slots/iteration and RNG continuation; then replay map/value separately. | Exact tensors with different behavior reveal configuration/source drift. Missing optimizer state invalidates same-training continuation. |
| N19 - Map inverse / logdet screen: abs error<=1e-8, condition<=1e8 | Inherited hypotheses. Conditioning can amplify errors; bounded diagonal scales do not bound shear. | Report absolute and scale-normalized round trips, logdet cancellation, Jacobian singular values and score parity at independent points. | Failure means unusable map algebra or ill-conditioning; a pass alone leaves tail coverage and whitening unproved. |
| N20 - Stress points: signed axes radii 2,4,6 | Inherited multiples of sqrt(d)=2. For N(0,I4), P(radius>r)=exp(-r^2/2)*(1+r^2/2). | These tail probabilities are about .4060, .003019, 2.894e-7; add random directions, cross-chart and independent posterior points. | Axes miss oblique shears and posterior tails. More axis points alone cannot justify global reliability. |
| N21 - GPU memory budget: 4 GiB pilot | Inherited resource hypothesis; no mathematical reason this is available or adequate. Growth is not a hard cap. | Measure complete-graph peak and compilation/transient memory under actual co-tenancy before cohort allocation. | Resource failure rejects that batch/topology; it is not training failure. Use an explicit allowed memory limit if a guarantee is required. |
| N22 - CPU sample-generation workers: powers of two, one thread initially | Proposed throughput search avoiding oversubscription; exact sequence is convenience. | Bound by affinity/memory, time repeated identical workloads, verify schedule-independent streams; choose smallest statistically unresolved throughput plateau. | Scaling stalls indicate overhead/bandwidth or threading. Sparse noisy timings cannot establish an optimum. |
| N23 - HMC chain parallelism: serial independent chains | Inspected binding default, not a claim that serial is fastest. | Price actual topology; verify chain independence and compare any proposed batched binding's values, states and resource costs. | Assuming training batching implies sampling batching gives wrong budget. Repair binding/cost forecast before launch. |

## A. Training architecture, optimizer and allocation

For this section a successful development contrast means valid maps and
material learning or downstream progress under its declared uncertainty and
cost allowance. It does not mean the selected architecture is globally optimal.
Only the plain beta-one grid has twelve initial histories; multiplying charts,
temperatures or batches multiplies the required work. Training bank size is
distinct from the number of optimizer updates.

| ID / parameter and value | Basis and present status | Smallest useful check and decision | Failure meaning and next action |
| --- | --- | --- | --- |
| A01 - Prior map: theta=mu+4*z | Derived exact Gaussian prior transport, up to fixed orthogonal permutations in initialized stages. | Check covariance 16*I, logdet 4*log(4), beta-zero transformed score -z and actual stage composition. | Any resolved identity failure is wiring/map error, not insufficient fitting. |
| A02 - Beta-zero optimizer updates: 0 | Derived from analytic prior endpoint, provided A01 passes. | Compare initial transformed density/score to N(0,I4) before positive-beta learning. | A failed check triggers implementation repair; adding updates would conceal it. |
| A03 - Plain NeuTra target: beta=1 | Scientific comparator requirement; M3 gives its objective. | Confirm full likelihood coefficient one and positive-temperature learning history through map export. | A canary map or partly tempered fit does not test properly trained plain NeuTra. |
| A04 - Initial architecture/LR search: widths (16,16)/(32,32), LR 5e-4/1e-3, two stages | Four inherited warm-start hypotheses. No theorem relates these exact widths or rates to q20 posterior complexity. | With paired roots/banks, compare learning per cost, numerical health, saturation and later independent pullback geometry. | Persistent learning indicates undertraining; plateau with capacity-response indicates restricted family; LR-response indicates optimization failure. Do not conflate them. |
| A05 - Activation / ordering: tanh / full reverse | Inherited architecture. Tanh derivative is 1-tanh^2(a); reversal changes which coordinates can condition later stages. | Inspect derivative/saturation distributions and cross-coordinate dependence by stage; test a declared ordering/activation contrast only if implicated. | Saturation may suppress gradient; poor ordering may limit correlations. An arbitrary activation switch is not evidence of a repaired posterior. |
| A06 - Scale-log cap: 2; extra linear scale paths off | Bounded diagonal exp(s) in [exp(-2),exp(2)] motivates protection; 2 is uncalibrated. Composed shears remain unconstrained. | Record scale outputs, their sensitivities, full Jacobian singular values and inverse errors; compare cap 3 only if saturation is implicated. | Saturated scale can limit fit; larger cap can worsen conditioning. Non-saturation does not establish adequate capacity or geometry. |
| A07 - Hidden weight initialization: Normal SD .02; output weights/biases 0 | Inherited initialization; zero final outputs protect prior initialization. Exact .02 is unsupported for this network's fan-in. | Verify same initial affine law across roots; record activation/gradient scales and when seed-dependent maps actually separate. | Identical initial maps are expected, not independent coverage. Vanishing updates or seed collapse requires initialization/optimization diagnosis. |
| A08 - Adam: beta1=.9, beta2=.999, epsilon=1e-7 | Inherited choices; M3 gives memory scales and limitations. Epsilon has units relative to gradient second moments. | Log actual moment/denominator scales, update norm relative to weights, and bias-correction iteration; compare a targeted alternative when dominated/stale. | Epsilon dominance suppresses learning; stale moments or lost slots can mimic plateau. No fixed update count follows from .999. |
| A09 - Gradient clipping: norm 10; alert if >10% updates clipped | Both constants inherited/proposed hypotheses respectively. Clipping bounds raw update input but biases stochastic gradient direction after expectation/Adam. | Preserve raw norm, clipping factor and resulting update; examine whether clipped events coincide with invalid scores or useful rare gradients. | Numerical pathology first requires target repair. Persistent clipping can justify 10-vs-30 development contrast; 10% alone neither vetoes nor proves harm. |
| A10 - Extra penalties / dropout: none, coefficients 0 | Method definition preserving M3 objective, not an assertion that regularization never helps. Stochastic dropout would alter the map unless explicitly handled. | Check implemented objective terms and deterministic frozen-map replay; assess instability/capacity via existing evidence first. | A hidden penalty changes the method. Proposed regularization must state its new objective and downstream bias/variance implications. |
| A11 - Batch-size candidates: 8,32,128, initially 32 | Exact sizes are hypotheses; independent gradient variance decreases as 1/B under M3. Hardware cost need not scale linearly. | Price actual native graphs; compare common roots at fixed measured update-work budget with uncertainty and all overhead reported. | A faster update can learn less per second. High noise, memory exhaustion or scalar fallback rejects the corresponding proposal for distinct reasons. |
| A12 - Cost measurement: one compilation, eight valid timed updates at beta=1 | Proposed short pricing pilot; 8 is not a statistically reliable tail estimate or training qualification. | Include warm caches, compilation, synchronization, validation/checkpoint overhead and each extra beta's cost. Check timing dispersion and forecast adequacy. | Variation or different branch costs makes reservation uncertain; extend/revise bounded pricing, not treat an optimistic mean as a guarantee. |
| A13 - Initial training seeds: three roots/config, twelve histories | Proposed bounded replication, not a robust seed-success estimate. Common roots pair configurations. | Preserve every trajectory including failure; compare between-root uncertainty separately from validation-row uncertainty. | A seed-sensitive result remains a candidate hypothesis. Three successful roots still provide weak reliability evidence. |
| A14 - Cumulative positive-beta training rungs: 128,512,2048,8192 | Proposed factor-four observation schedule; no convergence theorem specifies these counts. Every viable initial history is priced through 512. | Apply M4 progress decisions, optimizer telemetry and matched capacity/optimizer contrasts; charge cumulative increments once. | Progress at cap means undertraining or insufficient allocation. Early plateau can be capacity, optimization, coverage or validation resolution failure. |
| A15 - Final training count: selected funded checkpoint, unknown beforehand | Measured selection output, not a preset production label. Two plateau comparisons usually need reaching at least 2048. | Require learning history, resolved development assessment and map checks, then actual downstream sampling. | Neither reaching 8192 nor two flat comparisons proves adequate training. Record unresolved reason and repair or under-budgeted status. |
| A16 - LR schedule: initially constant, paired half-LR continuation | Proposed one-factor diagnostic. Local stability scale in M3 motivates testing rate sensitivity, not the exact factor .5. | Continue same checkpoint/slots on control and half-rate branches with matched evaluation rows and priced update/time comparison. | Half-rate improvement suggests optimization sensitivity; unchanged poor fit shifts attention to capacity or missing mass. Do not claim causal repair from one noisy root. |
| A17 - Conditional capacity repairs: widths (64,64)/two stages, or (32,32)/four stages | Proposed width-vs-composition contrasts; doubling is convenient, not derived. | Compare one changed factor, paired roots and equal declared cost; inspect heldout learning and downstream geometry. | Wider/deeper success implicates capacity within tested range. Failure may still be optimization/budget limitation rather than impossibility of flows. |
| A18 - Conditional optimizer/cap repairs: LR/2, clip 10/30, cap 2/3, beta2 .999/.99 | Proposed targeted alternatives. M3 explains altered step/memory scales; exact replacement values remain uncalibrated. | Choose contrast from observed failure, freeze it before evaluation and preserve parent. Price revalidation/tuning for changed maps. | An improvement only in clipping rate or loss is explanatory. Repair is useful only if its intended numerical/learning problem resolves and downstream remains valid. |
| A19 - Adam across temperatures: carry, one carry-vs-reset contrast if stalled | Proposed continuity choice. Objective changes by delta_beta*log L, so old moments may help or become stale. | Compare first gradient/moment alignment after beta change; test reset/carry from identical weights and independently paired streams. | Stale moment mismatch is optimizer-transfer failure. Within-beta loss of state remains a checkpoint bug, not this method contrast. |
| A20 - Save cadence: 128 updates plus decisions/boundaries/exit | Proposed recovery interval, not numerical science. Exact 128 comes from first rung. | Measure save cost and lost work in resume fixture; serialize weights, slots, iteration, beta and RNG counters. | Unreproducible continuation is an engineering veto. Excess overhead or exposure to lost work calls for a measured cadence adjustment. |

### A-batch. Precise batch comparison and optimizer caveats

The proposed common update-work allowance is `512*c_update(B=32)` seconds at
the same architecture/beta, with per-arm counts approximately
`floor(512*c32/cB)` frozen after pricing. It equalizes priced update work;
different compilation/validation cost means it does **not** equalize total
campaign cost. Report both, and use total cost for an efficiency claim. If
B=32 is infeasible, explicitly price and designate B=8 as the reference before
deriving counts; the unusable c32 cannot define a budget. Reuse the three B=32
compact-high histories only when they exactly match the grid's source, seeds,
optimizer, endpoint and evaluation scope. Up to nine batch histories otherwise
add work beyond the twelve-history grid. A different selected batch creates a
different training selection scope.

An alternative diagnostic for the uncalibrated LR scale is the local loss
change along the actual proposed optimizer direction v: estimate
`g^T v` and `v^T H_loss v`, and compare predicted
`eta*g^T v + eta^2*v^T H_loss*v/2` with an independent-batch loss change.
These are noisy/local quantities, so no newly invented cutoff is required.
Large disagreement calls the local approximation into question. This explains
what a step-size diagnostic would test without claiming it is already wired
or prescribing a Hessian computation more expensive than training itself.

## V. Training validation and map qualification

All objective intervals used for adaptive development below inherit M4's
selection warning. The .04-nat rule is optimization resolution, not a permitted
posterior error or a universal criterion for sufficient training.

| ID / parameter and value | Basis and present status | Smallest useful check and decision | Failure meaning and next action |
| --- | --- | --- | --- |
| V01 - Initial objective-validation rows: 3*256=768 | Inherited Monte Carlo bank construction. Exact 256/three-bank split is not calibrated. | Estimate paired variance/tails by bank; use M4 to price resolution. Validate stream independence and preserve shared rows for pairing. | Unstable variance makes nominal intervals unreliable; a large bank cannot repair an objective that misses mass. |
| V02 - Bank expansion: 768,3072,12288 | Proposed factor-four schedule; standard error halves under fixed-comparison IID assumptions. | Calculate required M from observed variance and h; extend within budget only if it can discriminate the decision. | Required rows beyond cap means unresolved comparison. It does not establish a plateau or justify selecting by point loss. |
| V03 - Validation cadence: start, rungs, before/after repair | Proposed balance of observation cost and decision opportunities. | Verify identical beta/objective and preserved checkpoint inputs; track every look and whether it influenced later fitting/selection. | Missing intermediate states prevent explaining failure; repeated adaptive looks remove naïve fixed-pair confidence interpretation. |
| V04 - Material objective change: delta=.01*d=.04 nat/draw | Proposed decision resolution; d-scaling is convenience, not evidence that .01 nat/dimension controls posterior error. | Price h=.02 using M4 and test whether this loss scale predicts useful downstream geometry/cost in development. | Unaffordable resolution or no downstream relationship invalidates this stopping rationale. Relabel/revise it before confirmation. |
| V05 - Paired objective uncertainty: mean(I)+/-1.96*s/sqrt(M) | Conditional CLT calculation; 95% is a design convention. Fixed banks reused adaptively are selection data. | Inspect bank disagreement, large contributions and variance stability; use fresh frozen-pair data for any nominal confidence claim. | Heavy tails or selection invalidates that interval's interpretation. Numerical finiteness alone cannot supply valid uncertainty. |
| V06 - Continue learning: lower endpoint>delta | Proposed development extension rule, conditional on meaningful M4 resolution and budget. | Keep training when material progress is resolved; when overlap remains, price more evidence or report inconclusive. | Failure to meet strict lower bound is not proof of no progress. Avoid premature stopping due to noisy validation. |
| V07 - Plateau candidate: within +/-delta for two successive comparisons | Proposed persistence screen, usually 128->512 and 512->2048. Overlapping evidence is not independent confirmation. | Check optimization health, same objective, capacity response and independent posterior coverage before interpreting plateau. | Flat loss can mean undercapacity, clipping/moments, missed modes or bank overfitting. Diagnose each before saying training is sufficient. |
| V08 - Deterioration: upper endpoint<-delta | Proposed development preservation rule, with M4 caveats. | Preserve earlier checkpoint and compare targeted LR/optimizer repair on declared independent/paired evidence. | Deterioration may be unstable optimization or noisy selection; it does not refute the transport family. |
| V09 - Seed viability: at least 2/3 roots improve and pass numerical checks | Inherited nomination concept. Exact two-thirds threshold is not calibrated reliability. | Define improvement consistently with V04-V05 and preserve the third outcome; report finite-sample uncertainty rather than success-rate certainty. | One or more failing seeds indicates sensitivity to initialization/optimization. Do not hide failures by selecting the one attractive map. |
| V10 - Cheap generated-map occupancy bank: 4096 | Inherited count; worst IID proportion SE is .0078125 under the proposal law. | Measure proposal sign mass and uncertainty, compare with independent posterior region evidence and log weights where resolved. | Proposal occupancy differs from posterior mass. A balanced proposal can still miss modes or generate low-density space. |
| V11 - Actual whitening: independent posterior pullbacks, no universal threshold | Exact transport would give N(0,I4); this is a target-specific empirical question. | Estimate pullback mean/covariance/eigenvalues, radial/tail behavior, cross-region coverage and score residual `grad_z log pi_T(z)+z`, with dependence-aware uncertainty. | Poor covariance suggests linear geometry; normal covariance with tail/score mismatch suggests non-Gaussian geometry. Bad reference makes the check unavailable, not passed. |

For V11, a useful descriptive covariance discrepancy is
`max_i abs(log(lambda_i(Cov(z))))`, meaningful only when the covariance estimate
is reliable and positive definite. Its acceptable magnitude must be calibrated
against downstream travel and total cost; no value is silently supplied here.
Mean/covariance checks alone miss a multimodal distribution with the same
moments as a Gaussian. For three successful independent training roots, even
an idealized two-sided 95% exact binomial lower bound on success probability is
only `.025^(1/3) approximately .2924`. The assumptions are stronger than a
post-selected architecture usually supports. This explains why three roots are
a limited exploration allocation, not reliability certification.

## E. Temperature ladders, charts and ensemble transitions

| ID / parameter and value | Basis and present status | Smallest useful check and decision | Failure meaning and next action |
| --- | --- | --- | --- |
| E01 - Single-map comparator: K=1, beta=1 | Scientific baseline requirement. M3 establishes objective and M6 kernel checks. | Price fully trained, independently tuned and posterior-assessed map; include training failures/cost. | An inadequately trained comparator makes ensemble advantage uninterpretable. |
| E02 - Initial ensemble: K=2 | Smallest nontrivial multi-chart hypothesis; historical nomination does not qualify rebuilt training. | Measure chart diversity on independent posterior points and each chart's kernel validity; compare cold-stream results at total cost. | Duplicate or undertrained maps do not test chart diversity. Two charts may still miss common mass. |
| E03 - Chart-count repair: K=4 | Inherited bounded doubling; no theorem prescribes four charts. | Fund four real training histories, examine complementary coverage and equal-accuracy total cost versus K=2. | More charts without diversity adds cost; poor map fit remains a training problem rather than proof that ensembles fail. |
| E04 - Initial bridge ladder: (0,.5,1) | Historical convenience hypothesis, no established overlap. M5 gives spacing scale. | Estimate log-likelihood spread, adjacent overlap, passage times and cold-chain region behavior using viable development samples. | A bottleneck can reflect spacing, map quality or poor within-slot movement; separate these before repair. |
| E05 - Ladder repair: (0,.25,.5,.75,1) | Inherited interval halving. M5 predicts local divergence roughly quarters only if variance/regularity remain comparable. | Reprice/train/tune all new scopes; compare valid cold estimates and total cost, not just swap rates. | High swap acceptance with slow travel is not repaired sampling. Additional replicas may cost more than they help. |
| E06 - Branching contrast: continuation versus half-chart restart at beta=.5 | Existing method hypothesis; half and .5 are convenient allocation/location choices. | Compare actual independent restarted learning against continuation with same total funding and final coverage criteria. | Identical affine initialization or too few updates does not test restart diversity. Failure may be inadequate training allocation. |
| E07 - Additional temperatures: bisect diagnosed interval in development only | Proposed localized repair motivated by M5; final number unknown. | Identify poorly connected interval using overlap and travel, then reserve full new training/tuning scope and freeze before retained sampling. | Unfundable bisection means under-budgeted development. Adaptive unrecorded temperatures would change the evaluated algorithm. |
| E08 - Chart selection probabilities gamma: uniform 1/K (.5 or .25) | M5 proves invariant mixture if each kernel is valid and weights are state-independent. Uniformity is an uncalibrated efficiency choice. | Verify nonnegativity, sum one, state independence, chart inverse/target factors and empirical selection counts. | Incorrect weighting can invalidate kernel. Valid but inefficient weights are an optimization question, not posterior bias by themselves. |
| E09 - Density-mixture alpha: uniform for diagnostic density, train-alpha off | Separate mixture-density definition; initial method excludes joint-mixture learning. Exact uniformity is convenience. | Check normalized component densities and alpha sum; trace that diagnostic alpha does not silently control gamma or training loss. | Conflation changes transition/objective. Any learned alpha needs its own loss and invariant-transition treatment. |
| E10 - Within/swap cadence: one HMC update per slot, alternating adjacent parity, initial 0 | Composition of invariant kernels preserves invariant distribution; deterministic alternation need not make the whole schedule reversible. Exact cadence is an efficiency hypothesis. | Check both parities, chart Jacobians and time indexing; measure slot travel and swap cost before testing a changed cadence. | Swap bookkeeping bug invalidates ensemble; excessive/insufficient swaps is an efficiency issue requiring a priced contrast. |
| E11 - Replica systems: four; three/five slots, hence 12/20 physical states | Count is derived; only four cold streams are independent across systems. Chart count does not multiply chains. | Preserve system/temperature/identity indices and separate seeds; assess only beta-one states for the posterior. | Treating hot replicas or multiple charts as independent posterior chains understates uncertainty and is wrong. |
| E12 - Region movement screen: both signs, >=4 cold crossings/chain | Inherited observability rule; exact four has no calibrated mixing or probability guarantee. | Compare reference region mass and event autocorrelation/MCSE; inspect dwell times and all-chain crossings. | Few crossings in substantial regions blocks promotion and motivates travel repair. A truly negligible region challenges the criterion; revise scope explicitly, not by hidden waiver. |
| E13 - Replica travel screen: >=16 aggregate round trips/orientation, >=1/identity | Inherited operational hypothesis. Dependent trips do not have justified Poisson SE 1/sqrt(16). | Record travel-time distribution, stuck identities and dependence; relate hot travel to cold-region exploration and accuracy. | Passing trip count can coexist with cold nonmixing. Failure localizes mobility; it is not by itself a rejection of the research direction. |

## H. HMC preparation, candidate search and starts

The [public HMC interface](/home/ubuntu/python/BayesFilter/docs/reference/hmc-tuning-interface.md)
and its [capability registry](/home/ubuntu/python/BayesFilter/bayesfilter/inference/tuning_contract.py)
were inspected. Use `tune_hmc_kernel` for the supported ordinary target/metric
path and `tune_fixed_transport_hmc_kernel` for a supported frozen nonlinear
transport. Kernel mechanics qualification and posterior equilibration are
separate. Do not route a residual NeuTra metric through an unrelated tuner or
describe q20 checkpoint export as completed before its actual binding is tested.

| ID / parameter and value | Basis and present status | Smallest useful check and decision | Failure meaning and next action |
| --- | --- | --- | --- |
| H01 - Kernel: fixed-trajectory Metropolis HMC, fresh Gaussian momentum | Method with invariance conditions in M6; exact transformed target includes logdet. | Verify full force/value agreement, reversibility/volume-preserving implementation and endpoint Metropolis calculation in supported route. Freeze map/metric/epsilon/L in retained sampling. | Missing Jacobian or incorrect integration/acceptance invalidates method. A valid but inefficient kernel remains a tuning candidate. |
| H02 - NeuTra mass: I4 in z | Current public-interface constraint, not proof of whitening. | Inspect independent pullback covariance and directional frequencies; verify supported checkpoint codec and latent score. | Residual anisotropy can force small steps. Repair transport or explicitly develop a supported method extension; do not silently claim mass adaptation exists. |
| H03 - Classical naive comparator: identity physical mass | Deliberately simple baseline, not the strongest classical method. | Confirm coordinate units and matched starting bank; report valid accuracy and complete cost. | Failure of this arm cannot establish NeuTra superiority over tuned classical HMC. |
| H04 - Tuned classical comparator: serious preparation, operational metric update required | Proposed supported baseline protocol; successful empirical update must actually execute. | Inspect serialized windows, covariance and both affine layers; verify downstream target/score and complete preparation cost. | Unchanged incumbent or missing affine layer is not a newly learned metric. Fix wiring or report unavailable tuned baseline. |
| H05 - Initial classical geometry: mu and scales (4,4,4,4), covariance 16*I | Prior-derived warm start. M6 prevents covariance/mass convention confusion. | Check issued coordinate transform, prior standardization and score chain rule; record any independent center/covariance's source. | Wrong convention or mismatched center gives wrong scale/derivative. Prior geometry may simply be poor for likelihood-dominated posterior. |
| H06 - Initial ordinary metric budget: 1000 transitions | Derived from current d=4 policy floor; exact minimum is inherited and uncalibrated for q20. | Inspect actual metric changes and covariance information using G rows, then separately test equilibration. | 1000 with trapped chains estimates local geometry. More adaptation may help, but count alone cannot establish sufficient metric or burn-in. |
| H07 - Tuning chain count: exactly 4 | Active acceptance implementation requirement; useful replication but not a sufficiency theorem. | Confirm distinct streams, all-chain telemetry and broad physical starts; apply actual four-chain evidence formula. | Duplicated streams or pooled-only health invalidates evidence. Four chains can all miss the same region. |
| H08 - Physical starting-bank proposal: two starts/sign, max 128 proposals in batches 32 | Proposed dispersion and bounded initialization cost. Without validity rejection, positive-sign probability is Phi(.65/4), about .5645. | Inspect prior draws in frozen order, archive invalid/rejected starts, verify both regions' reference relevance. Estimate sign-specific valid-start probability. | Under IID valid-start probability p, fewer than two after 128 has probability (1-p)^128+128*p*(1-p)^127. Numerical rejection or small p can make the cap unsuitable; report why, do not condition training/posterior. |
| H09 - Matching starts across methods: same physical bank before preparation | Fair-comparison requirement; equal physical states are meaningful despite different latent coordinates. | Apply map inverses and check physical replay; record preparation endpoints and cost for every arm. | Unequal favorable starts confound method cost. Common starts induce pairing, not independence between method estimates. |
| H10 - Confirmation starts: fresh bank and fresh checked binding/verification | Statistical separation plus interface requirement. Existing retained runner continues verified endpoints. | Verify selected settings can be bound and freshly verified on new starts without retuning from confirmation outcomes. | Arbitrary state reset bypasses supported provenance. If new verification fails, preserve failed confirmation and return to development. |
| H11 - Initial L grid: (3,5,9,13,18,25) | Inherited broad discrete hypotheses; no q20 optimality derivation. | Tune epsilon independently for each L; inspect duration/recurrence and downstream travel with full cost. | A viable short trajectory may move too little. Grid failure or max 25 limitation cannot exclude viable longer classical trajectories. |
| H12 - Initial epsilon: min(.1,.8*2/sqrt(lambda_max)) when curvature resolved | M6 derives 2/sqrt(lambda); .8 and .1 are unqualified seed constants. Lambda is measured in actual mass coordinates. | Use development-bank curvature, record negative/unresolved modes and observed locations, then inspect actual fixed-pair trajectory health. | Local curvature misses remote stiff regions; small epsilon may be safe but immobile. Ordinary preparation supplies its own measured seed. |
| H13 - Epsilon repair: factor 2, <=5 directional repairs; domain min/64 to 64*max | Existing bounded search; domain 64=2^(5+1) is dispatcher arithmetic, not a stability theorem. Explicit upper bound replaces upper endpoint. | Check complete interval and every child's fresh evidence; reserve stages before launch and record whether domain edge was hit. | Hitting edge means incomplete search under that bound, not no valid HMC kernel. Budget/schema changes require a new declared search. |
| H14 - Epsilon refinement: .8 and 1.25, one round | Inherited reciprocal local grid; exact spacing/round count uncalibrated. | Independently verify all surviving refined pairs; inspect sensitivity of mechanics and travel around coarse candidate. | Coarse-grid misses are tuning resolution limitations. No ranking follows from acceptance closest to .70. |
| H15 - L refinement: integer midpoints (4,7,11,15,21) | Derived floor midpoints of initial grid; choice to use floor/one refinement is policy convenience. | Check inclusion, uniqueness and funded independent evidence at each new L. | A missing midpoint stage leaves declared search incomplete; finite grid still does not prove optimum. |
| H16 - Longer travel: conditional L=50/100 | Proposed geometric extension when small epsilon makes travel short. Max 25 ordinary convenience config does not implement this override. | First verify a supported explicit binding and price duration/recurrence/integration cost; compare matched posterior accuracy. | Unsupported route is integration work; unaffordable travel is cost limitation. Neither warrants calling max 25 the best classical baseline. |
| H17 - Acceptance target: .70; practical [.65,.75], repair [.55,.85] | Shared operational heuristics; no general optimality/coverage theorem for these exact values. | Use full evidence predicate: interval overlaps practical band, lies within repair band, each chain in repair band, plus health/conflict/path checks. | Failure informs directional repair or more evidence. Passage establishes mechanics compatibility, not stationarity or efficiency. |
| H18 - Acceptance evidence: 4 chains*4 blocks*>=16 decisions, 90% interval | Implementation minimum 64/chain; t critical 2.3533634348 is derived for df 3 two-sided 90%. Valid coverage needs independent, suitably distributed chain summaries. | Inspect chain/time conflicts and effective information; verify actual interval arithmetic, not just 256 pooled decisions. | Short dependent blocks or nonstationary chains weaken confidence interpretation. Repeated looks are operational, not anytime-valid testing. |
| H19 - Pilot / measurement / fresh verification: 64 each, +32 startup discard each | Proposed minimum evidence allocation. 32 is startup handling, not equilibration. | Preserve separate stages/seeds and actual evidence counts; verify pilot cannot grant membership. | Using pilot or insufficient fresh decisions invalidates qualification. Passing verification still requires subsequent posterior warmup. |
| H20 - Inconclusive evidence rungs: 64/128/256 | Existing fourfold bounded information extension. Standard error need not follow IID 1/sqrt(n) under dependence. | Extend only declared unresolved evidence, preserving all looks and total work; report terminal interval and disposition. | Unresolved at 256 is inconclusive, not a pass from point acceptance or evidence that no pair exists. |
| H21 - Minimum movement / max repeats: .05/.95 per chain | Complementary inherited operational screens, not mixing-time estimates. | Count actual all-chain movement and exact repeat definition; separately inspect displacement and region travel. | High acceptance can coexist with tiny movement. Failure rejects this mechanics candidate; meeting 5% movement does not establish exploration. |
| H22 - Normalized return displacement floor: 1e-4 | Inherited units-dependent pathology threshold. Exact coefficient uncalibrated. | Inspect normalization from actual binding and compare displacement distributions to coordinate/prior/posterior scales. | Poor scaling can create false reassurance or false veto. Resolve units and tested sensitivity before changing threshold. |
| H23 - Path recurrence: lags 2..16, fraction .95, atol 1e-12/rtol 1e-10 | Inherited finite-lag recurrence screen. M6 explains periodic risk, not these cutoffs. | Compare observed near-returns to coordinate-scale rounding and expected trajectory periods; test known constant/periodic mechanics fixtures. | Matching tolerances can miss near-cycles or flag tiny-scale states. A pass excludes only the tested pathology, not general slow mixing. |
| H24 - Finite energy magnitude: 1000; legacy log-accept alert -1000 | Historical explanatory fields. exp(-1000) is effectively zero acceptance, but finite positive/negative energy errors have different meanings. | Preserve signed energy distributions, reversibility and target-status evidence; retain only existing native divergence/nonfinite vetoes. | Extreme finite values explain rejection/instability; do not invent a new universal 1000 veto or treat their absence as convergence. |
| H25 - Status cadence: per_chain_step, accepted/proposed and promised integration telemetry | Scientific numerical-health requirement; sentinel check in N11 is essential. | Inject status failure at each promised location and test connected consumer; record actual capability, not callback name alone. | Missing interior evidence leaves its claim unproved. A finite endpoint cannot certify all leapfrog states. |
| H26 - Tuning chunk size: 64, conditional 256 | Proposed checkpoint/deadline granularity, not total evidence count. | Measure native-call duration and peak memory, include compilation and test resume with no duplicated/lost decisions. | Overlong nonpreemptible calls violate forecast; change future chunking within supported scope or add explicit process bound. |
| H27 - Candidate/attempt/gradient caps: derived from funded search | Budget requirement; no inherited 100-candidate/100-unit cap receives default authority. | Enumerate scopes, L, stages, refinements and repairs; reserve mandatory remaining work for each admitted candidate using B accounting. | Unfundable queue means incomplete search. Charge failed calls and preserve verified members without claiming full coverage. |

The old failed-point curvature `33319.68301` implies the **local quadratic**
scale `.0109567`; old epsilon `.0275` is about 2.51 times that scale. These are
historical explanatory numbers, not fresh estimates for a trained map. The
underlying curvature/point provenance is required to reuse even that explanation;
the derivation does not transfer its tuning result. New map, temperature or
metric means a new curvature and fixed-pair evidence scope.

## G. Hidden ordinary metric and adaptation choices

If A is the actual issued affine scale, assess whitening using
`A^(-1)*Cov(theta)*A^(-T)` and the corresponding transformed score, not a field
named `mass` alone. For independent Gaussian samples, a variance estimate has
relative standard error approximately `sqrt(2/(n-1))`; covariance-moment
autocorrelation, heavy tails and unknown means complicate replacing n by an
ESS. With only eight effective squared deviations, even this idealized scale
is about .535. Thus the preparation's small ESS floors can check that some
information exists but cannot certify a precise covariance matrix.

| ID / parameter and value | Basis and present status | Smallest useful check and decision | Failure meaning and next action |
| --- | --- | --- | --- |
| G01 - Geometry budget: clip(ceil 20*d*m,1000,5000), m in[1,4]; legacy max 10000 | Inherited forecasting heuristic. At d=4, inner count<=320, hence initial 1000 regardless of m. | Inspect realized first-budget dispatch and adaptation evidence; price any additional supported attempt separately. | Heuristic pressure cannot enlarge d=4's initial budget. A quoted 10000 ladder does not prove it executes or succeeds. |
| G02 - Geometry-pressure coefficients: .25/.50/.05/.25, fallback 1.50 | Inherited weighted condition/anisotropy/clipping/nonpositive forecast. Exact weights lack calibration; d=4 floor dominates. | Vary diagnostic inputs without evaluating target to check dispatch sensitivity; compare predictions with measured preparation cost when available. | Insensitive floor or inaccurate forecast makes coefficients uninformative for adequacy. Do not interpret metadata as measured mass-learning difficulty. |
| G03 - Geometry initialization: scale .5, guard .8, jitter/eigenfloor 1e-9 | Inherited numerical hypotheses; no q20-derived coefficients. M2/M6 relate perturbation and curvature consequences. | Trace which initialization operations execute, their units, perturbation ratios, affine transform and score; inspect clipped directions. | Large regularization can conceal missing geometry. Wrong convention/flooring is numerical or preparation failure; declare repairs rather than quietly change target coordinates. |
| G04 - Ordinary bootstrap screen: clip(ceil 4*sqrt(d),32,1024), discard quarter | Inherited mechanics allocation; d=4 gives 32 and 8 discarded. Square-root dimensional scaling is a heuristic. | Check basic target/movement and that later candidate 64-decision minimum still executes. | Bootstrap failure localizes mechanics/preparation; bootstrap passage is neither kernel verification nor equilibration. |
| G05 - Windows for 1,000 steps: 100,200,400,200,100 | Derived constructor: 10% buffers, quarter slow span then doubling/truncation. Fractions are inherited schedule choices. | Inspect actual state, covariance and step-size updates at each boundary; assess covariance stability across independent evidence. | A precise window schedule can still learn trapped local geometry. Count and doubling do not prove sufficient adaptation. |
| G06 - Empirical metric shrinkage: .25 toward diagonal | Convex combination `.75*S+.25*diag(S)` preserves variances and damps correlations; exact .25 is uncalibrated bias/variance tradeoff. | Check SPD and transformed covariance, off-diagonal uncertainty and downstream movement against a separately priced shrinkage contrast if implicated. | Suppressed real correlations or noisy covariance both hurt geometry. Shrinkage preserves positive definiteness only under appropriate positive-variance inputs, not arbitrary invalid matrices. |
| G07 - Dense/diagonal minimum states: 64/32 at d=4 | Derived maxima from current 64 vs 4*d and 32 vs 2*ceil(log2(d+1)) screens. Sample covariance rank<=n-1 motivates enough states, not these exact minima. | Inspect covariance-of-moment information and rank/condition with dependence; distinguish many repeated rows from independent information. | Enough raw rows can still give singular/noisy metric. Repair exploration or gather information, rather than count duplicates as evidence. |
| G08 - Dense/diagonal ESS floor: 8/4 at d=4 | Derived from max(8,d+1)/max(4,ceil(log2(d+1))); exact floors are weak inherited screens. | Evaluate the statistic whose ESS is computed and covariance-element stability; compare to the Gaussian variance-error example above. | Parameter-mean ESS is not covariance ESS. Passing 8/4 cannot establish accurate dense/diagonal metric. |
| G09 - Dense/diagonal location check: split R-hat<=1.10/1.25 | Inherited preparation heuristics; these are not retained modern rank/folded criteria. | Inspect between-chain location/scale and actual formula; compare independent window covariance and reference-region evidence. | Common trapped chains can pass. A looser preparation screen cannot override failed retained diagnostics. |
| G10 - Dense condition/discrepancy: 1e8; dense .50, diagonal .75 | Inherited standardized conditioning and shrinkage-discrepancy screens; exact thresholds uncalibrated. | Read actual discrepancy normalization, report raw matrices/errors, evaluate coordinate rescaling and downstream numerical consequences. | False reassurance or rejection under rescaling signals bad calibration. Do not weaken threshold solely to obtain an issued metric. |
| G11 - Low-level window metadata: jitter 1e-6, floor 1e-9, rate .03, step range 1e-6..10 | Legacy config fields. Active empirical metric issues no absolute floor; actual step adaptation is dual averaging. | Trace realized consumer and serialize active settings, explicitly distinguishing metadata from executed operations. | Citing inactive controls as protection or tuning is unsupported. No calibration should be spent on a knob that does not execute. |
| G12 - Actual dual averaging: .05 shrinkage,10 pseudocount,.75 decay; target .70; shrink target min(10*eps,bound) | Inspected TFP defaults plus repository overrides. Stochastic-approximation averaging motivates decay between .5 and 1; it does not derive .75, .05 or 10 for q20. | Inspect adaptation curves, clipping at bounds and freeze stage; reset after metric changes, preserve exact state on interruption, verify resulting fixed pair independently. | Oscillation, saturation or stale state is adaptation failure. Stable acceptance alone does not establish a useful trajectory or posterior. |

## P. Posterior quantities, precision and reference agreement

The proposed .02 relative mean MCSE, .01 event MCSE and .05-SD quantile MCSE
are **decision requirements for this study**. They have consequences derived
in M7, but their practical adequacy has not been derived from a financial
decision. If a downstream decision is D(g), local error propagation gives
`Var(D(g_hat)) approximately grad(D)^T Cov(g_hat) grad(D)` when differentiable;
threshold decisions may instead require a bound on decision reversal. Such a
decision could justify tighter or looser tolerances. It cannot be inferred
from the optimizer's loss. No predictive horizon is added by this audit.

The R-hat and ESS source is Vehtari et al., *Rank-normalization, folding, and
localization*, locally preserved as [PDF](../../.localresources/papers/q20-pipeline-audit-20260915/rhat-1903.08008.pdf)
and [text](../../.localresources/papers/q20-pipeline-audit-20260915/rhat-1903.08008.txt).
The inspected basis is section 2, section 3 equations (1)-(13), section 4
equations (14)-(16) and its quantile discussion, and Appendix A's scale-mismatch
example. The authors explicitly describe 400 as chosen from practical
experience/simulations and say the application should determine adequate MCSE.
Their recommendation is strict R-hat<1.01; current owner/API policy uses <=1.01.
Neither is a proof of stationarity. No claim of complete original-author-code
equivalence is made here.

For split chains of length n, `Rhat=sqrt(Vplus/W)` with
`Vplus=(n-1)*W/n+B/n`. Rank normalization uses pooled average ranks for ties
and `Phi^(-1)((rank-3/8)/(S+1/4))`, S pooled draws. Folding applies to absolute
deviation from the pooled median before rank normalization. The fractions
3/8 and 1/4 are the cited normal-score approximation, not q20 tuning controls.
They and the square root belong to diagnostic correctness. Changing them to
obtain favorable numbers would compute a different statistic.

| ID / parameter and value | Basis and present status | Smallest useful check and decision | Failure meaning and next action |
| --- | --- | --- | --- |
| P01 - Means: all four, MCSE/SD<=.02 | Proposed accuracy requirement; M7 derives original-scale mean ESS>=2500 under CLT assumptions. | Identify each quantity by name and compute its actual mean-MCSE, finite moments, convergence and reference error. | Bulk ESS 400 does not satisfy this requirement. Missing moments or exploration invalidates the MCSE interpretation; otherwise extend funded retained evidence. |
| P02 - Quantiles: .025,.50,.975 each; MCSE/SD<=.05 | Proposed central 95% summaries and accuracy requirement. M7 explains local density and indicator ESS dependence. | Use actual quantile uncertainty, inspect ties/tails and reference-CDF brackets; compare .025/.975 directly. | Tail ESS at .05/.95 cannot substitute. Sparse tails or unstable quantile MCSE leaves interval endpoints unresolved. |
| P03 - Sign probability: MCSE<=.01 | Proposed one-percentage-point SE, not a one-point 95% confidence half-width. Worst ESS 2500 is derived. | Compute indicator-specific dependence/MCSE, visits and independent region-mass agreement. | Constant indicators may conceal missed mass. Valid but imprecise event estimates require more evidence or a better kernel. |
| P04 - Additional predictive quantities: none silently added | Scientific scope requirement. No mathematical choice of a useful horizon is possible without a prediction question. | If requested, specify time, conditioning, estimator, units and error propagation before adding computations. | An arbitrary horizon introduces a different task and unpriced accuracy requirement. Current posterior task is not predictive validation. |
| P05 - Independent posterior chains: 4 systems | Owner/interface requirement and source recommendation, not an optimal chain count. | Confirm independent systems and region-relevant dispersion; recompute between/within diagnostics and named functionals. | Four copies of one trajectory or hot slots counted as chains invalidates uncertainty. Four genuine chains can still share a missed mode. |
| P06 - Warmup: minimum 2000/window 1000/max 10000, Rhat<=1.05 | Owner operational policy. None of these numbers guarantees equilibration for q20. | Archive all chunks; evaluate current rank/folded statistic on latest window plus numerical/movement/reference-aware development evidence. | Failure at cap is unresolved equilibration. A passed window still cannot certify that an unvisited mode is absent. |
| P07 - Warmup check cadence: 1000, one passing window | Inherited operational allocation. Requiring more overlapping passes would not create independent evidence. | Inspect diagnostic path and comparison with independently initialized development runs; retain exact stopping history. | A transient pass is possible. Any changed persistence policy must be predeclared; do not retrospectively pick the favorable window. |
| P08 - Retained draws: 1000 then +1000 to 10000 per chain | Shared bounded precision schedule. M8 links cap to achievable ESS, not guaranteed sample quality. | Preserve cumulative prefixes; check each declared quantity and time/cost against remaining cap. | Cap exhaustion is insufficient evidence, not permission to loosen MCSE/R-hat or discard earlier unfavorable draws. |
| P09 - Retained R-hat: max rank-split/folded<=1.01 | Owner policy with empirical literature rationale. Exact cutoff is not a mathematical mixing bound. | Verify repaired formula and tie/constant behavior; apply to parameters and declared functionals, alongside coverage/reference checks. | High value vetoes promotion; low value can occur when all chains miss the same region. No burn-in guarantee follows. |
| P10 - Bulk / tail ESS screen: >=400 each pooled | Inherited empirical diagnostic floor, not derived q20 precision. Tail min uses .05/.95 indicators. | Check actual estimator identity and each monitored quantity; separately require original-scale mean and quantile MCSE targets. | Insufficient ESS weakens diagnostic reliability. Passing 400 cannot establish superiority, fine means or 2.5% quantile accuracy. |
| P11 - Extra warmup ESS floors: 0 initially | Existing compatibility policy, not a claim that warmup has enough information. | Report warmup ESS and interval stability without turning an undeclared floor into automatic stopping authority. | Low information limits interpretation of a R-hat pass. Required retained precision stays nonzero. |
| P12 - Primary mean-MCSE estimator: TFP positive-pairs autocorrelation estimator | Existing named implementation; distinct from initial-monotone sequence estimator. CLT/long-run variance assumptions are substantive. | Compare saved-draw arithmetic with an independent diagnostic and with P13 on viable chains; assess autocorrelation truncation and stability with n. | Disagreement suggests unresolved long-run variance/mixing, not permission to choose smallest MCSE. Correct formula can still be unreliable on short runs. |
| P13 - MCSE sensitivity: batch means/lugsail, b=floor(sqrt(n)), r=3, c=.5,minimum 20 batches | Inherited baseline parameters; square-root b makes both batch length/count grow, but finite-sample constants are uncalibrated. | Inspect LRV_b and LRV_floor(b/3), complete batch counts and dependence; lugsail=(LRV_b-.5*LRV_small)/.5. Reject unavailable estimates. | Negative/nonfinite/too-few batches cannot grant precision. Agreement on poorly exploring chains still does not prove coverage. |
| P14 - Repeated-look interpretation: operational precision stopping | Explicit limitation. Pointwise CLT intervals are not generally valid after adaptive stopping. | Preserve all looks; freeze horizon for any nominal final interval claim, or separately specify/calibrate a valid sequential procedure. | Without that procedure, report operational estimated precision and its limits; no invented anytime 95% coverage. |
| P15 - Constant/unvisited event: precision unavailable | Identifiability limitation of finite MCMC evidence. Zero empirical variance is not proof of a deterministic posterior event. | Check independent reference mass and visits. For truly IID Bernoulli draws only, zero events gives 95% upper bound 1-.05^(1/N); arbitrary ESS substitution is unjustified. | Unvisited non-negligible region is exploration failure. Independently established deterministic event can use its actual proof; absent proof, do not report zero uncertainty. |
| P16 - Mean/event equivalence margins: .10 pooled SD/.05 probability | Inherited research-resolution choices, not posterior properties. Estimated pooled SD makes the mean margin random. | Freeze a reference scale before comparison or propagate scale uncertainty; apply M7 simultaneous error-inclusive equivalence with actual independence assumptions. | A plug-in uncertain scale weakens nominal coverage. Failure can be bias, reference error or insufficient precision; distinguish before rejecting target. |
| P17 - Quantile equivalence margin: .20 pooled SD | Proposed resolution choice, not mathematically justified merely because quantiles are noisier. | Justify tolerated endpoint error for study purpose; account for reference-scale and quantile uncertainty as in P16/M7. | If needed error is smaller, the proposed margin is unsuitable. Large estimator noise means more evidence, not permission to enlarge margin. |
| P18 - Reference accuracy allocation: <=1/3 each permitted sampler-MCSE tolerance | Proposed error allocation; M7 derives variance ratio 1/9 and combined-SE factor 1.05409 at the limit. | Sum certified numerical/truncation errors where deterministic; use actual MCSE/covariance for stochastic reference and explicit bias bounds. | Spending the allowance independently on multiple error sources can exceed it. Unresolved total reference error blocks reference-based promotion. |
| P19 - Reference comparison confidence: 95% family across 17 quantities | Decision convention; 4+12+1=17 and Bonferroni critical 2.9738199 are derived. Across A methods use 17*A if joint family claimed. | Verify fixed family, adequate marginal intervals, independent method/reference randomness or covariance term, and selection/stopping limitations. | Bonferroni handles dependent quantities but cannot rescue invalid marginal MCSE, missed mass or adaptive interval claims. |
| P20 - Start-group comparison: 5 quantities,95% family, critical 2.5758293 | Derived critical for four means plus event; only two chains per initial-sign group. | Compute group-specific uncertainty, preserve initial labels after migration, propagate scale uncertainty and require stated equivalence margins. | Pooled precision may pass while groups remain unresolved. Small group count limits approximation and may require more confirmation evidence. |

For P13, at n=1000 the proposed b=31 gives 32 complete batches and 8 terminal
draws unused **by that MCSE estimator**; at n=10000, b=100 gives 100 batches
with no remainder. All retained draws still belong to the posterior archive.
The lugsail combination extrapolates two variance estimates; positivity is not
automatic. Neither r=3, c=.5 nor 20 batches has been calibrated here for q20.

The 17-reference and 5-start-group families each have a separate proposed 95%
level. A single simultaneous claim over both families must include all 22
comparisons per method, or explicitly split the family error allowance; two
separate 95% procedures do not automatically give joint 95% coverage.

The equivalence margins may be harder to satisfy than the separate MCSE goals.
For one mean with method SE .02 SD and reference SE .02/3 SD, the 17-quantity
critical consumes about .06269 SD of the .10-SD margin even when point
estimates coincide. Only about .03731 SD remains for observed difference and
reference bias. For a .975 quantile, allowed SE .05 SD and reference .05/3 SD
consume about .15673 SD of the .20-SD margin. These calculations explain why
meeting each MCSE requirement does not automatically pass equivalence.

## R. Reference-method choices introduced outside the parameter tables

No qualified reference method, final domain or node/draw count exists yet.
Reference uncertainty must distinguish target-code correctness, truncation,
integration error and stochastic MCSE. Shared code can share an error; an
independent sampler that uses the same wrong target cannot validate that target.

| ID / additional choice and value | Basis and present status | Smallest useful check and decision | Failure meaning and next action |
| --- | --- | --- | --- |
| R01 - Standardized coordinates: u=(theta-mu)/4 | Derived from fixed prior; favorable numerical scaling, not posterior whitening. | Verify Jacobian and integrand normalization; inspect where independently found posterior mass lies in u. | Incorrect Jacobian changes integrals. Concentration far from prior center requires different resolution/domain, not a claim of invalid posterior. |
| R02 - Box radii: R=4,6,8 | Proposed domain probes, no proven posterior coverage. Prior union bound is 8*Phi(-R). | Price expansions and establish posterior tail/moment bounds below allocated error; inspect multiple regions before trusting apparent stability. | A stable inside-box integral can miss a distant mode. Failed tail certification leaves reference unavailable. |
| R03 - Grid nodes per axis: 9,17,33; fourth powers 6561/83521/1185921 | Derived tensor-product counts; exact grid and quadrature rule remain proposals. | Price complete value calls first; name actual quadrature weights, refinement order and error estimator before execution. | More nodes without a trustworthy rule/tail analysis does not produce a certified reference. Four dimensions can still be too expensive. |
| R04 - Reference error budget: total<=one third of relevant sampler allowance | Decision allocation P18. Domain, discretization, arithmetic and bias bounds add conservatively when dependence is unknown. | Bound normalization and numerators jointly; sum error contributions, rather than giving each the full one-third allowance. | A grid-change difference is not automatically a rigorous integration bound; unresolved bias invalidates reference-based equivalence. |
| R05 - Reference fallback: independently specified sampler, counts undetermined | Alternative protocol, not an already qualified baseline. M7 defines quantity-specific stochastic accuracy. | Establish different implementation checks, region coverage, starts, kernel, seeds, stopping interpretation and total cost before use. | Two samplers stuck in the same mode do not validate each other. If neither reference can be validated/funded, report incomplete scientific qualification. |

For a nonnegative likelihood bounded above by Lmax and a certified normalizer
lower bound Zlower>0,

\[
 P_\pi(\theta\notin D)
 \leq\frac{L_{\max}}{Z_{\rm lower}}P_{\rm prior}(\theta\notin D),\qquad
 E_\pi[|g|1_{D^c}]
 \leq\frac{L_{\max}}{Z_{\rm lower}}E_{\rm prior}[|g|1_{D^c}].
\]

At R=4, the prior box-tail union bound is about .00025337; it is not the
posterior bound without the likelihood/normalizer ratio. A possible route to
Lmax is a proven positive lower bound vmin on every scalar innovation variance:
each Gaussian density is at most `(2*pi*vmin)^(-1/2)`, hence the 30-step product
at most `(2*pi*vmin)^(-15)`. Observation variance .3578897448 suggests a
candidate lower scale under positive UKF moment arithmetic, but a bound for
the **actual numerical program** including its repairs still needs checking.
An extremely loose Lmax/Zlower remains unusable.

For estimated numerator Nhat and normalizer Zhat, with certified errors eN,eZ
and Zhat>eZ, write ghat=Nhat/Zhat. Algebra gives
`abs(ghat-N/Z) <= (eN+abs(ghat)*eZ)/(Zhat-eZ)`.
This exposes normalization error rather than treating a stable numerator as
enough. For quantiles, a uniform CDF error bound eF gives an inverse-CDF bracket
using probabilities p-eF and p+eF where both lie in (0,1). The bracket's width
must meet the quantile error allowance; no unknown density is silently treated
as one. Monotonicity, all error sources and tail contribution must be included.

## B. Seeds, comparisons and cost allocation

| ID / parameter and value | Basis and present status | Smallest useful check and decision | Failure meaning and next action |
| --- | --- | --- | --- |
| B01 - New root namespace: (20260915,150001), role/arm/replicate/beta/chart/stage/block folds | Proposed labels for reproducibility, not optimal random integers. Folding is not a proof of distinct streams. | Enumerate bounded actual int32 schedule and compare with prior/current attempted seeds, including failed native calls. | Collision or accidental reuse breaks intended independence. Repair schedule before execution; printed namespace alone is insufficient. |
| B02 - Independent roles: initialization/training/selection/stress/reference/tuning/development/confirmation | Statistical design requirement; common rows for deliberate paired development comparisons are allowed and recorded. | Trace every consumer's RNG and bank identity, preserve reused rows and fresh confirmation boundaries. | Validation/confirmation leakage invalidates nominal uncertainty. Split random streams, not the 30 likelihood observations defining the target. |
| B03 - Training replications: 3 initially | Proposed exploration allocation, distinct from 3 validation banks. M4/V commentary quantify limitations. | Preserve all seeds and paired configuration effects; price extra replication from unresolved uncertainty. | A selected lucky root does not establish method reliability or broad trainability. |
| B04 - Final method replications: initial 3 independent four-chain systems/method | Proposed limited confirmation allocation, conditional on frozen trained maps; twelve cold chains across three replicates is not twelve independent method replications. | Freeze configurations and comparison family, reserve complete runs, preserve failures/timeouts and compute replicate-level uncertainty. | Few systems give weak ranking evidence and no population-wide claim over training realizations. Underfunding is not a reason to drop hard cases. |
| B05 - Efficiency statistic: total cost to meet all posterior/error criteria | Scientific comparison requirement at matched accuracy. Training amortization must state expected reuse count and include failed work. | Compare validated arms only; report preparation, learning, search, sampling and failures with a common cost unit. | ESS/gradient on invalid runs is not efficiency evidence. Excluding startup or favorable-only successes changes the question. |
| B06 - Efficiency uncertainty: paired log costs,95% interval | Proposed design; M8 gives t-based interval and exact-sign limitation. Three pairs cannot robustly verify its distributional assumptions. | Inspect paired differences and censoring; use predeclared method and report descriptive costs when inference assumptions are unsupported. | Overlapping/unstable evidence leaves viable methods unranked. Six unanimous signs would not establish broad robustness or power. |
| B07 - Further replication: normal n≈[(1.96+.8416)*s/Delta]^2 | Conditional approximate planning formula for two-sided .05,80% power. Exact confidence/power/effect are design choices. | Define worthwhile log-cost Delta, assess pilot variance uncertainty, freeze n before independent confirmation and price it. | Heavy tails, uncertain s or informative timeouts invalidates simple planning; revise statistical model or retain descriptive comparison. |
| B08 - Available campaign: 135275.83289109988s=37.5766 aggregate worker h | Saved amended budget, an authorization/accounting boundary, not an estimated sufficient duration. | Reconcile accumulated attempts and available allocation before launch; count each parallel worker separately. | Insufficient remaining budget requires a smaller explicit scientific comparison or incomplete status, not uncharged parallelism. |
| B09 - Included diagnostic allocation: 54299.37991617s=15.0832h | Saved sub-allocation included within B08. Decimal precision is ledger arithmetic, not timing accuracy. | Ensure every diagnostic charge is included once in total campaign and once in its sub-budget as applicable. | Adding diagnostic hours again double-counts authorization; double-charging expenditures misstates remaining capacity. |
| B10 - Per-arm cap: 28800s=8h | Existing operational boundary; no mathematical training/mixing justification. | Price complete logical arm across its processes and enforce declared stop/reporting behavior. | Cap exhaustion is an incomplete arm. Process splitting cannot create extra budget or silently count as a new independent experiment. |
| B11 - Training/rung max: 8192 per positive beta, further limited by funds | Proposed exploration cap, same count as A14. | Forecast continued learning and all downstream reserves before allocating each extension. | Unresolved progress at cap means insufficient allocation or unresolved method, not trained-to-convergence. |
| B12 - Repair reserve: one priced complete repair and revalidation | Proposed task-based contingency, not an unexplained percentage. | Specify the likely localized failure, required check/retraining/retuning and actual remaining phases; price them before cohort admission. | A nominal reserve without revalidation cost is inadequate. More repairs consume explicit remaining budget rather than renewing authority tokens. |
| B13 - Checkpoint/polling overhead: 128 updates, each sampler chunk,60s heartbeat | Proposed recovery/communication constants. They need cost/risk justification, not statistical calibration. | Measure save duration, interruption recovery/lost work and nonpreemptible call times. The checkpoint model below can inform revision. | Too much overhead or lost work is engineering debt. Heartbeat 60s does not bound GPU native-call execution or grant a deadline guarantee. |

### B-cost. Additional arithmetic and the meaning of the caps

At the obsolete measured update cost 2.0411379247379955s, cumulative
128/512/2048/8192 updates cost .07257/.29030/1.16118/4.64472 hours per
chart/beta. Twelve histories through 512 cost 3.48354 hours before all other
work. This observation came from a beta=.5, B=32 strict-backend pilot before
later source changes. It is a limited historical measurement, not a current
reservation rate. Measure the repaired source and all relevant temperatures.

With four chains and three stages each costing 64 decisions plus 32 startup,
one initial pair has 1152 transitions across its chains. For six chart/beta
scopes and L=(3,5,9,13,18,25), work estimate is
`6*4*3*(64+32)*sum(L+1)=546048`. This excludes refinements, repair, longer
evidence, preparation, compilation and reference work. It explains scope cost;
it does not prove those units accurately price every target call.

For checkpoint cost c seconds, interval t seconds and an illustrative constant
failure hazard lambda per second, expected fractional overhead is approximately
`c/t+lambda*t/2`, ignoring recovery cost and nonstationary failures. Differentiating
gives `t=sqrt(2*c/lambda)`. With no measured failure rate this is a diagnostic
model, not a determined cadence. It shows what information would justify 128
updates instead of preserving that number merely because it was first proposed.

Reserve all required reference, tuning, development, confirmation, reporting and
one complete repair before committing the initial training cohort. If twelve
viable histories cannot reach 512 while preserving that reserve, the proposed
grid is under-budgeted. A partial grid cannot be relabeled full selection. If
posterior IACT is near 16, the 2500-ESS mean goal alone reaches 10000 draws/chain
with four chains; quantiles and equivalence may require more. There is no
mathematical reason every requested criterion must fit the current caps.

## Corrections and unresolved decisions exposed by this audit

1. **The fixed fixture is not empirically justified finance calibration.** Its
   prior center equals the simulated free-coordinate truth and its data path
   starts at the initial mean. Preserve this target for the sampler question;
   make broader modeling claims only after a separate study.
2. **Absolute tolerances are incomplete evidence.** Record covariance/coordinate
   scales, conditioning, active repair branches and same-value derivatives.
   A shift/floor count alone misses changes applied on nominally valid rows.
3. **Adaptive validation intervals are selection summaries.** Fixed shared banks
   support paired development, but lose naïve95% interpretation after adaptive
   use. Final confidence statements need independently frozen comparisons and
   an explicit stopping/multiplicity design.
4. **The .04-nat plateau resolution and the training counts are unproven.** A
   sensitivity/learning protocol can justify a selected endpoint; neither a
   neat factor-four schedule nor Adam's memory scale supplies that evidence.
5. **Reference error sources share one total allowance.** Domain, discretization,
   arithmetic and other deterministic bias cannot each spend the full
   one-third allowance. Random scale estimates in equivalence margins also
   require freezing or uncertainty propagation.
6. **Passing diagnostic floors is weaker than requested precision.** Bulk ESS 400
   is not mean ESS 2500;5%/95% tail ESS is not 2.5%/97.5% quantile accuracy;
   R-hat<=1.01 does not establish burn-in or absence of missed modes.
7. **Fairness and integration remain practical blockers.** q20 map export and
   fresh-start confirmation binding are not checked. A classical route limited
   to L<=25 cannot establish a best tuned classical baseline when travel needs
   longer trajectories. Pricing must include the batch comparison and all
   temperature/chart scopes.

These are design corrections and explicit limitations, not new experiments.
The companion ledger now points here and incorporates the clarified reference
budget, adaptive-validation interpretation and estimated-scale treatment.

## Decision and inference status

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Mathematical/default audit documented | Every ledger parameter has a rationale/status, check and failure response; coverage/arithmetic recorded below | No numerical runs evaluated here | Most target-specific calibrations remain unexecuted | Resolve supported consumer and price complete protocol, then run declared focused checks when executing the campaign | Document completion is not pipeline qualification |
| Keep current target identifiable | Code constants/definitions inspected; model adequacy unproved | No silent target change authorized | UKF approximation and fixture generality | Perform independent target/numerical checks under stated scope | Financial realism or exact latent-posterior correctness |
| Treat learned maps as unqualified until evidence exists | Training/coverage/downstream criteria specified | Earlier canary lacks required training evidence | Capacity, optimizer, learned coverage and affordability | Complete real training selection and downstream assessment under funded plan | Six updates, loss decline or checkpoint replay proves whitening |
| Preserve method direction while diagnosing failures | Repair hypotheses discriminate possible causes | Corrupt implementation/evidence or exhausted budget stops affected work | Which failure mechanism dominates q20 | Follow first discriminating funded repair; preserve failures | An unsuccessful candidate disproves NeuTra or ensembles |

| Inference status | Current conclusion |
| --- | --- |
| Hard veto screen | No new stochastic candidate was run; historical production-audit failures remain separate evidence |
| Statistically supported ranking | None from this documentation/arithmetic audit |
| Descriptive-only differences | Historical timings and pilot observations retain only their original scope |
| Default-readiness | Exact new hyperparameters remain proposals; owner/backend requirements remain requirements |
| Next evidence needed | Qualified value/score/map consumer, adequate reference, priced target-specific training, frozen-scope tuning and independent posterior/efficiency evidence |

Post-audit challenge: even a well-documented protocol can be misleading if all
maps, diagnostics and the reference share the same missed region or numerical
error. The strongest alternative explanation for a later apparent success is
common undercoverage; the strongest alternative to a failed flow is inadequate
optimization or budget. Independent reference/coverage evidence and one-factor
repairs discriminate those explanations. The weakest present evidence is the
absence of an executed, fully funded q20 pipeline with the corrected consumer.

## Source anchors and documentation verification

The [original inventory](artifacts/ssl-lstm-q20-production-parameter-ledger-2026-09-15/r1/source-default-inventory.json)
preserves 31 source paths/hashes/default fields. Target/training sources are from
isolated checkout 3ec9affb; shared HMC sources are explicitly from main. Useful
anchors for the derivations and cautions added here are:

- [Fixture and synthetic data construction](../../bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py):
  `make_full_fixture`, `make_synthetic_observations` and target configuration.
- [Strict covariance classification and derivative route](../../bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py):
  `_principal_sqrt_covariance_classification`, strict factor builders and
  principal-square-root derivative consumers. The source-defined shift is
  inspected; global smoothness/complete numerical error bounds are not proved.
- [Batch reverse-KL training](../../bayesfilter/inference/neutra_weighted_training.py):
  map initialization, optimizer construction, batch loss and clipping.
- [Shared public interface](/home/ubuntu/python/BayesFilter/docs/reference/hmc-tuning-interface.md)
  and [capability registry](/home/ubuntu/python/BayesFilter/bayesfilter/inference/tuning_contract.py):
  supported coordinate/mass choices, candidate evidence and posterior assessment.
- The locally retained R-hat paper linked in section P. Mathematical M1-M8
  identities are derived here under their stated assumptions; no uninspected
  paper is cited as a guarantee for the proposed constants.

Verification of this revision is in
[documentation checks](artifacts/ssl-lstm-q20-parameter-mathematical-audit-2026-09-15/r2/document-validation.json)
and [parameter and troubleshooting coverage](artifacts/ssl-lstm-q20-parameter-mathematical-audit-2026-09-15/r2/parameter-coverage.json).
The unchanged derivations retain their original
[arithmetic examples](artifacts/ssl-lstm-q20-parameter-mathematical-audit-2026-09-15/r1/arithmetic-examples.json)
and [arithmetic validation](artifacts/ssl-lstm-q20-parameter-mathematical-audit-2026-09-15/r1/document-validation.json);
the r1 document hashes and line numbers describe the earlier revision.
Coverage checks compare every original table row to its corresponding T/N/A/V/E/H/G/P/B
row; R01-R05 and the mathematical/cost discussions cover additional prose choices.
The current revision also checks the D01–D18 symptom rows, their parameter
references and the pipeline/ledger links to the owner's investigation order.
These checks establish documentation coverage and arithmetic, not empirical
adequacy, literature-code equivalence, production call-chain correctness or
statistical calibration. Future numerical results should fill the **same
entries** with measured values and failure explanations, rather than create a
new unexplained default ledger.
