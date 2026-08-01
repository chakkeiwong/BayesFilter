# Contract E--TP Phase 8B Non-Oracle Continuation And Nonlinear Chart Handoff

metadata_date: 2026-07-15
status: ACTIVE_PHASE5_REPAIR
entry_result: `docs/plans/bayesfilter-contract-e-tp-phase8-progressive-score-lgssm-repair-result-2026-07-15.md`

## Objective

Replace the exact LGSSM future-continuation oracle with a fixed non-oracle
continuation basis, and prepare target-specific nonlinear short-prefix charts
without assuming that the failed one-step score interaction transfers.

## Entry Conditions

- Exact LGSSM continuation passes `T=2,10,50` at the frozen center.
- Compact `r_{t+1}c_t` candidate is rejected at `T=10`.
- Phase 5 model adapter density/transition audits remain passing.
- Contract E--TP remains experimental and excluded from canonical/default/HMC
  paths.

## Candidate Ladder

1. LGSSM finite look-ahead likelihoods at horizons `1,2,4,8`, changed one at a
   time and tested first at `T=10`.
2. A reviewed low-rank basis of continuation log likelihoods prepared only on
   preparation observations/parameters, never audit data.
3. Target-specific nonlinear finite look-ahead features using a fixed
   differentiable quadrature approximation to the **target** Markov transition
   and **target** observation likelihood for actual SV, KSC-SV, generalized
   SV, and predator--prey.  The continuation feature must not use the
   Gaussianized LEDH proposal likelihood in place of the target likelihood.
4. For any structurally singular client, use innovation-space continuation
   features with no full-state jitter or off-support anchors. Do not apply this
   rule to a full-rank target merely because it is high-dimensional.

## Evidence Contract

Primary criterion: the smallest non-oracle LGSSM basis must pass the original
`T=2,10,50` center value and componentwise Kalman score screens with same-scalar
AD/JVP/FD, strict positive fixed charts, and no sign reversal. Nonlinear rows
then require `T=1` plus their declared short prefixes before any full horizon.

Vetoes: audit leakage, feature selection by final results, target mismatch,
nonpositive/rank-deficient chart, runtime basis switch, stopped-gradient scalar
mismatch, or use of the LGSSM oracle continuation in nonlinear promotion.

Nonclaims: a short-prefix pass does not establish full-horizon accuracy,
cross-method equivalence, leaderboard/default/HMC readiness, or superiority.

## Skeptical Pre-Execution Audit

The audit found and repaired one material ambiguity in the original candidate
ladder: “finite proposal program” could have made the continuation feature a
function of the LEDH Gaussian proposal surface.  That would answer the wrong
question for actual and generalized SV.  The corrected feature is the target
continuation

\[
 b_{t,L}(x_t;\theta)
 =\int\prod_{s=t+1}^{t+L}
 f_\theta(x_s\mid x_{s-1})g_\theta(y_s\mid x_s)
 \,d x_{t+1:t+L},
\]

with the window truncated at the data horizon.  A fixed quadrature grid gives
the backward recursion

\[
 \widehat b_{s-1}(x_i)
 =\sum_{j=1}^{M}q_j f_\theta(z_j\mid x_i)
   g_\theta(y_s\mid z_j)\widehat b_s(z_j),
 \qquad \widehat b_{t+L}\equiv1.
\]

The executed feature is divided by one positive, parameter-dependent reference
value computed by the same recursion.  For nonlinear teacher clouds the
reference is the maximum continuation log value over the current teacher and a
fixed declared reference point, so the exponent is nonpositive.  This common
scaling does not change the span or the exact feature-matching equation; it
prevents overflow while retaining its total derivative.  No stop-gradient is
allowed.

The pre-execution audit also checked the required failure distinctions:

- a direct one-step quadrature identity isolates continuation implementation
  from LEDH and chart errors;
- an order/radius ladder diagnoses grid truncation and discretization;
- chart rank, positivity, residual, and condition diagnostics isolate reset
  feasibility;
- same-scalar AD/FD tests diagnose derivative wiring independently of target
  approximation; and
- dense target references diagnose the recursive Contract E--TP approximation.

The plan therefore passes skeptical audit after the target/proposal correction.
It does not yet approve any inherited nonlinear window, grid, radius, or chart
capacity as a scientific default.

## Phase 5 Scalar Nonlinear Preparation Contract

The first executable group contains actual SV, KSC-SV, and generalized SV.
Each row uses four features: mass, state, squared state, and one stabilized
target-continuation feature.  Four is a capacity hypothesis forced by the
one-dimensional moment basis plus one continuation functional, not a claim of
optimality.

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- |
| Look-ahead `1,2,4,8` | LGSSM-derived ladder; hypotheses only | LGSSM window transfers poorly | `T=10`, one window at a time |
| Legendre grid orders `33,65,129` | deterministic refinement hypothesis | tail truncation or aliasing | direct one-step and adjacent-rung gaps |
| Radius `8` for actual/KSC SV | existing dense scalar references; baseline | target mass outside interval | radius `8` versus `10` check |
| Radius `10` for generalized SV | target-specific convenience hypothesis | prior/likelihood mass outside interval | radius `10` versus `12` check |
| Center-only chart | current entry condition | active chart fails off center | center prefix first; no region claim |
| Float64 CPU | reference/debug exception | hides GPU/XLA behavior | later Phase 9 trusted GPU rung |

Actual SV uses exact `log(y^2)` observations and the exact log-chi-square
target likelihood.  KSC-SV uses the offset log-square observations and the KSC
mixture target.  Generalized SV uses raw observations and its raw zero-mean
normal target likelihood; its log-square observation is used only by the LEDH
proposal surface.  These identities are vetoes, not reporting metadata.

Predator--prey is not a positive-state target under the frozen BayesFilter
fixture.  Its transition and observation noises are additive Gaussian, its
declared domain policy is `diagnose_negative_after_noise`, and the frozen
dataset contains negative states.  Its Contract E--TP preparation therefore
uses real-plane finite support checks and must not clip or reject a candidate
solely because a coordinate is negative.

The execution order is primitive tests, direct one-step continuation checks,
`T=1`, then `T=10`.  A scalar row may proceed to `T=100` only after its prefix
passes its engineering gates and its reference gaps refine in the expected
direction.  No `T=1000/1008` run is authorized by this preparation record.

If look-ahead `8` fails at `T=10`, a look-ahead `9` diagnostic is allowed only
to test the full remaining-horizon mechanism at that prefix.  It is an oracle-
like finite-prefix diagnostic, not a transferable bounded-window candidate.
Only after that diagnostic may continuation-grid accuracy and teacher order be
varied, one at a time.

The repaired generalized-SV ladder adds fixed progressive target-continuation
marks with requested horizons `(1,4,9)` and a fixed overcomplete KKT chart with
two extra anchors. Near the record end, duplicate truncated horizons are
removed by the deterministic `(time index, record length, requested horizons)`
rule. Preparation chooses a strictly positive center reference and freezes the
Pearson--chi-square precision `diag(1/q0)`; runtime does not select anchors,
clip weights, or change the metric. This is a capacity hypothesis motivated by
the observed square-chart order instability. It must first pass `T=10`,
same-scalar FD, positivity, full-row-rank, and order/radius refinement. A pass
does not select these horizons as a default.

The first combined progressive/KKT candidate passed engineering checks but was
worse against the dense score reference than the prior full-prefix square
chart. Because that candidate changed feature set and chart family together,
it cannot identify the cause. Before any further capacity change, run two
controls: `(9)` with two-extra-anchor KKT to isolate chart family, and
`(1,4,9)` with zero-extra-anchor KKT to isolate the added features. In the
second control `m=K`, so the equality constraints have one feasible weight
vector and the KKT formula is algebraically the square solve regardless of its
metric. This audit repair is required even though the combined command
succeeded.

The controls show that feature matching alone leaves downstream functionals
underdetermined: two extra anchors with a max-min reference degrade even the
single full-prefix feature. The next repair changes only the KKT reference and
anchor design. For scalar state, choose fixed weighted-quantile anchors from
the center teacher, aggregate all center teacher mass into nearest-anchor
Voronoi cells to form `q0`, freeze `P=diag(1/q0)`, and apply the same exact KKT
feature correction at runtime. Unlike the max-min reference, `q0` approximates
the full teacher measure before correction. Test anchor counts `8,12,16` at
`T=10`, stopping at the first positive full-rank chart; then require teacher
orders 25 and 41 before any longer prefix. The counts are capacity hypotheses,
not defaults.

All three pure quantile rungs fail positivity; their most negative corrected
weights are approximately `-0.523`, `-0.216`, and `-0.017` for 8, 12, and 16
anchors. The failure is geometric: weighted quantile anchors approximate the
teacher distribution but do not guarantee that the exact feature target lies
in their positive convex hull. The next discriminating repair unions eight
weighted-quantile anchors with a proven positive LP basis. At the center, a
convex Pearson--chi-square projection finds the feasible positive weight vector
closest to the Voronoi-aggregated teacher mass. That vector and
`P=diag(1/q0)` are frozen for runtime. This changes the reference design only;
features, target, grid, teacher order, and exact equality gate remain fixed.

The chi-square projection reaches the equality target but its optimum is on the
boundary, so it fails strict positivity. Do not introduce a numerical weight
floor. Replace only the center reference solve by the KL information projection
of Voronoi teacher mass onto the same exact feature constraints. Its KKT form
is an exponential tilt of a positive reference and is therefore strictly
positive whenever the target lies in the relative interior. The runtime still
freezes that center solution and uses its local Hessian
`P=diag(1/q0)`; no iterative optimizer enters the executed scalar.

The first hand-written KL dual Newton/backtracking routine stalls at equality
residual `2.9e-3`. This is an offline optimizer failure, not a chart or method
result. Replace it with a scaled trust-region least-squares solve using the
analytic KL-dual Jacobian, while retaining strict positivity and maximum
equality residual `1e-12` as admission gates. No failed preparation artifact is
promoted.

The scaled trust-region KL solve also stalls above the exact equality gate,
indicating a boundary or severe-conditioning problem for that dual
parameterization. Replace it by a constructive feasible path. Start from the
strictly positive square basis. Insert all non-basis quantile anchors in their
Voronoi-mass proportions, and solve the compensating basis weights analytically.
If full insertion is strictly feasible, use it. Otherwise the feasible scale is
`(0, alpha_max)` and select its logarithmic-barrier analytic center
`alpha_max/2`, the unique maximizer of
`log(alpha)+log(alpha_max-alpha)`. This introduces no weight floor or fitted
tolerance and preserves the exact feature target by construction.

At a late concentrated teacher, eight weighted quantiles can map to only seven
distinct particles. This is an offline anchor-selection degeneracy. Freeze the
total tie rule in advance: retain the distinct weighted-quantile anchors and
fill any missing slots with highest-normalized-mass unused teacher points in
stable index order. The rule uses only the preparation teacher and does not
switch at runtime.

## Austria SIR Target Audit Amendment

The Austria fixture has state dimension 18, observation dimension 9,
`Q=I_18`, and initial covariance `I_18`; it is full rank. Phase 4 structural
deterministic completion is therefore inapplicable to this row and would change
the target.

The inspected author source contains two different finite programs:

- `models/sir_austria/st_process.mlx` adds Gaussian noise and clips susceptible
  coordinates at zero;
- `models/sir_austria/transition.mlx` evaluates an ordinary full-rank Gaussian
  transition density around the RK4 mean, with no atom or truncation term.

Clipping creates point masses on susceptible-coordinate zero hyperplanes, so
the sampling law is not the density represented by `transition.mlx`. An
ordinary invertible LEDH Jacobian cannot account for those atoms. The SIR
Contract E--TP observed-data row is consequently blocked until its target is
bound explicitly to one of these source programs. The differentiable density
program may be tested as a separately named source-density target; it must not
be described as the clipped simulator law. P90/P91 component evidence remains
valid only at its documented scope.

Same-scalar differentiation uses coordinate central FD of the exact executed
finite program.  The FD-only relative screen is `0.05*sqrt(p)` with an absolute
denominator floor recorded in the result.  This screen must not be reused as a
cross-method or target-agreement margin.

## Required Artifacts

- LGSSM look-ahead basis preparation/result ladder;
- one preparation record per nonlinear target stating feature program,
  innovation rule, support policy, capacity, and parameter scope;
- dense/streaming parity, support, same-scalar derivative, and short-prefix
  result artifacts;
- failure/repair ledger and phase result.

## Handoff

Only a non-oracle LGSSM basis that passes the frozen `T=50` criteria may become
the starting hypothesis for nonlinear chart preparation. Each nonlinear target
must still justify and test its own basis. After all short-prefix candidates
pass, hand off to Zhao--Cui comparator certification, paired all-model
comparisons, refinement, GPU/XLA scaling, and terminal synthesis.

## Stop Conditions

Stop for human direction only if the scientific target must change, audit data
would be required for preparation, campaign budget must expand, or no positive
fixed chart exists across the declared capacity ladder. Candidate failure is a
repair trigger, not a campaign stop.
