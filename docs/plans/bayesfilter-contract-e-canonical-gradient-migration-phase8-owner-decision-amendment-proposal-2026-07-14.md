# Phase 8 Owner-Decision Amendment Proposal

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `PROPOSAL_NOT_EXECUTION_AUTHORITY_OWNER_APPROVAL_REQUIRED`

## Purpose

This proposal reduces the Phase 8 decision boundary to choices that are
genuinely scientific. It does not authorize another Contract E target result.
It supersedes no accepted criterion until the owner approves it.

The earlier handoff mixed two different questions:

1. Is the canonical Contract E gradient adequate at the frozen leaderboard
   evaluation point?
2. Is one fixed numerical program adequate throughout the row's full prior box
   for future HMC?

The first is the Phase 8 LGSSM/leaderboard question. The second is a stronger
HMC-readiness question. A center result cannot prove the second, and requiring
the second before observing any lower-rung center result makes leaderboard
repair depend on an unplanned global numerical certificate.

## Facts Already Fixed By Repository Evidence

### Model, center, and coordinates

The authoritative row contract defines

```text
theta0 = (0.72, 0.55, 0.35, 0.35, 0.45)
theta  = (phi1, phi2, phi3, q_scale, r_scale)
u      = (atanh(phi1), atanh(phi2), atanh(phi3),
          log(q_scale), log(r_scale)).
```

The physical `benchmark_box` prior is

```text
phi1,phi2,phi3 in [-0.95, 0.95]
q_scale,r_scale in [0.05, 2.0].
```

Its exact transformed box is

```text
u_phi in [-1.8317808230648227, 1.8317808230648227]
u_q,u_r in [-2.995732273553991, 0.6931471805599453].
```

These bounds come from
`docs/plans/bayesfilter-filtering-value-gradient-benchmark-source-paper-scope-contract-2026-06-11.json`
and its independently checked generator/test. They are not inferred from a
Contract E output.

The physical-to-HMC gradient chain factors are

```text
(1-phi1^2, 1-phi2^2, 1-phi3^2, q_scale, r_scale).
```

### Existing scientific decisions

- For this `d=3,T=50` row, the accepted relative value-bias boundary is
  `0.001`.
- `0.05*sqrt(5)` is an FD-only implementation screen. It is not a Kalman,
  covariance, transport, or HMC-equivalence margin.
- Contract E--Chol is the only canonical-eligible route.
- The frozen row's pre-Contract-E transport settings are epsilon `0.5`,
  annealing scaling `0.9`, ten finite Sinkhorn steps, and row/column chunks
  `512`. They have independent pre-result provenance but remain baseline
  hypotheses, not established defaults for Contract E.
- The existing five leaderboard seeds are `81120..81124`. They are adequate for
  a fixed five-cell leaderboard aggregate but too few to validate a normal
  approximation or support a reliable stochastic equivalence claim by
  themselves.

## Recommended Scope Decision

Approve two distinct statuses rather than silently conflating them.

### Phase 8 center-scoped scientific gate (recommended)

The Phase 8 claimed parameter region is the frozen leaderboard center plus the
predeclared same-program FD endpoints. One ridge and one finite transport
program are immutable across all seeds, center calls, and endpoints. Passing
this gate may support the LGSSM leaderboard cell and Phase 9 migration, subject
to all value/gradient/statistical gates. It must be labeled
`center_scoped_lgssm_evidence` and cannot support HMC readiness.

This is recommended because the leaderboard row is evaluated at one frozen
theta, and it lets the current program answer that actual question without
pretending that finitely many points certify a continuous HMC domain.

### Full-box HMC gate (separate future gate)

Future HMC-facing status requires one fixed program to remain valid over the
entire transformed benchmark box. A finite corner/grid check is diagnostic
only. A valid full-box claim needs either:

- a checked analytic/interval/Lipschitz certificate covering the continuous
  box; or
- a separately declared probabilistic HMC-region claim based on a specified
  distribution and coverage statement.

Until that evidence exists, emit `HMC_READINESS_NOT_CHECKED`. This is not a
failure of the center-scoped candidate.

Owner decision required: approve the separation above, or require full-box HMC
certification before Phase 9.

## Recommended Gradient Equivalence Criterion

The prior plan used componentwise relative errors and unspecified near-zero
floors. That construction has no natural scale near a zero oracle component and
would require five arbitrary constants.

Use an HMC-coordinate first-order error contribution normalized by the oracle's
own coordinate scale. Let `e` be the mean LEDH-minus-Kalman gradient bias in
`u` coordinates. Let `r_k` be the maximum transformed-coordinate displacement
from the frozen center to either boundary of the authoritative benchmark box,
and define

```text
S_oracle = (1/5) * sum_j r_j * abs(g_Kalman,j)
C_grad,k = r_k * abs(e_k) / S_oracle.
```

The numerator is the exact worst-case first-order error contribution from
coordinate `k` over its declared coordinate radius. The denominator is the
average corresponding absolute oracle contribution across the five coordinates.
It must be finite and strictly positive at the frozen center or this metric is
undefined and the owner must choose another scale. The aggregate diagnostic

```text
R_grad = sum_k r_k * abs(e_k) / sum_j r_j * abs(g_Kalman,j)
```

is the ratio of the two weighted `L1` dual-norm bounds, but it is explanatory
only. The executable gate is componentwise `max_k C_grad,k`, which prevents
cancellation among error components. Because every component uses the same
global oracle scale, a large oracle contribution in one coordinate can make a
large relative error in a weak-oracle coordinate small on this metric. This is
intentional only if the owner approves global first-order contribution, rather
than per-component relative error, as the scientific loss. Every raw component
and its ordinary relative error when the oracle component is nonzero must still
be reported.

It has three advantages:

1. it measures a center-local first-order log-likelihood-gradient diagnostic in the declared HMC
   coordinates and benchmark-box scale;
2. it needs no componentwise near-zero floor; and
3. a full sign reversal is handled by a separate deterministic-oracle veto.

Use

```text
r = max(u0-lower_u, upper_u-u0) componentwise.

r = (2.739425806383947,
     2.4501621366392863,
     2.1972245773362187,
     1.945910149055313,
     2.197224577336219).
```

This is a local gradient comparison scaled by the full prior box; it does not
claim that a first-order Taylor expansion is accurate throughout that box or
that the candidate is chart-valid there.

Required equivalence boundary:

```text
max_k C_grad,k <= delta_grad.
```

No checked scientific argument maps the accepted `0.1%` value-bias boundary to
`delta_grad`, so this proposal does not invent one. The gate must also report
every physical and HMC-coordinate component and its sign. The separate hard
sign-reversal veto is defined operationally in the statistical section below.

Owner decision required: approve this componentwise metric and supply
`delta_grad`, or explicitly choose a different gradient loss and margin. A 95%
confidence level is not an effect-size margin, and the FD-only
`0.05*sqrt(5)` rule is not a candidate value.

### Kalman-only decision-support refinement

The subsequently reviewed comparator-only result found all five HMC-coordinate
Kalman gradient components nonzero at the frozen center; the minimum absolute
component is `0.2653223775603072`. Therefore the owner may instead choose the
more direct center-componentwise metric

```text
max_k abs(e_k)/abs(g_Kalman,k) <= delta_grad
```

without any near-zero floor for this exact center. This alternative avoids the
reviewed global metric's weak-coordinate masking and is recommended when equal
relative accuracy per parameter is the intended scientific loss. It remains
center-scoped and does not establish off-center or HMC-trajectory validity. The
full oracle table and both metric interpretations are in
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-kalman-decision-support-result-2026-07-14.md`.

## Numerical-Error Contract Without Independent Arbitrary Thresholds

Do not invent separate unexplained percentages for raw covariance, condition
number, Sinkhorn marginals, and chunk drift. Treat them as mechanisms and bind
their adequacy to downstream value/gradient stability.

For each fixed candidate, record:

- raw covariance residual in the frozen normalized Frobenius scale;
- all three Cholesky factor condition proxies and positive diagonals;
- full final-coupling row and column residuals in the frozen scales;
- identical-input drift across the predeclared chunk tilings; and
- value plus all five HMC-coordinate gradients.

A candidate is invalid if any chart, finiteness, identity, source, device, or
allocation veto fires. Among valid candidates, numerical selection uses only
the disjoint lower-rung seed `80920` and the downstream criteria below:

For every directed candidate edge `p -> c` that is used by the staged graph,
the lower-rung single-seed edge diagnostics are defined as

```text
DeltaV(p,c) = abs(L_c - L_p) / abs(L_Kalman).

If the owner selects the recommended componentwise-relative loss:
  DeltaG_k(p,c) = abs(g_c,u,k - g_p,u,k) / abs(g_Kalman,u,k).

If the owner selects the global-contribution loss:
  DeltaG_k(p,c) = r_k * abs(g_c,u,k - g_p,u,k) / S_oracle.
```

The edge passes iff `DeltaV(p,c) <= 0.001`, every `DeltaG_k(p,c) <=
delta_grad` under the one owner-selected loss, all required outputs are finite,
and the mechanism-specific
residual direction for that edge is satisfied. These are paired same-input
stability diagnostics, not Kalman equivalence tests. Only edges explicitly
listed in the staged graph are evaluated for selection; every evaluated edge is
retained in the artifact.

If no candidate passes its explicitly listed node or comparator-edge rule,
select none. This avoids pretending that a marginal residual or condition
number alone is a scientific error tolerance. Raw
covariance, conditioning, marginal, and chunk telemetry remain hard explanatory
diagnostics and repair triggers; the downstream stability check is the
adequacy gate.

Owner decision required: approve downstream-stability gating without separate
mechanism percentages, or supply separate mechanism budgets and their
scientific derivation.

## Pre-Result Candidate Ladder

The following finite candidate graph is proposed from pre-existing row identity
and dyadic refinement, not from the completed ridge-`4` smoke.

```text
transport baseline:
  epsilon = 0.5
  scaling = 0.9
  finite steps = 10
  row/column chunks = 512/512

transport schedules:
  steps = 10, 20, 40, 80

lower-rung chunk tilings:
  8/8, 16/16

production-shape chunk tilings (future only):
  256/256, 512/512

ridge scale:
  s0 = q_truth^2 = 0.1225
ridge candidates:
  s0 * 2^k for k in {-24,-20,-16,-12,-8,-4,0,4,8}
```

The ridge ladder is deliberately a bracketing hypothesis spanning numerical
regularization through large chart repair. It does not imply that every point
is scientifically plausible. No interpolation, extrapolation, combination, or
grid expansion is allowed after output.

The exact staged candidate graph is:

1. Ridge stage: evaluate the nine ridge candidates in increasing order at steps
   `80`, chunks `8/8`, stopping at the first node that passes every chart,
   identity, finiteness, and endpoint hard check. Freeze that smallest passing
   ridge. Ridge has no convergence edge: increasing it improves the Cholesky
   chart but can increase raw covariance displacement, so treating it as a
   monotone numerical refinement would be wrong. If no node passes, select
   none. Record every evaluated raw covariance residual, but let the later
   Kalman value/gradient gate decide scientific adequacy of the selected finite
   ridge.
2. Step stage: at the selected ridge and chunks `8/8`, evaluate steps
   `10`, `20`, `40`, and `80`. Comparator edges are exactly `10 -> 20`,
   `20 -> 40`, and `40 -> 80`. Select the first step count whose edge passes
   the deterministic downstream drift bounds and whose final row/column
   residuals do not exceed its comparator's residuals; otherwise select none.
3. Chunk stage: at the selected ridge and step count, evaluate chunks `16/16`
   and `8/8`. The sole comparator edge is `16/16 -> 8/8`; `8/8` is the finer-
   tiling reference. Require deterministic downstream drift within the approved
   bounds. If it passes, freeze `16/16` for the lower rung; otherwise select
   none. This does not select production chunks: a future production rung must
   separately freeze and compare `512/512 -> 256/256` before primary output.
4. Final candidate: rerun the selected tuple with the complete same-program FD
   ladder and all hard diagnostics. No selection statistic uses Kalman
   agreement; Kalman is reserved for the later scientific equivalence gate.

The lower-rung graph contains at most nine ridge nodes, three additional step
nodes because the selected `80/8/8` node is reused, one additional `16/16`
chunk node because the `8/8` node is reused, and one final verification node:
at most fourteen nodes. A shared node is reused by hash, not executed twice.
Selection is lexicographic:

1. reject invalid charts and hard vetoes;
2. apply only the exact step/chunk comparator edges above, using the edge
   formulas above;
3. choose the smallest stable ridge, then fewest stable steps; lower-rung chunks
   are frozen to `16/16` only if their sole comparator edge passes;
4. exact ties use the listed order; and
5. select none if no survivor exists.

Owner decision required: approve this ladder or provide an independently
provenanced replacement before any candidate run.

## Lower-Rung Execution Proposal

After approval of the scientific choices in a future owner-authorized campaign,
draft and review an exact harness subplan for one fresh CPU-hidden float64
attempt with:

```text
T=1, N=32, dataset seed=81100, estimator seed=80920
the complete staged graph above plus final FD endpoints
one attempt per graph node, 300-second cap per node
fresh versioned output directory
```

`N=32` exercises two or four blocks per particle axis under chunks `16` or `8`
and supports a nontrivial `d=3` covariance without using the already observed
`N=4` smoke as a numerical candidate. It is a feasibility/localization rung,
not scientific evidence. The exact harness path, command, result root, node
count, callable count, and aggregate wall-time cap must be frozen in that
subplan before execution. Only after its hard checks pass may a separately
reviewed `T=10`/increasing-`N` rung be drafted. No execution, GPU, or primary-
shape output is authorized by this proposal.

The `300` seconds per graph node proposal has a `14*300=4200` second worst-case
runtime before harness/review overhead. It does not fit the remaining original
eight-hour campaign window and is not executable in this campaign. Owner
decision required: approve the `T=1,N=32` scientific design for a future
explicitly bounded continuation and either its 4200-second node-cap envelope or
a smaller reviewed total budget. This approval still does not launch it; the
future exact harness subplan must first pass skeptical audit and review.

## Primary-Shape Statistical Proposal

The observation sequence is fixed once from dataset seed `81100`. All seeds
below are estimator seeds controlling stateless particle and residual-design
randomness; they do not generate new datasets. The float64 Kalman value and
gradient are therefore deterministic constants across estimator seeds.

Use disjoint estimator-seed sets:

```text
primary-shape feasibility/calibration seeds = 81020..81024
ordered primary audit pool = 81120..81183 (64 untouched canonical
                       primary-shape seeds)
frozen leaderboard aggregate = first five audit seeds 81120..81124
```

The five calibration seeds cannot select or repair numerical settings, choose
an interval method, or change the already fixed audit count. They may establish
only primary-shape feasibility, runtime, and descriptive variance. A failed
candidate stops; there is no primary-shape setting fallback. The frozen five-
seed leaderboard cell is a visible subset of the untouched audit pool; it
remains a descriptive leaderboard aggregate. Statistical equivalence, if
attempted, uses the complete predeclared audit prefix and reports the five-seed
subset separately.

For estimator seed `s`, define

```text
d_value,s = (L_ContractE,s - L_Kalman) / abs(L_Kalman)
e_s,k = g_ContractE,u,s,k - g_Kalman,u,k
z_grad,s,k = the owner-selected gradient loss applied to e_s,k.
```

The population targets are expectations of these six paired variables over the
declared stateless estimator-seed distribution, conditional on the fixed data,
prepared policy, and selected finite candidate. The declared pseudorandom
sampling model treats every distinct, domain-separated estimator key as an
independent uniform Philox stream. The listed integer labels are fixed indices,
not themselves a random sample; independence/exchangeability is an explicit
ideal-PRNG model assumption about their generated streams. Numeric adjacency of
labels is not treated as a source of randomness. Student/Bonferroni intervals
also require finite seed-level variances and the Student marginal model; the
artifacts do not prove those assumptions. A particle, time, endpoint, or repeat
is not a replicate.

Before any calibration output, freeze:

- one six-output family: relative value bias plus five signed gradient-loss
  contributions. Before any calibration output, the owner-selected loss is
  frozen to exactly one of:

```text
componentwise-relative: z_grad,k = e_k / abs(g_Kalman,u,k)
global-contribution:    z_grad,k = r_k * e_k / S_oracle
```

  The corresponding interval and pass rule use that same selected loss;
- two-sided Bonferroni Student intervals with familywise level `0.95`;
- calibration seeds excluded from audit intervals;
- a fixed audit count selected by the owner before calibration output: proposed
  options are `20`, `32`, or `64`, always using that exact prefix of the ordered
  audit pool and with no optional stopping;
- no power guarantee from the five calibration seeds. Their observed variance is
  descriptive only and cannot alter the fixed audit count;
- no data-dependent normality test or interval-method switch. Student coverage
  is accepted as a predeclared model assumption, and gross nonfinite/invalid
  seed behavior is a hard veto rather than an excuse to choose another method;
  and
- no calibration or audit seed may tune a numerical setting, interval method,
  or count.

The previously proposed plug-in power formula is rejected: five calibration
variance draws
plus a normal quantile do not establish `80%` Student-equivalence power. The
three audit counts are compute-budget options, not claims of adequate power.
Before launch, report a prospective detectable-width table over a declared
range of standard deviations; after the fixed-count audit, report achieved
interval widths and allow `inconclusive`.

Let `I_value` be the member of the simultaneous family for the mean of
`d_value,s`. Value equivalence passes only if

```text
-0.001 < lower(I_value) and upper(I_value) < 0.001.
```

Touching or crossing either boundary is inconclusive, and an interval wholly
outside on either side is nonequivalent for this center-scoped LGSSM criterion.

For the gradient decision, construct simultaneous intervals for all five signed
selected-loss contributions. The conservative component upper bounds are

```text
U_grad,k = max(abs(CI_low_k), abs(CI_high_k)).
```

Require every `U_grad,k <= delta_grad`. Also apply the sign-reversal veto below.
For the componentwise-relative loss, this is a floor-free center criterion only
because the comparator artifact proves every center oracle component is nonzero;
the selected loss must provide a different scale or stop if any denominator is
zero.

The sign-reversal veto is operationally defined from the deterministic Kalman
gradient and the implied Contract E interval. Let `I_z,k` be the simultaneous
interval for `z_grad,k`. Transform it back using the inverse selected loss:
`I_e,k = abs(g_Kalman,u,k)*I_z,k` for componentwise-relative, or
`I_e,k = (S_oracle/r_k)*I_z,k` for global-contribution. Then form the implied
Contract E gradient interval `I_ledh,k = g_Kalman,u,k + I_e,k`. If
`g_Kalman,u,k > 0` and the upper
endpoint of `I_ledh,k` is strictly below zero, or if `g_Kalman,u,k < 0` and the
lower endpoint is strictly above zero, the result is a hard sign-reversal
veto. If the deterministic oracle component is exactly zero, no sign direction
exists; the componentwise `U_grad,k` gate still applies.

Student intervals at `n>=20` remain model-based, not distribution-free. The
choice of `20` is a proposed minimum for a variance estimate with more than the
historical five draws, not a theorem guaranteeing normality. Every seed-level
value is retained, the approximation limitation is explicit, and no
superiority/ranking claim follows. If the owner requires distribution-free
coverage, a different design and larger compute budget must be approved before
calibration output.

Owner decisions required: approve the seed split, choose exactly one audit
count from `20`, `32`, or `64`, and approve the Student/Bonferroni model or
supply replacements. The maximum proposed cost is 69 primary-shape seed
evaluations (five calibration plus 64 audit), with no power claim. A wall-time/GPU
budget must be approved after one calibration-seed runtime is known and before the
remaining primary-shape evaluations.

## Skeptical Pre-Execution Audit

Status: `PASS_AS_DECISION_PROPOSAL_ONLY; EXECUTION_BLOCKED`.

- Wrong baseline: Kalman remains the exact LGSSM comparator; same-program FD is
  not substituted for it.
- Proxy promotion: covariance, marginal, condition, and chunk telemetry cannot
  promote a candidate without downstream value/gradient stability.
- Hidden defaults: every proposed number is tied to the frozen row, an accepted
  owner boundary, a standard inferential target, or is explicitly exposed as a
  proposal requiring approval.
- Stale context: raw-barycentric artifacts and their numerical results are not
  candidates or calibration data.
- Fairness: lower-rung, calibration, and audit seeds are disjoint; neither
  calibration nor audit data can tune.
- Missing stop condition: no selection, insufficient fixed-count precision, invalid chart,
  incoherent refinement, expired campaign clock, or insufficient approved
  compute all stop before promotion.
- Misleading success: center-scoped passage is labeled as such and cannot prove
  full-box HMC readiness.

## Minimal Owner Approval

An unambiguous approval can be as short as:

```text
Approve the recommended Phase 8 scientific amendment: center-scoped Phase 8
with full-box HMC readiness deferred; floor-free center-componentwise relative
gradient loss with delta_grad=<OWNER VALUE>; downstream-stability numerical
gate; exact staged ladder and T=1,N=32 lower-rung seed 80920 for a future
bounded continuation; calibration seeds 81020..81024; audit pool 81120..81183;
fixed audit count <20, 32, OR 64>; and Bonferroni/Student intervals with no
power claim. This approval freezes the scientific design but does not launch a
command or extend/reset the current campaign clock. The future exact lower-rung
harness subplan and later primary-shape GPU wall-time budget remain separate
gates.
```

Any changed item should be stated explicitly. Approval of this proposal would
not itself authorize a runtime command. It would not authorize primary-shape
GPU execution, Phase 9, HMC, leaderboard release, or a scientific completion
claim.
