# Actual-SV Overcomplete Analytical Chart Repair Plan

Date: 2026-07-17

Status: `COMPLETE_PHASE_8_NARROW_ENGINEERING_REPAIR`

Failure ledger item: `CE-07` in
`docs/plans/bayesfilter-contract-e-active-failure-ledger-2026-07-17.md`

## Scope And Phase Objective

Repair the experimental Actual-SV Contract E--TP `T=1000` fixed-square chart
failure without changing the four declared features, the Actual-SV target law,
the observation record, the finite-difference policy, or canonical Contract
E--Chol semantics.  The repair replaces the four-anchor square representation
with a fixed-shape overcomplete equality-constrained quadratic projection whose
runtime solution and total derivative are analytical linear solves.

Phase 0 first amends
`docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex` so the mathematical
target, reference distribution, Pearson metric, derivative, positivity
and stability gate, and scientific boundary are fully documented before code
changes.

This program does not include NAWM.  It does not repair the separate canonical
Sinkhorn wiring defect `CE-01`, and it cannot issue canonical, HMC, leaderboard,
or method-superiority claims.

## Research Intent Ledger

| Field | Binding intent |
| --- | --- |
| Main question | Can a fixed overcomplete analytical moment chart preserve the same Actual-SV finite feature program and remain numerically positive on the already-declared local `1e-5` FD neighborhood where the square chart fails? |
| Candidate | Four fixed features with `K>4` fixed anchor indices at every nonterminal time; center reference weights `r`; frozen `P=diag(1/r)`; analytical equality-constrained KKT solve. |
| Baseline | Existing four-feature/four-anchor square chart prepared at the center and executed at `T=1000`. |
| Expected failure mode | The selected anchor family does not contain the target moments in its positive convex hull over the local box; reference projection is nonpositive; KKT Gram conditioning makes the finite solve unreliable; fixed-shape recursion is wired incorrectly; or the new finite scalar is derivative-correct but scientifically inaccurate. |
| Promotion criterion | Select the smallest capacity that passes the predeclared design points only.  Freeze it, then require its held-out chart audit, own-scalar derivatives, original endpoint regression, and trusted GPU/XLA engineering gates to pass.  The route remains experimental. |
| Promotion veto | Any nonfinite output, feature mismatch beyond its floating-point residual gate, rank loss, nonpositive computed weight, dynamic anchor switching, manual/autodiff/FD mismatch, held-out local chart failure, wrong target/preparation identity, or GPU/XLA failure. |
| Continuation veto | The mathematical target cannot be made internally consistent; no `K<=25` candidate passes every design point; the predeclared ladder is under-budgeted or repeatedly times out after one localized repair; fixed-shape implementation cannot represent the declared program; artifacts are corrupted; or the bounded campaign budget is exhausted.  A failed capacity is a repair trigger, not automatically a research-direction veto, and an engineering timeout is not a mathematical capacity failure. |
| Repair trigger | Local algebra/JVP/VJP error, ragged anchor preparation, wrong frozen-dependency treatment, graph/XLA defect, or insufficient anchor capacity within the predeclared ladder. |
| Forbidden conclusion | No positive result proves a nonzero-radius HMC region, nonlinear filtering exactness, canonical Contract E--Chol correctness, general SV validity, leaderboard readiness, or superiority. |

## Mathematical Target

At nonterminal time `t`, retain the existing feature vector

\[
 \psi_{t,\theta}(x)
 =\bigl(1,x,x^2,h_{t,16,\theta}(x)\bigr)^\top
 \in\mathbb R^q,\qquad q=4.
\]

Choose one capacity `K` with `5 <= K <= 25`, constant over all nonterminal
times.  Preparation freezes `K` teacher-cloud indices per time step.  The
indices are fixed, while the particle locations at those indices and hence
their feature columns remain differentiable functions of `theta`.  Let

\[
 A_\theta=D^{-1}
 [\psi_{t,\theta}(z_1(\theta)),\ldots,
  \psi_{t,\theta}(z_K(\theta))],
 \qquad b_\theta=D^{-1}m_{t,\theta}.
\]

Here `D` is a positive diagonal row-scale matrix prepared at the center and
then frozen.  In particular, `D` is nonsingular and neither `D` nor its
selection rule is differentiated at runtime.

For positive frozen center reference weights `r` satisfying
`A_{theta_0} r = b_{theta_0}`, define

\[
 P=\operatorname{diag}(r_1^{-1},\ldots,r_K^{-1}),
 \qquad R=P^{-1}=\operatorname{diag}(r).
\]

The runtime program is

\[
 w_\theta=\arg\min_w
 \frac12(w-r)^\top P(w-r)
 \quad\text{subject to}\quad A_\theta w=b_\theta,
\]

with analytical solution

\[
 w_\theta=r+RA_\theta^\top
 (A_\theta R A_\theta^\top)^{-1}
 (b_\theta-A_\theta r).
\]

Implementation uses Cholesky or linear solves, never explicit matrix inverses.
The mass feature enforces `sum(w)=1` as part of the same equality system.

### Reference-Weight Construction

For a fixed anchor set, assign every center teacher particle to its nearest
anchor in state space with deterministic stable tie-breaking and aggregate the
normalized teacher masses into `v`.  Preparation requires every Voronoi cell
to receive strictly positive finite mass; hence `v_i>0` is a hard gate, not an
assumption inferred from distinct indices.  Let `V=diag(v)`.  Define the analytical
Pearson projection

\[
 r=v+VA_0^\top(A_0VA_0^\top)^{-1}(b_0-A_0v).
\]

This is the unique equality-constrained minimizer of

\[
 \frac12\sum_i\frac{(u_i-v_i)^2}{v_i}.
\]

Reject rather than clip, floor, or reassign cells if any `v_i` or computed
TensorFlow `r_i` is nonpositive or nonfinite, or if the construction is
rank-deficient or fails the declared residual and conditioning gates.  Offline
nonlinear or inequality-constrained optimization
is not part of this candidate.  Anchor indices come from deterministic
center-teacher weighted quantiles, with stable duplicate handling.  The
analytical definition of `r` is the only reference-weight construction.

### Total Derivative

Set

\[
 S=A R A^\top,\qquad c=b-Ar,\qquad \lambda=S^{-1}c,
 \qquad w=r+RA^\top\lambda.
\]

Because prepared `r` and `R` are frozen,

\[
 \dot S=\dot ARA^\top+AR\dot A^\top,
 \qquad \dot c=\dot b-\dot A r,
\]

\[
 \dot\lambda=S^{-1}(\dot c-\dot S\lambda),
 \qquad
 \dot w=R(\dot A^\top\lambda+A^\top\dot\lambda).
\]

The derivative must include teacher-target motion and motion of the feature
columns at the frozen indices.  It must not differentiate the offline index,
`r`, `R`, or row-scale selection.

### Finite-Program Positivity And Stability Gate

The claimed runtime object is the executed finite TensorFlow program.  Its
binding positivity condition is therefore that every computed weight is finite
and strictly positive.  The implementation additionally gates full row rank,
condition-roundoff, and scaled equality residual using the repository's
Higham-style floating-point model.  These checks detect an unreliable solve;
they are not represented as a rigorous componentwise sign certificate for the
unknown exact real-arithmetic solution.

For the selected candidate, an independent CPU high-precision audit
recomputes the deterministically selected weakest design case and weakest
held-out case at increasing precision and reports the sign and float64
discrepancy.  Weakest means the smallest finite TensorFlow weight, with ties
broken first by time index and then by the predeclared point order.  This is
corroborating numerical evidence, not
the BayesFilter algorithmic backend and not a proof by interval arithmetic.
If float64 positivity is not stable under the precision ladder, the candidate
fails.  Preparation, runtime, and tests share the finite-program rank,
condition-roundoff, equality-residual, finiteness, and computed-positivity
definitions.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific/engineering question | Does the analytical overcomplete candidate repair the known fixed-square positivity failure while preserving the same finite four-feature program and a correct total derivative? |
| Exact baseline | Frozen Actual-SV `T=1000`, order 25, continuation order 129/radius 10, lookahead 16 square-chart preparation and Phase 4 CPU result. |
| Primary criterion | The smallest design-passing capacity is frozen; that same candidate passes held-out finite-program chart gates, the exact `T=1000` center/four-endpoint scalar and FD gate, and GPU/XLA execution. |
| Promotion vetoes | Wrong target or features; nonconstant `K`; runtime index switching; nonfinite/rank/conditioning/residual/computed-positivity failure; derivative failure; held-out failure; or XLA/GPU failure. |
| Explanatory diagnostics | Minimum computed positive margin, high-precision sign audit, first weak time, condition numbers, capacity, same-target value/score differences, compile/warm time, memory, and descriptive CPU/GPU deltas. |
| Not concluded | HMC-region validity, exact nonlinear likelihood, canonical Contract E--Chol, general cross-model benefit, leaderboard readiness, or statistical superiority. |
| Artifacts | Phase result records plus fresh structured preparations/results under `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/`. |

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic | Promotion status |
| --- | --- | --- | --- | --- |
| Four features | Existing Actual-SV TP target and prior short-prefix evidence | Feature family can remain scientifically inaccurate despite chart repair | same-target score comparison after own-scalar derivative checks | frozen target, not newly promoted |
| `K=5..25` ladder | `q+1` is the smallest overcomplete system; 25 is the existing Actual-SV teacher order and therefore the full available initial anchor pool | repeated capacity search overfits known endpoints | choose smallest passing design candidate; final held-out points remain unused in selection | capacity hypothesis |
| Constant `K` over time | XLA loop shape requirement and fixed-program identity | weak times need more capacity than others | preparation must reject any rung that cannot provide exactly `K` distinct anchors at all times | required engineering invariant |
| Center weighted-quantile anchors | Quantiles cover center teacher probability with a simple deterministic rule and require no candidate-defining LP/SciPy route | duplicate quantiles, weak tails, or nonpositive analytical `r` | stable de-duplication, highest-mass-unused fill only for duplicates, anchor state/mass coverage, and held-out positivity | preparation hypothesis |
| Voronoi center mass `v` | Deterministic approximation of teacher mass represented by anchors, with `v_i>0` a hard preparation gate | empty or tiny cells invalidate the Pearson geometry or make `P` ill-conditioned | `v`, `r`, `S`, computed margin, and high-precision solve telemetry | statistical reference hypothesis |
| Analytical Pearson projection for `r` | Equality-constrained quadratic derivation | `r` can be nonpositive even when some positive representation exists | reject rung at first nonpositive or numerically unstable `r` | reviewed mathematical candidate |
| `P=diag(1/r)` | Pearson chi-square Hessian and local second-order KL geometry | tiny `r` creates ill-conditioning | KKT Gram, residual, computed positivity, and high-precision audit | reviewed mathematical candidate |
| Existing `1e-5` FD endpoints | Frozen Phase 4 gate and owner FD-only policy | known endpoints can be overfit | separate held-out local points | regression gate, not equivalence margin |
| Design points in normalized unconstrained coordinates | `theta=theta0+diag(1e-5,1e-5)z`, with `z=(0,0)`, the four signed axis unit vectors, and the four signed corners `(+-1,+-1)` | nonlinear interior failure can be missed | exact held-out normalized points below | convenience design, explicitly limited |
| Held-out normalized points | `(-1/2,-1/2)`, `(-1/2,+1/2)`, `(+1/2,-1/2)`, `(+1/2,+1/2)`, `(-3/4,-1/4)`, `(-3/4,+1/4)`, `(+3/4,-1/4)`, `(+3/4,+1/4)`, frozen before capacity execution | still sparse and not an HMC region | report failures without retuning | held-out diagnostic |
| Float64 first | Existing reference route | does not establish TF32 production behavior | trusted GPU float64 parity and memory | reference implementation only |

The held-out direction vectors and all design points must be written to the
Phase 1 specification before any candidate preparation executes.  They may not
be changed after results are observed.

## Skeptical Pre-Execution Audit

The plan was audited for wrong baselines, proxy promotion, hidden defaults,
missing stop conditions, unfair comparisons, stale context, environment
mismatch, and artifacts that would not answer the question.

Material findings and repairs:

1. More features were initially conflated with more anchors.  More features add
   constraints and can worsen positivity.  This plan freezes the same four
   features and increases only anchor capacity.
2. A per-time smallest passing `K` would create ragged loop state and different
   capacities across time.  This plan requires one global `K` for all 999
   projections.
3. A positive center weight alone did not protect against floating-point sign
   uncertainty.  A solve-level condition number also cannot certify the sign of
   the composed final weight.  The plan gates the executed finite program on
   computed positivity plus rank/conditioning/residual diagnostics and uses a
   separately labeled high-precision audit of the weakest saved solve inputs.
4. The earlier proposal risked calling anchor locations frozen.  Only anchor
   indices are frozen; selected particle locations and feature columns move
   with `theta` and are part of the total derivative.
5. Selecting `K` on the same four endpoints used for the final claim would make
   the final result a regression fit.  The known endpoints remain required
   design/regression points, while separately frozen held-out points can veto
   without causing retuning.
6. A penalized least-squares regression would only approximately match features
   and change the target.  The plan uses exact equality constraints.
7. Runtime inequality constraints would create active-set switching.  The plan
   uses an unconstrained analytical equality projection and fails closed if its
   prepared region is not interior.
8. The existing recursive KKT route uses Python time unrolling.  It is a
   reference only; production-shaped work requires a new `tf.while_loop` KKT
   core with constant tensor shapes.
9. The current general KKT implementation already has manual JVP/VJP evidence.
   The plan reuses it and adds recursive total-score and loop/XLA tests rather
   than inventing a second algebra.
10. The repaired route generally computes a different recursive finite scalar
    from the square route.  Square-chart value equality is not a gate; feature
    identity, own-scalar derivatives, held-out chart validity, and GPU/XLA are
    the binding gates.  Same-target differences remain required descriptive
    scientific evidence because no equivalence margin is justified.
11. Actual-SV TP is experimental and separate from Contract E--Chol/Sinkhorn.
    The plan forbids using this repair as evidence for `CE-01` closure.
12. NAWM appeared in stale historical scope.  It is explicitly excluded.
13. Selecting extra anchors only by descending teacher mass could cluster them
    near one mode, while an LP-selected seed basis would make SciPy part of the
    candidate-defining preparation.  Each `K` rung instead uses the `K`
    center-teacher weighted quantiles with probabilities
    `(j+1/2)/K`, `j=0,...,K-1`.  TensorFlow stable sorting, cumulative mass, and
    `searchsorted` select the indices.  Stable de-duplication is followed by
    highest-mass-unused fill only when repeated quantiles leave fewer than `K`
    distinct indices.  The rule is frozen before candidate execution and uses
    no NumPy/SciPy numerical algorithm.
14. Reusing the existing dense KKT primitive would form and solve a `K x K`
    precision system, contradicting the intended diagonal-P complexity and the
    new nonthrowing fail-closed semantics.  Phase 2 adds a specialized
    diagonal-P core using elementwise `R A^T`, one `q x q` Gram Cholesky/solve,
    and shared finite-program diagnostics.  A strict asserting wrapper remains
    test-only/reference-only.
15. Constant `K` is necessary but not sufficient for a loop-native graph.  The
    new recursion peels the first step with 25 teacher points, runs positive
    time steps with fixed `25K` teacher shape, pads lookahead-16 windows with an
    active count, and peels the terminal step with no projection.  Actual SV
    has two parameters, so the production manual score composition carries two
    forward JVP directions through the same fixed-shape loop.
16. The predecessor had no justified cross-method equivalence margin for
    Actual SV.  Phase 6 is therefore descriptive scientific evidence and may
    reveal a large mismatch, but it is not a pass/fail equivalence gate and
    cannot support scientific promotion.  This campaign closes only the chart
    and own-scalar engineering failure if its binding gates pass.
17. A timeout was initially misclassified as a failed capacity, which could
    bias capacity selection.  Timeouts are now engineering evidence only, with
    one bounded repair retry and a repeated-timeout continuation veto.
18. The Pearson construction had left `v_i>0` and the frozen nonsingularity of
    `D` implicit.  Both are now explicit preparation conditions; empty cells
    are rejected without clipping, flooring, or reassignment.
19. The high-precision audit was inconsistently described as both weakest-case
    corroboration and an all-case binding gate.  The finite TensorFlow program
    is binding for every case; only the deterministically weakest design and
    held-out cases receive corroborating increasing-precision recomputation.
20. The owner-selected 8192 MiB GPU cap now names the TensorFlow logical-device
    mechanism, must be set before initialization, records physical/logical
    devices, fails closed if unavailable, and forbids simultaneous memory
    growth.
21. A pilot had been assigned to Phase 1 before the candidate route exists.
    Phase 1 now freezes its commands and costing rule; Phase 3 executes the
    three measurements after Phase 2 implementation and before any capacity
    selection.

Audit verdict: `PASS_AFTER_REVISION_FOR_BOUNDED_REVIEW`.

### External Review Status

Claude health returned `CLAUDE_PROBE_OK`, and a one-file read probe returned
`ACTUAL_SV_PLAN_READ`.  The substantive one-file review and the subsequently
narrowed algebra-only review both returned no review text or verdict.  Under
the bounded review protocol, silence is not agreement and is classified as a
review prompt/tool limitation.  A later single-path retry was denied by the
platform external-data control, so no workaround was attempted.  Execution
therefore requires a fresh independent Codex mathematical/design review;
Claude unavailability does not waive any mathematical or scientific gate in
this plan.

The first independent Codex review returned `VERDICT: REVISE`.  It confirmed the
analytical minimizer and frozen-reference JVP, and identified material gaps in
the sign-certificate claim, scientific comparison criterion, preparation
backend boundary, dense-P complexity, loop shape, score composition,
design/audit wording, reproducible point geometry, compute allocation, and KSC
scope.  Findings 1--9 were patched.  Its focused re-review again confirmed the
algebra but found four execution-contract gaps: timeout classification/pilot
costing, strict positivity of `v` and frozen nonsingular `D`, high-precision
audit scope, and the exact GPU logical-memory contract.  Findings 17--20 above
patch those gaps.  A final focused verdict is required before Phase 0
execution.

A later local sequencing audit found and patched finding 21.  No candidate
experiment has run, and no capacity result has been observed while revising
these rules.

Final focused local review on 2026-07-17 found the four re-review findings and
the sequencing finding resolved.  The Pearson projection, runtime minimizer,
and frozen-reference JVP are mathematically consistent under the explicit
positive-`v`, positive-`r`, frozen nonsingular-`D`, and full-row-rank
conditions.  Timeout classification, audit separation, high-precision scope,
and GPU logical-memory configuration are now internally consistent.  Claude's
missing verdict is retained as a review limitation, not treated as agreement.
Final local verdict: `AGREE_FOR_PHASE_0_EXECUTION`.

## Phase 0: LaTeX Amendment

### Objective

Amend the overcomplete-chart section in
`docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex` before implementation.

### Required artifacts

- self-contained Actual-SV failure statement;
- distinction between features, anchors, and parameter charts;
- analytical construction of `r` from center Voronoi masses `v`;
- justification and definition of `P=diag(1/r)`;
- proposition and proof for the analytical center projection;
- proposition and proof for the runtime solution and total JVP;
- finite-program positivity/stability gate and high-precision corroboration;
- fixed-index/moving-location clarification;
- algorithm block for offline preparation and runtime execution;
- complexity `O(Kq^2+q^3)` with `q=4`;
- explicit nonclaims and connection to the time-748 counterexample.

### Checks and review

- verify every symbol is defined before use;
- compare equations with the TensorFlow KKT implementation;
- build `docs/main.pdf` with `latexmk`;
- inspect the relevant LaTeX log for undefined references and mathematical
  warnings;
- write a Phase 0 result record before code execution.

### Stop conditions

Stop if the proposed `r`, `P`, runtime solution, or derivative is internally
inconsistent, or if the document cannot distinguish frozen preparation from
runtime differentiation.

## Phase 1: Fixed Design And Preparation Specification

### Objective

Freeze the capacity ladder, design points, held-out points, deterministic
anchor construction, numerical stability gate, artifact schema, and exact
commands before evaluating a candidate.

### Required artifacts

- one machine-readable specification under the fresh output root;
- exact center `theta0=(0.2533471031357997,-0.916290731874155)`, unconstrained
  coordinate system, scale matrix `diag(1e-5,1e-5)`, normalized axis/corner
  design vectors, and normalized held-out vectors stated in this plan;
- target observation hash, preparation-source hash, feature settings, teacher
  and continuation quadrature settings, parameter order, and time order;
- design/audit separation and seed-free deterministic construction;
- global constant-`K` rule;
- exact TensorFlow weighted-quantile probabilities, stable sort/search,
  de-duplication, and fill rules producing exactly `K` indices at every time;
- route identity fields for anchors, `r`, `P`, scales, features, target, and
  prepared inputs.
- exact CPU/GPU commands, environment, thread/core caps, per-command timeout,
  per-phase attempt allocation, and budget ledger;
- an exact frozen pilot protocol for `K=5,T=10` and `K=25,T=10`, plus one
  center-only `K=25,T=100` timing probe, separately recording graph
  trace/compile, preparation, and warm evaluation costs after Phase 2 supplies
  the candidate implementation and before the full Phase 3 ladder is eligible;
- a conservative pre-launch projection rule that charges every possible rung at
  the greater of its measured endpoint-`K` per-step warm cost and the linear
  interpolation between the two endpoint pilots, adds the measured compile and
  preparation costs without amortizing them across distinct shapes, and
  assumes all 21 rungs survive to `T=1000`.  If that worst-case projection
  exceeds the remaining Phase 3 budget, revise the screening implementation or
  stop as under-budgeted rather than launch.

### Handoff condition

Proceed only when the specification can be consumed without hidden defaults by
both preparation and runtime harnesses and the command/budget manifest,
including the still-unexecuted pilot protocol and costing rule, receives a
focused review.  Pilot execution is a Phase 3 preflight after implementation.

## Phase 2: Analytical Primitive And Recursive Loop Implementation

### Objective

Use the existing KKT algebra as a checked reference and add a specialized
diagonal-P primitive plus the missing fixed-shape loop-native Actual-SV KKT
recursion and factory.  No KSC behavior or evidence is changed in this plan.

### Required implementation

- tensors shaped `[T-1,K]` for indices and reference weights; `R` is represented
  by those weights and no dense precision tensor is stored;
- diagonal-P algebra using a `q x q` Gram Cholesky/solve, with
  `O(Kq^2+q^3)` time and `O(Kq+q^2)` temporary storage;
- statically peeled first step, fixed `25K` positive-time teacher shape,
  padded lookahead windows with active counts, `tf.while_loop` middle recursion,
  and statically peeled terminal step;
- XLA enabled by default;
- exact teacher feature matching and nonuniform carried weights;
- a two-direction forward manual JVP through the complete recursion for the
  Actual-SV total score; primitive VJP/JVP duality remains a focused algebra
  test, while autodiff is an independent reference;
- a nonthrowing core returning rank, conditioning, residual, finiteness, and
  computed-positivity validity plus fail-closed poisoned claim outputs; an
  asserting wrapper is limited to focused tests/reference debugging;
- no NumPy/SciPy or runtime optimizer in the compiled path;
- no NumPy/SciPy numerical algorithm in candidate-defining preparation;
- historical square, dense-P, and Python-unrolled KKT paths remain explicitly
  scoped.

### Checks

- diagonal-P result versus the existing dense block KKT reference;
- exact features and mass;
- square equivalence at `K=q`;
- two-direction recursive JVP versus autodiff/FD and primitive VJP/JVP duality;
- deliberate rank, residual, and positivity failures;
- `T=1,2,10` eager/CPU-XLA checks;
- source/graph guard rejecting Python horizon unrolling and dense `K x K`
  precision solves in the candidate route.

## Phase 3: Capacity Preparation Ladder

### Objective

For `K=5,...,25`, construct one center preparation recursively and test all
predeclared design points.  Use a staged horizon screen: every rung first runs
`T=10`; survivors run `T=100`; only survivors of both run `T=1000`.  Select the
smallest `K` that passes all design points at all three horizons.  Stop at that
first pass and do not inspect held-out outcomes until selection is frozen.

Before the ladder, execute the frozen Phase 1 pilot protocol using the Phase 2
implementation and apply the conservative worst-case costing rule.  The ladder
is ineligible if that projection exceeds the remaining Phase 3 budget.

Phase 3 execution amendment, 2026-07-17: the frozen endpoint preparations
themselves supplied discriminating mathematical evidence before warm runtime
timing was possible.  `K=5` fails from a negative Pearson reference and `K=25`
fails the strict positive-Voronoi-mass gate.  They remain failed-capacity
artifacts and are not converted into timing observations.  For the timing-only
cost envelope, use the widest executable `T=10` pair, `K=7` and `K=23`, and
replace the ineligible `K=25,T=100` center timing probe with the widest
center-preparation-passing capacity at `T=100` (starting from `K=23` and
descending only when preparation itself supplies a mathematical failure).
Charge the already-observed failed endpoint preparation costs, every
surviving rung separately, and the greater of measured endpoint per-step warm
cost or linear interpolation.  Do not infer that `K=7` is selected until it
passes all design points at `T=10,100,1000`.  This repair changes only the
costing instrument; it does not change the ladder, target, points, gates, or
five-core-hour budget.

### Evidence per rung

- exact `K` at every time;
- anchor coverage and identities;
- `v`, analytical `r`, and `P` diagnostics;
- minimum computed runtime weight and its time/parameter point;
- `A` and `ARA^T` rank/conditioning;
- equality residual and feature residual;
- preparation runtime and failure classification;
- attempts, wall time, core-hour charge, and remaining budget.

After selection is frozen, recompute the deterministically weakest design case
at increasing CPU precision and require stable positive sign before Phase 4.
This is corroborating evidence for the selected finite-program result, not a
capacity-selection metric and not an interval proof.

### Stop conditions

- Stop successfully at the first passing design rung and freeze it.
- Stop blocked if no `K<=25` passes.
- Stop as an engineering-feasibility veto if the conservative pilot projection
  is over budget or a command repeats its timeout after one localized repair;
  do not classify either event as a failed `K`.
- Do not add features, widen the parameter box, change endpoints, or introduce
  a numerical floor inside this campaign.

## Phase 4: Held-Out Local Chart Audit

### Objective

Evaluate the frozen selected candidate at the predeclared held-out points
without retuning.

### Primary gate

Every time/point passes the binding finite-program finiteness, rank,
condition-roundoff, equality-residual, and computed-positivity checks.  The
deterministically selected weakest held-out case additionally passes the
corroborating high-precision sign-stability audit defined above.  A binding
finite-program failure, or an unstable weakest-case high-precision audit,
rejects the candidate; it does not authorize
trying the next `K` using the same held-out set because that would tune on audit
data.  A new capacity study would require a fresh held-out design.

## Phase 5: Original Failure And Own-Scalar Derivative Regression

### Objective

Execute `T=2,10,100,1000`, culminating in center plus the four exact `1e-5`
endpoints from Phase 4 of the predecessor campaign.

### Required gates

- all claim-bearing outputs finite;
- all chart certificates pass;
- exact feature matching;
- warm replay identity;
- manual score versus autodiff;
- manual score versus central FD under the existing FD-only policy;
- no first failure at time 748 or elsewhere;
- fixed program/preparation identity.

## Phase 6: Same-Target Scientific Diagnostic

### Objective

Because the overcomplete recursion defines a different finite scalar from the
square route, repeat the existing Actual-SV dense-reference value/score
comparison at `T=2,10,100` and the same adjacent teacher/continuation
quadrature refinement used by the predecessor where executable.  Report for
each parameter the candidate-minus-reference absolute and symmetric-relative
score differences, the value difference, and adjacent-refinement changes.

### Interpretation

Own-scalar derivative success is engineering evidence only.  No target-specific
equivalence margin has been justified, so this phase is descriptive and cannot
pass scientific equivalence or rank methods.  A large disagreement must remain
an explicit scientific gap and may motivate a separate feature-repair study;
it does not prevent the narrow `CE-07` chart/own-scalar repair classification.
No post-hoc accuracy threshold is allowed.

## Phase 7: Trusted GPU/XLA Certification

### Entry condition

Phases 0--5 pass and Phase 6 completes with honest descriptive artifacts.

### Execution

- trusted GPU/XLA, float64 reference candidate;
- hard TensorFlow logical-device memory limit `8192 MiB`, selected by the
  owner for this campaign, configured before any TensorFlow GPU initialization;
- record physical and logical GPU devices plus
  `logical_device_memory_limit_mib=8192` in the manifest, verify exactly one
  eligible logical GPU is exposed to the run, and fail closed if the logical
  configuration cannot be applied;
- do not enable or claim TensorFlow memory growth in this arm because it is
  mutually exclusive with the logical-device memory limit;
- fresh output root;
- compile/warm timing, graph topology, device placement, allocator peak,
  deterministic replay, and descriptive CPU/GPU differences;
- exception/failure artifacts for catchable TensorFlow failures.

### Stop conditions

Stop on invalid chart, OOM, native abort, wrong device, graph expansion, or
budget exhaustion.  GPU parity does not repair a scientific score failure.

## Phase 8: Terminal Review And Ledger Update

### Objective

Write a terminal result with separate engineering, numerical, and scientific
ledgers; run a post-run red team; review the result; update `CE-07` without
overclaiming.

### Decision cases

| Outcome | Ledger action |
| --- | --- |
| Chart, derivative, and GPU gates pass | Close `CE-07` as a local chart/own-scalar engineering defect; retain `CE-11` and record Phase 6 scientific differences without an equivalence claim. |
| Chart and derivative pass; Phase 6 shows a large descriptive mismatch | Close only the chart/own-scalar part of `CE-07` and add or retain a separate scientific score-accuracy gap. |
| Design or held-out positivity fails | Keep `CE-07`; record the rejected anchor/reference family. |
| Engineering implementation fails | Keep `CE-07`; record localized repair trigger and smallest reproducible failure. |

## Campaign Budget And Artifact Policy

- CPU/reference: at most 4 wall hours, 8 CPU core-hours, two TensorFlow
  intra-op threads, one inter-op thread, and one BLAS thread.
- Trusted GPU/XLA: at most 2 attempts and 1 hour total.
- Capacity ladder: exactly `K=5..25`, stopping at first design pass.
- One held-out evaluation only for the selected capacity.
- Phase 0/1 documentation and review: at most 0.5 CPU core-hours.
- Phase 2 implementation tests: at most 1.0 CPU core-hours and three localized
  test attempts.
- Phase 3 three-measurement pilot plus staged ladder: at most 5.0 CPU
  core-hours; `T=10` commands time out at 120 seconds, `T=100` at 600 seconds,
  and `T=1000` at 1800 seconds per rung.
- Phases 4--6 CPU audits: at most 1.5 CPU core-hours and two localized retries.
- Phase 7: at most two trusted GPU attempts, each with a 1800-second timeout,
  within the one-GPU-hour total.
- A timed-out command is an engineering/budget failure, never evidence that a
  capacity fails the mathematical design.  One localized retry is allowed
  after a recorded infrastructure or compilation repair within the same phase
  budget.  A repeated timeout triggers the engineering-feasibility continuation
  veto; it must not advance to another `K` as though positivity had failed.  If
  the pilot estimate implies the worst-case staged ladder exceeds the phase
  budget, revise the capacity-screening engineering before launch; do not
  silently expand compute.
- Every serious run writes a fresh versioned directory and records commit,
  dirty status, command, conda environment, CPU/GPU status, XLA/TF32/dtype,
  data/preparation hashes, deterministic inputs, wall time, output paths, plan,
  and result.
- Localized infrastructure repairs and retries are allowed inside this budget
  when target, features, points, criteria, hardware class, and scientific
  interpretation remain unchanged.

## Review And Execution Protocol

1. Run local skeptical mathematical/source review of this plan.
2. Request one bounded Claude read-only review of exactly this path.  Claude is
   advisory and cannot edit or authorize scientific claims.
3. Patch material findings visibly in this same plan and rerun focused review.
4. Execute Phase 0, write its result, then continue through the next eligible
   phase while no true continuation veto fires.
5. At each phase, run required checks, write a result/repair record, refresh the
   next phase if implementation evidence changes its details, and preserve all
   earlier attempts.

The user request to create, review, and execute this plan authorizes the bounded
local campaign.  No separate procedural launch token is required.  Platform
approval remains required for Claude and GPU commands.
