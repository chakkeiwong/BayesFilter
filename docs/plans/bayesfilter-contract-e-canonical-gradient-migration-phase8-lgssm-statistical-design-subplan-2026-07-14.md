# Phase 8 Subplan: LGSSM Statistical Design And Oracle Ladder

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `BLOCKED_HUMAN_DECISION_AND_IMMUTABLE_CAMPAIGN_BUDGET`

## Phase Objective

Define, freeze, and then execute a paired LGSSM oracle ladder that distinguishes
same-program derivative correctness from scientific agreement with the exact
Kalman observed-data likelihood. The ladder proceeds from CPU float64
reference/debug rungs to trusted GPU/XLA/TF32 rungs and, only after every design
gate passes, the `d=3,T=50,N=10000` multi-seed comparison.

## Entry Conditions

- Phases 0-7 are closed at their stated narrow gates.
- Contract E--Chol is the only canonical-eligible route; v1/raw routes remain
  historical diagnostics and cannot contribute to a pass.
- The production v2 factory remains empty; Phase 8 evidence cannot self-admit.
- The continuation clock remains `2026-07-14T01:32:19+08:00` through
  approximately `2026-07-14T09:32:19+08:00`; do not reset it.
- The owner accepted `0.1%` relative value bias for the specified `d=3,T=50`
  case. No Kalman-gradient equivalence margin has been accepted.
- The original Phase 5 canonical full-filter module hard-coded `tf.float64`.
  The Phase 8 shared-core repair now exposes float32 and float64 from the same
  numerical cores and passes the dtype-specific exact/XLA checks recorded in
  the Rung 0A dtype-repair result. This clears the hard-coded-dtype blocker, but
  it does not establish full `T=50` feasibility or Kalman agreement.

## Human Decision Before Final Results

No universal statistical construction determines a scientifically meaningful
gradient-equivalence margin. A 95% confidence interval controls repeated-sample
coverage for the chosen estimand; it does not define how large an error is
acceptable for HMC. Likewise, the owner-directed `0.05*sqrt(p)` rule is only an
individual-direction/coordinate FD implementation screen.

Before any `T=50,N=10000` Contract E value or gradient is observed, including a
pilot/canary, the owner must choose or approve a practical gradient error budget
and near-zero scale in the exact future HMC coordinates. The coordinates are

```text
u = (atanh(phi1), atanh(phi2), atanh(phi3), log(q_scale), log(r_scale))
```

and the compared gradients are obtained by the chain rule,

```text
g_u = ((1-phi1^2) g_phi1, (1-phi2^2) g_phi2,
       (1-phi3^2) g_phi3, q_scale g_q, r_scale g_r).
```

The recommended mixed componentwise scale is:

```text
e_k = mean_seed(g_LEDH,k - g_Kalman,k)
s_k = max(abs(g_Kalman,k), g_floor,k)
r_k = e_k / s_k
```

The value estimand uses `abs(L_Kalman)` directly after verifying that the oracle
value is nonzero; no value floor is introduced. One simultaneous 95% family
covers the relative value difference and all five `r_k`. Equivalence requires
the value interval to lie inside `[-0.001,+0.001]` and every gradient interval
to lie inside `[-delta_grad,+delta_grad]`. Each `g_floor,k` must be specified
from an HMC/numerical scale before primary-shape pilot results; it cannot be
fitted to make a component pass.

The amendment must also freeze disjoint pilot seeds, an ordered audit-seed pool,
pilot exclusion from inference, the simultaneous interval algorithm and random
seed, and a deterministic audit-count selection function before Rung 3. That
function must include the assumed alternative/bias used for power, multiplicity
adjustment, conservative pilot-variance rule, minimum and maximum audit counts,
power target, audit-pool prefix rule, and compute cap. Rung 3 may supply only the
pilot variance/runtime inputs to this frozen function; it cannot introduce
discretion. The amendment must also freeze any pilot tuning grid, its selection
statistic, deterministic tie-break and no-selection rule, and how selection is
accounted for when pilot variance enters the audit-count function. The selected
audit-pool prefix is then frozen before Rung 4. If the
owner does not select the margin and HMC-coordinate near-zero scale, this phase
stops after lower-shape design/harness work with a resumable blocker.

## Required Artifacts

- a frozen statistical-design amendment recording the owner margin/scale,
  transformed coordinates, simultaneous interval, pilot seeds, ordered audit
  pool, deterministic count function and bounds, or a fully specified valid
  sequential design, and compute ceiling;
- a dtype-repair result proving the full-filter float32 callable shares the
  canonical numerical core and preserves float64 Phase 5 certificates;
- a Phase 8 run manifest with exact code/prepared-input/observation hashes;
- a structured JSON result per rung and seed;
- a consolidated paired result with value and each gradient component;
- same-program FD, active-set, moment, row/column residual, Cholesky/ridge,
  memory, runtime, and device diagnostics;
- a Phase 8 result or blocker result; and
- a Phase 9 subplan only if the exact handoff conditions pass.

Every launch uses a fresh versioned output directory. Prior evidence is never
overwritten.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific question | At the declared LGSSM target and fixed finite Contract E program, are value bias and total-gradient bias small enough under the separately frozen practical equivalence regions? |
| Exact comparator | TensorFlow float64 Kalman observed-data log likelihood and autodiff gradient using identical observations, parameterization, transition-first timing, and stationary initial law |
| Engineering derivative criterion | Same callable center-primal identity plus the FD screen `0.05*sqrt(p)` with the Phase 1 step ladder, near-zero absolute FD scale, endpoint-collapse check, and identical branches/randomness; FD is not the Kalman criterion |
| Value promotion criterion | For `d=3,T=50`, the value member of one six-output simultaneous 95% family lies within `[-0.001,+0.001]` |
| Gradient promotion criterion | Every transformed-coordinate gradient member of the same simultaneous family lies inside the owner-approved `[-delta_grad,+delta_grad]` under the frozen mixed scale |
| Hard vetoes | Route/prepared-input mismatch; raw/v1 contribution; center-primal mismatch; nonfinite value/gradient; invalid chart; branch mismatch; failed same-program FD; missing diagnostics/artifact; hard-coded float64 mislabeled TF32; wrong GPU/XLA/TF32 provenance at production rungs |
| Explanatory diagnostics | Runtime, memory, Sinkhorn residuals, ridge, condition proxy, raw covariance residual, per-seed scatter, prefix drift, and historical raw/no-reset negative controls |
| Not concluded | Exact nonlinear filtering, posterior correctness, HMC readiness, superiority, universal numerical settings, v2 admission, leaderboard completeness, default readiness, or release readiness |

## Research Intent Ledger

| Field | Binding intent |
| --- | --- |
| Main question | Does the canonical finite LEDH estimator agree with the exact LGSSM oracle within a pre-result practical error budget? |
| Candidate | Contract E--Chol canonical value and total gradient |
| Expected failure | Finite Sinkhorn/reset/ridge bias or Monte Carlo scatter persists despite repaired derivative wiring |
| Promotion criterion | Both value and all five gradient equivalence intervals pass; engineering vetoes remain clear |
| Promotion veto | Any hard veto above |
| Continuation veto | Invalid comparator/target/data/artifact, unresolved owner margin before primary shape, failed shared-core dtype repair, exhausted campaign clock, or inadequate remaining budget for the frozen replication plan |
| Repair trigger | Lower-rung implementation, serialization, XLA, memory, branch, or diagnostic failure that can be repaired without changing target/design/budget |
| Explanatory diagnostic | Prefix/rung/seed trends and componentwise errors |
| Must not conclude | A failed candidate rejects Contract E as a research direction or a passed LGSSM result proves nonlinear/HMC readiness |

## Estimands And Statistical Design

The finite program is evaluated in physical coordinates
`(phi1, phi2, phi3, q_scale, r_scale)`, but the primary gradient comparison is
in the declared unconstrained HMC coordinates
`(atanh(phi1), atanh(phi2), atanh(phi3), log(q_scale), log(r_scale))`. For seed
`s`, record paired differences using the exact same observations and target:

```text
d_value,s = (L_contract_e,s - L_kalman) / abs(L_kalman)
d_grad,s,k = (g_contract_e,u,s,k - g_kalman,u,k)
             / max(abs(g_kalman,u,k), g_floor,u,k)
```

The observation sequence is fixed once from dataset seed `81100`; inference is
conditional on that sequence, not across newly simulated LGSSM datasets. An
estimator seed independently determines stateless TensorFlow base noise under a
frozen RNG algorithm and domain-separated tags: initial noise, every transition
time, and every realized residual-design time. For a seed, those tensors are
bitwise identical across value, score, autodiff where feasible, and every FD
center/endpoint. A particle, time step, FD endpoint, or repeated evaluation is
not a replicate.

The Kalman oracle is deterministic for the fixed observations. The primary
estimands are the audit-seed means of `d_value` and each `d_grad,k`. Report every
seed and one frozen simultaneous 95% interval family over all six means. The
default proposed construction is a Bonferroni family of Student intervals with
per-output two-sided alpha `0.05/6`, explicitly conditional on approximate
normality of seed-level differences; a frozen studentized max-statistic
bootstrap may replace it only if its algorithm, bootstrap seed, and minimum
audit count are fixed in the amendment. Resampling units are complete audit
seeds only.

The primary audit uses a fixed seed count. No repeated conventional-CI optional
stopping is allowed. The count is computed after a disjoint primary-shape pilot
by the already frozen deterministic selection function; pilot seeds are excluded
from every promotion interval. The function selects a prefix of the already
ordered audit pool and cannot be revised after the pilot. If a group-sequential
alternative is chosen, the amendment must instead name maximum seeds, look
schedule, alpha allocation/confidence-sequence construction, and compute budget.
If the remaining campaign clock cannot support the selected audit count, write a
resumable budget handoff rather than reduce replication.

The amendment must freeze the interval method's applicability check and failure
branch before any pilot or audit output is observed. If Student intervals are
selected, it must state the scientific justification for their minimum audit
count and a pilot-only diagnostic for the seed-level approximation; failure of
that diagnostic blocks promotion or invokes only a fully predetermined robust
alternative. If a bootstrap or sequential construction is selected, its own
minimum-count, resampling, and failure rules must be frozen instead. Audit data
may never be used to choose among interval methods or rescue a failed
applicability check.

## Baseline Ladder

| Rung | Execution | Role | Advancement condition |
| --- | --- | --- | --- |
| 0A | CPU-hidden tiny fixtures | Refactor/instantiate a shared-core float32 canonical callable and freshly recertify both dtypes | Fresh source-bound float64 and float32 gates below pass; Phase 0-6 compatibility passes; no scientific promotion |
| 0B | CPU-hidden float64 tiny frozen fixture | Recheck callable, oracle, serialization, same-scalar FD, and active-set diagnostics | All engineering vetoes clear; no scientific promotion |
| 1 | CPU-hidden float64 `T=1`, then `T=10`, increasing feasible `N` | Localize time/reset bias and compare manual score, autodiff where feasible, FD, and Kalman | No branch/chart/identity veto; trends recorded descriptively |
| 2 | Trusted GPU/XLA/TF32 float32 `T=1`, then `T=10`, increasing `N` | Establish compiled/device parity and bounded memory on the actual production dtype | Device/dtype provenance correct; no hard veto; prior evidence preserved |
| 3 | Trusted GPU/XLA/TF32 `T=50,N=10000`, frozen disjoint pilot seeds | Estimate runtime and audit-count variance after margins are frozen; may tune only predeclared candidate settings on pilot seeds | Pilot is permanently excluded from audit inference; candidate/settings and fixed audit count are frozen before Rung 4 |
| 4 | Trusted GPU/XLA/TF32 `T=50,N=10000`, untouched frozen audit seeds | Primary paired equivalence result | Both value and all five gradient members of the simultaneous family pass for any Phase 9 scientific handoff |

Historical raw reset and no-reset arms may be retained only as explanatory
negative controls. They cannot pass or replace the canonical candidate.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Exact Kalman comparator | LGSSM mathematics and checked TensorFlow oracle | Exact observed-data target | Timing/parameterization mismatch | Rung 0 identity audit and oracle autodiff/FD | Required |
| `N=10000,T=50,d=3` target | Owner campaign scope | Required leaderboard-scale case | Feasibility mistaken for accuracy | Separate canary and paired equivalence rungs | Frozen target |
| `0.1%` value margin | Owner decision | Accepted for specified case | Reused beyond fixture | Fixture identity in artifact | Frozen only for stated case |
| Gradient margin | Open owner decision | Must reflect practical HMC/numerical use | Arbitrary post-result threshold | Margin amendment before Rung 3 | Blocker |
| HMC-coordinate near-zero scale | Needed for components near zero | Relative error alone is unstable | Large floor can conceal error | Freeze component scales before Rung 3 | Blocker with margin |
| Simultaneous interval | Joint value-plus-five-gradient decision | Prevents separate marginal families from masquerading as joint evidence | Too few seeds, unsupported approximation, or invalid resampling unit | Freeze one six-output family, its applicability check, and blocking/predetermined fallback rule before observing pilot output | Required |
| Audit replication | Primary uncertainty evidence | A deterministic post-pilot fixed count avoids optional-stopping coverage loss | Underpowered, discretionary, or unaffordable audit | Before pilot freeze ordered audit pool and full power/count function | Required before Rung 3 |
| Pilot tuning | Optional target-specific numerical selection using pilot seeds only | Prevents inherited tiny-fixture settings from silently becoming defaults | Post-result grid expansion, subjective selection, or ignored selection variance | Freeze candidate grid, statistic, tie/no-selection rule, and count-function treatment before pilot | Required if tuning is used |
| Existing Sinkhorn/ridge/chunk settings | Historical/tiny prepared routes | They are candidate hypotheses, not target-specific defaults | Cross-shape transfer can fail for the wrong reason | Predeclare pilot-only tuning grid and numerical selection rules; never use audit seeds | Unreviewed hypotheses |
| Canonical dtype | Phase 5 module hard-codes float64 | Production policy requires float32/TF32 | Float64 GPU result mislabeled TF32 | Shared-core dtype repair and tiny parity | Active engineering blocker |
| TF32/XLA/GPU | Repository default target | Required production execution | CPU/float64 evidence mislabeled; graph drift | Trusted device manifest and CPU/GPU rung comparison | Required at production rungs |

## Skeptical Plan Audit

Decision: `PASS_FOR_REVIEW_REPAIR_AND_DTYPE_HARNESS_WORK_ONLY; PRIMARY_SHAPE_EXECUTION_BLOCKED`.

- Kalman is the scientific baseline; same-scalar FD is an engineering check and
  cannot promote oracle agreement.
- The `0.05*sqrt(p)` FD screen and historical `1%` rule are explicitly barred
  from becoming Kalman margins.
- Confidence coverage and acceptable effect size are separated; a 95% interval
  does not create an equivalence region.
- No primary-shape pilot or audit result can be inspected before the owner
  margin, HMC-coordinate scale, pilot/audit split, and inferential method are
  frozen.
- Pilot seeds are quarantined from audit inference; the plan states whether
  every seed is pilot or audit before launch.
- A fixed audit count removes conventional optional stopping; one six-output
  family covers value and all gradient components.
- Seed-level pairing and explicit stateless noise domains prevent particle/time
  pseudoreplication and mismatched FD randomness.
- The current full-filter float64 module cannot support a TF32 claim; shared-core
  float32 implementation and parity are explicit prerequisites.
- Lower-rung candidate failure triggers localization/repair unless it invalidates
  the target, comparator, data, artifact, or campaign budget.
- The campaign clock, fresh output roots, and explicit nonclaims prevent an
  incomplete run from being relabeled as promotion evidence.

## Pre-Mortem

The run could pass misleadingly if a generous near-zero floor hides a component,
separate intervals are treated as one joint pass, pilot seeds leak into the
audit, conventional intervals are repeatedly checked, a float64 graph is called
TF32, or the margin is chosen after seeing primary-shape output. The design
amendment, seed-level pairing, one simultaneous family, fixed audit count,
pilot/audit quarantine, and dtype repair address those risks.

The run could fail for engineering rather than scientific reasons through XLA
compilation, serialization, resource limits, or a missing diagnostic. Those are
repair triggers within the unchanged campaign contract. Scientific equivalence
is decided only by the frozen simultaneous regions; no qualitative sign or
"order-one" rule may override them after results.

## Dtype Repair Evidence Gate

The Rung 0A gate is frozen before repair results:

1. Fresh float64 certificates bind the new source/dependency hashes, retain
   `0 ULP` manual-JVP versus forward-autodiff on the frozen Phase 5 fixture, and
   reproduce the prior v2 center objective and score hexadecimal values exactly.
   The old source-bound certificates remain historical and are not relabeled.
2. Static source/AST inspection proves float64 and float32 factories invoke the
   same `_canonical_primal_core` and `_canonical_manual_jvp_core`; no copied
   dtype-specific numerical core is allowed. Dtype differences may affect only
   tensor dtype, constants, and required casts. There may be no dtype-conditioned
   algorithm, cadence, branch policy, reset semantics, or derivative
   composition inside or below those shared cores.
3. Fresh float32 eager checks require `0 ULP` manual-JVP versus forward-autodiff
   for per-batch and aggregate scores on the frozen tiny fixture. This is an
   equality between two traversals in the same dtype, not a scientific forward-
   error claim.
4. The float32 CPU-XLA callable must use one concrete value-and-score function,
   return finite values/scores, repeat bitwise at the center, preserve the eager
   center branch hash exactly, and keep every FD endpoint on the same branch.
   Its same-callable coordinate FD uses the existing `0.05*sqrt(5)` relative
   implementation screen plus the Phase 1 step/near-zero/collapse rules.
5. Float32-versus-float64 value/score differences are serialized and classified
   explanatory only. No cross-dtype adequacy tolerance is invented at Rung 0A;
   production adequacy remains the Kalman oracle decision under the frozen Phase
   8 regions.

The shared-core, zero-ULP manual-JVP/forward-autodiff agreement,
XLA-repeatability, branch, and explanatory cross-dtype portions of this gate
passed in the Rung 0A repair record. Later Phase 8 work ran a representable
seven-step ladder on the frozen fixture: all 35 cases passed the owner-directed
FD-only heuristic. A rigorous callable-error-bound FD certificate is
unconditionally `unsupported` because the required TensorFlow/XLA absolute
callable error bounds are absent. That formal certificate is retired as a
nonclaim rather than retained as an actionable phase blocker.

## Required Checks And Reviews

1. Audit exact model/data/timing/physical and HMC coordinates against the Kalman
   oracle before adding a new harness.
2. Refactor or instantiate the canonical full-filter callable from a shared
   numerical core for float64 and float32. Execute the frozen Dtype Repair
   Evidence Gate and issue fresh source-bound certificates before any production
   rung.
3. Draft the frozen margin/near-zero/pilot-seed/ordered-audit-pool/power/count-
   function/simultaneous-interval amendment and obtain owner approval before Rung 3 or
   any `T=50,N=10000` candidate command. The amendment must include the interval
   applicability/failure rule and, if pilot tuning is enabled, the exact grid,
   statistic, deterministic tie/no-selection rule, and treatment of selection in
   the audit-count calculation.
4. Build the smallest structured Rung 0B harness; run CPU-only with
   `CUDA_VISIBLE_DEVICES=-1` and record that choice.
5. Execute lower-shape Rungs 0B-2 sequentially with a fresh output root per attempt and
   repair only localized failures under the unchanged scientific contract.
6. Run trusted/escalated GPU commands for Rungs 2-4 and record device, dtype, TF32,
   XLA, memory, and wall time.
7. After Rung 3, freeze the selected numerical candidate and apply the already
   frozen count function to select an audit-pool prefix; never reuse pilot seeds
   in Rung 4 or revise the function.
8. Consolidate paired audit-seed results, intervals, hard vetoes, descriptive
   diagnostics, and nonclaims.
9. Obtain one terminal scientific/result review if material; reviewer
   unavailability is not a procedural blocker under current policy.
10. Write the Phase 8 result and draft Phase 9 only if handoff conditions pass.

## Forbidden Claims And Actions

- Do not invent, silently reuse, or data-fit a Kalman-gradient margin.
- Do not use `0.05*sqrt(p)` or `6%` actual-SV reference error as the LGSSM
  oracle-gradient criterion.
- Do not treat a 95% confidence interval as an effect-size justification.
- Do not run or inspect a primary-shape pilot before the complete margin,
  coordinate, noise, seed-split, interval, and power design is frozen.
- Do not use pilot seeds in the audit family or repeatedly inspect conventional
  audit intervals to choose when to stop.
- Do not inspect pilot output before freezing the candidate grid, selection
  statistic, tie/no-selection rule, variance treatment, and interval
  applicability/failure branch; do not use audit output to select or repair an
  interval method.
- Do not call a float64 full-filter execution TF32 evidence.
- Do not promote one-seed, lower-rung, CPU-only, FD-only, moment-only, or
  feasibility evidence.
- Do not register/admit v2, run HMC, migrate nonlinear rows, regenerate the
  leaderboard, or claim release readiness in Phase 8.
- Do not fall back to any raw-barycentric route.

## Exact Next-Phase Handoff Conditions

Phase 9 may begin only if the frozen Rung 4 design completes, the value member
of the six-output simultaneous family passes `0.1%`, all five transformed-
coordinate gradient members pass the owner-approved margin, same-program FD and center identity pass, every
numerical/device/artifact veto is clear, and no raw route contributes. If the
canonical candidate fails only its scientific equivalence region, write a
candidate-blocker result and do not migrate/regenerate nonlinear rows as if
LGSSM were cleared.

## Stop Conditions

Stop before another target numerical arm if the owner numerical-design
requirements are absent, and stop before any primary-shape execution if the
owner margin/design decision is absent. The seven-step same-callable FD-only
heuristic has passed; its rigorous callable-error-bound certificate is
unsupported and is not an active execution gate.
Stop the phase for invalid comparator/target/data, corrupted or unreproducible
prepared inputs, missing required diagnostics, a persistent nonlocal code/math
mismatch, exhausted campaign clock, or insufficient remaining budget for the
frozen seed design. A localized harness/XLA/serialization failure is a repair
trigger, not by itself a stop condition.
