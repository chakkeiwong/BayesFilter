# Canonical LGSSM Balancing And Kalman Repair Plan

Date: 2026-07-17

Status: `EXECUTED_WITH_SCIENTIFIC_AND_RESOURCE_GAPS`

Campaign ID: `canonical-lgssm-balancing-kalman-repair-20260717`

Output root:
`docs/benchmarks/artifacts/canonical_lgssm_balancing_kalman_repair_20260717/`

## Research Intent Ledger

| Field | Frozen intent |
| --- | --- |
| Main question | After the canonical Contract E--Chol caller consumes a terminally balanced row-quotient plan and gates both marginals, does its fixed finite LGSSM value-and-total-score program agree with the differentiated Kalman oracle at the declared center, and is any remaining error shared with the no-reset weighted LEDH recursion or specific to Contract E? |
| Candidate | Canonical `contract_e_chol_v1` LGSSM route with the complete total derivative and an explicitly bound terminal balancing schedule. |
| Baseline | Same prepared LGSSM proposal, importance-weight recursion, data, particles, and random streams with reset disabled. |
| Oracle | Deterministic float64 differentiated Kalman observed-data likelihood in HMC coordinates. |
| Expected failure mode | Final row quotient disturbs the target column marginal; after that repair, finite-particle bias/variance in the shared proposal/weight recursion may dominate reset-specific error. |
| Promotion criterion | For a declared horizon and particle count, all active canonical resets pass both consumed-plan marginals and the simultaneous interval for relative value bias lies in `[-0.001,0.001]`, while every simultaneous componentwise HMC-gradient interval lies in `[-0.05,0.05]`. |
| Promotion veto | Any nonfinite claim-bearing output; invalid physical/chart/Cholesky/covariance-gap state; either consumed-plan marginal outside its predeclared tolerance; same-scalar derivative failure; source/preparation/identity drift; or any interval wholly outside a declared boundary. |
| Continuation veto | Broken oracle identity, inability to construct the same canonical finite program in both arms, corrupted artifacts, campaign budget exhaustion, or failure of every predeclared balancing candidate on the frozen design and audit sets. A failed accuracy candidate is not by itself a continuation veto when the next phase is designed to diagnose it. |
| Repair trigger | Mechanical caller omission, marginal-gate omission, derivative/program mismatch, or harness defect with unchanged scientific contract. |
| Explanatory diagnostics | Per-seed errors, paired reset-minus-no-reset loss, particle-count trend, covariance gap, condition proxies, runtime, compilation, and allocator telemetry. |
| Forbidden conclusion | No method superiority, HMC readiness, full parameter-region validity, nonlinear-model validity, leaderboard completion, or default promotion. Passing one center-scoped screen is not a region certificate. |

## Default And Assumption Audit

| Choice | Provenance and status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| `delta_value=0.001` | Owner-approved existing LGSSM boundary; reviewed default | A 0.1% center value bias is the frozen leaderboard criterion. | Could be too weak for downstream HMC; keep center-scoped and do not infer trajectory validity. |
| `delta_grad=0.05` componentwise | Owner-approved existing LGSSM boundary; reviewed default | Every score component matters, and every Kalman component is nonzero at the center. | Does not control accumulated HMC energy error; explicitly forbid HMC-readiness claims. |
| FD tolerance `0.05*sqrt(5)` | Owner-selected heuristic, implementation screen only | Detects gross disagreement between the score and its own finite scalar. | Could pass a scientifically inaccurate score; keep completely separate from Kalman accuracy. |
| 16 paired seeds | Owner-selected exploratory audit count | Improves on the old five-seed nominal design and supports paired common-random-number intervals. | Student assumptions and power remain weak; report screen status and interval uncertainty, not general equivalence or ranking. |
| Float64 first | Existing canonical/Kalman reference policy; reference default | Separates algorithmic disagreement from TF32/float32 resolution. | Does not establish production float32/TF32 behavior; leave that for a later plan. |
| GPU/XLA production-shaped execution | Repository default | The final serious route must compile and execute on the intended backend. | Compilation cost may dominate; perform CPU-hidden focused correctness checks before charged GPU nodes. |
| Balancing candidates `0,1,2,5,10,20,50,100` | New finite hypothesis ladder | Includes the historical omission, small corrections, and the SIR-proven upper endpoint without assuming SIR transfer. | The grid may omit the true smallest pass; select only the smallest grid pass and call it a frozen tested schedule, not a mathematical optimum. |
| Marginal-only schedule selection | Mathematical separation requirement | Prevents tuning transport iterations against Kalman output. | A marginal-valid schedule can still be scientifically inaccurate; Kalman remains a separate later gate. |
| Particle ladder `N=128,256,512`, conditional `1024` | New diagnostic ladder | Extends beyond the previously inconclusive `32,64,128` range while remaining bounded. | May still be too small or nonmonotone; use paired intervals at every rung and do not select an observed-best `N`. |
| Horizons `T=2,10,50` | Existing LGSSM contract | `T=2` makes reset affect a later likelihood; `T=10,50` test recursion and target row. | Prefix success may not extrapolate; the plan explicitly requires conditional advancement. |

## Evidence Contract

For arm `a`, estimator seed `s`, horizon `T`, and particle count `N`, define

```text
z[a,s,value] = (L[a,s] - L_Kalman) / abs(L_Kalman)
z[a,s,k] = (g_hmc[a,s,k] - g_hmc_Kalman,k) / abs(g_hmc_Kalman,k)
d[s,j] = abs(z[ContractE,s,j]) - abs(z[NoReset,s,j]).
```

Use two-sided Bonferroni Student intervals with familywise level 0.95 for the
six mean signed errors of each arm and, separately, the six mean paired
absolute-loss differences.  The analysis unit is the estimator seed.  The
model assumes independent/exchangeable seed streams, finite variance, and
Student marginal coverage.  With 16 seeds this has no power guarantee and is
not distribution-free.

Classification:

- `screen_pass`: all six candidate mean-error intervals are contained in their
  declared equivalence regions and all hard vetoes pass;
- `screen_fail`: at least one interval lies wholly outside its equivalence
  region or a hard veto fires;
- `inconclusive`: every other boundary result;
- reset effect per quantity is lower error, higher error, or inconclusive only
  from the sign of its paired-loss interval;
- no global reset ranking is emitted unless all six paired-loss intervals have
  the same nonzero direction.

These are center- and design-scoped results.  Even `screen_pass` is not a
trajectory-region, posterior, HMC, default, or superiority certificate.

## Phase 0: Canonical Wiring Repair

### Objective

Make `balance_steps` an explicit canonical LGSSM input, propagate it through
primal and total-JVP calls, expose post-quotient marginal histories, include
`marginal_valid` in reset validity, and bind the schedule into route/preparation
identity wherever canonical identity is issued.

### Entry Conditions

- Active ledger `CE-01` identifies the omission.
- Streaming forward/JVP/VJP primitives already implement the finite terminal
  balance program.
- No old v1 or experimental TP artifact may be upgraded.

### Required Artifacts And Checks

- Focused tests proving a canonical caller cannot silently use
  `balance_steps=0`.
- A negative test where positive mass and Cholesky checks pass but the consumed
  marginal gate fails closed.
- Primal/manual-JVP/forward-AD and FD identity after balancing.
- Route/preparation identity changes when `balance_steps` changes.
- CPU-hidden XLA smoke, syntax checks, and affected caller tests.

### Evidence Boundary

This phase establishes engineering correctness of the declared finite program,
not Kalman accuracy or scientific validity.

### Handoff And Stop Conditions

Advance only if the caller, derivative, validity, telemetry, and identity all
bind the same balancing schedule. Stop on any unresolved program mismatch.

## Phase 1: Marginal-Only Schedule Freeze

### Objective

Select the smallest candidate `balance_steps` passing both consumed-plan
marginals at all active resets on frozen LGSSM preparation-design cases, then
check it once on a disjoint audit set without retuning.

### Frozen Design

- `T=2`, `N=128`, float64, canonical XLA route;
- data seed `81100`;
- design estimator seeds `81300..81307`;
- audit estimator seeds `81320..81327`;
- candidates in order: `0,1,2,5,10,20,50,100`;
- ridge, epsilon, scaling, annealing steps, chunks, reset mask, and preparation
  construction remain the existing reviewed LGSSM values;
- selection uses only finiteness, chart/Cholesky/covariance-gap checks, both
  marginal residuals, repeat identity, and resource completion. Kalman values
  and scores are not computed or inspected in this phase.

### Promotion And Vetoes

Select the first candidate passing all design cases.  If the selected candidate
fails any audit case, classify the schedule as not validated and stop; do not
try a larger candidate after audit.  The roundoff-derived marginal tolerance is
the current implementation criterion under test and must be reported with raw
residuals; this plan does not claim a general Sinkhorn convergence theorem.

### Required Artifact

One JSON ledger containing every attempted candidate/design result, the frozen
selection, untouched audit result, exact inputs and source hashes, and explicit
confirmation that no Kalman output influenced selection.

## Phase 2: Paired Particle-Count Diagnostic

### Objective

At `T=2`, compare active Contract E with the no-reset weighted baseline using
common random numbers over increasing `N`, determining whether errors are
shared or reset-specific without selecting a favorable observed rung.

### Frozen Design

- estimator seeds `81400..81415` at every particle count;
- `N=128,256,512`; run `N=1024` only if `N=512` is inconclusive and budget
  remains, not because a particular component looks favorable;
- the Phase 1 schedule and all other numerical settings are immutable;
- both arms use identical prepared inputs except the predeclared reset mask;
- exact Kalman value and HMC score are computed independently once.

### Required Checks

- same-scalar manual JVP against forward AD on the smallest rung and central FD
  on representative seeds;
- exact common-input hashes across arms and particle-count-specific preparation
  hashes;
- per-rung simultaneous intervals for both arms and paired reset effects;
- both marginal gates on the active arm and no misuse of inactive reset
  sentinels in the baseline;
- no `N` is promoted as optimal from descriptive means or tails.

### Handoff

- If a shared error remains and neither arm passes, advance to the next
  predeclared `N`; this is a candidate failure, not a harness veto.
- If Contract E uniquely fails while no-reset passes, write a reset-specific
  scientific blocker and stop before longer horizons.
- If Contract E passes or is inconclusive without a reset-specific veto at the
  largest executed `N`, proceed conditionally to Phase 3 using that largest
  predeclared executed `N`, not an observed-best rung.

## Phase 3: Conditional `T=2,10,50` Kalman Certification

### Objective

Run the canonical center-scoped value-and-score screen through the declared
LGSSM horizons using the Phase 1 schedule and the Phase 2 handoff particle
count.

### Sequence

Run `T=2`, then `T=10`, then `T=50`.  Each horizon uses 16 fresh paired seeds:

```text
T=2:  81500..81515
T=10: 81520..81535
T=50: 81540..81555
```

Advance after a `screen_pass` or scientifically inconclusive boundary result
with no hard veto. A `screen_fail` is a model-candidate failure and stops the
longer ladder unless the next horizon was predeclared specifically to diagnose
accumulation; this plan does not make that exception, so it stops.

### GPU/XLA Requirements

Serious nodes run on the trusted RTX GPU with XLA enabled and an 8192 MiB
TensorFlow logical-device limit, no memory growth, fresh versioned outputs,
structured exception artifacts, replay checks, graph loop inspection, runtime,
and allocator telemetry. GPU setup and use require escalated/trusted execution.

## Phase 4: Terminal Synthesis

Write:

- `docs/plans/bayesfilter-canonical-lgssm-balancing-kalman-repair-result-2026-07-17.md`;
- `docs/plans/bayesfilter-canonical-lgssm-balancing-kalman-repair-reset-memo-2026-07-17.md`;
- an update to the active failure ledger that closes or preserves `CE-01` and
  `CE-02` exactly according to evidence.

The result must contain engineering, numerical, and scientific ledgers; the
decision and inference-status tables; commands, environment, commit/source
state, seeds, hardware, wall time, artifacts, and post-run red team.

## Campaign Budget

- At most two repair attempts for a localized implementation/harness failure in
  each phase, with prior artifacts preserved.
- At most eight Phase 1 candidate nodes plus one fixed audit node.
- At most four particle counts times two arms in Phase 2.
- At most three horizons times two arms in Phase 3.
- Per CPU/XLA node cap: 15 minutes; per GPU/XLA node cap: 30 minutes.
- Total serious GPU wall-clock allocation: four hours.
- Every retry uses a fresh output directory and records failure classification,
  repair, focused regression, wall time, and remaining budget.

## Skeptical Plan Audit

Status: `PASS_AFTER_REVISION`.

Material findings and repairs:

1. The first draft risked transferring SIR's `20+100` schedule as an LGSSM
   default.  Repaired by a finite LGSSM-specific marginal-only ladder and an
   untouched audit set.
2. The first draft risked using Kalman accuracy during balancing selection.
   Repaired by forbidding Kalman computation in Phase 1.
3. The historical one-seed `N=32,64,128` diagnostic cannot support the proposed
   mechanism decision.  Repaired by 16 paired common-random-number seeds at
   each new rung and simultaneous intervals.
4. Five-seed nominal intervals were too weak for an equivalence claim.  Repaired
   by the owner-selected count 16 and by retaining explicit Student-model,
   power, and center-scope limitations.
5. Same-scalar FD, marginal convergence, and GPU parity were proxy risks.
   Repaired by assigning them implementation-veto roles only; Kalman
   value/score intervals remain the scientific criterion.
6. A failed candidate could have been confused with an invalid experiment.
   Repaired by separate continuation vetoes and candidate-failure handoffs.
7. The plan could have selected an observed-best `N`.  Repaired by always using
   the largest predeclared executed rung for handoff.
8. The existing canonical caller contains Python horizon unrolling.  This plan
   does not silently treat that as production XLA compliance: graph size and
   loop structure are recorded, and Phase 3 cannot claim production readiness
   unless the claim-bearing route satisfies repository XLA loop policy.

The plan now answers the stated question without changing the target, using a
proxy as the primary criterion, or inventing a numerical threshold from
observed output.

## Execution Feasibility Amendment

Review date: 2026-07-17.

Phase 2 measured `N=1024,T=2` at `1118.88 s` for Contract E and `1096.92 s`
for no reset.  Therefore a `T=10,N=1024` node is projected to exceed the
frozen 30-minute per-node cap even before allowing for the larger unrolled XLA
graph.  This is a plan-feasibility defect, not evidence that the mathematical
candidate failed.

The scientific contract, particle count, horizons, seed blocks, and caps are
unchanged.  Execute the fresh-seed `T=2` pair because it remains feasible.  If
that pair permits advancement, classify `T=10` and consequently `T=50` as
resource-blocked under this campaign rather than silently lowering `N`, using
fewer seeds, changing chunks, raising the cap, or treating a timeout as a
scientific screen failure.  A later campaign may revisit the longer horizons
only after the canonical time loop is converted to the repository-required
fixed-state `tf.while_loop` body and a reviewed resource pilot supports the
same scientific design.

The required same-scalar FD check is separated into a one-seed `N=128,T=2`
implementation-only node using the previously reviewed binary64 dyadic step
`2^-17`.  Its seeds are not used in the Kalman intervals, and its
`0.05*sqrt(5)` decision cannot promote Kalman accuracy.
