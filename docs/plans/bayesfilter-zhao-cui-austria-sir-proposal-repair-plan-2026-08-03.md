# Experiment plan: Zhao-Cui Austria SIR proposal repair (exact-mixture candidate)

Date: 2026-08-03

Status: `PLANNED; AUDIT_PASSED_WITH_DISCLOSED_SCOPE_TENSION; NOT_LAUNCHED`

Stage 0 (engineering: missing tests plus a prefix issuer for an untested in-repo
route) is unblocked. Stage A screening awaits owner confirmation that an
*analytic* repair is in scope; see "Authority and scope reconciliation".

Supersedes as execution authority:
`docs/plans/bayesfilter-zhao-cui-austria-sir-score-completion-plan-2026-08-02.md`
(historical; stopped at T3). That plan remains the reference for the
mathematical target, the manual score derivation, and the source-support
ledger. Its launch sequence is not resumed.

Reboot authority:
`docs/plans/bayesfilter-zhao-cui-austria-sir-score-completion-reset-memo-2026-08-03.md`

## Question

Does the already-implemented but never-exercised **exact-density guide-mixture**
proposal
(`compile_austria_sir_rank_one_mixture_proposal_branch`) satisfy the *unchanged*
frozen proposal-quality gate at T1, T2, and T3 over the *unchanged* `0.03`
parameter box, where the rejected persistent nine-guide family failed at three
T3 points?

The decision this informs is narrow: whether a repaired proposal exists that
permits *planning* the T5/T10/T20 staged screens for the frozen Austria SIR
finite-score program. It does not inform any likelihood-accuracy, HMC,
posterior, default, or production question.

## Preserved scientific target (unchanged)

Value and manual total derivative of the repository-defined frozen
importance-filter scalar with event order

```text
x0 -> transition -> y1 -> ... -> transition -> y20
```

and parameters

```text
(log_kappa_scale, log_nu_scale, log_obs_noise_scale).
```

| Field | Value |
|---|---|
| `target_id` | `zhao_cui_austria_sir_seed81120_latent_preclip_y1_y20_v1` |
| `target_seed` | 81120 |
| Route classification | `extension_or_invention` |
| Claim dtype | `float64` |
| Agreement rule | `abs(a-b) <= 5e-6 + 5e-6*max(abs(a),abs(b))` |
| ESS gate | `ESS/N >= 0.10` |
| Max-weight gate | `maximum normalized weight <= 0.10` |
| Parameter box | half-width `0.03`, plus the unchanged `0.10/0.25/0.50` ladder rungs |
| Particles | 1008 |

Nothing in this table is renegotiated by this plan. The gate is not weakened,
the box is not shrunk, and the target is not replaced.

## Mechanism being tested

### What changed relative to the rejected candidate

The rejected route (`compile_austria_sir_persistent_guide_program`) runs **nine
independent filters**, filter `c` using one fixed guide `theta_c` for *all* `T`
steps, and admits a parameter point if **at least one** of the nine satisfies
both gates. The repair candidate
(`make_rank_one_mixture_branch_tensor_compiler`) runs **one** filter whose
per-step proposal is a `K`-component mixture over the same guide locations, with
component weights recomputed at every step and for every particle.

### Derivation D1: the plan-2026-08-02 `K^-T` rejection does not apply here

Line 412 of the August 2 plan states: *"Per-time component switching is rejected
because its matching-path mass decays as `K^-T`; scoring only the realized
component is also wrong."*

Reproducing that argument fairly: if a route draws a component `c_t` at each
step and then evaluates the proposal density as `q_{c_t}(x_t | x_{t-1})`, the
density is misstated, and the subset of particles whose entire component
sequence `(c_1,...,c_T)` is constant has probability of order `K^-T`. Any
estimator whose validity or efficiency depends on such matched paths degenerates
exponentially in `T`.

The implemented mixture route does not evaluate the realized component's
density. With ancestor index `a` drawn from the auxiliary law, it draws
`c ~ P(. | a)` where

```text
P(c | a)  proportional to  (1/K) * W_{t-1}^{c,a} * predictive(y_t | x_{t-1}^a, theta_c)
```

([lines 846-863](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L846-L863)),
samples `x_t` from `q_c(. | x_{t-1}^a)`, and then evaluates

```text
log q_t(x_t | a) = logsumexp_c [ log P(c | a) + log q_c(x_t | x_{t-1}^a) ]
```

([lines 925-927](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L925-L927)),
i.e. the **exact conditional mixture density marginalised over the component**.
The module's own manifest already records this as
`"mixture_density": "exact_logsumexp_of_all_component_conditionals"`
([line 1583](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L1583)).

The importance weight assembled by the shared recursion
([`_evaluate_source_order_core`, lines 1002-1010](bayesfilter/highdim/zhao_cui_predator_prey_fixed_variant_tf.py#L1002-L1010))
is

```text
log w_t^i = log W_{t-1}^{a_i} + log f_theta(x_t^i | x_{t-1}^{a_i})
            + log g_theta(y_t | x_t^i) - log alpha_{t-1}^{a_i}
            - log q_t(x_t^i | a_i)
```

and references `q_t`, never `q_{c_i}`. The event "component sequence is
constant along a path" therefore appears nowhere in the weight, the estimator,
or its variance, and no `K^-T` factor arises.

**Verdict: the line-412 argument is an argument against realized-component
scoring and does not apply to this implementation.** This is a statement about
which argument applies, not a claim that the mixture passes the gate.

### Derivation D2: bounded per-step downside, unbounded upside — and what it does *not* prove

All terms of the mixture sum are nonnegative, so for any component `c`,
`q_t(x|a) >= P(c|a) * q_c(x | x_{t-1}^a)`. Taking `c* = argmax_c P(c|a)` gives
`P(c*|a) >= 1/K`, hence uniformly in `x`

```text
log q_{c*}(x | x_{t-1}^a) - log q_t(x | a)  <=  -log P(c*|a)  <=  log K = log 9 ~ 2.197 nats.
```

So per step the mixture's log-weight can exceed that of a filter using its own
most-probable component by at most `log K` nats, *uniformly*. The rejected
persistent-guide route has no such `theta`-independent bound: it is locked to a
single `theta_c` for all `T` steps, so its per-step log-weight mismatch scales
with the guide-to-`theta` distance and accumulates in `t`. The observed
stratification below is consistent with exactly that.

**What D2 does not establish.** ESS is a functional of the *variance* of the
log weights across particles, and D2 bounds only a pointwise density ratio. D2
therefore does not prove that the mixture attains `ESS/N >= 0.10` at T3, still
less at T20. Mixture viability is classified `not checked` and is precisely
what this experiment measures.

### Derivation D3: theta-independence is preserved

The manual score recursion is valid only for a **frozen** program, i.e. one
whose proposal tensors carry zero derivative with respect to `theta`. The
mixture compiler's entry point takes exactly `(observations, seed)`
([lines 732-742](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L732-L742));
`theta` is not an argument. All guide-dependent quantities use the compile-time
constant `guides` tensor. By inspection the emitted branch is
`theta`-independent, so the frozen structure is preserved. This is an
inspection-level verdict and is re-verified empirically by test T-M1 below
rather than assumed.

## New diagnostic evidence motivating this candidate

Derived from the preserved T3 artifact only; **no new runs**. Stratifying the 23
design points by Euclidean distance `d` in the `(log_kappa_scale,
log_nu_scale)` plane to the nearest guide node (nodes at `{-0.03, 0, +0.03}^2`):

| `d` | Points | Best-guide `ESS/N` at T3 | Gate |
|---|---:|---|---|
| `0` | 15 | 0.990 - 1.000 | all pass |
| `~0.0106` | 2 | 0.353, 0.550 | both pass |
| `0.0150 - 0.0168` | 6 | 0.046 - 0.367 | 3 pass, 3 fail |

Two consequences:

1. **The observation-noise coordinate is not the binding deficiency at half-width
   `0.03`.** Every `theta_3 = +/-0.03` point that is on-node in `(kappa, nu)`
   attains `ESS/N ~ 0.99`. The block is a `kappa/nu` guide-lattice resolution
   failure at off-node interior points. This supports keeping the guide's
   observation-noise coordinate fixed at zero, and it orders the nine-component
   family ahead of the twenty-seven-component family in the candidate ladder.

2. **Per-step ESS decay is bimodal.** On-node points lose about 1% of ESS per
   step; off-node points lose far more. The worst point
   `(0.018, -0.021, -0.012)` runs `0.8056 -> 0.3565 -> 0.0457`, i.e. per-step
   ratios `0.443` and `0.128`. Under a geometric-decay extrapolation — an
   **unverified assumption from three horizons** — every currently-passing
   off-node point projects to fail the `0.10` floor well before T20; the best
   off-node passer (`ESS/N = 0.550` at T3, ratio `~0.72`) projects to about
   `1.4e-3` at T20.

**Honesty caveat.** This stratification was computed over the full 23-point
design, which includes the three failed points. The mechanism is nevertheless
independently visible in the 20 *passing* points alone (`d = 0` at `~0.99`
versus `d ~ 0.0106` at `0.35-0.55` versus `d ~ 0.015-0.0168` down to just above
`0.10`), so it does not depend on the holdout failures. The three failed points
are used below only as a pass/fail regression gate, never to choose or order a
candidate.

**Consequence for plan design.** A repair that only rescues T3 would very likely
move the block to T5. Extrapolation is not accepted as evidence here; instead
Stage B measures T5 directly, at negligible cost, once Stage A passes.

## Claim-status ledger for the mixture route

| Claim | Status | Basis |
|---|---|---|
| Evaluates the exact conditional mixture density | `correct` | [lines 925-927](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L925-L927); derivation D1 |
| Preserves frozen `theta`-independence | `correct by inspection`, re-verified by T-M1 | derivation D3 |
| XLA-native, single `tf.while_loop`, no host callback | `correct by inspection`, re-verified by T-M3 | `jit_compile=True, autograph=False` at [lines 732-739](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L732-L739); loop at [lines 986-1001](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L986-L1001) |
| Rejected by the `K^-T` argument | **does not apply** | derivation D1 |
| Previously screened and failed | **unsupported** | see below |
| Attains `ESS/N >= 0.10` at T3 | `not checked` | this experiment |
| Attains the gate at T20 | `not checked` | out of scope for this plan |
| Manual score equals autodiff of the same scalar on this branch | `not checked` | test T-M1 |

**On "previously screened and failed":** the August 2 plan (line 521) refers to
*"failed per-time mixtures retained as negative evidence."* A repository sweep
finds **no caller anywhere** of
`make_rank_one_mixture_branch_tensor_compiler` or
`compile_austria_sir_rank_one_mixture_proposal_branch` outside their own module
— no test, no script, no runner — and **no artifact under `docs/`** that
references the mixture compiler. There is therefore no preserved artifact
supporting that sentence, and it is recorded here as an **unsupported claim in a
prior AI-generated plan**. The mixture route's status is `not checked`, not
refuted. This plan does not delete the prior sentence; it corrects the record.

## Authority and scope reconciliation (disclosed tension)

The August 2 plan, immediately after the `K^-T` sentence, predeclares at line 413:

```text
If no persistent guide is viable at any T3 domain point, reject the analytic
proposal route.
```

That T3 failure occurred. Read broadly, that sentence rejects **all** analytic
(non-learned) proposals and would require the repair to be a learned
higher-rank squared-TT/KR proposal. Read narrowly, "the analytic proposal route"
names the persistent-guide analytic route then being screened. This plan
proposes an analytic candidate, so the reading matters and is disclosed rather
than assumed away.

This plan relies on the narrow reading, for three stated reasons.

1. The 2026-08-03 reset memo is the newer authority and is the designated reboot
   document. It explicitly lists as a plausible repair *"a predeclared broader
   persistent-guide family whose locations, branch count, and proposal seeds are
   selected without tuning on the three failed points"* — an analytic,
   non-learned family. The memo therefore does not treat the T3 failure as
   rejecting analytic proposals as a class.
2. The evidence actually obtained is specific to one nine-guide persistent
   family. Generalizing it to all analytic proposals is not supported by the
   artifact; the T3 measurements bear on guide-lattice resolution under a
   single-guide-locked filter, which is the mechanism derivation D1 and D2
   change.
3. The memo's binding prohibitions ("Do Not Do These Things") and its required
   verdict vocabulary contain no prohibition on an analytic repair. This
   candidate violates none of them.

Two further disclosures, so no reader has to reconstruct them:

- The exact-mixture candidate is **neither** of the memo's two listed
  hypotheses. The memo introduces them with "Plausible hypotheses are", which is
  illustrative rather than exhaustive, and its operative instruction is that the
  plan "must preserve the scientific target and classify defaults before
  choosing a candidate." This plan does both.
- If the broad reading is the owner's intent, this plan is out of scope and
  should be replaced by a learned higher-rank squared-TT/KR plan. That decision
  belongs to the owner, not to this document. Nothing here is irreversible: the
  plan is unlaunched and no artifact directory has been created.

## Research intent ledger

| Field | Content |
|---|---|
| Main question | Does the exact-mixture proposal pass the unchanged T1/T2/T3 gate on the unchanged `0.03` box? |
| Candidate under test | `kappa_nu_cartesian_9` exact-density mixture, `proposal_standard_deviation_scale` from a predeclared ladder |
| Mechanism | Per-step, per-particle component reweighting replaces a single guide locked for all `T` steps (D1, D2) |
| Expected failure mode | Off-node `kappa/nu` interior points still degenerate, because a `log K`-bounded per-step density floor need not bound log-weight *variance* |
| Promotion criterion | Every point of selection design `S` satisfies both gates at T1, T2, and T3 |
| Promotion veto | Any `S` point failing either gate at any of T1/T2/T3; any non-finite output; missing `While`; any host callback |
| Continuation veto | Harness invalidity: prefix identity failure, `theta`-dependence of the branch, manual-score/autodiff disagreement outside the frozen rule, NumPy or Python numerical path in the claim-owned route |
| Repair trigger | Promotion veto on candidate `i` advances to candidate `i+1` in the predeclared ladder, evaluated on `S` only |
| Explanatory diagnostics | Per-step `ESS/N` trajectory, per-step decay ratio, per-step max weight, log-weight spread, realized component-selection entropy |
| Must not be concluded | T20 viability, likelihood accuracy, HMC readiness, posterior correctness, default or production readiness, statistical superiority over the rejected route |

## Design partition (predeclared before any run)

The unit design is `_domain_design()`: 1 origin, 6 signed axes, 8 sign corners,
8 mixed interiors, in that order (23 rows), scaled by each ladder half-width.

| Set | Definition | Role |
|---|---|---|
| `S` (selection) | The 23-point design **minus** unit rows 15, 16, 21 -> 20 points | Sole basis for choosing and ordering candidates |
| `R` (regression) | Exactly unit rows 15, 16, 21, i.e. `(0.5,-0.25,0.75)`, `(-0.5,0.25,-0.75)`, `(0.6,-0.7,-0.4)` scaled by `0.03` | Pass/fail admission gate only |
| `U` (untouched) | A fresh design generated from predeclared seed `70413`, evaluated exactly once | Reserved for a later claim; **not run in this plan** |

Rows 15, 16, 21 are the three observed T3 failures (mixed rows 0, 1, 6).

**Anti-tuning rule.** Candidates are ordered and selected using `S` only. The
first candidate that passes `S` at T1/T2/T3 is the single candidate carried to
`R`. If it fails `R`, the plan stops with `BLOCK_REPAIRED_PROPOSAL_QUALITY`;
the ladder is **not** walked further. Walking the ladder against `R` would be
tuning on observed failure and is forbidden.

## Candidate ladder (predeclared, ordered)

| # | Guide family | `K` | `proposal_standard_deviation_scale` | Rationale for this position |
|---|---|---:|---:|---|
| 1 | `kappa_nu_cartesian_9` | 9 | 1.0 | Same guide locations as the rejected route, so the mixture mechanism is the single changed design variable. Scale 1.0 is the exact locally-optimal-infectious / exact-transition-susceptible proposal, i.e. no defensive inflation. |
| 2 | `kappa_nu_cartesian_9` | 9 | 1.25 | Smallest predeclared defensive inflation; tests whether residual failure is tail-lightness rather than lattice resolution. |
| 3 | `kappa_nu_cartesian_9` | 9 | 1.5 | Larger inflation; trades proposal sharpness for tail coverage. |
| 4 | `full_cartesian_27` | 27 | 1.0 | Adds the observation-noise guide coordinate. Placed last because the new diagnostic shows that coordinate is not the binding deficiency at half-width `0.03`, and because it costs 3x the components. |

Ladder membership, ordering, and the scale values are fixed by this document
before execution. No fifth candidate may be added without a new plan revision
recorded before running it.

## Implementation tasks (bounded, before Stage A)

The mixture route is implemented but unexercised. Four gaps must be closed, and
closing them is engineering work whose correctness is established by focused
tests, not by the experiment.

1. **Prefix issuer.** The mixture route has no `prefix()` facility, whereas the
   persistent-guide route has
   [`AustriaSIRPersistentGuideProgram.prefix`](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L243).
   The August 2 review required staged horizons to be **literal tensor
   prefixes** of one frozen T20 parent. Add a repository-owned prefix view for
   the mixture branch that slices the parent T20 tensors; do **not** rely on
   `tf.random.stateless_uniform` producing shape-prefix-consistent streams.
   The compiler's uniforms are already time-major
   (`[steps, particles]`, `[steps, particles, 18]`,
   [lines 751-763](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L751-L763)),
   so slicing is well defined.
2. **Single-branch gate path in the runner.** The existing runner consumes
   `ess_by_time_and_guide` / `maximum_weight_by_time_and_guide` and takes a
   best-of-nine maximum. The shared core already returns `ess_by_time`,
   `maximum_normalized_weight_by_time`, `minimum_ess`, and `finite`
   ([lines 1002-1010](bayesfilter/highdim/zhao_cui_predator_prey_fixed_variant_tf.py#L1002-L1010)),
   so add a single-branch path that applies the **same frozen thresholds** with
   no per-point selection.
3. **Focused tests for the mixture route** (currently zero): T-M1 manual score
   versus `GradientTape` on the same scalar plus increment additivity; T-M2
   prefix identity against the T20 parent; T-M3 graph audit (`While` present, no
   `PyFunc`/`EagerPyFunc`/`MapDefun`); T-M4 `theta`-independence of every emitted
   branch tensor.
4. **Run manifest schema** for the new output root.

### Gate-strength note

With nine branches the gate read "**there exists** a guide with `ESS/N >= 0.10`
and max weight `<= 0.10`". With one mixture branch it reads "**the** branch
satisfies both". Removing the per-point best-of-nine selection removes a
multiple-comparison advantage, so the mixture faces the screen with no per-`theta`
selection available to it. A mixture pass therefore cannot be attributed to gate
relaxation. This is a statement about the screen's structure, not a prediction
about the outcome.

## Scope

- Variant: exact-density guide-mixture APF proposal for the frozen Austria SIR
  source-order program
- Objective: proposal-quality screening only; no training, no optimizer, no
  learned component
- Seed(s): proposal seed `31201` (unchanged from the rejected run, so the seed
  is not re-selected after seeing failure); target seed `81120`; untouched-design
  seed `70413` reserved and unused in this plan
- Training steps: N/A (no learned component in this plan)
- HMC/MCMC settings: N/A (no sampler in this plan)
- XLA/JIT mode: `jit_compile=True`, `autograph=False`, FP64, TF32 disabled
- Device: single visible GPU under
  `owner_designated_managed_session_visible_gpu_trusted`, 6,144 MiB
  logical-device limit configured before GPU initialization
- Expected runtime: seconds per stage (see budget)

## Stages

| Stage | Content | Entry condition | Exit verdict |
|---|---|---|---|
| 0 | Implementation tasks 1-4; focused tests T-M1..T-M4 plus the existing 15 tests | none | all tests pass, else `BLOCK_REPAIRED_PROPOSAL_IMPLEMENTATION_OR_XLA` |
| A | T1, T2, T3 screens on `S` for ladder candidates in order; then the first `S`-passing candidate on `R` | Stage 0 pass **and** owner confirmation that an analytic repair is in scope (see reconciliation section) | one of the three required verdicts |
| B | T5 screen on `S` and `R` for the admitted candidate; per-step decay measurement | Stage A = `PASS_REPAIRED_PROPOSAL_T1_T2_T3` | `PASS_T5_DECAY_PROBE` or `BLOCK_T5_PROPOSAL_QUALITY` |

Stage B exists because the decay diagnostic shows extrapolation is not
trustworthy and T5 costs approximately six seconds. It is entered only after
Stage A passes, consistent with the memo's prohibition on running T5 before the
staged T1/T2/T3 contract passes. T10 and T20 remain out of scope and require a
further plan.

## Success criteria

- Stage 0: 15 existing focused tests plus T-M1..T-M4 all pass; T-M1 agreement
  within `abs(a-b) <= 5e-6 + 5e-6*max(abs(a),abs(b))`.
- Stage A promotion: for the selected candidate, every point of `S` satisfies
  `ESS/N >= 0.10` **and** max normalized weight `<= 0.10` at T1, T2, and T3,
  with all outputs finite.
- Stage A admission: the same candidate satisfies both gates at all three points
  of `R` at T1, T2, and T3.
- Stage B: the admitted candidate satisfies both gates on `S` and `R` at T5.

## Diagnostics

Primary (promotion criterion):
- Per-point minimum-over-time `ESS/N` at T1, T2, T3.
- Per-point maximum-over-time normalized particle weight at T1, T2, T3.

Secondary (explanatory only; may not promote or rank):
- Per-step `ESS/N` trajectory and per-step decay ratio, per point.
- Per-step maximum weight and log-weight spread.
- Realized component-selection distribution and its entropy, per step.
- Distance `d` to the nearest guide node, joined to each point's ESS.
- Peak allocator bytes and wall time.

Sanity checks (continuation veto if failed):
- Prefix identity: T1/T2/T3/T5 branch tensors are exact slices of the T20 parent.
- `theta`-independence: every emitted branch tensor has zero gradient with
  respect to `theta`.
- Manual score versus `GradientTape` on the same scalar, within the frozen rule.
- Increment additivity: total value equals the sum of per-step increments; total
  score equals the sum of per-step increment scores.
- Graph audit: `While` or `StatelessWhile` present; no `PyFunc`,
  `PyFuncStateless`, `EagerPyFunc`, or `MapDefun`.
- No NumPy numerical path and no Python numerical loop in the claim-owned route.
- Recorded FP64, TF32 disabled, verified memory policy.

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Promotion status |
|---|---|---|---|---|---|
| Exact-mixture proposal | Already in-repo, unexercised | D1 shows the prior rejection argument does not apply; D2 bounds per-step downside by `log K` | Bounds density, not log-weight variance; may still degenerate | Stage A T3 on `S` | **hypothesis** |
| `kappa_nu_cartesian_9` first | Same locations as the rejected route | Isolates the mixture mechanism as the single changed variable; the new diagnostic shows the obs-noise coordinate is not binding at `0.03` | Nine components may be too coarse off-node | Stage A T3 on `S` | **hypothesis** |
| `proposal_standard_deviation_scale = 1.0` first | Module default | Exact locally-optimal proposal, no defensive inflation, no hidden tuning | Tails too light | Candidate 1 T3 max weight | **baseline** |
| Scale ladder `{1.0, 1.25, 1.5}` | Chosen here, before any run | Smallest-first inflation; bounded to three rungs | Ladder too short to rescue | Candidate 3 result | **convenience choice, bounded and declared** |
| Proposal seed `31201` | The rejected run's seed | Reusing it avoids post-hoc seed selection to erase the failure | Single-seed Monte Carlo variation | Stage B; multi-seed deferred | **baseline, not a superiority basis** |
| 1008 particles | Rejected run | Holds particle count fixed so the proposal is the changed variable | Too few particles off-node | Per-step ESS trajectory | **baseline** |
| `ESS/N >= 0.10` | August 2 plan, anchored to the author `N/10` reapproximation trigger in `full_sol.m:53` | Frozen before any result | Threshold may be loose or strict for this route | none; frozen | **reviewed default, not renegotiable here** |
| Max weight `<= 0.10` | August 2 plan | Frozen before any result | as above | none; frozen | **reviewed default** |
| FP64 / TF32 off / XLA / 6144 MiB | `AGENTS.md` and August 2 review | Repository governance | none identified | manifest fields | **reviewed default** |
| `S` / `R` split | This plan | Prevents selecting a repair on observed failures | `S` may be easier than `R` in ways not detected | `R` gate at Stage A | **reviewed default for this plan** |
| Geometric ESS decay | My extrapolation from three horizons | Used only to justify adding Stage B | Would be wrong to treat as a T20 prediction | Stage B measures T5 directly | **explicitly not promoted; extrapolation replaced by measurement** |

## Expected failure modes

1. Off-node `kappa/nu` interior points still fail at T3: the mixture's per-step
   density floor does not control log-weight variance. Most likely outcome if
   the block is genuinely lattice resolution rather than proposal-tail weight.
2. Candidate 1 passes T3 but fails T5, confirming the decay hypothesis and
   moving the block one horizon later. Stage B is designed to detect exactly
   this.
3. Max-weight veto without ESS veto: one particle dominates because the mixture
   concentrates on a single component. The component-entropy diagnostic
   distinguishes this from broad degeneracy.
4. Stage 0 implementation failure: prefix slicing, `theta`-independence, or the
   graph audit fails. This is `BLOCK_REPAIRED_PROPOSAL_IMPLEMENTATION_OR_XLA`
   and is an engineering result, not evidence about the mixture mechanism.
5. Manual score disagrees with `GradientTape` on the mixture branch, indicating
   the branch is not in fact frozen. Continuation veto.

## Pre-mortem

**How this run could pass while misleading us.**

- *Passes T3, doomed at T20.* The dominant risk. Mitigated by Stage B measuring
  T5 rather than extrapolating, and by refusing any T20 claim here.
- *Passes because `S` excludes the hard points.* `S` excludes exactly the three
  observed failures, which are the hardest known points. Mitigated by requiring
  `R` to pass as an admission gate and by forbidding ladder-walking against `R`.
- *Passes on a lucky seed.* Single frozen seed `31201`. Not mitigated within
  this plan; recorded as a non-claim, and multi-seed replication is required
  before any ranking or default statement.
- *Passes because the mixture and the rejected route consume different random
  streams.* The mixture compiler splits its seed four ways versus three
  ([line 743](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L743)
  versus [line 1046](bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py#L1046)),
  so particle realizations differ. The gate is an absolute threshold rather than
  a comparison, so this does not invalidate admission, but it does mean a pass
  is **not** a controlled single-variable comparison against the rejected route.
  Recorded as a non-claim.
- *Passes with degenerate component usage.* The mixture could collapse onto one
  component and merely reproduce a single-guide filter that happens to be well
  placed. The component-entropy diagnostic exposes this.

**How this run could fail for implementation or tuning reasons rather than
scientific ones.** Prefix slicing errors, a `theta` leak into the branch, an
XLA graph regression, or too short a scale ladder. Stage 0 tests separate the
first three. The fourth is bounded and declared, and a ladder exhaustion is
reported as candidate rejection, not as evidence against the mixture direction.

## Skeptical pre-execution audit

Audited as a skeptical developer before any command is run.

| Check | Finding |
|---|---|
| Wrong baseline? | No. The comparator is the *unchanged frozen threshold*, not the rejected route's numbers. No ranking against the rejected route is claimed. |
| Proxy metric promoted? | No. The promotion criterion is the same `ESS/N` and max-weight gate frozen before the rejected run. Per-step decay is explicitly explanatory and gates only budget allocation to Stage B. |
| Missing stop conditions? | No. Ladder is four candidates; `R` failure stops immediately; T10/T20 out of scope. |
| Unfair comparison? | The mixture faces a *stronger* screen (no best-of-nine selection). Random streams differ, which is recorded as a non-claim rather than hidden. |
| Hidden assumptions? | Two surfaced and handled: geometric ESS decay (demoted to motivation for Stage B, replaced by direct T5 measurement) and `theta`-independence of the mixture branch (inspection-level, re-verified by T-M4). |
| Stale context? | Verified. All five source hashes and all four artifact hashes match the terminal manifest byte-for-byte; the five preserved artifact directories are present; `run_zhao_cui_austria_sir_frozen_score_claim.py` is absent, consistent with stopping before Phase 5. **Base commit advanced during planning:** the memo records `f3ca5aa990fa0997414359983da2e93be8bee40c`, and `HEAD` is now `efce62b5aaf5b540811286511905a7765efe952c` from a concurrent session ("Add repaired KSC gaussian-sum NeuTra route..."). `git diff --name-only f3ca5aa9..efce62b5` touches no Zhao-Cui or Austria file, and all four relevant source hashes are byte-identical, so every line citation in this plan is valid at `efce62b5`. |
| Environment mismatch? | Same conda prefix, Python 3.11.14, TF 2.19.1, TFP 0.25.0, FP64, TF32 off, 6,144 MiB cap. |
| Would the artifacts answer the question? | Yes. Per-point per-horizon `ESS/N` and max weight against frozen thresholds directly answer T1/T2/T3 viability. |
| Tuning on observed failure? | Prevented by the `S`/`R` split plus the anti-tuning rule forbidding ladder-walking against `R`. |
| Gate weakened or box shrunk? | No. Both frozen values and the full `0.03/0.10/0.25/0.50` ladder are carried over unchanged. |
| Post-hoc seed selection? | No. The rejected run's seed `31201` is reused deliberately. |
| Unsupported prior claim carried forward? | Caught. The August 2 plan's "failed per-time mixtures" sentence has no preserved artifact and is recorded as unsupported rather than inherited. |
| Predeclared scope honored? | **Tension disclosed.** August 2 plan line 413 predeclares rejecting "the analytic proposal route" on T3 failure. This plan relies on the narrow reading, justified against the newer reset memo in "Authority and scope reconciliation" above. If the owner intends the broad reading, this plan is out of scope. |
| Candidate within the memo's listed hypotheses? | No, and disclosed. The memo's two hypotheses are illustrative ("Plausible hypotheses are"); its operative requirement is target preservation plus default classification, both satisfied. |

**Audit verdict: PASS WITH DISCLOSED SCOPE TENSION.** One material defect was
found and repaired during the audit: the first draft relied on geometric
extrapolation to argue T20 infeasibility, which would have promoted a three-point
extrapolation into a research decision. It is replaced by Stage B measuring T5
directly.

One material tension was found and is **not** repaired because it is not the
agent's decision: whether August 2 line 413 rejects analytic proposals as a
class. Stage 0 is engineering work that is useful under either reading (it adds
missing tests and a prefix issuer to an untested in-repo route), so Stage 0 may
begin. **Stage A should not begin until the owner confirms an analytic repair is
in scope**, since a broad reading would make Stage A's screening effort
misdirected rather than merely negative.

## Compute and attempt budget

Reference wall times from the preserved artifacts: preflight 13.98 s, T1 4.91 s,
T2 5.21 s, T3 5.63 s, each a full 4-rung x 23-point sweep; peak allocator
100,834,816 bytes.

| Item | Budget |
|---|---|
| Stage 0 focused tests | 3 attempts, <= 5 min total |
| Stage A screens | <= 4 candidates x 3 horizons, <= 15 min total |
| Stage B screens | 1 candidate x 1 horizon, <= 5 min |
| Infrastructure repair attempts | <= 5, each with a fresh versioned output directory |
| Total campaign | <= 45 min GPU wall time |

A failed candidate or repair attempt consumes budget, not authority. Stop for
new direction only if budget is exhausted, a continuation veto fires, or the
scientific contract would have to change.

## Output root

```text
docs/plans/artifacts/zhao-cui-austria-sir-proposal-repair-20260803/
```

One fresh subdirectory per launch (`stage0-tests-01`, `stageA-t1-cand1-01`, ...).
The five preserved directories under
`docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/` are
read-only inputs and must not be deleted, overwritten, or reused.

## Command

Stage 0 (CPU-hidden focused tests; GPU intentionally hidden):

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run \
  -p /home/chakwong/anaconda3/envs/tf-gpu \
  python -m pytest -q \
  tests/highdim/test_zhao_cui_austria_sir_fixed_variant_tf.py \
  tests/highdim/test_zhao_cui_austria_sir_rank_one_proposal_tf.py
```

Stage A (trusted GPU; one invocation per horizon, `--candidate` walking the
declared ladder):

```bash
/home/chakwong/anaconda3/bin/conda run -p /home/chakwong/anaconda3/envs/tf-gpu \
  python scripts/run_zhao_cui_austria_sir_mixture_proposal_screen.py \
  --horizon 3 --particle-count 1008 --seed 31201 --candidate 1 \
  --design selection \
  --output-dir docs/plans/artifacts/zhao-cui-austria-sir-proposal-repair-20260803/stageA-t3-cand1-01
```

Exact commands are recorded in each stage manifest as run.

## Interpretation rule

- If Stage 0 fails: `BLOCK_REPAIRED_PROPOSAL_IMPLEMENTATION_OR_XLA`. This is an
  engineering result and carries no information about the mixture mechanism.
- If a continuation veto fires (prefix identity, `theta`-dependence,
  score/autodiff disagreement, NumPy or Python numerical path):
  `BLOCK_REPAIRED_PROPOSAL_IMPLEMENTATION_OR_XLA`, and the harness is repaired
  before any further screening.
- If every ladder candidate fails `S` at T1, T2, or T3:
  `BLOCK_REPAIRED_PROPOSAL_QUALITY`. The exact-mixture candidate family is
  rejected for this box; the *research direction* is not rejected, and the next
  discriminating artifact is a refined `kappa/nu` lattice or an XLA-native
  higher-rank squared-TT/KR proposal with target-specific L1 tuning on disjoint
  calibration and validation data.
- If a candidate passes `S` but fails `R`: `BLOCK_REPAIRED_PROPOSAL_QUALITY`.
  Stop; do not advance the ladder.
- If a candidate passes both `S` and `R` at T1, T2, and T3:
  `PASS_REPAIRED_PROPOSAL_T1_T2_T3`. This permits Stage B and, separately,
  *planning* T10/T20 in a future plan. It does not establish T20 viability,
  likelihood accuracy, HMC readiness, posterior correctness, default readiness,
  or superiority over the rejected route.
- If Stage B passes: record `PASS_T5_DECAY_PROBE` and report the measured
  per-step decay ratios as descriptive evidence only.
- If Stage B fails: record `BLOCK_T5_PROPOSAL_QUALITY`. Stage A's verdict stands
  as reported; the T20 program remains blocked.

## Non-claims

- No T20 score-completion claim.
- No exact physical-likelihood claim.
- No source-faithful Zhao-Cui Austria parameter-score claim; the author Austria
  example fixes `kappa` and `nu` and sets parameter dimension `d = 0`.
- No HMC, posterior, default, or production-readiness claim.
- No statistical superiority claim over the rejected persistent-guide route: a
  single frozen seed, differing random streams, and threshold-based screening
  cannot support a ranking.
- No claim that passing T3 implies passing T5, T10, or T20.
