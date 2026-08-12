# GenUT Four-Model NeuTra Readiness Plan

Date: 2026-08-04

Status: `EXECUTED_WITH_THREE_MODEL_ADMISSION_AND_AUSTRIA_VETO`

Models: LGSSM, KSC-SV, Austria SIR, predator-prey

Artifact root:
`docs/benchmarks/artifacts/genut_four_model_neutra_readiness_20260804/`

## Objective And Boundary

Close the engineering and numerical gaps needed to try a real, batch-native
NeuTra optimizer update against the finite GenUT posterior for each model.  A
model passes only if its exact frozen finite value program, its total forward
derivative, posterior prior/chart terms, status telemetry, GPU/XLA execution,
and one optimizer update all pass their declared gates.

This campaign does **not** run a serious NeuTra training, select a learned
transport, tune HMC, or draw posterior samples.  “NeuTra-update ready” below
means eligible for that later target-specific training campaign.  It does not
mean HMC-ready, converged, posterior-correct, or default-ready.

## Research Intent Ledger

| Field | Frozen intent |
|---|---|
| Main question | Can the current finite GenUT program provide genuine leading-batch posterior values and scores, with bounded time memory, that the repository NeuTra trainer can consume for all four requested targets? |
| Candidate | A TensorFlow batch-native forward-JVP GenUT recurrence, with no sample-row mapping and the current Contract-E Chol reset plus selected diagonal higher-moment correction. |
| Exact baseline | The existing scalar `finite_value_score` evaluated with the same observations, fixed particle noises/design, controls, arithmetic, and posterior terms. |
| Expected failure mode | GPU memory grows as `B*N^2`; Austria branch sensitivity/replay can invalidate rows; KSC `T=1000` may be computationally infeasible; target-chart boundaries may create invalid points. |
| Promotion criterion | Per-model pass of value/score/status parity, same-scalar finite differences, replay, GPU/XLA batch capacity with `B>1`, and one finite NeuTra optimizer update. |
| Promotion veto | Any score not equal to the derivative of the same finite value program; row-mapped scalar execution; nonfinite/invalid required row; failed posterior recomposition; missing memory growth/GPU/XLA; failed replay; no finite optimizer update. |
| Continuation veto | Shared batch core fails scalar parity after a focused repair, GPU capacity cannot execute `B=2`, or the bounded campaign budget expires.  A model-specific failure blocks that model, not the other models. |
| Repair trigger | Shape/XLA/harness error, missing status field, localized transpose error, or stale target binding with unchanged target and budget. |
| Explanatory diagnostics | Runtime, allocator peak, score magnitude, per-step residual traces, descriptive distance to an external filter, and batch-size throughput. |
| Must not be concluded | No posterior accuracy, stochastic superiority, HMC convergence, transport quality, or production/default readiness follows from this campaign. |

## Frozen Posterior Targets

All targets are deterministic conditional on the recorded fixed GenUT noises
and residual design.  Training and later endpoint evaluation must use the same
finite target identity.

| Model | Data and likelihood | Unconstrained coordinate, prior, and Jacobian | Scope controls |
|---|---|---|---|
| LGSSM | Identifiable diagonal `m=3,T=50` benchmark, dataset seed `81100`, parameterized finite GenUT likelihood in physical `(phi1,phi2,phi3,q_scale,r_scale)` | Five independent probit maps onto the declared benchmark box `(-0.95,0.95)^3 x (0.05,2)^2`; uniform physical box prior; complete probit log-Jacobian. This is the prior declared in `scripts/filtering_value_gradient_benchmark_emit_source_paper_scope.py`, not the unrelated 18-dimensional exact-LGSSM NeuTra target. | `N=1008`; July `lgssm_T50` tuning selected `2/8/8/1e-5` and diagonal higher moments `4/0.2/1e-5`; the new batch route must revalidate it. |
| KSC-SV | Seven-component KSC transformed-observation finite GenUT likelihood on frozen seed `81101`, `T=1000` raw observations transformed by `log(y^2+1e-8)` | Existing two-probit map to physical `(gamma,beta) in [0.1,0.9]^2`; independent uniform physical prior; complete two-probit log-Jacobian. | `N=1008`. Historical `T=10` settings are **only a warm start**; a `T=1000` tuning artifact is required. |
| Austria SIR | Parameterized `d=18,T=20`, frozen `y1:y20` observation hash `cd794ad6...0f07`, fixed GenUT noises/design | Existing identity chart in three log-scale coordinates and independent `Normal(0,0.5^2)` prior; zero chart Jacobian. | `N=1008`; current-source tuning artifact `genut_austria_antithetic_ensemble_20260803/tuning_attempt01/result.json`, selecting `epsilon=8`, `8/8`, ridge `1e-5`, diagonal higher moments `4/0.2/1e-5`. |
| Predator-prey | Additive-Gaussian RK4 model, frozen seed `81104`, `T=20`, initial observation before the first transition | Existing six-probit map to the Zhao-Cui parameter box, independent uniform physical prior, complete six-probit log-Jacobian. | `N=1008`; July `predator_prey_T20` tuning used transition-then-observe event order, so its settings are only a warm start and this event-order scope requires retuning. |

The LGSSM chart is a reviewed target-definition choice, not an empirical
finding.  It converts the already-declared proper benchmark-box prior into a
smooth unconstrained posterior and prevents invalid stationary/scale values.

## Evidence Contract

### Primary pass criteria

For each model:

1. The posterior adapter accepts `theta[B,p]` and returns `value[B]`,
   `score[B,p]`, and row-aligned status tensors in one graph-native call.
2. No Python sample loop, `tf.map_fn`, `tf.vectorized_map`, `tf.while_loop`
   over posterior rows, scalar callback, or scalar fallback is used.
3. At valid points, batch rows agree with independent scalar finite GenUT
   evaluations using the identical finite program.  FP32 tolerances are
   `2e-4` relative for value and `2e-3` relative for score unless the plan's
   diagnostic records a tighter model-derived scale.
4. Central finite differences of the same batched posterior value agree with
   the returned score: maximum symmetric relative error `<=0.05`, with at
   least two step sizes showing a stable plateau and no branch/status change.
5. Same-process replay is exact.  Two fresh GPU processes must agree to
   `2e-6` relative for value and `2e-5` relative for score and must make the
   same validity decision.  This is deliberately not bitwise cross-process
   equality because GPU reduction ordering is not guaranteed.
6. Trusted GPU/XLA runs use FP32 GenUT tensors, TF32 enabled, verified memory
   growth, `B>=2`, and record allocator current/peak bytes.  The largest
   successful capacity rung is descriptive; only `B=2` is required here.
7. The repository batching guard accepts the adapter and one real
   `PlainDenseIAFTransport` optimizer update is finite, advances the optimizer
   step, consumes `B>1`, and records no scalar or row-mapped fallback.
8. A tangent-free leading-batch value endpoint exists and agrees with the
   value returned by the value/score route.  This is required for later
   endpoint Metropolis evaluation and for measuring the score-computation
   cost separately.

### Vetoes and diagnostics

- `program_valid`, finite value/score, positive reset factors, Sinkhorn
  marginal validity, and posterior-domain validity are hard vetoes.
- A changed validity branch anywhere in a finite-difference stencil vetoes
  that stencil; it may not be reported as score agreement.
- Any Austria cross-process failure repeats once with TF32 disabled as an
  explanatory localization.  This does not silently change the target; if
  FP32-no-TF32 passes, a new arithmetic-scope tuning artifact is required
  before promotion.
- Runtime, peak memory, and throughput are explanatory only.
- LGSSM Kalman agreement is an additional numerical diagnostic and veto for a
  gross target error; it is not a substitute for same-finite-program parity.

### Preserved evidence

The campaign writes fresh attempt directories containing `result.json`,
`run_manifest.json`, replay payloads, per-model parity/FD/capacity rows, and a
terminal result note.  Every serious manifest records Git commit, exact
command, conda environment, TensorFlow version, GPU, XLA/TF32/dtype, memory
policy, data hashes, fixed-noise hashes, controls/tuning artifact, seeds, wall
time, plan/result paths, and any retry classification.

## Implementation Plan

1. Add batched model callbacks.  Extend the GenUT adapter contract with
   value/JVP callbacks operating on `theta[B,p]`, particles `[B,N,d]`, and
   tangents `[B,N,d,p]`.  Implement the four adapters with tensor broadcasting,
   not posterior-row mapping.
2. Add a batch-native Sinkhorn value/JVP kernel over `[B,N,N]` and
   `[B,N,N,p]`, preserving the scalar finite iteration counts, row quotient,
   validity checks, and diagnostics.
3. Add a batch-native Contract-E reset wrapper.  The repository reset forward
   is already batch-native.  Its directional JVP will flatten the internal
   `(B,p)` pair only for the reset call, then restore `[B,N,d,p]`; this is
   simultaneous tensor algebra, not sample-row iteration.
4. Add only the selected batch-native diagonal third/fourth-moment
   correction.  Pairwise/projected variants remain unsupported in this route
   because their experiments did not justify inclusion.
5. Add two recurrences: tangent-free batch value/status, and bounded-memory
   batch forward-JVP value/score/status.  Reverse-mode autodiff through time is
   rejected because KSC `T=1000` would retain the complete transport history.
6. Add repository-issued posterior adapters and stable target signatures for
   the four frozen targets.  The score is the total derivative of likelihood,
   prior, and chart Jacobian in the unconstrained coordinate.
7. Add focused CPU-hidden mechanics/parity tests at small exact-policy
   fixtures, then trusted GPU/XLA tests at the real `N=1008` scopes.
8. Run a capacity ladder `B=2,4` and stop at the first OOM/resource failure.
   No batch size is promoted as a universal training default.
9. Run same-scalar FD, scalar-row parity, posterior recomposition, same- and
   cross-process replay, then one optimizer update per passing model.
10. Emit a per-model readiness matrix.  Later serious NeuTra training and HMC
    are authorized only for cells that pass every gate.

## Default And Assumption Audit

| Choice | Provenance/status | Why used | Failure mode | Earliest diagnostic |
|---|---|---|---|---|
| `N=1008` | Existing GenUT model comparisons; baseline, not universal default | Common requested accuracy/cost scope | `B*N^2` exceeds memory or score remains unstable | `B=2,T=1` capacity smoke, then full horizon |
| FP32+TF32 | Repository DPF/GenUT production direction; reviewed default | Expected GPU throughput | Cross-process reduction/branch instability | replay gate; one FP32-no-TF32 localization for Austria only |
| Batch `B=2,4` | Capacity hypotheses | Smallest legal NeuTra batch and one growth rung | Too small for useful training or OOM | allocator peak and one update; no quality claim |
| Diagonal HM `4/0.2/1e-5` | Current selected GenUT controls | Current finite value target | Cross-model or horizon transfer is stale | per-scope tuning identity and scalar parity |
| KSC `T=1000` | Established KSC NeuTra target | User asked for KSC-SV, not the historical `T=10` diagnostic | Runtime makes GenUT training impractical | bounded `T=10,100,1000` timing/tuning ladder |
| One optimizer update | Engineering smoke only | Tests the actual trainer interface | Could pass despite unusable optimization or posterior | explicitly no training/HMC claim |
| LGSSM probit box chart | Declared benchmark prior, reviewed target-definition choice | Proper smooth unconstrained posterior | Prior/chart could dominate or differ from another desired scientific prior | exact recomposition and manifest binding |
| Fixed GenUT noise/design | Required deterministic finite target | NeuTra/HMC requires one stable target | Seed/design-specific approximation | record identity; no broad approximation claim |

## Skeptical Plan Audit

Audit date: 2026-08-04. Verdict: `PASS_AFTER_REVISION`.

Material findings and revisions:

1. **Wrong KSC scope:** available GenUT controls were tuned at `T=10`, not the
   requested `T=1000`.  They were downgraded to a warm start and a `T=1000`
   scope gate was added.
2. **Memory-invalid derivative design:** reverse-mode differentiation of the
   full value recurrence would retain `T` transport graphs.  It was replaced
   by bounded-memory forward JVP.
3. **False batching risk:** wrapping `finite_value_score` with a map would
   violate the NeuTra batching policy.  The plan now requires leading-batch
   tensor algebra in every shared and model callback.
4. **Hidden endpoint cost:** the prior zero-force smoke computed and discarded
   tangents.  A true tangent-free value endpoint is now a required gate.
5. **Unfair posterior comparison:** prior GenUT artifacts were likelihood-only
   points.  Exact priors, charts, and Jacobians are now frozen per model.
6. **Stale LGSSM target risk:** the existing 18-dimensional exact NeuTra target
   is not the five-parameter GenUT target.  The declared five-dimensional
   benchmark-box prior/chart is bound instead.
7. **Proxy promotion:** one optimizer update cannot prove training or HMC
   quality.  The campaign language and promotion boundary now stop at
   “eligible to try target-specific NeuTra training.”
8. **Austria shared-failure ambiguity:** Austria failure blocks only Austria
   unless LGSSM or a focused shared primitive demonstrates the same defect.

The baseline is the exact same scalar finite program, criteria are target
aligned, failure branches are recorded, stop conditions are explicit, and the
artifacts answer the stated engineering-readiness question.  No material
unexamined default remains; uncertain settings are explicitly hypotheses with
early diagnostics and nonclaims.

### Execution Audit Addendum: LGSSM Kalman Comparator

The initially suggested `_kalman_value_score` helper predicts before every
observation.  The frozen NeuTra target instead observes the stationary initial
state at `t=0` and transitions only before `y_1,...,y_49`.  Calling that helper
unchanged would compare different likelihoods and could falsely veto the
candidate.  The final diagnostic therefore uses a target-aligned TensorFlow
Kalman recurrence, checks it against the preserved exact source oracle at the
data-generating parameter, and adds the identical uniform-box prior and probit
Jacobian in NeuTra coordinates.

This remains a gross-error veto, not an accuracy ranking.  Before execution,
the deliberately loose thresholds are frozen as absolute posterior-value error
`<=10`, maximum absolute posterior-score error `<=10`, and positive score
direction cosine.  These thresholds can expose a wrong dataset, event order,
chart, or parameterization while allowing the known one-seed `N=1008` GenUT
approximation error.  Passing them must not be called Kalman equivalence.

Attempt 02 reproduced the preserved source-oracle likelihood to `7.4e-8` but
failed an auxiliary `1e-9` equality tolerance that had no float32-boundary
justification.  This is a reference-harness tolerance defect, not a target or
candidate failure.  The auxiliary replay tolerance is repaired to `1e-6`,
while all three predeclared gross-error thresholds above remain unchanged.

### Terminal Scope-Tuning Audit Addendum

The first terminal audit found that LGSSM numerical revalidation had passed but
the target still bound a historical scalar-route tuning artifact and reported
`scope_tuned_current_scalar_source_batch_revalidation_required`.  Under the
current per-scope tuning policy this is insufficient for serious NeuTra
training.  LGSSM is therefore added to the same deterministic, disjoint
calibration/validation tuner used for KSC-SV and predator-prey.  Its final
claim, replay, capacity, Kalman, and one-step-training artifacts must all be
rerun against the resulting exact-scope artifact; the earlier LGSSM passes
remain engineering history and cannot establish eligibility.

### Terminal Scalar-Parity Audit Addendum

The implementation unit suite checks batch/scalar parity only on small
fixtures.  That is insufficient for the plan's per-model real-scope claim.
For LGSSM, KSC-SV, and predator-prey, a final diagnostic will therefore run
the independent scalar `finite_value_score` at the claim center with the exact
same fixed data, particles, design, controls, chart, prior, and Jacobian.  The
predeclared real-scope tolerances remain `2e-4` relative for posterior value and
`2e-3` relative for posterior score.  The scalar route is diagnostic only and
is forbidden as a NeuTra training fallback.

### LGSSM Arithmetic-Scope Repair

The real-scope TF32 diagnostic failed the frozen score-parity tolerance: value
relative error was `6.0e-6`, but maximum posterior-score relative error was
`2.21e-2` at `B=2` and `1.70e-2` at `B=1`, versus `2e-3`.  The mismatch
persisted when higher-moment correction was disabled, so it is not cross-row
interference or solely the higher-moment extension.  The identical `B=1`
diagnostic with TF32 disabled passed.  Therefore the TF32 LGSSM scope is
rejected for NeuTra eligibility under this campaign.

FP32 without TF32 is a repair candidate, not an automatic promotion.  Because
arithmetic mode is part of the tuning scope, LGSSM must be freshly retuned and
must repeat claim, derivative, replay, `B=4`, Kalman, real-scope scalar parity,
and one-step training gates with TF32 disabled.  KSC-SV and predator-prey stay
on their independently passing TF32 scopes.  This model-specific arithmetic
exception does not change the repository-wide default or establish a speed or
accuracy ranking.

## Budget, Attempts, And Stop Conditions

- Engineering: focused implementation and tests, at most two localized repair
  retries per shared primitive.
- GPU campaign: at most two capacity rungs per model, one FD/parity run, two
  fresh replay processes, and one optimizer update per model.
- Tuning: one bounded warm-start grid per stale model scope, at most eight
  arms; KSC uses the `T=10,100,1000` timing ladder and stops if projected full
  grid cost exceeds the remaining budget.
- Total serious GPU wall-time budget: 90 minutes.  Stop an individual command
  at 20 minutes.  Never overwrite an attempt directory.
- Stop immediately on GPU memory-policy failure, absent GPU/XLA, output-root
  reuse, target/tuning identity mismatch, `B=2` OOM, shared scalar-parity
  failure after focused repair, or total budget exhaustion.

## Execution Commands

Commands are implemented by the campaign harness and recorded verbatim in its
manifest.  The intended ladder is:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/highdim/test_cubature_genut_batch.py \
  tests/test_genut_neutra_targets.py

TF_FORCE_GPU_ALLOW_GROWTH=true python \
  docs/benchmarks/run_genut_four_model_neutra_readiness.py \
  --output-root \
  docs/benchmarks/artifacts/genut_four_model_neutra_readiness_20260804/attempt01
```

The second command is a trusted GPU/XLA command and must use the repository's
escalated GPU execution policy.
