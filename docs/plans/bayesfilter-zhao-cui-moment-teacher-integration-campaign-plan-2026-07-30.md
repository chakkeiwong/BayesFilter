# Zhao-Cui Moment-Teacher Integration And Feasibility Campaign

Date: 2026-07-30
Status: executed; LGSSM and predator-prey passed, Austria SIR tuning veto closed the campaign
Route under test: `zhao_cui_moment_teacher_gpu_fp32_no_tf32_xla_v1`
Classification: `extension_or_invention`
Parent plan: `docs/plans/bayesfilter-zhao-cui-moment-teacher-plan-2026-07-30.md`

Terminal result:
`docs/plans/bayesfilter-zhao-cui-moment-teacher-integration-campaign-result-2026-07-31.md`.

## Research Intent Ledger

| Item | Predeclared answer |
|---|---|
| Main question | Can the independently fitted squared-TT recursion supply higher-moment targets to the canonical particle/OT/Contract-E finite program while preserving a correct same-program analytical score and remaining executable on the selected GPU route? |
| Candidate | Canonical Contract E-Chol particle filter plus graph-native TT fit, carried marginal, TT shape contractions, and explicit-target bounded correction. |
| Baseline | The same prepared particle program with empirical-target Contract E and no TT target substitution. Kalman is an additional value/score oracle only for LGSSM. |
| Expected failure mode | TT fit/chart invalidity, incomplete TT or particle tangent, scope-mismatched tuning, excessive XLA resource use, or nonlinear TT rank/fit failure. |
| Promotion criterion | Same-program derivative parity passes; LGSSM `T=2,10,50` value and score pass their predeclared Kalman/empirical-target checks; selected GPU/XLA graph is finite and contains no host callback. |
| Promotion veto | Any value/score mismatch, missing derivative term, wrong likelihood owner, invalid canonical reset, missing or mismatched tuning artifact, non-finite output, forbidden runtime backend, or corrupted artifact. |
| Continuation veto | Invalid harness/target, inability to issue canonical identity for the actual route, exhausted repair budget, or an upstream gate that remains failed after its bounded repairs. A candidate accuracy failure alone is not a research-direction veto, but it blocks later claim runs. |
| Repair trigger | Local trace/shape/XLA/tuning/serialization failure with unchanged scientific target and budget. Preserve the failed attempt and retry in a new output directory. |
| Explanatory diagnostics | TT fit residual and condition, represented normalizer, shape residuals, transport marginal residuals, allocator peak, compile/runtime, and descriptive nonlinear baseline differences. |
| Nonclaims | No HMC readiness, posterior correctness, statistical superiority, source-faithful Zhao-Cui filter, global default change, or broad nonlinear validity follows from this campaign. |

## Evidence Contract

The claim target is the total analytical derivative of the exact finite scalar
returned by the hybrid particle/TT program. The particle normalized-weight
increment remains the only likelihood increment. The TT scale shift and TT
normalizer never enter that scalar. The TT lane affects later particle
increments only through the explicit shape correction and carried corrected
cloud.

The exact baseline is the existing canonical Contract E-Chol program using the
same observations, frozen innovations, reset mask, residual design, ridge,
Sinkhorn schedule, balance schedule, dtype, TF32 mode, XLA mode, particle count,
and chunk policy. Kalman value and score are an oracle only for LGSSM and do not
replace comparison with that finite-particle baseline.

Hard veto diagnostics are finite-program derivative parity, canonical reset
validity, finite outputs, fixed-branch identity, exact tuning-scope match,
required chunk policy, GPU memory-growth verification, XLA execution, and the
absence of `PyFunc`/`EagerPyFunc`. Fit residuals and timing are explanatory
unless the frozen tuning protocol assigns a threshold before claim execution.

Artifacts are written under unique attempt directories rooted at
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_integration_20260730/`.
Every serious result records the Git commit, dirty source hashes, command,
environment, GPU/memory policy, dtype/TF32/XLA settings, prepared-data ID,
seeds, tuning artifact and scope digest, chunk selection, wall time, and output
paths.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
|---|---|---|---|
| FP32, TF32 off, GPU, XLA on | Selected route; attempt 12 | Passed full-teacher FP32/FP64 mechanics while TF32 failed and saved only 11% | Rerun integrated FP32/FP64 parity and inspect graph/device before model claims |
| Contract E-Chol streaming reset | Repository policy; canonical baseline | Only reset eligible for canonical score claims | Factory identity and reset-validity tests fail closed |
| Exact-divisor chunks | Repository policy | Required active DPF geometry | Configuration-time selector and mismatch test |
| Fixed TT rows, bases, ranks, schedule | Hypotheses requiring scope-specific tuning | Required for a differentiable fixed branch | Calibration grid records fit validity/conditioning; claim consumes frozen selection |
| Fixed ridge and correction controls | Hypotheses requiring scope-specific tuning | No cross-scope universal setting is assumed | Calibration/validation selection with untouched claim preparation |
| One direction per TT replay, stacked over parameters | Implementation choice | Existing verified TT JVP is directional; the parameter count is static | Compare every stacked coordinate against centered finite differences of the same value program |
| Batch size one for nonlinear feasibility | Deliberate descriptive scope | The plan asks for one seed; avoids silently claiming multi-seed inference | Manifest and result label differences descriptive only |
| `N>1000` nonlinear particle count | Parent plan requirement | Tests non-fixture feasibility | Use policy-issued chunk and report allocator peak; stop on resource veto |

No setting selected at `T=2` transfers as tuned evidence to `T=10` or `T=50`.
Likewise, neither LGSSM settings nor one nonlinear model's settings are tuned
for another model.

## Skeptical Plan Audit

The audit checked wrong baselines, proxy promotion, missing stop conditions,
unfair comparisons, hidden defaults, stale context, environment mismatch, and
whether each command answers its stated question.

Findings and repairs:

1. The existing TT-only GPU gate cannot answer end-to-end score correctness.
   It remains mechanics evidence only; this campaign adds a same-program
   particle/TT derivative gate.
2. `tt_particle_contract_e_step_reference_jvp` uses an eager historical reset
   helper. It is forbidden here. Integration starts from
   `ledh_contract_e_canonical_lgssm_tf` and the repository streaming
   Contract E-Chol core.
3. `ledh_contract_e_tp_tf` is a different projection algorithm, not canonical
   Contract E-Chol. It is not the integration baseline.
4. A frozen teacher tangent would compute a different score. Each parameter
   coordinate replays TT fitting, carried marginalization, whitening, shape
   targets, correction, and subsequent particle increments.
5. A single inherited control set would violate per-scope tuning policy.
   Each horizon/model/count/backend scope gets a repository-issued tuning
   record selected on calibration/validation preparations and frozen before
   its claim run.
6. One-seed nonlinear differences cannot support ranking. They are feasibility
   diagnostics only and run only after LGSSM validity passes.
7. HMC is not included. Its fixed adapter, proposal behavior, convergence, and
   posterior gates require a later plan after this finite-program route passes.

Audit verdict: `PASS_FOR_STAGED_EXECUTION`. The first executable phase is a
small deterministic integration fixture. Later phases are conditional and
must not run across a failed upstream hard veto.

### Audit Amendment Before Nonlinear Execution

The original audit missed one material ambiguity: it required the LGSSM value
and score to pass an MCSE-aware Kalman check but did not state the numerical
continuation threshold. This was discovered after the frozen six-seed LGSSM
claims had run and before any nonlinear adapter or experiment was executed.
The existing LGSSM results therefore are not represented as a prospectively
calibrated unbiasedness or promotion test.

For the narrower decision to continue to one-seed descriptive nonlinear
feasibility, the repaired screen is:

- every claim row passes the predeclared finite, reset, tuning-scope,
  mean/covariance, GPU/XLA, TF32-off, and no-host-callback hard gates;
- every horizon/coordinate has at least one positive and one negative error
  among the six frozen seeds; and
- for the six value/score quantities at each of the three horizons,
  `abs(mean error) / MCSE <= 3`.

The last rule is a feasibility anomaly screen, not an equivalence margin. Under
an ideal normal approximation, a two-sided three-standard-error screen across
18 checks has union-bound familywise false-alarm probability at most about
`18 * 0.0027 = 0.0486`. Six seeds make the MCSE estimate noisy, so passing does
not prove zero bias. The earlier `0.5 MCSE` criterion applies only to paired
TF32 numerical displacement relative to FP32-no-TF32 and is not reused for a
finite-particle estimator-versus-Kalman comparison.

Audit-amendment verdict: this repaired screen may authorize Phase 5 only. It
cannot support method ranking, default readiness, posterior correctness, HMC
readiness, or a claim that the estimator is unbiased.

## Implementation And Test Phases

### Phase 1: Canonical Composition

Add a repository-owned moment-teacher route that:

- constructs the independent adjacent TT targets and analytical target
  directions from fixed prepared rows;
- replays graph-native fixed ALS and normalized marginal recursion;
- contracts per-time TT shape values and tangents;
- executes the existing canonical particle likelihood, streaming OT, and
  Contract E-Chol reset;
- applies the explicit TT target correction with total source/weight,
  streaming-transport, target, whitening, and correction tangents;
- carries the corrected particle cloud and the TT marginal; and
- returns the particle likelihood and its same-program score.

The route factory must issue identity from the actual top-level value,
gradient, and reset callables and bind the complete prepared input. Caller
stamping is forbidden.

### Phase 2: Deterministic Correctness

Run CPU/FP64 reference tests with `CUDA_VISIBLE_DEVICES=-1`:

- identity and tuning fail-closed tests;
- particle likelihood unchanged at the current step by TT-only quantities;
- particle weighted mean/covariance preserved after correction;
- all score coordinates versus centered finite difference of the same value
  program;
- empirical-target zero-correction tie-out;
- ordered co-skew and symmetric co-kurtosis masks;
- concrete graph has TensorFlow control flow and no host callback.

Any derivative mismatch is an implementation veto. Centered finite difference
is validation only and never the runtime score.

### Phase 3: Offline Tuning

For each exact claim scope, run a bounded calibration/validation grid over only
the controls applicable to this route: TT rank/basis/fit rows/sweeps/ridge,
defensive weight, Sinkhorn and balance counts, and diagonal/pairwise correction
steps/strength. Selection prioritizes hard validity, then heldout represented
fit error and correction residual; timing breaks ties only among valid arms.
Freeze the selected controls in a repository-issued artifact. Claim data are
untouched during selection.

### Phase 4: GPU/XLA And LGSSM

Run escalated/trusted GPU commands with memory growth configured before device
initialization and TF32 disabled before graph execution.

1. Integrated deterministic FP32/FP64 parity and graph/device gate.
2. Scope-specific `T=2` claim against Kalman and empirical-target Contract E.
3. Fresh scope-specific `T=10` tuning and claim.
4. Fresh scope-specific `T=50` tuning and claim.

Value and each score coordinate must be reported with absolute difference,
relative difference using a near-zero floor, and finite-particle MCSE where
replicates exist. Numerical parity is not promoted into a model-accuracy
criterion. Kalman discrepancies are interpreted against the predeclared
finite-particle uncertainty, not assumed to be zero.

### Phase 5: Nonlinear Feasibility

Only if Phase 4 has no hard veto:

- predator-prey, `T=20`, one seed, `N>1000`;
- score-admissible Austria SIR, declared observed-data horizon, one seed,
  `N>1000`.

Each receives its own tuning artifact. Compare the hybrid with empirical-target
Contract E using identical prepared randomness. Report validity, runtime,
memory, value, score, fit/shape/transport diagnostics, and paired differences.
Do not rank methods from these one-seed results.

## Budget And Stop Conditions

- Implementation: one primary attempt plus at most two localized repair
  attempts per failed gate.
- Deterministic tests: expected under five minutes per attempt.
- GPU mechanics: at most three launches, expected under ten minutes total
  excluding compilation.
- LGSSM: at most three tuning/claim attempts per horizon and no more than six
  GPU-hours total.
- Nonlinear: at most three tuning/claim attempts per model and no more than
  eight GPU-hours total.
- Never overwrite an attempt directory.

Stop before later phases for wrong target, incomplete derivative, invalid
canonical identity, tuning mismatch, non-finite/corrupt output, GPU memory
policy failure, persistent XLA failure, or exhausted phase budget. A failed
candidate gate blocks promotion and triggers only an in-budget localized repair;
it does not justify changing the scientific target or relaxing thresholds.

## Planned Commands

Focused CPU checks use explicit GPU hiding:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/highdim/test_zhao_cui_moment_teacher_integration.py
```

All GPU detection and execution commands use trusted/escalated permissions:

```bash
nvidia-smi
bash scripts/run_zhao_cui_moment_teacher_integration_campaign.sh
```

The campaign launcher must create unique attempt roots, validate the tuning
scope before launch, and stop rather than launching nonlinear work if any
LGSSM hard veto is present.
