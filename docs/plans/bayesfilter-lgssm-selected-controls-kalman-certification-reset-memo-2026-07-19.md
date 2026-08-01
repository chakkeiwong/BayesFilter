# Reboot Reset Memo: LGSSM Particle-Count Ladder And Nonlinear Handoff

Date: 2026-07-19

Status: `READY_FOR_REBOOT_FRESH_N2000_N5000_PLAN_REQUIRED`

Owner direction: after reboot, test the current canonical LGSSM at
`T=50,N=2000` and `T=50,N=5000`. Each particle count must receive its own
Kalman-blind Sinkhorn/balance calibration. If the larger-particle LGSSM ladder
closes the value-and-score gates, close `T=10` at the successful particle count
and continue to the nonlinear-model testing program with independent tuning for
every model and horizon.

## Read First After Reboot

1. Read this memo completely.
2. Read the controlling result:
   `docs/plans/bayesfilter-lgssm-selected-controls-kalman-certification-result-2026-07-19.md`.
3. Inspect `git status --short`. The worktree is intentionally dirty and
   contains shared uncommitted transport optimization plus this campaign's new
   plan, test, aggregator, result, memo, and artifacts. Do not reset, restore,
   delete, or overwrite unrelated work.
4. Run trusted/elevated `nvidia-smi` and a trusted TensorFlow GPU probe before
   debugging CUDA or launching a run.
5. Create one concise serious-campaign plan under `docs/plans` with an evidence
   contract, compute/attempt budget, fresh versioned output root, stop
   conditions, and the phases below. Perform the required skeptical plan audit
   before implementation or GPU execution.

No execution process from this lane needs to survive the reboot. All completed
evidence is stored in versioned JSON and Markdown files.

## Frozen Scientific State

The current production-shaped route is canonical Contract E--Chol with total
derivative identity
`contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1`.
Contract E--Chol remains the only reset eligible for canonical value, score,
leaderboard, or HMC-facing evidence.

The exact completed scopes are:

| Scope | Selected controls | Engineering result | Kalman result |
| --- | --- | --- | --- |
| `T=10,N=1024,float32/TF32,GPU/XLA` | `sinkhorn_steps=20`, `balance_steps=3` | pass | inconclusive |
| `T=50,N=1024,float32/TF32,GPU/XLA` | `sinkhorn_steps=20`, `balance_steps=8` | pass | score fail |

At `T=50,N=1024`, all direct transport gates pass, but the fourth HMC-score
coordinate, `q_scale`, fails against Kalman:

```text
parameter order:      [phi1, phi2, phi3, q_scale, r_scale]
LEDH q_scale score:   -0.8833980
Kalman q_scale score: -0.6710117
mean relative error:  -31.65%
simultaneous 95% CI:  [-43.59%, -19.71%]
required region:      [-5%, +5%]
```

Across the 16 particle seeds, the `q_scale` HMC-score standard deviation is
`0.10555` and the standard error of the mean is `0.02639`. The observed score
difference is about eight ordinary standard errors, so adding estimator seeds
at fixed `N=1024` would estimate the same bias more precisely rather than
plausibly remove it.

The exact Kalman `q_scale` predictive-score increments at `T=50` have RMS size
`1.0537` but sum to only `-0.6710`. The total is cancellation-sensitive. A
finite-particle error that is small relative to the time-local terms can be
large relative to the final score.

## Why The Next Test Changes N

Insufficient particle count is a plausible leading hypothesis, not an
established cause. At `T=2`, the earlier current-family particle ladder showed
descriptive narrowing of `q_scale` error and seed dispersion as `N` increased:

| N | Mean relative `q_scale` error | Across-seed SD |
| ---: | ---: | ---: |
| 128 | `3.85%` | `9.08%` |
| 256 | `4.44%` | `5.86%` |
| 512 | `2.29%` | `3.33%` |
| 1024 | `0.69%` | `2.86%` |

This is consistent with finite-particle error decreasing with `N`, but it does
not prove a convergence rate at `T=50`. The `N=2000` and `N=5000` runs are a
discriminating particle-scaling experiment. A material decrease in the
`q_scale` bias supports the insufficient-`N` hypothesis. A plateau, sign-stable
bias, or deterioration indicates a remaining long-horizon algorithmic
approximation or implementation problem.

## Binding Post-Reboot Sequence

### Phase 0: Harness And Policy Generalization

Before any research run, remove the current harness-only particle-count
hardcoding without changing the finite algorithm:

- `docs/benchmarks/run_canonical_lgssm_fused_ot_loop_repair.py` currently
  permits only `N=(128,1024)`;
- `docs/benchmarks/run_ledh_offline_ot_tuning_campaign.py` currently hardcodes
  `NUM_PARTICLES=1024` and assumes `K=N` in its gates and scope;
- generalize both through an explicit positive `--num-particles` argument;
- derive chunks only through
  `bayesfilter.highdim.transport_chunk_policy.select_transport_chunks`;
- bind `N`, chunk policy, chunk sizes, block grid, horizon, dtype, route, and
  control family into `LEDHTuningScope` and all selection/claim artifacts;
- preserve XLA `tf.while_loop`, float32/TF32, the 8192 MiB TensorFlow logical
  limit, structured exception artifacts, and no Python horizon loop;
- add tests that reject cross-`N`, cross-horizon, wrong-chunk, and stale tuning
  artifacts.

The active chunk policy is not tunable:

| Particle count | Required chunk `K` | Exact block grid |
| ---: | ---: | ---: |
| 2000 | 2000 | `1 x 1` |
| 5000 | 2500 | `2 x 2` |

Do not revive small chunks, fallback chunks, or any policy other than
`dpf_transport_exact_divisor_cap3000_v1`. For `N=5000`, `K=2500` is required
because it is the largest divisor not exceeding 3000.

The same-cloud geometry cache is optional and changes memory scaling. The
scientific baseline for this ladder is the uncached streamed route used by the
selected-control certification. Do not enable the cache merely for speed. An
exact-scope cached arm may be used only after same-input value/total-score and
marginal parity plus memory-cap checks. The current cache assumes a one-block
same-cloud solve and therefore must not be forced onto the `N=5000,K=2500`
multi-block scope.

### Phase 1: `T=50,N=2000` Independent Tuning And Claim

`T=50,N=2000` is a new scope. The `N=1024` pair `(20,8)` is only a warm-start
hypothesis.

Required procedure:

1. Run a one-seed resource probe under the 8 GiB limit.
2. If 16 simultaneous seeds do not fit, use a fixed compiled seed microbatch
   size and aggregate independent seed artifacts. Do not reduce `N`, alter
   chunks, disable XLA, or omit the total score to make the claim fit.
3. Use new calibration and validation seed blocks and a disjoint untouched
   claim block. Record the exact seed policy before observing candidate output.
4. At fixed initial `sinkhorn_steps=20`, tune the cheaper `balance_steps`
   control first. The previous balance ladder may be used as an explicit
   warm-start grid, not a default.
5. Only after exhausting the declared balance ladder may the plan increase
   `sinkhorn_steps`; for every Sinkhorn rung, retest the full cheaper balance
   ladder.
6. Selection uses only direct numerical/engineering gates. Kalman values and
   scores must not be computed or inspected during calibration or validation.
7. Freeze the selected pair, run the untouched 16-seed claim, then compare the
   same claim output with the exact differentiated Kalman value and HMC score.

Direct tuning/claim gates remain:

```text
TV_col <= 1e-4
E_row  <= 1e-2
finite value and total score
valid chart and reset
bitwise within-run replay
exact work accounting
StatelessWhile present
python_horizon_unroll == false
correct TF32/GPU/XLA/chunk/scope identity
```

Kalman output is a later scientific gate, never a tuning objective.

`N=2000` is an intermediate discriminating rung. A valid numerical run that
fails the Kalman screen does not stop `N=5000`; proceed unless an engineering,
resource, target-identity, artifact, or campaign-budget continuation veto fires.

### Phase 2: `T=50,N=5000` Independent Tuning And Claim

Repeat the full Phase 1 protocol as a distinct scope. Do not transfer the
selected `N=2000` controls as anything stronger than a warm start. The required
chunk identity is `K=2500`, `2 x 2` blocks.

The primary scientific screen remains the previously frozen LGSSM center
screen over 16 untouched estimator seeds:

```text
relative value-bias simultaneous 95% interval inside [-0.001, +0.001]
every HMC-score relative-bias simultaneous 95% interval inside [-0.05, +0.05]
```

Use two-sided Bonferroni-Student intervals over the six outputs: value plus five
score coordinates. Report raw scores, absolute errors, seed SDs, standard
errors, predictive-increment energy, and average-OPG diagnostics, but do not
replace the frozen primary criterion retrospectively.

Interpretation rules:

- `screen_pass`: all intervals are contained and all hard gates pass;
- `screen_fail`: any interval is wholly outside or a hard gate fails;
- `inconclusive`: all other cases;
- do not widen thresholds after seeing output;
- do not call a descriptive decrease a proven `1/N` convergence rate;
- preserve `N=2000` even if `N=5000` passes.

### Phase 3: Close LGSSM `T=10`

The old `T=10,N=1024` result is inconclusive, not closed. If
`T=50,N=5000` passes, independently tune `T=10,N=5000` as another new scope
and run its untouched Kalman claim under the same value/score interval rules.
Do not reuse the `T=50,N=5000` controls without calibration.

LGSSM is successful for nonlinear handoff only when:

1. `T=50,N=5000` is `screen_pass`;
2. `T=10,N=5000` is `screen_pass`;
3. all production GPU/XLA, marginal, replay, work, memory, route, chunk, and
   tuning-scope identities pass; and
4. no same-scalar derivative or target-identity veto remains.

If `N=5000` is inconclusive, write a fresh precision/power amendment before
adding seeds. If it fails, do not continue nonlinear transfer; first run the
time-local score decomposition described below.

## Required Failure Diagnostic If N=5000 Does Not Pass

On identical observations and random streams, compare active Contract E,
no-reset, and exact Kalman `q_scale` score increments. Decompose the finite
score by time into:

- stationary initial-covariance contribution;
- transition/proposal contribution;
- observation-weight and likelihood-normalization contribution;
- carried previous-weight contribution; and
- Contract E moment/weight/transport reset contribution.

Each reported derivative component must be checked against the derivative of
the same partial finite scalar. Do not change Sinkhorn/balance settings to fit
Kalman when direct OT marginals already pass.

## Conditional Nonlinear Continuation

Only after the LGSSM success conditions above pass, create and skeptically
review a fresh nonlinear master plan. The plan must test whether the generic
LGSSM repairs carry over without transferring numerical controls.

Every nonlinear model/horizon is a separate tuning scope. If its executed
route contains streaming Sinkhorn and terminal balancing, calibrate a new
`sinkhorn_steps` and `balance_steps` pair for that exact model, horizon, data
regime, `N`, dtype, route, and chunk identity. A setting from LGSSM or another
nonlinear model is only a warm start. If an experimental TP route has
feature/chart controls rather than Sinkhorn controls, do not mislabel them;
tune its actual route-specific controls independently as well.

Respect the registered model horizons:

| Model | Next valid horizons after LGSSM closes |
| --- | --- |
| Actual SV | independently tuned `T=10`, then `T=50` |
| KSC-SV | independently tuned `T=10`, then `T=50` |
| Generalized SV | repair the existing score-feature failure at `T=10` before any `T=50` claim |
| Predator--prey | independently tuned `T=10`, then its registered full target `T=20`; `T=50` would be a new target and requires owner direction |
| Austria SIR | independently tuned `T=10`, then its registered full target `T=20`; `T=50` is not the registered row |

For every nonlinear scope, use disjoint calibration, validation, and untouched
claim data/seeds; retain same-scalar derivative checks, model-specific
reference/comparator checks, GPU/XLA/TF32 evidence, and honest nonclaims. Do not
claim that an LGSSM pass proves nonlinear score correctness.

## Controlling Artifacts

- Certification plan:
  `docs/plans/bayesfilter-lgssm-selected-controls-kalman-certification-plan-2026-07-19.md`
- Certification result:
  `docs/plans/bayesfilter-lgssm-selected-controls-kalman-certification-result-2026-07-19.md`
- Main aggregate:
  `docs/benchmarks/artifacts/lgssm_selected_controls_kalman_20260719/attempt01/aggregate.json`
- `T=10` node:
  `docs/benchmarks/artifacts/lgssm_selected_controls_kalman_20260719/attempt01/t10_s20_b3_s16.json`
- `T=50` node:
  `docs/benchmarks/artifacts/lgssm_selected_controls_kalman_20260719/attempt01/t50_s20_b8_s16.json`
- `T=50` no-reset diagnostic:
  `docs/benchmarks/artifacts/lgssm_selected_controls_kalman_20260719/attempt01/t50_no_reset_s20_b8_s16.json`
- Per-model tuning policy:
  `docs/plans/bayesfilter-ledh-per-model-scope-tuning-master-program-2026-07-19.md`
- Generic transport optimization result:
  `docs/plans/bayesfilter-ledh-generic-transport-optimization-phase1-result-2026-07-19.md`

The certification aggregate SHA-256 is
`dc6f0422bc6cf4117fcfc83e70b7d1057817f68367d816ec4c5e9237262067d6`.
The recorded Git commit is
`9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b`; the relevant working source also
contains uncommitted generic transport optimization, so source hashes in each
new artifact remain mandatory.

## Forbidden Shortcuts And Claims

- Do not reuse `(20,8)` as the `N=2000` or `N=5000` selected setting without
  independent tuning.
- Do not tune against Kalman value or score.
- Do not change `TV_col`, `E_row`, value, or score thresholds after seeing a
  failure.
- Do not reduce chunks below repository policy, disable XLA, use Python horizon
  loops, use NumPy as the algorithm backend, silently switch precision, or
  remove the total score to obtain a run.
- Do not interpret a completed command, finite output, or marginal pass as
  Kalman score correctness.
- Do not describe one-seed or descriptive particle scaling as a statistically
  supported convergence rate.
- Do not continue to nonlinear models unless both LGSSM `T=50,N=5000` and the
  independently tuned `T=10,N=5000` claims pass.
- Do not create `T=50` predator--prey or SIR rows without explicit target/data
  registration and owner direction.
- Do not claim HMC readiness, posterior correctness, parameter-region validity,
  method superiority, or complete leaderboard readiness from this ladder.

## Reboot Handoff Summary

The immediate post-reboot task is not to rerun `N=1024`. Generalize and test the
harness, then independently tune and claim `T=50,N=2000`, followed by
`T=50,N=5000`, under the exact chunk policy and 8 GiB GPU limit. `N=2000` is an
intermediate particle-scaling rung and does not block the planned `N=5000` rung
on scientific failure alone. If `N=5000` passes, independently close
`T=10,N=5000`; only then launch the separately reviewed nonlinear per-scope
tuning and carryover program.

