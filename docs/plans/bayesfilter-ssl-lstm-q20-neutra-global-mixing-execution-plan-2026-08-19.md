# SSL-LSTM q=20 NeuTra global-mixing execution plan (2026-08-19)

Status: `EXECUTED_TERMINAL_UNDER_BUDGETED`

This is the campaign plan authorized by the user's 2026-08-19 request to plan,
review, and execute the SSL-LSTM q=20 NeuTra continuation. It operationalizes,
but does not rewrite, the mathematical correction in
`bayesfilter-ssl-lstm-q20-neutra-global-mixing-repair-plan-2026-08-19.md`.
Historical artifacts remain evidence under their original claims. This plan
uses fresh, versioned output roots and preserves the dirty shared worktree.
Here `q=20` is the locked SSL-LSTM state/complexity setting. The inferred
physical parameter vector has exactly four coordinates, and every parameter
diagnostic below covers all four.

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Can a target-specific weighted NeuTra transport make one exact SSL-LSTM q=20 pullback fixed-HMC kernel forget initialization and traverse both currently known material sign regions? |
| Candidate mechanism | Weighted forward-KL dense IAF training from eight receipt-verified annealed-SMC terminal populations, followed by exact transformed-target fixed HMC. The SMC rows are optimization replay only. |
| Claimed target | `log pi_z(z) = log pi_theta(T_phi(z)) + log_abs_det(J_T_phi(z))`, with the full explicit score pullback and original target status. |
| Candidate failure | A trained transport or tuned kernel is finite but mode-locked, nonconverged, or invalid. This rejects that candidate, not the target or NeuTra direction. |
| Harness failure | Receipt mismatch, wrong target/adapter identity, scalar fallback, shape error, nonfinite parity check, missing GPU memory growth, or artifact overwrite risk. Repair locally before interpreting a candidate. |
| Promotion criterion | One frozen transport and one common fixed-HMC kernel pass the canonical sequential policy, all finite/status gates, parameter and mode-indicator modern R-hat/ESS gates, and per-chain cross-sign transitions. |
| Promotion veto | Any invalid target status, nonfinite state/value/score/Jacobian/log acceptance, stale identity, batch-size-one or row-mapped training, failed R-hat/ESS, or any retained chain that remains conditional on its starting region. |
| Continuation veto | Exact target/transport parity cannot be established, receipt-verified replay is unavailable, trusted GPU/XLA execution cannot start, or the total campaign compute/attempt budget is exhausted. |
| Repair trigger | A finite mode-locked canary triggers the predeclared capacity/learning-rate screen. Failed serious training triggers the one predeclared lower-rate repair rung. Failed HMC tuning triggers the second frozen candidate, not conditional pooling. |
| Explanatory diagnostics | Training/selection/audit loss, replay ESS, acceptance, energy-error summaries, occupancy, transition counts, SMC mass interval, and runtime. None promotes a posterior alone. |
| Must not be concluded | Exhaustive mode discovery, statistical superiority, universal NeuTra defaults, exact SMC posterior authority, parameter identifiability, production/default readiness, or predictive equivalence from HMC diagnostics alone. |

## Existing evidence and baseline ladder

The baseline ladder is descriptive; this campaign does not rank methods.

| Rung | Existing evidence | Role here |
|---|---|---|
| Naive/local | The old single-region reverse-KL NeuTra transport | Failure baseline: local movement is not global posterior sampling |
| Best tested classical | Dense-mass physical replica HMC moved across signs but failed warm-up R-hat (`1.3175 > 1.05`) | Independent sampler comparator; not an upstream archive gate |
| Scope-limited mass control | Eight annealed-SMC populations passed their two-known-region receipt/ESS gates | Receipt-bound weighted optimization replay and explanatory mass interval only |
| Plain proposed | `(32,32)`, three-stage IAF, learning rate `1e-3`, 20 updates, one seed | Mechanics canary only |
| Enhanced proposed | Factorial target-specific capacity/rate screen with two seeds, then exact common-kernel HMC | Candidate under test |

A 32-start target-query search found only the two known stationary regions, but
that is bounded negative evidence and does not prove that no additional mode
exists. Any successful result remains scoped to the tested target and bounded
mode-discovery evidence.

## Evidence contract

### Engineering ledger

The executable path must demonstrate all of the following before scientific
interpretation:

1. the replay tensor paths, SHA-256 values, dtypes, and shapes match
   `receipt-recovery-v1.json`, whose recovery gates and status must pass;
2. the train/selection split uses `central-00` through `central-06`, while
   `central-07` remains unused by the new campaign for training, checkpoint
   selection, or candidate nomination until all choices have been frozen;
3. every optimization update uses a TensorFlow/XLA batch of 600 rows, with no
   Python per-row target call, `tf.map_fn`, `tf.vectorized_map`, or scalar
   fallback;
4. the exact base target and frozen transport expose value, score, combined
   status, Jacobian, and explicit pullback operations with matching identities;
5. HMC samples retain `[draw, chain, parameter]` ordering and mode labels are
   explicitly transposed to `[chain, draw]`; and
6. every output root is fresh and fails closed if it already exists.

### Sampler ledger

The primary sampler criterion applies to retained draws from one common exact
kernel:

- warm-up uses `bayesfilter_neutra_sequential_hmc_v1`, is archived, and is
  excluded from all estimates;
- recent-window maximum rank-normalized split/folded R-hat is `<= 1.05` before
  retained sampling;
- retained parameter and sign-indicator R-hat is `<= 1.01`; the binary
  sign-indicator calculation is tie-aware and supplemental to the direct
  per-chain crossing gate, but a nonfinite or undefined result still vetoes;
- minimum bulk ESS is `>= 1000` and minimum tail ESS is `>= 400` over the four
  parameters and sign indicator;
- every chain visits both known sign regions and records at least one retained
  transition; and
- all state, target, score, log-acceptance, target-status, movement, and exposed
  energy-error gates pass. Native divergence is reported as
  `not_exposed_by_kernel` when TFP does not expose it.

The bulk ESS floor is derived from the downstream request for 1,000 posterior-
predictive paths. The tail ESS floor is inherited from the reviewed q=20 and
three-mode confirmation protocols. They are target-specific evidence settings,
not universal defaults.

### Scientific endpoint ledger

Only after sampler admission, draw one retained posterior parameter value with
replacement per predictive path and simulate 1,000 complete raw observation
paths at each `T` in `{10, 20, 30, 50, 100}`. Compare them with 1,000 paths
generated from the true parameter using
`bayesfilter.testing.posterior_predictive_tf.posterior_predictive_energy_test`.
For each horizon, the statistic is the biased empirical whole-path energy
distance

`2 mean ||X_i-Y_j|| - mean ||X_i-X_j|| - mean ||Y_i-Y_j||`,

with zero diagonals retained in the within-arm V-statistic terms. Use 9,999
balanced-label permutations, the plus-one Monte Carlo p-value, and reject the
T-specific equality null iff `p < 0.01`. Report five separate decisions; do not
form an omnibus, combined, or familywise pass/fail decision. The minimum
attainable p-value is `0.0001`.

A fresh true-vs-true mechanics calibration runs first at `T=20`, `n=32`, and
999 permutations. It passes only when source/target identities, disjoint seed
domains, path shapes, finiteness, XLA placement, balanced permutation geometry,
energy-distance invariants, replay, and artifact checks pass. Its realized
p-value is explanatory and cannot pass or fail the campaign because a single
valid null realization may fall on either side of `0.01`. A harness failure
closes the material tests and triggers one local repair; a second harness
failure is a continuation veto. This is not a repeated-null Type-I calibration.

At a material horizon, `p < 0.01` is evidence against exact equality of those
two finite-dimensional path laws. `p >= 0.01` means only
`NOT_DISTINGUISHED_AT_1_PERCENT`; it is not predictive-equivalence evidence.
Passing all horizons does not prove equality or model adequacy. Report energy
statistics, permutation uncertainty, and descriptive path summaries.

## Diagnostic roles

| Diagnostic | Role |
|---|---|
| Receipt hash/dtype/shape and target identity | Hard engineering veto |
| Batch-native/XLA/memory-growth receipt | Hard training/runtime veto |
| Inverse/forward/Jacobian and value/score/status parity | Hard engineering veto |
| Canary loss, replay ESS, acceptance, occupancy | Explanatory or nomination only |
| Canary per-chain sign crossing | Mechanics nomination; failure triggers capacity repair |
| Selection weighted NLL | Candidate nomination only; observed differences are descriptive |
| Untouched audit finite/parity checks | Hard frozen-candidate veto |
| HMC parameter/sign R-hat and ESS | Promotion criterion and veto |
| Per-chain retained sign transitions | Anti-pooling promotion veto |
| True-vs-true mechanics calibration | Hard harness veto; realized p-value explanatory only |
| Five energy/permutation tests | T-specific equality decisions after sampler admission; no joint decision |

No descriptive metric is silently upgraded to a promotion criterion.

## Default and numeric assumption audit

| Choice | Provenance and status | Justification | Failure mode and early diagnostic |
|---|---|---|---|
| TensorFlow/TFP, float64, XLA, TF32 off | Reviewed repository policy/established q=20 target route | Preserves current exact target identity and numerical route | Device/XLA/identity receipt fails closed |
| GPU 1 with memory growth | Inherited prepared runner; execution convenience, not algorithm default | Keeps the campaign on one visible device while allowing shared allocation | Trusted probe verifies one logical GPU and growth before tensors |
| Six training banks, one selection bank, one audit bank | Inherited eight independent SMC populations; reviewed split | Preserves independent bank-level selection/audit boundaries | Canary audit leakage is explicitly removed; receipt check binds all paths |
| Training batch size 600 | Derived: six receipt-bound populations x 100 terminal rows each | Uses the complete training partition in every batch-native update | Manifest records shape `[600,4]`; any scalar/row-mapped fallback vetoes |
| Canary: width 32, 3 stages, `1e-3`, 20 updates, seed 1 | Prepared mechanics runner; convenience hypothesis | Smallest existing end-to-end GPU/XLA check | Never used for promotion; mode lock triggers screen |
| Canary HMC 64 discarded + 64 retained transitions per chain | Prepared mechanics runner; convenience hypothesis | Exercises compiled warm-up, retained traces, transformation, and chain-axis handling in 128 transitions | Shortness forbids convergence or posterior claims; invalid mechanics vetoes |
| Screen capacities `(64,64)/3` and `(128,128)/6` | Smaller target-specific arm plus validated three-mode capacity precedent; hypotheses | Brackets a moderate and established multimodal-control capacity | Neither is promoted by transfer; both are tested on q=20 |
| Screen rates `1e-3` and `3e-4` | Existing q=20 warm start plus a convenience factor-of-about-three lower arm; hypotheses | Tests whether the warm start is unstable or too aggressive | Finite gradients/loss and heldout curve expose failure |
| Screen seeds 2 and 3 | Deterministic reproducibility seeds; convenience choices | Distinct from canary seed 1 and sufficient to expose a single-seed accident | Two seeds do not estimate a success rate; no ranking claim |
| Tanh activation, scale cap 2, full-reverse stage permutation, initialization scale 0.02, Adam `(0.9,0.999,1e-7)`, gradient clip 10 | Inherited canary/class settings; warm-start hypotheses, not promoted defaults | Holds non-capacity architecture and optimizer mechanics fixed while screening capacity and rate | Checkpoint finiteness, clipping telemetry, selection curves, and the frozen audit expose instability; failure does not justify transfer or promotion |
| Update ladder `250, 2000, 8000` | Early diagnostic plus successful three-mode checkpoint near 8,750; budget ladder hypothesis | Detects immediate instability before paying the full inherited-scale budget | If all arms are still improving at 8,000, stop as under-budgeted rather than promote |
| One repair rate `1e-4` at `(128,128)/6`, seeds 2 and 3 | Predeclared lower-rate repair hypothesis | Distinguishes optimization instability from capacity failure | Allowed only if both primary rates fail for numerical/training reasons |
| Four HMC chains | Repository minimum and existing q=20/control protocols | Enables multi-chain modern diagnostics | Any nonmoving/invalid chain vetoes |
| Initial tuning `step=0.05`, `L in {3,5,10,15}` | Three-mode tuning warm start, not transferred evidence | Bounded target-specific fixed-HMC grid with `L>=2` | Tuning output decides viability; acceptance never establishes convergence |
| Tuning target acceptance 0.70, pass band `[0.55,0.90]`, repair band `[0.40,0.95]` | Inherited three-mode tuning protocol; warm-start hypotheses | Permits target-specific dual averaging while rejecting numerically unusable fixed kernels | Acceptance is kernel-health/tuning evidence only; full modern R-hat verification and sequential gates remain mandatory |
| Tuning verification in raw physical coordinates | Derived from the four-parameter posterior claim | Initialization forgetting must hold for the claimed physical parameters, not only a latent chart | A latent-only pass cannot admit the kernel |
| Per-call tuning forecast with 1.25 overrun allowance | Measured q=20 canary rate plus a convenience 25% engineering allowance; resource hypothesis only | Refuses a full-chain call that cannot leave the 180 s closeout reserve | A refusal means under-budgeted; it cannot reject a kernel or establish runtime scaling |
| Sequential chunks of 500; warm-up 2,000--10,000; retained 2,000--10,000 | Canonical repository policy | Enforces recent-window readiness and cumulative retained evidence | Hard cap may leave the campaign under-powered; report, do not relax |
| Predictive `n=1000`, horizons `10,20,30,50,100`, per-test alpha `0.01` | User-established q=20 diagnostic protocol recorded in the 2026-08-18 gap-closure plan | Preserves the requested five finite-horizon endpoint | Separate tests only; non-rejection is not equivalence and other horizons are untested |
| 9,999 material permutations; 999 canary permutations | Inherited reviewed q=20 energy diagnostic; derived p-value resolutions `0.0001` and `0.001` | Resolves the 1% material threshold and bounds mechanics cost | Record exceedance count and seed; no exact-p-value or power claim below the resolution |
| Predictive mechanics canary `T=20`, `n=32` | Inherited reviewed q=20 energy mechanics contract; convenience shape/runtime check | Exercises a non-default horizon and the complete statistic before material seeds | Its p-value is never a calibration or scientific gate |
| Total GPU wall cap 28,800 s | Convenience resource ceiling inherited from the prior q=20 campaign, not a scientific threshold | Bounds this authorized local campaign to eight GPU-hours | Stop under-budgeted if a required gate needs more compute |

## Pre-mortem

The campaign could pass while misleading us if balanced starts create pooled
occupancy without transitions, if selection leakage reaches the audit bank, if
the replay omits a narrow mode, or if a short canary is mistaken for
convergence. Per-chain transitions, reserved audit use, bounded mode-discovery
nonclaims, and the sequential HMC gate address these risks.

It could fail for engineering or tuning reasons rather than scientific ones if
the runner calls the adapter with the wrong keyword, loses chain/draw ordering,
uses stale replay tensors, overwrites evidence, or inherits an unsuitable
learning rate/capacity. Focused source checks, receipt validation, fresh roots,
and the factorial screen distinguish these explanations before rejecting the
candidate mechanism.

## Execution phases

### Phase 0: plan review and source audit

Before any GPU process:

1. self-audit this plan against the skeptical checklist below;
2. obtain one bounded read-only Claude review of exactly this plan;
3. revise material findings and record the review;
4. inspect the canary, target adapter, fixed transport wrapper, full-chain HMC
   shape contract, and replay receipt schema; and
5. repair only the named lane files, adding focused tests for each discovered
   defect.

Known required prelaunch repairs are:

- replace the invalid `fixed_transport=` constructor keyword with `transport=`;
- keep `central-07` out of canary validation;
- verify replay hashes/dtypes/shapes against the recovered receipt;
- require a fresh output root; and
- bind output manifests to this execution plan.

### Phase 1: CPU checks and trusted GPU preflight

Run syntax and focused CPU-only checks with GPU hidden:

```bash
python -m py_compile \
  bayesfilter/inference/neutra_global_mixing.py \
  tests/test_ssl_lstm_q20_neutra_global_mixing.py \
  docs/benchmarks/run_ssl_lstm_q20_neutra_gpu_preflight_2026_08_19.py \
  docs/benchmarks/run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py

CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
python -m pytest -q \
  tests/test_ssl_lstm_q20_neutra_global_mixing.py \
  tests/test_ssl_lstm_q20_gap_closure_campaign.py \
  tests/test_ssl_lstm_q20_gap_closure_mode_discovery.py
```

Then run one trusted, escalated GPU probe with
`TF_FORCE_GPU_ALLOW_GROWTH=true` and `CUDA_VISIBLE_DEVICES=1`. It must call the
repository memory helper before any logical-device or tensor initialization and
record one visible logical GPU. The preflight aggregate cap is 300 seconds,
including an unchanged infrastructure retry. A gateway rejection is
infrastructure evidence and stops the GPU lane without scientific
interpretation.

Preflight command, under trusted GPU execution:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
timeout 300s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_gpu_preflight_2026_08_19.py \
  --device 1 \
  --output docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/gpu-preflight.json
```

For benchmark commands, `--device 1` is a physical-device selector consumed by
the runner before TensorFlow import: the runner sets `CUDA_VISIBLE_DEVICES=1`,
after which TensorFlow correctly names the sole visible logical device
`GPU:0`. The shell commands therefore do not also set `CUDA_VISIBLE_DEVICES`.
The manifest records both the requested physical selector and the visible
logical device; treating `--device 1` as a post-mask logical index is a hard
configuration error.

### Phase 2: mechanics canary

Fresh root:

`docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/gpu-canary`

Command, under trusted GPU execution:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
timeout 1800s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py \
  --device 1 --updates 20 --hidden-width 32 --stages 3 --seed 1 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/gpu-canary
```

The aggregate phase cap is 1,800 s, a convenience bound for a
20-update/128-transition-per-chain mechanics run. Any permitted local
harness-repair retry consumes the unused remainder of this same cap; it does
not receive another 1,800 s. A finite result may nominate Phase 3. Cross-sign labels do not
establish convergence. A finite mode-locked result rejects only this small
candidate and still triggers Phase 3. A target/adapter/harness failure is
repaired once and retried with unchanged scientific settings; a second failure
stops the lane.

### Phase 3: target-specific training screen and frozen audit

Only after a finite mechanics run, implement and review the campaign runner at:

`docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py`

Training and HMC use separate immutable runners. The training artifact binds the
training-runner SHA-256 and complete frozen transport state; the HMC runner may
consume but cannot rewrite that artifact. This prevents a later sampler repair
from changing the source identity of the program that selected the transport.

Run all eight primary cells from the capacity, rate, and seed grid through the
update ladder. Check selection loss every 250 updates and restore each arm's
lowest observed selection-loss checkpoint. Within each seed, nominate the
descriptively lowest selection-loss finite arm; this is an optimization choice,
not a statistical superiority claim. An exact numerical tie is resolved in
favor of the smaller capacity and then the lower learning rate.

After all choices are frozen, evaluate `central-07` exactly once for each of the
two nominated seed candidates. Require finite audit loss/ESS, receipt and
identity match, and inverse/forward/Jacobian/value-score-status parity. The
audit metric explains generalization but does not rank candidates.

Primary root:

`docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/training-screen`

The screen has a 7,200 s aggregate phase cap. The two repair cells, checkpoint
validation, and frozen audits draw from the same cap; they receive no additional
time. For a seed only, if all four primary arms fail for numerical training
reasons, run its one predeclared `1e-4` repair cell. Thus at most two repair
cells can run, and a viable primary arm suppresses the repair for that seed. If
no candidate passes frozen audit, stop with a training-protocol negative result.

Planned command, after the runner exists and its focused tests pass:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
timeout 7200s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py \
  --device 1 \
  --capacities 64x3,128x6 --learning-rates 1e-3,3e-4 \
  --seeds 2,3 --update-ladder 250,2000,8000 \
  --time-cap-seconds 7050 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/training-screen
```

### Phase 4: fixed-kernel tuning and sequential HMC

For each frozen candidate that passes audit:

1. tune the target-specific fixed-HMC grid with the repository tuning helper;
2. reject `L=1`, NUTS, per-mode kernels, and runtime transport retuning;
3. run a short common-kernel mechanics screen;
4. if the kernel is finite, execute the canonical sequential controller; and
5. apply parameter and sign-indicator diagnostics plus the anti-pooling gate.

The tuning helper uses `initial_step_size=0.05`, ordered leapfrog feasibility
ladder `(3,5,10,15)`, target acceptance `0.70`, acceptance band
`[0.55,0.90]`, repair band `[0.40,0.95]`, budget schedule `(32,64,128)`, 16
tuning results, 64 screening results after 16 burn-in steps, and a 2,000-result
raw-physical-coordinate modern-R-hat verification after 64 burn-in steps. Each
length is passed to the repository tuner as a singleton grid, and the first
length that passes its complete verification is frozen. Later lengths are not
run and no descriptive ranking is made. These are inherited tuning warm starts
and evidence minima, not transferred q=20 successes. The four initial chains are the two mapped known
representatives, a `+0.05` first-coordinate perturbation of the plus
representative, and a `-0.05` first-coordinate perturbation of the minus
representative. They are initialization-forgetting probes, not mode weights.

Attempt the frozen seed-2 candidate first. If and only if it fails tuning or a
global-mixing gate while budget remains, attempt the frozen seed-3 candidate.
Stop after the first passing candidate, so there is no unreachable "both pass"
selection branch. Do not concatenate candidates or their chains. At most one
candidate may be admitted for the posterior bank; this fallback order is
predeclared and does not rank training seeds.

The HMC cap is the adaptive campaign remainder, computed before launch as
`28,800 - actual_preflight_and_canary_gpu_wall - actual_training_gpu_wall -
3,600_predictive_reserve`. It is shared across both candidates. It may exceed
the original provisional 14,400 s allocation, but the total campaign cap is
unchanged and the predictive reserve is not borrowed. Exhaustion before the
minimum sequential evidence is an under-budgeted result, not a convergence
failure and not permission to weaken the gates. The HMC manifest records the
input wall-time artifacts and the derived timeout before any chain launches.

The completed training screen consumed `902.9125 s`. At launch, the runner
combined it with only the successful `3.2075 s` preflight and `790.2905 s`
mechanics canary. It therefore used a prior-wall subtotal of `1,696.4104 s`, a
derived HMC remainder of `23,503.5896 s`, the floored shell bound `23,503 s`,
and an internal `23,323 s` bound reserving 180 seconds for atomic terminal
closure. The post-run audit found that this subtotal omitted `19.8 s` from the
failed preflight and launch-invalid canary. Correct prior wall was
`1,716.2104 s`; under the same policy the external/internal bounds would have
been `23,483 s` and `23,303 s`. Actual HMC wall was well below both bounds and
the terminal decision is unchanged. Before sequential sampling, the runner
also records a non-scientific sufficiency estimate derived from the selected
kernel's own 2,064-transition verification wall and the 4,000-transition
canonical sequential minimum. If the measured remainder cannot cover that
estimate plus the closeout reserve, it stops as `UNDER_BUDGETED` without
weakening a sampler gate.

The public fixed-transport tuner has no native deadline. Each of its full-chain
calls is therefore wrapped by a pre-call resource veto derived from the frozen
canary's measured `771.3013279580 s` for `(64+64)*5 = 640` leapfrog
transitions, scaled to the requested work and multiplied by the convenience
allowance `1.25`. The allowance is not a runtime law or scientific threshold.
A refused call is archived and terminates the campaign as under-budgeted; the
tuner cannot convert it into evidence that the length or transport failed.
The canonical sequential controller separately calls the same kind of budget
check before every 500-result chunk using the selected kernel's own verification
rate. Every controlled stop writes `result.json`, `manifest.json`, and an
artifact hash inventory.

Planned command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
timeout 23503s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_hmc_2026_08_19.py \
  --device 1 --campaign-wall-cap-seconds 28800 --predictive-reserve-seconds 3600 \
  --time-cap-seconds 23323 \
  --training-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/training-screen \
  --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc
```

### Phase 5: posterior-predictive endpoint

Only after Phase 4 admission, adapt the existing TensorFlow
`posterior_predictive_energy_test` route in a fresh runner, preserving one
retained parameter draw per path and the exact statistic, permutation, and
horizon contracts declared above. Run the true-vs-true mechanics calibration
first, then the material comparison. Use TensorFlow/XLA on the trusted GPU
route with the same memory-growth policy. The aggregate predictive cap is
3,600 s across both launches and any local pre-material harness repair.

Planned command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
timeout 3600s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_posterior_predictive_2026_08_19.py \
  --device 1 \
  --hmc-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc \
  --sample-count 1000 --horizons 10,20,30,50,100 --alpha 0.01 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/predictive
```

Write a terminal result and reset memo whether the campaign passes, rejects a
candidate, stops for infrastructure, or exhausts its budget.

## Attempt and artifact budget

| Item | Maximum |
|---|---:|
| Claude plan review | One initial review plus one exact-path clarification if requested |
| GPU preflight | Two attempts only when the first is an unchanged gateway/infrastructure retry |
| Mechanics canary | One launch plus one local harness-repair retry |
| Primary training cells | Eight |
| Training repair cells | Two |
| Frozen HMC candidates | Two; never pooled |
| Predictive calibration/material launches | One each |
| Total GPU wall | 28,800 s |

The wall accounting is aggregate, not per retry. Preflight and mechanics consume
their actual measured GPU wall up to their respective `300 s` and `1,800 s`
caps. Training plus repair/audit may consume at most `7,200 s`. Predictive
calibration plus material retains `3,600 s`. HMC receives only the remainder of
the unchanged `28,800 s` total after those actual costs and the predictive
reserve are deducted. No phase may borrow the predictive reserve or weaken a
scientific gate; the campaign records cumulative GPU wall and stops at
`28,800 s`.

## Post-canary execution amendment

The reviewed plan had SHA-256
`9c2e03bfd5cb121421c8e50b5a2434131efbf65e719c77f5967836be06e0df84`
when the preflight and mechanics runs launched. The successful preflight at
`r1/gpu-preflight.json` has SHA-256
`28be07fcc83be539b9b643f2127094f025706d0a92b733873a85ddeb56b50a45`.
The first mechanics launch initialized TensorFlow before the memory helper and
was launch-invalid; its preserved failure artifact triggered a local harness
repair. The unchanged retry at `r1/gpu-canary-retry-01/result.json` has SHA-256
`27685b3f22936659b5b5b34bc07c675eb16c47ddb86a465bfd83fb31c31d7bfa` and
passed exact value/score/status/Jacobian and runtime validity checks.

The retry was finite but every chain stayed in its initial sign region: chain
region counts were `[[64,0],[0,64],[64,0],[0,64]]`, every transition count was
zero, and pooled counts were `[128,128]`. This rejects the 20-update mechanics
candidate and triggers Phase 3; it does not reject the exact target, harness, or
NeuTra direction. The HMC call alone took `771.3013 s` for 128 transitions per
chain, with `790.2905 s` total canary wall. That measured cost makes the earlier
fixed `14,400 s` HMC subdivision a material under-budgeting risk for the
canonical minimum of 2,000 warm-up plus 2,000 retained transitions per chain,
in addition to tuning verification. Runtime need not scale linearly, so this is
an engineering budget warning rather than an extrapolated cost claim.

Skeptical amendment verdict: `PASS_WITH_ADAPTIVE_REMAINDER`. The scientific
target, candidates, data split, promotion gates, vetoes, hardware class,
predictive reserve, attempt counts, and total `28,800 s` campaign cap are
unchanged. Only the provisional internal HMC subdivision and runner ownership
are corrected using measured evidence. If the canonical evidence still cannot
fit, the result is `UNDER_BUDGETED`, not permission to shorten warm-up,
retained sampling, tuning verification, or diagnostics.

### Phase 3 prelaunch audit

The training-only runner has SHA-256
`8c17fafcb25a5656ac1e90734f4b157703ddfad490b45ce01063c52779cd0c9f`.
Syntax checks and the three focused q=20 suites passed with `21 passed`. The
audit included a real weighted-state serialize/mutate/restore round trip.

| Skeptical check | Phase 3 disposition |
|---|---|
| Wrong data or baseline | All arms consume the same six receipt-verified training banks and the same `central-06` selection bank; SMC remains optimization replay only |
| Proxy promotion | Selection loss nominates within a seed only, audit loss does not rank, and neither can admit HMC |
| Missing stop condition | An internal `7,050 s` cap leaves 150 seconds for atomic closure before the aggregate `7,200 s` external cap; an incomplete factorial screen cannot nominate |
| Unfair comparison | All eight primary cells run the same 8,000-update ladder and 250-update selection cadence before nomination; incomplete arms are ineligible |
| Hidden defaults | Activation, scale cap, permutation, initialization, optimizer, clipping, capacities, rates, seeds, and update ladder are recorded above and in every state |
| Environment mismatch | Source and tests require memory growth before TensorFlow-using project imports, exactly one logical GPU, float64 XLA, and TF32 disabled |
| Audit leakage | The runner writes `nominations-before-audit.json` before the only audit loader call; that loader accepts exactly `central-07`, and each nominee has one validation call |
| Failure misclassification | Expected nonfinite training is an arm failure; state/receipt/identity or unexpected audit errors are harness failures and cannot reject the mechanism |
| Artifact insufficiency | Every completed arm has an atomic full model/optimizer state, semantic state hash, file hash, selection history, source hash, and progress receipt |

Phase 3 audit verdict: `PASS_FOR_BOUNDED_GPU_EXECUTION`. This is permission to
run the training screen under the existing campaign authorization, not evidence
that either transport will pass frozen audit or HMC.

### Phase 3 result and Phase 4 amendment audit

The training result at `r1/training-screen/result.json` has SHA-256
`886a617eb60895bc97bc6530b74ef9e2578abee64771992fb29495c471cd92c7`;
its manifest has SHA-256
`556e34a3ad9975c10cd5db327fbff2b0c71f82f46da4b840eb1ed11b7f6f1c76`,
and its 30-entry artifact graph has no hash mismatch. All eight primary cells
completed, so no repair cell ran. Seed 2 nominated width 64, three stages,
`3e-4`, update 3,750; seed 3 independently nominated the same capacity/rate at
update 3,250. Their selection losses (`5.4350` and `5.4596`) and audit losses
(`5.6654` and `5.7231`) are descriptive only. Both candidates passed the frozen
audit and exact pullback parity, so both remain viable for ordered HMC attempts;
no ranking between them is statistically supported.

The ordered-leapfrog amendment above responds to measured q=20 transition cost.
It retains every declared length and the full per-length verification, but stops
at the first passing length because later viable kernels are unnecessary to the
question "does one common kernel pass?" It does not select a winner using
acceptance, speed, or descriptive R-hat differences. If no attempted length
passes before the cap, the candidate is tuning-failed or the campaign is
under-budgeted according to the artifact actually reached; unrun lengths are not
called failures.

| Skeptical check | Phase 4 disposition |
|---|---|
| Wrong target or state | HMC must verify all training artifact hashes, reload the exact nominated state, reproduce its semantic/tensor hashes, and bind the q=20 target/adapter/geometry identities |
| Proxy promotion | Tuning acceptance and mechanics occupancy cannot admit; complete fixed-kernel verification plus canonical sequential R-hat/ESS and per-chain crossing remain required |
| Search fairness | Leapfrog lengths have a predeclared order and identical tuning/verification contracts; stopping at first pass is feasibility selection, not a ranking claim |
| Conditional pooling | The retained callback appends a tie-aware sign indicator to all four physical parameters and separately requires both signs and a transition in every chain |
| Missing stop condition | Candidate order, leapfrog order, HMC remainder, predictive reserve, internal closeout, tuning failure, sequential veto, and under-budget classification are explicit |
| Runtime extrapolation | The tuning pre-call veto uses the frozen canary rate plus a declared 1.25 convenience allowance; the sequential veto uses measured same-kernel verification wall. Both are engineering stops only and cannot reject a kernel or support a scientific claim |
| Environment mismatch | The new consumer must repeat trusted one-GPU memory-growth, float64, XLA, and TF32-off validation before loading TensorFlow target code |
| Artifact loss on timeout | Every completed sequential chunk is atomically archived before the controller proceeds; completed tuning lengths and mechanics outputs use fresh subroots |

Phase 4 amendment verdict: `PASS_FOR_ORDERED_FEASIBILITY_EXECUTION`. The total
campaign cap, predictive reserve, target, candidates, common-kernel requirement,
and sampler promotion/veto thresholds are unchanged.

### Phase 4 terminal prelaunch audit

The core-consolidation route ledger referenced by the persistent repository
guard was an ignored artifact and was absent from the worktree. Git history,
unreachable objects, and saved agent-session searches contained no recoverable
copy. The ledger was regenerated from its versioned NeuTra plus HMC/sampling
discovery contract. It now classifies 40 discovered consumers exactly once and
excludes three explicit shared-controller/export/policy plumbing files. The
current q=20 runner is `active_claim_bearing`, binds
`bayesfilter_neutra_sequential_hmc_v1`, and audits the ledger before target
construction.

The shared controller now accepts an optional resource callback and refuses a
chunk before TensorFlow execution when the callback cannot preserve closeout.
No existing caller changes behavior when the callback is absent. The q=20
runner also verifies the returned canonical policy identity and writes complete
terminal artifacts for controlled under-budget and harness-failure paths.

Focused controller, route-policy, and q=20 checks passed with `43 passed`.
The expanded controller/campaign/end-to-end set produced `109 passed` and two
unrelated baseline failures: both require the absent ignored P0 registry
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p0/attempt-04-20260715T1658/target_registry.json`.
Those failures predate and do not exercise this campaign's modified path.
Compilation and `git diff --check` pass.

Terminal prelaunch verdict: `PASS_FOR_BOUNDED_HMC_EXECUTION`. The route guard,
long-call stop conditions, canonical policy binding, complete terminal closeout,
training identities, and sampler diagnostics are now checked. This verdict is
execution permission only, not evidence that any HMC candidate will pass.

Every serious artifact records Git commit and dirty status, exact command,
environment/conda env, CPU/GPU and memory-growth receipt, dtype/TF32/XLA,
target/replay/transport identities, data receipt hash, seeds, batch size,
sample-wise fallback status, wall time, output paths, plan path, and result
path. `N/A` is used only when a field genuinely does not apply.

## Skeptical pre-execution audit

| Risk | Audit finding and disposition |
|---|---|
| Wrong baseline | The failed reverse-KL and dense physical routes are historical comparators, not promotion authorities. SMC is replay/support evidence only. |
| Proxy promotion | Loss, replay ESS, acceptance, occupancy, and canary crossing cannot promote. Exact sequential HMC and predictive gates remain primary. |
| Missing stop conditions | Numerical, identity, audit, sampler, infrastructure, attempt, and wall-time vetoes are explicit. |
| Unfair comparison | The campaign does not claim a method ranking. Architecture/rate differences nominate downstream candidates only. |
| Hidden defaults | Capacity, rates, seeds, update ladder, HMC grid, ESS floors, and time caps have provenance and failure diagnostics above. |
| Stale context | Fresh roots plus plan, target, replay receipt, transport, and Git identities prevent silent reuse. |
| Environment mismatch | CPU checks hide GPU; serious training/HMC requires trusted GPU, XLA, float64, TF32 off, and verified memory growth. |
| Artifact cannot answer question | Training artifacts nominate only; the retained common-kernel HMC artifact and predictive result answer their separate ledgers. |
| Audit leakage | The inherited canary incorrectly evaluated `central-07`; no GPU canary ran, and this plan requires removal before execution. |
| Conditional-chain pooling | Per-chain crossing plus mode-indicator R-hat/ESS is a hard veto; candidates and starts are never concatenated to create weights. |

## Review record and dispositions

A bounded read-only Claude Code review of exactly this plan returned
`VERDICT: REVISE`. The preserved review is
`docs/reviews/bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-plan-claude-review-2026-08-19.md`.

| Review finding | Disposition |
|---|---|
| Predictive test and calibration rule underdeclared | Bound the repository energy/permutation implementation, exact statistic, permutations, alpha rule, mechanics-calibration gate, and nonclaims above |
| Phase and retry budgets could exceed the total | Made every retry/repair consume its phase aggregate and reconciled the `27,300 s` planned maximum with the `28,800 s` campaign cap |
| Physical versus logical GPU index was ambiguous | Defined `--device 1` as the pre-import physical selector and `GPU:0` as the post-mask logical device; removed duplicate shell masking |
| Numeric provenance missing for batch/canary/predictive settings | Added derived or inherited provenance and failure diagnostics for each |
| Four-parameter coverage was implicit | Declared parameter dimension four and complete diagnostic coverage |
| Audit-bank wording conflicted with inherited source | Distinguished the unexecuted inherited leak from the new campaign boundary; prelaunch repair remains mandatory |
| HMC candidate ordering contained an unreachable branch | Frozen seed 2 first, seed 3 only on failure, and stop at the first pass |

The self-audit also verified the exact transformed-target constructor and sample
axis contracts, the replay receipt inventory, and the repository HMC tuning
helper signature. The binary indicator is retained as a supplemental diagnostic
while direct per-chain crossing remains the anti-pooling authority.

Audit verdict: `PASS_AFTER_REVIEW; EXECUTION_REQUIRES_LISTED_PRELAUNCH_REPAIRS`.
No GPU experiment may run while any known repair above is unresolved. The
review changed the plan but did not authorize stronger scientific claims.

## Result-note requirements

The terminal note must include:

- a decision table with primary criterion, veto status, uncertainty, next
  action, and nonclaim;
- an inference-status table covering hard vetoes, viable candidates,
  statistically supported ranking, descriptive-only differences,
  default-readiness, and next evidence;
- separate engineering, sampler, and scientific ledgers;
- a failure classification separating implementation, tuning, diagnostic, and
  scientific evidence;
- a post-run red team with the strongest alternative explanation, evidence
  that would overturn the decision, and the weakest evidence component; and
- exact run manifests and artifact paths.

## Terminal execution outcome

The reviewed campaign executed through the bounded HMC phase and stopped as
`UNDER_BUDGETED_HMC`. The HMC launch bound the pre-result version of this plan
at SHA-256
`309340acaf5a0702ffd0f8999f062aceba4c4e87bd595d34ac28e4cc11e2f81c`.
This terminal appendix changes the current file hash but does not change the
launch contract or any predeclared gate.

Candidate seed 2 with `L=3` completed a 2,000-result-per-chain verification.
All samples and target/status telemetry were finite, but the observation-weight
coordinate had rank-normalized split R-hat `1.134678` and folded
rank-normalized split R-hat `1.129265`, above the `1.01` tuning gate. That
kernel is rejected. Candidate seed 2 with `L=5` passed its finite short screen
at selected step size `0.249446` and descriptive acceptance `0.763694`, but its
required long verification was not run: the canary-anchored forecast was
`15,546.542 s` plus `180 s` closeout while only `12,540.174 s` remained.

The resource stop is a continuation veto for this campaign. It is not a
convergence failure for `L=5`, a rejection of transport seed 2, a rejection of
transport seed 3, or evidence against the NeuTra direction. `L=10`, `L=15`,
transport seed 3, canonical sequential sampling, and posterior-predictive tests
were not run. The predictive reserve was not borrowed.

The post-run budget audit found that the HMC runner's prior-wall input omitted
the `9.5 s` failed preflight and `10.3 s` launch-invalid canary. It therefore
overstated the HMC remainder by `19.8 s`. Correct cumulative GPU wall is
`12,499.045 s`, not the successful-launch subtotal `12,479.245 s`. No campaign
cap was breached, and the corrected smaller remainder leaves the `L=5`
resource refusal unchanged. This is an engineering accounting defect in the
terminal artifact's budget block, not a target, sampler, or result-integrity
failure. The immutable artifact is preserved rather than silently rewritten.

Terminal records:

- `docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-result-2026-08-19.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-reset-memo-2026-08-19.md`
- `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc/result.json`
- `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc/manifest.json`
- `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc/artifact-hashes.json`
