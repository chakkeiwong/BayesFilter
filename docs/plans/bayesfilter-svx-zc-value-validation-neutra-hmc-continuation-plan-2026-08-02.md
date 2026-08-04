# SVX-ZC Value Validation, NeuTra, And HMC Continuation Plan

Date: 2026-08-02

Status: `COMPLETE`

## Objective

Complete the nominated SVX-ZC value-capacity campaign, treating score values as
diagnostics only, then conditionally run target-specific batched NeuTra and the
shared sequential HMC controller for the same frozen finite likelihood.

The value target is the promotion question. Score magnitude, score direction,
and finite-difference agreement are recorded to explain downstream mechanics,
but they do not veto value validation or select a value capacity.

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Does `(degree=10, rank=2, order=25)` remain value-stable at the frozen local validation points, and can that exact finite program support target-bound NeuTra/HMC mechanics? |
| Candidate | SVX-ZC `zhao_cui_fixed_adjacent_state_squared_tt_v1`, UKF initialized, degree 10/rank 2/order 25. |
| Expected failure modes | Value prefix or numerical invariant failure; missing target identity; scalar-only target path; training status failure; HMC health/convergence failure; branch-sensitive score diagnostics. |
| Value promotion criterion | Center quadrature and all four frozen validation points pass the requested first-two-significant-digit likelihood prefix against degree, rank, and order neighbors, with finite/mass/positivity/conditioning invariants. |
| Value veto | Nonfinite value/increment, mass or positivity failure, conditioning failure, frozen-scope drift, or failed value-prefix comparison. |
| Score role | Diagnostic only. Report finite score, finite-difference agreement, branch hashes, componentwise sign/direction, score norm, and capacity deltas. Do not reject a value candidate because score changes by 20% if the overall direction remains consistent. |
| NeuTra promotion criterion | A repository-issued target signature and independently tested batch-native value/status adapter, scope-bound tuning artifact, valid GPU/XLA training artifact, and frozen transport parity. |
| HMC promotion criterion | Same target adapter and frozen transport, shared sequential controller, finite/status/energy diagnostics, warm-up readiness, modern R-hat, declared ESS, and downstream posterior gates. |
| Continuation veto | Invalid value harness/artifact, no repository-owned batch-native adapter, target-signature mismatch, corrupted transport, exhausted budget, or a true HMC convergence/health veto. |
| Repair trigger | Localized runner, serialization, adapter batching, GPU/XLA, training, or HMC resource failure under the unchanged scope and budget. |
| Explanatory diagnostics | Score changes, acceptance, training loss, runtime, fit residuals, condition estimates, and branch index changes. |
| Not concluded | Exact likelihood, exact score, statistical superiority, universal capacity, posterior correctness from one run, production readiness, or cross-scope transfer. |

## Frozen value scope

- Model: actual transformed SV / `SVX-ZC`.
- Data seed: `81101`; horizon `T=10`; center
  `[0.2533471031357997, -0.916290731874155]`.
- Validation points: center plus the four predeclared axis offsets `+/-0.05`.
- Dtype: TensorFlow `float64`.
- Value execution: deliberate CPU-only, non-XLA diagnostic route with
  `CUDA_VISIBLE_DEVICES=-1`; UKF initialization is the default.
- Nominee: degree 10, rank 2, quadrature order 25.
- Value neighbors: degree `(12,2,25)`, rank `(10,4,25)`, order `(10,2,33)`.
- Value rule: first two significant digits agree; the third may change.
- Invariants: finite values/increments, marginal mass error `<=1e-10`, queried
  density nonnegative within `-1e-14`, solved condition number `<=1e10`, and
  frozen scope identity equal across compared cells.

## Execution phases

### Phase 1: Value-only validation

Run:

```text
python docs/benchmarks/run_svx_zc_capacity_self_convergence_tuning_20260801.py \
  --output-root docs/plans/artifacts/bayesfilter-svx-zc-value-validation-neutra-hmc-20260802/value-attempt01
```

This is a fresh 35-cell maximum (17 center calibration cells, two center
quadrature cells, and four validation points times four cells). Only the value
and declared numerical invariants control the terminal status.

After the value run, compute score diagnostics for the nominee and the four
validation-point nominees using the existing fixed-branch score function. Score
diagnostics must be written under a separate directory and cannot alter the
value decision. Report the score direction as componentwise sign agreement and
cosine similarity to the center score, not as a hard accuracy claim.

### Phase 1B: HMC-fixed initializer value validation

The completed Phase 1 run exposed an HMC target-identity issue: its UKF warm-
start cores were rebuilt at each validation parameter. Runtime parameter-
dependent initialization would be retuning and is forbidden inside HMC.
Therefore freeze the UKF cores once at the center for every compared capacity,
then repeat the four-point value comparisons with those cores unchanged. This
phase uses the same value-only criterion and invariants. Score remains
diagnostic. Phase 2 cannot issue an HMC target identity unless Phase 1B passes.

### Phase 2: Target identity and batch-native adapter

If Phase 1 returns `SELF_CONVERGED_VALUE`, build a scope-bound adapter around
the exact nominated finite value program. It must:

- bind the data, model, route, capacity, UKF initializer, horizon, dtype, and
  all numerical controls in a repository-issued signature;
- expose finite value, score, and status tensors;
- preserve the leading batch dimension through value, score, status, loss, and
  gradient operations;
- avoid Python sample loops, `tf.map_fn`, `tf.vectorized_map`, and scalar
  fallback for training;
- include an independent posterior total-value/score recomposition with prior,
  chart, and Jacobian; and
- fail closed on stale or mismatched capacity/value artifacts.

Run focused singleton, batch-parity, permutation, finite/status, and signature
tests before any training. If a true batch-native implementation cannot be
proven without changing the mathematical target, stop with an adapter blocker;
do not register a scalar wrapper as NeuTra-ready.

### Phase 3: Scope-specific NeuTra training

Only after Phase 2 passes, use disjoint calibration/validation data and a fresh
scope tuning artifact. Run the repository GPU TensorFlow/XLA path with memory
growth verified before initialization. Use a bounded recipe screen (500 steps,
batch size 128), select one recipe without ranking unsupported stochastic
differences, then run one fresh 5,000-step final training artifact. Score is
diagnostic during training and held-out evaluation; it is not a value-capacity
promotion gate.

### Phase 4: Sequential NeuTra HMC

Only after valid frozen transport and public tuning handoff, use the shared
`bayesfilter.inference.neutra_hmc` sequential controller. Retain warm-up chunks
but exclude them from posterior estimates. Require the repository warm-up,
R-hat, ESS, finite/status, movement, and declared energy-error policies. Record
acceptance as diagnostic only. A health or convergence veto blocks HMC claims;
it does not invalidate the value result.

## Budget and artifacts

- Phase 1: one fresh CPU run, maximum 35 value cells and 30 minutes.
- Phase 1B: one fresh CPU run, maximum 16 validation cells and 30 minutes.
- Score diagnostics: at most five center/validation score calls plus their
  existing finite-difference ladders, separate from value promotion.
- Phase 2: focused tests and one adapter implementation attempt, maximum 60
  CPU minutes.
- Phase 3: one GPU recipe screen and one selected 5,000-step training run,
  maximum 8 GPU hours.
- Phase 4: one bounded sequential HMC campaign with cumulative retained cap
  10,000 per chain, maximum 8 GPU hours.

Fresh roots:

```text
docs/plans/artifacts/bayesfilter-svx-zc-value-validation-neutra-hmc-20260802/
  value-attempt01/
  frozen-initializer-value-attempt01/
  score-diagnostics-attempt01/
  adapter-attempt01/
  neutra-hmc-attempt01/
```

Each serious artifact must include command, git commit/status, environment,
data seed/hash, dtype/device/XLA, memory policy, wall time, plan path, and
nonclaims. No prior artifact is overwritten.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| Value-only promotion | User instruction in this task | Score could be unstable while value is stable | Preserve score direction/norm diagnostics | Reviewed policy |
| 20% log-beta score movement accepted | User instruction, conditional on direction | Large derivative bias could affect HMC | Cosine/sign diagnostics and HMC health gates | Diagnostic tolerance, not accuracy claim |
| Degree-10/rank-2 nominee | Active 2026-08-01 value tuning artifact | Center-only nomination may not validate locally | Four frozen validation points | Warm-start hypothesis until Phase 1 |
| Batch-native adapter requirement | Repository NeuTra policy | Scalar route could be incorrectly promoted | Source inspection plus batch parity/permutation tests | Binding policy |
| GPU/XLA NeuTra | Repository owner directive | CPU-only training would not support claims | Memory-growth/device/XLA manifest | Binding execution target |

## Skeptical plan audit

- Wrong baseline risk is controlled by using the nominated finite value program,
  not a dense or KSC substitute.
- Score is explicitly diagnostic, so a proxy score threshold cannot silently
  become a value promotion criterion.
- Value and HMC have separate continuation vetoes; HMC failure does not erase
  value evidence, and value passage does not waive adapter/HMC gates.
- The plan stops before training if the current scalar implementation cannot
  satisfy batch-native training. This prevents a successful-looking scalar
  loop from being mislabeled as NeuTra evidence.
- The main hidden assumption is that the current finite route can be expressed
  batch-natively without changing its target. The Phase 2 focused source and
  parity checks are the earliest test of that assumption.

Audit verdict: `PASS_FOR_VALUE_EXECUTION_AND_CONDITIONAL_NEUTRA_HMC`; no GPU
training or HMC launch is valid until the value gate and adapter gate pass.

## Execution ledger, 2026-08-02

- Phase 1 passed as `SELF_CONVERGED_VALUE`; 35/35 value cells completed.
- Phase 1B passed as `SELF_CONVERGED_FIXED_INITIALIZER_VALUE`; all 16 cells
  passed with center-built cores held fixed.
- The separate score diagnostic preserved sign `[-,+]` at all five points;
  center-relative cosine similarities were `1.0`, `0.999416`, `0.999341`,
  `0.987775`, and `0.978692`. This remains explanatory only.
- Phase 2 focused tests passed: scalar value parity, finite-difference score,
  batch permutation/status, posterior recomposition/binding, and CPU XLA.
- GPU attempt 01 stopped before training because UKF core construction inherited
  the process device and issued a different target signature after GPU
  initialization. Repair: build the one-time frozen initializer on CPU; the
  resulting signature is device-stable
  `deccdda78028706d0987322d30b9798f0f4d8b518c6773451338e83bf14d1cab`.
- GPU attempt 02 completed the first 500-step batch-128 XLA training job, then
  exhausted GPU memory in an unnecessary flattened 1,024-row held-out score
  diagnostic. Repair: preserve the eight independent 128-row diagnostic
  batches. This changes neither training batch size nor target semantics.
- GPU attempt 03 completed the four-recipe screen. All recipes passed hard
  value/status/parity gates. The one-seed reverse-KL differences are descriptive
  only, so no ranking is supported. The deterministic representative is the
  smallest viable architecture, `svx_zc_narrow_lr1e3`; the screen result SHA-256
  is `97f646904803123c1d487a811595d22b565e450c7f915cd01a4c55a1e1c00e1a`.
- The final continuation must consume that preserved result through the strict
  screen handoff, run exactly one fresh 5,000-step final training, then native
  tuning and shared sequential HMC. It must not rerun or reinterpret the screen.

Renewed skeptical audit: `PASS_FOR_SELECTED_FINAL_TRAINING_TUNING_AND_HMC`.
The comparator is the declared identity-affine baseline; recipe loss has not
been promoted to a statistically supported ranking; stop conditions remain
target/status, training, tuning, and HMC health/convergence vetoes; the selected
screen artifact is hash- and target-bound; GPU/XLA/memory-growth and batch-128
execution were observed directly; and the final artifact will answer the stated
NeuTra/HMC question without changing the value-capacity decision.

### Final-training and tuner repair ledger

- The selected `svx_zc_narrow_lr1e3` final training completed 5,000 GPU/XLA
  updates at batch 128. The held-out 1,024 rows were all valid, with zero
  floors; frozen/trainable value, logdet, pullback, and logdet-score parity
  passed. This admits the frozen transport to HMC tuning but is not convergence
  evidence.
- The legacy anchor tuner then stopped before retained sampling. Its six
  32-draw rounds had finite samples, finite target/log-acceptance values, valid
  target status, zero floors, and no hard vetoes. It nevertheless exhausted its
  repair budget because point estimates of mean acceptance did not fall inside
  the exact `[0.65,0.75]` band. Observed mean acceptance probabilities were
  `0.946716`, `0.941609`, `0.448993`, `0.831063`, `0.608396`, and `0.629035`.
- This is a repair trigger for the already reviewed generic broad-grid route,
  not evidence against the target, transport, or HMC direction. The replacement
  route independently tunes epsilon for `L=(3,5,9,13,18,25)`, screens three
  replications over four chains, applies the frozen 90% replication-mean
  interval compatibility rule, and adds nonrecursive same-epsilon `L+/-1`
  coverage. It preserves the complete viable set as unranked.

Skeptical repair audit: `PASS_FOR_PRESERVED_TRANSPORT_BROAD_GRID`. The target,
transport, data, GPU/XLA backend, fixed-identity mass, acceptance target, total
campaign budget, and downstream convergence gates remain unchanged. Only the
known-wrong small-sample point-estimate acceptance decision is replaced by the
approved statistical protocol. All broad-grid draws are discarded; survival
cannot establish HMC convergence or posterior validity.

- Broad-grid attempt 01 completed the independent L=3 epsilon tune but exhausted
  the 13.5 GiB GPU during its 128-draw four-chain screen. No primary candidate
  completed, so this is shared-execution invalidity, not candidate rejection.
  The target-specific retry uses 65 draws per chain, the smallest integer above
  the reviewed `>64` confirmation boundary. Four chains, three replications,
  primary grid, independent epsilon tuning, interval classification, one-hop
  coverage, fixed transport, and all health gates remain unchanged.
- Broad-grid attempt 02 passed. Exactly one pair survived, so no stochastic
  ranking or discretionary representative choice is needed: independently tuned
  `L=25`, `epsilon=0.8434292653387`, grand mean acceptance `0.787289`, and 90%
  working interval `[0.739104,0.835475]`. The same-epsilon `L=24` coverage row
  did not pass and does not veto its parent.
- Sequential HMC consumes that unique hash-bound pair without retuning. Because
  the 128-draw screen exceeded memory while 65 draws passed, warm-up and retained
  execution use 65-result GPU/XLA chunks. The controller still requires at least
  2,000 warm-up transitions per chain, evaluates the latest 1,000-transition
  window at R-hat `<=1.05`, requires retained modern R-hat `<=1.01` plus declared
  bulk/tail ESS, and retains the 10,000-per-chain caps.

Final pre-sampling audit: `PASS_FOR_SHARED_SEQUENTIAL_HMC`. The kernel is the
only viable unranked pair; the transport and grid artifacts are independently
SHA-256 bound; acceptance is not used as convergence evidence; 65-result tiling
was directly proven on the same GPU and kernel shape; warm-up remains excluded;
and finite/status/movement/R-hat/ESS gates remain active. The run can still fail
scientifically at the convergence or truth-tail gates, which will not invalidate
the completed value or training evidence.

## Terminal interpretation

The final result must state separately whether the value harness passed, whether
the adapter was admitted, whether NeuTra training completed, and whether HMC
converged. A score direction that remains aligned is descriptive evidence only;
it cannot establish exact score correctness or score convergence.
