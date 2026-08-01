# Zhao-Cui Coupled Nonlinear Blockwise TTSIRT-APF Rung-2 Plan

Date: 2026-07-22

Status: `PASS_ENGINEERING_RUNG2`

Route classification: `extension_or_invention`

## Research Intent

| Field | Contract |
| --- | --- |
| Main question | Can a training-base fitted squared-TT represent a genuinely coupled nonlinear 2D adjacent target well enough that twelve independently replicated blocks form a finite, non-collapsed `d=24` fixed-proposal APF with an analytical score of the same finite scalar? |
| Candidate mechanism | Project 2D initial and 4D `(x_previous,x_current)` square-root targets on fixed quadrature clouds, compress by deterministic TensorFlow TT-SVD, apply L1-aware density refinement, carry the fitted current-state marginal by paired-core contraction, compile a fixed conditional KR proposal through a Gaussian-quantile coordinate map, freeze a weight-aware predictive auxiliary genealogy at the reference parameter, then combine twelve blocks under one shared ancestor genealogy. |
| Expected failure mode | Four-coordinate fit or numerical inverse error compounds over twelve blocks, or even the frozen weight-aware predictive auxiliary law cannot prevent ESS collapse, causing failure despite finite scalar mechanics. |
| Promotion criterion | All training, retained-marginal, measure, conditional-density, and inverse gates pass; selected validation and untouched audit diagnostics are finite; candidate same-scalar analytical score/FD max error `<=0.05`; minimum ESS fraction `>=0.20` at `d=24,T=3,N=512`; GPU/XLA placement and memory growth are verified. |
| Promotion veto | Audit leakage into tuning; missing L1 selection; stale ALS decision path; non-finite target/training/map/value/score; zero defensive mass; wrong reference/physical measure; wrong ancestor law; generic retained tensor-product grid; conditional proposal RMS log-density error `>0.75`; score/FD failure; predictive fitted ESS fraction below `0.20`; missing GPU or memory-growth verification. |
| Continuation veto | The normalized 2D/4D target cannot be represented as a finite full-support density, the exact conditional comparator is algebraically invalid, or the fixed-branch scalar lacks a tractable exact parameter score. A failed rank/L1 candidate is only a repair trigger. |
| Repair trigger | Unstable cross-order target log normalizers trigger coordinate-scale/design repair; fit failure triggers fresh rank/degree/L1 tuning; downstream collapse with valid scalar blocks triggers predictive auxiliary or larger-block repair. |
| Explanatory diagnostics | Calibration/validation/audit cross-entropy, centered log-shape RMS, cross-order target log-normalizer agreement, ranks, conditional log-density RMS/tails, ESS, log-weight spread, compile/warmed time, and allocator bytes. |
| Must not be concluded | No source-faithful Zhao-Cui filter, Austria SIR, NAWM, exact pseudo-marginal likelihood, HMC convergence, production KR closure, default readiness, or statistical superiority. |

## Model And Exact Comparator

One synthetic block has state `x=(s,i)` and parameters
`theta=(log_kappa_scale,log_nu_scale,observation_offset)`. Its transition mean
is the nonlinear SIR-inspired Euler map

```text
infection = kappa * exp(theta[0]) * s * i
recovery  = nu * exp(theta[1]) * i
m_s       = s - delta * infection
m_i       = i + delta * (infection - recovery)
```

with a fixed positive-definite Gaussian process covariance. The observation is
`y = i + theta[2] + epsilon`, with fixed Gaussian variance. The initial state
is Gaussian with fixed mean and covariance. This is a BayesFilter synthetic
nonlinear target, not a Zhao-Cui model reproduction.

Although the transition mean is nonlinear in the previous state, conditional
on a selected previous particle the current prior is Gaussian and the
observation is linear-Gaussian. Therefore the exact predictive likelihood and
the exact fully adapted Gaussian conditional are available by a two-dimensional
Kalman update. The matched ladder is:

1. exact conditional with exact predictive auxiliary probabilities;
2. exact conditional with uniform auxiliary probabilities; and
3. fitted TTSIRT conditional with uniform auxiliary probabilities; and
4. fitted TTSIRT conditional with the frozen reference-parameter predictive
   auxiliary probabilities.

All arms use the same model, observations, particles, fixed base random
numbers, scope, and online dtype. The auxiliary law and proposal determine the
realized genealogy and states, so those outputs are not asserted to be
identical across arms. Arm 1 is an oracle ceiling; arm 2 isolates the effect of
uniform genealogy with the exact proposal; arm 3 preserves the rejected
uniform-genealogy fitted comparator; and arm 4 is the candidate. Cross-arm
numerical differences are descriptive only. The primary evidence is arm 4 ESS
plus the analytical derivative of that same fixed-branch scalar.

Twelve identical model blocks are replicated to state dimension `24` and
observation dimension `12`. Proposal uniforms are independent by block, while
all blocks use the same ancestor genealogy. This intentionally tests
within-block coupling and high-dimensional weight composition without claiming
cross-block model coupling.

## Zhao-Cui Operation Classification

| Operation | Anchor | Classification |
| --- | --- | --- |
| Adjacent target and retained previous marginal | Zhao-Cui Eqs. (9)-(11), `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:339`; pinned `models/full_sol.m:72-120` | `source_faithful` operation |
| Squared-TT plus defensive mass | Zhao-Cui Eq. (13), paper line 539; pinned `deep-tensor.dev/src/SIRT.m:74-85` | `source_faithful` operation; tuned scalar is local |
| Paired-core marginal and generic conditional KR | Zhao-Cui Proposition 2/KR construction, paper lines 592-670; pinned `@TTSIRT/marginalise.m:19-85` and `@TTSIRT/eval_cirt_reference.m:43-100` | `source_faithful` operation |
| Freeze samples, settings, ranks, and schedules | Zhao-Cui Algorithm 3, paper lines 890-924; pinned `models/full_sol.m:21-43` | `fixed_hmc_adaptation` |
| Synthetic nonlinear model, `(previous,current)` reorder, Gaussian-quantile map, quadrature TT-SVD initializer, tensor-product training design, numerical grid inverse, block product, APF scalar, and analytical score | Not the cited author route | `extension_or_invention` |

The assembled route is `extension_or_invention`. Source-grounded constituent
operations do not upgrade the whole route to source-faithful status.

## Training And Tuning Protocol

All fitting is TensorFlow float64 on CPU. The historical ridge ALS path is not
eligible for the rung-2 decision. Each target uses the opt-in batched
training-base squared-TT optimizer with no sample-wise map or NumPy numerical
path.

The initial target is the normalized initial posterior. At each later time,
the target relative to the tensor-product uniform reference measure is

```text
p_hat_previous_reference(z_previous)
* f_physical(x_current | x_previous)
* g_physical(y_t | x_current)
* |dx_current/dz_current|
/ lambda_current(z_current).
```

`p_hat_previous_reference` is obtained by paired-core marginalization of the
previous fitted adjacent density. Each target is normalized on its own fixed
design before training. The current-state marginal is then carried to the next
time step. No generic retained tensor-product grid is used as an online
filtering route.

Cloud roles are disjoint:

- calibration: Gauss-Legendre order `8` per active axis;
- validation/selection: Gauss-Legendre order `9` per active axis; and
- final audit only: Gauss-Legendre order `10` per active axis.

The debug smoke may use orders `3/4/5` and is nonclaiming.

Stage A is a structural screen with `l1_weight=0` as a comparator only:

- degree in `{4,6}`;
- TT-SVD internal rank cap in `{8,12}`;
- Gaussian-quantile scale in `{0.20,0.22}` after the v1 algebraic-map route
  sampled states as extreme as `i=89.1` and `s=-65.0` and failed with
  conditional RMS `1707.8` and ESS fraction `0.00195`;
- defensive mass fixed initially at `1e-6`, with cross-order log-normalizer and tail
  diagnostics that trigger a fresh tau/scale repair rather than promotion.

Stage B repeats the selected structure with `l1_weight` in
`{0,1e-6,1e-5}`. Every arm uses the same fixed quadrature clouds, seeds, and
deterministic TT-SVD initialization rule. The initial target is identical, but
later sequential targets are not identical across L1 arms: each arm carries
its own fitted current-state marginal into the next adjacent target. Learning
rate `3e-4`, L2 weight `1e-8`, gradient clip `10`, and 32 density-refinement
steps are target-specific hypotheses, not defaults. Validation excess
cross-entropy, computed as quadrature `KL(target || fitted)`, selects or vetoes;
raw differential cross-entropy is explanatory only because its target-entropy
term changes with the coordinate scale. Audit data are not evaluated until the
structure and L1 arm are frozen. A positive-L1 arm must improve the zero-L1
arm's maximum validation KL by at least
`max(0.005, 0.02*abs(zero_l1_metric))`; otherwise the zero-L1 comparator is
selected while preserving L1 tuning as the required procedure.

The selected frozen cores, coordinate scale, rank, degree, defensive mass,
training traces, calibration/validation roles, and untouched audit result form
the tuning artifact for this exact scope. They are not transferable defaults.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Nonlinear SIR-inspired Euler block | smallest coupled model with exact Gaussian conditional | isolates nonlinear adjacent fitting from conditional-oracle uncertainty | too easy or unrepresentative | nonlinear-vs-linear target difference and explicit nonclaim | diagnostic invention |
| Twelve independent 2D blocks | rung-1 composition plus within-block coupling | keeps `d=24` while moving from scalar to 4D adjacent fits | hides cross-block dependence | later coupled-across-block rung | diagnostic hypothesis |
| `T=3,N=512` | one extra particle rung over rung 1 | detects immediate sequential collapse within bounded cost | too short for long-horizon claims | explicit nonclaim and per-time ESS | convenience hypothesis |
| Gaussian-quantile scales `{0.20,0.22}` | v1 tail failure plus bounded v2 discriminator | maps the uniform defensive reference to a Gaussian full-support physical law matched to the model noise scale | a fixed center/scale may still miss nonlinear posterior geometry | cross-order log-normalizer, realized state extrema, and conditional audit | repaired hypotheses |
| Degree/rank grid `{4,6}` x `{8,12}` | deterministic TT-SVD discriminator | tests materially different polynomial and separation capacity without optimizer-init confounding | insufficient degree/rank or TT-SVD projection bias | validation metric, discarded singular-value mass, and conditional-density audit | repaired hypotheses |
| TT-SVD initializer | fixed tensor-product quadrature projection in the orthonormal Legendre basis | the v1 random-core 4D prefit remained near loss `1.0`; TT-SVD gave one-block RMS `0.339-0.479` and ESS `0.668-0.671` | dense coefficient tensor is limited to this 4D block compiler and is not a generic high-dimensional route | realized ranks, discarded singular-value mass, validation KL | extension hypothesis |
| L1 grid | Zhao-Cui lane policy, target-specific magnitudes | includes zero comparator and positive penalties | penalties ineffective or destructive | recorded regularization and validation-KL differences | hypotheses |
| Training steps/LR | P75/P86 optimizer surface with much smaller target | bounded first ladder | undertraining or overshoot | trace at every 8 steps and final/best ratio | hypotheses |
| Uniform auxiliary fitted comparator | rung-1 mechanism isolation | tests the fitted proposal before the predictive-auxiliary repair | avoidable ESS collapse | exact-uniform arm and fitted-predictive arm separate mechanisms | deliberate baseline |
| Frozen fitted predictive auxiliary | v2 CPU repair after uniform fitted ESS `0.1166` | uses reference previous weights plus predictive observation likelihood while keeping the online branch parameter-independent | reference-anchor mismatch or residual genealogy collapse | fitted-uniform vs fitted-predictive ESS and same-scalar score | repaired hypothesis |
| Grid-CDF inverse | existing diagnostic implementation | only available fixed conditional inverse | density/sample mismatch | exact-conditional log-q and roundtrip gates | diagnostic extension |
| Float32 TF32 XLA online | repository default | required production-oriented execution target | cancellation or unsupported op | CPU precheck, score/FD, device and finite gates | reviewed execution default |

## Evidence Contract

The exact baseline is generated by the same nonlinear transition mean and
Gaussian covariance used by the candidate model. Fit/validation/audit metrics
may select, explain, or veto a proposal but cannot replace the downstream APF
ESS and same-scalar score gates. Acceptance, runtime, and cross-arm value
differences are explanatory only. A one-seed pass establishes engineering
viability at this rung, not stochastic superiority.

The candidate conditional-density audit evaluates heldout pairs against the
analytical Gaussian `p(x_t|x_previous,y_t)` at the exact rounded states used by
the online branch. It reports weighted RMS, q95, q99, and maximum absolute log
density error. RMS `>0.75` is a candidate veto; tail quantities are explanatory
at this sample size.

Artifacts will be written under
`docs/benchmarks/artifacts/zhao_cui_coupled_nonlinear_ttsirt_apf_rung2_20260722/`
using a fresh directory for every attempt.

## Skeptical Audit

- Wrong baseline: avoided by exact predictive-auxiliary and exact
  uniform-auxiliary arms at the same nonlinear scope.
- Proxy promotion: validation/audit KL does not replace ESS or score gates.
- Missing stop conditions: non-finite training, target-design collapse,
  conditional mismatch, score failure, ESS failure, budget exhaustion, and
  missing GPU policy are explicit stops.
- Unfair comparison: all four arms share observations, particles, model, fixed
  base randomness, dtype, and horizon; only the declared proposal and
  auxiliary mechanisms differ. Realized genealogies and states are allowed to
  differ as consequences of those declared mechanisms.
- Hidden assumptions: capacity, scale, L1, training schedule, auxiliary law,
  and block factorization are recorded as hypotheses rather than defaults.
- Stale context: rung 1 proved only independent scalar Gaussian composition;
  it is not used as nonlinear evidence.
- Environment mismatch: fitting is intentionally CPU float64; only the online
  claim arms use GPU float32 TF32 XLA, with an explicit CPU precheck.
- Artifact fitness: the structured result records tuning roles, frozen cores'
  identity, target/model scope, score/ESS gates, device, allocator bytes, and
  nonclaims.

The skeptical audit passed after repairing audit leakage, the invalid unit-mass
gate, adjacent Jacobian ownership, retained-marginal axes, L1 selection, and
exact-branch randomness layout. The user authorized this bounded scope on
2026-07-22 and required shared-GPU memory growth. Debug attempt 6 passed every
harness/correctness gate: inverse roundtrip max error `4.87e-7`, same-scalar
score/FD max error `6.90e-4`, finite CPU-XLA execution, frozen-audit isolation,
and deterministic repeatability. Its deliberately tiny degree-2/rank-2 fit
failed the separate candidate screens and is not candidate evidence. The
v1 canonical CPU precheck then rejected the algebraic/random-core candidate:
conditional RMS `1707.8`, ESS fraction `0.00195`, while the exact-uniform arm
retained `0.5297`. This did not invalidate the scalar, score, inverse, or XLA
harness. Bounded repair diagnostics found that the Gaussian-quantile map plus
fixed TT-SVD initialization passes all one-block numerical screens. The v2 CPU
precheck repaired the proposal fit but rejected the fitted uniform-genealogy
arm at ESS fraction `0.1166`. The v3 predictive-auxiliary repair passed the CPU
precheck and the terminal GPU/XLA run. The GPU candidate reached ESS fraction
`0.25457`, conditional log-density RMS `0.30145`, score/FD maximum error
`1.93e-4`, and inverse roundtrip maximum error `2.98e-6`; all declared gates
passed with memory growth configured and verified before logical-device
initialization.

## Pre-Mortem

The run could pass misleadingly because replicated independent blocks are much
easier than a genuinely coupled 24D target, because `T=3` hides long-horizon
collapse, or because the numerical inverse passes roundtrip while sampling a
slightly different density. These are handled by strict nonclaims and the
exact conditional-density audit.

The run could fail for tuning rather than scientific reasons if the coordinate
scale under-resolves mass, the short optimizer schedule undertrains, or the L1
grid is ineffective. Cross-order log-normalizer, trace, validation, and conditional-density
diagnostics distinguish those repair triggers from failure of the fixed-TT/APF
architecture.

## Budget And Stop Conditions (Closed)

- Debug implementation/smoke: at most 6 CPU-hidden harness-repair attempts,
  still bounded by 2 minutes total. Attempts 1--4 exposed only empty-theta,
  vector-diagnostic, and keyword-only harness defects; none is candidate
  evidence.
- Each canonical attempt: at most 8 Stage-A structures and 3 Stage-B L1 arms.
- Canonical CPU training/precheck budget: 10 minutes total, one repaired
  infrastructure retry within the same contract.
- Trusted GPU/XLA claim budget: 2 minutes, one attempt after occupancy check.
- Preserve every attempt in a unique output directory.
- User authorization was received on 2026-07-22; stop only if a scientific
  veto fires or the declared compute budget is exhausted.

The terminal GPU attempt completed in `119.63 s`, within the declared two-minute
budget. This scope is closed; any multi-seed or longer-horizon work requires a
fresh plan, tuning scope, budget, and versioned output root.

## GPU And Shared-Device Policy

The canonical claim runner must set `TF_FORCE_GPU_ALLOW_GROWTH=true` before
TensorFlow import, configure and verify memory growth on every visible physical
GPU before logical-device initialization, disable whole-device preallocation,
record allocator current/peak bytes, and fail closed on missing placement or
policy. Offline training and branch compilation remain on CPU. Before launch,
record `nvidia-smi` occupancy and defer while the shared device is materially
busy.

## Commands Executed

Nonclaiming debug smoke:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python \
  docs/benchmarks/run_zhao_cui_coupled_nonlinear_ttsirt_apf_rung2.py \
  --output-root /tmp/zhao_cui_coupled_nonlinear_rung2_debug \
  --debug-smoke --cpu-reference
```

Canonical v3 CPU precheck after the predictive-auxiliary repair:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python \
  docs/benchmarks/run_zhao_cui_coupled_nonlinear_ttsirt_apf_rung2.py \
  --output-root docs/benchmarks/artifacts/zhao_cui_coupled_nonlinear_ttsirt_apf_rung2_20260722/cpu_reference_attempt03 \
  --cpu-reference
```

Canonical trusted GPU claim after the CPU and occupancy gates passed:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONDONTWRITEBYTECODE=1 python \
  docs/benchmarks/run_zhao_cui_coupled_nonlinear_ttsirt_apf_rung2.py \
  --output-root docs/benchmarks/artifacts/zhao_cui_coupled_nonlinear_ttsirt_apf_rung2_20260722/gpu_attempt01
```
The v2 CPU precheck repaired target/map/fit quality but the uniform fitted arm
still reached only ESS fraction `0.1166`; its fitted predictive auxiliary
diagnostic on the debug scope reached `0.6472`. The v3 repair therefore used
the frozen predictive auxiliary arm as the candidate and retained the uniform
arm as an explanatory comparator.

The v3 CPU precheck passed all declared gates at `d=24,T=3,N=512`: selected
degree `6`, TT-SVD rank cap `12`, Gaussian scale `0.22`, and `l1_weight=0.0`
after the complete L1 grid; validation KL `0.00264`, untouched audit KL
`0.00496`, cross-order log-normalizer spread `0.00856`, conditional log-density
RMS `0.30145`, predictive fitted ESS fraction `0.25457`, same-scalar score/FD
error `2.69e-4`, and inverse roundtrip error `2.98e-6`. Positive L1 did not
meet the predeclared `0.005` improvement margin. The subsequent GPU/XLA run
reproduced the pass with predictive fitted ESS fraction `0.25457`, same-scalar
score/FD error `1.93e-4`, `/GPU:0` placement, TF32/XLA enabled, and memory
growth verified on `/physical_device:GPU:0`. The terminal decision is
`PASS_ENGINEERING_RUNG2`; this does not transfer the selected controls to a new
scope or establish any of the stated nonclaims.
