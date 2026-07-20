# SSL-LSTM Matrix-Free Filter Derivatives Plan

Date: 2026-07-20  
Status: `PHASE_6_COMPLETE_NO_DOWNSTREAM_RERUN_NOMINATION`  
Tier: Tier 2 material research engineering

## Question And Motivation

Can BayesFilter remove the dense local state-Jacobian bottleneck from the
selected-direction SSL-LSTM score without changing the fixed UKF, Fixed-SGQF,
or fixed-replay scalar and score?

For the state-complexity rung `latent_dim=hidden_dim=q`, the augmented state
has dimension `n=3q`, the structural UKF integrates over `a=4q` augmented
state/innovation coordinates, and it uses `R=8q+1` points.  The current
selected four-direction score constructs a pointwise transition Jacobian with
shape `[R,n,n]` before multiplying it by a tangent tensor with shape
`[p,R,n]`, where `p=4`.  At `q=20`, the Jacobian alone contains 579,600
float64 entries at every filtering step.  The SSL-LSTM construction performs
dense gate-level matrix operations for all `n` output directions, so the
transition derivative can scale as `O(T R q^3)=O(T q^4)` even though only four
directional products are required.

The candidate computes those products directly.  It propagates the selected
state tangents through the LSTM gates and adds the innovation and direct
parameter tangents without ever forming `[R,n,n]`.  For fixed `p`, the local
transition work becomes `O(p R q^2)` and the Gaussian covariance/factor
derivative work remains `O(p q^3)`, giving an expected overall selected-score
order of `O(T p q^3)`.  This is an operation-count argument, not a prospective
runtime claim.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Does a matrix-free local JVP preserve the current fixed-filter value and selected score while removing dense SSL-LSTM state Jacobians? |
| Exact baseline | Current dense-Jacobian UKF, Fixed-SGQF, and fixed Zhao-Cui replay implementations on identical parameters, observations, rules/clouds, branches, dtype, and selected directions. |
| Candidate mechanism | Optional batched local JVP callbacks on the existing derivative contracts, with dense callbacks retained as compatibility fallbacks. |
| Expected failure mode | Wrong tangent-axis convention, missing direct parameter term, wrong innovation injection, gate-chain-rule error, XLA shape failure, or accidental dense fallback. |
| Promotion criterion | Dense/JVP local products and end-to-end values/scores agree within declared float64 tolerances; finite differences pass; the matrix-free route raises no dense-Jacobian sentinel; focused eager/XLA tests pass. |
| Promotion veto | Any scalar/score mismatch, finite-difference failure, non-finite output, branch change, dense sentinel invocation, or public compatibility regression. |
| Continuation veto | Invalid test fixture, corrupted source artifact, unresolved concurrent edit in an in-scope file, or inability to reproduce the baseline. |
| Repair trigger | A focused parity failure triggers inspection of the smallest local JVP term before any benchmark or broader filter migration. |
| Explanatory only | Tensor counts, asymptotic operation counts, wall time, RSS, allocator memory, and compile time. |
| Must not conclude | Posterior correctness, filter approximation accuracy, HMC readiness, NeuTra quality, runtime superiority, q=20 admission, or production/default readiness. |

## Evidence Contract

The scientific/engineering question is local derivative equivalence and
elimination of dense state-Jacobian materialization.  The comparator is the
existing dense analytic route, not finite differences alone and not a weaker
filter.  The primary pass criterion is same-input float64 value/score parity
plus finite-difference agreement.  Non-finite output, changed fixed branch,
failed adjoint identity where a VJP exists, or any invocation of a guarded
dense SSL-LSTM Jacobian vetoes the candidate.  Runtime and memory are
explanatory in this pass.  The result is preserved in the result note beside
this plan; no HMC or model-quality conclusion follows from a pass.

## Source And Boundary Classification

- UKF and Fixed-SGQF local JVP wiring is BayesFilter numerical engineering.
- The fixed SSL-LSTM Zhao-Cui adapter is already classified
  `fixed_hmc_adaptation_with_extension_likelihood`; its fixed replay and
  recentering vocabulary is anchored to Zhao--Cui JMLR 2024 Sections 1--3 and
  5 and to
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/computeL.m:24-47`.
  Replacing `J @ tangent` by the algebraically identical JVP does not alter the
  source route.  It remains a `fixed_hmc_adaptation` derivative implementation
  detail and creates no source-faithful SSL-LSTM claim.
- LEDH already exposes manual streaming JVP/VJP operators in
  `bayesfilter/highdim/ledh_contract_e_streaming_tf.py` and manual moment/root
  JVP/VJP primitives in `ledh_contract_e_reset_tf.py`.  This pass documents
  their conformance but does not rewrite or relabel the LEDH algorithm.

## Phases

### Phase 1: Self-Contained Mathematical Documentation

Add a section to the SSL-LSTM chapter deriving the tangent recursion, mapping
the generic JVP/VJP design to UKF, Fixed-SGQF, fixed Zhao-Cui replay, and LEDH,
and separating algebraic equivalence from runtime and scientific claims.

### Phase 2: Backward-Compatible Derivative Contract

Add optional transition and observation JVP callbacks to the UKF and
Fixed-SGQF derivative records.  Central helpers choose the JVP when supplied
and otherwise retain the current dense Jacobian path.  Do not remove, rename,
or change existing callback semantics or public exports.

### Phase 3: SSL-LSTM Direct JVP

Implement TensorFlow float64 batched transition-state and observation-state
JVPs in the SSL-LSTM adapter.  Wire the UKF and Fixed-SGQF adapters to combine
state propagation, innovation propagation where applicable, and the same
direct parameter derivatives as the baseline.

### Phase 4: Fixed Replay Migration

Replace dense state-Jacobian multiplication in the existing fixed replay with
the direct SSL-LSTM state JVP.  Preserve fixed seeds, particles, likelihood,
weights, recentering, manifest classification, and artifact schema.

### Phase 5: Focused Verification

Required checks:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  tests/test_ssl_lstm_matrix_free_derivatives_tf.py \
  tests/test_ssl_lstm_sgqf_ukf_adapters.py \
  tests/test_ssl_lstm_zhaocui_fixed_adapter.py -q

CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_fixed_sgqf_scores_tf.py -q

python -m compileall \
  bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py \
  bayesfilter/nonlinear/fixed_sgqf_derivatives_tf.py \
  bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py \
  bayesfilter/nonlinear/ssl_lstm_zhaocui_fixed_adapter.py

cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The new focused suite must include:

- dense Jacobian product versus direct JVP at `q in {1,2,5}`;
- central finite differences for state directions;
- end-to-end UKF selected-score dense-fallback/JVP parity;
- end-to-end Fixed-SGQF and fixed-replay parity on small fixtures;
- eager/XLA parity for the direct local JVP;
- a sentinel proving the selected UKF and fixed-replay JVP routes do not call
  `ssl_lstm_transition_state_jacobian`;
- shape checks through `q=20` that do not constitute a timing or admission run.

## Resource Stop

This pass is CPU-hidden focused verification only and should stop before any
command expected to exceed five minutes.  It authorizes no GPU benchmark,
HMC, NeuTra training, q=20 ladder rerun, or long stochastic acquisition.  A
later trusted GPU/XLA benchmark requires a refreshed result contract with
allocator/RSS evidence and uncertainty-aware repeated timings.

## Skeptical Pre-Execution Audit

The plan was audited for wrong baselines, proxy promotion, hidden assumptions,
unfair comparisons, missing stops, environment mismatch, and artifacts that
would not answer the question.

1. The baseline is the identical fixed scalar and derivative engine; no filter
   change is hidden inside the optimization.
2. Finite differences are a veto/check, not the sole comparator.  Dense
   analytic parity and end-to-end score parity are required.
3. The asymptotic argument is not treated as observed speedup or q=20
   admission.  No performance promotion occurs in this pass.
4. The plan does not assume a single filter-wide optimizer.  It shares local
   JVP semantics while preserving filter-specific covariance, factor,
   weighting, transport, and branch recursions.
5. Forward JVP is appropriate only because the active complexity target has
   four selected directions.  Full-chart SSL-LSTM inference may require a
   reverse/VJP filter recurrence and is explicitly not declared solved.
6. Zhao-Cui route identity is preserved and source-faithfulness is not claimed.
7. LEDH is not needlessly rewritten; existing streaming JVP/VJP work is the
   starting point for later interface unification.
8. Dense fallbacks protect other models and public callers.  Compatibility
   tests are required before the phase can close.
9. CPU hiding is explicit for debug/reference tests; GPU/XLA performance
   evidence is deferred to a separately recorded trusted run.

Audit verdict: `PASS_AFTER_SCOPE_REPAIR`.  The original broad architecture was
narrowed to the smallest implementation that directly addresses the observed
selected-direction bottleneck without overclaiming full reverse-mode or
filter-family unification.

## Result Artifact

`docs/plans/bayesfilter-ssl-lstm-matrix-free-filter-derivatives-result-2026-07-20.md`

## Phase 6: Trusted GPU/XLA Dense-versus-JVP Benchmark

### Question And Exact Comparator

On the identical four-direction principal-root UKF target, does the admitted
matrix-free SSL-LSTM JVP produce a large enough observed engineering effect to
justify rerunning the q=20 target/NeuTra capacity preflight?

The baseline is the current derivative object with only
`transition_jvp_fn=None` and `observation_jvp_fn=None`; this forces the existing
dense Jacobian products while preserving the same fixture, observations,
parameter point, direct parameter derivatives, sigma-point rule, covariance
and principal-root recursion, float64 dtype, GPU, XLA setting, and source
revision.  The candidate differs only by retaining the admitted JVP callbacks.

### Prospective Evidence Contract

| Role | Phase-6 contract |
| --- | --- |
| Primary hard pass | Every dense/JVP pair at q=5,10,20 has finite values/scores and maximum same-point value/score absolute difference `<=1e-10`. |
| Performance nomination | Nominate a bounded q=20 downstream rerun only if the q=20 median fresh-process warm-time ratio `JVP/dense <=0.80`, neither q=20 allocator peak nor host high-water RSS exceeds dense by more than 10%, and no timing cell is contaminated. |
| Small-rung guard | At q=5 and q=10, a ratio above 1.10 is recorded as an observed crossover/regression and triggers explanation, but does not invalidate q=20 because fixed overhead may dominate small rungs. |
| Hard vetoes | Non-finite output, parity failure, wrong device/XLA/dtype, memory-growth failure, allocator peak above 28 GiB, host high-water RSS above 64 GiB, subprocess timeout above 1,200 seconds, malformed/missing artifact, or source drift. |
| Timing contamination | Prelaunch physical-GPU utilization above 50%, a new foreign compute PID on the selected GPU, or device mismatch makes that cell timing explanatory-only and blocks downstream nomination. It does not turn a correct candidate into a numerical failure. |
| Explanatory only | First-call/compile wall, individual warm calls, allocator current/peak, process VmRSS/VmHWM, `ru_maxrss`, and all ratios from three repetitions. |
| Not concluded | Statistical superiority, q=20 admission, end-to-end NeuTra/HMC acceleration, posterior correctness, full-chart scalability, or a filter-family default. |
| Artifact | `docs/plans/artifacts/ssl-lstm-matrix-free-filter-derivatives-2026-07-20/gpu-xla-benchmark/summary.json` plus per-cell JSON/log files and the refreshed result note. |

Each `(q, arm)` runs in a fresh process because TensorFlow allocator peaks and
XLA caches cannot be reset into a genuinely independent in-process baseline.
Use three paired repetitions, alternating arm order by repetition, and five
synchronized warm evaluations per process.  Evaluate the same center and
fixed shell points in every arm.  Run q in increasing order and stop only on a
hard veto; a small-q performance miss is not a continuation veto for q=20.

### GPU And Resource Contract

- Prefer physical GPU 1 only when it has no foreign compute process; otherwise
  use physical GPU 0.  The 2026-07-20 preflight found a foreign Python process
  on GPU 1, so Phase 6 must use GPU 0 and not interfere with the other lane.
- Set `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import and call
  `set_memory_growth` before logical-device initialization.
- Enable XLA JIT and TF32 policy, while retaining float64 tensors.
- Record the trust basis
  `owner_designated_managed_session_visible_gpu_trusted`.
- Run no HMC, NeuTra training, package operation, network action, or default
  change.

### Phase-6 Skeptical Audit

1. The earlier q=20 isolated target used only about 0.31 GiB TensorFlow
   allocator peak and 3.03 GiB process RSS; therefore the historical 36 GiB
   observation is not assumed to be live GPU tensor memory.  Both allocator
   and host metrics are recorded, and wall time is the main practical question.
2. Comparing a local JVP microkernel would not answer whether the complete
   selected score improves.  The benchmark executes the full identical
   principal-root UKF value/score.
3. Running arms in one process would bias allocator and compilation evidence.
   Fresh subprocesses and alternating order repair that baseline flaw.
4. Three repetitions cannot statistically rank noisy GPU timing distributions.
   Ratios are descriptive nomination evidence only; no superiority language is
   authorized.
5. GPU 1 is excluded while another lane owns it.  GPU 0 background graphics
   are measured prospectively, and high-utilization cells cannot nominate a
   downstream run.
6. A q=5 slowdown can be a fixed-overhead crossover.  It does not silently
   become a continuation veto for q=10 or q=20.
7. A large q=20 effect still cannot admit q=20 or establish end-to-end NeuTra
   benefit; it can only justify the next bounded target/NeuTra capacity check.

Audit verdict: `PASS_WITH_FRESH_PROCESS_AND_CONTAMINATION_REPAIRS`.
