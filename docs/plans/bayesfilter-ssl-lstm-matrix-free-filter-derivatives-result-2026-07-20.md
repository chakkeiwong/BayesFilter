# SSL-LSTM Matrix-Free Filter Derivatives Result

Date: 2026-07-20  
Status: `MATRIX_FREE_CORRECT_WARM_MEMORY_REDUCED_DOWNSTREAM_RERUN_NOT_NOMINATED`

## Outcome

The selected-direction SSL-LSTM derivative path now applies local transition
and observation derivatives as batched JVPs instead of materializing dense
pointwise state Jacobians.  The same optional contract is available to the
structural UKF and Fixed-SGQF engines, with their previous dense callbacks
retained as compatibility fallbacks.  The fixed SSL-LSTM Zhao-Cui replay now
uses the same direct state and observation JVPs.

The SSL-LSTM chapter now derives the JVP equations, explains the expected
change from quartic to cubic selected-score scaling, distinguishes forward JVP
from reverse VJP, maps the design to UKF, Fixed-SGQF, fixed Zhao-Cui replay,
and LEDH, and states the same-scalar validation and nonclaim boundaries.

## Decision Table

| Decision field | Result |
| --- | --- |
| Decision | Admit the matrix-free local JVP as the SSL-LSTM selected-direction derivative implementation; retain dense fallbacks for other providers. |
| Primary criterion | Pass. Dense products, direct JVPs, finite differences, end-to-end UKF/SGQF/replay values and scores, CPU XLA, and q=20 shapes passed. |
| Veto diagnostics | No non-finite output, parity mismatch, branch change, compatibility failure, or dense-Jacobian sentinel invocation in the admitted paths. |
| Main uncertainty | GPU 0 background utilization contaminated at least one arm in every q=10 and q=20 pair, so timing ratios are descriptive only. |
| Next justified action | Do not rerun the expensive q=20 NeuTra lane from this evidence. Profile/fuse the JVP or obtain a clean dedicated-GPU q=20 comparison before reconsidering. |
| Not concluded | q=20 admission, runtime superiority, posterior correctness, filter accuracy, HMC readiness, NeuTra quality, full-chart scalability, or default/production readiness. |

## Evidence Status

| Evidence question | Status |
| --- | --- |
| Hard veto screen | Passed for the focused fixed computations. |
| Statistically supported ranking | None; no stochastic or repeated runtime ranking was attempted. |
| Descriptive-only differences | q=20 median paired warm JVP/dense ratio `0.898`; warm allocator peak ratio `0.321`; host high-water ratio `0.962`. Timing was contaminated. |
| Default readiness | Not established. The callbacks are optional and the change promotes only the SSL-LSTM selected-direction implementation. |
| Next evidence needed | A fused/profiled JVP repair or uncontaminated dedicated-GPU q=20 timing if an end-to-end rerun is still desired. |

## Implemented Artifacts

- `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py`
  adds optional transition/observation JVP callbacks and validates their output
  before adding the unchanged direct parameter derivative.
- `bayesfilter/nonlinear/fixed_sgqf_derivatives_tf.py`
  adds the analogous optional point-map JVP callbacks with dense fallbacks.
- `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py`
  implements the direct gate, cell, hidden, latent, and observation JVPs and
  wires them to UKF and Fixed-SGQF.
- `bayesfilter/nonlinear/ssl_lstm_zhaocui_fixed_adapter.py`
  replaces dense state/observation Jacobian products by the direct JVPs.
- `tests/test_ssl_lstm_matrix_free_derivatives_tf.py`
  contains the focused parity, finite-difference, sentinel, XLA, and q=20
  shape checks.
- `docs/chapters/ch28a_neural_network_state_space_model_applications.tex`
  contains the self-contained mathematical and filter-family design section.

## Zhao-Cui Source Boundary

The source gate was checked directly after implementation:

- Zhao and Cui, JMLR 2024, Sections 1--3 define the sequential TT/KR route and
  accompanying particle operations; Section 5.2 defines Gaussian linear
  preconditioning from a sample mean and covariance and a Cholesky factor.
- Author snapshot
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/computeL.m:24-47`
  normalizes weights, computes the weighted mean/covariance, takes a regularized
  Cholesky factor, and optionally applies the quantile stretcher.
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:64-70`
  invokes `computeL` and stores the resulting affine frame.

The local derivative change is classified `fixed_hmc_adaptation`: it replaces
`D_x f` times a carried particle tangent by the algebraically identical JVP.
It does not change fixed randomness, the replay likelihood, or recentering and
does not support a source-faithful SSL-LSTM Zhao-Cui claim.  The SSL-LSTM
likelihood remains a BayesFilter extension.

## Verification

### Focused matrix-free suite

Command:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  tests/test_ssl_lstm_matrix_free_derivatives_tf.py -q
```

Result: `10 passed` in 4.73 seconds.  This includes dense/JVP parity at
`q in {1,2,5}`, directional finite differences, UKF/SGQF/replay parity,
dense-call sentinels, q=20 `[4,161,60]` transition-tangent shape, and CPU XLA
parity.

### Focused compatibility suite

Command:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  tests/test_ssl_lstm_matrix_free_derivatives_tf.py \
  tests/test_ssl_lstm_sgqf_ukf_adapters.py \
  tests/test_ssl_lstm_zhaocui_fixed_adapter.py \
  tests/test_ssl_lstm_complexity_target_tf.py \
  tests/test_fixed_sgqf_scores_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py -q --disable-warnings \
  --deselect='tests/test_nonlinear_sigma_point_scores_tf.py::test_model_b_analytic_score_matches_finite_difference[tf_svd_ukf_score-tf_svd_ukf]'
```

Result: `53 passed, 1 deselected` in 178.24 seconds.

The deselected generic historical SVD-UKF case was not hidden as a candidate
failure.  It was rerun alone in the working tree and in an isolated detached
worktree at clean commit `3250e0cb708eef7f8cbeafb62b2fd27741e3554f`.
Both produced exactly the same pre-existing mismatch:

- analytic score `[-1.13555311, -5.31685265, -1.11112528]`;
- finite difference `[-1.13765430, -5.31931913, -1.10456913]`;
- maximum absolute difference `0.00655615`.

Because the clean baseline fails identically, this is an existing historical
SVD-UKF finite-difference defect, not a regression caused by the optional JVP
contract.  It remains a separate numerical issue.

### Static and document checks

- Python `compileall`: pass for all four changed numerical modules and the new
  test module.
- `git diff --check`: pass.
- Full monograph build:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`: pass,
  431 pages, `docs/main.pdf`.
- The new matrix-free labels and `zhao2024ttsequential` citation resolve.
  The build still reports 11 undefined citations and four duplicate labels in
  unrelated pre-existing high-dimensional chapters.

### Trusted GPU/XLA dense-versus-JVP benchmark

The reviewed Phase-6 supervisor ran 18 fresh TensorFlow processes: q in
`{5,10,20}`, dense and JVP arms, and three alternating-order repetitions.  Each
process compiled the identical four-direction principal-root UKF score, then
evaluated the same five fixed parameter points.  GPU 1 was excluded because a
foreign Python workload owned it; the run used physical GPU 0 without
interfering with the other lane.

Command:

```bash
python docs/benchmarks/benchmark_ssl_lstm_matrix_free_gpu_xla_2026_07_20.py \
  --mode supervisor --physical-gpu 0 --authorize-gpu-benchmark
```

The first non-trusted attempt stopped before any worker because `nvidia-smi`
returned exit 9.  In accordance with the GPU trust policy, the identical
command was rerun in the trusted context.  It completed in `1100.40` seconds.

| q | Median warm JVP/dense | Warm allocator JVP/dense | Host HWM JVP/dense | Timing status |
| ---: | ---: | ---: | ---: | --- |
| 5 | 1.895 | 0.859 | 0.978 | two clean pairs; JVP slower |
| 10 | 1.086 | 0.838 | 0.969 | every pair contaminated |
| 20 | 0.898 | 0.321 | 0.962 | every pair contaminated |

All 15 paired point comparisons had zero value difference and maximum score
difference `1.78e-15`, well inside the `1e-10` veto threshold.  There were no
non-finite outputs, device/XLA failures, memory-growth failures, timeouts, or
resource-cap violations.

At q=20 the post-compile, reset warm allocator peak was exactly `12.614 MiB`
in every dense process and `4.050 MiB` in every JVP process.  The cold
compile/first-call peak was identical at `363.892 MiB`, so the JVP reduces
steady derivative allocation but not the dominant cold allocation.  Process
high-water RSS was about `2.39--2.41 GiB` dense and `2.30 GiB` JVP.  These
figures do not explain or reproduce the earlier 36-GiB process report, which
therefore belongs to a different or cumulative execution context rather than
this isolated selected-score call.

The q=20 paired warm ratios were `0.898`, `0.383`, and `0.978`, but every pair
contained at least one prelaunch utilization reading above 50 percent.  The
median also missed the prospective `<=0.80` nomination threshold.  The
predeclared decision is therefore
`DO_NOT_NOMINATE_DOWNSTREAM_RERUN_FROM_CURRENT_EVIDENCE`.  This is not a
candidate-correctness rejection: the JVP is exact and reduces warm allocation.
It means the present implementation has not shown a sufficiently large,
uncontaminated end-to-end speed benefit to justify restarting the expensive
q=20 NeuTra/HMC program.

Machine-readable evidence is under
`docs/plans/artifacts/ssl-lstm-matrix-free-filter-derivatives-2026-07-20/gpu-xla-benchmark/`.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `3250e0cb708eef7f8cbeafb62b2fd27741e3554f` plus the documented dirty working-tree changes |
| Branch | `main` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-matrix-free-filter-derivatives-plan-2026-07-20.md` |
| Result | this file |
| Environment | Python 3.13.13; TensorFlow 2.20.0; TensorFlow Probability 0.25.0 |
| Device | CPU-hidden debug/reference checks plus trusted physical GPU 0 benchmark; GPU 1 left to the other workload |
| XLA | CPU XLA local parity and GPU XLA full-score benchmark |
| TF32 | Enabled by repository policy; tensors remained float64 |
| Seeds | Deterministic algebraic fixtures; fixed replay tests use recorded stateless manifest seeds, including `(20260720,11)` and `(20260720,13)` |
| Data version | Deterministic in-repository SSL-LSTM fixtures; no external data |
| Focused wall time | 4.73 seconds |
| Compatibility wall time | 178.24 seconds |
| GPU benchmark wall time | 1100.40 seconds across 18 fresh worker processes |
| Final document wall time | 2.5 seconds; initial full rebuild 7.5 seconds |
| Output paths | changed source/tests, benchmark script, Phase-6 JSON/log artifacts, this result, and `docs/main.pdf` |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Optional callbacks preserve old constructors; dense fallbacks pass; compilation, XLA smoke, q=20 shape, dense-call guards, and 18 fresh GPU/XLA workers pass. |
| Numerical validity | Same-input dense/JVP local and end-to-end values/scores agree; directional finite differences pass; GPU maximum score difference is `1.78e-15`. |
| Scientific interpretation | No change to the filter approximation, posterior target, model, data, or sampler was claimed. No scientific ranking follows. |

## Post-Run Red Team

The strongest alternative explanation is now supported: dense SSL-LSTM
Jacobians were only one component of the q=20 cost.  The JVP cuts the warm
allocator peak by about 68 percent but produces only a contaminated median
warm-time reduction of about 10 percent.  Principal-root derivatives,
covariance tangents, XLA kernel structure, and launch/synchronization overhead
remain material.

The weakest evidence is timing because GPU 0 background utilization varied
widely.  Nevertheless, the result is strong enough for the negative decision:
even the contaminated median misses the prospective 20-percent nomination
threshold, q=5 is clearly slower on two clean pairs, and cold memory is
unchanged.  A profiler-guided fused JVP or a dedicated-GPU q=20 rerun could
reopen the performance question.  Full-chart inference remains open because
forward JVP cost grows with the number of directions; a memory-style
reverse/VJP filter recurrence is still the appropriate algorithmic phase for
that different problem.
