# SSL-LSTM q=20 GPU replay, batching, and strict-eigensystem reuse result

Date: 2026-09-04  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-plan-2026-09-04.md`  
Status: `P3_FACTOR_CACHED_PARITY_PASS_RAW_CACHED_VETO_P1_CANARY_PENDING`  
Git base: `2dae412e450a5b44f46e375b810a7ad81aa78aeb`  

## Decision summary

The new full-replay profile is configured with a 10,000-second material wall
cap and a fresh seed/output namespace.  The historical 7,800-second profile is
unchanged and was not resumed.  The q=20 target and the four HMC chains are
already batch-native on the GPU.  A GPU batch-size diagnostic shows a large
throughput benefit when independent rows are submitted together, but the outer
candidate-pair loop remains serial because different leapfrog counts and
stateless random streams have not been made exactly equivalent in one grouped
transition.

Reusing an eigensystem is computationally promising, but the reuse location
matters.  The raw-covariance candidate reduces the q=20 graph from six to two
`SelfAdjointEigV2` nodes and is descriptively about 2.9 times faster at batch 4,
but its score fails the declared parity tolerance.  The narrower
strict-factor candidate reuses the eigensystem of the exact safe matrix used to
construct the factor.  It reduces the graph from six to four eigensolver nodes,
is about 1.5 times faster at batch 4, and passes the current center and varied
q=20 score fixtures.  It remains opt-in until a near-degenerate and downstream
transition check pass.

## Evidence contract and roles

The measured quantity is the batch-native q=20 value/analytic score/status
program, not a posterior sample or whitening diagnostic.  The strict
`tensorflow_eigh_strict` implementation is the authority.  Eigensolver counts
and timings are explanatory performance diagnostics.  Value/score differences
and status mismatches are hard candidate vetoes.  No result in this note is
evidence for posterior correctness, convergence, IID-Gaussian whitening, mode
discovery, sampler superiority, or high-dimensional scaling.

## Commands and environment

The focused regression was run with GPUs hidden:

```text
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_ssl_lstm_q20_phase9a_repair_runner.py \
  tests/test_experimental_batched_svd_sigma_point_tf.py
```

This focused run completed with `61 passed`; the warnings were dependency
deprecations only.

The final diagnostic runs used GPU 0, TensorFlow 2.20.0, Python
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, XLA enabled, `TF32=1`, and
`TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import.  The artifact records
one RTX 4080 SUPER, verified memory growth, and one trace for each compiled
backend.  The benchmark command was:

```text
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 TF32=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/benchmark_ssl_lstm_q20_eigh_reuse_2026_09_04.py \
  --output <fresh-json-path> --batch-size 4 --repeats 2
```

The benchmark enforces the focused-test tolerances
`value rtol=1e-10`, `score rtol=1e-9`, and `atol=1e-10`.  These are diagnostic
comparison tolerances copied from the new factor/score regression; they do not
assert that two finite-precision HMC programs have identical trajectories.

## Harness repair record

Four early attempts were infrastructure failures and are preserved by their
shell history and timestamps rather than treated as numerical evidence:

| Attempt | Failure | Repair |
|---|---|---|
| 01 | Executing the benchmark by path did not put the repository root on `sys.path`. | Insert the resolved repository root before package imports. |
| 02 | The target module created TensorFlow constants before the memory-growth helper. | Configure memory growth before importing the target module. |
| 03 | The component API returns likelihood, likelihood score, prior, prior score, and status. | Unpack the five-value contract and form the posterior value/score. |
| 04 | The stored compiled object is a `tf.function`, not a concrete graph. | Call `get_concrete_function()` before graph inspection. |

The earlier artifacts `gpu0-attempt-05.json`, `gpu0-attempt-06.json`,
`gpu0-batch1.json`, `gpu0-batch8.json`, and `gpu0-batch16.json` predate the
explicit numerical tolerance pass condition and report status-only parity.
They remain diagnostic timing evidence, but their `PASS_DIAGNOSTIC_PARITY`
labels must not be interpreted as score parity.  The final artifacts below use
the corrected pass condition.

The factor-basis follow-up artifacts are
`gpu0-factor-cached-center-valid.json` and
`gpu0-factor-cached-varied.json`.  They include both reuse candidates so their
candidate-specific parity decisions remain visible; the top-level status is
`FAIL_DIAGNOSTIC_PARITY` because the raw-basis candidate is intentionally
vetoed even though the factor-basis candidate passes.

## Batching evidence

The target calls `batch_prior_likelihood_value_score_status` with a static
`[B,4]` tensor.  Time recursion is a TensorFlow `while_loop`, and the linear
algebra is over the leading batch axis.  The fixed-transport runner passes a
`[4,4]` state bank, so the four HMC chains are evaluated in one compiled target
call.  This is genuine GPU batching.

The following steady-state numbers are descriptive single-process measurements
on identical center-row inputs.  `per row` is the mean divided by the batch
size; it is not an uncertainty interval.

| Batch | Strict mean (s) | Strict per row (s) | Raw-reuse mean (s) | Raw-reuse per row (s) | Strict/raw-reuse |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8258 | 0.8258 | 0.2745 | 0.2745 | 3.01x |
| 4 | 0.8816 | 0.2204 | 0.3025 | 0.0756 | 2.91x |
| 8 | 1.0479 | 0.1310 | 0.3711 | 0.0464 | 2.82x |
| 16 | 1.2620 | 0.0789 | 0.4729 | 0.0296 | 2.67x |

Thus batching helps materially.  The raw-reuse columns are timing diagnostics
only; that candidate is rejected by the score-parity gate below.  The
measurements do not authorize batching different HMC candidate pairs: the
existing runner deliberately keeps the
candidate loop serial, while `step_size`, leapfrog count, and random seed are
dynamic inputs to one reusable graph.  A grouped-candidate implementation must
first reproduce state, target, gradient, log-acceptance, random-stream, and
call-count semantics on a deterministic fixture.  The earlier grouped-HMC
prototype failed that test and remains unadmitted.

These timings apply to the target call, not to the complete replay.  For
example, the strict-factor target ratio is approximately
`0.89 / 0.57 = 1.56`.  If the target consumes a fraction `p` of total wall time,
the corresponding Amdahl bound is

\[
 S_{\rm total}(p)=\frac{1}{(1-p)+p/1.56}.
\]

That is about `1.22x` at `p=0.5` and `1.38x` at `p=0.8`; it approaches `1.56x`
only when the target is essentially all of the runtime.  Increasing the batch
size beyond the existing four-chain bank could improve throughput, but it
would require more independent chains or a new candidate-grouped transition
and would increase activation/eigensolver memory.  No memory-capacity or
semantic-equivalence evidence for that change exists yet.

## Eigensystem-reuse evidence

Two candidates were measured.  `tensorflow_eigh_strict_cached` uses the raw
covariance eigensystem for the strict scalar-identity guard shift and the
Sylvester solve.  `tensorflow_eigh_strict_factor_cached` instead computes the
eigensystem of the exact safe matrix passed to the strict root and reuses that
basis for the Sylvester solve.  For a real symmetric safe covariance
`C_s=V\operatorname{diag}(\lambda)V^T`,

\[
 C_s^{1/2}=V\operatorname{diag}(\sqrt{\lambda_i})V^T,
 \qquad
 FX+XF=R \Longrightarrow
 X=V\left[\frac{(V^TRV)_{ij}}
 {\sqrt{\lambda_i}+\sqrt{\lambda_j}}\right]V^T .
\]

The identity is valid in exact arithmetic only after row classification and the
same guard shift are fixed.  No eigensystem is cached across time steps or HMC
calls.  The factor-basis candidate keeps the strict factor arithmetic and only
removes the separate eigensolve of that factor; the raw-basis candidate also
changes the finite-precision basis used by the derivative.

On the repeated-center q=20 batch-4 fixture, the strict backend had a steady
mean of about 0.89 s, the raw candidate about 0.30 s, and the factor-basis
candidate about 0.57 s.  The graph library contained six, two, and four
`SelfAdjointEigV2` nodes, respectively.  All outputs were finite, status codes
and row classes matched, and each backend traced once.

The corrected numerical comparison gives separate results for the two
candidates:

| Fixture / candidate | Max value error | Max score error | Score tolerance status | Status/class equality |
|---|---:|---:|---|---|
| Center, raw covariance reuse | `4.73e-11` | `2.15e-9` | **FAIL** (largest ratio 1.42) | Pass |
| Center, strict-factor reuse | `0` | `9.91e-12` | **PASS** (largest ratio 0.0114) | Pass |
| Varied, raw covariance reuse | `1.37e-9` | `5.19e-9` | **FAIL** (largest ratio 48.5) | Pass |
| Varied, strict-factor reuse | `0` | `1.16e-10` | **PASS** (largest ratio 0.0610) | Pass |

The varied-row artifact is
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-04/eigh-reuse/gpu0-varied-batch4.json`.
Its placement eigenvalues are around `1e-12`, with minimum adjacent gaps as
small as `4.8e-15`.  This near-degenerate, poorly conditioned spectrum explains
why mathematically equivalent eigenbases can produce materially different
finite-precision derivatives.  The raw candidate's failure does not prove that
its algebraic formula is wrong; it proves that the current implementation is not
numerically interchangeable with the strict score path under the declared
contract.  The factor-basis candidate passes these local fixtures, but that is
not yet a trajectory or posterior result.

## Decision tables

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action |
|---|---|---|---|---|
| Keep target batching | Batch-native static signature and throughput | Hidden row loop or trace growth | One GPU and descriptive timings | Retain existing four-chain batching; measure grouped candidates only with an exact fixture |
| Promote raw covariance eigensystem reuse | Fastest graph with mathematical identity | Score parity failed on center and varied q=20 rows | Whether a guarded/block spectral treatment can preserve derivatives | Reject for replay; retain as historical diagnostic |
| Promote strict-factor eigensystem reuse | Removes one factor eigensolve while retaining the strict safe matrix | Local center/varied fixtures pass; downstream trajectory sensitivity is untested | Near-degenerate derivative behavior over the full recursion | Keep opt-in candidate; run near-degenerate and one-step/full-chain checks |
| Launch 10,000-second full replay | Fresh profile/cap is configured | P1 canary has not run; cached candidate is not admitted | All-screen survivor cost may exceed 10,000 s | Run the strict-baseline canary only after its entry closeout; do not use cached output |

### Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Raw candidate vetoed by score parity; factor candidate passes current local parity; no GPU memory-growth or finite-value veto in the diagnostic |
| Statistically supported ranking | None; one process and a few repeats provide no ranking evidence |
| Descriptive-only differences | Batching throughput, eigensolver counts, compile/steady timings |
| Default-readiness | Not ready; strict baseline remains the only replay route |
| Next evidence needed | More near-degenerate parity, one-step/full-chain sensitivity for factor reuse, then a fresh strict-baseline canary |

## Campaign state and next step

The fresh profiles are:

- `phase9a_full_replay_canary_gpu10000_v1`: scope `3/1`, cap 1,800 s;
- `phase9a_full_replay_gpu10000_v1`: scopes `0/6`, cap 10,000 s.

No canary or six-scope replay was launched in this diagnostic phase, so the
10,000-second cap is configured but not shown sufficient.  The active master
remains Phase 9B-blocked.  The next smallest discriminating work is to test the
strict-factor candidate on explicit near-degenerate rows and compare one-step
and full-chain HMC sensitivities.  Until those checks pass, replay uses
`tensorflow_eigh_strict`; the raw candidate is rejected and the factor-basis
candidate remains opt-in.  A canary and any later replay must use a fresh
attempt root and the closeout requirements in the active plan.

## Post-run red-team

The strongest alternative explanation for the timing result is that the
benchmark measures a controlled target call rather than the complete HMC
campaign; Python orchestration, TFP transition work, chart construction, and
serialization may dominate the replay.  The strongest alternative explanation
for the raw-candidate parity failure is finite-precision sensitivity from the
nearly degenerate placement spectrum rather than an algebraic error.  A
downstream one-step/full-chain comparison on the actual HMC state bank would
overturn the claim that the factor-candidate discrepancy is operationally
negligible.  Until that evidence exists, the factor route remains a performance
hypothesis only.
