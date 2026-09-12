# SSL-LSTM q=20 GPU replay, batching, and strict-eigensystem reuse plan

Date: 2026-09-04  
Status: `P3_FACTOR_CACHED_PARITY_PASS_RAW_CACHED_VETO_P1_CANARY_PENDING`
Governing master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Runner: `docs/benchmarks/run_ssl_lstm_q20_phase9a_fresh_tuning_preflight_2026_08_31.py`  
Launcher: `scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu10000.sh`

## Purpose

The previous Phase 9A replay was stopped by a valid resource veto under a
7,800-second profile.  This continuation keeps the target, bridge, chart
protocol, candidate grid, and HMC evidence contract fixed while testing two
performance hypotheses:

1. the target kernel already batches the four chains, but candidate pairs may
   be grouped only after a semantically equivalent transition has been built;
2. the batched strict principal-square-root route may reuse one strict-factor
   eigendecomposition for its Sylvester derivative.

The continuation introduces a fresh source-owned profile with a 10,000-second
full-replay material cap.  The old 1,800/7,800-second profiles and their partial
calls remain historical and are never resumed.

## Research intent ledger

| Item | Binding definition |
|---|---|
| Question | Can the frozen six-scope q=20 mechanics schedule complete on GPU0 within 10,000 seconds, and can either batching or strict eigensystem reuse reduce cost without changing the computed target or score? |
| Baseline | Existing batch-native TensorFlow/TFP target, `tensorflow_eigh_strict`, four-chain batched HMC call, serial candidate-pair orchestration, and the frozen C5 chart protocol. |
| Candidate mechanisms | (A) target/candidate batching grouped by a fixed leapfrog count; (B) opt-in raw-covariance reuse `tensorflow_eigh_strict_cached`; (C) opt-in strict-factor reuse `tensorflow_eigh_strict_factor_cached`; (D) the unchanged baseline. |
| Primary performance criterion | A candidate is eligible for timing only after exact value/score/diagnostic parity with the baseline on valid, repaired, and invalid fixtures. Completion is measured against the 10,000-second cap. |
| Promotion criterion | Every required parity, target-status, memory-growth, XLA, provenance, and durable-artifact gate passes; then the six-scope replay completes with all declared calls and held-out checks. |
| Promotion veto | Any value/score/state mismatch beyond the declared tolerance; changed invalid-row classification; target or seed identity mismatch; nonfinite output; missing call; retracing or memory-policy violation; output collision; or claim beyond the computed quantity. |
| Continuation veto | The 10,000-second campaign cap is exhausted, required artifacts cannot be made durable, a semantic equivalence repair fails three scoped times, or continuation would change target, data, hardware class, privacy boundary, or the frozen scientific contract. |
| Explanatory diagnostics | Eigensolver-op counts, compile versus steady-state time, allocator readings, acceptance, movement, ESS, R-hat, and per-call timing. These do not establish convergence or sampler quality in this plan. |
| Must not conclude | No IID-Gaussian whitening, mode discovery, posterior correctness, convergence, sampler superiority, high-dimensional scaling, production readiness, or default promotion. |

## Fixed identities

- Target signature: `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
- Parameter measure: `theta in R^4`; the internal q=20 filter state is not the
  HMC sampling measure.
- Bridge: Gaussian prior plus likelihood temperature bridge, with beta ladder
  `(0, 0.5, 1)` and the repository properness receipt.
- Transport: C5 `phase8-k2-compact-high-l3-pure`, two `(16,16)` tanh charts,
  pure continuation, fixed state-independent chart weights `(0.5, 0.5)`.
- HMC tuning: `measured_joint_grid_v1`, epsilon
  `(0.25, 0.55, 0.85, 1.20)` crossed with leapfrog lengths `(3, 8)`;
  every declared pair is measured before selection.
- Numerical route: TensorFlow/TFP, float64, `tensorflow_eigh_strict`, XLA on,
  TF32 recorded, GPU0, memory growth configured before TensorFlow import, no
  pfor or row-mapped scalar target.
- Fresh profile IDs:
  `phase9a_full_replay_canary_gpu10000_v1` (scope 3/1, 1,800-second canary)
  and `phase9a_full_replay_gpu10000_v1` (scope 0/6, 10,000-second replay).

## Budget and artifact boundary

The 10,000-second number is the material wall cap for the six-scope full replay,
not a promise that every possible candidate pattern fits.  The canary has its
own 1,800-second bound, so the maximum campaign envelope is 11,800 seconds if
the canary and full replay are both launched.  This interpretation is recorded
explicitly; it does not silently convert the historical cap.

All attempts write below
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-04/phase9a-full-replay-gpu10000/<attempt>/`.
An existing attempt directory is a hard error.  Each attempt must retain its
start receipt, per-call start/complete/failure records, chart checkpoints,
scope records, run manifest, and a result/closeout note.

The old timing gives a descriptive forecast, not a guarantee.  With the
observed two surviving beta-zero scopes, one GPU is expected to take roughly
5,500--5,800 seconds.  If all 48 screen candidates survive, selection can
push the schedule to roughly 12,500--13,200 seconds and the 10,000-second cap
will correctly stop it.  The replay must therefore report completion coverage,
not treat a cap stop as a scientific failure.

## Default and assumption audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|
| GPU0, XLA, TF32, memory growth | Repository policy and prior q=20 GPU manifests | Device contention, late allocator setup, or unsupported op | Trusted device/memory receipt before target import | Reviewed execution policy |
| Four chains in one target batch | Existing `GaussianLikelihoodBridge` and fixed-transport runner | Shape specialization or a hidden row loop changes graph semantics | Static signature and trace-count receipt | Existing baseline |
| Candidate grouping by fixed `L` | HMC trajectories with different `L` cannot share one scalar loop budget | Per-row seeds, step sizes, or accept/reject state become coupled | Exact grouped/scalar/row-loop fixture | Hypothesis only |
| Raw-covariance eigensystem reuse | Algebraic identity for a scalar identity shift of an SPD covariance | Near-degenerate eigenvectors change the finite-precision score | Value/score/diagnostic parity on valid, repaired, and invalid rows | Opt-in diagnostic; GPU score vetoed |
| Strict-factor eigensystem reuse | The safe-factor eigensystem also diagonalizes its SPD factor in exact arithmetic | Finite-precision basis sensitivity or downstream trajectory drift | Value/score/diagnostic parity plus one-step/full-chain sensitivity | Opt-in candidate; local GPU parity passes |
| 10,000-second replay cap | User instruction; larger than the historical 7,800-second cap | Still insufficient when all candidates pass screens | Canary projection and durable cap receipt | New reviewed campaign cap |
| Eight-pair grid and two selection replications | Frozen C5/Phase 9A evidence contract | Long candidates can exhaust the cap before held-out checks | Per-call coverage table | Baseline, not a universal tuning default |

## What is already batched

The production q=20 bridge calls
`batch_prior_likelihood_value_score_status(theta)` with a static `[B,4]`
tensor.  The target's filter recursion uses `tf.while_loop` over time and
TensorFlow linear algebra over the leading batch axis.  The fixed-transport
runner passes a four-chain state bank, so four chains are already evaluated in
one compiled target call.

The prior target diagnostic measured, for equal total rows, a descriptive
2.68x total-time reduction for batch 16 and a 3.97x per-row reduction for batch
32 relative to serial batch-4 calls.  The grouped HMC prototype was finite but
not semantically equivalent: state, target, and log-acceptance differed because
TFP stateless-seed consumption and candidate partitioning were different.  It
remains unadmitted.  Candidate-level batching is therefore a new kernel design,
not a switch on the existing runner.

## Strict eigensystem reuse: mathematical condition

Let a valid symmetric positive-definite covariance be

\[
C = V\,\operatorname{diag}(\lambda_1,\ldots,\lambda_d)V^T,
\qquad \lambda_i>0.
\]

The strict implementation sends
\(C_s=C+\rho I\) to the root operation, where \(\rho\) is the fixed guard
margin.  On a roundoff-repaired row it sends \(C_s=C+2\rho I\); on a
classified-invalid row it sends a replacement scalar matrix.  For the valid
and repaired cases,

\[
C_s=V\,\operatorname{diag}(\lambda_i+\delta)V^T,
\quad
F=C_s^{1/2}=V\,\operatorname{diag}(s_i)V^T,
\quad s_i=\sqrt{\lambda_i+\delta},
\]

with \(\delta=\rho\) or \(2\rho\).  Thus the covariance eigensystem is
mathematically sufficient for both the factor and the Sylvester equation

\[
F X + X F = R.
\]

Writing \(\widehat R=V^T R V\), the unique SPD solution is

\[
X=V\left[\frac{\widehat R_{ij}}{s_i+s_j}\right]_{ij}V^T.
\]

The candidate backend `tensorflow_eigh_strict_cached` may use this identity only after the exact safe matrix
and row classification are known.  It must retain the existing replacement and
invalid-row behavior.  It may not cache an eigensystem across time steps or
HMC calls, because the covariance changes with the state and parameters.

The current batched strict score graph has one covariance `SelfAdjointEigV2`,
one strict-root `SelfAdjointEigV2`, and one factor-Sylvester `SelfAdjointEigV2`
in each placement step (and the corresponding three in the innovation step).
For q=20 the placement matrix is 80x80 and the innovation matrix is 1x1.
The raw-covariance candidate can reduce the large placement eigensolver count
from three to one per time-step, while the safer strict-factor candidate
reduces it from three to two by reusing the eigensystem of the exact safe
matrix used to construct the factor.  These are performance hypotheses, not
correctness results; the strict baseline remains the authority until parity is
shown.

The first GPU timing run shows why the parity clause is binding: q=20 placement
spectra can have eigenvalues near `1e-12` and adjacent gaps near `1e-15`.
Reusing the raw covariance eigenbasis is therefore an ill-conditioned
finite-precision transformation even though the displayed identity holds in
exact arithmetic.  The strict-factor candidate instead reuses the basis from
the same safe matrix used by the strict root.  It passed the current center and
varied-row score fixture; this is still local parity evidence, not downstream
trajectory equivalence.

For the strict-factor candidate, let

\[
C_s=V\operatorname{diag}(\lambda_i)V^T,\qquad
F=C_s^{1/2}=V\operatorname{diag}(\sqrt{\lambda_i})V^T.
\]

For a symmetric right-hand side `R`, the Sylvester solution is

\[
X=V\left[\frac{(V^T R V)_{ij}}
{\sqrt{\lambda_i}+\sqrt{\lambda_j}}\right]_{ij}V^T.
\]

The implementation computes this eigendecomposition once per covariance and
uses it for both `F` and `X`; it does not cache across time steps or HMC calls.

## Phases and mandatory repair/refresh

Every phase has a closeout.  A passing phase still writes a receipt, reruns the
smallest exact regression, and refreshes the next phase's assumptions and
command.

| Phase | Work | Closeout and next entry | Real stop condition |
|---|---|---|---|
| P0 | Review source call graph, profiles, graph-op counts, and this evidence contract | Record the skeptical audit and profile hashes; refresh P1 | Target/profile identity or artifact root cannot be established |
| P1 | Run the fresh GPU0 canary with the unchanged strict baseline | Record call coverage, compile/steady timing, memory, and cap projection; refresh P2 | Identity, memory-growth, numerical, or second bounded resource failure under the unchanged canary |
| P2 | Implement and test raw-covariance and strict-factor eigensystem reuse; compare both with the strict baseline on valid, repaired, invalid, and q=20 batch fixtures | Preserve failed fixtures, retain the raw candidate as vetoed, and refresh P3 with the strict-factor candidate only after its declared value/score tolerance passes | Any unexplained value/score/diagnostic mismatch after three scoped repairs |
| P3 | Run a paired GPU timing diagnostic for baseline and both reuse candidates with identical inputs and static signatures | Record eig-op counts, compile/steady time, allocator, and parity; keep each candidate diagnostic-only until its own downstream sensitivity gate passes | Candidate is not parity-clean or violates XLA/memory policy |
| P4 | Design/measure candidate batching grouped by fixed `L`; first use deterministic per-row transition mechanics | Record exact grouped/scalar/row-loop receipts; do not integrate on mismatch | Seed/state semantics cannot be made equivalent |
| P5 | Launch the six-scope replay under the new 10,000-second profile only with a passing P1/P2/P3 closeout | Write terminal decision and inference-status tables; refresh or keep Phase 9B blocked | Cap exhaustion, missing scope, or any promotion veto |

Candidate batching and eigensystem reuse are independent.  A failed batching
candidate does not invalidate the cached-eigensystem candidate, and a timing
improvement never waives parity or the replay's scientific gates.

## Execution ledger

The bounded focused checks and GPU timing diagnostics were executed before the
strict-baseline canary.  This out-of-order diagnostic work does not authorize a
replay and is recorded so the canary entry remains explicit.

| Date/phase | Evidence | Decision |
|---|---|---|
| 2026-09-04 P0 | Python compilation, shell syntax, profile identity checks, and 61 focused tests pass | P0 source and harness checks pass |
| 2026-09-04 P2 | Strict-factor value/score parity passes CPU and compiled valid/repaired/invalid fixtures; raw-basis candidate is retained only as a diagnostic | Strict-factor candidate remains opt-in pending q=20 GPU evidence |
| 2026-09-04 P3 | GPU0 XLA batch-size diagnostics; six versus two versus four eigensolver nodes; batch-4 raw and strict-factor steady means about 0.30 s and 0.57 s versus strict 0.89 s | Both performance hypotheses supported descriptively; raw parity still vetoed |
| 2026-09-04 P3 repair | Raw reuse fails q=20 GPU score parity on center and varied rows; strict-factor reuse passes both, with minimum placement gaps near `1e-15` | Raw candidate vetoed; strict-factor candidate remains opt-in pending downstream sensitivity; retain strict replay baseline |

The authoritative closeout is
`docs/plans/bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-result-2026-09-04.md`.
The previous status-only benchmark artifacts are diagnostic timing records and
must not be read as parity approval.

## Exact commands

Static/profile checks:

```text
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_ssl_lstm_q20_phase9a_repair_runner.py \
  tests/test_experimental_batched_svd_sigma_point_tf.py
bash -n scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu10000.sh
```

Canary (GPU0, fresh attempt ID):

```text
BAYESFILTER_PHASE9A_ATTEMPT_ID=canary-20260904-01 \
bash scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu10000.sh \
  --profile phase9a_full_replay_canary_gpu10000_v1 \
  --scope-start 3 --scope-limit 1
```

Full replay, only after the canary and parity closeouts:

```text
BAYESFILTER_PHASE9A_ATTEMPT_ID=full-20260904-01 \
bash scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu10000.sh \
  --profile phase9a_full_replay_gpu10000_v1 \
  --scope-start 0 --scope-limit 6
```

The launcher sets `CUDA_VISIBLE_DEVICES=0` and
`TF_FORCE_GPU_ALLOW_GROWTH=true` before importing TensorFlow.  A trusted GPU
permission is required at launch; a nontrusted device failure is not evidence
that the machine or environment is broken.

## Skeptical audit before execution

The plan passed the required pre-execution audit with these repairs:

- It does not reuse the terminal 7,800-second plan or partial calls; the new
  profile, plan hash, seed namespace, and artifact root are distinct.
- Completion and timing are not treated as evidence of posterior quality.
  Acceptance, R-hat, ESS, and movement remain explanatory or veto diagnostics
  under the declared roles.
- The cap is explicit about canary versus full-replay accounting and includes a
  worst-case projection that can still exceed 10,000 seconds.
- Candidate batching is blocked until exact state, target, gradient,
  log-acceptance, seed, and call-count equivalence is demonstrated.  No pfor,
  `tf.map_fn`, or row-mapped scalar target is introduced.
- Eigensystem reuse is restricted to mathematically valid per-call identities,
  with explicit repaired and invalid-row tests.  The raw-basis candidate is
  vetoed by GPU score parity; strict-factor reuse has only local parity so far.
  No cross-call cache or hidden arithmetic substitution is permitted.
- GPU memory growth, XLA, static signatures, fresh outputs, manifests, and
  stop conditions are bound before any long run.

Audit verdict: `PASS_NEW_GPU10000_PERFORMANCE_DESIGN_READY_FOR_FOCUSED_TESTS`.
This verdict authorizes the bounded profile and parity work, not a whitening,
posterior, HMC, or production claim.

## Closeout requirements

The result note must state the actual command, Git commit/status hash,
environment, GPU and allocator receipt, seeds, wall time, per-call coverage,
parity results, decision table, inference-status table, strongest alternative
explanation, and next action.  A cap stop is reported as a resource result and
does not promote partial selections.  Phase 9B remains blocked unless a later
plan independently satisfies its sequential warmup, retained-sample, travel,
replica, and posterior gates.
