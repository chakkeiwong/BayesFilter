# Canonical LGSSM Fused OT And Loop Performance Repair Result

Date: 2026-07-18
Campaign ID: `canonical-lgssm-fused-ot-loop-repair-20260718`
Status: `PERFORMANCE_REPAIR_PASS_TF32_MARGINAL_GATE_OPEN_T10_T50_NOT_RUN`

## Outcome

The identified implementation performance defects are repaired on the executed
LGSSM T=2 route. The production factory now uses one `tf.while_loop` horizon,
one joint primal-plus-five-tangent Sinkhorn/balance state per active time step,
one transport tile traversal per active time step, and no diagnostic Sinkhorn,
balance, or marginal tile traversal. An all-inactive prepared reset mask is
specialized at factory construction and executes zero Contract E/OT work.

The exact workload that previously took `885.02 s` for one T=2, N=1024,
16-seed Contract E node now takes `29.15 s` including graph tracing, XLA
compilation, first execution, warm replay, reporting, and Kalman comparison.
Its synchronized warm execution is `3.8895 s`. This is about `30.4x` lower
whole-node wall time than the prior Contract E artifact and `227.5x` lower when
the old whole-node wall time is compared descriptively with the new warm call;
the latter is not a like-for-like timing claim because the old harness did not
separate compilation.

The float64 T=2 node is hard-valid. A marginal-only design/audit selected
`balance_steps=2` under the new `TV_col <= 1e-4` and `E_row <= 0.01` criteria.
At N=1024 float64, maximum `TV_col=1.7853e-5` and `E_row=3.9250e-4` pass.

The production float32/TF32 T=2 node compiles and runs in `0.8921 s` warm with
the same correct work counts and `0.890 GB` peak allocator memory. It is not
scientifically valid under the frozen marginal criterion: `E_row=7.6008e-4`
passes, but `TV_col=1.7715e-4` exceeds `1e-4`. Therefore the conditional
supervisor correctly stopped before T=10 and T=50. The campaign attempt budget
is exhausted. T=10 and T=50 remain unexecuted, not failed.

## Claimed And Computed Quantities

| Item | Claimed target | Quantity computed | Verdict |
| --- | --- | --- | --- |
| finite value | same fixed Contract E--Chol finite particle scalar | fused functional-loop value compared with separated pre-repair finite program on tiny fixtures | correct within declared float64 bounds on checked fixtures |
| total score | total derivative of the same scalar in five physical coordinates | joint manual JVP propagated with shared primal state | correct within declared bounds against the separated manual JVP and forward AD fixtures |
| OT work | one shared primal/JVP solve per active time step | explicit counters in T=2 GPU artifacts | correct: two state constructions and two transport sweeps for T=2; zero diagnostic solver/sweeps |
| inactive reset | no OT work when reset is statically all inactive | factory specialization and source/graph/work tests | correct on checked route |
| horizon topology | TensorFlow functional time recursion | one `StatelessWhile`, 472--473 top-level operations | correct; no production Python horizon unroll |
| marginal validity | `TV_col <= 1e-4`, `E_row <= 0.01` per active reset | direct probability-scaled errors from the fused coupling | float64 correct at T=2; TF32 wrong relative to the declared TV gate at `balance_steps=2` |
| LGSSM oracle agreement | particle value/score compared with differentiated Kalman filtering | aggregate value and physical-score differences | descriptive at T=2; no statistical equivalence conclusion |

## Repair Details

1. `ledh_contract_e_reset_tf.py` now exposes a JVP-from-forward helper so the
   Contract E reset reuses the exact moments, Cholesky factors, injected cloud,
   and affine factorization.
2. `ledh_contract_e_streaming_tf.py` now builds a joint primal/JVP finite
   Sinkhorn and terminal-balance state. For the active exact one-tile policy
   (`K=N`), it constructs the transient coupling once and accumulates payload,
   row mass, column mass, post-quotient column mass, and payload tangents from
   that tile.
3. Direct `TV_col` and `E_row` replace roundoff-level marginal validity on the
   fused path. Historical roundoff telemetry remains reporting-only.
4. `ledh_contract_e_canonical_lgssm_tf.py` now has a fused per-step value/score
   kernel and one `tf.while_loop` horizon with TensorArray telemetry. The factory
   specializes all-inactive masks and routes current claim work through the
   functional core. The old separated primal/manual-JVP functions remain as
   reference authorities.
5. New harnesses separate trace, compile-plus-first-execution, and warm timing;
   emit work counters, graph metrics, memory, marginal errors, device/XLA/TF32
   provenance, Kalman comparisons, and structured exceptions; and never
   overwrite an artifact.
6. Preparation identity binds
   `contract_e_probability_marginals_tvcol_erow_v1`, both tolerances, terminal
   balance count, and the existing chunk-policy identity.

## Balance Selection

The selection was Kalman-blind at T=2, N=128, float64 with design seeds
`81300..81307` and untouched audit seeds `81320..81327`.

| Balance steps | Design pass | Maximum `TV_col` | Maximum `E_row` | Warm execution |
| ---: | --- | ---: | ---: | ---: |
| 0 | no | `3.6311e-4` | `2.3723e-2` | `0.1328 s` |
| 1 | no | `1.1785e-4` | `1.0048e-2` | `0.1283 s` |
| 2 | yes | `5.5156e-5` | `5.6450e-3` | `0.1295 s` |

The untouched two-step audit passes with `TV_col=3.6635e-5` and
`E_row=6.8997e-4`. Candidates `3,5,8` were not executed after the first positive
passing count, as predeclared.

The subsequent TF32 N=1024 result demonstrates that this float64-selected count
does not transfer automatically across precision. A future continuation must
perform a TF32-specific marginal-only selection/audit. It must not merely set
five because five was previously discussed.

## Performance Evidence

| Node | Status | Trace | Compile + first | Warm | Peak allocator | Graph ops |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T=2,N=128,1 seed,float64,balance 1 | expected marginal candidate failure; harness valid | `7.224 s` | `8.778 s` | `0.186 s` | `0.002 GB` | 470 |
| T=2,N=1024,16 seeds,float64,balance 2 | hard-valid | `6.808 s` | `14.269 s` | `3.890 s` | `1.647 GB` | 472 |
| T=2,N=1024,16 seeds,float32/TF32,balance 2 | performance passes; marginal veto | `6.674 s` | `15.516 s` | `0.892 s` | `0.890 GB` | 473 |

The graph operation count is essentially independent of T by construction, but
only T=2 was executed in this campaign. T=10/T=50 runtime scaling is not
measured and must not be inferred solely from the loop topology.

## T=2 Kalman Diagnostics

These are descriptive aggregate differences, not uncertainty-supported
equivalence results.

| Precision | Value difference | Physical score differences `(phi1,phi2,phi3,q,r)` |
| --- | ---: | --- |
| float64 | `0.00174757` | `(0.0127366, -0.00169792, -0.0323609, -0.0307094, -0.0185994)` |
| float32/TF32 | `0.00701972` | `(0.0159308, -0.000534444, -0.0323704, -0.0533142, -0.0289076)` |

No coordinate changes sign relative to the corresponding aggregate score, but
the TF32 marginal veto fires before a precision-promotion or longer-horizon
claim can be made.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| accept the fused performance implementation for further testing | passed finite-program parity, work accounting, functional graph, float64 T=2 validity, and GPU/XLA resource gates | no engineering or float64 T=2 veto | multi-block future N>3000 path and longer-horizon runtime untested | retain fused route and guardrails | no production scientific admission |
| reject `balance_steps=50` as required by the new float64 T=2 marginal target | two steps passed design and audit | none for float64 checked set | cross-model/cross-precision transfer | keep count identity-scoped, not universal | no convergence theorem or universal optimum |
| do not promote TF32 balance count 2 | `TV_col=1.7715e-4 > 1e-4` | marginal hard veto fired | smallest passing TF32 count unknown | new bounded TF32 marginal-only ladder, then T=2 precision replay | no claim that TF32 algorithm is generally invalid |
| do not claim T=10/T=50 results | nodes were not launched after the T=2 veto | campaign budget exhausted | longer-horizon performance and validity unknown | authorize a continuation only after TF32 balance selection | no T=10/T=50 failure claim |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Result |
| --- | --- |
| Engineering correctness | PASS for checked scope: fused primitive/reset/horizon parity; all-inactive zero OT; one active joint solve per step; no diagnostic recomputation; XLA; 8 GiB cap; chunk-policy guards; structured artifacts. |
| Numerical validity | PASS for float64 T=2. TF32 T=2 fails the declared column-TV marginal gate at the transferred balance count. |
| Scientific interpretation | T=2 Kalman comparisons are descriptive. No equivalence, HMC, nonlinear, posterior, default-readiness, or leaderboard claim is established. |

## Inference Status

| Question | Status |
| --- | --- |
| hard veto screen | float64 T=2 passes; TF32 T=2 column-TV veto fires |
| statistically supported ranking | none attempted |
| descriptive-only differences | timing, memory, Kalman value/score differences, and TF32-vs-float64 drift |
| default readiness | implementation topology remains a viable production candidate; TF32 scientific admission is not ready |
| next evidence needed | TF32-specific marginal-only count selection/audit, repeated T=2 precision gate, then conditional T=10 and T=50 nodes |

## Attempt And Failure Record

- Attempt 01: valid performance smoke; one-step balance candidate failed both
  marginal criteria as expected.
- Attempt 02: selected/audited float64 count two. One prior permission review
  timeout did not launch a process and consumed no campaign budget.
- Attempt 03: hard-valid T=2,N=1024,16-seed float64 node. Two prior permission
  review timeouts did not launch processes.
- Attempt 04: harness failed before compile because an existing float64 Tensor
  was passed to `tf.convert_to_tensor(..., float32)`; fixed with an explicit
  boundary decision.
- Attempt 04b: preparation rejected an already-float32 observation Tensor
  because preparation owns the raw-float64 conversion; harness repaired to pass
  the raw dataset Tensor.
- Attempt 05: TF32 XLA node compiled and ran, then stopped on the declared T=2
  column-TV veto. No T=10/T=50 node was launched.

## Checks

- focused primitive/canonical/preparation/chunk-policy suite: `51 passed`;
- new fused primitive parity/work test: passed;
- fused loop versus separated finite-program value/score test: passed;
- all-inactive zero-OT factory test: passed;
- Python compilation for all touched implementation and harness modules: passed;
- `git diff --check`: passed;
- trusted `nvidia-smi`: RTX 4080 SUPER visible;
- trusted TensorFlow probe: TensorFlow 2.19.1 sees GPU 0, CUDA build 12.4,
  capability `sm_89`;
- trusted float64 and TF32 XLA compilation: passed.

The two TensorFlow Probability `distutils.version` deprecation warnings and
duplicate CUDA factory-registration startup warnings are unrelated to the
result.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit at execution | `fbc3b6e9aaf882b8275bfb94aaa2ff43cc4c5a98` |
| Worktree | dirty shared research worktree; unrelated edits preserved |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu` |
| TensorFlow | 2.19.1 |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 8192 MiB TensorFlow logical limit, no memory growth |
| XLA | `jit_compile=True`; trusted GPU compiled cluster observed |
| Chunk policy | `dpf_transport_exact_divisor_cap3000_v1`; N=1024,K=1024, one block |
| Float64 T=2 seeds | `81500..81515` |
| TF32 T=2 seeds | `81500..81515` |
| Plan | `docs/plans/bayesfilter-canonical-lgssm-fused-ot-loop-performance-repair-plan-2026-07-18.md` |
| Artifact root | `docs/benchmarks/artifacts/canonical_lgssm_fused_ot_loop_repair_20260718/` |
| Attempt ledger | `docs/benchmarks/artifacts/canonical_lgssm_fused_ot_loop_repair_20260718/attempt-ledger.md` |

## Post-Run Red Team

The strongest alternative explanation for the speedup is not algorithmic
fusion alone: reducing terminal balance from 50 to 2 also removes substantial
work. The explicit counters isolate the topology result (one solve versus
repeated solves), while the float64 timing combines both legitimate repairs.
A like-for-like 50-step fused timing was not run and is unnecessary for the
new target, but it would be required to apportion the speedup exactly.

The conclusion would be overturned by a same-prepared-input fixture where the
fused value/score differs materially from the separated finite program, by a
reachable graph showing hidden solver reconstruction, or by counters that do
not match a profiler/HLO audit. Current focused parity, graph, and work evidence
does not show such a defect.

The weakest scientific evidence is the absence of T=10/T=50 and the one-batch
T=2 Kalman comparison. Those omissions are explicit. The correct next step is
not to relax `TV_col`; it is to select a TF32 balance count under the same
marginal-only design/audit procedure, then repeat the conditional ladder.
