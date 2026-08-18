# Austria GenUT NeuTra Root-Cause Execution Result

Date: 2026-08-18

Plan:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-handoff-2026-08-17.md`

Review:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-second-review-reply-2026-08-17.md`

## Status

`CPU_AUTHORITY_PASS_GPU_EAGER_PASS_GPU_GRAPH_WITHIN_MODE_IDENTITY_FAIL`

Superseded former status (preserved for history):
`PHASES_0_TO_6_CPU_AUTHORITY_PASS_GPU_CONFIRMATION_BLOCKED`. The trusted GPU
launch blocker was cleared on 2026-08-18 by a Claude Code session using its own
trusted execution boundary; see "Current-Source GPU Endpoint Confirmation"
below.

Austria remains blocked from NeuTra and HMC. This execution repaired and
validated the batch diagonal candidate's value/score identity in the CPU
diagnostic lane, but the required RTX 5080 endpoint confirmation could not be
started because the trusted-command approval service returned HTTP 502 before
process creation.

The tested callable is still `batch_diagonal_candidate`, not the promoted
dual-cap route. No dual-cap, default-promotion, tuning, posterior-correctness,
or HMC claim follows from this note.

## Research Intent And Evidence Contract

Question: can the public score return the derivative of the same finite
`batch_finite_value` scalar while remaining graph/XLA compatible?

Baseline: frozen Austria SIR target, `T=20`, `N=1008`, state dimension `18`,
observation dimension `9`, target signature
`4845e7322685e19650024e5886e47d89c8b9c4b70c5d36a639c9b1218d39b5c3`.

Primary criterion: public value equals the tangent-free value and public score
equals independent TensorFlow forward autodiff of that same value program at
`T=1,2,20`.

Hard vetoes: changed frozen inputs, nonfinite output, unequal same-program
value, score/value route mismatch, finite escape after tangent invalidity, or
failed graph/XLA compilation.

Explanatory only: central-FD regression quality, ULP differences between
independently executed diagnostic paths, conditioning, runtime, and tangent
norm growth.

Nonclaims: exact nonlinear Austria likelihood, dual-cap correctness, posterior
correctness, NeuTra readiness, HMC readiness, tuning validity, or statistical
superiority.

## Implementation Changes

In `bayesfilter/highdim/cubature_genut_batch_tf.py`:

1. The public score remains a fixed-direction TensorFlow
   `ForwardAccumulator` derivative of the complete `batch_finite_value`
   program. Reverse-mode differentiation was tested but rejected because it
   differs materially from forward differentiation on the ill-conditioned
   long Austria recursion.
2. The value and JVP higher-moment routes share one primal correction core;
   previous independent manual primal routes remain diagnostic-only.
3. For statically known shapes, Sinkhorn iterations and horizon recurrences
   are Python-unrolled before TensorFlow tracing. Unknown-shape callers retain
   bounded `tf.while_loop` fallbacks. This avoids TensorList shape failures
   when forward autodiff traces nested loops.
4. The Austria transition-first policy uses a Python static branch, avoiding an
   unnecessary `tf.cond` in the graph. The dynamic transition policy retains
   `tf.cond`.
5. Redundant `tf.ensure_shape` calls inside recursive bodies were removed;
   shape invariants are established at the function boundary and by the loop
   state.
6. Diagnostics from the shared value/JVP program preserve the existing
   fail-closed validity mask and add the higher-moment feasibility/conditioning
   fields already exposed by the repaired route.

## Validation

Focused test:

```text
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/highdim/test_genut_batch_primal_parity.py
5 passed
```

Existing batch regression:

```text
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/highdim/test_cubature_genut_batch.py
4 passed
```

Derivative authority artifact:

`docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/attempt04/derivative_cpu.json`

Results from the terminal artifact:

| Horizon | Value identity | Maximum public-score error vs independent forward AD | Relative error | ULP error |
|---:|---|---:|---:|---:|
| 1 | exact | 0 | 0 | 0 |
| 2 | exact | 0 | 0 | 0 |
| 20 | exact | 0 | 0 | 0 |

The central finite-difference `h^2` regression remains poor. That is an
explanatory diagnostic in the reviewed plan, not a correctness veto, because
the FP32 perturbation ladder is numerically unstable for this route.

The preserved manual-JVP diagnostic was also compared against independent
forward AD by horizon. Its discrepancy begins around `T=3`, grows to order
`5` by `T=10`, and is order `10^2` by `T=20`. This confirms that the previous
time-recursive hand score was wrong relative to the complete finite scalar;
the new public route does not use it.

## GPU Gate

Trusted `nvidia-smi` succeeded:

```text
GPU 0: NVIDIA GeForce RTX 5080, 16303 MiB total, 2831 MiB used
GPU 1: NVIDIA GeForce RTX 4080 SUPER, 16376 MiB total, 11 MiB used
```

The planned RTX 5080 launch with `tftwogpu`, TF32, deterministic operations,
and memory growth was requested, but the approval service returned `502 Bad
Gateway` before process creation. This is an infrastructure/permission
blocker, not a failed scientific attempt. No GPU artifact should be interpreted
as a failed endpoint.

## Current-Source GPU Endpoint Confirmation (2026-08-18, Claude Code)

Executed under the restart memo
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-reset-memo-2026-08-18.md`
resume procedure, with trusted GPU access from the Claude Code execution
boundary. Pre-launch checks all passed: git commit
`dae37183bf4421682b2ad991e2dc0d0f3c53f260`, all six relevant source SHA-256
values matched the handoff table (current
`cubature_genut_batch_tf.py` = `ae8cbfb...a976e`), trusted occupancy probe saw
GPU 0 (RTX 5080, 5170 MiB used, 4%) and GPU 1, and
`repair_validation_attempt07` was confirmed absent before launch.

### Eager (attempt07) — PASS

Artifact:
`docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/repair_validation_attempt07/endpoint_gpu0_eager.json`

- `status=COMPLETE`, frozen identity guard `PASS`, source hashes match the
  current source, TensorFlow `2.20.0-dev0+selfbuilt`, `/GPU:0` (RTX 5080),
  memory-growth mode configured before logical initialization and verified on
  all physical GPUs, TF32 enabled, deterministic-ops env `1`, targets
  constructed on `/CPU:0`. Wall time `90.76` s.
- `T=20`, four corrections: value-only and score-carried value both
  `-683.0018921`, `exact_equal=true`, zero absolute/relative/ULP error,
  identical value SHA-256; finite value and score; `program_valid=[true]` on
  both endpoints.
- `T=1`, zero corrections: `-31.12767029`, `exact_equal=true`, zero error,
  finite, valid.
- All eager gates in the memo pass.

### Graph (attempt08) — WITHIN-MODE IDENTITY FAIL (confirmation veto)

Artifact:
`docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/repair_validation_attempt08/endpoint_gpu0_graph.json`

- `status=COMPLETE`, frozen identity guard `PASS`, source hashes match,
  same environment/device/TF32/deterministic/memory-growth checks as eager.
  Wall time `2342.75` s (~39 minutes; see budget note below).
- `T=1`, zero corrections: value `-31.12767029`, `exact_equal=true`, zero
  error, bitwise equal to the eager `T=1` value (same value SHA-256). The
  upstream/zero-correction control is therefore mode-stable.
- `T=20`, four corrections: **within-mode identity failed.** Value-only
  `-683.0575562`; value carried by the score endpoint `-682.4954834`; absolute
  gap `0.56207275`, scale-relative `8.23e-4`, approximate ULP error `9209`.
  Both finite, both `program_valid=[true]`, but `exact_equal=false` with
  different value SHA-256 hashes.
- The graph `T=20` score vector is also grossly different from eager: graph
  score range `[-2123.37, -463.27]` versus eager `[404.53, 2976.94]` — sign
  and magnitude disagreement, not rounding-level drift. Recorded as
  explanatory; no score tolerance exists.
- Cross-mode explanatory observation: eager value-only `-683.0018921` versus
  graph value-only `-683.0575562` (diff `0.0557`), matching the historical
  stale attempt06 eager/graph pair exactly at printed precision.

Per the memo's step-5 rule and decision table ("Eager passes, later mode fails
within-mode identity" → compiler-mode confirmation failure), execution stopped
here. XLA and invariants were NOT run. Both artifacts are preserved; nothing
was overwritten.

### Classification

Compiler-mode confirmation failure, current source. The failure is localized
to graph-mode tracing of the `T=20` four-correction route: the value program
and the ForwardAccumulator-carrying value program compile to different primal
arithmetic under `tf.function` (non-XLA graph), while eager executes the shared
primal identically. The bitwise-equal `T=1`/zero-correction control in graph
mode indicates the divergence is in the higher-moment correction (or its
interaction with forward-mode tracing), not the upstream recursion. This is a
current-source GPU confirmation failure under the declared gates — not a
research-direction rejection, not a dual-cap verdict, and not evidence against
the CPU derivative authority, which remains exact.

Budget deviation note: the memo's 15-minute combined endpoint ceiling was
exceeded by the graph arm alone (~39 minutes, dominated by graph tracing of the
Python-unrolled `T=20` recursion). The process was already consuming compute
when the ceiling passed and was allowed to complete to obtain a terminal
artifact rather than an uninterpretable `RUNNING` file. Recorded as a stop
condition hit; no further endpoint processes were launched.

### Next Justified Action

A bounded repair/localization plan for the graph-mode value-vs-JVP primal
divergence at `T=20` (for example, endpoint-level bisection over correction
steps 0→4 in graph mode, then localization of the first unequal tensor inside
the traced correction core), under a fresh reviewed scope. No tolerance may be
invented for the observed gap. NeuTra, HMC, tuning, cross-model, and dual-cap
phases remain blocked.

## Decision Table

| Decision | Primary criterion | Veto status | Next action | Not concluded |
|---|---|---|---|---|
| Keep the shared-primal repair and forward-mode public score | Passed exactly at `T=1,2,20` in terminal CPU authority artifact; passed exactly on GPU eager | GPU graph `T=20` within-mode identity FAILED (gap `0.562`); confirmation vetoed at the graph arm | Bounded graph-mode localization/repair plan; XLA and invariants deferred | Dual-cap admission, Austria NeuTra/HMC, tuning, posterior correctness |
| Reject reverse-mode substitution | Long-horizon reverse and forward derivatives disagree materially | Would change the derivative numerical program | Do not replace forward score merely for graph convenience | No claim about which FP32 derivative is closer to an unavailable exact real-arithmetic derivative |

## Next Phase

Superseded on 2026-08-18 after the GPU confirmation attempt: item 1 partially
completed (eager PASS, graph FAIL, XLA not run). The active next phase is a
bounded graph-mode divergence localization plan as described in
"Current-Source GPU Endpoint Confirmation" above. Items 2–4 remain gated
behind a passing three-mode confirmation and are unchanged in substance:

1. Rerun the frozen endpoint on RTX 5080 using the repaired `tftwogpu`
   environment and record eager, graph, and XLA results.
2. If GPU endpoint identity passes, run the full fail-closed and batch
   composition checks on GPU.
3. Run cross-model value/score regressions only after the Austria GPU gate.
4. Treat any fresh repair as invalidating the Austria tuning scope; perform
   fresh scope-specific tuning before any NeuTra proposal.
