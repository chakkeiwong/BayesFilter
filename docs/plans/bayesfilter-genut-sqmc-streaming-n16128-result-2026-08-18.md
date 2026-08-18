# GenUT SQMC Streaming and N=16128 Result and Reset Memo

Date: 2026-08-18  
Plan: `docs/plans/bayesfilter-genut-sqmc-streaming-n16128-plan-2026-08-18.md`

## Direct Result

The exact tiled transport implementation and focused parity tests are in
place, but the requested `N=16128` campaign was not launched. The required
same-source GPU route replay could not be run because the trusted GPU command
permission service returned `502 Bad Gateway` twice. The archived dense rows
also cannot be used as the primary parity baseline: their source hashes differ
from the current streaming source.

It would be wrong to report the existing `N=16128` target as a successful
streaming result. The first streamed smoke rows are retained as engineering
artifacts only and do not clear the route-level parity gate.

## Implementation

The implementation adds an explicit `transport_plan_mode` with `dense` and
`streaming` branches. The streaming branch evaluates the same finite
multiplicative Sinkhorn program in exact-divisor tiles, preserving the cost
scale, 16 alternating updates, denominator floor, row quotient, raw/post-
quotient marginal diagnostics, Contract-E reset, trust-region correction, and
analytical score recursion. For `N=16128`, the active repository policy selects
`K=2688` and a `6 x 6` block grid.

Changed implementation and harness files:

- `bayesfilter/highdim/genut_guided_proposal_tf.py`
- `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`
- `docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py`
- `tests/highdim/test_genut_sqmc_score_blocking.py`

## Verification

| Check | Status | Evidence |
|---|---|---|
| Dense/streamed FP64 multi-tile transport parity | PASS | focused test |
| Dense/streamed FP32 multi-tile transport parity under CPU XLA | PASS | newly added focused test |
| CPU XLA streaming compilation | PASS | focused test |
| Trust-region reset parity on the current small route | PASS | focused test |
| Full Austria `T=2,N=36` value/score/final-particle parity | PASS | focused test |
| Focused suite | PASS | `10 passed` |
| Python compilation | PASS | touched Python files |
| Patch whitespace check | PASS | `git diff --check` |
| Current-source GPU dense/streamed route replay | BLOCKED | fresh dense row completed, matched streamed row must be rerun after one-block baseline repair; wrapper intermittently returns `502 Bad Gateway` |
| `N=16128` all-variant run | NOT RUN | continuation veto not cleared |

## Retained GPU Artifacts

The following rows were finite and structurally valid, but are not scientific
or parity evidence until the current-source dense comparator is completed:

- `docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/smoke_attempt01/result.json`
  (`N=1008`, repaired permutation, seed `97701`)
- `docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/smoke_attempt02/result.json`
  (`N=4032`, repaired permutation, seed `97701`)

The archived dense comparator in
`docs/benchmarks/artifacts/genut-sqmc-particle-count-trust-region-20260817/claim_attempt01/result.json`
uses different hashes for the transport/reset module, the route wrapper, and
the harness. Its differences from the streamed rows are therefore historical
source drift, not an isolated streaming error or a streaming success.

The one-block discrepancy was then repaired: when the active policy selects
`K=N`, streaming now reuses the dense arithmetic directly, while `N>3000`
continues to use bounded tiled evaluation. The fresh dense baseline that
motivated this repair is `smoke_attempt03/result.json`: value
`-681.3641357421875`, score `(294.1878052, -133.5684662, 5.8378406)`, with
the same input hashes as the earlier `N=1008` streamed row. Two post-fix
streamed retries returned `502 Bad Gateway` before creating an artifact.

For reference, the streamed rows were:

| N | Value | Score `(j0,j1,j2)` | TV | Unique ancestors |
|---:|---:|---|---:|---:|
| 1008 | `-682.6151123` | `(393.73398, -154.54193, 8.82004)` | `1.87e-6` | `1008` |
| 4032 | `-681.6522217` | `(-663.29675, 155.72716, 5.47599)` | `6.65e-6` | `4032` |

These are descriptive engineering outputs only; they do not establish score
correctness, an SQMC rate, a route ranking, or particle-count improvement.

## Decision Table

| Decision | Primary criterion | Status | Next justified action | Not concluded |
|---|---|---|---|---|
| Keep exact streaming implementation | source structure and focused parity | PASS | retain explicit dense comparator and tile policy | large-N feasibility |
| Accept route-level streaming parity | same-source GPU replay after one-block repair | BLOCKED | rerun streamed `N=1008` on GPU1, then `N=4032` | historical artifact comparison |
| Run `N=16128` | parity gate plus resource/validity checks | NOT AUTHORIZED BY PLAN YET | execute one repaired-permutation row, then four variants, after parity pass | all-variant result |

## Reset State

Resume from the plan and this memo. The next run must use a fresh attempt
directory and record current source hashes. The one-block `K=N` path now uses
the dense arithmetic baseline, so rerun the same-source streamed `N=1008`
repaired-permutation row with seed `97701` and compare it with
`smoke_attempt03`. Require the frozen value/score parity tolerances. Then
repeat the gate at `N=4032`. Only after both pass should `N=16128`, `K=2688`,
seed `97701`, and the four reviewed SQMC routes be attempted. Do not relax
tolerances, substitute the annealed streaming OT solver, or use the stale dense
artifact as a parity oracle.

The strongest current conclusion is engineering-only: the tiled helper is
structurally bounded and passes deterministic small-case parity tests. The
weakest point remains the unexecuted current-source GPU route comparison.

## 2026-08-19 Claude Code Session Update (Supersedes Tables Above)

Session: Claude Code, trusted execution boundary, 2026-08-19 ~03:05-03:20 HKT.
Git HEAD: `a13c481b40ec62286967d2e67e2d79f0228bb5b6` (moved from the memo's
`dae37183`; all nine in-scope SHA-256 hashes recomputed and verified identical
to the memo table, and the streaming files are now tracked by Git). Environment:
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, TensorFlow
`2.20.0-dev0+selfbuilt`, Python 3.11.

### Session preflight evidence

- Trusted `nvidia-smi`: GPU0 RTX 5080 6% util / 12295 MiB free; GPU1
  RTX 4080 SUPER 0% util / 16035 MiB free. GPU1 selected per availability rule.
- Device-order correction (recorded deviation from the memo command): CUDA
  enumeration order on this machine differs from `nvidia-smi` PCI order, so all
  session GPU commands added `CUDA_DEVICE_ORDER=PCI_BUS_ID` to make
  `CUDA_VISIBLE_DEVICES=1` provably select the RTX 4080 SUPER. The mandated
  probe passed with device identity confirmed
  (`NVIDIA GeForce RTX 4080 SUPER`, compute capability 8.9, PCI `09:00.0`),
  exactly one physical and one logical GPU, memory growth `True` verified
  before initialization, matmul on `/GPU:0` printing `512.0`.
- CPU-hidden focused suite (`CUDA_VISIBLE_DEVICES=-1`): `10 passed`.
  `py_compile` and `git diff --check` passed.

### Fresh post-repair GPU pairs (attempts 04-07)

All four rows: `--stage smoke`, seed `97701`, route `repaired_permutation`,
reset `trust_region`, TF32 `true`, XLA `true`, verified memory growth on the
RTX 4080 SUPER, `score_child_block_size=126`, finite, program-valid,
row-valid, permutation-valid, zero saturation, full unique ancestors.
Provenance comparison passed on every invariant field (git commit, source
hashes, model, horizon, seed, controls, initial/process/ancestor input hashes)
before any value comparison, for both pairs.

| Attempt | Plan | N | Chunks | Value | Score `(j0,j1,j2)` | Row seconds | Alloc peak bytes |
|---|---|---:|---|---:|---|---:|---:|
| 04 | dense | 1008 | `1008` | `-682.2589111328125` | `(-603.212646, 46.615898, 9.699236)` | 43.58 | 679926016 |
| 05 | streaming | 1008 | `1008` | `-682.2589111328125` | identical to 04 | 44.49 | 679926272 |
| 06 | dense | 4032 | `4032` | `-681.9028930664062` | `(167.980194, -104.879349, 5.878852)` | 120.40 | 8285741824 |
| 07 | streaming | 4032 | `2016x2016` | `-681.6522216796875` | `(-663.296753, 155.727158, 5.475986)` | 126.60 | 8285744896 |

### Gate outcomes

- `N=1008` parity gate: `PASS` with bit-exact equality of value and all three
  score coordinates, as predicted by the one-block `K=N` dense-arithmetic
  repair. The one-block repair is therefore admitted for GPU route parity.
- `N=4032` parity gate: `FAIL` (continuation veto for the ladder). Absolute
  value difference `0.2507 > 0.05`; per-coordinate relative score differences
  `4.95`, `2.49`, and `6.85e-2`, all far above `5e-3`. All validity screens
  (finite, TV `<= 1e-4`, saturation, ancestry, permutation) passed on both
  rows; the failure is numerical parity only.
- `N=16128`: not launched. Both parity gates were required; the second failed.

### Determinism cross-checks (descriptive)

- Attempt 07 is bit-identical in value and score to pre-repair attempt 02,
  confirming the tiled `N=4032` route is deterministic and was not altered by
  the one-block repair (which only touches `K=N`).
- Attempt 06 is bit-identical to the older 20260817 dense claim artifact's
  `N=4032` row despite intervening wrapper/harness source drift, confirming
  the dense route is deterministic and stable across that drift.
- The dense/streamed difference is therefore a systematic, reproducible
  arithmetic-path difference, not run-to-run noise.

### Primitive-level GPU diagnostic (debugging-only, no research decision)

A focused GPU diagnostic compared `_dense_sinkhorn_barycentric_value` and
`_streaming_sinkhorn_barycentric_value` directly at `N=4032`, `K=2016`, with
the route's actual controls (`epsilon=8.0`, 8 Sinkhorn steps, 8 balance
steps), XLA-compiled, on synthetic Gaussian clouds:

| TF32 | Max abs barycentric diff | Barycentric scale | Relative |
|---|---:|---:|---:|
| on | `1.24e-5` | `1.79e-1` | `~7e-5` |
| off | `5.2e-8` | `1.79e-1` | `~3e-7` |

Marginal-mass and residual diagnostics were similarly small; `cost_scale` was
bit-identical. Interpretation: the tiled transport primitive itself agrees
with dense to about `1e-5` under TF32 (and to FP32 roundoff without TF32). The
`O(0.25)` value and `O(10^2)` score divergence over the full `T=20` route is
therefore attributable to downstream amplification of tiny TF32
reduction-order perturbations by the nonlinear trust-region correction and
discrete pipeline stages, not to a tiled-reduction implementation bug. This is
the same amplification mechanism previously observed in the one-block case.
This attribution is supported but not proved end-to-end: the diagnostic used
synthetic clouds, and no full-route TF32-off arm was run because TF32 is a
frozen control in this campaign.

### Updated decision table

| Decision | Primary criterion | Status | Next justified action | Not concluded |
|---|---|---|---|---|
| Admit one-block `K=N` repair | fresh same-source GPU `N=1008` pair | `PASS` (bit-exact) | none; admitted | tiled parity |
| Admit tiled route (`N>3000`) | fresh same-source GPU `N=4032` pair | `FAIL` (parity veto) | user decision on next direction (see below) | that the tiled route is wrong at primitive level (evidence points against) |
| Launch `N=16128` rows | both parity gates pass | `NOT CLEARED` (second gate failed) | none under current plan | feasibility at `N=16128` |
| Promote method/default/HMC/NeuTra status | none exists in this campaign | `OUT OF SCOPE` | none | any promotion claim |

### Inference status

| Row | Status |
|---|---|
| Hard veto screen | `N=4032` streamed and dense rows both pass all validity screens; the parity gate itself failed |
| Statistically supported ranking | none; one seed, no uncertainty analysis |
| Descriptive-only differences | all value/score/runtime/allocator differences |
| Default-readiness | not established for the tiled route |
| Next evidence needed | user-authorized diagnosis arm (see options) |

### Budget accounting

GPU row time this session: `43.58 + 44.49 + 120.40 + 126.60 = 335.1` seconds,
plus about 25 seconds for the primitive diagnostic. Cumulative campaign GPU row
time including attempts 01-03: about 625 seconds. No `N=16128` GPU time
consumed. Localized infrastructure repairs used: 0 of 2. No gateway failures
occurred in the Claude Code session; the previous Codex approval-gateway
blocker did not reproduce under the Claude Code trusted boundary.

### Candidate next actions (user decision required)

The `N=4032` parity failure is a promotion veto for the tiled route under the
frozen gates, and a continuation veto for the `N=16128` ladder. It is not
evidence that the tiled implementation is wrong: primitive-level parity is
tight, and the frozen `0.05` / `5e-3` full-route gates implicitly assumed the
route is not chaotically sensitive at `T=20`, which the one-block episode had
already called into question. Options for the owner:

1. Authorize a bounded TF32-off diagnostic arm (dense and streamed `N=4032`,
   diagnostic-only, explicitly outside the frozen TF32 control) to test
   whether the divergence collapses without TF32, discriminating
   "TF32-seeded chaos" from "tiling-order-seeded chaos".
2. Authorize a revised parity criterion that acknowledges trajectory-level
   sensitivity (for example, distributional or diagnostic-level comparison
   instead of trajectory value/score equality) with review.
3. Stop the streaming `N=16128` direction here and retain the dense route.

### Nonclaims

No exact Austria-SIR observed-data score, no SQMC variant ranking, no variance
rate, no `N=16128` behavior, no HMC/NeuTra/production/default readiness, and
no equivalence to the annealed streaming OT algorithm. The trust-region
amplification attribution is a supported working hypothesis, not a proved
mechanism.
