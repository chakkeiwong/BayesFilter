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
