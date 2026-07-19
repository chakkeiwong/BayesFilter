# Phase 8 Forecast Chunk Repair Native Review

Date: 2026-07-17

Verdict: `AGREE_COMPILE_CANARY_ONLY`

The failed target-pilot repair reached the external `1260`-second timeout while
XLA compiled the statically unrolled `256`-draw forecast program. No receipt was
written. The terminal filter and staged covariance audit had already passed on
the exact G/H pilot inputs, so this is a forecast graph-shape resource failure,
not evidence of a covariance recurrence or a predictive difference.

The reviewed repair adds optional draw-axis chunking to the compiled forecast
API. Terminal states are still computed and audited once for the full draw
matrix. Each forecast chunk slices the already materialized draw, terminal, and
innovation tensors, calls one reusable fixed-shape XLA program, and concatenates
the outputs in original draw order. The default remains the original unchunked
execution. The pilot alone freezes chunk size `16`.

## Findings

1. A chunk size larger than the draw dimension now fails closed. A partial last
   chunk is supported by a separately shaped compiled program; the pilot avoids
   that branch because `256` is divisible by `16`.
2. Chunked and unchunked size-two forecasts are tensor-exact on the CPU-hidden
   XLA route. Full innovation-bank signatures and tensor hashes are unchanged.
3. The pilot records chunk size in both per-chart provenance and pooled
   calibration configuration, and its trace gate now targets the size-16
   forecast program rather than the failed size-256 program.
4. Chunk size `16` is not yet trusted GPU compile evidence. A separate
   engineering-only canary uses 32 tiled A0 points, two chunks, fresh
   innovations, no retained samples, and a `600`-second internal cap. Its pass
   authorizes exact-prefix validation only, not another pilot or calibration.

## Audit

| Risk | Disposition |
| --- | --- |
| Wrong baseline | Tensor-exact unchunked output is the engineering comparator |
| Proxy promotion | Compile canary cannot establish target-pilot or calibration success |
| Hidden semantic change | Forecast equations, terminal audit, bank bytes, draw order, horizons, and replications are unchanged |
| Resource ambiguity | Failed attempt is preserved separately; canary has a fresh output and finite cap |
| Leakage | Canary uses tiled A0 points and does not name or read Phase 7 retained shards |
| Environment | Serious canary remains trusted GPU/XLA, `float64`, TF32 recorded |

Checks: two focused API/parity tests passed; five pilot-contract tests passed;
58 predictive-statistics tests passed; Python compilation and `git diff
--check` passed.

No unresolved material finding remains for the bounded compile canary.
