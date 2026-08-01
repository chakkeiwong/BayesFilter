# Phase 4 Complexity And Graph Audit

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `LOCAL_NO_DENSE_WIRING_AND_PRODUCTION_SHAPE_FEASIBILITY_SUPPORTED`

## Analytic Live State

Let batch size be `B`, particle count `N`, geometry dimension `d`, payload
dimension `r`, row chunk `Cr`, and column chunk `Cc`.

The augmented payload has `r=d+1`. Streaming potentials and particle fields use
`O(B*N*(d+r))` storage. A transport block uses `O(B*Cr*Cc)`, query/key blocks
use `O(B*(Cr+Cc)*d)`, and payload blocks use `O(B*(Cr+Cc)*r)`. Contract E adds
`O(B*N*d + B*d^2)` state. The repaired VJP-local row action adds
`O(B*N*d^2)`, which is bounded for the production `d=3` chart.

No term is `O(B*N^2)` or `O(T*N^2)`. Runtime remains quadratic in `N` for the
exact chunked transport because all row/column block pairs are visited; this
phase establishes bounded live memory, not subquadratic compute.

## Source Audit

The owned streaming composition contains no NumPy, autodiff tape,
`ForwardAccumulator`, dense transport-matrix construction, denominator floor,
mass clipping, stopped mass, raw reset, or historical stopped-scale/key route.
The generic transport VJP distinguishes geometry width from payload width.

The preflight harness and owned module contain no literal `N x N` allocation.
The tiny dense comparator exists only in the CPU-hidden test module.

## Local XLA HLO Audit

The frozen `B=2,N=4,d=2` forward HLO was obtained deliberately on CPU-XLA:

| Field | Value |
| --- | --- |
| HLO bytes | `1,409,090` |
| SHA256 | `f0a703435a9e7ace6980c8e1f062a741b7997d96b69d62eef8b24d4cde694689` |
| `while(` occurrences | `25` |
| static `[2,4,4]` transport-shape occurrences | `0` |

This supports the local graph-wiring claim only. It is not a proof about every
compiler optimization or production shape.

## Measured Trusted-GPU Evidence

At `B=1,N=10000,d=3`, float32, TF32 enabled, XLA JIT, chunks `1024`, two finite
Sinkhorn steps:

| Run | Status | GPU allocator peak | Compile/execute | Warm execute |
| --- | --- | ---: | ---: | ---: |
| Forward | Valid chart | `67,960,576` bytes | `6.907 s` | `0.175 s` |
| Analytic VJP after repair | Valid chart, finite cotangents | `84,859,392` bytes | `19.556 s` | Not run |

Both are far below the existing `14000 MiB` engineering memory ceiling. This is
descriptive one-fixture feasibility evidence, not a runtime guarantee or a
statistical comparison.

## Boundaries

Supported: the selected forward and analytic VJP graphs executed at the required
production particle count without observed dense-state memory growth or OOM.

Not established: chunk-accumulation error, finite-Sinkhorn convergence,
row-mass adequacy, reset/covariance adequacy, general derivative accuracy,
full-time filtering feasibility, admission, or scientific validity.
