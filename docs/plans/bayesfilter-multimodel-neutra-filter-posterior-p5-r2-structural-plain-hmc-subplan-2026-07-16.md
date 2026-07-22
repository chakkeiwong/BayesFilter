# P5 R2 Subplan: Structural Same-Target Plain HMC

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `REVIEWED_READY_FOR_EXECUTION`

## Objective And Entry Conditions

Admit or block a same-target plain-HMC comparator for typed `STR-UKF` identity
`e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665`.
Entry requires byte-identical CPU/GPU identity and recomposition artifacts,
valid recursive hashes, the unchanged frozen T=100 dataset, and trusted GPU
availability with memory growth.

## Evidence Contract

| Field | R2 contract |
| --- | --- |
| Question | Can fixed-kernel HMC produce a health-valid, converged retained sample from the exact typed structural UKF posterior? |
| Baseline | Source-coordinate Euclidean HMC on the admitted posterior; a target-bound affine mass is a predeclared geometry repair only if source geometry fails |
| Kernel nomination | among health-valid finite probes, maximize minimum rank-normalized bulk ESS; grid order breaks ties |
| Comparator pass | separate retained warm-up; recent warm-up modern R-hat `<=1.05`; retained modern R-hat `<=1.01`; minimum bulk ESS `>=1000`; minimum tail ESS `>=400`; all health/status checks valid |
| Hard vetoes | target/identity/hash drift, nonfinite state/value/score/energy, divergence, unmoved chains, invalid status, warm-up or retained cap, corrupted archive |
| Explanatory only | acceptance, short-probe R-hat/ESS, runtime, truth distance, posterior summaries |
| Not concluded | NeuTra quality, filter exactness, truth recovery, calibration, superiority, robustness, or readiness |

Modern R-hat is the maximum of rank-normalized split and folded rank-normalized
split R-hat. Warm-up is fully archived, excluded from posterior summaries, and
never pooled with retained draws.

## Source-Geometry Attempt

- Four chains start at the source truth center plus fixed dispersed offsets.
- Step grid: `0.005`, `0.01`, `0.02`, `0.04`, `0.08`, `0.16`.
- Eight leapfrog steps; each probe has 64 burn-in transitions and 128 draws.
- Warm-up chunks: 1,000 draws per chain; minimum 2,000; recent window 1,000;
  maximum 10,000.
- Retained chunks: 2,000 draws per chain; minimum 4,000; maximum 10,000.
- Probe root `(20260716,16000)`, warm-up root `(20260716,16101)`, retained
  root `(20260716,16201)`.

Acceptance cannot nominate or admit a kernel. Probe R-hat and ESS are short-run
diagnostics only; only the sequential retained gate admits the comparator.

## Geometry Repair

If all source probes are invalid, or a selected healthy source kernel reaches
the warm-up cap with stable finite/status-valid target evaluations but fails
modern R-hat, classify `SOURCE_GEOMETRY_FAILURE`. A follow-up subplan may build
one target-bound affine mass from tuning-only source warm-up or a checked
posterior-mode Hessian. That repair must use fresh warm-up/retained seeds and
cannot reuse failed warm-up for inference. No target, prior, data, diagnostics,
caps, or posterior can change.

## Artifacts, Handoff, And Stops

Write per-probe checkpoints, tuning selection, separate warm-up/retained chunk
and cumulative tensors, diagnostics, posterior summary, identity replay,
manifest, cell ledger, and recursive hashes.

On pass move only `STR-UKF` to `COMPARATOR_ADMITTED` and draft the target-
specific training subplan. On source-geometry failure write the bounded affine
repair subplan. On target/status failure reopen R1B. Stop after one source
attempt plus one reviewed affine repair, three identical infrastructure
repairs, or 6 GPU-hours in R2.

## Skeptical Pre-Execution Audit

Decision: `PASS`.

The comparator is target-identical, uses the shared archived sequential
controller, does not use acceptance as a pass, retains but excludes warm-up,
and has fixed 10,000-draw caps. The truth-centered initializer is a synthetic
fixture convenience and not a promotion metric. A failed source geometry
triggers the predeclared mass repair rather than target or threshold changes.
No proxy metric can establish comparator admission.
