# Phase 8 C1 strict-backend cost-rescue result

Date: 2026-08-30  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-strict-backend-cost-subplan-2026-08-30.md`  
Status: `PASS_C1_STRICT_BACKEND_COST_FEASIBLE`

## Question and evidence contract

This arm asked whether the existing `tensorflow_eigh_strict` principal-square-
root implementation both preserves the q=20 target semantics and completes the
frozen K=2 two-batch cost pilot. The old compiled-backend C1 allocation remains
closed; this is a new diagnostic repair arm. The primary pass required the
previous parity receipt plus a complete `PASS_PHASE8_COST_PILOT` manifest at
the original validation size 256, with finite target/status rows, checkpoint
replay, learned-map reliability, route scan, XLA and memory-growth receipts,
and allocator peak at most 4 GiB.

## Receipts

The parity prerequisite passed at
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-strict-backend/backend-parity/attempt-02-native-eigh/result.json`.
The cost receipt is
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-strict-backend/cost-pilot/attempt-01-native-eigh-n256/run_manifest.json`.

| Batch | Median steady update (s) | Peak TensorFlow allocation (bytes) | Cross-density work | Cross-density time (s) | Reliability |
|---:|---:|---:|---:|---:|---|
| 8 | 1.2217690244724508 | 1701254912 | 32 | 0.05610800697468221 | pass |
| 32 | 2.0411379247379955 | 1701394688 | 128 | 0.05610022600740194 | pass |

Both charts at both batch sizes completed the beta=0 preflight, beta=0.5
training, five valid optimizer updates, final checkpoint capture, and replay.
The replay maximums were zero for forward and log-determinant values and
`8.881784197001252e-16` for the latent round trip. The manifest selected B=32
under the frozen rule because it was finite, below the allocator cap, and its
median update time was below four times the B=8 median. The complete process
wall time was `261.52175762609113` seconds.

The strict target calls and updates ran on one RTX 4080 SUPER logical GPU with
XLA and TensorFlow memory growth configured before logical initialization. The
static route scan passed with no row mapping or pfor tokens. Compatibility
receipts were present and passed. Loss, pullback residuals, occupancy, and
gradient values remain diagnostic only.

## Decision table

| Decision | Primary criterion | Hard veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close strict-backend C1 rescue as feasible | Complete parity plus two-batch cost/reliability manifest | Pass; no numerical, memory, checkpoint, or route veto | Cost under larger K, longer training, and full calibration is unmeasured; two-row parity does not cover every batch layout | Refresh Phase 8 calibration with strict backend, add a larger-batch parity check, and run the predeclared architecture/optimizer screen under a new bounded subplan | No whitening, mode discovery, posterior correctness, HMC readiness, statistical superiority, or high-dimensional scaling |

## Inference status

| Evidence class | Result |
|---|---|
| Hard veto screen | Pass for this feasibility scope: parity, finite/status, replay, reliability, route scan, XLA, memory growth, and allocator cap all passed |
| Statistically supported ranking | None; one feasibility arm and no stochastic sampler comparison |
| Descriptive-only differences | Strict-backend timing, selected B=32, training loss/improvement, and pullback diagnostics |
| Default readiness | The strict backend is a viable q=20 candidate route, not a repository-wide default or scientific promotion |
| Next evidence needed | Batch-size-dependent parity, target-specific architecture/optimizer/lineage calibration, then untouched sequential HMC validation |

## Classification and red-team

The earlier timeout was a graph-cost limitation of the compiled callback route,
not evidence against the target or transport mathematics. This arm repairs
that feasibility limitation without changing the target. The strongest
alternative explanation is that the strict and compiled paths agree on the
small parity batch but diverge for unusual covariance branches or larger
training batches; the next larger-batch parity check is therefore required.
The result cannot be used to infer an IID Gaussian pullback, mode coverage, or
sampler quality.

