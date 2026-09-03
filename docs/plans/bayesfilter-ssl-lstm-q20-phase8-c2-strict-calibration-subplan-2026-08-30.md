# Phase 8 C2 strict-backend architecture and optimizer screen

Date: 2026-08-30  
Status: `READY_FOR_C2_SCREEN_AFTER_B8_PARITY`

Parent program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`  
C1 feasibility result:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-strict-backend-cost-result-2026-08-30.md`

## Scope and research question

The old compiled-backend C1 allocation is closed. C1 strict-backend feasibility
has passed under its separate repair subplan, so this is a new C2 calibration
allocation. The target is the unchanged q=20 SSL-LSTM bridge in four sampled
parameters; the only execution-backend choice is the parity-passed
`tensorflow_eigh_strict` route. No confirmation stream, HMC chain, or posterior
promotion run is allowed here.

Question: among the predeclared target-specific capacity and learning-rate
hypotheses, which architectures produce finite, reliable reverse-KL charts
after 32 fresh-Gaussian updates at `beta=0.5` on two independent initialization
roots, at the measured selected batch size B=32?

## Evidence contract

| Item | Contract |
|---|---|
| Exact baseline | The beta=0.5 reference-affine start for the same architecture/root, evaluated on the same held-out latent bank. |
| Candidate grid | `compact-high`: hidden `(16,16)`, `tanh`, `lr=1e-3`; `compact-low`: `(16,16)`, `tanh`, `lr=5e-4`; `wide-high`: `(32,32)`, `tanh`, `lr=1e-3`; `wide-low`: `(32,32)`, `tanh`, `lr=5e-4`. |
| Roots | Initialization roots `(20260830,12001)` and `(20260830,12002)`; training root `(20260830,22001)`; validation roots `(20260830,42001)` and `(20260830,42002)`; stress root `(20260830,52001)`. These are disjoint from reserved Phase 9 roots. |
| Training | One static batch of 32 fresh IID standard-normal latent rows per update; 32 updates per architecture/root; one batch-native target call per update; XLA and strict backend enabled. |
| Primary C2 pass | Every attempted map has finite/status-valid updates, exact start/final checkpoint replay, finite held-out density/score residuals, and a passing self/cross/reference learned-map reliability screen. At least one architecture remains viable on both roots. |
| Hard vetoes | Target/bridge/backend identity drift, invalid row, nonfinite update, scalar or pfor route, checkpoint mismatch, failed inverse/Jacobian/score reliability, memory-growth/XLA/device failure, output overwrite, or timeout. |
| Nomination rule | Among hard-valid rows, retain all rows whose paired held-out improvement interval has upper endpoint below zero. If more than one capacity/rate remains without supported separation, nominate the lower-cost architecture as a calibration representative; this is parsimony, not scientific superiority. |
| Explanatory diagnostics | Loss, gradient norm, clipping, reverse-KL improvement, residuals, sign occupancy, and timing. None establishes whitening, mode discovery, convergence, or sampler quality. |

## Required B=8 batch-layout parity

Before C2 training, run the new same-input q=20 parity harness at B=8,
comparing `compiled_custom_op` and `tensorflow_eigh_strict` values, analytic
scores, and statuses. The frozen tolerances are value absolute `1e-8`, score
absolute `1e-7` or scaled relative `1e-7`. A failure stops C2; timing cannot
excuse a semantic mismatch. This check is a batch-layout diagnostic, not a
claim that all possible covariance branches are equivalent.

The first attempt incorrectly placed both routes inside an XLA function and
hit the custom op's missing XLA GPU kernel. The repaired prerequisite passed in
graph mode for `compiled_custom_op` (`jit_compile=false`) versus XLA for
`tensorflow_eigh_strict` (`jit_compile=true`). This asymmetry is intentional,
recorded in the parity result note, and does not change the strict C2 training
route:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c2-backend-parity-result-2026-08-30.md`.

## C2 implementation and diagnostics

For each of the eight architecture/root rows:

1. Construct the transport from its stateless initialization seed and admit a
   fixed beta=0 preflight with the Gaussian-prior reference affine map.
2. Rebuild a fresh beta=0.5 transport from the checkpoint and evaluate a
   256-row validation bank before optimization.
3. Run 32 updates using fresh stateless Gaussian batches. Invalid rows reject
   the update and preserve the pre-update state; no replacement-row sampling is
   allowed.
4. Capture and replay the final checkpoint on the same validation bank.
5. Evaluate centered pullback log-density and pullback-score residuals and the
   paired final-minus-start reverse-KL interval.
6. Run the learned-map reliability screen on self, cross, reference, and the
   two declared sign-region points. The screen is a hard implementation gate;
   residual magnitude is not a whitening promotion metric.

All target calls use the strict analytic score path. The runner must contain no
`tf.map_fn`, `tf.vectorized_map`, `GradientTape.jacobian`, `batch_jacobian`, or
explicit pfor. Training and validation banks are disjoint by seed role.

## Budget and output ownership

The new C2 allocation is `4,200` command-wall seconds: `600` seconds for the
B=8 parity prerequisite and `3,600` seconds for the eight-row C2 screen. These
are bounded allocations, not performance forecasts. A timeout consumes the
corresponding stage allocation and closes that stage; one localized artifact
repair is allowed only in a fresh attempt directory under the same cap. The
remaining campaign reserve is not silently reassigned.

```text
docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c2-strict-calibration/
```

Every manifest records Git commit/dirty state, exact command, Python/conda and
TensorFlow versions, target/bridge/backend identities, seeds, static batch,
XLA/TF32, GPU and memory-growth receipts, target-call counts, wall time, and
source hashes. Each retry has a fresh directory.

## Commands

Batch-layout parity:

```text
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
timeout --signal=TERM --kill-after=60s 600s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_backend_parity_batch_2026_08_30.py \
--batch-size 8 \
--output-dir docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c2-strict-calibration/backend-parity/attempt-01-b8
```

C2 screen (after parity):

```text
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
timeout --signal=TERM --kill-after=60s 3600s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_phase8_c2_strict_calibration_2026_08_30.py \
--output-dir docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c2-strict-calibration/screen/attempt-01-eight-rows
```

## Default and assumption audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|
| Strict eigen backend | Passed C1 parity and cost rescue | Unseen branch/layout drift | B=8 parity plus per-row status/reliability | Reviewed q=20 candidate, not global default |
| B=32 | C1 measured cost rule selected it | Larger training memory or update instability | Allocator receipt and finite update screen | Frozen C2 hypothesis |
| Four architecture rows | Existing Phase 8 target-specific warm-start grid | Grid may miss a useful capacity/optimizer | Paired held-out improvement and reliability | Bounded search, no cross-model transfer |
| 32 updates | Original C2 screen definition | Undertraining or overfitting | Start/final paired diagnostics and update trace | Calibration cap, not quality proof |
| 256 validation rows | Frozen Phase 8 calibration bank | MC error and tail under-resolution | Residual uncertainty and explicit nonclaims | Nomination only |
| Two roots | Original C2 root count | Seed-specific failure | Per-root receipts and paired comparison | Required replication, not statistical ranking |
| Four-GiB allocator cap | C1 resource policy | Shared-device contention | TensorFlow current/peak telemetry | Hard resource veto |

## Pre-mortem and skeptical audit

| Apparent result | Alternative explanation | Cheapest discriminator | Action |
|---|---|---|---|
| All rows improve loss | All maps remain in one sign region | Held-out sign occupancy and cross-map screen | Continue to C3 branching/lineage test; do not promote |
| One row fails | Learning rate/capacity instability rather than backend error | Per-update finite/status trace and neighboring rate | Retain failed row, do not transfer its setting |
| Reliability passes | Finite local map but poor global pullback | Self/cross/reference banks and radial stress | Treat as nomination only; require Phase 9 retained diagnostics |
| Timeout | Graph or resource issue rather than numerical failure | Stage markers and allocator telemetry | One bounded harness repair; close on cap exhaustion |
| B=8 parity passes | Rare covariance branch differs | Status/reliability on every C2 row | Keep backend identity bound; no broad equivalence claim |

Audit verdict: `PASS_FOR_B8_PARITY_THEN_BOUNDED_C2_SCREEN`.

## Stop and refresh rules

Do not launch C3 lineage or C4 joint refinement until this result note is
written. A complete C2 screen refreshes the next subplan with measured per-row
cost and preserves all failed rows. A hard target/backend or budget veto stops
the campaign at C2; a candidate-specific failure triggers the smallest planned
capacity/rate repair and is not treated as evidence against the reverse-KL
direction.
