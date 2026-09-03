# Phase 8 C3 lineage and temperature-overlap pilot

Date: 2026-08-30  
Status: `CLOSED_PASS_HARD_SCREEN_NO_BRANCHING_NOMINATION`

Parent program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`  
Preceding result:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c2-strict-calibration-result-2026-08-30.md`

## Scope and research question

C2 established that the strict q=20 target and four small transport hypotheses
can execute, but its held-out pullback score residuals were large. C3A asks a
narrower question before any HMC: does positive-temperature branching create
different, reproducible chart lineages with usable adjacent-temperature overlap,
or do pure continuation and restart lineages collapse to the same local chart?

This is a calibration diagnostic. It does not estimate posterior mode masses,
prove mode discovery, establish whitening, or consume Phase 9 confirmation
streams.

## Evidence contract

| Item | Contract |
|---|---|
| Target and backend | q=20 SSL-LSTM, four sampled coordinates, proper Gaussian-prior bridge, `tensorflow_eigh_strict`, TensorFlow/TFP float64, GPU/XLA, memory growth before initialization |
| Arms | `pure_continuation` and `positive_temperature_branching`; both use K=2 and L3 `(0, 0.5, 1)`; the branching arm restarts component 1 at beta=0.5 while component 0 continues |
| Architectures | `compact-high` `(16,16)`, tanh, 1e-3 and `compact-low` `(16,16)`, tanh, 5e-4; these are C2 viable representatives, not promoted defaults |
| Roots | Two fresh lineage roots `(20260830, 13001)` and `(20260830, 13002)`; training root `(20260830, 23001)`; overlap bank roots `(20260830, 43001)` and `(20260830, 43002)`; no Phase 9 roots |
| Training | B=32; 16 fresh IID Gaussian updates for each positive beta and component; optimizer state resets at each beta; no invalid-row replacement |
| Hard pass | Every declared chart has finite/status-valid updates, immutable checkpoint replay, finite target evaluations, and a reliable inverse/logdet/score screen on self, cross, reference, and declared points |
| Overlap diagnostic | For each arm/root, generate a common 64-chain physical bank at each ladder level, evaluate the proper bridge, compute adjacent swap log-ratio acceptance and finite/status counts using `ProperBridgeReplicaExchange`; this is descriptive overlap evidence only |
| Diversity diagnostic | Compare beta-one chart physical means, covariance summaries, sign-region occupancy, and cross-chart distances on disjoint banks; no summary is a mode-discovery criterion |
| Nomination | A branching arm is worth carrying forward only if it is hard-valid on both roots and has a descriptively nonzero adjacent-overlap screen plus a larger cross-lineage distance than pure continuation. No superiority claim is made without a predeclared uncertainty analysis |

## Lineage and checkpoint protocol

`TemperedLineageController` is the authority for beta ladder, component IDs,
parent indices, stateless seed derivation, and immutable lineage checkpoints.
Each arm writes a manifest for beta indices 0, 1, and 2. At beta 0 all charts
use the exact reference-affine Gaussian map and receive no optimizer update. At
beta 0.5 the pure arm restores each beta-zero chart; the branching arm restores
component 0 and constructs a fresh reference-affine component 1. At beta 1
both components continue from their beta-half checkpoints. Every checkpoint is
reconstructed before the next update and replayed on a fixed bank.

The actual objective remains the independent fresh-Gaussian reverse-KL trainer.
Lineage metadata never substitutes a particle measure, and no map arithmetic
or replacement-row conditioning is permitted.

## Budget and artifacts

The fresh C3A allocation is `7,200` command-wall seconds, with a `600` second
preflight/parity allowance and a `6,600` second pilot cap. A timeout or
localized harness failure consumes this allocation and is retried only once in
a fresh attempt directory. The output root is:

```text
docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3-lineage-overlap/
```

Every manifest records command, git commit/dirty state, target/bridge/lineage
signatures, exact seeds, backend, XLA/TF32, GPU and memory-growth receipts,
target calls, wall time, checkpoint hashes, overlap metrics, and nonclaims.

## Skeptical pre-execution audit

| Risk | Check and disposition |
|---|---|
| Wrong target or stale C2 map | Reconstruct the bridge and require the frozen q=20 target signature; C3 uses fresh transports and roots |
| Branching mistaken for discovery | Region occupancy and chart distance are explanatory only; no mode claim is allowed |
| Pure and branching arms not comparable | Same K, L3, architecture, B, update count, overlap bank sizes, and target-call accounting; only parent/restart policy differs |
| Checkpoint overwrite or mutable chart | Hash every checkpoint and restore before each beta; fail closed on replay mismatch |
| Invalid row selection | Fixed preflight bank, finite repair ladder, and trainer rejection preserve the pre-update state; no retry-until-valid batch |
| Swap overlap computed from a wrong law | Use the exact proper bridge and `ProperBridgeReplicaExchange`; record cross values/statuses and call count |
| Large C2 residuals hidden by overlap | Report pullback score residuals and overlap separately; C3 may nominate a lineage but cannot promote whitening |
| GPU allocator or graph failure | Require strict backend, XLA, one visible GPU, memory growth, static B=32, and a 4-GiB per-arm cap |
| Underpowered comparison | Two roots are a calibration replication, not a statistical ranking; result note must label all differences descriptive |

Audit verdict: `PASS_FOR_BOUNDED_C3A_LINEAGE_OVERLAP_PILOT`.

## Stop, repair, and refresh rules

Stop C3A on a target/bridge signature mismatch, nonfinite/status-invalid target,
checkpoint/replay mismatch, forbidden row mapping/pfor, memory-growth/XLA
failure, or exhausted cap. A single arm failure is candidate evidence and
triggers the smallest planned repair; it is not evidence against the proper
bridge or reverse-KL direction. After the result note, refresh a C3B subplan
for K=4/L5 or the next smallest ladder repair only if C3A supplies valid
overlap and lineage receipts. Phase 9 tuning and HMC remain closed.

## Closeout

C3A completed on 2026-08-30 with all eight rows hard-valid. The initial runner
omitted covariance and sign-occupancy summaries named in the evidence contract;
the focused repair subplan
`bayesfilter-ssl-lstm-q20-phase8-c3-diversity-repair-subplan-2026-08-31.md`
reconstructed every beta-one chart from immutable checkpoints and supplied the
missing diagnostics on fresh banks. See
`bayesfilter-ssl-lstm-q20-phase8-c3-lineage-overlap-result-2026-08-31.md`.

Positive branching was not nominated: its mean/covariance diversity was not
consistently larger than pure continuation across the two roots and two
architectures. Both arms remain mechanically viable comparators. The result is
diagnostic only; C2's large pullback-score residuals still block whitening and
Phase 9 remains closed. The next approved step is a fresh, audited L5 ladder
repair that keeps pure continuation as the comparator and tests the same single
positive-temperature restart event under finer spacing.
