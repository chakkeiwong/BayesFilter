# Phase 8 C3B L5 temperature-ladder repair

Date: 2026-08-31  
Status: `CLOSED_PASS_HARD_SCREEN_L5_OVERLAP_SIGNAL_NO_ARM_PROMOTION`

Parent program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`

Preceding evidence:

- `docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-lineage-overlap-result-2026-08-31.md`
- `docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3-lineage-overlap/attempt-01/run_manifest.json`
- `docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3-lineage-overlap/diversity-repair-2026-08-31/attempt-01/run_manifest.json`

## Scope and question

C3A showed that an L3 ladder `(0, 0.5, 1)` is mechanically valid but did not
produce a consistent positive-branching diversity signal. C3B asks whether
finer adjacent temperature spacing improves proper-bridge overlap while
holding the branching intervention fixed. It is a calibration diagnostic, not
a mode-discovery or sampler-confirmation run.

The target remains the frozen q=20 SSL-LSTM four-parameter bridge. No posterior
draws, HMC transitions, mixture-weight fitting, or target-derived particle
replay is introduced.

## Evidence contract

| Item | Frozen choice and role |
|---|---|
| Target/backend | q=20 proper Gaussian-prior bridge; target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`; `tensorflow_eigh_strict`; TensorFlow/TFP float64; GPU/XLA; memory growth before device initialization |
| Ladder | `BETAS=(0.0,0.25,0.5,0.75,1.0)`; L5 is the only changed factor relative to C3A |
| Arms | `pure_continuation` and `positive_temperature_branching`; branching restarts component 1 only at beta `0.5`, then continues it through `0.75` and `1.0` |
| Charts | K=2 independent transports; maps are never averaged |
| Architectures | C3 calibration representatives `compact-high` `(16,16), tanh, 1e-3` and `compact-low` `(16,16), tanh, 5e-4`; neither is a promoted default |
| Roots | Two fresh initialization roots `(20260831,15001)` and `(20260831,15002)`; training root `(20260831,25001)`; overlap/diversity roots `(20260831,45001)` and `(20260831,45002)`; no Phase 9 roots. Corresponding pure/branching rows use arm-neutral folded seeds for paired training and diagnostic banks. |
| Training | Static batch B=32; 16 optimizer updates for every positive beta, component, arm, architecture, and root; optimizer state resets at each beta; fixed preflight bank and no invalid-row replacement |
| Hard pass | All rows finite and status-valid; every checkpoint hash/context and replay passes; proper bridge cross-values and adjacent ratios are finite; learned-map self/cross/reference/declared reliability passes; one-GPU memory growth and 4-GiB row cap pass |
| Overlap diagnostic | Record acceptance and log-ratio summaries for all four adjacent pairs. Compare the minimum and median acceptance descriptively with the corresponding C3A L3 rows; no universal acceptance threshold is asserted |
| Diversity diagnostic | On fresh disjoint 256-row banks, record chart means, covariance matrices/diagonal summaries, coordinate-2 sign fractions, and pairwise distances. These are explanatory nomination signals only |
| Nomination | A branch is only a candidate for a later review if it is hard-valid on both roots and its overlap/diversity contrasts are directionally consistent across roots. This is not a statistical ranking or promotion gate |

## Lineage and checkpoint protocol

`TemperedLineageController` owns the L5 ladder, component identities, seeds,
parent indices, and immutable lineage checkpoints. At beta 0 both components
use the fixed reference-affine Gaussian initialization and receive no update.
At beta 0.25 both arms continue from beta 0. At beta 0.5 only the branching
arm's component 1 receives a fresh reference-affine restart; component 0 and
the pure arm continue. Both components continue from their beta-0.5 snapshots
at beta 0.75 and beta 1.0. Every checkpoint is restored and replayed before
the next positive-beta update. A checkpoint from one beta is never overwritten
by a later chart.

The objective remains independent fresh-Gaussian reverse KL at each level. A
lineage restart is a new initialization, not a posterior sample or a particle
replacement rule.

## Prerequisites and artifacts

The runner must refuse to start unless the C2 strict-calibration result and C3A
hard-screen/diversity-repair manifests have the expected target and backend
identities. It writes only to this fresh root:

```text
docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3b-l5-ladder/attempt-01/
```

The manifest records the exact command, git state, Python/TFP versions, target
and bridge signatures, ladder/lineage signatures, seeds, per-level target
calls and update counts, checkpoint hashes, overlap/diversity diagnostics,
allocator telemetry, route scan, wall time, and nonclaims.

## Skeptical pre-execution audit

| Risk | Check and disposition |
|---|---|
| Finer ladder accidentally changes the target | Reconstruct the bridge and require the frozen target signature and properness receipt. |
| Branch arm is not isolated | Same maps, architecture, batch, updates, roots, and bank sizes; arm-neutral folded seeds pair the stochastic inputs, so only component-1 restart at beta 0.5 differs. |
| L5 cost exceeds the remaining C3 allocation | Four positive levels are measured explicitly; a 5,000-second material cap is below the remaining 7,200-second C3 allocation after C3A and repair. |
| Overlap uses an invalid shortcut | Evaluate all states with `ProperBridgeReplicaExchange` and the full prior-plus-beta-likelihood bridge. |
| Diversity summaries become mode evidence | Sign labels, covariance, and distances are explicitly descriptive; no basin, mass, or discovery conclusion is permitted. |
| Checkpoint mutation or stale scope | Restore/hash every level before continuing; bind target, bridge, backend, ladder, root, and architecture in scope metadata. |
| Learned-map reliability hidden by analytic fixtures | Apply self, cross, reference, declared, and physical-score checks to every beta-one chart before row completion. |
| Graph-policy or allocator failure | Scan the inference routes for forbidden row mapping/pfor tokens; require one visible GPU, pre-import memory growth, static B=32, and a 4-GiB row cap. |
| Short runs overinterpreted | Two roots and 16 updates are calibration evidence only; no ranking, default, whitening, posterior, HMC, or scaling claim. |

Audit verdict: `PASS_FOR_BOUNDED_C3B_L5_CALIBRATION_DIAGNOSTIC`.

## Attempt-01 repair

The first launch reached the beta-0.25 component-0 final checkpoints for all
eight rows, then every row stopped with `IndexError: list index out of range`.
The failure was in the new harness: `parent_index` from
`TemperedLineageController` is a component index within the previous beta
slice, but the runner incorrectly used it as an index into the beta-slice
list. No candidate row or overlap result was produced. The repair now reads
the immediately preceding beta slice (`beta_records[-1][component_index]`) for
continuations, preserves the immutable checkpoint semantics, and records the
failed attempt at
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3b-l5-ladder/attempt-01/run_manifest.json`.

This is a localized harness retry under the unchanged target, method, budget,
and evidence contract. The fresh retry directory is `attempt-02`.

## Attempt-02 closeout

The repaired attempt completed all eight rows in
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3b-l5-ladder/attempt-02/run_manifest.json`
with status `PASS_PHASE8_C3B_L5_LADDER` in 3,080.149 seconds. The complete
interpretation is recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3b-l5-ladder-result-2026-08-31.md`.

The L5 minimum and median adjacent-acceptance diagnostics were 0.3856 and
0.5101, compared descriptively with 0.2350 and 0.3482 for the preceding L3
rows. Positive branching produced a larger beta-one mean-distance summary than
pure continuation for both roots of each architecture, but covariance and
sign-occupancy contrasts were mixed. These finite-bank contrasts do not support
a statistical ranking or a default branching policy. C3B is closed as a
hard-valid calibration diagnostic; no arm is promoted and Phase 9 remains
closed.

The original attempt-02 manifest did not include the imported C3 runner in its
source-hash map. A supplemental provenance receipt records that dependency and
the unchanged helper hash; it repairs metadata provenance only and does not
alter the numerical evidence.

## Budget and stop/repair rules

The fresh C3B allocation is 5,000 command-wall seconds, one attempt directory,
and no Phase 9 budget. Stop on target/bridge/lineage mismatch, invalid target
status, checkpoint or replay failure, forbidden route token, memory-growth/XLA
failure, allocator-cap breach, missing diversity field, or exhausted cap. A
single candidate-row failure is recorded as candidate or harness evidence; the
smallest repair may be attempted only in a fresh directory within the same
allocation. Do not relax the target, replace invalid rows, or promote a branch
because it has a larger descriptive distance.

After the result, write a result note with decision and inference-status tables
and a post-run red-team. If the hard screen passes, refresh C3B/C4 only after
checking whether finer spacing supplied a reproducible diagnostic signal.
Phase 9 tuning and HMC remain closed in every outcome.
