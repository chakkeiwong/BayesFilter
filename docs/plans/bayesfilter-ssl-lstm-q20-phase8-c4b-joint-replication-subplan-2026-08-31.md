# Phase 8 C4B K=4 joint-mixture replication subplan

Date: 2026-08-31  
Status: `CLOSED_PASS_REPLICATION_NO_PROMOTION_C5_REFRESH_REQUIRED`

Parent program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`

Preceding feasibility:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4a-joint-feasibility-result-2026-08-31.md`

## Purpose and boundary

C4B asks whether the C4A K=4 joint-mixture implementation and its resource
receipt replicate under a fresh initialization root and a second viable C2
architecture. It is the last calibration step before a C5 freeze decision. It
does not test retained HMC, posterior correctness, whitening, mode discovery,
or high-dimensional scaling.

Each row compares independent and joint reverse-KL copies restored from the
same four beta-0.5 start checkpoints. The joint objective and trainable alpha
logits are the only intervention. Maps remain individual bijections and are
never averaged. C4B does not reuse C4A checkpoints or target-derived particles.

## Evidence contract

| Item | Frozen choice and role |
|---|---|
| Target/measure | Frozen q=20 SSL-LSTM proper Gaussian-prior bridge on `theta in R^4`; target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`; current bridge properness receipt is required |
| Backend | TensorFlow/TFP float64, strict `tensorflow_eigh_strict`, XLA and TF32 enabled, one visible GPU, memory growth before logical-device creation |
| Components | `K=4`; fixed categorical chart bank; no state-dependent chart selection |
| Rows | `compact-high` `(16,16), tanh, 1e-3` at fresh root `(20260831,62001)` and `compact-low` `(16,16), tanh, 5e-4` at fresh root `(20260831,62002)`; both are target-specific calibration hypotheses, not defaults |
| Initialization | Reference-affine prior chart at beta 0, then beta 0.5 start with no optimizer update; four distinct stateless component seeds per row |
| Updates | Static `B=32`; 8 independent updates per chart and 8 joint updates; forecast 16 joint updates from measured compile/steady times |
| Banks | Fresh disjoint 128-row objective, 64-row reliability, and 256-row diversity banks per row; no C3/C3B/C4A or Phase 9 roots |
| Joint alpha | Uniform initial logits, `train_alpha=true`; report positivity and entropy only, never posterior mass |
| Work contract | Every joint update: one target call over `K*B=128` rows and `K^2B=512` cross-density units. Every independent chart update: one target call and zero training cross-density units. |
| Hard pass | Both rows finite/status-valid; exact work counts; byte-replayable checkpoints; four-map reliability; positive normalized alpha; no exact parameter-state duplicates; GPU/XLA/memory-growth and 4-GiB caps; each 16-update forecast <=3,600 s |
| Interpretation | Joint-minus-independent held-out objective, alpha movement, residuals, diversity, and timing are descriptive only. No statistical ranking from two rows. |
| Artifact | Fresh root `docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c4b-joint-replication/attempt-01/` |

## Procedure

1. Verify the C4A pass manifest, result note, target/backend identity, and
   source-hash/properness receipts. Reconstruct the q=20 bridge and record the
   current signature.
2. For each row, create four fresh reference-affine charts, run beta-0 and
   beta-0.5 fixed-bank preflights, and capture immutable shared start
   checkpoints. Restore independent and joint copies from those exact states.
3. Run the eight-update independent comparator and eight-update joint trainer
   with row-specific stateless roots. Record every target call, cross-density
   count, compile/steady time, loss, gradient, and validity field.
4. Restore all final checkpoints into fresh objects. Run held-out objective,
   learned-map reliability, pullback residual, alpha, diversity, duplicate-state,
   and allocator checks. Preserve each row even if the other row fails.
5. Compute each row's predeclared 16-update joint resource forecast. A resource
   failure skips that row's joint arm and does not relax the cap or alter B/K.
6. Write a result note with the required decision and inference-status tables,
   a post-run red-team, and a C5 refresh recommendation. C4B success permits
   only a C5 freeze subplan; it does not open Phase 9.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Promotion status |
|---|---|---|---|---|
| Two rows / two architectures | C4A result and C2 viable representatives | Two rows still cannot establish a ranking | Per-row hard screen and paired objective receipt | Replication hypothesis |
| `K=4`, `B=32` | C4A measured work/resource path | Quadratic density or allocator cost scales poorly | `[4,4,32]` telemetry and 4-GiB cap | Target-specific candidate |
| 8 updates / 16 forecast | C4A bounded pilot | Short trajectory may hide instability | Finite updates and explicit forecast | Feasibility only |
| Fresh roots | Reproducibility requirement | Seed-specific result or overlap with prior evidence | Seed ledger and disjoint-bank IDs | Required replication control |
| Compact-low rate `5e-4` | C2/C3B viable calibration hypothesis | Rate may undertrain joint objective | Held-out start/final diagnostics | Hypothesis, not default |

## Skeptical pre-execution audit

| Risk | Check and disposition |
|---|---|
| C4A result is accidentally reused as training state | Read C4A only for status/signatures; construct all C4B charts afresh. |
| Rows are not comparable | Independent and joint copies are restored from byte-identical per-row checkpoints; only objective/alpha policy differs. |
| Joint loss/work is misreported | Assert target call count 1, cross work 512 per update, shape `[4,4,32]`, and held-out shape `[4,4,128]`. |
| Independent accounting is hidden | Assert one target call per chart and report aggregate four calls per update. |
| Alpha is called a mode mass | Schema and result note restrict alpha to a variational-density diagnostic. |
| Short runs are overinterpreted | Two rows and eight updates are explicitly descriptive; no ranking or default change is allowed. |
| Learned maps pass only analytic fixtures | Run q=20 self/cross/reference/declared/physical-score reliability on every final map. |
| Exact duplicate maps are missed | Compare immutable parameter-state hashes; report near duplicates without a post-hoc distance gate. |
| Hidden graph or allocator violation | Scan all route and runner sources; require pre-import memory growth, one GPU, XLA, TF32, and 4-GiB cap. |
| Resource forecast is optimistic | Use first compiled update plus median of seven steady updates; retain raw timings and actual wall time. |
| Confirmation contamination | Use roots disjoint from all prior calibration and reserved Phase 9 roots; do not read retained samples. |

Audit verdict: `PASS_FOR_BOUNDED_C4B_K4_REPLICATION_DIAGNOSTIC`.

## Budget and repair rules

The C4B command cap is 3,600 wall seconds for both rows, one fresh attempt
directory, and no Phase 9 budget. Preserve a failed row and any partial
checkpoints. One localized harness/serialization repair may run in a fresh
directory under the unchanged target, K, B, objective, hardware, and total
cap. Stop on target/bridge mismatch, invalid status, wrong work count,
checkpoint failure, nonfinite update, reliability failure, exact duplicate
state, memory-growth/XLA failure, allocator breach, or exhausted forecast.

If both rows pass, refresh C5 to decide whether a K=4 joint representative can
be retained alongside the independent K=2 comparator. If only one row passes,
retain the result as a row-specific feasibility hypothesis and do not promote
the joint arm. If the resource screen fails, close the optional joint arm with
`SKIP_JOINT_ARM_RESOURCE_ENVELOPE`. All outcomes leave whitening, Phase 9,
posterior, and HMC gates closed.

## Required result interpretation

The result note must include a decision table, an inference-status table with
hard-veto, statistical-ranking, descriptive-difference, default-readiness, and
next-evidence rows, and a post-run red-team. It must distinguish a replicated
implementation/resource result from evidence about approximation quality.

## Closeout, 2026-08-31

The audited attempt completed both rows in `876.4273084410233` seconds with
status `PASS_PHASE8_C4B_JOINT_REPLICATION`. All predeclared hard checks passed:
exact target/cross-density work, finite target status, checkpoint replay,
four-map learned-transport reliability, positive normalized alpha, distinct
parameter-state hashes, XLA/GPU/memory growth, the 4-GiB allocator cap, and the
16-update forecast cap.

The objective contrasts have opposite signs across the two rows and are not
paired because the arms used separate held-out latent banks. Pullback score
residuals remain large. C4B therefore closes as an implementation/resource
replication only, with no joint-arm, architecture, whitening, HMC, posterior,
or default promotion. The terminal interpretation is in
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4b-joint-replication-result-2026-08-31.md`.
The next authorized action is a metadata-only C5 freeze refresh; Phase 9 remains
closed pending its own skeptical audit and evidence contract.
