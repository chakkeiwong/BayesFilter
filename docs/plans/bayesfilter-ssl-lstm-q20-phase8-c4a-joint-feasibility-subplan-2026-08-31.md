# Phase 8 C4A K=4 joint mixture-RKL feasibility subplan

Date: 2026-08-31  
Status: `CLOSED_PASS_RESOURCE_AND_IMPLEMENTATION_SCREEN_C4B_REFRESH_REQUIRED`

Parent program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`

Preceding calibration:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3b-l5-ladder-result-2026-08-31.md`

## Purpose and boundary

C4A is the smallest q=20 test of the optional exact joint mixture reverse-KL
trainer on the GPU/XLA route. It answers whether a four-chart bank can execute
the joint objective with the measured quadratic cross-density term under the
existing memory and wall-time envelope. It is a feasibility and implementation
diagnostic, not a chart-quality, whitening, mode-discovery, posterior, HMC, or
architecture-selection experiment.

The independent and joint arms start from the same four immutable beta-0.5
chart checkpoints. This isolates the objective implementation: the joint arm
changes the loss and trains `alpha` logits, while the independent arm trains
each chart separately. Maps remain individual bijections; no maps are averaged.
The fresh charts are deliberately not copied from C3B, so C3B remains a
separate L5 lineage diagnostic.

## Evidence contract

| Item | Frozen choice and role |
|---|---|
| Target and measure | Frozen q=20 SSL-LSTM proper Gaussian-prior bridge on `theta in R^4`; target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`; bridge signature must be reconstructed and recorded |
| Backend | TensorFlow/TFP float64, `tensorflow_eigh_strict`, XLA and TF32 enabled, exactly one visible GPU, memory growth configured before logical-device creation |
| Components | `K=4`, independent categorical charts with unique IDs; K=4 is inherited from the reviewed Phase 8 C4 proposal and remains a target-specific hypothesis |
| Architecture/root | `compact-high`: hidden layers `(16,16)`, tanh, learning rate `1e-3`; one fresh initialization root `(20260831,61001)`; this is a feasibility representative, not a ranking |
| Initialization | Reference-affine prior chart at beta 0, then beta 0.5 with no optimizer update; four distinct stateless component seeds derived from `(20260831,61001)` |
| Batch and updates | Static `B=32`; 8 independent updates per chart and 8 joint updates; 16 joint updates are forecast from measured compile and steady times. Eight is a bounded pilot hypothesis, not a convergence setting. |
| Validation banks | Fresh disjoint 128-row base banks for objective comparison, 64-row reliability banks, and a 256-row diversity bank; all roots are distinct from C3/C3B and reserved Phase 9 roots |
| Joint parameters | Uniform initial `alpha` logits, `train_alpha=true`; alpha entropy is recorded descriptively and is never interpreted as posterior regional mass |
| Work accounting | Each joint update must report one batch-native target call over `K*B=128` rows and `K^2*B=512` cross-density work units. Independent target calls are recorded per chart. |
| Hard pass | Finite/status-valid updates; exact work counts; checkpoint and replay hashes; all four learned maps pass self/cross/reference/declared reliability; alpha is finite and normalized; no exact final chart duplicates; GPU/XLA/memory-growth and 4-GiB allocator checks pass; 16-update joint forecast is at most 3,600 seconds |
| Diagnostics | Paired held-out joint-loss and pullback-residual changes, chart distances, alpha entropy, compile/steady times, target calls, cross-density work, and memory. These nominate or explain only. |
| Artifact | Fresh manifest under `docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c4a-joint-feasibility/attempt-01/` |

## Mathematical and implementation checks

For latent rows (z_{i,b}\sim\phi), the joint trainer must evaluate

\[
\mathcal L_{\rm joint}
= \sum_{i=1}^{K}\alpha_i\frac1B\sum_{b=1}^B
\left[\log\sum_{j=1}^{K}\alpha_jq_j(T_i(z_{i,b}))
      -\log\widetilde\pi_{0.5}(T_i(z_{i,b}))
      -\log|\det J_{T_i}(z_{i,b})|\right].
\]

The target is evaluated once on the flattened `K*B` physical rows. The
cross-density tensor has shape `[K,K,B]`; its work count is `K^2 B`, while
target work is `K B`. The runner must verify these identities from returned
telemetry rather than infer them from elapsed time.

The independent comparator uses the same initial checkpoints and fresh rows
folded by the same component seeds. It has four target calls per update and no
cross-component mixture term. This is an accounting comparator, not a claim
that the two arms have equal wall cost.

## Procedure

1. Verify the C3B pass manifest, target signature, strict backend, and
   provenance-repair receipt. Reconstruct the q=20 proper bridge and record its
   current properness receipt.
2. Construct four fresh reference-affine charts, run the fixed beta-0 and
   beta-0.5 preflights, and capture one immutable start checkpoint per chart.
3. Restore two independent copies of every start checkpoint. Run the
   independent comparator and the joint trainer from those copies with the
   frozen seeds and eight updates. Preserve every update's loss, validity,
   target calls, cross-density work, gradient and elapsed-time fields.
4. Restore final checkpoints into fresh objects. Evaluate the fixed held-out
   banks, reliability screen, alpha normalization/entropy, exact state hashes,
   and allocator telemetry. No target-derived particle bank or replay is
   introduced.
5. Calculate the joint 16-update forecast from the measured compiled update
   and median steady update. If it exceeds 3,600 seconds, record
   `SKIP_JOINT_ARM_RESOURCE_ENVELOPE`; do not extend the run or relax the cap.
6. Write the manifest and a result note with decision and inference-status
   tables. A pass authorizes only a separately refreshed C4B calibration
   subplan; it does not authorize C5 freeze or Phase 9.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Promotion status |
|---|---|---|---|---|
| `K=4` | Parent Phase 8 optional-arm proposal | Quadratic density work or duplicate charts erase any benefit | Exact `[4,4,32]` work receipt and final state-hash test | Hypothesis |
| `B=32` | Strict C1 cost pilot | Larger batch can exceed allocator or compile budget | 4-GiB cap and measured update times | Reviewed target-specific feasibility value |
| 8 pilot updates / 16 forecast | Bounded C4A feasibility budget | Too short to say anything about optimization quality | Start/final held-out loss and explicit nonclaim | Convenience pilot, not convergence evidence |
| `compact-high` | Viable C2/C3B calibration representative without ranking | Capacity or rate may be unsuitable for joint objective | Four-map reliability and finite update screen | Feasibility representative |
| Trainable uniform alpha | Joint-trainer API and simplex parameterization | Boundary collapse or misleading mass interpretation | Finite normalized logits and entropy receipt | Optional mechanism only |
| 4-GiB cap and 3,600 s forecast | Reviewed parent C4 envelope | Resource failure may be mistaken for mathematical failure | Separate target/cross work, compile time, allocator peak | Hard resource veto |

## Skeptical pre-execution audit

| Risk | Check and disposition |
|---|---|
| Joint loss is not the claimed mixture objective | Verify flattened target rows, `[K,K,B]` cross density, and direct `mixture_reverse_kl_terms` identity on the run's tensors. |
| Independent and joint arms are unfairly initialized | Capture one start checkpoint per component and restore byte-identical copies before either arm. |
| Quadratic work is hidden | Require `target_call_count=1`, `cross_density_work=512` for each joint update and report independent counts separately. |
| Alpha is mistaken for posterior mass | Record entropy only; prohibit regional-mass or mode claims in the result schema. |
| A short pilot is treated as optimization evidence | Eight updates are explicitly feasibility-only; held-out changes are descriptive and cannot promote a map. |
| Learned-map fixtures are too optimistic | Run the q=20 self/cross/reference/declared/physical-score reliability screen on all four final maps. |
| Exact duplicate charts pass as diversity | Hash every final chart; exact duplicates reject the joint arm. Near duplicates are reported descriptively, not threshold-rejected post hoc. |
| Target-derived replay or particle circularity enters | Source scan and manifest prohibit posterior draws, particle replacement, and target-derived training banks. |
| Graph policy or allocator failure | Static route scan, pre-import growth verification, one visible GPU, XLA, and 4-GiB cap are hard checks. |
| Forecast extrapolation is misleading | Record compile and every steady update; forecast uses the predeclared 16-update formula and remains a resource screen only. |
| C3B metadata is silently treated as numerical input | Only its signatures/provenance are read; no C3B checkpoint is used to initialize C4A. |

Audit verdict: `PASS_FOR_BOUNDED_C4A_K4_JOINT_FEASIBILITY_DIAGNOSTIC`.

## Budget, repair, and stop rules

The C4A command cap is 3,600 wall seconds, one fresh attempt directory, and no
Phase 9 budget. A failed row is preserved. A localized harness or serialization
repair may run once in a fresh directory under the same cap only when the
target, K, B, objective, hardware, and evidence contract are unchanged. Stop
on target/bridge mismatch, invalid status, checkpoint failure, nonfinite joint
loss/gradient, wrong work count, reliability failure, exact chart collapse,
memory-growth/XLA failure, allocator breach, or exhausted forecast envelope.

If C4A passes its hard screen and the forecast fits, refresh C4B with a fresh
architecture/root replication before any candidate freeze. If the resource
envelope fails, close the optional joint arm as `SKIP_JOINT_ARM_RESOURCE_ENVELOPE`
and continue the independent K=2 path. Neither outcome opens Phase 9.

## Closeout

C4A passed all hard checks in 454.691 seconds. The measured 16-update joint
forecast was 117.723 seconds and the largest allocator peak was
3,401,816,064 bytes. The joint and independent copies had no exact duplicate
state hashes, and the joint alpha vector remained finite, positive, and close
to uniform. The detailed result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4a-joint-feasibility-result-2026-08-31.md`.

The lower joint held-out objective is descriptive evidence from one root and
eight updates only. It does not promote the joint arm. A fresh C4B
architecture/root replication is required before C5 freeze; whitening and
Phase 9 remain closed.

## Required result interpretation

The result note must state separately whether the implementation hard screen
passed, whether the resource forecast passed, whether any statistical ranking
is supported (expected: none), which differences are descriptive only, and
what evidence remains before C5 or Phase 9. It must include a post-run
red-team identifying the strongest alternative explanation, an overturning
failure, and the weakest part of the evidence.
