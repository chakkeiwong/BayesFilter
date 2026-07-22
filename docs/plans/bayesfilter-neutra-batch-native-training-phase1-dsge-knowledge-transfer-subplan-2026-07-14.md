# Phase 1 Subplan: DSGE Batch-Native Knowledge Transfer

Date: 2026-07-14

Master program:
`docs/plans/bayesfilter-neutra-batch-native-training-knowledge-transfer-master-program-2026-07-14.md`

## Phase Objective

Convert the relevant `~/python` NeuTra experience into a source-anchored local
design specification. Transfer mechanisms, failure lessons, evidence structure,
and execution alternatives without importing target-specific DSGE mathematics,
NumPy training code, Python optimizer loops, or `tf.map_fn` fallbacks.

## Entry Conditions Inherited From Phase 0

- Generic NeuTra training fails closed without an inspected batch method.
- Current exact LGSSM serious training is blocked before artifacts or updates.
- Legacy BayesFilter optimizer fixtures are non-executable migration evidence.
- The scalar SVD/eigh graph-status LGSSM remains the numerical authority.
- No GPU work or target-specific training is authorized by this documentation
  phase.

## Required Artifacts

- A source-anchored transfer specification at
  `docs/plans/bayesfilter-neutra-batch-native-training-phase1-dsge-knowledge-transfer-spec-2026-07-14.md`.
- A machine-readable source ledger under
  `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase1/`.
- Phase 1 result/close record.
- Drafted and suitability-reviewed Phase 2 batch-materialization subplan.

## Required Source Anchors

| Source | Required inspection |
| --- | --- |
| `~/python/src/dsge_hmc/experiment_adapters/ssm_equivalence.py` | genuine batched adapter, analytical batched prior path, and forbidden `tf.map_fn` fallback |
| `~/python/scripts/train_nk_svd_ukf_neutra_phase2_canary.py` | compiled fixed-shape batch target and custom-gradient score bridge; Python optimizer loop and NumPy diagnostics classified as non-transferable |
| `~/python/scripts/run_neutra_paper_style_at_baseline.py` | persistent CPU worker sharding as an alternative target-evaluation topology, not pre-generated training data |
| SGU and Rotemberg serious launch summaries | exact `B=480`, 96 workers, five target rows per worker, GPU flow/CPU target split, and non-performance status of launch metadata |
| `bayesfilter/linear/experimental_batched_kalman_tf.py` | leading batch-axis tensor algebra and one time-axis loop |
| `bayesfilter/linear/kalman_svd_derivatives_tf.py` | scalar SVD/eigh graph-status value/score/status semantics to preserve |

## Transfer Classification

Every cited mechanism must be classified as exactly one of:

- `transfer_reusable`: shape/control-flow/evidence mechanism applicable to the
  LGSSM target;
- `alternative_topology`: useful comparator or repair route but not selected by
  default;
- `target_specific_hypothesis`: DSGE choice requiring LGSSM-specific evidence;
- `rejected_policy_incompatible`: Python loop, NumPy implementation,
  `tf.map_fn`, scalar replay, host callback, or target/status drift; or
- `historical_evidence_only`: artifact that establishes prior execution but not
  current correctness or speed.

## Training-Sample Semantics

Reverse-KL NeuTra does not require a fixed pre-generated training dataset in the
selected route. Stateless base noise is generated inside the compiled training
program, transformed by the flow, and evaluated by the batch target. The DSGE
96-worker topology sharded each current optimizer step's target value/score
batch across persistent CPU processes. It is therefore an alternative
target-evaluation topology, not evidence for offline sample generation.

If a future objective uses replay samples or an external dataset, that is a new
scientific/training contract and must use the repository's separate multicore
CPU sample-generation policy.

## Required Checks And Reviews

1. Hash each required source file and artifact inspected.
2. Record line anchors for every transferred or rejected mechanism.
3. Verify that every master-program Phase 2-7 design dependency has a transfer
   classification or is explicitly target-local.
4. Search the specification for unsupported speed claims and for ambiguous uses
   of “batch,” “sample generation,” “training data,” and “XLA.”
5. Confirm that no DSGE model default, architecture, optimizer value, threshold,
   or timing is silently promoted to an LGSSM default.
6. Run `git diff --check` on Phase 1 documentation/artifacts.
7. Perform a local next-subplan suitability review; Claude review is optional
   because the program-level substantive call was unavailable after successful
   health/path probes.

## Evidence Contract

| Item | Phase contract |
| --- | --- |
| Question | Is the exact prior NeuTra knowledge relevant to this LGSSM program captured and correctly bounded? |
| Pass criterion | Every material mechanism has exact anchors and a transfer classification; Phase 2 can be implemented without rediscovering topology or conflating target evaluation with training-data generation. |
| Hard veto | Unsupported source claim, target/status substitution, policy-incompatible route labeled reusable, or imported DSGE default presented as LGSSM evidence. |
| Explanatory only | Historical run time, worker count, batch size, launch status, and DSGE loss curves. |
| Artifact | Transfer spec, source ledger, and Phase 1 result. |
| Nonclaims | Documentation does not establish LGSSM parity, speed, transport quality, HMC readiness, or scientific validity. |

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Selected topology is GPU/XLA batch target | BayesFilter owner policy and NK compiled route | FP64 SVD batch kernel may be slower than CPU shards | Phase 6 compares target and training timings; CPU shards remain repair alternative | implementation hypothesis |
| One time loop, no sample loop | Kalman dependence plus local batched kernel | graph may retain inefficient broadcasts | Phase 3 graph/source audit and Phase 6 component timing | reviewed mechanism |
| Custom-gradient score injection | NK canary and current generic trainer | value/score could come from different programs | Phase 5 identical-call and finite-difference objective checks | reviewed mechanism |
| CPU workers are not selected default | Owner GPU training policy | selected GPU kernel may be impractical | Phase 6 repair can nominate CPU shard topology under a refreshed subplan | alternative topology |
| DSGE batches 128/480 are not LGSSM defaults | historical source | cargo-cult batch size can distort memory/performance | Phase 6 batch ladder while final comparison retains 128 | target-specific hypothesis |

## Skeptical Subplan Audit

- Wrong source baseline: both the optimized compiled NK route and the
  persistent CPU-worker route are inspected; neither is treated as universal.
- Proxy promotion: launch manifests establish configuration, not measured
  performance or correctness.
- Hidden fallback: `tf.map_fn` analytical-prior and tensor-builder paths are
  explicitly classified as rejected.
- Sample ambiguity: in-graph base noise, target-evaluation batches, offline
  replay data, and posterior/HMC samples are separated.
- Environment mismatch: DSGE artifacts may name other GPUs/paths; only topology
  is transferred, not device identity or runtime.
- Missing handoff: Phase 2 receives an exact tensor/derivative shape table and a
  list of local functions to vectorize.

Audit verdict: **PASS WITH REQUIRED CLASSIFICATION DISCIPLINE**. The phase is
documentation-only, bounded, and directly prevents the observed knowledge-loss
failure. The noted `tf.map_fn` and CPU-worker nuances are incorporated rather
than deferred.

## Forbidden Claims And Actions

- Do not call the DSGE codebase uniformly batch-native.
- Do not describe CPU worker target evaluation as pre-generated training data.
- Do not copy NumPy or Python optimizer-loop implementation into BayesFilter.
- Do not substitute Cholesky/QR/sigma-point math for the frozen SVD/eigh target.
- Do not claim that historical SGU/Rotemberg launch configuration proves speed,
  completion, correctness, or LGSSM suitability.
- Do not run GPU training or change model/optimizer code in Phase 1.

## Exact Next-Phase Handoff Conditions

Phase 2 may begin when the transfer specification and source ledger pass local
checks, Phase 1 records no unresolved target/status ambiguity, and the Phase 2
subplan gives exact `[B,18]` materialization outputs, derivative shapes, scalar
parity tests, and no-loop/no-NumPy checks.

## Stop Conditions

Stop only if a required source is missing/corrupt, the scalar authority cannot
be identified, source inspection shows that a planned mechanism changes the
target/status law, or Phase 2 cannot be specified without a material scientific
choice. Reviewer unavailability or stale historical timing is not a blocker.

## Phase-End Procedure

1. Run all required local checks.
2. Write the Phase 1 result/close record.
3. Draft or refresh the Phase 2 batch-materialization subplan.
4. Review Phase 2 for consistency, correctness, feasibility, artifact coverage,
   default/assumption coverage, and boundary safety.
5. Continue if no real blocker exists.

