# Phase 2 Subplan: True-Batched And Nested Fixture Tensor Algebra

Date: 2026-07-11
Status: `PHASE_CLOSED_LOCAL_GATE_PASSED_PHASE3_REVIEW_PENDING`

## Phase Objective

Replace Python row construction of all 16 batched model/derivative tensors with
TensorFlow tensor algebra and repair the scaling fixture so `P=50` is an exact
prefix of `P=150`, `B=1/4` select stable rows from one canonical `B=16` proposal
cloud, and observations/base tensors are identical across `P/B` within a fixed
dimension/dtype/timestep cell.

This phase establishes fixture semantics and graph structure only. It does not
edit QR/Kalman algorithmic kernels, vectorize parameter-axis derivative helpers,
repair batched autodiff, change timing boundaries, execute XLA, or probe GPU.

## Entry Conditions Inherited From Phase 1

- Phase 1 v2 method isolation, strict JSON, stage recovery, schedule identity,
  status, and resume contracts passed 51 focused tests.
- The GPU-hidden, explicitly non-JIT one-method integration smoke completed and
  exact resume returned `reusable_exact_match`.
- All 15 historical Phase 0 hashes and all deferred algorithmic hashes match.
- Current lane implementation hashes are recorded in the Phase 1 result.
- Git HEAD may move because another authorized lane commits unrelated work;
  this lane gates on exact declared-path hashes and stops only for overlap.
- Claude remains policy-blocked before probe. Bounded Codex substitute review is
  required and labeled weaker than Claude.

## Historical Fixture Defects Binding This Phase

- `_batched_model_tensors` loops over static `B` and stacks scalar rows.
- `make_fixture` uses `tf.linspace(-0.2,0.2,P)`, so the first 50 parameter values
  at `P=50` are not the first 50 values at `P=150`.
- `_make_parameter_batch` centers and scales offsets using the requested `B`, so
  proposal rows change when `B` changes.
- Observations are generated after derivative-bearing parameterization is
  attached to the fixture. Cross-`P` identity is therefore not explicit enough
  for a fair scaling contract even where current bases happen to be sparse.
- Existing v2 fixture-version constants describe these historical semantics and
  must change prospectively so stale Phase 1 method records cannot resume.

## Locked Replacement Fixture Contract

- For fixed `(dimension,dtype,timesteps)`, build one parameter-independent base
  model and generate one deterministic observation sequence from that base
  model only. No derivative direction or requested `P/B` may affect it.
- Define the canonical length-`P` parameter vector by index, not by endpoint
  interpolation:
  `theta[j] = -0.2 + 0.4 * j / 149` for zero-based `j`. Thus `P=50` is the exact
  prefix of `P=150`. The maximum supported scaling `P` for this program is 150;
  other values up to 150 use the same prefix rule.
- Keep the existing deterministic `_make_slots` order. Every derivative basis at
  `P<=150` is an exact prefix of the `P=150` basis in the fixed dimension.
- Define one canonical `B=16` proposal cloud using fixed row coordinates
  `linspace(-1,1,16)` and the existing sine direction pattern evaluated on the
  canonical parameter indices. Proposal identity is the integer source-row ID.
- Locked row selections are `B=1 -> [7]`, `B=4 -> [2,7,8,13]`, and
  `B=16 -> [0,1,...,15]`. These are nested as identity subsets even though array
  order/contiguity differs. Any other `B` is rejected by the scaling harness
  unless a later reviewed plan adds an explicit row map.
- Cross-`P/B` timing may be compared descriptively only with the recorded nested
  fixture identities. Because parameter dimension and batch shape change the
  computed target, method ranking remains within the same `(P,B)` cell; no
  cross-cell timing difference may be attributed solely to `P` or `B`.

## Required Artifacts And Write Set

- Surgical fixture/orchestration changes only in
  `scripts/benchmark_kalman_qr_parameter_count_scaling.py`.
- Fixture identity/version updates in
  `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py` and only
  directly required pure helpers in `scripts/kalman_qr_benchmark_contract.py`.
- New focused tests in `tests/test_kalman_qr_batched_fixture.py`; surgical
  additions to `tests/test_kalman_qr_parameter_count_scaling_harness.py` only if
  required for stale-resume/fingerprint checks.
- Graph diagnostic JSON:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase2_graphdef_2026-07-11.json`.
- Phase 2 result:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase2-batched-fixture-result-2026-07-11.md`.
- Refreshed Phase 3 subplan and its deterministic review records
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase3-subplan-codex-substitute-review-round1-2026-07-11.md`
  through round 5 only if reached.
- Temporary fixture diagnostics only under `/tmp/kalman_qr_phase2_fixture/`.

Read-only in Phase 2:

- `bayesfilter/linear/kalman_qr_derivatives_tf.py`
- `bayesfilter/linear/kalman_qr_tf.py`
- `bayesfilter/linear/qr_factor_tf.py`
- all historical `*2026-07-09*` artifacts and runner source
- Phase 1 `/tmp` smoke artifacts

## Required Checks, Tests, And Review

1. Preserve the scalar `_model_tensors` implementation as the independent
   reference. Compare all 16 tensor outputs of the new batched path against
   `tf.stack([_model_tensors(row), ...])` at `B=1,4`, `P=3,50`, and both
   float32/float64. Require exact shapes/dtypes and `assert_equal` where the
   algebra is operation-order identical; otherwise use predeclared roundoff
   tolerances `float32 rtol=atol=2e-6`, `float64 rtol=atol=2e-13` and report the
   observed maxima.
2. AST/source test the timed fixture call graph: `_batched_model_tensors` and
   directly called fixture helpers contain no Python `for`/`while`, list
   comprehension, `tf.map_fn`, `tf.vectorized_map`, or `tf.numpy_function` over
   batch rows. TensorFlow broadcast/einsum/matmul is required.
3. For each dtype at `dimension=10,T=8`, build `P=50/150` and `B=1/4/16` and
   record/check:
   - identical base-model and observation hashes across all six cells;
   - exact `P=50` parameter and derivative-basis prefixes of `P=150`;
   - exact proposal row IDs per the locked map;
   - common selected row values are exact prefixes across `P` and exact row
     matches across `B` after indexing by row ID;
   - fixture/data/source/version hashes stored in the diagnostic artifact.
4. Trace only a wrapper from `[B,P]` to all 16 fixture tensors with
   `tf.function(jit_compile=False)` at `dimension=10,P=50,B=1/4/16`. Record
   GraphDef node count, serialized byte size, static output shapes, and a
   normalized structural digest. No callable execution, `.numpy()`, XLA
   compile, GPU probe, or Kalman score is allowed in this structural diagnostic.
   The `[B,P]` input is already selected before tracing; canonical-cloud
   construction and locked row selection are untimed fixture-identity checks
   outside this wrapper and must not introduce selection constants into its graph.
5. Define the normalized GraphDef representation prospectively in the diagnostic
   code and tests. For each node preserve ordered node name, op type, ordered
   inputs, device, and every serialized attribute except the explicitly allowed
   B-dependent leading tensor-shape dimension values for the `[B,P]` input and
   tensors derived from that leading axis. No constant payload is allowed to
   vary across B. Preserve constant dtype, rank, element count, payload digest,
   consumer list, and all non-B dimensions; reject any new,
   removed, reordered, or differently connected node, any changed op/device,
   any changed non-shape attribute, and any changed constant role/consumer.
   The normalizer must itself have unit fixtures proving it detects an op,
   edge, attribute, duplicate node, constant-role/payload/element-count, dtype,
   rank, and non-B-dimension mutation.
6. Graph gate: node count and normalized structural digest must be exactly equal
   across `B=1/4/16`. Raw serialized byte counts and ratios are explanatory only
   and have no pass threshold. Any structural mismatch or B-dependent node
   growth is a veto and repair trigger, not evidence against TensorFlow batching.
7. Update the fixture contract, parameter-batch, and observation-generation
   version strings in the v2 supervisor. Test that a Phase 1 record is rejected
   after this prospective fixture-version change, and require the exact resume
   rejection reason to identify fixture/config/schedule fingerprint mismatch
   rather than accepting a generic failure.
8. Run only GPU-hidden, non-JIT tests/diagnostics:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_batched_fixture.py
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_batched_fixture.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py
git diff --check -- \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_batched_fixture.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py
```

9. Rehash all read-only algorithmic and historical paths at close. Write the
   Phase 2 result, refresh Phase 3 with actual formulas/shapes/graph evidence,
   and review Phase 3 before advancing.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does true batched TensorFlow fixture algebra reproduce stacked scalar tensors while enforcing nested `P/B` fixture identity and eliminating B-axis graph duplication? |
| Exact baseline | Phase 1 v2 harness plus historical `_batched_model_tensors`, `tf.linspace(...,P)`, and B-dependent offset generation. |
| Primary criterion | All 16 tensor parity, nested identity/hash tests, source-structure gate, fixture-version stale-resume rejection, and GraphDef gate pass. |
| Promotion vetoes | Tensor/shape/dtype mismatch; base/observation hash mismatch; broken parameter/derivative prefix; wrong proposal IDs/values; residual batch loop/map; GraphDef node growth; stale Phase 1 resume; strict artifact failure. |
| Repair triggers | Any focused fixture, nested-identity, structure, graph, or fingerprint test failure within the write set. |
| Explanatory only | Exact GraphDef bytes, trace duration, and observed roundoff maxima. |
| Not concluded | Analytical score correctness, autodiff correctness, XLA viability, warm runtime, GPU readiness, method ranking, HMC/posterior/default/production/scientific validity. |

## Skeptical Pre-Execution Audit

- Baseline is stacked scalar fixture output, not a score or historical timing.
- Graph size is a structure gate only; smaller GraphDef cannot promote runtime.
- Locked parameter indices and proposal row IDs prevent regenerated cross-cell
  targets from masquerading as scaling evidence.
- Observations come from the base model, preventing derivative count from
  changing the data.
- Exact normalized structure and node equality answer the B-unrolling question
  without promoting raw GraphDef bytes or executing XLA.
- Version strings force old Phase 1 artifacts to fail resume after semantics
  change.
- CPU diagnostics hide GPU before TensorFlow import and remain non-JIT.
- A fixture mismatch is a common-harness continuation veto until repaired; a
  graph byte difference inside the declared envelope is explanatory only.

Audit status: `PASSED_AFTER_CODEX_SUBSTITUTE_REVIEW_ROUND3`.

## Forbidden Claims And Actions

- Do not edit Kalman/QR algorithmic or analytical derivative helpers.
- Do not repair the batched-autodiff tape/reduction bug.
- Do not change measurement/timing boundaries.
- Do not run XLA, CUDA/GPU detection, a score benchmark, or any comparison grid.
- Do not use NumPy in the implementation path; NumPy is allowed only in tests or
  reporting/reference inspection under repository policy.
- Do not delete/overwrite historical artifacts or unrelated dirty work.
- Do not call graph-size reduction speed evidence or cross-cell method ranking.

## Exact Next-Phase Handoff Conditions

All conditions are conjunctive:

- All 16 tensor parity checks pass at every declared dtype/P/B case.
- Nested parameter/basis/proposal/base/observation checks and hashes pass.
- Source-structure, normalizer mutation tests, and exact normalized GraphDef
  gates pass.
- Fixture version change rejects Phase 1 resume artifacts.
- Strict diagnostic JSON parses and contains exact command/environment/source/
  fixture provenance and nonclaims.
- `py_compile`, focused pytest, scoped `git diff --check`, closing read-only
  hashes, and no-worker check pass.
- Phase 2 result maps each fixture defect to code/test evidence.
- Refreshed Phase 3 subplan receives exact `VERDICT: AGREE`.

## Stop Conditions

- Tensor algebra cannot reproduce scalar fixture semantics within declared
  tolerances without changing the model target.
- Nested `P/B` identity cannot be enforced under the locked replacement contract.
- Static shape contracts required later by XLA cannot be retained.
- An in-scope focused check remains broken after the implementation repair loop.
- Another lane changes a declared Phase 2/read-only path and the overlap cannot
  be reconciled safely.
- New package/network/model-file/product/scientific authority is required.
- The same material review blocker fails to converge after five rounds.

An ordinary fixture implementation bug or graph-gate failure is a repair
trigger, not an automatic stop. Localize, patch within the write set, and rerun
the smallest discriminating check.

## Mandatory Phase-End Sequence

1. Run every required local check.
2. Write the Phase 2 result/close record.
3. Refresh the Phase 3 subplan from actual evidence.
4. Review Phase 3 and repair/recheck it before advancing.
