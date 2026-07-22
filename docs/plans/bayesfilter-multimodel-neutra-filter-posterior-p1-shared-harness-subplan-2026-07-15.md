# P1 Subplan: Shared Multi-Model NeuTra Campaign Harness

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `P1_COMPLETE_REVIEWED_REOPENED_STATUS_IDENTITY_REPAIR_CLOSED`

## Phase Objective

Build and admit one generic campaign control plane that can issue a typed
posterior identity only after a complete target passes independent
recomposition, then bind that identity to batched value/status and value/score
interfaces, GPU/XLA NeuTra training, frozen transport loading, shared sequential
NeuTra HMC, manifests, and cell-state transitions without target substitution.

## Inherited Entry Conditions

- P0 attempt 04 is complete and validated at
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p0/attempt-04-20260715T1658/`.
- P0 issued zero posterior signatures and classified all eleven model cells
  `TARGET_BLOCKED`. Their inventory `scope_identity` values are explicitly
  ineligible as HMC, training, or transport target signatures.
- P1 may issue a typed target identity only for its repository-owned complete
  synthetic canary. Missing model-cell contracts remain explicit and P1 must
  not fabricate or advance them.
- Shared LGSSM HMC and training utilities are usable foundations, not assumed
  universal recipes.

## Cell Scope

All cells at the registry/state-guard level. P1 uses one exact lightweight
synthetic Gaussian posterior canary and negative identity fixtures; it does not
run any model-cell HMC/training or promote any P2-P6 cell.

## Required Artifacts

- Generic registry loader and repository-owned typed target-identity issuer. The
  issued digest binds the mathematical target contract, dtype, exact inspected
  batch adapter and dependency closure, recomposition result, and cell/canary
  scope. A bare caller string or P0 scope identity is never enough.
- Posterior-identity dossier/recomposition interface that is independent of the
  production final target assembler and checks total unconstrained value/score.
- Campaign manifest/result schemas and append-only cell-state transition helper.
- Guarded training, transport-loading, plain-HMC, and NeuTra-HMC entry points
  requiring the same issued identity and adapter binding.
- GPU training launcher that requires batching, XLA, memory growth, target
  binding, fresh output roots, and serious-run provenance.
- CPU sample-generation helper with deterministic worker/seed partitioning.
- Static/graph guards for NumPy, host callbacks, eager host conversion, and
  Python sample-axis loops in active training/target paths.
- Unit and integration tests, tiny CPU reference artifacts, trusted GPU/XLA
  canary, P1 result/run manifest, updated ledgers, and refreshed P2 subplan.

## Required Checks And Reviews

1. Target/signature positive and negative tests: changed prior, observations,
   filter setting, transform, dtype, or dependency hash must change identity;
   caller-stamped and cross-cell artifacts fail closed.
2. Posterior recomposition checks: prior plus filter likelihood plus full
   unconstraining Jacobian agrees in total value and total score; deliberately
   omitted Jacobian/prior, wrong observations/filter/dtype/chart, and final-
   assembler reuse fail the dossier gate.
3. Batch checks: singleton and multi-batch parity, batch permutation, nonfinite
   status propagation, fixed shapes, deterministic stateless seeds, and no
   scalar-loop fallback.
4. Shared controller checks: retained warm-up separation, recent-window modern
   R-hat, cumulative retained diagnostics, 10,000 caps, minimum four chains,
   distinct tuning/warm-up/retained seeds, and archive callbacks.
5. Training checks: graph-native reverse-KL step, capacity/config serialization,
   target/frozen-artifact binding, checkpoint replay, heldout quarantine, and
   fail-closed GPU/XLA/memory-growth policy. State-transition tests distinguish
   `RECIPE_REJECTED` from `CELL_CANDIDATE_REJECTED` and require complete
   tried/selected/rejected/untried candidate ledgers.
6. CPU generation checks: worker-count invariance under domain-separated seeds,
   deterministic ordering, hash-stable data, and batched payloads.
7. Trusted GPU canary under escalated permissions trains only the synthetic
   Gaussian canary and records device, framework,
   XLA, TF32, dtype, memory growth, compile time, run time, and peak memory.
8. Discovery-complete source audit over active campaign paths for NumPy and
   Python sample-axis loops. Diagnostic/reporting exceptions are classified.
9. Focused pytest, compile/import checks, JSON schema validation, artifact
   replay, and scoped `git diff --check`.

## Evidence Contract

| Field | P1 contract |
| --- | --- |
| Question | Can one fail-closed harness execute registered targets without identity drift, hidden host execution, or artifact loss? |
| Baseline | Existing shared sequential HMC/training components and a complete analytic Gaussian canary |
| Primary pass | All signature, posterior-recomposition, candidate-state, batching, archive, CPU-generation, and trusted GPU/XLA canary tests pass |
| Vetoes | Scalar fallback; target substitution; failed or implementation-circular posterior recomposition; generic recipe failure mislabeled cell rejection; cross-cell artifact load; warm-up pooling; wrong modern R-hat; no memory growth; NumPy/host callback/sample loop in active path; missing manifest fields |
| Explanatory only | Canary loss, acceptance, runtime, and throughput |
| Not concluded | Any nonlinear cell works, training recipe adequacy, posterior correctness, filter accuracy, or production readiness |

## Default And Assumption Audit

P1 must not hard-code LGSSM dimensions, affine charts, topology, learning rate,
filter status schema, or comparator estimands. The shared defaults permitted are
execution invariants only: TensorFlow/TFP, batched inputs, XLA on, GPU training,
memory growth, fresh roots, signature binding, at least four HMC chains, retained
warm-up, modern diagnostics, and 10,000 caps. All scientific and optimization
choices stay cell-owned.

## Repair Triggers

Localized compile, device, multiprocessing, serialization, schema, or test
failures are contract-preserving repair triggers. Repair in a fresh attempt,
run the smallest failing regression plus the harness integrity suite, then
replay the trusted canary if GPU graph/source changed. A canary candidate metric
failure is irrelevant unless it exposes an interface or numerical invalidity.

## Forbidden Claims And Actions

- No P2-P6 cell state beyond `TARGET_FROZEN` changes in P1.
- No scalar Python wrapper presented as batched execution.
- No NumPy in active training/target generation and no CPU serious training.
- No universal architecture, optimizer, or HMC tuning default inferred from the
  canary or LGSSM.
- No package/environment mutation without separate authority.

## Handoff Conditions

P2 begins when P1 tests and trusted GPU canary pass, target-negative tests fail
closed, manifest replay succeeds, no active-path batching policy violation
remains, the P1 result and run manifest are complete, and P2 is refreshed with
the actual exact-SV signatures, routes, commands, margins, and remaining budget.

## Stop Conditions

Stop program-wide if the shared harness cannot preserve target identity or
archives after three repairs, or completed LGSSM evidence is shown to be
contaminated. Stop P1 locally for unavailable trusted GPU after escalated device
and framework probes, irreconcilable environment mismatch, or exhaustion of 16
CPU-hours plus 2 GPU-hours. Cell-specific missing adapters do not stop P1.

## Compute And Attempt Budget

At most 16 CPU wall-hours and 2 trusted GPU wall-hours. The GPU bucket covers the
initial common canary and common harness/schema/serialization/reporting repairs,
including a later P1 reopen triggered by P2-P6; it never charges a cell. Three
repair attempts apply per materially identical harness defect. GPU canaries are
tiny and never claim-bearing for nonlinear cells. Exhausting this bucket before
common repair fires the shared-harness budget continuation veto.

## Skeptical Pre-Execution Audit

The full skeptical audit is in
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p1-skeptical-audit-2026-07-15.md`.
It passed after removing the false assumption that P0 had frozen any model
target, separating the mathematical SSM signature from the campaign-issued
execution identity, and restricting runtime evidence to the synthetic canary.

## Exact Commands And Output Roots

CPU-hidden focused checks:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_neutra_campaign.py tests/test_multimodel_neutra_p1_canary.py tests/test_neutra_batching.py tests/test_neutra_training.py tests/test_neutra_hmc.py tests/test_tensorflow_gpu_memory_policy.py
```

Trusted GPU/XLA canary:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true python docs/benchmarks/run_multimodel_neutra_p1_canary.py --output-root docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p1/attempt-01-<UTC_TIMESTAMP>
```

The launcher must reject an existing output root. The canary budget is at most
64 training steps, batch size 64, one four-chain 16-draw HMC health smoke, and
two GPU wall-hours including repairs. Canary loss, acceptance, and runtime are
explanatory only.
