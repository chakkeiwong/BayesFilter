# Phase 9 Subplan: Trusted GPU/XLA Score-Memory Ladder

Date: 2026-07-10

Status: `GATE_A_IMPLEMENTATION_APPROVED_GPU_EXECUTION_SEPARATELY_BLOCKED`

## Phase Objective

Build a production-policy-compliant evidence harness for compact nonlinear LEDH
scores, then run a gated trusted GPU/XLA ladder that measures per-seed score
memory and checks same-scalar finite differences without reviving historical
routes or batching full-row seeds in one component call.

Phase 9 does not start with a full-row run. It starts with a harness gate
because the current nonlinear score CLIs do not record or enforce XLA JIT,
managed-session GPU trust, device placement, reset memory statistics, terminal
progress artifacts, or score-only/FD-only separation.

## Research Intent Ledger

| Field | Intent |
| --- | --- |
| Main question | For each current LEDH row, can the compact same-scalar score execute under the default trusted GPU/XLA/TF32 policy at the admitted row shape within the `14000 MiB` per-seed score budget and pass all-coordinate same-scalar FD? |
| Candidate/mechanism | Compact forward-sensitivity, no-time-history, no-autodiff score route with sequential fixed-seed execution. |
| Exact comparator | Central finite differences of the row-matched value-only scalar at the same fixed seed, target, transport policy, precision declaration, and parameter coordinates. |
| Expected failure mode | XLA compile incompatibility, nonfinite score/value, missing terminal artifact, wrong device/provenance, per-seed peak above budget, or score/FD mismatch. |
| Promotion criterion | All five admitted seeds have trusted GPU/XLA compact score shards, value-only FD shards, finite outputs, row-matched metadata, max per-seed score peak at or below `14000 MiB`, and both per-seed and aggregate all-coordinate FD pass under the frozen thresholds below. |
| Promotion veto | Historical/manual provenance, CPU or non-XLA evidence, non-production precision, wrong target/shape/seeds, missing reset score-memory peak, nonfinite output, FD failure, or incomplete seed set. |
| Continuation veto | Harness cannot emit a terminal artifact; tiny XLA compact kernel cannot compile/execute after a bounded local repair; device/trust provenance is false; artifact corruption; or a prefix at the same particle count exceeds the memory budget. |
| Repair trigger | Compile failure with a specific unsupported operation, missing evidence field, aggregation mismatch, or prefix-only numerical mismatch. |
| Explanatory diagnostics | Compile time, wall time, value peak, score peak below the hard budget, per-coordinate errors below a pass threshold, Sinkhorn residuals, and process-level GPU reservation. |
| Must not be concluded | Runtime superiority, posterior correctness, HMC readiness, exact nonlinear likelihood correctness, native actual-SV correctness for KSC, scientific superiority, or default-policy promotion beyond the existing owner directive. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering/scientific question | Can each compact score route produce valid trusted full-row GPU/XLA memory and same-scalar correctness evidence? |
| Baseline/comparator | Admitted row-matched forward artifact; compact route tested in Phase 8; LGSSM one-seed score-only artifact for harness shape only. Historical fixed-SIR manual-VJP memory is explicitly not a compact baseline. |
| Primary criterion | Row-matched full seed set, compact provenance, `float32`, TF32 enabled, `jit_compile=True`, managed-session trusted GPU device evidence, finite score and value, reset per-seed score peak `<=14000 MiB`, and aggregate all-coordinate FD pass. |
| Promotion vetoes | Any missing primary field; any historical route; any non-XLA/non-GPU execution represented as production evidence; any memory budget failure; any FD failure; any target or coordinate mismatch. |
| Continuation vetoes | Invalid harness/artifact; tiny XLA failure that cannot be locally repaired; prefix OOM/budget failure at the intended particle count; corrupt or incomplete seed shard. A candidate row failure does not stop unrelated rows unless it invalidates shared harness logic. |
| Explanatory only | Runtime, compile time, sub-budget peak differences, external `nvidia-smi`, prefix FD errors, and value/reference deltas. |
| Not concluded | Claims listed in the research intent ledger. |
| Artifact | Per-rung structured JSON/logs, row aggregate JSON/Markdown, and Phase 9 result. |

## Pre-Execution Skeptical Audit

Gate A implementation resumed only after checking the plan against stale
context, baseline, proxy-metric, stop-condition, environment, and artifact
risks. The admitted forward artifacts remain target/shape comparators rather
than score baselines; the historical fixed-SIR manual-VJP and one-seed LGSSM
artifacts remain non-admission evidence. CPU-hidden tests answer only harness
and artifact-contract correctness, not GPU execution, memory admission, or
finite-difference admission. The reviewed continuation and repair rules remain
binding, and no GPU command is allowed until the Gate A result plus exact
command manifest receives a fresh bounded substitute-review `VERDICT: AGREE`.

The audit found one material artifact-identity gap: the initial Gate A
validator did not bind transport plan mode, AD mode, gradient mode, annealing
scale, or annealing convergence threshold. Gate A verification was held while
those fields and adversarial tests were added. With that repair, each planned
CPU command produces evidence for the stated Gate A question without silently
changing the comparator, target, promotion criterion, or continuation veto.

## Baseline Discipline

- LGSSM has one valid trusted `N=10000,T=50`, seed `81120`, score-only
  compact artifact with `jit_compile=true` and a TensorFlow reset-memory peak
  of about `719.671 MiB`. It is not admitted because same-scalar FD and the
  other four seeds are absent.
- The July 6 fixed-SIR `3166.769 MiB` artifact used
  `manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot`.
  That historical route is diagnostic-only and must not be used as a compact
  comparator or admission input.
- Admitted forward artifacts are target/shape comparators, not score evidence.
  Their full compile/first-call times show the expected scale: predator-prey
  about `34s`; actual-SV about `1105s`; generalized-SV about `1188s`; KSC-SV
  about `1055s`.
- No cross-model runtime or memory ranking is allowed. Each row is judged
  against the same hard evidence fields and its own admitted target.

## Frozen Finite-Difference Criteria

These values are the current production-precision CLI policies and are frozen
before any Phase 9 GPU result. Pass uses the existing rule
`max_abs_error <= atol OR max_relative_error <= rtol`. Every singleton seed
must pass, and the arithmetic-mean aggregate score versus aggregate FD must
also pass. A failure cannot be repaired by changing step or tolerance after
seeing the result; any alternative precision/step arm requires a revised,
reviewed plan and remains diagnostic until then.

| Row | FD step | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: | ---: |
| LGSSM | `1.0e-3` | `5.0e-3` | `5.0e-3` |
| fixed-SIR | `1.0e-3` | `1.0e-2` | `5.0e-2` |
| predator-prey | `1.0e-4` | `5.0e-3` | `5.0e-3` |
| actual-SV | `1.0e-4` | `5.0e-3` | `5.0e-3` |
| generalized-SV | `1.0e-4` | `5.0e-3` | `5.0e-3` |
| KSC-SV | `1.0e-4` | `5.0e-3` | `5.0e-3` |

## Gate A: Evidence Harness Compliance

Before any Phase 9 GPU command, implement or extend a shared nonlinear score
harness so every nonlinear row provides:

- a score kernel invoked through `tf.function(..., jit_compile=True)` by
  default;
- no production `jit_compile=False` fallback;
- `--device-scope`, `--cuda-visible-devices`, `--device`, and
  `--expect-device-kind` controls consistent with existing value runners;
- `GPU_TRUST_BASIS =
  "owner_designated_managed_session_visible_gpu_trusted"` in trusted artifacts;
- physical/logical GPU, output-device, dtype, TF32, XLA, TensorFlow, host, git,
  seed, target, and command provenance;
- atomic started/initialized/completed/failed terminal JSON artifacts;
- separate `score-only` and `fd-only` execution so FD never recomputes the
  score kernel;
- TensorFlow score-memory reset immediately before the measured score call and
  `score_gpu_memory_info_before/after` fields;
- exactly one seed per raw runtime shard;
- a read-only aggregation path that validates all five fixed seeds, shape,
  target, parameter order, compact route, precision, trust, XLA, device,
  finite outputs, FD metadata, and per-seed memory before constructing an
  admitted score artifact;
- fixed-SIR as a real executable row in the shared harness rather than only an
  import-time normalizer;
- tests modeled on the existing LGSSM shard aggregation checks, including
  duplicate/missing seed, forged trust/JIT/device, historical route, wrong
  target/shape/precision, missing peak, and FD-failure rejection.

Gate A passes only after CPU-hidden compile/contract tests and a bounded local
substitute review agree. CPU-hidden tests may validate artifact logic and XLA
default wiring but cannot claim GPU execution.

### Gate A implementation scope

This reviewed subplan may authorize only the following implementation work:

- add one shared runner at
  `docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py`;
- add tensor-only compact score/value adapter entry points to the five
  nonlinear score modules only where required for XLA compilation;
- add focused contract tests at
  `tests/highdim/test_ledh_compact_score_gpu_xla_harness.py` and update existing
  model-specific tests only for the new entry points;
- reuse the existing admitted value builders/cores and compact score math;
- add artifact validation/aggregation helpers without changing the public
  BayesFilter API or admission thresholds.

Gate A must not change target density math, parameter transformations,
transport policy, compact derivative equations, row tolerances, shared score
admission semantics, or defaults outside this evidence harness. If tensor-only
extraction requires a mathematical change rather than moving existing
operations behind tensor inputs, stop and revise the plan.

### Gate A required CPU-hidden checks

After implementation, run:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py \
  tests/highdim/test_ledh_compact_score_gpu_xla_harness.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_compact_score_gpu_xla_harness.py \
  tests/highdim/test_ledh_score_wiring_phase8_cross_model.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Then rerun affected model-specific shards. Write
`docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-a-harness-result-2026-07-10.md`
and a concrete GPU execution manifest containing exact commands, paths,
timeouts, expected fields, and per-rung stop decisions.

No Gate B/C/D command is authorized merely because Gate A code/tests pass.
The Gate A result and exact GPU execution manifest require a fresh bounded
review with `VERDICT: AGREE` before the first GPU/CUDA command.

## Gate B: Trusted GPU/XLA Preflight

Run escalated/trusted `nvidia-smi` and a TensorFlow device probe first. Record
the managed-session trust basis. Then run one-seed tiny XLA score-only and
FD-only preflights for each nonlinear row.

Suggested initial shape per row: the existing tiny fixture shape already used
by its model-specific contract test. A preflight passes only if:

- `jit_compile=true` and no non-JIT fallback ran;
- compact route and singleton seed are recorded;
- score/value tensors are finite and on GPU;
- production `float32` plus TF32 is recorded;
- reset score peak is present;
- FD-only consumes the saved score shard and does not call the score kernel;
- a terminal JSON artifact is emitted on success or failure.

An XLA compile failure is a repair trigger. Preserve its exact error and make
the smallest kernel/harness repair. Do not silently run eager/non-JIT and do not
continue that row to a larger rung until the tiny XLA gate passes.

Exact Gate B/C/D commands are intentionally deferred until the harness CLI
exists and `--help` plus CPU-hidden parser tests establish its real option
names. Invented commands in this pre-implementation plan would not be
executable evidence. The post-Gate-A execution manifest is mandatory and must
freeze the exact commands before any GPU result is observed.

## Gate C: One-Seed Prefix Ladder

Use seed `81120`, `N=10000`, row production transport settings, and increasing
time prefixes. Prefixes nominate feasibility only; they cannot admit a score.

| Row | Prefix rungs | Full time |
| --- | --- | ---: |
| LGSSM | existing `T=50` score-only baseline; add FD-only for seed `81120` | 50 |
| fixed-SIR | `T=1`, `T=5`, `T=20` | 20 |
| predator-prey | `T=1`, `T=5`, `T=20` | 20 |
| actual-SV | `T=4`, `T=50`, `T=250`, `T=1000` | 1000 |
| generalized-SV | `T=4`, `T=50`, `T=252`, `T=1008` | 1008 |
| KSC-SV | `T=4`, `T=50`, `T=250`, `T=1000` | 1000 |

Run score-only first at each rung. Run FD-only only after the score shard is
finite, trusted, and below budget. Stop the row before the next rung if a score
prefix at `N=10000` exceeds `14000 MiB`, is nonfinite, lacks a terminal
artifact, or fails XLA/device/trust checks. A prefix FD mismatch triggers
diagnosis; it is not evidence that a later prefix will repair correctness.

The long-SV full-time rung may take materially longer than the admitted value
compile times because it includes the compact sensitivity and multiple
value-only perturbations. Keep score and each FD perturbation in separate
visible processes/artifacts where needed so memory is measured per computation
and failures are recoverable.

## Gate D: Full Fixed-Seed Evidence

Only after seed `81120` passes the full-time rung may seeds `81121` through
`81124` run at the exact admitted shape. Execute them as separate trusted
processes, never as one five-seed kernel call.

Aggregate only after all five seed score and FD artifacts exist. The aggregate
must use the arithmetic mean over the fixed seeds, disclose segmented
execution, use the maximum per-seed reset score peak for the memory gate, and
state that it is not monolithic batch memory or runtime evidence.

If a row fails, classify it directly:

- harness/implementation failure;
- XLA compile failure;
- numerical validity failure;
- memory-budget failure;
- finite-difference correctness failure;
- incomplete evidence;
- or passed the full score screen.

Candidate rejection must not be represented as rejection of the compact-score
research direction or as evidence against unrelated rows.

## Required Run Manifest

Every serious artifact must record:

- git commit plus dirty-worktree disclosure;
- exact command and runner path;
- Python, TensorFlow, and environment/conda identity;
- host, physical/logical devices, output devices, and CUDA visibility;
- `gpu_trust_basis`;
- `jit_compile`, dtype, TF32 state, and device expectation;
- row id, target policy, theta coordinates/order, source value artifact;
- `N`, `T`, seed, transport mode, iterations, epsilon, and chunk sizes;
- score-only or FD-only stage;
- reset memory before/after and peak source;
- compile/first-call, elapsed time, terminal status, and artifact paths;
- plan and result paths.

## Pre-Mortem

| Risk | How the run could mislead | Cheap discriminator |
| --- | --- | --- |
| Eager score labeled production | Finite GPU output appears valid but violates XLA default policy. | Require `jit_compile=true` in code, artifact, and adversarial validator test. |
| Process reservation confused with score peak | External GPU reservation overstates or understates measured score memory. | Reset TensorFlow memory stats immediately before score call; preserve external status as explanatory only. |
| Five seeds multiply peak | A batched kernel passes numerically but OOMs or changes the memory claim. | Raw shard validator requires exactly one seed; aggregate offline. |
| FD recomputes score | Correctness stage hides score memory/time and can pass through the wrong route. | Separate process/stage and monkeypatch test forbidding score call in FD-only. |
| Prefix pass promoted | Short `T` result is treated as full-row evidence. | Artifact status remains prefix/tiny and full validator checks exact admitted time. |
| Long compile mistaken for hang | Full SV XLA compile is killed despite prior value compiles taking 18-20 minutes. | Atomic progress artifact plus bounded log polling and row-specific timeout derived from baseline. |
| Target substitution | KSC, actual-SV, or generalized-SV shares implementation pieces and silently changes likelihood semantics. | Aggregate validator matches admitted target policy and parameter order exactly. |
| Descriptive metric ranked | Lower memory/runtime below the budget is called superior. | Report only pass/fail and descriptive values; no ranking without uncertainty design. |

## Stop Conditions

- Gate A review returns `VERDICT: REVISE` with an unresolved material issue.
- A trusted GPU preflight cannot establish XLA/device/trust provenance.
- The tiny XLA compact kernel cannot be repaired within five bounded review
  rounds without changing the admitted target or score math.
- A prefix at `N=10000` exceeds `14000 MiB`, emits nonfinite output, or lacks a
  valid terminal artifact.
- FD correctness fails and the mismatch cannot be localized without changing
  the declared comparator or tolerance after seeing results.
- Required seed/shape/target/provenance evidence is missing or corrupt.
- A run would require a package install, external fetch, destructive git
  action, or unreviewed default/public API change.

## Skeptical Plan Audit

- Wrong baseline: corrected. Historical fixed-SIR manual score memory is not a
  compact comparator; row-matched admitted values and compact prefix shards are
  the baselines.
- Proxy promotion: blocked. Tiny and prefix results only nominate the next
  rung; score-only evidence cannot admit without FD.
- Missing stop conditions: corrected with explicit harness, XLA, artifact,
  memory, finite-output, and FD continuation vetoes.
- Unfair comparison: blocked. No cross-model ranking; every row uses its own
  target, parameter coordinates, and admitted shape.
- Hidden assumption: corrected. Sequential seed aggregation is disclosed and
  is not a monolithic batch memory/runtime claim.
- Stale context: corrected. Phase 8 post-repair test results and the June 26
  XLA policy are binding; older manual-route artifacts are diagnostic only.
- Environment mismatch: blocked. All GPU/CUDA/XLA commands require trusted or
  escalated execution and managed-session provenance.
- Artifact sufficiency: corrected. Atomic per-stage artifacts plus offline
  aggregation answer compilation, device, memory, correctness, and admission
  separately.

Audit result: the plan is suitable for bounded review of Gate A implementation
only. No Phase 9 code edit is allowed until that review agrees. GPU commands
remain separately blocked until the implemented harness, Gate A result, and
exact execution manifest receive a later `VERDICT: AGREE`.
