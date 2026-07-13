# Phase 1 Subplan: Harness Failure Isolation And Artifact Integrity

Date: 2026-07-11
Status: `PHASE_CLOSED_HANDOFF_PASSED`

## Phase Objective

Make each benchmark method independently selectable and executable in one
method/case subprocess, make failure stages and terminal status unambiguous,
emit strict JSON, and make resume decisions fail closed on schema, source,
configuration, and result integrity.

This phase repairs evidence plumbing only. It does not vectorize fixture or QR
math, repair the true-batched autodiff formula, separate promoted timing
boundaries, or launch a comparison grid.

## Entry Conditions Inherited From Phase 0

- Phase 0 inventory/result exist and all local Phase 0 gates passed.
- Git HEAD is `52ee244498988e046a6356f926003b581103083b`.
- The seven-path opening/closing source fingerprint in the Phase 0 inventory is
  stable.
- No Kalman QR benchmark worker was active at Phase 0 close.
- Nine CPU and two GPU historical JSON artifacts are all
  `historical_debug_only_nonresumable_under_repaired_schema`.
- Claude review remains policy-blocked before probe. A fresh bounded Codex
  substitute review is required and must be labeled weaker than Claude.
- The dirty worktree and all unrelated changes must be preserved.

## Phase 0 Facts Binding This Phase

- `benchmark_case` executes three methods in one process, so failure evidence is
  coupled.
- The isolated-grid parent writes `run_status=complete` even when rows contain
  errors.
- The old runner resumes solely from `summary.run_status == "complete"` and can
  return success despite GPU-preflight failure.
- JSON writes do not set `allow_nan=False`.
- The diagnostic batched-autodiff tape-scope bug belongs to Phase 4; Phase 1
  may isolate that method but must not repair or promote it.
- Host materialization inside timing belongs to Phase 5; Phase 1 may label the
  current boundary but must not claim repaired timing.

## Required Artifacts And Write Set

- `scripts/kalman_qr_benchmark_contract.py`: pure standard-library schema,
  fingerprint, strict-JSON, terminal-status, and resume-decision helpers.
- Surgical orchestration changes to
  `scripts/benchmark_kalman_qr_parameter_count_scaling.py`.
- New runner
  `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py`.
- Focused tests in `tests/test_kalman_qr_benchmark_contract.py` and, only if
  needed for method-selection integration,
  `tests/test_kalman_qr_parameter_count_scaling_harness.py`.
- Phase result:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase1-harness-integrity-result-2026-07-11.md`.
- Refreshed Phase 2 subplan:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase2-batched-fixture-subplan-2026-07-11.md`.
- Phase 2 review records under deterministic paths
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase2-subplan-codex-substitute-review-round1-2026-07-11.md`
  through `round5` only if reached.
- Temporary integration artifacts may be written only below
  `/tmp/kalman_qr_phase1_harness/`.

The following historical paths are read-only in Phase 1:

- `docs/benchmarks/run_kalman_qr_core_batch_grid_overnight_2026_07_09.py`
- every `docs/benchmarks/*2026-07-09.json`
- every `docs/benchmarks/*2026-07-09.md`
- historical logs under `docs/benchmarks/logs/`

## Schema And Status Contract

- New artifacts use schema `bayesfilter.kalman_qr_batched_xla_repair.v2`.
- Stable method identifiers are
  `batch_native_analytical_qr_score`, `scalar_analytical_row_loop`,
  `autodiff_row_loop_qr_score`, and `batched_static_autodiff_probe`.
- A scheduled unit is exactly one `(case_id, method_id)` subprocess. No child
  may execute an unselected benchmark method. Fixture setup shared by the
  selected method is allowed inside that child. Dispatch uses a closed builder
  registry plus a selection guard; each child record contains
  `selected_method_id` and an `invoked_method_ids` ledger that must equal the
  singleton selected method.
- Before any child launch, the supervisor atomically persists a canonical
  `schedule_manifest` containing the ordered expected `(case_id, method_id)`
  identities, declared aggregate checks, schema, and all five fingerprints.
  Its `schedule_fingerprint` is computed independently of returned child rows.
  Completeness is always validated against this manifest, never inferred from
  observed results.
- Stage vocabulary is exactly `fixture`, `trace`, `compile_first_execution`,
  `warm_execution`, `materialization`, `parity`, and `artifact_write`.
  Each method record stores the last entered stage, terminal stage, elapsed
  stage data when available, return code/timeout/signal, and bounded error tail.
- Every launch receives a cryptographically random `attempt_id` and a new
  attempt-specific progress-journal path created with exclusive-create
  semantics before the child starts. Existing or nonempty attempt paths fail
  closed and are never appended. Every event binds `attempt_id`, `case_id`,
  `method_id`, all five fingerprints, and `resume_key`. Before entering each
  stage, the child appends one strict-JSON line to that fresh journal, flushes
  it, and calls `os.fsync`. Only newline-terminated records that pass strict
  decoding and exactly match the supervisor's launch identity count. On normal completion the child
  atomically writes its result. On timeout, signal, or hard exit, the supervisor
  synthesizes the terminal method record from process status plus the last
  fully flushed journal event; no child result is required for stage recovery.
  A missing first event for the current attempt is a structurally recorded
  `crashed` result with `last_entered_stage=null`, not an inherited or invented
  stage. Journals from prior attempts are immutable evidence and never inputs
  to a later attempt's stage recovery.
- Method terminal states are `passed`, `failed`, `timed_out`, `crashed`, and
  `interrupted`. `pending` and `running` are nonterminal.
- Top-level `complete` means every expected method/case identity occurs exactly
  once, every method state is `passed`, mandatory aggregate checks pass, and
  the artifact is strict/schema-valid.
- Top-level `complete_with_failures` means every expected identity occurs
  exactly once in a terminal state and the artifact is structurally valid, but
  at least one method or aggregate gate failed.
- Top-level `failed` means the harness cannot provide a structurally complete,
  trustworthy schedule record: corrupt/missing/duplicate child result,
  orchestrator exception, artifact-write failure, or source/config drift during
  the run. `interrupted` remains a separate nonterminal top-level state.
- A top-level success exit requires `complete`. `complete_with_failures`,
  `failed`, and `interrupted` return nonzero, including GPU-preflight failure.
- The schedule manifest's `mandatory_aggregate_checks` is a closed subset of
  `identity_integrity`, `record_integrity`, `finite_output_metadata`,
  `expected_dtype_shape`, and `comparator_parity`. Synthetic Phase 1 schedules
  may declare only the first two and must label themselves
  `harness_contract_test_only`. Any numerical/comparison schedule must declare
  all five. `comparator_parity` is computed only by the aggregate parent after
  every required output-bearing sibling exists; a lone method child cannot
  assert aggregate parity.
- Schedule, progress, child-result, aggregate, and resume-cache writes use a
  sibling temporary file, file flush plus `os.fsync`, and `os.replace` where the
  artifact is replaceable. Progress journals are append-only with flushed,
  `fsync`ed newline-delimited records. A crash must not truncate or overwrite a
  previously reusable sibling method record. A partial final line ends valid
  recovery for that attempt at its preceding complete line; retries always use
  a different exclusive journal, so no append occurs after the partial line.

## Fingerprint And Resume Contract

- `source_fingerprint` is SHA-256 over a sorted manifest of this closed
  repo-relative execution set and each file's SHA-256:
  `bayesfilter/__init__.py`, `bayesfilter/linear/__init__.py`,
  `bayesfilter/diagnostics.py`, `bayesfilter/results_tf.py`,
  `bayesfilter/structural.py`, `bayesfilter/linear/dtypes_tf.py`,
  `bayesfilter/linear/types_tf.py`, `bayesfilter/linear/qr_factor_tf.py`,
  `bayesfilter/linear/kalman_qr_tf.py`,
  `bayesfilter/linear/kalman_qr_derivatives_tf.py`,
  `scripts/kalman_qr_benchmark_contract.py`, and
  `scripts/benchmark_kalman_qr_parameter_count_scaling.py`. The new runner is
  additionally required when it is the supervisor. Missing paths fail closed.
  Git HEAD and full dirty status are recorded separately for provenance;
  unrelated documentation dirt does not silently change the execution digest.
- `config_fingerprint` is SHA-256 over canonical strict JSON containing method,
  dimension, parameter count, timesteps, batch size, dtype, device, JIT, CPU
  thread request, repeats, subprocess timeout, XLA flags, TF32 setting, jitter,
  jitter-update policy, fixture-contract version, materialization/timing-boundary
  version, and method-specific option map. Unknown method-specific options are
  rejected; output paths and timestamps are excluded.
- `runtime_fingerprint` is SHA-256 over canonical strict JSON containing the
  interpreter real path, Python implementation/version, platform, and exact
  distribution names/versions providing TensorFlow, TensorFlow Probability,
  and NumPy. Missing metadata fails closed.
- `fixture_fingerprint` is SHA-256 over canonical strict JSON containing the
  fixture-contract version, deterministic/random declaration, seed (`null` for
  the current deterministic fixture), dimension, parameter count, timesteps,
  batch size, dtype, parameter-batch construction version, observation-
  generation version, and explicit external input hashes (empty for the current
  generated fixture). Later phases may add tensor hashes prospectively without
  weakening this identity.
- `schedule_fingerprint` binds the canonical ordered schedule identities and
  mandatory aggregate checks.
- `resume_key` binds schema plus source, config, runtime, fixture, and schedule
  fingerprints for the exact method/case identity.
- A method/case record is reusable only when schema, case identity, method
  identity, all five fingerprints, and resume key match; its terminal state is
  `passed`; required output metadata exists; outputs were finite; invocation
  ledger is the expected singleton; and no integrity field is missing. Source,
  config, runtime, fixture, and schedule fingerprints are recomputed immediately
  before launch/resume and after the schedule finishes; drift makes the
  top-level artifact `failed` and prevents reuse.
- Historical v1 artifacts, failed/timed-out/crashed/interrupted/running records,
  malformed JSON, duplicate identities, and any fingerprint mismatch are never
  reusable. Rejection records an exact reason.
- Successful matching method records may be reused individually from a valid
  v2 `complete_with_failures` artifact; failed siblings must be rerun. The
  supervisor must still recompute the top-level status from the complete
  expected identity set.

## Required Checks, Tests, And Review

1. Pure contract tests prove canonical fingerprints are order-stable and change
   for every field/path in the closed source, config, runtime, fixture, and
   schedule contracts. They also prove missing paths/metadata and unknown
   method-option keys fail closed.
2. Parameterized status tests cover `complete`, `complete_with_failures`,
   `failed`, and `interrupted`; missing and duplicate identities must fail
   closed.
3. Resume tests cover exact-match passed reuse, failed sibling rerun, v1
   rejection, source mismatch, config mismatch, malformed JSON, missing finite
   evidence, and incomplete method metadata.
4. Strict-JSON tests use raw `NaN`, `Infinity`, and `-Infinity` tokens plus
   nested non-finite values. They require `allow_nan=False` on encoding and
   `parse_constant` rejection on decoding for every schedule, progress, child,
   aggregate, historical/resume input path. Atomic-write tests prove a hard exit
   cannot replace a valid prior artifact with a partial file.
5. Method-selection tests prove each child command names exactly one method and
   instrument the builder registry with fail-fast/counting spies: the selected
   builder is dispatched once, its returned callable executes only at the
   declared trace/compile/warm/repeat stages, and every unselected builder and
   callable has zero invocations. The child method-dispatch ledger must equal
   the selected singleton. Synthetic
   failure/timeout/crash must not erase passed sibling records.
6. Stage tests cover all seven exact stages. Subprocess fixtures hard-exit and
   timeout immediately after every flushed stage marker; the supervisor must
   recover that exact last stage from the journal and synthesize the correct
   terminal state. Tests also cover no marker, a partial trailing line, an
   existing-path exclusive-create failure, a stale valid journal from an earlier
   attempt, wrong case/method/fingerprint/resume identities, and a retry after a
   partial line. Only current-attempt matching events may affect recovery.
7. Schedule tests persist the expected identities before launch and reject
   missing, extra, and duplicate returned identities. They test every allowed
   aggregate-check subset, require the first two checks for all schedules,
   require all five for numerical schedules, and prove a child cannot assert
   aggregate parity.
8. A tiny GPU-hidden, explicitly non-JIT integration smoke may run at
   `dimension=2,P=2,T=2,B=1` for one selected method only. It is debug evidence,
   writes below `/tmp/kalman_qr_phase1_harness/`, and cannot support math,
   timing, XLA, or default claims.
9. Run:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_benchmark_contract.py
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py
git diff --check -- \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py
```

If the optional integration test file is not needed, omit its path from the
pytest command and record why. Do not run pytest collection against a missing
path.

10. Recompute every historical path/hash recorded in the Phase 0 inventory and
    require exact equality. Then write the Phase 1 result, refresh Phase 2 with actual source hashes and
   exact diagnostics, then obtain bounded read-only review of the exact Phase 2
   subplan for consistency, correctness, feasibility, artifact coverage, and
   boundary safety. Maximum five rounds for the same material blocker.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can one method fail without erasing other method evidence or being mislabeled complete/resumable? |
| Exact baseline | Phase 0 source fingerprint and historical v1 all-method execution/status behavior. |
| Primary criterion | Focused tests prove one-method subprocess isolation, exact stage localization, strict JSON, correct top-level/exit status, and exact fingerprinted per-method resume behavior. |
| Promotion vetoes | Failed method silently skipped; unselected method executed; failed/stale/mismatched artifact reused; invalid JSON; missing/duplicate method identity; missing failure stage; GPU failure omitted from exit status. |
| Repair triggers | Any focused contract or integration test failure within the declared write set. |
| Explanatory only | Exception text, stage durations, and the optional tiny smoke runtime. |
| Not concluded | Numerical parity, vectorization correctness, XLA viability, runtime ranking, GPU readiness, or any HMC/posterior/default/production/scientific claim. |

## Skeptical Pre-Execution Audit

- Baseline is the exact Phase 0 fingerprint, not a passing historical row.
- The primary criterion is harness behavior under tests; timings and historical
  pass labels cannot promote the phase.
- Method isolation is tested with synthetic failures before any framework run.
- Resume eligibility binds execution-affecting source and config but does not
  use unrelated documentation dirt as a hidden invalidator.
- The optional TensorFlow smoke is GPU-hidden before import and explicitly
  non-JIT debug evidence; it is not a production/XLA proxy.
- Old runner/output paths are immutable, so repair cannot erase the failure
  baseline.
- A method failure is a repair trigger or method-local result. Only corrupt or
  structurally incomplete harness evidence is a continuation veto.

Audit status: `PASSED_LOCALLY_AWAITING_BOUNDED_REVIEW`.

## Forbidden Claims And Actions

- Do not edit Kalman/QR algorithmic math or derivative helpers.
- Do not repair the batched-autodiff tape/reduction semantics in Phase 1.
- Do not move materialization out of timing or claim timing fairness yet.
- Do not launch XLA compilation, GPU/CUDA probing, a grid, or a comparison run.
- Do not overwrite/delete/migrate historical 2026-07-09 artifacts.
- Do not reset, clean, commit, or alter unrelated dirty work.
- Do not weaken XLA-on defaults; the optional non-JIT smoke is a labeled debug
  exception only.

## Exact Next-Phase Handoff Conditions

All conditions are conjunctive:

- Required schema/status/fingerprint/resume/method/stage tests pass.
- Any optional integration smoke used is GPU-hidden, finite, and labeled debug;
  its absence is justified if pure/synthetic integration tests carry the gate.
- `py_compile`, focused pytest, and `git diff --check` pass.
- Phase 1 result maps each Phase 0 harness defect to a code change and focused
  test, records exact commands/hashes, and preserves all nonclaims.
- A close check rehashes every historical record named by the Phase 0 inventory;
  all paths and hashes match exactly.
- Phase 2 subplan is refreshed from actual Phase 1 evidence and receives exact
  `VERDICT: AGREE` from the bounded reviewer.

## Stop Conditions

- Method isolation cannot be represented without editing algorithmic math.
- A structurally complete partial-result schema cannot be made unambiguous.
- Strict JSON and fail-closed resume semantics cannot be established within the
  declared write set.
- Required source changed unexpectedly outside Phase 1 edits.
- An in-scope focused check remains broken after the implementation repair loop.
- A new human/runtime/package/network/model-file/product/scientific boundary is
  required.
- The same material review blocker fails to converge after five rounds.

An ordinary synthetic or method-local failure is not by itself a reason to
stop; localize, patch within the write set, and rerun the smallest focused check.

## Mandatory Phase-End Sequence

1. Run every required local check.
2. Write the Phase 1 result/close record.
3. Refresh the Phase 2 subplan from actual evidence.
4. Review Phase 2 and repair/recheck it before advancing.
