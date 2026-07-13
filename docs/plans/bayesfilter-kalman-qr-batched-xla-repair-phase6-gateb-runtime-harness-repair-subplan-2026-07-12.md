# Phase 6 Gate B Runtime-Harness Repair And Reauthorization Subplan

Date: 2026-07-12

Status: `CODEX_SUBSTITUTE_REVIEW_CONVERGED_NO_TARGET_REPAIR_EXECUTION_ACTIVE_NO_FRESH_TARGET_AUTHORITY`

Supervisor/executor: Codex in the current conversation.

Reviewer: Claude Opus at max effort was requested as read-only reviewer, but the
trusted execution layer rejected the external-disclosure call before its probe.
No repository bytes were sent and this is not Claude liveness evidence. Fresh
bounded Codex read-only substitute review is active and is explicitly weaker
than Claude review. Neither reviewer is an execution authority.

Parent Gate B subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-trace-pilot-subplan-2026-07-12.md`.

Parent Gate B subplan pre-repair SHA-256:
`d1c46aacdc6e15c234a3c3d739837d0fcb6fd8c57dbdc6c78f8c532ef0cc1214`.

Parent Phase 6 subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-subplan-2026-07-11.md`.

Parent Phase 6 SHA-256:
`b7b653d8febfa341dd2e8b53e8c274246eb49b6afcc59e4bca27126d3b33769b`.

## Phase Objective

Repair two common-invalidity defects exposed by the first authorized Gate B
trace launch, preserve that launch and its authority as immutable invalid-harness
evidence, and construct a fresh, separately reviewed `r2` Gate B namespace and
authority. Only after the repair passes focused and consolidated local checks,
bounded read-only review, strict archive validation, proposal review, detached
attestation validation, and a final skeptical runtime audit may the exact Gate B
trace/pilot command run again.

The defects in scope are:

1. the trace child replaced the exact invoked interpreter token
   `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` with its resolved symlink target
   `/home/ubuntu/anaconda3/envs/tfgpu/bin/python3.13`, so otherwise valid child
   evidence failed exact schedule provenance;
2. the supervisor tried to encode successful-process/invalid-evidence as
   `failed:invalid_child_evidence` while retaining truthful return code zero, but
   the process schema accepted only positive return codes for every `failed`
   record. The rejected transition then caused exception recovery to close the
   durable running record as `interrupted:supervisor_recovery`.

This is an engineering harness repair. It does not alter Kalman mathematics,
the target grid, methods, tolerances, budgets, XLA policy, or scientific gates.

## Entry Conditions Inherited From The Blocked Gate B Run

All conditions are conjunctive:

- Gate A remains closed with its recorded local pass and Claude `AGREE` review;
- the first Gate B proposal remains at SHA-256
  `e1b4cabba3dfd1ca292c4d7842d02ba86273001275b5f0a3b69ed0851a0ec823`,
  authority ID
  `ff9913a2bb8ad101fb9e4edab1a021aeff627228830c125b1296dbeaf874e837`;
- the first attestation remains at SHA-256
  `583e7842c3af2ebe0e00598224a86dd5cbf9c2627f0a22ca0638e25f870cd153`;
- the durable first trace ledger remains at SHA-256
  `9def8eb0f728a2262353002234bcbfe39aaa79bd49632e0737df2e0c8bf830ef`,
  non-final `state=running`, `update_index=2`, with the first record durably
  terminalized as `interrupted:supervisor_recovery` and no target worker alive;
- the first child artifact remains at SHA-256
  `a4235d0d399bde8148fbf07138aefae17b41cceaf6ecae7dd7e06fd36e333f31`;
- the first child progress journal remains at SHA-256
  `71585306fc1f579e780c059f99f3e2595fb02aac61030345b4a5a585c02bc365`;
- the old budget state and lease remain preserved at SHA-256
  `7cb269c18d4232851c0f620bc120e024ef303527c91945f11a24a56692a5deea`
  and
  `f0e1c0ce9876b0da20ef8e2715713a598921bc3d0d35a7d8068d617d7c14384e`;
- no pilot ledger was created and no process remains from the first launch;
- protected Kalman source hashes remain exactly:
  `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b`
  for `bayesfilter/linear/kalman_qr_tf.py`,
  `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57`
  for `bayesfilter/linear/kalman_qr_derivatives_tf.py`, and
  `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401`
  for `bayesfilter/linear/qr_factor_tf.py`;
- the opening hash ledger remains at SHA-256
  `9261e0c560ede29dc6893e0ffe3769cd762b38f3dd651af6dfcfa2f90dce1911`,
  with its known 144-entry scope and 36 omitted historical JSON counterparts;
- unrelated HMC, nonlinear, SSL-LSTM, and other-lane changes remain out of scope;
- this exact subplan receives bounded read-only `VERDICT: AGREE` before source
  edits; while external review remains policy-blocked, the agreeing artifact must
  be a fresh Codex substitute review labeled `codex_substitute_weaker`.

Pre-edit lane hashes for surgical attribution:

| Path | SHA-256 |
| --- | --- |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `efcd2925a8c12a69b0a6950c6315b1e9c53876019317a6e0fb970ea0d7cd4a6c` |
| `scripts/kalman_qr_benchmark_contract.py` | `730e4374ca046dcf7cc1aa76af69d1c7942c08eac17120006577ec37ca1ec06c` |
| `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py` | `5de6f8e276abf5e9312207fb5de748d8a55c196547371b835fa764ae48a15de2` |
| `tests/test_kalman_qr_phase6_cpu_xla_gates.py` | `5d26b60c5edd7996a1eee334cad6ac606e15a52f6c7df9af0c4ea448171c7268` |
| `tests/test_kalman_qr_phase6_gatea_runtime_controls.py` | `d0485a4ed1ae2305a75f15bb8e600ea48e43a22037af5e4d9659f8ce3d0798b6` |
| `tests/test_kalman_qr_phase6_import_discovery_cli.py` | `e9b4dada5902835105cab7e6ff7006c6c5ce10197eef2166842a2d83f3190715` |

## Observed Evidence And Hypotheses

The first child is not method evidence. Its process returned zero and its trace
payload says `passed`, but the enclosing Gate B evidence contract rejected it.

| Hypothesis | Smallest discriminating check | Current status |
| --- | --- | --- |
| The trace predicate failed only because child `command_argv[0]` resolved the interpreter symlink. | Element-wise compare reviewed schedule argv with embedded child argv, then recompute every terminal predicate. | Confirmed for the first observed rejection; all other checked predicates passed. |
| `sys.orig_argv` preserves the exact interpreter and script tokens supplied to the child process. | Invoke a no-TensorFlow temporary diagnostic through the reviewed interpreter token and assert exact element-wise equality. | Supported by a local `-c` probe; must be locked by a focused test before use. |
| Accepting resolved-path equivalence would repair the run safely. | Mutate the interpreter token to a different symlink with the same target. | Rejected design: it weakens exact reviewed invocation and is forbidden. |
| A zero-return process can truthfully end as common invalidity when its evidence is invalid. | Synthetic terminal transition with `reason=invalid_child_evidence`, `classification=common_invalidity`, return code zero, and an otherwise truthful process record. | Confirmed schema gap; implementation repair pending. |
| Broadly accepting return code zero for ordinary `failed` records is safe. | Try `child_nonzero_exit` with return code zero. | Rejected design: ordinary failed states must remain positive-return-only. |
| The old fixed artifact paths can be reused after source repair. | Inspect old trace, child, journal, proposal, attestation, budget state, and lease existence. | Falsified: reuse would overwrite or contaminate immutable evidence. A fresh namespace is required. |

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can the Gate B supervisor truthfully bind the exact reviewed child invocation and durably encode successful-process/invalid-evidence without overwriting the first failed authority lineage? |
| Candidate/mechanism | Raw invocation capture from `sys.orig_argv`; reason-aware failed-process validation; generation-scoped `r2` paths and authority; immutable `r1` archive manifest. |
| Exact baseline | The first Gate B proposal, attestation, trace ledger, child, journal, budget state/lease, their hashes above, and the pre-edit source/test hashes. |
| Expected failure mode | Hidden argv normalization remains; invalid-evidence fallback still double-transitions; fresh proposal accidentally binds old paths; stale files are deleted or imported; tests pass only through mocked equivalence. |
| Primary pass criterion | Focused tests demonstrate exact argv capture and truthful terminalization; consolidated Gate A suite passes; protected and `r1` hashes match; bounded review agrees with its actual strength recorded; fresh `r2` proposal/attestation pass strict authority validation. |
| Promotion veto | Any weakening from exact argv equality, ordinary failed/return-code mismatch acceptance, stale overwrite/import, protected drift, failed consolidated gate, invalid archive/proposal/attestation, or surviving process. |
| Continuation veto | Ambiguous ownership or mutable `r1` evidence; inability to construct a disjoint `r2` namespace; corrupted required artifact; new human/package/network/model/default/scientific authority; five non-converging review rounds for the same blocker. |
| Repair trigger | Any focused or consolidated check failure, reviewer `REVISE`, namespace collision, archive mismatch, or authority validation failure. |
| Explanatory only | Child elapsed time, GraphDef size/counts, stdout/stderr tails, and the fact that the first child internally reached `passed`. |
| Must not conclude | No Kalman correctness, CPU-XLA viability, method ranking, speed, GPU, HMC, posterior, default, production, or scientific claim from this repair. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Exact main question in the research intent ledger. |
| Baseline/comparator | Byte-identical `r1` lineage and strict synthetic state-machine cases; no target timing comparison. |
| Primary pass/fail criterion | Exact invocation bytes equal the reviewed schedule; reason-aware terminal semantics accept only truthful combinations; old and new namespaces are disjoint; all declared local gates pass. |
| Promotion veto diagnostics | Exact mismatch, stale/overwritten file, ordinary failure with zero return code, double transition, relaunch after recovery, invalid ledger, protected hash drift, or invalid authority. |
| Continuation veto diagnostics | Corrupt/ambiguous evidence or ownership, unsafe cleanup, missing required diagnostics, or boundary requiring new human authority. |
| Explanatory only | Runtime duration, trace internals, graph metrics, and test duration. |
| Not concluded even if passed | No method/backend outcome and no production or scientific readiness. |
| Preserved result | Archive manifest, focused/consolidated test logs, repair result, reviewer records, fresh proposal/review/attestation, final trace/pilot ledgers if separately authorized. |

## Required Artifacts

### Immutable `r1` Lineage

- `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_pilot_budget_2026-07-11.json`;
- `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_budget_attestation_2026-07-11.json`;
- `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-budget-review-round1-2026-07-11.md`, SHA-256
  `166bd5594d01371e6b08186f885e1cc3f06130d0072aa4a5363802412dc6e0e8`;
- `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-preflight-2026-07-12.md`, SHA-256
  `136c6d9d76a30df1068c2e05df1e5e7632ebf1fa9fe9288dda075a768cf7aa90`;
- the four Gate B subplan review records and the post-proposal artifact-repair
  review under `docs/reviews/`, with exact path/digest records captured by the
  archive constructor;
- `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_trace_census_2026-07-11.json`;
- `/tmp/kalman_qr_phase6_cpu_xla/trace/3ad6c9faf97fe6395a287e17.json`;
- `/tmp/kalman_qr_phase6_cpu_xla/trace/3ad6c9faf97fe6395a287e17.jsonl`;
- `/tmp/kalman_qr_phase6_cpu_xla/budget_state/gate_b-ff9913a2bb8ad101fb9e4edab1a021aeff627228830c125b1296dbeaf874e837.json`;
- its `.lease` file;
- new strict archive manifest:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r1_invalid_harness_archive_2026-07-12.json`.

The archive manifest must record exact absolute path, byte count, SHA-256, role,
and disposition for each item; exact element-wise schedule/child argv
differences; predicate localization; no-live-process evidence; source/test
pre-edit hashes; and explicit nonclaims. Its validator must rehash every listed
file. It must not embed the multi-megabyte child or ledger bytes.

### Repair And Review

- this subplan;
- `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-harness-repair-result-2026-07-12.md`;
- substitute plan review round 1:
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-repair-subplan-codex-review-round1-2026-07-12.md`;
- substitute plan review round 2:
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-repair-subplan-codex-review-round2-2026-07-12.md`;
- bounded plan review records under `docs/reviews/`, labeled with actual
  reviewer and strength;
- bounded implementation/result review records under `docs/reviews/`, labeled
  with actual reviewer and strength;
- focused and consolidated logs under
  `/tmp/kalman_qr_phase6_gateb_runtime_repair/`.

### Fresh `r2` Namespace

- work root: `/tmp/kalman_qr_phase6_cpu_xla_gateb_r2/`;
- import discovery:
  `/tmp/kalman_qr_phase6_cpu_xla_gateb_r2/import_discovery.json`;
- proposal:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_2026-07-12.json`;
- proposal review:
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r2-budget-review-round1-2026-07-12.md`, with preserved round `2..5` files only if needed;
- detached attestation:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_attestation_2026-07-12.json`;
- trace ledger:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_trace_census_2026-07-12.json`;
- pilot ledger:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_cpu_xla_pilot_2026-07-12.json`;
- fresh budget state/lease under the `r2` work root and fresh authority ID;
- final Gate B result:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r2-trace-pilot-result-2026-07-12.md`.

All `r2` paths must be absent before construction. The `r2` proposal must bind
this exact reviewed repair subplan as its plan and must contain exact path/digest
authority inputs for the `r1` archive manifest, the final repair result, the
final agreeing plan review, and the final agreeing implementation/result review.
It must also bind current source/runtime/schedule identities and only `r2`
runtime paths. The proposal reviewer must validate every one of those records.
Any later byte change to a bound plan, result, review, archive, source, command,
schedule, or runtime input invalidates the proposal and requires the affected
review and authority construction to repeat.

## Write Set

Implementation edits are limited to:

- `scripts/benchmark_kalman_qr_parameter_count_scaling.py`;
- `scripts/kalman_qr_benchmark_contract.py`;
- `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py`;
- `tests/test_kalman_qr_phase6_cpu_xla_gates.py`;
- `tests/test_kalman_qr_phase6_gatea_runtime_controls.py`;
- `tests/test_kalman_qr_phase6_import_discovery_cli.py` only if the new import
  namespace requires its exact-path update;
- new focused file
  `tests/test_kalman_qr_phase6_gateb_runtime_repair.py`;
- the plan/result/review/archive/authority artifacts named here;
- the visible execution ledger and stop handoff for status only.

No `bayesfilter/linear/*.py`, HMC, nonlinear, SSL-LSTM, unrelated test, package,
model, or environment file may be edited.

## Repair Design

### A. Exact Child Invocation

- Add one small helper that returns a validated copy of `sys.orig_argv`.
- Require a non-empty list of strings whose script token resolves to the current
  benchmark script; do not fall back silently to `sys.executable` plus
  `sys.argv`.
- Write that raw invocation into trace child `command_argv` without resolving,
  normalizing, canonicalizing, or accepting path aliases.
- Keep schedule and process-record equality exact and element-wise.
- Fail closed if `sys.orig_argv` is absent, malformed, or does not identify the
  current script.

### B. Truthful Invalid-Evidence Terminalization

- Keep ordinary `failed` process validation positive-return-only.
- Make process validation reason-aware so only
  `failed:invalid_child_evidence` may retain exactly return code zero with
  `timed_out=false`. Positive return codes retain ordinary honest child-failure
  semantics; negative returns and timeouts retain their existing crash/timeout
  semantics.
- Require `classification=common_invalidity` and require the candidate's normal
  terminal semantics to be invalid; never use this route for a valid child
  failure or passed record.
- Persist exactly one terminal transition. A transition-validation exception
  must not convert a completed child into a second recovery outcome.
- Common invalidity must prune later launches according to the existing ledger
  policy; it must not relaunch or reinterpret the child.

### C. Immutable `r1` / Fresh `r2` Boundary

- Introduce explicit constants for archived `r1` paths and active `r2` paths.
- Use the `r2` work root for all import, child, journal, sidecar, dependency,
  budget, lease, trace, and pilot paths.
- Require strict absence before first construction and refusal on unexpected
  bytes; never unlink `r1` evidence.
- Bind the archive manifest into the new proposal and revalidate it at proposal
  construction and runtime-authority validation.
- Update future Gate C predecessor constants to consume only closed `r2` Gate B
  artifacts; do not import the interrupted `r1` record.

## Required Checks, Tests, And Reviews

### No-Target Diagnosis And Archive Gate

1. Recompute the exact element-wise argv diff and every first-record predicate.
2. Strictly parse the child, journal, proposal, attestation, trace, budget state,
   and lease where their schemas permit; record the trace as non-final and
   invalid for method interpretation.
3. Prove no matching worker/process group remains without counting the probe
   command itself.
4. Write and strictly validate the archive manifest; rehash every listed path.
5. Recheck the protected, parent-plan, opening-ledger, and pre-edit hashes.

After implementation, run the exact archive constructor/validator below. It
must exit zero, write the declared JSON on first construction, and on every
later invocation rehash and validate the identical manifest without overwriting
nonidentical bytes. Full output goes to the declared log.

```bash
mkdir -p /tmp/kalman_qr_phase6_gateb_runtime_repair
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-archive-r1 \
  --output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r1_invalid_harness_archive_2026-07-12.json \
  > /tmp/kalman_qr_phase6_gateb_runtime_repair/archive.log 2>&1
```

### Focused Repair Tests

Tests must cover at least:

- exact `sys.orig_argv` preservation for interpreter symlink spelling;
- rejection when `sys.orig_argv` is absent, malformed, has a different script,
  or contains a different interpreter token even if it resolves to the same
  executable;
- trace child envelope equality to exact reviewed `child_command_argv`;
- zero-return `failed:invalid_child_evidence` accepted only with
  `common_invalidity` and invalid normal child semantics;
- zero-return `child_nonzero_exit` and other ordinary failures rejected;
- positive-return honest child failure remains valid and is not reclassified;
- exactly one durable terminal transition when the child process returns zero
  but evidence is invalid;
- no exception-recovery double transition, no relaunch, and common-invalidity
  pruning after that transition;
- all `r1` paths unchanged and all generated `r2` paths disjoint;
- old trace/child/journal bytes cannot be imported, resumed, deleted, or
  overwritten by the `r2` schedule;
- new proposal accepts only the exact archive manifest and repair-plan digest;
- stale or unexpected `r2` files cause refusal rather than unlink/reuse.

No test may invoke a target TensorFlow trace, XLA compile, or Kalman filter run.
A tiny Python argv subprocess with no TensorFlow import is allowed.

### Local Closure Gate

- exact compilation command below exits zero;
- the dedicated focused repair file below exits zero without deselection;
- the exact consolidated GPU-hidden Gate A suite below exits zero with one
  declared CPU-XLA deselection;
- scoped `git diff --check` and trailing-whitespace scan;
- AST/source checks proving exact equality was not replaced with resolved-path
  equivalence;
- all protected hashes and all `r1` archive hashes unchanged;
- no matching worker/process group;
- strict absence of runtime `r2` trace/pilot/budget artifacts before authority.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  scripts/kalman_qr_benchmark_contract.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_phase6_gateb_runtime_repair.py \
  > /tmp/kalman_qr_phase6_gateb_runtime_repair/py_compile.log 2>&1

CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_phase6_gateb_runtime_repair.py \
  > /tmp/kalman_qr_phase6_gateb_runtime_repair/focused_pytest.log 2>&1

CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py \
  tests/test_kalman_qr_phase6_gatea_runtime_controls.py \
  tests/test_kalman_qr_phase6_import_discovery_cli.py \
  tests/test_kalman_qr_measurement_boundaries.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_batch_native_autodiff.py \
  tests/test_kalman_qr_phase6_gateb_runtime_repair.py \
  --deselect=tests/test_kalman_qr_batch_native_autodiff.py::test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value \
  > /tmp/kalman_qr_phase6_gateb_runtime_repair/consolidated_pytest.log 2>&1

git diff --check -- \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  scripts/kalman_qr_benchmark_contract.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py \
  tests/test_kalman_qr_phase6_gatea_runtime_controls.py \
  tests/test_kalman_qr_phase6_import_discovery_cli.py \
  tests/test_kalman_qr_phase6_gateb_runtime_repair.py

rg -n '[[:blank:]]+$' \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  scripts/kalman_qr_benchmark_contract.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py \
  tests/test_kalman_qr_phase6_gatea_runtime_controls.py \
  tests/test_kalman_qr_phase6_import_discovery_cli.py \
  tests/test_kalman_qr_phase6_gateb_runtime_repair.py
```

The `rg` command must exit `1` with no output, meaning no match. The dedicated
focused file carries the AST/source-shape assertions: exact schedule/child argv
equality remains present, the trace envelope uses the validated raw invocation,
and no resolved-path equivalence appears in a Phase 6 child-evidence validator.

After the local closure gate, obtain bounded read-only review of the smallest
exact repair result path. If external review remains policy-blocked, use a fresh
Codex substitute and label it weaker. If the reviewer requests source or a test,
send only that exact path next. On `REVISE`, patch in scope, rerun focused checks
and then the closure gate. Stop after five material rounds for the same blocker.

### Fresh Authority Gate

1. Freeze the reviewed repair subplan, final agreeing plan review, repair result,
   final agreeing implementation/result review, and archive bytes; do not edit
   them after proposal construction.
2. Run only the exact no-target `r2` proposal constructor.
3. Strictly validate proposal schema, archive input, source/runtime/schedule,
   exact paths, unchanged 3045-second budget, and fresh authority ID.
4. Obtain bounded read-only review of exactly the proposal path. Use the actual
   reviewer-strength label; a Codex substitute remains weaker than Claude.
5. Create a detached attestation binding the exact proposal, repair plan, and
   agreeing review path/digests.
6. Run strict runtime-authority validation and all no-worker/absence/hash checks.
7. Record a final skeptical runtime audit. Only exact `PASS` may cross into the
   target run.

Exact no-target commands, all expected to exit zero, are:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-prepare-proposal gate_b \
  --output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_2026-07-12.json \
  > /tmp/kalman_qr_phase6_gateb_runtime_repair/proposal.log 2>&1

/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-create-attestation gate_b \
  --budget-contract docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_2026-07-12.json \
  --review-path docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r2-budget-review-round1-2026-07-12.md \
  --output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_attestation_2026-07-12.json \
  > /tmp/kalman_qr_phase6_gateb_runtime_repair/attestation.log 2>&1

/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-validate-authority gate_b \
  --budget-contract docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_2026-07-12.json \
  --budget-attestation docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_attestation_2026-07-12.json \
  > /tmp/kalman_qr_phase6_gateb_runtime_repair/authority_validation.log 2>&1
```

The implementation must add these closed CLI modes. The archive and proposal
commands are no-target/import-only operations; attestation creation and authority
validation read only bounded governance JSON/Markdown. None may construct a
fixture, trace a function, import a target filter, compile XLA, or run a method.

### Exact Fresh Runtime Command

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  timeout --signal=TERM --kill-after=45s 3000s \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-pilot \
  --dimensions 10 20 30 --parameter-counts 50 150 \
  --batch-sizes 1 4 16 --timesteps 120 \
  --dtype float32 --device cpu --cpu-threads 1 --jit-compile \
  --trace-child-timeout-seconds 60 --xla-child-timeout-seconds 60 \
  --xla-cell-timeout-seconds 160 \
  --budget-contract docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_2026-07-12.json \
  --budget-attestation docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_budget_attestation_2026-07-12.json \
  --trace-output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_trace_census_2026-07-12.json \
  --output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_cpu_xla_pilot_2026-07-12.json
```

The target roster, order, 60-second child execution caps, five-second TERM and
five-second KILL/reap grace, 70-second lifecycle cap, 160-second paired-cell cap,
3000-second outer deadline, 45-second outer KILL grace, and 3045-second authority
remain unchanged from `r1`. This is a fresh invalid-harness repair authority,
not a candidate-budget reset or a response to an unfavorable method outcome.

## Skeptical Pre-Execution Audit

Status: `PASS_FOR_PLAN_REVIEW_AND_NO_TARGET_REPAIR_ONLY`.

- Wrong baseline: the comparator is the byte-preserved first authority lineage,
  not a reconstructed or cleaned ledger.
- Proxy promotion: child-internal `passed`, trace duration, graph metrics, and
  unit tests cannot become method or backend outcomes.
- Missing stop conditions: archive drift, namespace collision, invalid
  authority, unsafe ownership, failed local gates, and review non-convergence
  are explicit stops.
- Unfair comparison: target roster, order, methods, timeouts, budget, device,
  threads, dtype, and JIT remain unchanged; only common harness identity and
  evidence paths change prospectively.
- Hidden assumption: raw invocation preservation is tested with the actual
  interpreter symlink spelling and fail-closed negative mutations.
- Stale context: pre-edit, protected, plan, opening-ledger, and `r1` hashes are
  checked before edits, after tests, before proposal, and after runtime.
- Environment mismatch: repair tests are CPU/no-target. The later Gate B command
  remains deliberate GPU-hidden CPU XLA under the existing reviewed policy.
- Artifact fitness: the archive manifest answers why a fresh authority is
  legitimate; focused state-machine tests answer the repair; trace/pilot
  ledgers, not test outcomes, answer the later Gate B target question.
- Misleading pass: a repaired harness does not validate the Kalman method.
- Misleading fail: the first common-invalidity event invalidated the harness
  authority, not the candidate or research direction; the planned repair is
  exactly the next discriminating phase.

## Forbidden Claims And Actions

- Do not call the first child, trace ledger, or interrupted record method,
  backend, CPU-XLA, correctness, or performance evidence.
- Do not overwrite, delete, rename, truncate, chmod, normalize, finalize, resume,
  or import any `r1` runtime artifact or budget state.
- Do not accept resolved-path, inode, realpath, basename, or executable-equivalent
  provenance in place of exact reviewed argv equality.
- Do not falsify the process return code, synthesize a positive code, or call a
  zero-return invalid-evidence process an ordinary child failure.
- Do not weaken ordinary failed-state semantics.
- Do not change target roster, methods, ordering, timeouts, caps, budget,
  tolerances, device, thread count, dtype, JIT, TF32, or promotion criteria.
- Do not launch any trace, XLA, scalar reference, Gate C, GPU, HMC, or comparison
  workload before fresh authority passes every gate above.
- Do not edit protected algorithm files or unrelated other-lane files.
- Do not treat any reviewer as execution authority or as authorization to cross human,
  runtime, model-file, funding, product/default, release, or scientific-claim
  boundaries.

## Exact Next-Phase Handoff Conditions

All are conjunctive:

- archive manifest exists, strictly validates, and all `r1` bytes still match;
- exact argv capture and truthful invalid-evidence terminalization are covered by
  focused positive and negative tests;
- consolidated Gate A suite, compile, whitespace, source-shape, protected-hash,
  no-worker, and namespace checks pass;
- repair result is written with run manifest, decision table, inference-status
  table, alternative explanation, nonclaims, and test evidence;
- bounded implementation/result review returns exact `VERDICT: AGREE` with its
  actual reviewer-strength label;
- fresh `r2` proposal, proposal review, attestation, and runtime-authority checks
  all pass without changing the old budget or target contract;
- final skeptical runtime audit is recorded as `PASS`;
- only then may the exact fresh runtime command run;
- after that run, the Gate B `r2` result must distinguish harness validity,
  candidate outcome, and research-direction status. Refresh and review the
  dedicated Gate C subplan only for the eligible branch below; a valid trace
  rejection writes a Gate C blocker/decision subplan, and harness-invalid
  outcomes return to repair.

Exact pre-runtime handoff state:
`GATE_B_R1_ARCHIVED_REPAIR_REVIEWED_R2_AUTHORITY_VALID_RUNTIME_READY`.

Exact post-runtime handoff state:
`GATE_B_R2_CLOSED_GATE_C_SUBPLAN_REVIEWED_RUNTIME_STILL_BLOCKED`.

That post-runtime state is emitted only when `trace_common_valid=true`; the
trace ledger is final and passes every ledger, evaluator, authority,
process-cleanup, and no-common-invalidity check; and the pilot ledger is final,
has no pending/running/interrupted or common-invalidity record, and passes every
ledger/authority/process-cleanup check. Pilot records may honestly be `passed`,
`failed`, `timed_out`, or `crashed`; non-passed method-local outcomes reject only
those candidates under the cap.

If the trace ledger is final and harness-valid but the typed structural trace
gate is honestly false, every pilot record must be final
`not_launched:trace_gate_not_passed`. Gate B may close as a valid negative trace
result, Gate C is not eligible, and the exact handoff is
`GATE_B_R2_VALID_TRACE_REJECTION_GATE_C_BLOCKED`.

If any common invalidity, malformed/non-final ledger, `interrupted` record,
authority mismatch, missing diagnostic, unsafe ownership, surviving process, or
supervisor failure occurs, write a repair/blocker result and emit
`GATE_B_R2_HARNESS_INVALID_REPAIR_REQUIRED`. Do not claim Gate B closed, do not
draft an executable Gate C authority, and do not cross into Gate C.

## Stop Conditions

- Any `r1` path or hash changes unexpectedly.
- Exact provenance can be repaired only by weakening equality.
- Truthful zero-return invalid-evidence terminalization cannot be represented
  without weakening ordinary failure semantics.
- `r2` paths are not disjoint or contain unexplained bytes.
- Required local checks fail after bounded in-scope repair.
- bounded review does not converge within five material rounds for the same
  blocker.
- A live or stale process identity cannot be proved safe.
- Proposal, archive, attestation, or runtime authority cannot be strictly
  validated.
- Continuing requires new package/network/credential/model-file/funding,
  product/default/release, scientific-claim, or other human authority.

## Mandatory Close Sequence

At the end of this repair subplan:

1. run every required local and artifact check;
2. write the repair result/close record;
3. construct and review the fresh Gate B authority, or write a blocker result;
4. after an authorized Gate B run, apply the exact valid-pass, valid-trace-reject,
   or harness-invalid branch above; only the valid-pass branch drafts/refreshes
   an executable dedicated Gate C subplan;
5. review the next subplan for consistency, correctness, feasibility, artifact
   coverage, and boundary safety;
6. visibly repair and rerun focused checks for any fixable issue, stopping after
   five material review rounds for the same blocker;
7. advance only when the exact handoff conditions are satisfied.
