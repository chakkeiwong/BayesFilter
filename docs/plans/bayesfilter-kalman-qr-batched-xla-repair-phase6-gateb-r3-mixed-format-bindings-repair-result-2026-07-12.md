# Phase 6 Gate B R3 Mixed-Format Bindings Repair Result

Date: 2026-07-12

Status: `LOCAL_R3_REPAIR_PASSED_RESULT_REVIEW_ROUND2_PENDING_NO_TARGET_AUTHORITY`

Parent subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-mixed-format-bindings-repair-subplan-2026-07-12.md`.

Parent subplan SHA-256:
`3af2959c719e62b4fb02d9e7c78b3be86521d7e62b757d35d2e4acede679ba1a`.

Final agreeing subplan review SHA-256:
`99523d5660988dc08cf5509391a3f7c6ff0ba51cd4352a9da09272f2d1bc4b27`.

## Result

The R3 no-target harness repair passed its local engineering gate. Gate B
authority inputs can now contain one strict JSON archive plus three Markdown
files: every input remains required, present, byte/base64/SHA bound, and in the
exact proposal order, but Markdown is not falsely required to parse as JSON.
Proposal, attestation, Phase 4/5 evidence, and runtime predecessors retain
their semantic strict-JSON requirements.

Deterministic trace bindings, live authority, and the complete initial trace
ledger are now constructed and validated before the budget lease or state is
created. A final parent guard immediately before spawn converts detected drift
into a durable all-`not_launched:common_invalidity` trace ledger with zero
spawn. Each authorized child receives a digest-bound authority snapshot and
validates it before importing TensorFlow. Persistent terminal drift preserves
the real timeout, signal, or return code while invalidating the evidence as
common invalidity.

The immutable R2 failed generation is archived at
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r2_invalid_harness_archive_2026-07-12.json`,
SHA-256
`40e6a186a28cd15d4ab3901f516854a6d84065fcb9759716108ab8e103e7834d`.
It contains the exact 13-file generation inventory, six required absences,
closed directory inventories, truthful running-budget/released-lease state,
root-cause classification, and nonclaims. R2 was not resumed, closed, deleted,
renamed, imported, or otherwise mutated.

No target trace, TensorFlow concrete-function trace, XLA compile, or Kalman
execution was run in this repair. Fresh R3 proposal construction, proposal
review, attestation, final skeptical runtime audit, and the proposal-bound Gate
B command remain downstream gates. This result does not establish that the
original memory or performance problems are fixed.

## Implemented Repairs

| Repair | Implementation | Evidence |
| --- | --- | --- |
| Mixed-format authority | `_phase6_bindings_valid` requires valid/present authority blobs and exact ordered proposal path/SHA projection without requiring `strict_json` for that category only | Real one-JSON/three-Markdown file test; byte count, base64, path, digest, order, and presence mutations fail |
| Semantic JSON boundary | Proposal, attestation, Phase 4/5 evidence, and runtime predecessor artifact still require parsed strict JSON | Independent category mutations fail; source-shape scan confirms the predicates remain |
| Pre-budget deterministic gate | `run_phase6_pilot` builds/revalidates trace bindings and constructs/checks the initial trace ledger before lease/open | Entrypoint order test and deterministic-failure integration show no work root, budget, lease, trace, pilot, or child artifact |
| Final parent guard | Revalidation occurs after child-path safety preparation and immediately before snapshot/spawn | A real Markdown mutation after budget open yields zero spawn, durable common-invalidity trace evidence, closed budget state, and released lease |
| Child-entry guard | A durable snapshot binds full bindings, one schedule row, exact argv, exact path, file digest, CPU environment, source/runtime identity, and live authority bytes before TensorFlow import | A real mutation after the parent guard launches a subprocess but produces a typed failure envelope with all target-work counters zero; a poison TensorFlow module is never imported |
| Terminal guard | Completion revalidates authority before committing terminal evidence | Persistent drift records `authority_revalidation_failed`, common invalidity, and the actual return code `7` |
| Checkpoint claim boundary | Target work uses captured governance identity after entry; terminal revalidation detects drift present at that checkpoint | Tests cover parent, child-entry, and terminal checkpoints; no claim is made for an ABA mutation restored entirely between checkpoints |
| Fresh R3 namespace | Active plan, work root, discovery, proposal, attestation, trace, pilot, and schedule child paths use R3; R2 constants are historical/archive only | Active-path scan, absent-root checks, exact schedule checks, and immutable R2 hash checks |
| R2 archival | Exact closed `--phase6-archive-r2` CLI builds and validates the archive without writing under R2 | Archive validator and six independent mutation tests; R2 import/state/lease hashes unchanged |

No protected `bayesfilter/linear/*.py` algorithm source was edited by this
repair.

## Evidence Contract Assessment

| Field | Observed result |
| --- | --- |
| Engineering question | Passed locally: mixed-format path/digest authority is accepted without semantic weakening, deterministic trace preflight precedes budget mutation, and checkpoint drift fails closed. |
| Exact baseline | Immutable R2 proposal, review, attestation, audit, failed result, import discovery, running budget state, released lease, and protected source hashes. |
| Primary criterion | Compile, 42 focused tests, exact consolidated GPU-hidden suite, real mutation races, archive validation, source-shape/static checks, protected hashes, no-worker check, and strict R3-root absence passed. |
| Hard vetoes | No accepted tampered/reordered/missing input, semantic JSON weakening, R2 drift/reuse, budget mutation on deterministic preflight failure, spawn after parent-detected drift, child target work after entry drift, terminal outcome falsification, protected drift, or surviving process was observed. |
| Explanatory only | Test durations, warning count, file sizes, and earlier failed test-order diagnostics. |
| Not concluded | No trace-structure result, CPU-XLA viability, Kalman numerical correctness, memory reduction, performance improvement, scalability, GPU readiness, method ranking, HMC/posterior validity, default readiness, production readiness, or scientific validity. |

## Visible Repair Loop

The first consolidated attempt reached `402 passed, 1 deselected` but failed
the child-entry test because a preceding synthetic cell-cap test had created
empty directories under the active R3 root. Codex did not accept that run as
evidence. The cell-cap test was isolated to a temporary work root, the empty
test directories were removed, and the exact consolidated suite was rerun.

A later consolidated run passed `403 passed, 1 deselected`; Codex then
strengthened the parent and child boundary tests from synthetic callback
failures to real bound-file byte mutations. The focused and consolidated
suites were rerun again after those changes. Only the final post-change runs
below carry the local evidence burden.

These failures were test-harness isolation and evidence-quality repairs. They
did not show a Kalman, TensorFlow, or XLA failure because the target node was
never run.

Claude Opus/max result review Round 1 then returned `REVISE`, SHA-256
`bd582675e6ce546eb5c43b931fc5277e5f1c19d821127a3612f6baf128c5cde8`.
The substantive repair boundaries were accepted, but the record omitted the
literal compile/static commands and left its final logs only under `/tmp`.
Codex preserved the final logs under `docs/benchmarks`, wrote a strict JSON
check manifest with exact commands, exit codes, byte counts, hashes, and check
outcomes, disclosed that the scoped lane files are untracked and therefore
require `git diff --no-index --check`, and reran the focused suite. No target,
TensorFlow, XLA, or Kalman execution was required or performed for this review
repair.

## Local Checks

The durable closed check report is
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_bindings_repair_check_manifest_2026-07-12.json`,
byte count `17084`, SHA-256
`85a0bbf7f00e8f25a75b7fe2aa431516e579cd156e950f91789b4bd9e57aff17`.
It records the literal commands and outputs summarized below. The manifest is
the evidence contract for command reproducibility; the human-readable result
does not replace it.

### Compilation

~~~bash
CUDA_VISIBLE_DEVICES=-1 \
PYTHONPYCACHEPREFIX=/tmp/kalman_qr_phase6_gateb_r3_binding_repair/pycache \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py \
  tests/test_kalman_qr_phase6_gatea_runtime_controls.py \
  tests/test_kalman_qr_phase6_gateb_runtime_repair.py \
  tests/test_kalman_qr_phase6_import_discovery_cli.py
~~~

- exit code: `0`;
- durable log:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_bindings_repair_py_compile_2026-07-12.txt`;
- byte count: `0`;
- SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

### Focused Repair Suite

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_phase6_gateb_runtime_repair.py
```

Result: `42 passed in 19.31s`.

- exit code: `0`;
- durable log:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_bindings_repair_focused_pytest_2026-07-12.txt`;
- byte count: `100`;
- SHA-256:
  `abc4c9180021bb04018bd2e8a9ecd552001577033aa0dacfd2ccb978020cd87a`.

### Consolidated GPU-Hidden Suite

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py \
  tests/test_kalman_qr_phase6_gatea_runtime_controls.py \
  tests/test_kalman_qr_phase6_import_discovery_cli.py \
  tests/test_kalman_qr_measurement_boundaries.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_batch_native_autodiff.py \
  tests/test_kalman_qr_phase6_gateb_runtime_repair.py \
  --deselect=tests/test_kalman_qr_batch_native_autodiff.py::test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value
```

Result: `403 passed, 1 deselected, 6634 warnings in 214.43s`.

- exit code: `0`;
- durable log:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_bindings_repair_consolidated_pytest_2026-07-12.txt`;
- byte count: `16581`;
- SHA-256:
  `9a4de98dc4634e6612b89f5b12cad16588e4100125ed0a4857309923ce63a99f`.

The warnings are existing Gast/AutoGraph deprecations. They are explanatory,
not promotion evidence. GPU was deliberately hidden before framework imports.
The single deselected node is the target CPU-XLA check forbidden until fresh
R3 authority passes.

### Round 1 Focused Recheck

After repairing the result evidence, the exact focused command above was rerun
with GPU hidden. It exited `0`: `42 passed in 19.32s`.

- durable log:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_bindings_repair_round1_focused_recheck_2026-07-12.txt`;
- byte count: `100`;
- SHA-256:
  `015e438f603d63f0c9ea266b38787331239d9de6ba4e76472f21048540f46941`.

### Static, Hash, And Namespace Checks

The literal commands, return codes, and outputs are in the bound manifest. The
applicable checks were:

- closed `rg -n '[ \t]+$' ...` scan over the seven lane implementation/test
  paths and this result: expected no-match exit `1`, no output;
- seven exact `git diff --no-index --check /dev/null <path>` commands: each
  exited `1` because a nonempty untracked file differs from `/dev/null`,
  with zero diagnostic bytes, so Git reported no whitespace error;
- GPU-hidden exact Python source assertion: all six predicates true, including
  byte-valid/non-JSON authority inputs, retained semantic JSON requirements,
  and binding/revalidation/initial-ledger checks before lease creation;
- GPU-hidden exact R2 archive validator: exit `0`, all `13` closed checks
  true;
- exact `sha256sum` commands: protected, reviewed-lineage, R2, implementation,
  and durable-log hashes matched the manifest;
- exact no-worker Python scan: exit `0`, matching PID list `[]`;
- exact namespace command
  `test ! -e /tmp/kalman_qr_phase6_cpu_xla_gateb_r3 && test ! -L /tmp/kalman_qr_phase6_cpu_xla_gateb_r3`:
  exit `0`.

The ordinary scoped `git diff --check` command also exited `0`, but it is
explicitly informational and not a pass criterion because all seven lane
implementation/harness files are untracked. The no-index commands above are
the applicable Git checks. Compilation, tests, the closed whitespace scan,
source assertions, archive validation, and hashes carry the evidence burden.

## Protected State

| Path | Closing SHA-256 |
| --- | --- |
| `bayesfilter/linear/kalman_qr_tf.py` | `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b` |
| `bayesfilter/linear/kalman_qr_derivatives_tf.py` | `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57` |
| `bayesfilter/linear/qr_factor_tf.py` | `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401` |
| reviewed R3 subplan | `3af2959c719e62b4fb02d9e7c78b3be86521d7e62b757d35d2e4acede679ba1a` |
| agreeing R3 subplan review | `99523d5660988dc08cf5509391a3f7c6ff0ba51cd4352a9da09272f2d1bc4b27` |
| immutable R2 archive | `40e6a186a28cd15d4ab3901f516854a6d84065fcb9759716108ab8e103e7834d` |
| R2 import discovery | `8ae6086bd6b8bbebd7bf236536a80cb6b8befa993a9e686801c451e8fec4c8ac` |
| R2 running budget state | `a4cc284b64d6527a7357171f4c47395a7f29f7fed7e50b15563257feae09390f` |
| R2 released lease | `ae711efe84056ae416d5fe2d2d40751b91afaa7f3a2e3530f095fb501a03b456` |

## Closing Implementation Hashes

| Path | SHA-256 |
| --- | --- |
| `scripts/kalman_qr_benchmark_contract.py` | `f52a20624eb3c8c72c59cc2809f4cd870de4c3c84276fed97f308bc4f0a75e64` |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `baf62b85f885073d0b72b5c13af0463ac5566f2429c16d5c98a542aa24c8eec9` |
| `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py` | `66c2f43acc4312640ee39af777830e72fc4fb5112fcb8844453a53bcde2b03eb` |
| `tests/test_kalman_qr_phase6_cpu_xla_gates.py` | `08941fa01eeba361a6939facc4a238d00691ea3d65b1705ed70e50f6e7df1447` |
| `tests/test_kalman_qr_phase6_gatea_runtime_controls.py` | `376bdfcd2b67c1dc6e7393b9606308e4edc60ff8ced08e7a395347bb29b4c270` |
| `tests/test_kalman_qr_phase6_gateb_runtime_repair.py` | `8cfd51086291b5f29c07662421b88cc7d48d83152f36038970f2e0a3d7d5ecb8` |
| `tests/test_kalman_qr_phase6_import_discovery_cli.py` | `a203abb1fc553598749e31ed3b33b8ab566b1dfb9846132c7fad2c0c9dcccfa9` |

The shared worktree remains dirty and these lane files are untracked in the
current index. Unrelated other-lane changes were not modified or used as this
lane's evidence.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Command | Exact compile, focused, and consolidated commands above |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; CPython 3.13.13; Linux 6.8.0-124-generic x86_64 |
| CPU/GPU status | Deliberate GPU-hidden local checks; no target GPU or CPU-XLA run |
| JIT/XLA | Target JIT node deselected; no Gate B XLA compile authorized or run |
| Data version | N/A; deterministic contract tests and synthetic fixtures |
| Random seeds | N/A; no stochastic experiment |
| Wall time | Focused `19.31s`; consolidated `214.43s`; Round 1 focused recheck `19.32s`; target runtime N/A |
| Output artifacts | R2 archive, durable strict JSON check manifest, and four durable `docs/benchmarks/*bindings_repair*.txt` logs named above |
| Plan file | Parent R3 subplan above |
| Result file | This file |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept local R3 harness repair for bounded result review | Passed | No local repair veto fired | Real Gate B trace/XLA behavior remains unmeasured; Claude may find an artifact or boundary defect | Bounded Claude Opus/max review of exactly this result; repair on `REVISE`; freeze only on `AGREE` | No target, memory, performance, default, production, or scientific claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the local engineering repair only |
| Statistically supported ranking | None; no stochastic or method comparison ran |
| Descriptive-only differences | Test wall times, warning count, file sizes, and repair-loop counts |
| Default-readiness | Not evaluated |
| Next evidence needed | Agreeing result review, fresh exact R3 proposal/review/attestation, final skeptical audit, then proposal-bound Gate B trace/pilot evidence |

## Evidence Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Local repair gate passed for mixed-format binding, budget ordering, parent/child/terminal checkpoint invalidation, namespace isolation, and R2 archival |
| Numerical/runtime validity | Not evaluated; no target trace, XLA compile, or Kalman execution ran |
| Scientific interpretation | Not evaluated; no evidence may cross from the engineering ledger |

## Negative Evidence Classification

R2 failed before fixture, selected-method construction, concrete-function
tracing, XLA, or Kalman target work. That invalidated the R2 harness execution,
not the Kalman implementation, CPU-XLA target, memory-repair hypothesis, or
performance-repair hypothesis. The observed failure triggered exactly this R3
harness repair.

The local repair-loop failures described above invalidated particular test
runs until repaired. They did not invalidate the target, data, math, or
scientific direction. The later repair phases remain justified because no true
continuation veto fired.

## Post-Run Red Team

- Strongest alternative explanation: unit and subprocess integration tests may
  miss a real filesystem/process interaction in the proposal-bound Gate B run.
- Result that would overturn this close: any fresh-authority validation failure,
  accepted tampered input, budget mutation before deterministic preflight,
  spawn after parent-detected drift, target work after child-entry drift,
  terminal outcome falsification, R2/protected drift, or surviving worker.
- Weakest evidence: filesystem checkpointing cannot detect an ABA mutation that
  is introduced and restored entirely between checkpoints. The claim assumes
  cooperative repository writers and covers only drift present at a checked
  boundary.
- Reviewer boundary: Claude review is advisory and cannot authorize runtime,
  human, product/default, funding, model-file, release, or scientific claims.

## Handoff

Local repair closure conditions are satisfied. The next exact steps are:

1. Claude Opus/max read-only review of this result at the fixed final path;
2. visible repair and focused rerun for any `REVISE`, maximum five material
   rounds for one blocker;
3. freeze the agreeing result review and its hashes in the runtime contract;
4. construct the fresh R3 proposal only from an absent R3 work root;
5. review the proposal, create detached attestation, validate authority, and
   complete the final skeptical runtime audit;
6. only then run the exact proposal-bound Gate B command once.

Until those steps pass, the exact handoff state is
`LOCAL_R3_REPAIR_PASSED_RESULT_REVIEW_ROUND2_PENDING_NO_TARGET_AUTHORITY`.
