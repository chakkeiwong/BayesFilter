# Phase 6 Gate B Runtime-Harness Repair Result

Date: 2026-07-12

Status: `LOCAL_REPAIR_GATE_PASSED_RESULT_REVIEW_PENDING_NO_FRESH_TARGET_AUTHORITY`

Parent repair subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-harness-repair-subplan-2026-07-12.md`.

Parent repair subplan SHA-256:
`575cf2fc7661bef2be6282ca57570d27bbf4490f2b01ecaad9e5cbb4c5efb004`.

## Result

The no-target Gate B runtime-harness repair passed its local closure gate. The
trace child now preserves the exact reviewed interpreter and script tokens from
`sys.orig_argv`. The Phase 6 process schema now accepts return code zero only
for `failed:invalid_child_evidence`, while ordinary failed records remain
positive-return-only. A real supervisor-loop integration test proved that
zero-return invalid evidence produces one durable terminal transition, does not
enter exception recovery, does not relaunch, and prunes all remaining rows as
common invalidity.

After the first bounded result review, Codex's mandatory skeptical audit found
that `run_phase6_pilot` validated the proposal's reviewed command but did not
independently require the live supervisor process to have that exact argv. It
also did not require the prelaunch work root to contain only the proposal-bound
import-discovery file. The first review is therefore superseded and is not an
authority input. The repaired entrypoint now validates its own exact
`sys.orig_argv` against the proposal command, refuses alternate authority/output
tokens, requires a strictly absent work root before proposal construction, and
requires exactly one bound discovery file with no trace, pilot, budget, lease,
or child state before runtime.

The first `r1` authority lineage remains immutable invalid-harness evidence. Its
archive manifest was reconstructed after the repair and retained byte-identical
SHA-256
`caacd7144a0e6b7767487d7cc3a48145702983487ac1ab6885f5f97ba2f9607a`.
All active Gate B child, import-discovery, budget-state, proposal, attestation,
trace, and pilot paths now use a disjoint `r2` namespace. No `r2` proposal,
attestation, trace ledger, pilot ledger, budget state, lease, child artifact, or
import-discovery byte exists at this close point.

This result closes only the local engineering repair gate. It does not authorize
or report a target trace, XLA compile, Kalman execution, CPU-XLA outcome, method
outcome, or scientific claim. Bounded read-only review of this exact result,
fresh proposal construction and review, detached attestation, strict authority
validation, and a final skeptical runtime audit remain required before Gate B.

## Implemented Repairs

| Repair | Implementation | Focused evidence |
| --- | --- | --- |
| Exact child invocation | `scripts/benchmark_kalman_qr_parameter_count_scaling.py` validates and copies `sys.orig_argv`; no resolved-path equivalence is accepted for interpreter provenance | Real interpreter-token subprocess, negative mutations, and source-shape assertions in `tests/test_kalman_qr_phase6_gateb_runtime_repair.py` |
| Truthful zero-return invalid evidence | `scripts/kalman_qr_benchmark_contract.py` makes process validation reason-aware only for `failed:invalid_child_evidence`; positive ordinary failures remain unchanged | Positive/negative process-schema tests plus real supervisor-loop integration |
| Single terminal transition | `phase6_execute_ledger` validates normal child semantics first, then commits one common-invalidity transition when evidence is invalid | One launch; events begin `running, failed`; no recovery callback; all later records `not_launched:common_invalidity` |
| Immutable `r1`, fresh `r2` | Explicit historical and active constants separate old plan/artifacts/work root from new reviewed repair plan and `r2` paths | Archive revalidation, path-disjointness tests, stale-`r2` preservation test, protected hashes |
| Four-input Gate B authority | The Gate B proposal must bind the exact `r1` archive, this result, final agreeing plan review, and final agreeing implementation/result review | Independent digest mutation for each input and review-verdict/strength mutations |
| Closed no-target authority modes | Added exact-CLI `--phase6-create-attestation`, `--review-path`, and `--phase6-validate-authority` modes | Synthetic governance-only tests cover first-write-only output, strength binding, live-worker veto, and runtime-artifact absence |
| Exact live supervisor authority | `run_phase6_pilot` copies validated `sys.orig_argv`, compares it byte-for-byte with both the closed command builder and proposal command, and rejects alternate output/authority tokens | Exact positive case plus missing, interpreter, script, and output-token mutations |
| Strict prelaunch namespace | Proposal construction requires the `r2` work root absent; authority validation and runtime require exactly the proposal-bound discovery file and no other `r2` state | Missing root, wrong discovery, extra child file, trace output, and live-worker mutations |

No `bayesfilter/linear/*.py` algorithm source was edited by this repair.

## Evidence Contract Assessment

| Field | Observed result |
| --- | --- |
| Engineering question | Passed locally: exact reviewed argv can be preserved and truthful zero-return invalid evidence can close once without reusing or mutating `r1`. |
| Exact baseline | Frozen `r1` proposal, attestation, trace, child, journal, budget state/lease, archive manifest, reviewed repair plan, and protected algorithm hashes. |
| Primary criterion | Compile, 32 focused tests, exact consolidated GPU-hidden Gate A suite, static checks, archive revalidation, protected hashes, no-worker check, and strict `r2` runtime-artifact absence passed. |
| Hard vetoes | None fired locally. No exact-equality weakening, ordinary zero-return failure acceptance, `r1` drift, stale-byte overwrite, double transition, relaunch, protected drift, or live target worker was observed. |
| Explanatory only | Test wall times, warning count, line counts, and synthetic executor event count. |
| Not concluded | No Kalman correctness, trace structural pass, CPU-XLA viability, method ranking, speed/scalability, GPU readiness, HMC/posterior correctness, default readiness, production readiness, or scientific validity. |

## Local Checks

### Compilation

The exact reviewed compile command exited zero and wrote an empty log:

- log: `/tmp/kalman_qr_phase6_gateb_runtime_repair/py_compile.log`;
- SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

### Focused Repair Suite

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_phase6_gateb_runtime_repair.py
```

Result: `32 passed in 8.17s`.

- log: `/tmp/kalman_qr_phase6_gateb_runtime_repair/focused_pytest.log`;
- SHA-256:
  `f9ad612eb493c3096956523a48b88c7ab3a8f2e4b20de9ec757cb91806250ba3`.

The suite contains no target TensorFlow trace, XLA compile, or Kalman filter
run. Its only real subprocess is a tiny standard-library argv probe.

### Consolidated GPU-Hidden Gate A Suite

The exact consolidated command from the repair subplan exited zero with the
declared CPU-XLA node deselected:

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

Result: `379 passed, 1 deselected, 6634 warnings in 197.15s`.

- log: `/tmp/kalman_qr_phase6_gateb_runtime_repair/consolidated_pytest.log`;
- SHA-256:
  `165fda93b28986a8c404c909da91a5dd51eb007f6bd089026cb3f60e6822428c`.

The warnings are Gast/AutoGraph deprecations already present in the preceding
Gate A/Phase 5 evidence. They are not promotion evidence and did not hide a
failed node. GPU was deliberately hidden before test imports. The single
deselected node is the target CPU-XLA runtime check forbidden at this gate.

### Static, Hash, And Namespace Checks

- scoped `git diff --check`: passed;
- trailing-whitespace scan: empty, expected `rg` exit 1;
- exact argv source-shape checks: passed;
- `r1` archive reconstruction: exited zero and retained identical bytes;
- protected algorithm hashes: unchanged;
- reviewed repair subplan/review hashes: unchanged;
- exact process-table scan: no target supervisor or child;
- repository `r2` proposal/attestation/trace/pilot paths: absent;
- `/tmp/kalman_qr_phase6_cpu_xla_gateb_r2`: absent after removal of empty
  test-created directories.

## Protected State

| Path | Closing SHA-256 |
| --- | --- |
| `bayesfilter/linear/kalman_qr_tf.py` | `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b` |
| `bayesfilter/linear/kalman_qr_derivatives_tf.py` | `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57` |
| `bayesfilter/linear/qr_factor_tf.py` | `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401` |
| reviewed repair subplan | `575cf2fc7661bef2be6282ca57570d27bbf4490f2b01ecaad9e5cbb4c5efb004` |
| agreeing plan review round 3 | `39f41d15394e721a7a481d36abf65ac340dea656f6a96181f40e1577a0e1b9f5` |
| immutable `r1` archive | `caacd7144a0e6b7767487d7cc3a48145702983487ac1ab6885f5f97ba2f9607a` |

## Closing Implementation Hashes

| Path | SHA-256 |
| --- | --- |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `d299c5ffed10a30c93a0e8685b47bfe9433efb66cf722fe64b53bf4ef39651fc` |
| `scripts/kalman_qr_benchmark_contract.py` | `1f81eb7453ec58a4b0b0227897fb9252fed2ffdacf6fa0d8c0392e5f056f783b` |
| `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py` | `197bd4357e42aa308dfd44c99c47fbb948534593c4d141a9c4ba073a45c1c07a` |
| `tests/test_kalman_qr_phase6_cpu_xla_gates.py` | `7251aa289ff2d1041a51d5d79f2c478686b000f31e213fc35b6af3063bc22f20` |
| `tests/test_kalman_qr_phase6_gatea_runtime_controls.py` | `1d29f3063a47967e542fd3c97730caf4c0eadd258ecd6f3feeb3dac4263298d0` |
| `tests/test_kalman_qr_phase6_import_discovery_cli.py` | `0f8a97c0f4e20cd1958d83ae9b9e5dbf36972efc6e6626c7c2902d8dd104b830` |
| `tests/test_kalman_qr_phase6_gateb_runtime_repair.py` | `d035c68471a7e2493ef9c9c2afce07920ff6d9c85ac7d138c63bc6964677f51c` |

The shared worktree remains dirty and these lane files are opening-untracked in
the current Git index. Other-lane files were not modified, reverted, or used as
this lane's evidence.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Interpreter | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, CPython 3.13.13 |
| Environment | `tfgpu`; TensorFlow 2.20.0; TensorFlow Probability 0.25.0 |
| CPU/GPU status | Deliberate GPU-hidden local checks; no target GPU or CPU-XLA run |
| JIT/XLA | Target JIT test deselected; no XLA compile authorized or run |
| Data/seeds | N/A; deterministic contract tests and synthetic process/evidence fixtures |
| Wall time | Focused 8.17s; consolidated 197.15s; target runtime N/A |
| Output artifacts | Three logs above and immutable `r1` archive; no `r2` runtime artifact |
| Plan | Reviewed repair subplan named above |
| Result | This file |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the local harness repair for bounded review | Passed | No local repair veto fired | Real target behavior remains unmeasured; reviewer may find a boundary or artifact-contract defect | Bounded read-only review of this exact result; repair if needed; only then construct/review fresh `r2` authority | No method/backend/CPU-XLA/default/production/scientific claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for local engineering repair only |
| Statistically supported ranking | None; no stochastic or method comparison ran |
| Descriptive-only differences | Test wall times, warning counts, file sizes, and event counts |
| Default readiness | Not evaluated |
| Next evidence needed | Agreeing result review, exact fresh proposal/review/attestation, final skeptical audit, then separately authorized Gate B trace/pilot evidence |

## Evidence Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Local repair gate passed for exact argv, terminal semantics, namespace isolation, and authority plumbing |
| Numerical/runtime validity | Not evaluated; no target trace, XLA compile, or Kalman execution ran |
| Scientific interpretation | Not evaluated; no claim may cross from the engineering ledger |

## Negative And Repair Evidence

The first `r1` child internally reached `passed`, but its exact invocation did
not match the reviewed schedule and the supervisor could not truthfully encode
return-code-zero invalid evidence. That is common-invalidity harness evidence,
not evidence against either Kalman method or CPU XLA. The local repair loop also
found stale synthetic tests that paired `invalid_child_evidence` with positive
return codes or assumed empty Gate B authority inputs. Those fixtures were
updated to the reviewed truthful semantics; production validators were not
weakened.

Result review round 1 returned a bounded substitute `AGREE`, but Codex did not
accept it as closure because the subsequent skeptical audit found the live
supervisor argv/prelaunch namespace gap above. The result and implementation
were visibly repaired, focused and consolidated gates were rerun, and only a
fresh round-2 review of the new bytes may become an authority input.

## Post-Run Red Team

- Strongest alternative explanation: unit and synthetic integration tests may
  miss a real child-envelope or process-lifecycle interaction. The separately
  authorized Gate B run is the discriminating evidence and must remain behind
  fresh authority.
- Result that would overturn this conclusion: any exact interpreter-token
  drift, zero-return ordinary failure acceptance, double terminal transition,
  relaunch after invalid evidence, `r1` byte drift, stale `r2` reuse, or an
  invalid proposal/attestation passing strict validation.
- Weakest evidence: the integration child is synthetic and its evidence blobs
  are intentionally invalid. It proves state-machine behavior, not target
  TensorFlow behavior.
- Reviewer boundary: an agreeing review proves only that the bound plan/result
  bytes survived bounded scrutiny. It cannot authorize execution or lower any
  human, runtime, product, funding, model-file, or scientific boundary.

## Handoff

Local repair handoff conditions are satisfied. Gate B remains blocked until all
of the following hold:

1. bounded read-only implementation/result review of this exact file returns
   `VERDICT: AGREE` with its actual strength recorded;
2. this result and that review are frozen;
3. the exact no-target `r2` proposal is constructed and strictly validates all
   four authority inputs, source/runtime identities, schedules, paths, and the
   unchanged 3045-second budget;
4. bounded review of exactly that proposal returns `VERDICT: AGREE`;
5. detached attestation and exact authority validation pass;
6. a final skeptical runtime audit records exact `PASS` with no worker, no
   stale runtime byte, and all frozen hashes current.

Until then the exact handoff state is
`LOCAL_REPAIR_GATE_PASSED_RESULT_REVIEW_PENDING_NO_FRESH_TARGET_AUTHORITY`.
