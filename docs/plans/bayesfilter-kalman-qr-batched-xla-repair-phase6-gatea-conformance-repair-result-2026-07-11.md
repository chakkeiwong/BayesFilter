# Phase 6 Gate A Conformance Repair Result

Date: 2026-07-12

Status: `LOCAL_GATE_PASSED_CLAUDE_REVIEW_PENDING_GATE_B_RUNTIME_BLOCKED`

Parent repair subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatea-conformance-repair-subplan-2026-07-11.md`.

Parent Phase 6 subplan SHA-256:
`b7b653d8febfa341dd2e8b53e8c274246eb49b6afcc59e4bca27126d3b33769b`.

## Result

Gate A's local conformance gate passed. The Phase 6 supervisor now fails closed
across reviewed schedule authority, shared budget leases, process-group
lifecycle, interruption recovery, immutable pilot import, exact `P=150`
routing, common-invalidity pruning, and terminal correspondence. No target
trace, XLA, scalar-reference, GPU, HLO, or benchmark workload was launched.

This result does not itself close Gate A. A bounded Claude Opus max-effort
read-only review of this result must return exact `VERDICT: AGREE`; any
material finding returns to this same repair lane. Gate B runtime remains
blocked behind its dedicated subplan, exact proposal, separate proposal review,
and detached attestation.

## Implemented Repairs

| Repair | Source anchor | Focused evidence |
| --- | --- | --- |
| Deferred SIGTERM covers spawn, durable `running`, completion, and terminal persistence; cleanup also runs after identity/callback exceptions | `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py:793` | Runtime controls at lines 1017, 1056, 1207, 1244, and 1291 |
| Shared budgets bind boot identity, immutable deadline, monotonic observations, command order, and an exclusive `flock` lease | `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py:282` | Runtime controls at lines 582, 680, 752, 767, and 791 |
| Lease acquisition releases its descriptor and lock after every post-`flock` exception | `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py:292` | `tests/test_kalman_qr_phase6_gatea_runtime_controls.py:767` |
| Absolute lifecycle deadlines include prelaunch, spawn, callback, execution, cleanup, and persistence; valid prelaunch expiry prunes durably | `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py:2095` | Runtime controls at lines 1207 and 1586 |
| Recovery preserves malformed-present bytes and refuses stale deterministic child files | `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py:685` | Runtime controls at lines 1092 and 1191 |
| Recovery signals only a live leader with exact PID/PGID/start-token identity; dead leader plus live group is an unauthenticated hard stop | `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py:685` | `tests/test_kalman_qr_phase6_cpu_xla_gates.py:2653` and adjacent mismatch tests |
| Gate C preflights the exact full imported pilot roster, preserves imported provenance, detects imported common invalidity before spawn, and recovers stale owned `running` state before rejecting inconsistent prior work | `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py:2095` | Runtime controls at lines 459, 1456, and 1522; Phase 6 gate test at line 2190 |
| Routing schema `p150_routing.v3` stores an immutable prelaunch ledger-prefix snapshot and reconstructs it from final events/records | `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py:1390` | Runtime controls at line 1338, including dependency drift and snapshot tampering |
| Terminal overlays have a closed structural schema and exact final-ledger correspondence; later common invalidity globally invalidates earlier historical eligibility without rewriting it | `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py:1515` | Fully valid zero-spawn integration at `tests/test_kalman_qr_phase6_gatea_runtime_controls.py:1456` |
| Gate B trace/pilot and Gate C scalar ledgers can honestly encode executor-declared common-invalidity and shared-budget pruning states | `scripts/kalman_qr_benchmark_contract.py:2670` | Fully valid common-invalidity integration and prelaunch-expiry test |

## Evidence Contract Assessment

| Field | Observed result |
| --- | --- |
| Engineering question | Passed locally: reviewed Phase 6 authority, lifecycle, routing, import, and evidence state machines fail closed under focused mutations and harmless subprocess simulations. |
| Exact baseline | Parent Phase 6 subplan at the SHA-256 above and opening ledger SHA-256 `9261e0c560ede29dc6893e0ffe3769cd762b38f3dd651af6dfcfa2f90dce1911`. |
| Primary criterion | Local compile, consolidated focused suite, static checks, protected hashes, exact opening-ledger parse, and no-worker check passed. Claude review remains pending. |
| Hard vetoes | None fired locally. No target command launched; no owned process group survived; no authority, import, routing, or terminal-evidence mutation passed. |
| Explanatory only | Test duration, warning count, file size, and implementation line count. |
| Not concluded | No target numerical correctness, trace stability, CPU-XLA viability, GPU readiness, speed/scalability, method ranking, HMC/posterior correctness, default readiness, production readiness, or scientific validity. |

## Local Checks

### Compilation

The following passed with zero output:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/kalman_qr_benchmark_contract.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py \
  tests/test_kalman_qr_phase6_gatea_runtime_controls.py \
  tests/test_kalman_qr_phase6_import_discovery_cli.py \
  tests/test_kalman_qr_measurement_boundaries.py
```

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
  --deselect=tests/test_kalman_qr_batch_native_autodiff.py::test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value
```

Result: `347 passed, 1 deselected, 6634 warnings in 193.07s`.

The warnings are Gast/AutoGraph deprecations already observed in Phase 5. They
are not promotion evidence and did not hide a failed test. GPU was intentionally
hidden before TensorFlow import. The deselected test is the explicit CPU-XLA
runtime test, which is forbidden in Gate A.

Additional focused results before consolidation were `26 passed in 104.70s`
for runtime controls and `197 passed in 59.12s` for the contract, Phase 6 gate,
import-discovery, and measurement modules.

### Static And Boundary Checks

- scoped `git diff --check`: passed;
- trailing-whitespace scan over all Gate A touched paths: empty;
- exact process-table filter for the Phase 6 supervisor/work directory: empty;
- stale routing `v2` and unimplemented descendant-token wording scan: empty;
- strict opening-ledger parse: two headers plus 144 data entries;
- opening-ledger SHA-256:
  `9261e0c560ede29dc6893e0ffe3769cd762b38f3dd651af6dfcfa2f90dce1911`.

The opening ledger omitted 36 historical JSON counterparts. This is a known
coverage limitation: it is not a complete repository or complete historical
artifact inventory, and no such claim is made.

## Protected State

| Path | Opening/closing SHA-256 |
| --- | --- |
| `bayesfilter/linear/kalman_qr_tf.py` | `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b` |
| `bayesfilter/linear/kalman_qr_derivatives_tf.py` | `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57` |
| `bayesfilter/linear/qr_factor_tf.py` | `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401` |
| parent Phase 6 subplan | `b7b653d8febfa341dd2e8b53e8c274246eb49b6afcc59e4bca27126d3b33769b` |

No protected algorithm or parent-plan bytes changed in Gate A.

## Closing Artifact Hashes

| Path | SHA-256 |
| --- | --- |
| `scripts/kalman_qr_benchmark_contract.py` | `730e4374ca046dcf7cc1aa76af69d1c7942c08eac17120006577ec37ca1ec06c` |
| `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py` | `5de6f8e276abf5e9312207fb5de748d8a55c196547371b835fa764ae48a15de2` |
| `tests/test_kalman_qr_benchmark_contract.py` | `402c21f8c09d1befacd6a7bcc7189b8adfb28217821d0745f701e4a7b54240bf` |
| `tests/test_kalman_qr_phase6_cpu_xla_gates.py` | `5d26b60c5edd7996a1eee334cad6ac606e15a52f6c7df9af0c4ea448171c7268` |
| `tests/test_kalman_qr_phase6_gatea_runtime_controls.py` | `d0485a4ed1ae2305a75f15bb8e600ea48e43a22037af5e4d9659f8ce3d0798b6` |
| `tests/test_kalman_qr_phase6_import_discovery_cli.py` | `e9b4dada5902835105cab7e6ff7006c6c5ce10197eef2166842a2d83f3190715` |
| `tests/test_kalman_qr_measurement_boundaries.py` | `8ff8014691b539796cd19e39296d7fbbc33d5ab073016888ea95385543ce4534` |
| Gate A repair subplan | `bae0bd1d529ed646e76a56a336a40f43b170e2487b1d5f0fd3266eb67e18ee9f` |

The shared worktree is dirty and these paths are opening-untracked in the
current Git index. Other-lane files were not modified, reverted, or interpreted
as this lane's evidence.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Interpreter | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, CPython 3.13.13 |
| Device | Deliberate GPU-hidden local checks; no target runtime |
| GPU setting | `CUDA_VISIBLE_DEVICES=-1` before test imports |
| Random seeds/data | N/A; deterministic contract tests and harmless process simulations |
| Target-workload wall time | N/A; forbidden and not run |
| Plan | Gate A repair subplan named above |
| Result | This file |
| Runtime artifacts | Temporary pytest/process files only; no Phase 6 target artifact |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept Gate A local implementation evidence | Passed | No local conformance veto fired | External bounded review has not yet agreed; target behavior remains unmeasured | Claude Opus max-effort read-only review of this exact result, repair if needed, then review the dedicated Gate B subplan | No target/XLA/GPU/scientific claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for Gate A engineering simulations and validators only |
| Statistically supported ranking | None; no stochastic method comparison ran |
| Descriptive-only differences | Test times and warning counts only |
| Default readiness | Not evaluated |
| Next evidence needed | Converged Gate A review, reviewed exact Gate B proposal, detached attestation, then target trace census before pilot XLA |

## Handoff

Local Gate A handoff conditions are satisfied. Remaining conditions are:

1. bounded Claude review of this exact result returns `VERDICT: AGREE` within
   five material rounds;
2. the dedicated Gate B subplan converges under a separate bounded review;
3. exact Gate B proposal creation and validation complete without target work;
4. a proposal-specific review record ends with `VERDICT: AGREE` and a detached
   `claude_opus_max` attestation validates;
5. only then may the exact Gate B command be launched.

Until all five hold, status remains
`LOCAL_GATE_PASSED_CLAUDE_REVIEW_PENDING_GATE_B_RUNTIME_BLOCKED`.
