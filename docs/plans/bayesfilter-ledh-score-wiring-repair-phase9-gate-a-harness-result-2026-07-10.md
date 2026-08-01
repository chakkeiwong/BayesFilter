# Phase 9 Gate A Result: Compact Score GPU/XLA Evidence Harness

Date: 2026-07-10

Status: `PASSED_CPU_HIDDEN_REVIEW_PENDING_GPU_EXECUTION_BLOCKED`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| The shared five-row nonlinear compact-score harness is ready for bounded review of trusted GPU Gate B. | Passed syntax, parser, artifact-contract, adversarial, tensor-adapter parity, cross-model, shared-contract, and model-specific CPU-hidden checks. | No historical-route, target, transport-identity, precision, seed-set, source-hash, device-provenance, reset-memory-schema, FD-label-forgery, or aggregate-budget test veto remains. | No nonlinear compact kernel has compiled or executed under GPU XLA through this harness. Full-row memory and same-scalar FD are not checked. | Review this result, the runner/tests, and the exact execution manifest. Run Gate B only if the fresh bounded substitute review returns `VERDICT: AGREE`. | GPU viability, score admission, full-row numerical validity, runtime superiority, posterior correctness, HMC readiness, exact nonlinear likelihood correctness, leaderboard completion, or scientific superiority. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the harness emit and validate recoverable singleton-seed XLA score/FD artifacts without changing the compact score equations or admitting incomplete evidence? |
| Exact comparator | Existing row-specific eager compact score and value-only routes at the same tiny fixed inputs. |
| Primary criterion | Passed. The five tensor-only adapters reproduce their existing eager routes, the runner hard-codes XLA JIT, and offline admission requires exact full-row singleton shards plus recomputed FD and memory gates. |
| Promotion vetoes | Exercised adversarially: forged JIT/trust/device/precision/route/target/source/transport/budget/seed/memory/FD fields, incomplete seed sets, duplicate seeds, score-reference mismatch, and over-budget aggregate. |
| Explanatory only | Test wall times, source hashes, and CPU-hidden tiny numerical values. |
| Artifact | This result, `docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py`, `tests/highdim/test_ledh_compact_score_gpu_xla_harness.py`, and the exact execution manifest. |

## Claimed And Computed Quantities

| Item | Verdict |
| --- | --- |
| Claimed target | Harness wiring for each row's compact forward-sensitivity score of the admitted realized finite-`N` LEDH observed-data log-likelihood estimator. |
| Quantity actually computed | CPU-hidden tiny eager tensor-adapter outputs plus parser, source, validator, aggregation, and schedule tests. |
| Relationship | The adapter outputs equal the pre-existing tiny eager score/value outputs within `atol=rtol=1e-10`. This establishes extraction parity for those fixtures, not GPU-XLA compilation or full-row correctness. |
| Support | Final combined test run and model-specific shards listed below. |
| Unproved | GPU compilation, GPU placement, reset-memory values, prefix/full-row finiteness, per-seed FD, aggregate FD, and full admission. |

## Implementation Result

Added the shared runner:

- `docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py`.

Added focused tests:

- `tests/highdim/test_ledh_compact_score_gpu_xla_harness.py`.

Added private tensor-only adapters to the five nonlinear score modules:

- `_prepare_compact_xla_inputs` freezes row data and fixed randomness outside the compiled call;
- `_compact_score_tensor_outputs` passes tensor `theta` through the existing compact score equations;
- `_value_tensor_outputs` passes tensor `theta` through the existing value-only same-scalar equations.

The runner provides:

- `tf.function(..., jit_compile=True)` score and value kernels with no non-JIT production fallback;
- one seed per runtime shard and separate `score-only` and `fd-only` processes;
- atomic `started`, `initialized`, terminal `completed`, blocked, or `failed` JSON;
- reset TensorFlow score-memory before/after snapshots;
- physical/logical/output-device, trust, XLA, precision, environment, git, source, target, theta, transport, seed, shape, command, output, timing, plan, and result provenance;
- offline exact-five-seed aggregation with no monolithic batch-memory or runtime claim;
- source-value and score-reference hashes;
- frozen recursive hashes for all reachable local Python dependencies and the
  reviewed governance/command artifacts;
- deterministic serialized hashes for every prepared tensor leaf, binding
  fixed observations and randomness across score-only and FD-only processes;
- a deterministic, parser-tested `91`-entry exact-command JSON whose literal
  argv is enforced for trusted runtime and aggregate commands;
- recomputation of singleton and aggregate all-coordinate FD pass rather than trust in a declared `status` label;
- the frozen `14000 MiB` maximum per-seed reset score-memory gate.

No target-density equation, parameter transform, compact derivative recurrence,
finite-difference threshold, transport default, public BayesFilter API, or
shared admission schema was changed by Gate A.

## Skeptical Audit Repairs

The pre-execution audit found that the first harness draft did not bind all
transport-defining fields. Post-test red-team inspection found additional
artifact-validity gaps. Verification was held until all were repaired:

- bound transport plan mode, AD mode, gradient mode, annealing scale, and
  convergence threshold;
- bound row, source, theta, chunks, memory budget, runner, plan/result, command,
  environment, git, device, and output identity;
- replaced executable authority from Gate C/D substitution templates with a
  complete frozen command artifact;
- bound separate score/FD processes to identical serialized prepared inputs;
- require exact fixed-seed membership and reject missing, duplicate, or
  unexpected seeds;
- require finite/nonnegative reset-memory snapshots before and after score;
- require singleton objective, likelihood, aggregate score, and per-seed score
  consistency;
- derive FD errors from every finite parameter entry and independently apply
  the frozen absolute-or-relative pass rule;
- require exact FD-to-score content and file-hash binding;
- preserve a terminal blocked artifact when aggregate memory exceeds budget;
- emit an `initialized` progress state and compile/first-call timing fields.

## Local Checks

All commands deliberately set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow
import and `MPLCONFIGDIR=/tmp`. They are wiring evidence only.

| Check | Result |
| --- | --- |
| Final syntax check | Passed. |
| Final focused harness contract before combined rerun | `76 passed, 2 warnings in 20.46s`. |
| Final combined harness, parity, Phase 8 schedule, and shared score contract | `149 passed, 2 warnings in 20.16s` after binding the Gate B authorization-review hash. |
| LGSSM plus fixed-SIR model-specific shard | `39 passed, 2 warnings in 78.14s`. |
| Predator-prey plus actual-SV model-specific shard | `45 passed, 2 warnings in 172.41s`. |
| Generalized-SV plus KSC-SV model-specific shard | `38 passed, 2 warnings in 56.83s`. |
| CPU-hidden `--help` parser check | Exit `0`; confirmed the implemented option surface. CUDA initialization warnings are CPU-hidden sandbox noise and are not GPU-health evidence. |

The final combined run includes all five tensor-adapter equivalence tests. The
three model-specific shards were run before the final validator-only hardening;
no row module changed after those shards.

## Source Artifact Identity

| Row | Frozen source value artifact SHA-256 |
| --- | --- |
| fixed-SIR | `38a7da0ef1f32f96e74d4f62676d823af2fbe1b4267d88dbfa0c39c4156ba9b8` |
| predator-prey | `17eaaf23302fa68e802eef686b167e4b31cc3dba755503f9b74343d2ca29ef45` |
| actual-SV | `3811268078d07e0ac4c2fcd9400af156a5918503e404937d516391ce0f034c16` |
| generalized-SV | `5afb71144576bdb0070080f684b5d5b41f33de77889105b10bcd78e36b77dd77` |
| KSC-SV | `9883721faf8af9fbe96ef75c209f86eda5732aec6ca5e602980d4cf27338b3b6` |

## Three-Ledger Status

| Ledger | Status |
| --- | --- |
| Engineering correctness | Passed for parser, adapter parity, artifact generation/validation, seed sharding, stage separation, and offline aggregation logic. |
| Numerical and runtime validity | Not checked on GPU. CPU-hidden tiny parity is the only numerical evidence. |
| Scientific interpretation | No scientific claim is supported or attempted. |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | No Gate A engineering veto remains after hardening. GPU/runtime vetoes are not checked. |
| Statistically supported ranking | None; no stochastic candidate ranking was run. |
| Descriptive-only differences | Test durations and historical memory values remain descriptive only. |
| Default-readiness | The evidence harness is review-ready; no nonlinear score row is newly admitted. |
| Next evidence needed | Fresh bounded review, then trusted GPU/XLA Gate B score-only and FD-only preflights under the exact manifest. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus the disclosed current dirty worktree. |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`, Python `3.11.14`, TensorFlow `2.19.1`, host `DESKTOP-RF1Q5IJ`. |
| CPU/GPU status | CPU-only checks with GPUs intentionally hidden. No Phase 9 GPU command ran. |
| Data version | Admitted forward-scalar artifacts dated 2026-07-07 with hashes above. |
| Seeds | Tiny execution fixture `81120`; fixed full-row artifact identities `81120..81124` without full execution. |
| Commands | Syntax, combined, model-specific, generator-currentness, and parser commands described above; all future nonlinear commands are frozen in the exact-command JSON. |
| Output artifacts | This result, the execution manifest, and the exact-command JSON; no Phase 9 runtime JSON exists yet. |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md`. |
| Result file | This file. |

## Post-Run Red Team

- Strongest alternative explanation: eager tiny parity and source-level XLA
  wiring may pass while a captured tensor, unsupported operation, device
  placement, or tensor lifetime fails under actual GPU XLA.
- Result that would overturn this decision: Gate B fails to emit a terminal
  artifact, compiles only without XLA, places outputs off GPU, changes a frozen
  target/transport field, or exposes a validator acceptance path for forged
  evidence.
- Weakest evidence: no compiled nonlinear GPU call has run through the shared
  harness.
- Candidate versus direction: a Gate B row failure rejects that current row
  execution until repaired; it does not reject the compact-score recurrence or
  unrelated rows unless shared harness logic is invalidated.

## Gate Boundary

GPU execution remains blocked. The next action is a fresh bounded local
substitute review of this result, the exact execution manifest, runner, tests,
and tensor-only adapters. Only `VERDICT: AGREE` authorizes the trusted GPU
preflight and Gate B commands. Gate C/D remain conditional on their explicit
predecessor stop decisions.
