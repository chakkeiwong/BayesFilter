# BayesFilter Fixed-Metric Grid Process-Parallel Option Result

Date: 2026-07-20

Status: `COMPLETE`

Plan: `docs/plans/bayesfilter-fixed-metric-grid-process-parallel-option-plan-2026-07-20.md`

## Outcome

BayesFilter now provides an opt-in `process_parallel` execution mode for the
fixed-metric HMC `L`-grid search. Serial remains the default. Each spawned child
imports an application `module:factory`, constructs its target-specific tune
and screen callbacks, and executes one complete candidate. Round 0 and optional
refinement are separate barriers.

The public API now includes `FixedMetricGridExecutionConfig`,
`FixedMetricCandidateRunners`, `run_fixed_metric_candidate`, and
`run_fixed_metric_grid_search`. MacroFinance's existing six-process CCMA worker
now calls the public single-candidate function rather than BayesFilter's private
`_run_candidate` helper.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Retain serial default and expose opt-in spawned processes | Passed: deterministic serial and process candidate mechanics match for Round 0 and refinement | No fork, environment, factory-scope, seed, ordering, or failure-scope veto fired | Real-target throughput and capacity remain application/device dependent | Applications may opt in with an importable child factory and explicit environment | No speedup, convergence, mass-adequacy, retained-sampling, or scientific claim |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Result |
| --- | --- |
| Engineering correctness | Real `spawn` tests passed for deterministic equality, reversed order, deterministic completion callbacks, local rejection, shared bootstrap failure, target/resource propagation, and fail-closed configuration. Public imports and compile checks passed. |
| Numerical/sampler validity | Candidate payloads, seeds, tuned steps, screens, extensions, survivors, and refinement agree with the deterministic serial oracle. The fixture does not run HMC. |
| Scientific interpretation | Not applicable. All tuning draws remain discarded and no retained or scientific run was launched. |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Passed for scheduling/API scope. Invalid factory/bootstrap behavior produces shared invalidity; target and resource vetoes propagate. |
| Statistically supported ranking | None attempted. |
| Descriptive-only differences | Prior CCMA timing evidence is outside this API result and is not used as the pass criterion. |
| Default-readiness | Serial remains the default; process mode is an explicit option. |
| Next evidence needed | A target-specific application run is needed only to establish device capacity or speedup for that application. |

## Verification

CPU-only checks intentionally hid accelerators with `CUDA_VISIBLE_DEVICES=-1`.
The BayesFilter focused suite passed as bounded groups because one combined
spawn-heavy invocation was terminated by the command envelope after nine dots
without a pytest summary. The bounded commands produced valid terminal results:

- non-spawn/config/serial group: `23 passed, 9 deselected`;
- real-spawn semantic and failure-scope groups: `9 passed` across bounded
  commands;
- total focused BayesFilter cases: `32 passed`;
- MacroFinance focused integration: `56 passed`;
- `py_compile`, public import smoke, and `git diff --check`: passed;
- `ruff`: unavailable in the environment.

TensorFlow Probability emitted two existing `distutils` deprecation warnings;
no focused check failed.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit at verification | `3250e0cb708eef7f8cbeafb62b2fd27741e3554f` (dirty shared worktree preserved) |
| Python | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` |
| Environment | `tfgpu`; `PYTHONPYCACHEPREFIX` under `/tmp` |
| CPU/GPU | CPU-only; `CUDA_VISIBLE_DEVICES=-1`; no GPU initialized for a scientific run |
| Random seeds | Deterministic fixture uses the grid config's candidate-derived seeds; no stochastic HMC executed |
| Wall time | Final-code BayesFilter bounded groups approximately 67 seconds total; MacroFinance integration 9.99 seconds |
| Output artifacts | This result note and pytest terminal results; no raw samples or target states |
| Plan file | `docs/plans/bayesfilter-fixed-metric-grid-process-parallel-option-plan-2026-07-20.md` |
| Result file | `docs/plans/bayesfilter-fixed-metric-grid-process-parallel-option-result-2026-07-20.md` |

## Post-Run Red Team

The strongest alternative explanation is that equality holds only because the
callbacks are deterministic and cheap. That does not weaken the scheduling
contract result, but it means this work cannot establish real-target speedup,
GPU capacity, or absence of TensorFlow initialization costs. A target-specific
run showing changed seeds, candidate payloads, failure scope, or survivor
semantics would overturn the scheduling-equivalence conclusion. The weakest
evidence is target-specific runtime performance, which was deliberately outside
this API phase.

Applications must avoid target/TensorFlow construction at module import time.
GPU child factories must set and verify TensorFlow memory growth before target
construction even though BayesFilter also requires
`TF_FORCE_GPU_ALLOW_GROWTH=true` in the declared worker environment.
