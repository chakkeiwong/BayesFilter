# q=20 72-core process-parallel canary result

Date: 2026-09-03  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`  
Artifact root: `docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03/canary/attempt-02/`  
Status: `PASS_CANARY_FULL_RUN_AUTHORIZED`

## Verdict

The composite canary passed.  It exercised the current q=20 target and fixed
transport verification path in isolated CPU processes at all three requested
barriers: `8x4` screen workers, `2x8` selection workers, and `6x4` scope
finalization workers.  A fixed-seed serial reference and a process worker
returned matching samples, target log probabilities, acceptance traces, and
scores within `1e-9` for samples/values and `1e-8` for scores.

This authorizes the full staged diagnostic in the controlling plan.  It does
not authorize Phase 9B, posterior claims, or a CPU default.

## Evidence

| Check | Result |
|---|---|
| Host affinity | 256 logical CPUs visible; 72 required worker cores available |
| Screen barrier | 8 workers, four CPUs each, 8/8 tasks durable, zero worker failures |
| Selection barrier | 2 workers, eight CPUs each, 2/2 tasks durable, zero worker failures |
| Scope-finalize barrier | 6 workers, four CPUs each, 6/6 tasks durable, zero worker failures |
| CUDA visibility | `CUDA_VISIBLE_DEVICES=-1`; TensorFlow GPU list empty in every child |
| Compilation | `jit_compile=true` in every child; strict `tensorflow_eigh` target signature matched |
| Serial/process parity | Samples, values, log-acceptance, and scores passed declared tolerances |
| Wall time | `581.3556926490273` seconds, below the `1200` second canary cap |
| Maximum worker RSS | approximately `1.20 GB` in the short fixture calls; descriptive only |

The complete machine-readable summary is
`.../canary/attempt-02/canary_summary.json`.  The fixture checkpoints are the
pre-existing failed-replay charts and are explicitly mechanics-only; they are
not reused as fresh tuning evidence.

## Repair history

Attempt 01 stopped before the barriers because the first absolute-path launch
did not add the repository root to `sys.path`, and the parent also checked a
healthy worker's return code before joining it.  No numerical result was used.
The localized repair inserted the source root before repository imports and
joined workers before interpreting return codes.  Focused topology tests,
Python compilation, and whitespace checks passed before the fresh attempt-02
launch.  Attempt 01 remains preserved under the same artifact root.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Admit full staged run | topology, identity, finite/status, and fixed-seed parity | no canary veto | full-call CPU time and contention | prepare fresh charts, then run all barriers | no convergence or posterior claim |
| Keep CPU route diagnostic | repository GPU-default policy | CPU exception explicitly labeled | CPU/GPU timing relation is unmeasured | leave GPU canonical lane unchanged | no CPU default or GPU speedup |

## Inference status

| Inference class | Status |
|---|---|
| Hard veto screen | Passed for canary engineering checks |
| Statistically supported ranking | None |
| Descriptive-only differences | Short-call wall time, RSS, and acceptance |
| Default readiness | Not assessed; GPU default unchanged |
| Next evidence needed | Fresh chart preparation and complete six-scope staged diagnostic |

