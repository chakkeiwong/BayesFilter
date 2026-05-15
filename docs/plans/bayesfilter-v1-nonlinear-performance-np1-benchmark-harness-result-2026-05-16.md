# BayesFilter V1 Nonlinear Performance NP1 Benchmark Harness Result

## Date

2026-05-16

## Governing artifacts

- Master program: `docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md`
- NP0 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np0-inventory-result-2026-05-16.md`
- NP1 plan: `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-plan-2026-05-15.md`

## Phase purpose

Upgrade the nonlinear benchmark harness to emit NP1-grade CPU-only manifest and row schema evidence without changing production BayesFilter modules or making GPU/XLA/default-policy claims.

## Skeptical plan audit

Audit target: whether a bounded worker can produce an NP1 artifact that answers the schema question without silently turning a tiny smoke run into a performance or correctness promotion.

Findings:

- Wrong-baseline risk: avoided by using the existing nonlinear benchmark harness as the only implementation target and by keeping production nonlinear modules unchanged.
- Proxy-metric risk: value rows keep exact affine parity only for Model A and keep one-step projection metrics explanatory-only; no Model B/C exact nonlinear likelihood claim is introduced.
- Missing-branch-link risk: score timing rows are forced to reference distinct `branch_precheck` rows through `branch_precheck_id`, and blocked score executions would be recorded as `skipped` rather than omitted.
- Compile-overclaim risk: CPU-only smoke rows use eager mode only and record `compile_warmup_policy` explicitly, so this result does not certify graph/XLA support.
- CUT4 unfairness risk: NP1 requires explicit CUT4 point-cap handling, so a synthetic over-cap row is emitted as `row_role=skipped` with `skip_category=cut4_point_cap` rather than silently omitted.
- Environment-mismatch risk: the manifest records `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import and the logical devices seen at runtime, but TensorFlow still emitted `cuInit` stderr noise; under project policy this is sandbox/environment noise, not GPU evidence.

Audit outcome: pass for NP1 smoke scope. The artifact answers the harness-schema question for CPU-only tiny rows and preserves the planned non-claims.

## Evidence contract

Question:

- Does the nonlinear CPU benchmark harness now emit NP1-required manifest fields, explicit row roles, branch-precheck links, CPU visibility policy, CUT4 cap skips, and per-row metadata needed for later NP2/NP3 benchmarking?

Baseline:

- Pre-NP1 `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py` behavior as inspected in NP0.

Primary criterion:

- The harness writes JSON and Markdown artifacts whose rows explicitly include `row_role`, path/mode/device metadata, branch-precheck linkage, manifest linkage, skip metadata, and non-implication text.

Veto diagnostics:

- Production nonlinear modules changed.
- Score timing rows lack `branch_precheck_id` linkage.
- CPU-only rows fail to record hidden-GPU policy.
- CUT4 over-cap cases are silently absent.
- Tiny smoke output is used to claim backend ranking, GPU/XLA support, or exact Model B/C likelihood quality.

Explanatory diagnostics only:

- Model A exact affine log-likelihood parity.
- First-call versus steady-call wall times on tiny eager rows.
- RSS deltas.
- Model A branch-linked score timing rows.

What is not concluded:

- any broad performance ranking;
- any NP2 value fast-path promotion;
- any NP3 score fast-path promotion;
- any graph/XLA certification;
- any GPU evidence;
- any exact nonlinear likelihood correctness statement for Models B-C;
- any HMC or Hessian readiness statement.

## Scope executed

This NP1 worker only edited the CPU benchmark harness and wrote NP1 artifacts under the allowed benchmark/result paths. It did not modify production BayesFilter nonlinear modules, tests, source map, reset memo, GPU harnesses, or git state beyond local uncommitted files.

## Files changed

- `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py`
- `docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.json`
- `docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.md`
- `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md`

## Commands run

```bash
python docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py --requested-device cpu --repeats 1 --output docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.json --markdown-output docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.md
```

## Run manifest

| Field | Value |
| --- | --- |
| Git commit | `81c647b157612a233dde1a9fbd847647a5b03b8f` |
| Dirty worktree | `true` |
| Python | `3.11.14` |
| TensorFlow | `2.19.1` |
| TensorFlow Probability | `0.25.0` |
| CPU/GPU policy | `cpu_only_hidden_gpu_before_tensorflow_import` |
| `CUDA_VISIBLE_DEVICES` | `-1` |
| Requested device | `cpu` |
| Visible logical devices | `['/device:CPU:0']` |
| JSON artifact | `docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.json` |
| Markdown artifact | `docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.md` |
| Governing plan | `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-plan-2026-05-15.md` |
| Governing result | `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md` |

## Validation performed

### Harness schema validation

Confirmed in the JSON artifact:

- allowed `row_role` values appear explicitly: `value_timing`, `score_timing`, `branch_precheck`, and `skipped`;
- score timing rows are represented for cubature, UKF, and CUT4, and each has `branch_precheck_id` pointing to a distinct branch-precheck row;
- value timing rows split `return_filtered=False` from `return_filtered=True` for cubature, UKF, and CUT4;
- each row carries explicit `model`, `backend`, `path`, `dtype`, `T`, `dims`, `parameter_dim`, `point_count`, `return_filtered`, `mode`, `requested_device`, `actual_device`, `device_trust_label`, `gpu_intentionally_hidden`, `compile_warmup_policy`, `command`, `environment_id`, `artifact_path`, `skip_category`, `skip_reason`, and `non_implication_text` fields;
- the manifest records the exact command, git commit/dirty state, Python/TF/TFP versions, CPU visibility policy, artifact paths, and governing plan/result paths;
- a CUT4 over-cap case is emitted as `row_role=skipped` with `skip_category=cut4_point_cap` and concrete `skip_reason`.

### Narrow smoke execution

- Tiny CPU-only Model A value timing rows ran for cubature, UKF, and CUT4 with both `return_filtered=False` and `return_filtered=True`.
- Affine value parity against the exact linear-Gaussian Kalman reference remained at machine precision (`~1e-16`).
- Score branch precheck rows passed on the tiny Model A parameter grid for all three backends.
- Branch-linked score timing rows ran for cubature, UKF, and CUT4 on the selected Model A tiny row.  The score timing rows are timing/provenance evidence only; their parity status is `measured_against_branch_precheck_only`, not a finite-difference or Hessian validation.

## Key observations

1. The NP1 harness now records row roles and branch-precheck linkage explicitly, which was the main contract required for later NP2/NP3 benchmark phases.
2. CPU-only visibility is preserved in the manifest (`CUDA_VISIBLE_DEVICES=-1`), and the artifact records only a visible CPU logical device.
3. Score timing rows now link to branch precheck rows, but NP3 still needs stronger score parity or finite-difference evidence before promoting a score fast path.
4. The synthetic CUT4 skip row satisfies the plan requirement that over-cap cases be emitted explicitly instead of silently disappearing from the artifact.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept NP1 smoke harness/schema artifact | passed: JSON/MD artifacts carry explicit manifest, row-role, return-filtered split, branch-link, device-policy, and skip metadata | passed for smoke scope: no production module edits, no silent CUT4 omission, no missing branch linkage; score timing rows link to branch prechecks | whether the same schema cleanly supports broader tiny/small ladders beyond this minimal smoke | use this harness schema as the NP2 baseline for value-path timing and as the NP3 baseline for branch-linked score timing, with stronger score-parity evidence before optimization promotion | no broad performance claim, no default policy, no graph/XLA/GPU claim, no exact Model B/C likelihood claim |

## Continuation labels

- NP2 continuation label: `NP1_SCHEMA_READY_FOR_NP2_VALUE_BASELINE`
- NP3 continuation label: `NP1_SCHEMA_READY_FOR_NP3_BRANCH_LINKED_SCORE_BASELINE`

## Phase exit label

`NP1_COMPLETE_CPU_SMOKE_SCHEMA_ARTIFACT_WRITTEN`
