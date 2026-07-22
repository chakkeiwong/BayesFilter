# P0 Attempt And Repair Record

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

## Attempts

| Attempt | Classification | Outcome | Repair |
| --- | --- | --- | --- |
| `attempt-01-20260715T1626` | `EVIDENCE_REPORTING` | Core JSON schema passed, but selected scholarly-audit skill required six separate source/metadata/snowball/claim/omission ledgers that were absent. | Preserve attempt; add the six ledgers and validation gates. |
| `attempt-02-20260715T1645` | `HARNESS_INFRASTRUCTURE` | Builder raised `TypeError` because `_budget()` returned `None` after a function-boundary patch error. Validation did not run. | Move budget return into `_budget()` and add direct budget/scholarly regression assertion. |
| `attempt-03-20260715T1652` | `EVIDENCE_REPORTING` | Content validator passed, but run manifest reconstructed an explicit `--generated-at` argument that the actual command had not supplied. | Make generated timestamp explicit in both executed command and command manifest. |
| `attempt-04-20260715T1658` | `PASS` | Compile, build, validator, budget, source-ledger, classification, hash, and exact-command checks passed. | Closeout candidate. |

All attempts use separate directories under
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p0/`.
No prior evidence was overwritten. No attempt imported TensorFlow, used GPU,
trained a model, or ran HMC.

## Focused Regression

```text
python -m py_compile \
  docs/benchmarks/build_multimodel_neutra_p0_registry.py \
  docs/benchmarks/check_multimodel_neutra_p0_registry.py

python docs/benchmarks/build_multimodel_neutra_p0_registry.py \
  --output-root docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p0/attempt-04-20260715T1658 \
  --generated-at 2026-07-15T16:58:00+08:00

python docs/benchmarks/check_multimodel_neutra_p0_registry.py \
  --output-root docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p0/attempt-04-20260715T1658
```

Validator result:

```json
{"cell_count": 11, "passed": true, "per_cell_gpu_hours": 40, "program_gpu_hours": 442, "target_blocked_count": 11, "target_signatures_issued": 0}
```

