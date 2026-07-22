# HNN-NeuTra Exact-Gradient Comparison Phase 1 Harness Result

Decision: `PASS_PHASE1_HARNESS_READY_FOR_TRUSTED_GPU_CANARIES`.

## Implemented Repair

- Added a new repair-only comparison module and runner. Historical P4/P5
  result schemas and artifacts remain readable and unchanged.
- Bound the executable primary arm set to exactly `learned_residual` and
  `true_gradient`. Zero residual remains historical explanatory evidence only.
- Added synchronized supervision, optimizer, tuning, matched-mechanics, and
  adaptive-chain timing.
- Added production-shape precompilation so sampling execution time excludes its
  compile probe while cold compile cost remains explicit.
- Added three alternating synchronized matched-timing repeats.
- Added physical-coordinate HNN-versus-exact interval and pooled-MCSE agreement.
- Added preparation, tuning, warm sampling, seconds/minimum-bulk-ESS, reuse,
  break-even, and guarded from-scratch ledgers.
- Kept candidate rejection separate from infrastructure/process failure.

## Checks

| Check | Result |
| --- | --- |
| CPU-hidden focused tests | `33 passed`, two upstream TFP deprecation warnings |
| Python compilation | pass for all changed Python modules and runner |
| CLI `--help` | pass; four exact cell choices and `--canary` exposed |
| `git diff --check` | pass for the repair file set |
| Static arm audit | exact two-arm set present; zero residual absent from primary factory |
| Timing audit | explicit materialization, precompile separation, alternating repeats present |
| Cost audit | supervision and HNN grid wall charged; unknown common chart cost is not set to zero |

CPU-only checks intentionally used `CUDA_VISIBLE_DEVICES=-1`; they are
engineering evidence only, not GPU/XLA or scientific evidence.

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Exact baseline | implemented as complete transformed target gradient |
| HNN candidate | fresh target-local force in serious runs; historical force only for canary |
| Endpoint | common value-only complete transformed posterior |
| Matched mechanics | implementation and assertions present |
| Tuned validity | existing modern adaptive diagnostic path reused with synchronization |
| Accuracy comparison | physical means, intervals, MCSE, truth-tail implemented |
| Performance comparison | matched time and tuned seconds/minimum bulk ESS implemented |
| Not concluded | no scientific result before Phase 3/4 serious runs |

## Phase 2 Commands

All commands require trusted/escalated GPU execution and write fresh roots:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
docs/benchmarks/run_hnn_neutra_exact_comparison_2026_07_18.py \
--cell PP-UKF --canary \
--output-root docs/plans/artifacts/hnn-neutra-exact-gradient-comparison-repair-20260718/phase2-canary/PP-UKF/attempt-01
```

Repeat exactly for `PP-SGQF`, `SIR-SGQF`, and `STR-UKF`, each in its named
fresh attempt directory.

## Decision And Inference Status

| Field | Status |
| --- | --- |
| Primary criterion | engineering harness checks passed |
| Hard veto status | none fired |
| Main uncertainty | exact-gradient target graphs have not yet compiled on trusted GPU in this repair route |
| Next justified action | run trusted device probe, then four two-transition canaries |
| Statistically supported ranking | none; no research run occurred |
| Default readiness | not established |

Phase 2 was reviewed for baseline identity, endpoint boundary, XLA/GPU policy,
memory growth, fresh artifacts, structural semantics, and bounded execution.
No real continuation veto fired.
