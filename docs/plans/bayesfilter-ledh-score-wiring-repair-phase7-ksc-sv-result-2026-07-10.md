# Phase 7 Result: KSC-SV Compact Precision Gate

Date: 2026-07-10

Status: `PASSED_KSC_SV_COMPACT_PRECISION_GATE`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| KSC-SV now uses an explicitly named compact sequential-seed score route with production precision and hardened full-admission boundaries. | Passed focused CPU-hidden wiring tests: compact no-autodiff score execution, same-scalar finite differences through a value-only objective, KSC target preservation, and one compact component call per seed. | Full admission rejects non-compact nested provenance, false no-autodiff/same-route declarations, non-production precision, mismatched particles/time/seeds, target substitution, and exact-native actual-SV overclaim. | No trusted full `N=10000,T=1000` KSC-SV score-memory run was launched. | Review this result and the Phase 8 cross-model subplan. | No full KSC-SV score admission, native actual-SV likelihood correctness, leaderboard rebuild, HMC readiness, posterior correctness, runtime ranking, or scientific superiority claim. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Is KSC-SV wired so its existing compact score route satisfies production precision and full-admission boundaries without being relabeled as native actual-SV likelihood evidence? |
| Baseline/comparator | The pre-phase compact KSC-SV implementation and admitted KSC forward artifact; the score module defaulted to `float64`/TF32 disabled and lacked precision plus nested full-row guards. |
| Primary criterion | Passed. The default diagnostic calls `_compact_value_and_score_across_seeds`, which evaluates `_compact_value_and_score_from_components` sequentially per fixed seed. FD perturbations call the value-only objective. Tiny no-autodiff and same-scalar checks pass. |
| Veto diagnostics | Passed. Defaults are `float32`/TF32-enabled; full construction requires compact nested provenance, production `score_precision`, admitted-value shape/seeds, and trusted memory metadata. The shared contract rejects target substitution and exact-native actual-SV overclaim. |
| Explanatory diagnostics | Tiny CPU-hidden compact/value parity and finite-difference checks only. |
| Artifact | This result and focused tests; no new admitted score JSON was produced. |

## Claimed And Computed Quantities

| Item | Classification |
| --- | --- |
| Claimed target | Score of the realized finite-`N` LEDH log-likelihood estimator for the admitted `ksc_log_chi_square_gaussian_mixture_surrogate` row. |
| Quantity checked | Compact forward sensitivity of the tiny fixed-randomness KSC Gaussian-mixture scalar versus central finite differences of its value-only route. |
| Relationship | Equal within the declared tiny finite-difference tolerance. This is not the native actual-SV likelihood. |
| Support | `tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py` and the final focused pytest result below. |

## Changed Files

- `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py`
- `tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py`

## Implementation Summary

- Set module and CLI score defaults to `float32` with TF32 enabled.
- Added explicit score-precision metadata.
- Added `_compact_value_and_score_across_seeds` and kept the old manual-named
  function only as a compatibility alias. The default diagnostic calls the
  compact wrapper, not the alias.
- Preserved sequential per-seed component evaluation to avoid an unreviewed
  full-row peak-memory multiplication.
- Kept FD perturbations on the value-only same-scalar objective.
- Added nested compact provenance, parameter, full shape, and seed checks.
- Preserved `ksc_log_chi_square_gaussian_mixture_surrogate`, the synthetic
  unconstrained coordinate system, and
  `claims_exact_native_actual_sv_likelihood = false`.

## Local Checks

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py
```

Result: passed.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Final post-batching-repair result:

```text
68 passed, 2 warnings in 23.16s
```

Source search found compact sequential-seed score wiring, value-only FD,
production precision defaults, explicit score precision, full-row guards, the
KSC target policy, and exact-native nonclaim.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus current scoped uncommitted changes |
| Command | Final focused pytest command shown above |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`; TensorFlow `2.19.1` |
| CPU/GPU status | CPU-only wiring diagnostic; `CUDA_VISIBLE_DEVICES=-1` intentionally hid GPUs before framework import |
| Data version | Admitted KSC-SV forward artifact dated 2026-07-07 |
| Random seeds | Tiny test seed `81120`; two-seed call-path fixture `81120,81121`; synthetic full-admission fixture `81120` through `81124` without a full computation |
| Wall time | `23.16s` for the final focused pytest run |
| Output artifacts | This result and test output; no admitted score JSON |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase7-ksc-sv-subplan-2026-07-10.md` |
| Result file | This file |

## Post-Run Red Team

- Strongest alternative explanation: tiny KSC tests can pass while the full
  `N=10000,T=1000` route exceeds memory or develops scale-specific numerical
  failure.
- Result that would overturn this gate: the default diagnostic reaches the
  compatibility alias/historical route, batches all full-row seeds into one
  component call, accepts forged nested metadata, changes the target policy,
  or permits an exact-native actual-SV claim.
- Weakest evidence: no trusted full-row GPU score-memory artifact exists for
  KSC-SV in this phase.

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | No wiring/admission veto fired in the focused suite. |
| Statistically supported ranking | None; no candidates were ranked. |
| Descriptive-only differences | None used for a decision. |
| Default-readiness | Wiring defaults satisfy the reviewed policy; full-row score admission remains unproved. |
| Next evidence needed | Cross-model regression gate, then trusted full-row GPU memory evidence in Phase 9. |

## Boundary Notes

- CPU-hidden tests are wiring evidence only.
- Synthetic full-admission fixtures test contract logic; they are not full-row
  runtime or memory evidence.
- KSC-SV remains a Gaussian-mixture surrogate row and is wrong relative to any
  claim that it computes exact native actual-SV likelihood.

## Next Phase Handoff

Phase 8 may start only after fresh review of the repaired Phase 6 result, this
Phase 7 result, and the Phase 8 subplan.

Fresh Codex substitute re-review returned `VERDICT: AGREE`; Phase 8 may start
within its CPU-hidden cross-model scope.
