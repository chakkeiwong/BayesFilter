# HMC Rank-Normalized Tuning Verifier Repair Result

Date: 2026-07-13

Status: `PASS_CODE_REPAIR_FRESH_RETUNE_REQUIRED`

## Outcome

The tuning verifier implementation bug is fixed. Deterministic HMC tuning and
Phase 7 now consume one shared R-hat implementation whose decision statistic is

`max(rank-normalized split R-hat, folded rank-normalized split R-hat)`.

This repair did not rewrite the historical tuning artifact, which predates the
repair. Its classical split-free R-hat pass is not a modern-R-hat pass. No
retuning, Phase 7 retry, retained sampling, Phase 8 scientific runtime, or
NeuTra work ran under this repair.

## Code Changes

- `bayesfilter/inference/hmc_convergence.py` now owns the shared
  `rank_normalized_split_rhat_summary` implementation and the exact diagnostic
  definition.
- `rank_normalized_hmc_diagnostics` uses that shared R-hat summary before adding
  Phase 7 bulk/tail ESS diagnostics.
- `bayesfilter/inference/hmc.py::_rhat_summary_from_retained_samples` no longer
  computes classical split-free R-hat. It calls the shared modern summary and
  retains the existing aggregate verifier fields.
- The tuning verifier and its public summary now record the exact definition,
  maximum rank-normalized split component, and maximum folded component.
- Verifier lifecycle tests now use deterministic well-mixed draws with enough
  samples for a valid split diagnostic.

## Regression Evidence

A deterministic scale-mismatch fixture used aligned chain locations with chain
scales `0.5`, `1.0`, `2.0`, and `3.0`. Its rank-normalized split location
component was below `1.01`, while its folded component was above `1.01`.

The shared helper, tuning summary, and Phase 7 diagnostic all rejected the same
draws and returned the same combined maximum R-hat. This directly covers the
failure mode missed by the old tuning implementation.

## Checks

All commands deliberately set `CUDA_VISIBLE_DEVICES=-1` for TensorFlow tests.
They were CPU-only code/regression checks, not HMC experiments or GPU evidence.

| Check | Result |
| --- | --- |
| Shared diagnostic plus verifier suite | `26 passed, 2 warnings` |
| Tuning outer loop plus Phase 7 controller suite | `94 passed, 2 warnings` |
| Tuning driver suite | `15 passed, 2 warnings` |
| Public tuning API plus sequential checkpoint contract | `47 passed, 2 warnings` |
| Python compilation | Passed |
| Scoped whitespace validation | Passed |
| Active-path classical-R-hat scan | No classical/split-free calculation remains in the deterministic tuning/Phase 7 path |

The warnings are the existing TensorFlow Probability `distutils.version`
deprecations.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close implementation repair | Passed: tuning and Phase 7 use one shared modern-R-hat implementation and the scale-mismatch regression agrees exactly | No code, test, compilation, or contract veto fired | Whether the historical selected kernel passes the repaired tuning gate is untested | If desired, create a fresh retuning experiment plan and new artifacts | Historical-kernel rejection by repaired tuning, posterior convergence/recovery, sampler ranking, or production/default readiness |

## Evidence Classification

- Engineering correctness: the implementation mismatch is repaired and focused
  contracts pass.
- Sampler validity: not reevaluated; no HMC transition ran.
- Scientific interpretation: unchanged. The historical Phase 7 candidate remains
  rejected under its recorded serious screen, while the broader HMC direction
  and target remain neither proved nor disproved.

## Handoff

The code repair is complete. All prior tuning artifacts must be treated as
pre-repair artifacts. A fresh retune is required before any kernel can claim to
have passed the corrected tuning verifier, and any serious sampling retry needs
a separate reviewed experiment plan.
