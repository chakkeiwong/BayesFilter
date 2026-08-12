# q=20 CPU-XLA Parallel Seed-A/Seed-B Training Result

Date: 2026-08-01
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-cpu-xla-parallel-training-plan-2026-08-01.md`
Aggregate artifact: `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1/summary.json`

## Outcome

The parallel supervisor completed both independently seeded CPU-XLA streams.
Both children exited with code `0`, wrote valid terminal artifacts, and passed
the declared CPU diagnostic screen. The supervisor aggregate status is
`COMPLETED` after `27,366.63 s` (`7.60 h`), below the `40,000 s` cap.

| Seed | Terminal step | Best step | Stop reason | LR reductions | Result | Wall time |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| A | 2,250 | 1,500 | `plateau_after_lr_repair` | 1 (`4e-4 -> 2e-4`) | `CPU_XLA_DIAGNOSTIC_SCREEN_PASSED` | `20,920.77 s` |
| B | 3,250 | 2,500 | `plateau_after_lr_repair` | 2 (`4e-4 -> 2e-4 -> 1e-4`) | `CPU_XLA_DIAGNOSTIC_SCREEN_PASSED` | `27,362.71 s` |

Seed A stopped earlier because its plateau controller fired after the first
learning-rate repair. Seed B continued through step 3,250, then stopped after
its second repair. These are declared adaptive stop decisions, not failures.

## Validity Screens

- Both runs used CPU-only execution with CUDA hidden before TensorFlow import.
- Both parent and worker target paths recorded `jit_compile=True`; child logs
  contain TensorFlow `Compiled cluster using XLA` receipts.
- Seed A used CPUs `0..24`; seed B used CPUs `25..49`.
- Each child used `25` persistent workers, `4` rows per worker, and training
  batch size `100`.
- Thread audits passed for both children. Maximum recorded process-tree native
  threads were `802`; configured compute cores remained within the limit of
  `50`.
- No finite-value, support, round-trip, memory, affinity, or artifact veto
  fired.
- Final support probes were finite. Round-trip maxima were
  `7.11e-15` (seed A) and `2.66e-15` (seed B); shell inverse radius was `4.0`.
- Final 256-row audit mean loss was `41.368657` (seed A) and `40.836181`
  (seed B).

## Inference Status

| Evidence class | Status | Interpretation |
| --- | --- | --- |
| Hard veto screen | Passed for both seeds | No implementation/resource/artifact veto was observed. |
| Statistically supported ranking | None | Two seeded CPU streams do not support ranking one seed or method as superior. |
| Descriptive-only differences | Seed B ran longer and has lower final audit mean loss | These are descriptive observations, not uncertainty-supported rankings. |
| Default readiness | Not eligible | This is a CPU diagnostic exception; it does not change the GPU NeuTra default. |
| Next evidence needed | Target-specific GPU/XLA claim-bearing training and downstream validation | CPU diagnostic success is insufficient for HMC, posterior, or scientific claims. |

## Decision Table

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept both CPU streams as completed diagnostics | Both terminal artifacts finite and screen-passed | None fired | One campaign, two seeds; no statistical ranking | Preserve artifacts and use them as CPU-XLA diagnostic evidence | No GPU equivalence, posterior correctness, HMC readiness, or scientific validity |
| Retain adaptive controller behavior | Plateau repairs and stop reasons were recorded | No nonfinite/support failure | Plateau policy is descriptive for this target/scope | Review the validation trajectory before any new training budget | No claim that either seed converged scientifically |

## Post-Run Red Team

The strongest alternative explanation is that the two concurrent CPU lanes
changed throughput and plateau timing relative to the isolated `25x4` timing
benchmark. The run artifacts preserve the actual wall times and controller
history, so those timing differences must not be generalized as a universal
topology ranking. A result that would overturn the completion classification
would be a missing/corrupt terminal artifact, nonzero child exit, or a failed
finite/support/affinity/resource screen; none occurred here.

## Nonclaims

This result is CPU-XLA diagnostic evidence only. It does not establish
posterior correctness, convergence, HMC readiness, transport promotion,
GPU equivalence, architecture superiority, default readiness, or broad
scientific validity.
