# NeuTra All-Executable-Models End-to-End Terminal Result

Date: 2026-07-19

Campaign: `bayesfilter-neutra-all-executable-models-e2e-20260718`

Status: `TRAINING_COMPLETE_TUNING_GATE_OPEN`

## Scope and evidence

The executable registry contains five cells and seven blocked inventory cells.
Four cells completed fresh target-specific training, parity, held-out checks,
and native fixed-transport tuning. `PP-SGQF` did not complete because its
original target identity hashed device-dependent serialized tensor bytes; the
identity was repaired and CPU/GPU parity was demonstrated afterward, but the
planned one infrastructure retry was consumed before a fresh full run.

Fresh transport artifacts:

| Cell | Training | Tuning result | Hard veto / classification | Sampling |
| --- | --- | --- | --- | --- |
| `LGSSM-EXACT` | 5,000 segmented steps, terminal freeze | `no_viable_candidate` | verifier modern folded R-hat failed; one screen log-accept veto | not run |
| `PP-UKF` | 5,000 segmented steps, terminal freeze | `no_viable_candidate` | ladder budget exhausted; no hard numerical veto | not run |
| `SIR-SGQF` | 5,000 segmented steps, terminal freeze | `no_viable_candidate` | screen log-accept nonfinite/missing | not run |
| `STR-UKF` | 5,000 segmented steps, terminal freeze | `no_viable_candidate` | verifier energy-error and modern folded R-hat vetoes | not run |
| `PP-SGQF` | not rerun after identity repair | not available | identity continuation blocker consumed retry budget | not run |

The four trained cells all have immutable five-segment checkpoints, unchanged
config hashes, exact parent lineage, and terminal-only frozen transports. No
cell reached sequential warm-up, retained sampling, truth-tail evaluation, or
posterior accuracy evidence. The seven blocked inventory cells were not
launched and are not failures.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Training composition | Fresh finite target-specific transport with parity | passed for four cells | no downstream HMC admission | preserve transports and repair tuning gate | no NeuTra sampler validity |
| Native tuning | acceptance 0.70 in `[0.65,0.75]` plus configured verifier | failed for all four | fixed 1,000-draw verifier and screen diagnostics | revise tuner/sampling contract before rerun | no claim that transport is scientifically wrong |
| Posterior convergence | sequential physical-coordinate folded R-hat/ESS | not evaluated | no admitted kernel | run only after an admitted tuned kernel exists | no convergence claim |
| Truth recovery | all parameters `p_truth >= 0.05` | not evaluated | no retained draws | run only after convergence | no accuracy/calibration claim |
| Default readiness | end-to-end pass | not ready | no sampler result | require a coherent repaired campaign | no superiority/ranking claim |

## Interpretation

The scientific question remains open. The result invalidates neither the
trained transports nor the NeuTra direction. It shows that the current
end-to-end promotion gate is too strict or internally mismatched for this
campaign: the native tuner performs a fixed 1,000-result rank/folded-R-hat
handoff screen, while the sequential sampler is separately designed to extend
warm-up and retained draws adaptively to 10,000 per chain. A candidate that
would become valid at 2,000--10,000 draws is rejected before that adaptive
controller can run. This is a plan/tuning-contract issue, not evidence of
posterior failure.

The other vetoes remain real diagnostics: nonfinite log-accept screens and
energy-error limits must be investigated rather than relabeled. Acceptance
near 0.65--0.75 by itself is not sufficient for promotion.

## Required reset

Do not retrain the four preserved transports. Create a new reviewed plan that:

1. explicitly separates step-size tuning from convergence assessment;
2. uses the native tuner for acceptance and finite/status/energy health, but
   lets the existing sequential controller own adaptive folded-R-hat/ESS caps;
3. declares how a fixed-kernel verification result is admitted when its R-hat
   is above 1.01 but finite and otherwise healthy, without silently weakening
   the final retained convergence gate;
4. adds a bounded nonfinite-log-accept repair/localization arm for SIR-SGQF and
   the affected LGSSM screen rounds; and
5. reruns `PP-SGQF` only after the repaired cross-device identity is included in
   its fresh manifest.

The next run must reuse the frozen payloads by hash and must not claim a
scientific pass until sequential physical-coordinate folded R-hat, ESS, health,
and truth-tail artifacts exist.

## Nonclaims

This result does not establish NeuTra correctness, posterior correctness,
HMC readiness, convergence, calibration, truth recovery, speed, superiority,
default readiness, or failure of the scientific idea. It also does not rank the
models. It is a valid training-completion and tuning-gate diagnostic.
