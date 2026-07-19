# Phase A4 HMC Acquisition Repair Plan

Date: 2026-07-14 (Asia/Shanghai)

Status: `AUTHORIZED_REPAIR_IMPLEMENTATION_PENDING`

Authorization: On 2026-07-14 the owner authorized exactly one fresh A4 HMC
repair attempt using `step_size=0.19625`, `8` leapfrog steps, the original four
starts, fresh artifacts, and the remaining `7.6295 GPU-hours` with sequential
stopping. Forecast calibration is authorized only if this repair produces an
admitted archive.

## Objective

Test whether halving the HMC step size while preserving trajectory length
repairs the state-dependent zero-acceptance pathology seen in chain 0, without
changing the locked A1 target, affine geometry, dispersed starts, four-chain
requirement, or sampler-admission thresholds.

## Entry Conditions

- The prior acquisition result is
  `BLOCKED_INVALID_CALIBRATION_INPUT_REPAIR_REQUIRED`.
- The balanced retained attempt remains rejected and immutable.
- Target semantic SHA-256 remains
  `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e`.
- A0 affine geometry, initial states, diagnostic thresholds, native-divergence
  qualification, and private-archive contracts remain unchanged.
- All prior trusted-GPU wall time, including failed attempts, remains charged
  to the same `8 GPU-hour` cap.

## Repair Mechanism

Use the already predeclared constant-trajectory candidate:

```text
step_size = 0.19625
num_leapfrog_steps = 8
trajectory_length = 1.57
```

The mechanism under test is reduced integration error at the same nominal
trajectory length. This is a kernel repair, not a new posterior, initialization
filter, chain deletion, or relaxation of admission criteria.

## Evidence Contract

| Field | Prospective requirement |
| --- | --- |
| Question | Does the half-step/eight-leapfrog kernel produce four finite moving chains and an admissible retained archive where the balanced kernel produced a zero-acceptance chain? |
| Exact comparator | Rejected balanced-kernel attempt `segment-0.json`, used only as failure context; the new attempt must independently pass all gates |
| Tuning gate | Fresh 64-retained/32-burn-in four-chain screen; all chains move; each acceptance in `[0.20,0.95]`; finite samples/target/log-accept telemetry; GPU/XLA placement; no positive native divergence if available |
| Retained gate | Fresh 250-burn-in/250-retained archive; four chains move; per-chain and aggregate acceptance `[0.20,0.95]`; maximum rank-normalized split R-hat `<=1.05`; every bulk/tail ESS `>=100`; every mean MCSE/SD ratio `<=0.10` in both latent and A1 free coordinates |
| Hard vetoes | Target/source/geometry drift; malformed or nonfinite archive; any unmoved retained chain; positive native divergence if exposed; missing diagnostic; budget exhaustion; or reuse of rejected `segment_0` as samples/current state |
| Explanatory only | Runtime, maximum finite log-accept ratio, target range, posterior means/SDs, and descriptive contrast with the rejected balanced attempt |
| Nonconclusions | No sampler superiority/ranking, posterior correctness, HMC readiness beyond A4 input admission, predictive equivalence, NeuTra readiness, or default readiness |
| Result artifact | New repair-specific tuning/archive/result paths; never overwrite prior attempts |

## Skeptical Audit

| Challenge | Disposition |
| --- | --- |
| Wrong baseline | The locked A1 posterior remains the target; the rejected balanced kernel is failure context, not a weak scientific comparator |
| Proxy promotion | Tuning acceptance only selects a kernel for the serious retained gate; it cannot admit calibration input |
| Missing stop | Any unmoved retained chain is again a continuation veto; no repeated seed hunting is allowed |
| Post-hoc rescue | The half-step candidate was in the original prospective ladder before `segment_0`; the repair requires fresh artifacts and cannot delete or reuse failed samples |
| Unfair comparison | Same target, affine geometry, starts, trajectory length, thresholds, chain count, and budget accounting |
| Resource risk | Before launch, recompute remaining budget from all prior GPU artifacts and require projected tuning plus smallest retained rung to fit |
| Misleading pass | A passing repair only admits HMC draws for A4 calibration; it does not validate the posterior or establish superiority over the rejected kernel |

Audit decision: `READY_FOR_SEPARATE_AUTHORIZATION`; not executed under the
blocked acquisition result.

## Sequential Steps

1. Add repair-specific labels/seeds and strict lineage support without changing
   production BayesFilter code.
2. Rerun focused CPU-hidden tests and one native review of the repair diff.
3. Run exactly one fresh 64-draw tuning screen for `(0.19625,8)`.
4. If tuning passes, run exactly one fresh 250/250 retained archive from the
   original fixed dispersed starts, not from `segment_0` final states.
5. If the retained gate passes, stop HMC and hand the new archive to A4
   forecast calibration. If any hard veto fires, write a blocker and stop.
6. Extend retained draws only for R-hat/ESS/MCSE promotion vetoes after every
   chain has moved; never extend a zero-movement chain.

## Frozen Repair Namespace And Seeds

| Role | Frozen value |
| --- | --- |
| Public artifact root | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/repair-01/` |
| Private archive root | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/repair-01/private/` |
| Tuning label / seed | `repair_01_tune_smaller_step`; `[20260714,1521]` |
| Retained labels / seeds | `repair_01_segment_{i}`; `[20260714,1530+i]` |
| Tuning output | `repair-01/tune.json` |
| Retained outputs | `repair-01/segment-{i}.json` |
| Repair runner | `docs/benchmarks/run_ssl_lstm_a4_hmc_repair_2026_07_14.py` |
| Focused tests | `tests/test_ssl_lstm_a4_hmc_repair.py` |
| Result | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-repair-result-2026-07-14.md` |

The repair runner must bind the four prior GPU receipts and their current
SHA-256 values, charge all four wall times, and fail closed if any receipt
drifts. It must refuse an existing public output or private archive member.
The original acquisition harness remains byte-identical so its prior source
bindings remain verifiable.

## Exact Commands

All CPU checks deliberately hide GPUs. GPU commands use the trusted managed
session, TensorFlow/TFP, XLA JIT, and the repository `tfgpu` environment.

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_ssl_lstm_a4_hmc_repair.py

/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_a4_hmc_repair_2026_07_14.py tune

/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_a4_hmc_repair_2026_07_14.py segment --segment-index 0
```

Later segment commands are eligible only when the immediately preceding
segment has status `EXTEND`, every chain moved, no hard veto fired, and the
projected command fits the remaining shared budget. An admitted segment stops
HMC immediately. A `HARD_VETO` or non-selected tuning screen stops the repair.

## Forbidden Actions

- Do not overwrite or mutate the failed canary, tuning, or `segment_0`
  artifacts.
- Do not delete chain 0, change its start, shorten burn-in, or select a seed by
  observed acceptance.
- Do not use remaining GPU budget as permission to bypass the prior veto.
- Do not run forecast calibration until a fresh four-chain archive passes.
- Do not run NeuTra or A5 confirmation.
