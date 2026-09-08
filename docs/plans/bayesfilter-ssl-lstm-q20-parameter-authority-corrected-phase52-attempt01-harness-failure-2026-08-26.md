# Corrected Parameter-Authority Phase 52 Attempt 01 Harness Failure

Date: 2026-08-26  
Classification: `HARNESS_INVALID_WRONG_PILOT_ROUTE`  
Scientific interpretation: none  
Retry: `attempt-02` with the verified corrected-theta generator

## Verdict

Attempt 01 is wrong relative to the Phase 52 input contract. The declared
input was a fresh corrected-theta pilot with schema
`bayesfilter.ssl_lstm.q20.corrected_theta_authority_pilot.v1`, status
`PASS_THETA_MEASURE_PILOT`, measure `theta_R4`, a completed calibration block,
and M0 seed `[20260826, 5201]`.

The command actually selected
`docs/benchmarks/run_ssl_lstm_q20_particle_authority_pilot_2026_08_25.py`.
It computed schema `bayesfilter.ssl_lstm.q20.authority_pilot.v1`, status
`PASS_GATE`, no top-level theta measure or corrected calibration receipt, and
M0 seed `[20260826, 5301]`. These objects are different. A passing status from
that route cannot be relabeled as the required corrected-theta pilot.

This invalidates the launch harness only. It does not invalidate the q=20
target, the finite support/geometry fixture, either proposal law, or the Phase
52 research question. No boundary, bootstrap report, NeuTra training, or HMC
run was launched from the invalid receipt.

## Preserved evidence

| Item | Value |
|---|---|
| command | `docs/benchmarks/run_ssl_lstm_q20_particle_authority_pilot_2026_08_25.py --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/pilot-01 --particles 256 --calibration-particles 64 --seed 20260826 5201 --arms both --mutation identity` |
| invalid receipt | `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/pilot-01/pilot.json` |
| invalid receipt SHA-256 | `6d795217d5774cd62f88c70db3e2df43ddb36143a1f74fe04f01eb33b003592b` |
| wrong runner SHA-256 | `1c173031c994a74600b0b6db910aff4666ce62eaf7412f29e2710199c193dd10` |
| wall time | `351.91348750598263 s` |
| CPU/GPU posture | `CUDA_VISIBLE_DEVICES=-1`; no physical or logical GPU |

The valid Phase 52 fixture remains separate at
`phase52-fresh-paired-uncertainty-replication/fixture/result.json`, with status
`PASS_V3_4_FRESH_PAIRED_FIXTURE`. It did not consume the invalid pilot.

## Root cause and repair

The plan used a similarly named historical particle-authority generator rather
than the corrected parameter-authority generator used by the valid Phase 47
receipts. The verified route is
`docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py`,
whose SHA-256 is
`c0b793ab10bd8d69cec22347c3beba00b5dd15e77e129f61b25d8dc585b9b703`.
That hash exactly matches the runner hash preserved by Phase 47.

The corrected generator maps a root seed `[a,b]` to M0 seed `[a,b+100]` and
C0 seed `[a,b+101]`. Attempt 02 therefore uses root seeds
`[20260826,5101]` through `[20260826,5106]` to produce the frozen M0 seed
ledger `[20260826,5201]` through `[20260826,5206]`. All valid retry artifacts
are written below `phase52-fresh-paired-uncertainty-replication/attempt-02/`;
the invalid attempt is preserved and never overwritten or pooled.

## Decision table

| Decision | Primary status | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| accept attempt-01 pilot | fail | wrong schema, measure, seed, and route | none; mismatch is direct | exclude it from Phase 52 | candidate quality |
| retain fixture | pass | no fixture gate fired | finite 2D scope only | reuse unchanged fixture receipt | q=20 invariance |
| retry pilot generation | repair authorized | target/method/budget unchanged | fresh corrected receipts still needed | use attempt-02 verified route | proposal superiority |

## Inference status

| Evidence class | Status |
|---|---|
| hard veto screen | fired for attempt-01 harness |
| statistically supported ranking | none |
| descriptive-only differences | none interpreted |
| default readiness | not ready |
| next evidence needed | six passing attempt-02 corrected-theta pilots |

The campaign lower-bound wall time after the valid fixture and invalid pilot is
`34874.57283953301 s`, leaving `29925.427160466992 s` under the authorized
`64800 s` cap. The repair changes no target, data, method, criteria, hardware
class, privacy boundary, or global budget.
