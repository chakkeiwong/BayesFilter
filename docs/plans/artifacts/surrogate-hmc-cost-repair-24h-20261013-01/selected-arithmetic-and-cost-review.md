# Selected arithmetic and consumer-cost review

The five frozen selected programs pass the owner 1% arithmetic screen at the
four saved HMC starting positions. The comparison is to the same finite filter
in FP64 at identical physical inputs, not a model likelihood/score oracle.
The scaled error is abs(candidate-reference)/max(1,abs(reference)); percentages
below multiply this by 100. SIR and range use their explicitly optional FP64
recursion/FP32 pair scopes; FP32/TF32-off remains the default. N4032/K2016.

| Model | Likelihood | Physical score | Coordinate force | Actual consumer wiring |
|---|---:|---:|---:|---:|
| lgssm | 0.0000331% | 0.0026661% | 0.0029084% | 0.0000000% |
| ksc | 0.0002411% | 0.0301020% | 0.0308725% | 0.0000000% |
| predator_prey | 0.0000548% | 0.0379819% | 0.0379167% | 0.0000000% |
| sir_d18 | 0.0000031% | 0.0976928% | 0.0978606% | 0.0000000% |
| range_bearing | 0.0009345% | 0.3054078% | 0.2116925% | 0.0000000% |

All four rows remain finite and complete their horizon. Actual B4 consumer
forces equal the grouped B2 reconstruction at these inputs. Source-closure
checks for every accepted artifact match the recorded pre-run hashes; see
selected-consumer-precision-review.json. The initial LGSSM artifact is preserved
but superseded because its diagnostic source changed during execution. The
isolated LGSSM rerun has matching pre-run and archived source hashes. The
controller now detects that mismatch automatically. Three focused CPU tests
pass, covering thresholds, binding, source snapshots and targeted retry.

The first paired KSC result had nonstationary timings and a 1.111 ratio. Two
fresh process repeats with continuous GPU telemetry yielded:

| Repeat | Baseline mean seconds | Candidate mean seconds | Candidate/baseline |
|---|---:|---:|---:|
| repeat-01 | 9.9480 | 10.0019 | 1.00542 |
| repeat-02 | 9.8966 | 10.0361 | 1.01410 |

Both repeat force discrepancies are 4.77e-7 scaled. These observations do not
support a KSC speed improvement. They also do not reproduce the initial 11%
slowdown. Few paired times and incomplete WSL process visibility preclude a
statistical speed ranking. Preserve the first observation, the telemetry and
every individual time in ksc-monitored-repeats-01; do not replace them by one
favorable number. This closes the large-regression suspicion sufficiently to
continue targeted transport cost work, not a universal speed claim.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain selected arithmetic evidence | Five models within 1% at four starts | Health, wiring and source checks pass | Coverage outside starts | Preserve frozen evidence; test next cost candidate separately | Posterior correctness or convergence |
| Retain shared-Gaussian repair | All-model parity passes | No new invalid row in checked region | KSC shows no clear timing saving | Address remaining transport cost under phase 2 | All-model speed superiority |
| Exclude initial LGSSM precision artifact | Source snapshot mismatch | Provenance veto | None for replacement artifact | Use isolated rerun | Old artifact accepted retroactively |

| Inference status | Finding |
|---|---|
| Hard veto screen | Selected arithmetic/health/wiring pass; initial LGSSM provenance veto repaired by rerun |
| Statistically supported ranking | None |
| Descriptive-only differences | All latencies, ratios and observed starts-region errors |
| Default-readiness | FP32 remains default; SIR all-FP32 numerical failure remains open |
| Next evidence needed | Full-horizon parity and actual-consumer performance for any further repair; genuine adaptive posterior runs require the existing target/tuner conditions |

Post-run red team: start-region arithmetic may hide worse errors along a real
HMC trajectory. A later valid row exceeding 1% would overturn arithmetic
coverage for that region. The weakest cost evidence is sparse repeats with
incomplete external-process visibility. Two-step mechanics are not convergence
or inference evidence. The computational master remains in progress.
