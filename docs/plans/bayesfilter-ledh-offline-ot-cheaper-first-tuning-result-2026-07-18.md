# LEDH Offline OT Cheaper-First Tuning Result

Date: 2026-07-18  
Campaign ID: `ledh-offline-ot-cheaper-first-tuning-20260718`  
Status: `RECLASSIFIED_UNTUNED_T50_BASELINE_2026-07-19`

> Reclassification, 2026-07-19: the T=50 execution used settings tuned for the
> different T=10 scope. It is therefore an untuned T=50 baseline failure, not a
> valid T=50 tuned-claim veto. The numerical measurements remain correct. The
> planning and promotion interpretation was wrong.

## Outcome

The offline cheaper-first tuner worked as designed. At LGSSM `T=10,N=1024`
it rejected `(sinkhorn_steps,balance_steps)=(20,2)` and selected `(20,3)`.
The selected pair passed a fresh 16-seed T=10 claim and an independent
one-seed T=50 resource witness, but failed the fresh 16-seed T=50 numerical
claim. The conditional nonlinear phase was therefore not executed.

This is a numerical transfer failure, not an infrastructure failure. The T=50
claim was finite, bitwise replayable, GPU/XLA/TF32 compliant, well below the
8 GiB cap, and completed in 127.30 seconds. Four of 800 seed-time resets failed
only the maximum row-marginal gate. All column-total-variation errors passed.

## Decision Table

| Decision | Primary criterion status | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the offline tuner as an engineering harness | PASS: cheaper-first ordering, fixed-pair binding, fresh partitions, structured artifacts, GPU/XLA/TF32 and `K=N` checks all executed | No harness, replay, nonfinite, resource, work-count, or identity veto | Search grid and seed coverage are bounded | Keep the tuner and its regression tests | Universal optimality or default readiness |
| Accept `(20,3)` for the tested T=10 domain | PASS: tuning partitions and fresh 16-seed T=10 claim | `TV_col=1.3848e-5`, `E_row=0.004685`, all charts/resets valid | Finite seed evidence only | Use as a T=10 candidate, not a universal schedule | T=50 or nonlinear validity |
| Reclassify `(20,3)` T=50 execution | Untuned baseline FAIL: four of 800 resets exceed `E_row<=0.01` | Worst `E_row=0.0267197`; `TV_col=3.8266e-5` passes; finite and replayable | It does not distinguish which T=50-specific controls will be selected | Tune the T=50 scope on fresh calibration/validation seeds and reserve these seeds as historical baseline evidence, not tuning data | No conclusion about a properly tuned T=50 run |
| Correct the nonlinear stop | Previous stop was wrong planning | A failure in an untuned LGSSM T=50 baseline is not a continuation veto for independent model tuning | Each nonlinear route still needs an implemented route-specific tuner | Give every model/route/scope its own tuning phase and claim partition | No all-model failure or success conclusion |

## Detailed Results

| Node | Pair | Result | `TV_col` max | `E_row` max | Wall time | Peak allocator |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| T=10 tuning, 16 seeds | `(20,2)` | FAIL validation | `2.8695e-5` | `0.0205871` | `30.65 s` | `0.895 GB` |
| T=10 tuning, 16 seeds | `(20,3)` | PASS calibration and validation | `1.1880e-5` | `0.00832105` | `23.87 s` | `0.895 GB` |
| T=10 fresh claim, 16 seeds | `(20,3)` | PASS | `1.3848e-5` | `0.00468516` | `25.39 s` | `0.900 GB` |
| T=50 resource witness, one seed | `(20,3)` | PASS | `2.0201e-5` | `0.00532150` | `35.54 s` | `0.900 GB` |
| T=50 fresh claim, 16 seeds | `(20,3)` | FAIL row gate | `3.8266e-5` | `0.0267197` | `127.30 s` | `0.942 GB` |

The four failing T=50 reset states were:

| Estimator seed | Zero-based time | `TV_col` | `E_row` |
| ---: | ---: | ---: | ---: |
| 81720 | 16 | `2.4680e-5` | `0.0241443` |
| 81723 | 20 | `3.8266e-5` | `0.0267197` |
| 81728 | 3 | `1.9091e-5` | `0.0111896` |
| 81734 | 47 | `1.8536e-5` | `0.0103934` |

The T=50 node used one `StatelessWhile`, no Python horizon unrolling, one
Sinkhorn state, one terminal-balance state, and one transport sweep per time
step, with zero diagnostic solver reconstructions and zero marginal tile
sweeps. The 8192 MiB TensorFlow limit was active. XLA compilation and runtime
were therefore feasible; the failure is the fixed pair's row-marginal accuracy.

## Nonlinear Compatibility Inventory

| Model family | Current route | Same OT controls? | Disposition |
| --- | --- | --- | --- |
| Latent pre-clipping SIR | `ledh_contract_e_latent_sir_tf` | Yes, exposes `steps` and `balance_steps` | Conditional adapter required; current route is float64/candidate and reports older residual fields |
| Scalar SV | Contract E--TP | No | Separate TP feature/order tuner |
| Generalized SV | Contract E--TP | No | Separate TP feature/order tuner |
| KSC-SV | Contract E--TP | No | Separate TP feature/order tuner |
| Predator-prey | Contract E--TP | No | Separate TP feature/order tuner |

Calling TP basis, feature, order, or lookahead settings “Sinkhorn tuning” would
be wrong relative to those implementations.

## Evidence And Verification

Primary result:
`docs/benchmarks/artifacts/ledh_offline_ot_cheaper_first_tuning_20260718/attempt01/campaign_result.json`.
The same directory contains the selection artifact, every candidate/node JSON,
and the run manifest. The run used commit `9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b`
with recorded source hashes, TensorFlow 2.19.1, RTX 4080 SUPER, float32/TF32,
XLA JIT, seeds, command, and wall time.

Post-run focused verification: Python compilation passed; tuner tests passed
`5/5`; `git diff --check` passed. Structured artifact review confirmed that
the selected pair matches preparation and execution identities and that the
T=50 failure is confined to the four row-marginal states above.

The executed v1 node status string combined direct-gate and node-cap failures.
The artifact's separate `within_node_cap=true` and residual fields make the
classification unambiguous. The post-run harness now emits
`FAIL_DIRECT_GATE` and `FAIL_NODE_CAP` separately; no scientific rerun was
needed for this reporting-only repair.

## Inference Status

| Question | Status |
| --- | --- |
| Hard veto screen | T=10 tuned scope passes; untuned T=50 baseline fails the row-marginal screen |
| Statistically supported ranking | None; the iteration ladder is deterministic numerical selection on fixed seed sets |
| Descriptive-only differences | Runtime, allocator use, and continuous residual margins among passing candidates |
| Default readiness | No; there are no cross-scope LEDH defaults |
| Next evidence needed | Fresh T=50 calibration/validation plus a new untouched T=50 claim, using the same cheaper-first search and unchanged gates |

## Post-Run Red Team

The strongest alternative explanation is horizon-domain mismatch: the T=10
state distribution did not include row-marginal cases as difficult as the four
seen at T=50. This weakens transfer of the selected pair, not the tuner order or
Contract E derivation. A fresh T=50 tuning ladder that passes new calibration,
validation, and untouched claim seeds would overturn the transfer rejection.
The weakest evidence is tail coverage from only 16 T=50 seeds; consequently no
universal failure rate or model-wide probability statement is made.
