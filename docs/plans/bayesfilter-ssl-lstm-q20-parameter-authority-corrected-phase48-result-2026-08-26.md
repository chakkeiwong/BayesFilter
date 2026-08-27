# Corrected Parameter-Authority Phase 48 Result

Date: 2026-08-26  
Version: `v3.0-independent-proposal-mutation`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase48-subplan-2026-08-26.md`  
Status: `PASS_V3_0_INDEPENDENT_MH_REPORT_REPAIR_TRIGGERED`

## Question and scope

Phase 48 compared identity mutation with an independent-proposal
Metropolis-Hastings (MH) kernel after each nonterminal tempering stage. The
candidate was sampled from the declared defensive mixture `q(theta)` in the
four-dimensional parameter measure, and the acceptance ratio used the same
`q` and the q=20 target `V`. The 60-dimensional UKF state remained internal.

This was a finite mutation diagnostic. It was not a posterior, whitening,
HMC, LEDH, or NeuTra-promotion run.

## Execution and hard-gate evidence

The analytic fixture passed as `PASS_V3_0_INDEPENDENT_MH_FIXTURE`. The q=20
GPU boundary passed as `PASS_V3_0_INDEPENDENT_MH_BOUNDARY` after reproducing
the Phase 47 initial clouds and identity endpoints exactly in all three
replicates. The CPU-hidden report passed as
`PASS_V3_0_INDEPENDENT_MH_REPORT`.

| Gate | Result |
|---|---|
| target and measure | passed; q=20 target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d727`, measure `theta_R4` |
| independent-MH algebra fixture | passed; beta-zero ratio identity and beta-one movement |
| candidate proposal law | passed; candidates sampled from and scored by the same defensive mixture |
| target/status handling | passed; no invalid candidate was accepted |
| pairing and replay | passed; same initial clouds/resampling seeds and Phase 47 identity final hashes |
| tensor validity | passed; all retained endpoints finite with shape `[256,4]` |
| GPU policy | passed; two RTX 4080 SUPER devices, memory growth verified before logical-device use, XLA and TF32 enabled |
| artifact integrity | passed; unique output root and source/command hashes recorded |

The q=20 boundary wall time was `1521.3247332019964 s`; the fixture and
report wall times were `0.4511733159888536 s` and `0.3468089979724027 s`.
The boundary receipt SHA-256 is
`65a4a6e7b4123fc8d73fef945b81345c011e7892594cc4079b8136f44a08b275`.

## Descriptive result

The report branch was `independent_mh_does_not_reduce_variability`. The
independent-MH arm moved in every replicate: mean move fractions were
`0.455566`, `0.422852`, and `0.469727`, with zero invalid candidates accepted.
The paired finite spreads were:

| Diagnostic | Identity spread | Independent-MH spread | Relation |
|---|---:|---:|---|
| weighted theta mean[0] | 1.256141 | 1.383798 | larger |
| covariance off-diagonal maximum | 0.836754 | 1.972146 | larger |
| negative-mode mass | 0.164751 | 0.062855 | smaller |
| retained root count | 21 | 13 | smaller |
| weighted ESS fraction | 0.040061 | 0.021027 | smaller |

These are three finite replicates with two proposals per nonterminal stage and
no uncertainty model. They are descriptive only and do not rank the kernels.

## Mathematical and MathDevMCP audit

With `b=beta`, `qc=log q(theta')`, `qx=log q(theta)`, `vp=V(theta')`, and
`vx=V(theta)`, direct substitution gives

`((1-b)*qc+b*vp)+qx-qc-((1-b)*qx+b*vx)`
`= b*((vp-qc)-(vx-qx))`.

The first MathDevMCP invocation used unconstrained symbols named
`bridge_current` and `bridge_prime`; its finite counterexample was therefore
not applicable because it did not include their defining equations. A
re-encoded expression with the bridge definitions substituted was certified by
the SymPy backend (`lhs - rhs = 0`). The separate code-structure query was
reported as structurally inconclusive because it was phrased in prose rather
than the exact expression; the implementation was instead checked by the
explicit fixture, source inspection, and the q=20 receipt. MathDevMCP output
is audit evidence, not a general proof oracle.

## Decision and inference status

| Decision | Primary criterion | Status | Limitation | Next action | Not concluded |
|---|---|---|---|---|---|
| retain theta target | target/status/measure/pairing gates | pass | none | retain parameter authority | posterior correctness |
| promote IID whitening | finite mutation clouds | veto | finite clouds do not identify a Gaussian law | keep whitening closed | IID Gaussian law |
| promote independent MH as default | paired spread | defer | three replicates and two steps; no uncertainty model | test mutation depth under the same proposal | superiority/default readiness |
| reopen HMC or canonical LEDH | density and downstream gates | veto | whitening/posterior gates remain closed | keep routes closed | HMC/LEDH readiness |

| Inference class | Status |
|---|---|
| hard veto screen | passed after the v3.0 fixture/boundary/report chain |
| statistically supported ranking | none |
| descriptive-only difference | nonlocal MH moved particles but did not reduce the declared spread vector |
| default readiness | not ready |
| next evidence | a depth diagnostic that distinguishes insufficient mutation depth from proposal-support bias |

## Research-direction classification and repair

The valid v3.0 result is a candidate-method/depth failure, not a target,
measure, data, or harness failure. Two accepted proposals per stage are not
enough to conclude that independent MH cannot mix the finite cloud: the later
stage move fractions remained nonzero. The next smallest discriminating phase
therefore keeps the exact target, proposal, seeds, and identity comparator but
increases the independent-MH depth to eight proposals per nonterminal stage.

No result from this phase authorizes whitening, posterior claims, HMC,
canonical LEDH, NeuTra training, or a default change.

## Artifacts and manifest

- Fixture: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase48-independent-proposal-mutation/fixture/`
- Boundary: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase48-independent-proposal-mutation/q20-paired/result.json`
- Report: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase48-independent-proposal-mutation/report/result.json`
- Runner: `docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase48_2026_08_26.py`
- Reporter: `docs/benchmarks/report_ssl_lstm_q20_parameter_authority_corrected_phase48_2026_08_26.py`

The run manifests record the dirty-tree state, TensorFlow 2.20.0, the hidden
GPU report lane, the trusted GPU memory-growth policy for the boundary, seeds,
source hashes, commands, and artifact roots.
