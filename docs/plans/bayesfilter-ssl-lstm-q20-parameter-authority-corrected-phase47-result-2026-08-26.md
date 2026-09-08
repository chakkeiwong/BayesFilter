# Corrected Parameter-Authority Phase 47 Result

Date: 2026-08-26  
Version: `v2.9-invariant-mutation-diagnostic`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase47-subplan-2026-08-26.md`  
Status: `PASS_V2_9_MUTATION_REPORT_REPAIR_TRIGGERED`

## Question and scope

Phase 47 tested whether two symmetric theta-space random-walk Metropolis
steps after each nonterminal annealing stage reduce finite support variability
relative to identity mutation. Both arms used the same q=20 target in
`theta in R^4`, the same defensive proposal density, schedule, initial cloud,
resampling seeds, and target/status API. The 60D UKF state remained internal.

This was a finite invariant-mutation diagnostic. It was not a posterior,
whitening, HMC, LEDH, or NeuTra-promotion run.

## Execution and repair record

The first q=20 attempt completed all per-replicate computations but returned
`PHASE47_MUTATION_BOUNDARY_FAIL`. Its arm gate map incorrectly included the
metadata field `terminal_resampling: false` in `all(gates.values())`; therefore
otherwise-valid arms were rejected by construction. The attempt is retained at
`phase47-invariant-mutation/q20-paired/` with no scientific interpretation.

The harness repair removed that metadata field from the pass conjunction while
retaining it in the manifest. The unchanged experiment was rerun at
`phase47-invariant-mutation/attempt-02/q20-paired/` and passed
`PASS_V2_9_MUTATION_BOUNDARY`. The CPU-hidden report then passed
`PASS_V2_9_MUTATION_REPORT`.

## Hard-gate evidence

| Gate | Result |
|---|---|
| analytic MH fixture | passed; `PASS_V2_9_MH_FIXTURE` |
| target and measure | passed; q=20 target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d727`, measure `theta_R4` |
| target/status validity | passed for every retained endpoint and candidate status gate |
| paired initial cloud | passed; exact initial tensor hashes matched between arms in all three replicates |
| paired resampling seeds | passed; same deterministic offsets in all three replicates |
| finite tensors and summaries | passed; all `[256,4]` tensors and weights finite |
| GPU execution | passed; two RTX 4080 SUPER devices, memory growth verified before logical-device use, XLA and TF32 enabled |
| artifact integrity | passed; unique roots, no overwrite, source and command hashes recorded |

The repaired GPU boundary wall time was `1522.8043500939966 s`. The fixture
wall time was `0.3921517609851435 s`; the CPU report wall time was
`0.3203752610133961 s`.

## Descriptive paired receipt

The report branch was
`mh_rejuvenation_does_not_reduce_variability`. The independent-replicate
spreads were:

| Diagnostic | Identity spread | MH spread | Relation |
|---|---:|---:|---|
| weighted theta mean[0] | 1.256141 | 1.152790 | MH no larger |
| covariance off-diagonal max | 0.836754 | 2.428978 | MH larger |
| negative-mode mass | 0.164751 | 0.209962 | MH larger |
| retained root count | 21 | 9 | MH no larger |
| weighted ESS fraction | 0.040061 | 0.112723 | MH larger |

The MH move fractions were `0.193848`, `0.191895`, and `0.185547` in the
three replicates, so the negative branch is not explained by zero movement.
These are descriptive values from three finite replicates, with two mutation
steps per nonterminal stage and no uncertainty model. They do not rank the
methods.

## Mathematical validity

At beta `beta`, the bridge is

`pi_beta(theta) proportional to q(theta)^(1-beta) exp(V(theta))`.

For the symmetric proposal `theta_prime = theta + sigma xi`, the implemented
log acceptance ratio is

`min(0, log pi_beta(theta_prime) - log pi_beta(theta))`.

Invalid target/status candidates are rejected. The analytic standard-normal
fixture passed finite, acceptance, movement, mean, and second-moment screens.
This establishes an implementation check for the finite kernel, not a
finite-run invariance theorem for the q=20 target.

## Decision table

| Decision | Primary criterion | Status | Veto or limitation | Next action | Not concluded |
|---|---|---|---|---|---|
| retain theta target | target/status/measure and pairing gates | pass | none | retain parameter authority | posterior correctness |
| promote IID whitening | finite mutation clouds | veto | finite clouds do not identify a Gaussian law; residual support variability persists | keep whitening closed | IID Gaussian law |
| promote local MH as default | paired spread diagnostic | defer | three replicates, two steps, no uncertainty interval | test a nonlocal invariant proposal in a new scope | superiority/default readiness |
| reopen HMC or canonical LEDH | density and downstream gates | veto | no whitening or posterior gate passed | keep routes closed | HMC/LEDH readiness |

## Inference-status table

| Evidence class | Status |
|---|---|
| hard veto screen | passed after the harness repair |
| statistically supported ranking | none |
| descriptive-only differences | MH moved but did not reduce the declared support-spread vector |
| default readiness | not ready |
| next evidence needed | independent-proposal mutation and longer validation with uncertainty/downstream checks |

## Post-run red team

The strongest alternative explanation is scale/locality: an isotropic proposal
with `sigma=0.35` may move within a mode without crossing the separated
proposal components, so this result does not test all invariant rejuvenation
kernels. The weakest evidence is the three-replicate, two-step design with no
Monte Carlo interval. Evidence that would overturn the current repair decision
is a fresh, paired independent-proposal kernel using the exact q density that
reduces support variability and remains valid under an unchanged target and
measure. Even that result would remain role-limited until downstream validation.

## Research-direction classification

The harness failure was an implementation artifact and was repaired. The valid
Phase 47 result is a candidate-method failure for this local MH configuration,
not a target, measure, data, or research-direction failure. The next phase is
therefore a bounded repair rather than a continuation veto.

## Artifacts and manifest

- Failed attempt: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase47-invariant-mutation/q20-paired/result.json`
- Repaired boundary: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase47-invariant-mutation/attempt-02/q20-paired/result.json`
- Report: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase47-invariant-mutation/attempt-02/report/result.json`
- Fixture: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase47-invariant-mutation/fixture/result.json`
- Runner: `docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase47_2026_08_26.py`
- Reporter: `docs/benchmarks/report_ssl_lstm_q20_parameter_authority_corrected_phase47_2026_08_26.py`
- Fixture runner: `docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase47_fixture_2026_08_26.py`

The repaired boundary receipt SHA-256 is
`0a3d3008a279603fff4b67e18af46d5074fc357fe2b4f5481093332b7e2fe690`; the
report receipt SHA-256 is
`ca0047fe019944d9cafbd212bb45c71c22489c30fe9e9d41d1158e7786cddf89`.

MathDevMCP was not used as a proof oracle for this implementation-specific
kernel. The algebra above is the declared derivation, and the fixture is the
finite code check; neither supports an IID, posterior, or whitening claim.
