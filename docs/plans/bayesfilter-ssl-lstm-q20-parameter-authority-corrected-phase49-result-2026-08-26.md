# Corrected Parameter-Authority Phase 49 Result

Date: 2026-08-26  
Version: `v3.1-independent-proposal-depth`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase49-subplan-2026-08-26.md`  
Status: `PASS_V3_1_INDEPENDENT_MH_DEPTH_REPORT_REPAIR_TRIGGERED`

## Question and scope

Phase 49 increased the independent-proposal MH depth from two to eight
proposals at each nonterminal annealing stage. The target remained the
batch-native q=20 SSL-LSTM log target in `theta in R^4`; the 60-dimensional UKF
state remained internal. The initial cloud, defensive-mixture base density,
annealing schedule, resampling seeds, and identity replay boundary were
unchanged from Phase 47/48. No NeuTra update, HMC run, whitening admission, or
canonical LEDH route was launched.

This is finite support evidence. It is not a posterior, convergence,
whitening, mode-discovery, or superiority result.

## Hard-gate evidence

The repeated analytic fixture passed as `PASS_V3_1_INDEPENDENT_MH_DEPTH_FIXTURE`.
The trusted GPU/XLA boundary passed as
`PASS_V3_1_INDEPENDENT_MH_DEPTH_BOUNDARY`; all three initial clouds and the
Phase 47 identity endpoint hashes were reproduced. The CPU-hidden report
passed as `PASS_V3_1_INDEPENDENT_MH_DEPTH_REPORT` with branch
`depth8_does_not_reduce_variability`.

| Gate | Result |
|---|---|
| target, measure, and depth | passed; target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d727`, measure `theta_R4`, eight steps |
| repeated MH algebra fixture | passed; beta-zero identity over eight repetitions and beta-one movement |
| candidate/status handling | passed; no invalid candidate was accepted |
| pairing and replay | passed; exact initial clouds, resampling seeds, and Phase 47 identity endpoint hashes |
| finite tensor/artifact checks | passed; all retained endpoints finite with shape `[256,4]` and unique output root |
| GPU policy | passed; two RTX 4080 SUPER devices, memory growth before logical-device use, XLA and TF32 enabled |

The fixture, boundary, and report wall times were `0.48292820202186704 s`,
`5568.329515483987 s`, and `0.3576144660473801 s`. The boundary source and
artifact hashes are recorded in its `run_manifest`.

## Descriptive result

The depth-eight independent arm had nonzero movement at every nonterminal
stage. Mean move fractions by replicate were approximately `0.443237`,
`0.438354`, and `0.444214`; the candidate-safe fraction remained close to the
declared `0.20`, and zero invalid candidates were accepted.

The report compared three paired depth-eight rows with the frozen Phase 48
depth-two independent-MH rows:

| Diagnostic spread | Depth 8 | Frozen depth 2 | Relation |
|---|---:|---:|---|
| weighted theta mean[0] | 0.430109 | 1.383799 | smaller |
| covariance off-diagonal maximum | 1.214406 | 1.972146 | smaller |
| negative-mode mass | 0.069278 | 0.062855 | larger |
| retained root count | 10 | 13 | smaller |
| weighted ESS fraction | 0.030223 | 0.021027 | larger |

Because the predeclared three-metric condition failed on negative-mode mass,
the branch is `depth8_does_not_reduce_variability`. These are three finite
replicates with no uncertainty model and are descriptive only; no kernel is
ranked.

## Mathematical audit

With `q` the fixed defensive-mixture density and
`pi_beta(theta) proportional to q(theta)^(1-beta) exp(beta V(theta))`, each
independent proposal from `q` used

`log a = min(0, bridge_q(theta') - bridge_q(theta) + log q(theta) - log q(theta'))`,

where `bridge_q=(1-beta)log q+beta V`. Direct substitution gives

`log a (before min) = beta[(V'-log q')-(V-log q)]`.

The MathDevMCP invocation with unconstrained bridge symbols returned an
inapplicable counterexample. After substituting the bridge definitions, the
SymPy backend certified `lhs-rhs=0`. The finite fixture and source/receipt
checks support implementation alignment; they do not prove finite-run
invariance or a population limit.

## Decision and inference status

| Decision | Primary criterion | Status | Limitation | Next action | Not concluded |
|---|---|---|---|---|---|
| retain theta target | target/status/measure/pairing gates | pass | none | retain parameter authority | posterior correctness |
| promote IID whitening | finite mutation clouds | veto | finite clouds do not identify a Gaussian law | keep whitening closed | IID Gaussian law |
| promote depth-eight MH | paired spread against frozen depth two | defer | three replicates, no uncertainty model, mixed metrics | test proposal-support construction | superiority/default readiness |
| admit HMC or canonical LEDH | density and downstream gates | veto | whitening/posterior gates remain closed | keep routes closed | HMC/LEDH readiness |

| Inference class | Status |
|---|---|
| hard veto screen | passed |
| statistically supported ranking | none |
| descriptive-only difference | more depth produced movement and reduced some spreads, but did not repair the primary mode/ESS variability condition |
| default readiness | not ready |
| next evidence | a proposal-support/overlap repair under the same target and theta measure |

## Research-direction classification and repair

The valid result is a candidate-depth failure, not a target, measure, data, or
harness failure. Nonzero movement at all four active stages means that a
two-step-depth explanation is incomplete, while the mixed spread relations
leave proposal-support overlap unresolved. The next phase therefore keeps the
base annealing density and target fixed and tests a separately evaluated
full-support candidate law with the exact non-symmetric MH correction.

No result from Phase 49 authorizes whitening, posterior claims, HMC,
canonical LEDH, NeuTra training, or a default change.

## Artifacts

- Fixture: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase49-independent-proposal-depth/fixture/`
- Boundary: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase49-independent-proposal-depth/q20-paired/result.json`
- Report: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase49-independent-proposal-depth/report/result.json`
- Runner: `docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase49_2026_08_26.py`
- Reporter: `docs/benchmarks/report_ssl_lstm_q20_parameter_authority_corrected_phase49_2026_08_26.py`

