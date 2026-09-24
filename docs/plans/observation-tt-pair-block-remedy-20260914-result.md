# Pair-block remedy: sampler repaired, filtering promotion rejected

The completed comparison passes the declared mean and log-evidence reference
screens in d1 and d4. It still rejects promotion: the pair proposal has larger
observed conditional mean errors than cheap heuristics in ordinary/large d1
observations and than the SGQF Gaussian in large d4 observations. Four particle
seeds on one observation sequence do not establish a statistical ordering.
This is a valid diagnostic rejection of the tested configuration, not rejection
of the pair representation or observation-aware TT research direction.

## Evidence and implementation diagnosis

The numerical target is the conditional density
`q(u|v) = (h(u,v)^2 + tau) rho(u) / (Z(v) + tau)`, where
`Z(v) = integral h(u,v)^2 rho(u) du`. Campaign-01 computed a different quantity
for Z when the core bond dimension exceeded one. Its contraction
`nab,nakc,nald,kl->ncd` reused index a and summed b independently. The correct
Gram recurrence uses `nab,nakc,nbld,kl->ncd`. With a two-by-two positive semidefinite matrix
`E = [[1,-2],[-2,4]]` and core column `(1,0)`, the old contraction gives -1
while the required quadratic form gives +1. MathDevMCP checked this explicit
counterexample and a separate quadratic-form identity.

Replay-01 reproduced campaign-01 at d4 t=2, seed 210102, saving the complete
previous particles, charts, cores and random inputs. The failing conditional
mass was -0.00869397759638696. The bracket was valid and CDF residual was
5.551115e-16; changing bisection or its threshold would not repair the error.
Only the contracted index was corrected. The rejection guard is unchanged;
the exception now retains time, seed and diagnostics.

An independent test expands all Hermite coefficient combinations and computes
their squared norm. It failed before repair and passes afterward, in eager
and compiled CPU reference execution. Thirteen tests in the two focused suites
pass, including independent CDF, zero-polynomial, marginal and row-sampling
checks. GPU replay-02 then completed all four d4 seeds using the saved fits.
The master retry and replay-02 produce exactly equal serialized d4 pair runs.
The checked consumer chain is master particle filter → observation-guided pair
sampler → compiled pair sampler → conditional normalizer. This combines
consumer execution with the independent integral check.

Campaign-02 reloaded all fits without retraining; all six d1/d4 proposal-file
hashes equal campaign-01. Fixture identity and the complete frozen scope were
checked before execution, and rebuilt SGQF charts matched saved charts. The
defective routine is absent from fitting and retained-density construction.
The d1 pair runs are exactly equal to the original runs: their one-core
contraction has no nontrivial internal bond and was unaffected. They were
recomputed to establish this rather than assumed valid.

The [terminal summary](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-02/terminal_summary.json)
preserves the checks, full comparison and budget. The failed campaign remains
archived; its partial RUNNING result is superseded by its FAILED terminal
manifest. The corrected completion is
[campaign-02](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-02/run_manifest.json).

## Filtering results

Both dimensions use T=20, N=512, degree/rank 3, 1024 rows, four fitting sweeps
and 128 proximal steps. Seeds are 2101–2104; fitting/reference seeds are
64100/74100. The original diagnostic-02 fixture and selection remain frozen.
d4 selected the configuration; d1 is a transfer diagnostic. The repaired run
reuses exposed observations and is not a fresh holdout.

| Dimension | Pair mean log evidence | Particle-seed SE | Reference | Reference gap | Mean/evidence screens |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | -20.975034 | 0.050027 | -20.972621 | -0.002413 | PASS / PASS |
| 4 | -63.677796 | 0.071604 | -63.641241 | -0.036555 | PASS / PASS |

d1 uses independent scalar quadrature. d4 uses four transition-filter references
with 32768 particles, reference log-evidence SE 0.016875. The predeclared
screens are `abs(gap) <= 3.182446 * combined SE + 0.15` for log evidence and
the analogous coordinate/timewise mean screen with additive `0.15*sigma`.
These are diagnostic screens, not simultaneous confidence guarantees.
All six methods pass both screens in both dimensions.

The conditional RMSE below compares the four-seed average filtering mean with
the reference, pooling only the indicated observations and state coordinates.
Regimes use `max(abs(y)/beta) <= 0.5`, between 0.5 and 2, and `>=2`.
d1 has 7/10/3 observations in the three regimes; d4 has 1/9/10. In particular,
d4 near-zero evidence consists of one timepoint, not replicated sequences.

| d | Proposal | Near zero | Ordinary | Large |
| --- | --- | ---: | ---: | ---: |
| 1 | Transition bootstrap | 0.030203 | 0.018188 | 0.014385 |
| 1 | Stationary prior | 0.036207 | 0.025927 | 0.029007 |
| 1 | SGQF Gaussian | 0.079116 | 0.050992 | 0.018675 |
| 1 | Scalar predictive TT | 0.013610 | 0.024908 | 0.016814 |
| 1 | Scalar guided TT | 0.015054 | 0.026387 | 0.021022 |
| 1 | Pair TT | 0.016192 | 0.024908 | 0.019715 |
| 4 | Transition bootstrap | 0.090113 | 0.035104 | 0.054022 |
| 4 | Stationary prior | 0.073189 | 0.070619 | 0.104458 |
| 4 | SGQF Gaussian | 0.045774 | 0.044386 | 0.047386 |
| 4 | Scalar predictive TT | 0.065791 | 0.052730 | 0.058269 |
| 4 | Scalar guided TT | 0.065547 | 0.057570 | 0.059305 |
| 4 | Pair TT | 0.027447 | 0.034611 | 0.048544 |

The heuristic set was constructed before execution: transition bootstrap uses
the exact conditional dynamics, stationary prior tests whether conditioning
helps, and SGQF Gaussian tests whether TT fitting adds value beyond the moment
guide. The pair route loses descriptively to transition in ordinary and large
d1 observations, to SGQF in large d1 observations, and to SGQF in large d4
observations. The last comparison is 0.048544 versus 0.047386; it is a promotion
veto under the frozen rule, not statistically supported inferiority.

All pair draws/densities and brackets were finite/valid. Minimum polynomial
conditional masses were 0.030365 (d1) and 0.007332 (d4); maximum CDF residuals
were 5.55e-16 and 9.99e-16. The scalar guided frozen-proposal analytical-score
check passes; pair gradients and derivatives through refitting were not checked.

## Representation and fitting remain separate questions

The earlier diagnostic-02 common Gaussian audit gives d4 relative amplitude RMS
0.442461 for grouped scalar TT, 0.108114 for interleaved scalar TT, 0.116883
for the selected weighted rank-three pair fit, and 0.131604 for its allowed
eight-sweep/256-step repair. These are descriptive, and different initializers
confound attribution to representation. Pair cores use more coefficients at
the same nominal rank. The test uses a Gaussian incoming law at t=1; downstream
fitting uses recursive retained TT densities.

The selected pair validation residual was 0.065773 versus 0.068324 after the
solver repair. The baseline KKT residual was 0.010057, and the repair gave
0.011450; neither passed the 0.001 convergence diagnostic. Validation selected
the baseline; common-audit errors did not select a configuration. The single
allowed repair was exercised, so no claim of converged fitting is justified.
Large later recursive fit residuals remain a fitting/representation concern.

A remaining default-audit gap is the inherited 0.05 defensive mixture mass.
The implementation sets `tau = Z * 0.05 / 0.95`, so the joint approximation
assigns five percent mass to the Gaussian base. Positive tau supplies full
support; the particular five-percent value has no calibration evidence in this
campaign. It is a frozen hypothesis, not a reviewed numerical default. A large
base component can waste proposals in uninformative regions, while a small one
can leave poor conditional coverage; the recorded conditional masses, CDF
checks and regime errors expose some consequences but do not calibrate this
choice. Future promotion needs a dedicated coverage/non-harm justification
using fresh calibration. This limitation is additional to the observed
heuristic veto and is not repaired by the successful index correction.

## Decisions and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the localized index correction | Independent coefficient integral agrees | Old formula refuted; corrected finite/bracket checks pass | General numerical conditioning beyond this scope | Retain correction and regression | Universal sampler certification |
| Withhold pair promotion | Both reference screens pass | Heuristic underperformance in stated regimes | Four seeds, one sequence, d1 transfer, imperfect fit convergence | Document completion; design fresh calibration only in a new campaign | Superiority, default or HMC readiness |
| Keep research direction open | Pair structure remains mathematically viable | No unresolved sampler continuation veto after repair | Degree/rank, solver convergence and sequence generalization | Next study should separate solver realization from representation capacity | Rejection of paired representation |

| Inference category | Status |
| --- | --- |
| Hard veto screen | Original implementation rejected; corrected sampler passes tested validity checks. Heuristic promotion veto remains. |
| Statistically supported ranking | None. Viable numerical candidates are indistinguishable in ranking under this evidence. |
| Descriptive-only differences | All regime RMSE comparisons, fitting errors, seed SEs, runtime and apparent scalar/pair differences. |
| Default-readiness | Rejected; no HMC or pair total-gradient evidence. |
| Next evidence needed | Fresh calibration with a justified solver/capacity protocol; multiple untouched observation sequences and predeclared paired uncertainty analysis of conditional errors. |

Post-run red team: the strongest alternative explanation for the apparent pair
advantage over scalar TT is initialization/extra coefficients plus particle
variation, not pairing alone. A converged matched-capacity comparison across
independent sequences could overturn the interpretation. The weakest evidence
is conditional generalization, especially the single d4 near-zero observation.
Passing a broad reference tolerance does not establish filtering efficiency.

## Resources, mathematics and manuscript

The repaired GPU/XLA campaign used tftwogpu, TensorFlow 2.20.0-dev0+selfbuilt,
float64, RTX5080, threads 2/1 and verified memory growth; exact command,
dependency/data hashes, seeds, device details, allocator use and wall time are
in its manifest. Replay-01 took 14.598617 s; replay-02 12.479128 s; campaign-02
48.464704 s. Two CPU tests are conservatively charged 120 s in total because
startup was not separately metered. About 1824.933 s remain under that ledger,
but all three full-launch slots and the allowed solver repair are consumed.
No new experiment is launched merely because time remains.

Eight MathDevMCP theorem audits remain inconclusive and the previous document
tree audit failed. Nine scoped identities, one weight identity and the new
quadratic-form identity were verified; the old formula has a checked numeric
counterexample. These checks do not certify the full manuscript. The existing
eight proposition proofs are preserved; the forward Gram derivation and
counterexample explain the implementation correction at its mathematical point
of use. The protected pre-pair and pre-recovery sources remain unchanged.
The manuscript targets a reader familiar with Gaussian filtering, importance
sampling and tensor notation; the new unit follows conditional normalization,
its implementation failure, and the resulting statistical comparison.
Human assessment of the revised prose remains pending; continued drafting and
rendered inspection are authorized, with no claim of human acceptance.

The [48-page manuscript PDF](artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.pdf)
now includes the corrected derivation and completed comparison. Final pdflatex
passes succeeded with no overfull boxes, undefined references or rerun request.
Existing math-in-bookmark warnings in older headings remain; the page content
is unaffected. Rendered review covered all pages at overview scale and the
abstract/pair derivation/results at reading scale. The weighted objective was
split into three aligned rows, the long result path now wraps, and the main
alternative-explanation paragraph stays together around the table float.
The [preservation audit](artifacts/observation-tt-pair-block-20260914/manuscript-preservation-audit.json)
checks both protected sources: no labels, citations, paths or inline mathematics
were removed; all display mathematics is retained, with the single reviewed
layout-only change. Build logs, PDF/source hashes and render images are in
[the rendered-review record](artifacts/observation-tt-pair-block-20260914/rendered-review/review.md).
