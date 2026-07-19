# NeuTra Cross-Repository Model Evidence Ledger

Date: 2026-07-17

Status: `ACTIVE_COUNT_AND_COST_BOUNDED_DIAGNOSTIC_POLICY`

## Purpose

Report how broadly learned NeuTra has actually been exercised across
BayesFilter and `/home/chakwong/python` without inflating the count with seeds,
dimensions, transport arms, tuning candidates, or implementation canaries.

A model family is counted once for shared scientific dynamics or target
structure. A posterior-target configuration is counted separately when the
filter, solution order, or likelihood construction materially changes the
posterior. A configuration is counted as tested only when a learned NeuTra
transport reached transformed HMC and produced retained sampler diagnostics.
Target preflights, score certificates, training-only runs, tiny canaries, and
launch-readiness artifacts are excluded.

## Counts

| Unit | Count | Interpretation |
| --- | ---: | --- |
| Distinct model families tested | `9` | learned NeuTra reached transformed HMC |
| Distinct posterior-target configurations tested | `12` | materially different likelihood/filter/solution configurations |
| Clean or strong historical diagnostic configurations | `9` | passed the preserved result's own primary sampler/quality screen |
| Qualified or marginal configurations | `3` | useful evidence, but with an explicit budget, parent-artifact, telemetry, or convergence caveat |

These are conservative counts. They do not count repeated seeds, LGSSM
fixtures, target dimensions, HMC candidates, affine controls, or multiple
transport arms as additional models.

## Evidence Ledger

| Family | Configuration | Historical status | Evidence anchor |
| --- | --- | --- | --- |
| LGSSM | exact Kalman likelihood | clean pass on the original fixture plus an independent new fixture | `docs/plans/bayesfilter-neutra-hmc-core-consolidation-and-robustness-reset-memo-2026-07-15.md` |
| Predator-prey | fixed UKF likelihood | clean six-physical-mean same-target confirmation | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-terminal-result-2026-07-16.md` |
| Predator-prey | fixed SGQF likelihood | clean six-physical-mean same-target confirmation | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-terminal-result-2026-07-16.md` |
| Parameterized SIR | fixed SGQF likelihood | clean three-physical-mean same-target confirmation | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-terminal-result-2026-07-16.md` |
| Funnel | paper-scale dense-IAF target | clean under the preserved local Gate-1 quality screen | `/home/chakwong/python/docs/plans/neutra-gate1-two-track-closure-result-2026-05-05.md` |
| Ill-conditioned Gaussian | paper-scale dense-IAF target | strong local Gate-1 diagnostics | `/home/chakwong/python/docs/plans/neutra-gate1-two-track-closure-result-2026-05-05.md` |
| German logistic regression | gamma-scales dense-IAF target | qualified: diagnostics were good, but the conservative paper-budget ratio gate remained open | `/home/chakwong/python/docs/plans/neutra-gate1-two-track-closure-result-2026-05-05.md` |
| NK-like analytic surrogate | smooth changing-Hessian target | strong one-seed paper-style baseline; zero divergences, R-hat `1.00158`, minimum ESS `2541.5` | `/home/chakwong/python/docs/plans/neutra-gate3-nk-mild-score-matching-closure-2026-06-04.md` |
| Small/real NK | real-NK local posterior | qualified: all 49 workers completed and the selected row had R-hat `1.00077` and ESS `9011.3`, but the parent timed out before canonical reduction | `/home/chakwong/python/docs/plans/BayesFilterDSGE/neutra-real-nk-baseline-manual-closeout-result-2026-06-15.md` |
| Small/real NK | principal-sqrt SVD-UKF posterior | qualified: viable continuation candidate, but R-hat `1.01774` exceeded `1.01` and native divergence telemetry was unavailable | `/home/chakwong/python/docs/plans/BayesFilterDSGE/nk-svd-ukf-neutra-promotion-decision-2026-06-20.md` |
| Rotemberg NK | linear-Kalman posterior | clean successful serious baseline with calibrated HMC and passing R-hat/ESS gates | `/home/chakwong/python/docs/plans/BayesFilterDSGE/rotemberg-linear-kalman-neutra-success-closeout-2026-06-30.md` |
| Rotemberg NK | second-order SVD, 4D minimal principal-sqrt posterior | clean completed-valid serious baseline; zero selected-candidate divergences, R-hat `1.00324`, minimum ESS `2248.3` | `/home/chakwong/python/docs/plans/BayesFilterDSGE/rotemberg-second-order-svd-4d-minimal-principal-sqrt-baseline-phase6-serious-launch-result-2026-07-01.md` |

The count groups the two predator-prey filters, the two Small/real-NK
posteriors, and the two Rotemberg solution/filter posteriors into one family
each, while retaining all six as separate posterior-target configurations.

## Explicit Exclusions

- Exact stochastic volatility, KSC stochastic volatility, and the structural
  deterministic model in the current BayesFilter program stopped at filter,
  source-route, target, or comparator gates before learned NeuTra retained
  HMC. They are attempted program cells but not tested NeuTra configurations.
- SGU linear and second-order lanes reached readiness/canary stages but not a
  serious retained NeuTra result, so they are not counted.
- The earlier full Rotemberg second-order launch that failed before HMC
  candidates is not counted separately from the later successful 4D minimal
  configuration.
- Affine, diagonal, TriL, IAF, RealNVP, score-matching, learning-rate, seed,
  dimension, and HMC-grid variants do not create new model counts.

## Prospective Cost-Bounded Truth Diagnostic

Future synthetic model tests use the generating parameter at the prior mean
and start with one independent dataset seed. Plain-HMC agreement is optional
debugging evidence, not the primary scientific screen.

For retained physical-parameter samples, define the two-sided posterior truth-
tail probability

\[
p_{\mathrm{truth},j} = 2\min\{F_j(\theta_j^{\mathrm{true}}),
1-F_j(\theta_j^{\mathrm{true}})\},
\]

where `F_j` is the empirical posterior CDF. The implementation should use a
finite-sample smoothed empirical proportion and record retained effective
sample size. This quantity is a posterior-tail diagnostic, not a frequentist
hypothesis-test p-value.

| First-seed result | Action and label |
| --- | --- |
| sampler validity passes and every parameter has `p_truth >= 0.05` | stop; `ONE_SEED_DIAGNOSTIC_PASS` |
| no parameter is below `0.003`, but at least one has `0.003 <= p_truth < 0.05` | run one fresh data seed; `MARGINAL_ONE_SEED` |
| any parameter has `p_truth < 0.003` | stop; `ONE_SEED_DIAGNOSTIC_FAILURE` and investigate |
| R-hat/ESS, finite/status, or divergence screen fails | `SAMPLER_INCONCLUSIVE`; repair sampling before interpreting truth recovery |

On the second seed, a repeated `p_truth < 0.05` for the same parameter is
`REPEATED_PARAMETER_CONCERN`. If the original parameter passes, no parameter
has a severe `p_truth < 0.003` miss, and sampler validity passes, record
`TWO_SEED_DIAGNOSTIC_ACCEPTABLE` with both parameter-wise tables. A marginal
miss on a different parameter is recorded but does not automatically trigger a
third seed.

No multiplicity correction is required for this exploratory diagnostic. The
result remains parameter-wise, and the number of inspected parameters is
reported. A one-seed pass supports only:

> NeuTra passed a one-seed central-truth diagnostic for this model/filter
> configuration.

The truth-at-prior-mean design is deliberately favorable. The aggregate count
supports the practical statement that NeuTra is worth trying across a broad
set of sophisticated posterior geometries; it is not a calibration theorem or
a claim of universal reliability.

## Retrospective Boundary

The twelve historical configurations used different contemporaneous evidence
contracts. None is retroactively classified as a truth-tail pass unless its
preserved artifact actually contains the generating truth and the required
posterior-tail calculation. The historical count reports breadth of genuine
learned-NeuTra HMC experience; the prospective ladder standardizes the cheaper
scientific diagnostic going forward.

## Retrospective Diagnostic Addendum, 2026-07-17

The preserved learned-NeuTra retained archives for LGSSM exact Kalman,
predator-prey UKF, predator-prey SGQF, and parameterized SIR SGQF were checked
under the prospective numerical tail ladder in
`docs/plans/bayesfilter-neutra-retrospective-truth-tail-diagnostic-result-2026-07-17.md`.

| Configuration | Minimum `p_truth` | Classification |
| --- | ---: | --- |
| LGSSM exact Kalman | `0.0678083` | `ONE_SEED_DIAGNOSTIC_PASS` |
| Predator-prey UKF | `0.212549` | `RETROSPECTIVE_ONE_SEED_TAIL_PASS_NONCENTRAL_TRUTH` |
| Predator-prey SGQF | `0.215049` | `RETROSPECTIVE_ONE_SEED_TAIL_PASS_NONCENTRAL_TRUTH` |
| Parameterized SIR SGQF | `0.372664` | `ONE_SEED_DIAGNOSTIC_PASS` |

All 33 inspected generating values were inside their empirical 95% posterior
intervals and no parameter crossed the marginal `0.05` or severe `0.003`
threshold. LGSSM and SIR are genuinely central-truth fixtures under their
declared priors. The predator-prey fixture is not: `K=114` differs from its
physical-uniform prior mean `120`, and `s=0.3` differs from its prior mean
`0.6`. Those two configurations therefore retain a qualified retrospective
label and are not retroactively described as central-truth passes.

No second seed is nominated by the predeclared cost-bounded ladder. These four
classifications strengthen the historical breadth ledger but do not establish
calibration, coverage, filter exactness, universal reliability, production
readiness, or default readiness.
