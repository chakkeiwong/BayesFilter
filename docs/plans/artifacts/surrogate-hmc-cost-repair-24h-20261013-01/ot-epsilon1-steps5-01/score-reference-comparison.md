# Saved-point scores before and after the five-iteration OT change

Report date: 2026-10-14. This is a descriptive comparison of existing artifacts;
no additional GPU work was run. Checkout: ledh-graceful-failure-execution-20260917,
HEAD 10884528ca8b1a7463895256d30b9655d69c0955, with existing working-tree changes.

## Question and scope

Does changing the selected OT settings to epsilon=1 and five Sinkhorn iterations
move the reported likelihood scores closer to independent model references?
The table uses the saved dataset and nominal parameter point only. Both arms use
N=4032, chunk=2016, GPU/XLA and TF32 off. Arithmetic is FP32 except for SIR's
existing FP64 recursion/FP32 pairwise exception. Other controls are frozen.
These are likelihood scores in the listed model parameter coordinates, not
posterior scores and not uniformly derivatives with respect to physical scales.

| Model | Horizon | Before: epsilon, iterations | After: epsilon, iterations |
|---|---:|---|---|
| SIR | 50 | 0.5, 40 | 1, 5 |
| Range bearing | 20 | 4, 40 | 1, 5 |
| KSC SV | 50 | 0.25, 40 | 1, 5 |
| Predator-prey | 50 | 1, 40 | 1, 5 |

Except for predator-prey, this changes both epsilon and iteration count. It does
not isolate an iteration-count effect. The current KSC target is the repository's
seven-mixture KSC model.

## Scores

The reference column contains numerical reference estimates, not exact oracles.

| Model | Parameter | Before | After | Numerical reference |
|---|---|---:|---:|---:|
| SIR | log kappa scale | -175.025414 | -170.166902 | -157.110397 |
| SIR | log nu scale | 98.351819 | 90.864080 | 104.578310 |
| SIR | log observation-noise scale | 56.140837 | 59.453225 | 56.860701 |
| Range bearing | log process-noise SD scale | -0.709281 | -0.761364 | -0.334787 |
| Range bearing | log observation-noise SD scale | -6.738297 | -6.670021 | -5.795149 |
| KSC SV | Phi-inverse(gamma) | -2.049715 | -2.079398 | -2.002601 |
| KSC SV | log beta | -0.774762 | -0.703509 | -0.824124 |
| Predator-prey | r | -34.027348 | -34.000763 | -33.991138 |
| Predator-prey | carrying capacity | -0.768811 | -0.765257 | -0.769203 |
| Predator-prey | half saturation | -0.002593 | -0.001614 | -0.003692 |
| Predator-prey | s | 4.021271 | 3.970699 | 4.043382 |
| Predator-prey | u | -4.763106 | -5.001551 | -4.478284 |
| Predator-prey | v | 5.226709 | 5.518272 | 4.875862 |

Against these point references, SIR's first component moves closer and the other
two move farther away. Range bearing's process-noise component moves farther
away and its observation-noise component moves closer. Both KSC components move
farther away. Predator-prey's r moves closer while its other five components move
farther away. These are descriptive distances, not statistically supported
rankings of the two methods over parameter space or fresh datasets.

## Reference construction and limits

KSC: the current endpoint audit supplies a 1601-node refined integration score.
Its 1201/1601-node log-likelihood difference is about 2.84e-14; value agreement
alone is not a bound on score error. The reference uses the current rounded data.

Range bearing: 16 replicated bootstrap particle filters with 65,536 particles
provide a Fisher-identity score estimate. Score MCSEs are 0.065023 and 0.022681.
The observations and parameter point match the current evaluation. Finite-particle
bias is not bounded by these MCSEs.

SIR and predator-prey: the displayed reference pools two rank-20 trajectory
importance/local-quadratic fits, with 100,000 paths per fit. These are approximate
numerical scores. Between-fit standard errors do not capture all path-sampling,
rank, support or quadratic-radius error and are not full uncertainty bounds.

The SIR and predator-prey references use the original FP64 datasets and parameter
values, whereas the current inputs were rounded through FP32. Rounding the saved
reference inputs to FP32 reproduces the current inputs exactly. Maximum absolute
observation changes are 7.24e-6 for SIR and 3.69e-6 for predator-prey; the largest
predator-prey parameter change is 2.38e-8. The resulting reference-score change
has not been measured, so these are same-dataset comparisons with a precision
mismatch, not exact same-input oracle checks.

Independent bootstrap Fisher estimates using 131,072 particles and four
replications provide an additional uncertainty check:

| Model | Score vector | Reported score standard errors |
|---|---|---|
| SIR | -80.284536, 89.195049, 56.022546 | 36.570085, 10.174775, 0.507036 |
| Predator-prey | -33.893204, -0.782343, -0.003296, 4.115885, -4.611189, 5.044826 | 0.268423, 0.019891, 0.000477, 0.038154, 0.103319, 0.126382 |

SIR's independent references disagree substantially, especially on the first
component. Its oracle accuracy is unresolved; the smaller between-fit errors of
the displayed importance reference must not be interpreted as resolving this.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Report the saved-point comparison; preserve selected settings | Component distances reported, no uniform improvement | Existing arms finite; SIR reference disagreement and precision mismatch preclude oracle certification | One particle design/point and approximate references | If a default change is pursued, match reference inputs and extend paired comparisons over the existing scope | No default, posterior, convergence or broad accuracy promotion |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new non-finite result found; reference limitations block stronger accuracy claims |
| Statistically supported ranking | None established by this saved-point comparison |
| Descriptive-only differences | Component movements listed above |
| Default-readiness | This report does not establish a new default |
| Next evidence needed | Matched-input reference uncertainty and replicated comparisons across declared datasets and parameters |

Post-run audit: reference approximation and input precision, rather than OT
accuracy alone, can explain apparent gaps. Better-matched, better-resolved
references could alter the component comparisons. No global score norm is used
to rank methods across parameters with different units.

## Evidence paths

Paths below are relative to the repository root.

- Before/after: this directory's report.json and each model's result.json;
  select dataset=saved and position=0. The existing verification.json records
  the paired-run checks and source identity.
- KSC: docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-scope-advance-01/ksc/mechanics/ksc/endpoint-audit.json, refined_score.
- Range bearing: docs/plans/artifacts/ledh-range-bearing-particle-ladder-20261010-02/N4032/references/1704-0-65536.json, score and mcse[1:].
- SIR and predator-prey: docs/plans/artifacts/ledh-independent-validation-20261008-01/report-20261009-011148/comparison.json, datasets with data_seed=26100831, reference.zhao_rank20 and reference.bootstrap_N131072.

Reporting consumed zero additional GPU time. The campaign checkpoint's remaining
budget stays 15.22069209597675 GPU hours; this report does not mark the master
HMC campaign complete.

## Clarification: the 1,482.48% SIR maximum

The nominal-point table above and the maximum use different subsets of the
experiment. The maximum covers 18 SIR evaluations (three datasets times six
parameter points), each with three score components. It occurs on fresh dataset
26101401, position 2, at model log-scale parameters (-0.0175, -0.0175, -0.0175).

| SIR component | Before | After | Signed change |
|---|---:|---:|---:|
| log kappa scale | -46.244229 | 4.421015 | 50.665244 |
| log nu scale | -1.325783 | -20.980306 | -19.654523 |
| log observation-noise scale | -13.302526 | -6.497191 | 6.805336 |

The implemented percentage is 100 * abs(after-before) / max(1,abs(before)).
For log nu: 100 * 19.654523172059427 / 1.3257831440132581 =
1482.484014132471 percent. The small denominator magnifies the percentage.
However, kappa also changes sign and the Euclidean vector change is 113.77435%
of the baseline norm in these log-scale coordinates. Thus the difference at
this point is not solely a denominator effect. Log likelihood changes by
-0.20334976935509985.

The saved nominal point shown earlier has maximum component change 7.613219%
and vector change 4.566981%. Neither metric measures error to an oracle.
Nominal-point references do not determine which arm is more accurate on this
fresh dataset at position 2. These descriptive results establish neither a
statistical ranking nor posterior impact. Existing controls are unchanged.
Recomputed from saved JSON with Python standard-library arithmetic; zero
additional GPU time.
