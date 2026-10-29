# SIR neural KDM adaptation: implementation and independent comparison

21 October 2026. Implementation and the specified independent comparisons are
complete:2496 valid paired evaluation rows across64 datasets. All learned
candidates have higher observed loss than recursive Gaussian filtering in every
required regime. The tested candidates are not promoted.

The SIR-specific neural proposal is implemented as an adaptation of the
replicated bearings method. It trains all six neural weight/bias tensors through
the actual corrected terminal LEDH moments. The completed neural candidate
does not meet the predeclared improvement criterion. The simple recursive
Gaussian estimator has lower observed loss in every required regime, triggering
the heuristic promotion veto. Independent model-score accuracy is unevaluated.

## What is implemented and checked

The network receives the complete standardized observation sequence and predicts
Gaussian-mixture displacements and log bandwidths. Every component uses its
parent's conditional mean and Cholesky factor. The physical proposal mixes this
neural density with LEDH, selects one transformation, and evaluates the complete
mixture density in the importance denominator. Mixture probabilities are fixed
within a candidate; every network parameter is trained. This is neither the
original bearings IWSG learner nor full controlled-SMC twisting.

The endpoint retains the shared UKF covariance lifecycle, GenUT correction, OT
and Contract E reset. Neural-parameter autodiff trains the proposal; model
parameters retain the analytical recursive derivative. The actual-consumer
tests check density agreement against TFP, zero-network parity, invalid-scale
rejection, total model-parameter finite differences, and terminal-loss gradients
to all six tensors. Source and executable evidence are mapped in
[consumer-mapping.md](consumer-mapping.md) and
[math-code-review.md](math-code-review.md). Phase-A repairs pass59 CPU checks
plus GPU checks; neural-consumer23, compiler-repair30, study-harness25,
restricted-comparison12 and terminal-analysis7 focused checks also pass.
These checks establish the tested implementation properties, not estimator
superiority or true-model score accuracy.

The population target is posterior first/raw-second moment MSE. The actual
training quantity is independent simulated-label squared loss. Under the stated
conditional independence, their difference is an additive constant independent
of the learner. Percentage changes have different denominators. Latent-selected
strata require a nonzero cross term in general and retain only the declared
prediction-loss interpretation. The manuscript supplies derivations; MathDev's
probability formalization and terminal structural audit remain inconclusive.
The checked algebra and numerical evidence are not a formal proof certificate.

## Scope and frozen selection

This diagnostic fixes SIR d18,T5,N64,theta0,FP64,TF32off,GPU/XLA, OT(1,5),
terminal balance32 and chunk64. These controls are not a scope-tuned canonical
default. Calibration uses256 paths; validation32 paths with two paired draws;
final evaluation64 fresh paths with four paired particle draws. The train,
validation and final namespaces are disjoint. All final comparisons condition
on the selected fits; they do not estimate variability across repeated complete
training/search campaigns.

All16 neural configurations completed32 updates; two continued to128. Validation
selected K2,width16,beta.9,Adam.001,seed263044001 at32 updates, with loss
.6786334119165994. The optional512 continuation timed out after362 valid updates,
producing no completed checkpoint. It is excluded from selection. The uncompleted
longer rung limits optimization conclusions.

The fresh restricted comparison fits quadratic cSMC-derived and positive-RBF
messages from observations, then screens pooled3/time-specific12 adapters with
two direction seeds and two step sizes. All18 fits/160 updates complete with
finite losses, accepted updates and no coefficient-bound hits. Validation selects
the pooled cSMC and time-specific RBF24-update continuations at rate.15 and
seed263048002. These have validation losses.7525414589562959 and
.7659842365221505, respectively. They use beta.5 and two training paths per
update, versus the neural four-path batch and beta search. They are restricted
baselines, not fully tuned classical optima or training-efficiency comparators.

## Independent results

The completed neural holdout contains all1472 valid rows with unchanged package
source. Trained loss is.7706365074, initialization.7740662759, plain LEDH
.7652113873 and recursive Gaussian.5335124316. The trained-minus-initialization
difference is-.0034297685,95% paired dataset-bootstrap interval
[-.0095097836,.0024543246]. Trained-minus-plain is+.0054251201, interval
[-.0501252611,.0631362082]. Neither primary interval supports a ranking; the
required10% label-loss reduction also fails. Full blocks, regime counts and
interpretation are in [neural-holdout-result.md](neural-holdout-result.md).

The restricted comparison contains all1024 required valid rows. No package
source changes, missing regimes or pairing failures occur. All834 package-source
hashes match the neural study. The combined diagnostic analysis preserves the
original neural decision and uses5000 dataset-cluster bootstrap draws with
seed263045001. Its intervals condition on the frozen fits and are exploratory,
without adjustment for multiple comparisons.

| Method | Total | First moments | Raw second moments |
| --- | ---: | ---: | ---: |
| Trained neural adaptation | .770637 | .557711 | .983562 |
| Same neural network, untrained | .774066 | .559156 | .988976 |
| Plain LEDH | .765211 | .552860 | .977563 |
| Conditional-only proposal | .759526 | .547718 | .971334 |
| cSMC-derived, unadapted | .871385 | .655728 | 1.087042 |
| cSMC-derived, adapted | .863626 | .644080 | 1.083172 |
| Positive RBF, unadapted | .875033 | .658915 | 1.091151 |
| Positive RBF, adapted | .872854 | .656278 | 1.089430 |
| Recursive Gaussian | .533512 | .378388 | .688637 |

The adapted-minus-unadapted cSMC difference is-.00775892,
95% interval[-.05066537,.03271678]; the RBF difference is-.00217936,
interval[-.00809018,.00367577]. Neither establishes an adapter benefit.
The neural-minus-adapted-cSMC difference is-.09298969,
interval[-.13785872,-.05768208]; neural-minus-adapted-RBF is-.10221735,
interval[-.14089076,-.06872114]. These exploratory intervals favor the neural
candidate over these restricted fits. They do not rescue its failed primary
comparison, demonstrate a learning gain, or compare full source methods.

The adapted restricted candidates also have higher loss than plain LEDH:
cSMC excess.09841481, interval[.03411166,.16749319]; RBF excess.10764247,
interval[.04444369,.17703519]. Recursive Gaussian has lower observed loss than
all four restricted arms in every required regime. No candidate clears the
heuristic test, even where a complex-to-complex difference is favorable.

| Regime (datasets) | Neural | cSMC unadapted | cSMC adapted | RBF unadapted | RBF adapted | Recursive Gaussian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All (64) | .770637 | .871385 | .863626 | .875033 | .872854 | .533512 |
| Low observations (31) | .833695 | .965950 | .964081 | .972892 | .965729 | .598323 |
| High observations (33) | .711399 | .782552 | .769260 | .783105 | .785607 | .472630 |
| Nearer boundary (2) | 1.338364 | 1.416375 | 1.550776 | 1.674356 | 1.673232 | .959331 |
| Interior (62) | .752323 | .853805 | .841460 | .849249 | .847035 | .519776 |

The two boundary datasets do not establish population-tail behavior. The latent
regimes retain only their prediction-loss interpretation. Every method/block,
conditional pairing and machine-readable heuristic verdict is preserved in
[completion-analysis.json](completion-analysis.json). The primary neural
report also includes the predictive and observation-only heuristic results.

## Decision and uncertainty

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not promote the neural candidate |10% reduction plus negative paired intervals against plain and initialization fails | Pairing/source/numerics pass; conditional heuristic veto fires | Longer optimization is incomplete; capacity and finite filtering effects are not isolated | Archive this completed comparison; any next hypothesis requires fresh evaluation data | No rejection of all neural proposals or simulated-moment training; no score benefit |
| Do not promote the restricted adapters | Adapted-minus-unadapted intervals include zero | All rows valid; each candidate fails the conditional heuristic test | Restricted capacities, fixed beta and smaller training batches | Retain as freshly measured comparators for this completed diagnostic | No verdict on fully twisted cSMC, original neural KDM or optimal classical methods |
| Retain implemented adaptation as a diagnostic path | Actual-consumer properties pass focused tests | Invalid proposals reject; no canonical admission is issued | Only the stated small SIR scope has been evaluated | Use a new protocol and untouched data for a distinct future hypothesis | No universal default, HMC readiness or original-method replication claim |

| Inference item | Status |
| --- | --- |
| Hard veto screen | All numerical/evidence validity checks pass; every learned candidate fails the heuristic promotion screen. These are candidate failures, not continuation vetoes. |
| Statistically supported ranking | None for neural primary comparisons or either restricted adapter's learning gain. Exploratory pooled intervals favor neural over the restricted fitted proposals, while both adapted restricted proposals have higher loss than plain LEDH. |
| Descriptive-only differences | Training reduces loss.443% relative to initialization; trained loss is.709% higher than plain. Conditional comparisons are exploratory; boundary n=2. |
| Default-readiness | No; canonical per-scope tuning and model-score validation remain absent. |
| Next evidence needed | A distinct future hypothesis needs a fresh reviewed protocol, untouched paired evaluation and independent downstream score evidence. |

Post-run red team: a finite optimization search can fail even when the population
risk identity is correct. Restricted capacity, limited training and finite-
particle correction/reset effects are distinct explanations, not established
root causes. The result does establish that this selected candidate fails the
predeclared comparison. A fresh, adequately budgeted protocol with paired gains
and no heuristic veto could overturn that candidate-level conclusion. Weakest
evidence concerns boundary behavior and extrapolation to model scores.

## Reproduction and resource accounting

Worktree branch is ledh-refactor-with-policy-fix, base
4d6915cc66e701c4c7b657e094306d298fdd3e76, with preserved working-source snapshots.
The neural evaluation is ../attempt-16-finalized/final-holdout; restricted
evaluation is ../attempt-19-restricted/study. Each contains its exact command,
seeds, data identity, environment, source snapshot and result. GPU memory growth
is verified on the RTX4080SUPER. The scoped XLA command-buffer workaround remains
recorded; GPU/XLA itself remains enabled. Failed attempts remain preserved.

The combined reporting command ran from this worktree with exit status0:

```sh
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  scripts/analyze_sir_neural_kdm_completion.py \
  --neural-study docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-16-finalized/final-holdout \
  --restricted-study docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-19-restricted/study \
  --output docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-01/completion-analysis.json
```

This is CPU-only post-run reporting with GPU devices intentionally hidden;
it does not execute or retrain either filtering method.

Before attempt19,13475seconds were charged against the22051.197895057-second
post-engineering allowance, leaving8576.197895057. Attempt19's cap is
8276.197895057, with300seconds reserved outside it. Terminal charge and remaining
allowance are now recorded:3685seconds charged for attempt19, with
4891.197895057seconds remaining. Its measured wall time is3684.356911139seconds;
the child reports3679.644338852seconds. Total charges after engineering are
17160seconds; including attempts02--11's1715seconds, total is18875 against
the original23766.197895057-second allowance. There is no compute expansion or
reclaimed failed charge. The remaining allowance does not justify retuning on
this now-exposed holdout.

Review: [terminal-review.md](terminal-review.md). Manuscript and master-program
updates record the completed comparison and the unresolved score objective.
The rebuilt [104-page PDF](../../../../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.pdf)
contains the neural design and comparisons in Sections15.12--15.14, with the
result tables on pages96--97. Both final builds are clean, and the changed
result/status pages have been visually inspected.
