# Generic NeuTra five-stage training result (2026-08-15)

## Outcome

The target-agnostic five-stage controller is implemented and its mechanics are
supported by focused tests. The fixed dense-IAF training recipe repaired the
100-dimensional reverse funnel in both tested seeds, while matched cold joint
training failed the untouched known-law gate in both seeds.

The recipe did not generalize uniformly. At the matched 1,000-update budget,
cold joint training passed the correlated-Gaussian gate while staged training
did not. Both routes remained undertrained on the banana target. Both routes
collapsed to one component of the separated three-component mixture. Thus the
controller is reusable, but this particular stage allocation is not a universal
NeuTra training rule.

## Claimed and computed quantities

| Item | Status |
|---|---|
| Claimed target | A generic controller for phased reverse-KL training, followed by an untouched target-specific validation callback. |
| Quantity computed | Held-out reverse-KL selected each phase checkpoint and learning rate; 131,072 fresh proposal draws were compared with exact known-law moments and tails. |
| Equality of target and computation | Correct for the stated controller and known-law proposal diagnostics. Training loss alone was not treated as distributional correctness. |
| Supporting artifacts | Campaign result and per-cell manifests/results under `docs/plans/artifacts/neutra-generic-five-stage-training-2026-08-15/`. |
| Not evaluated | HMC mixing, posterior correctness for SSL-LSTM, universal model generality, and repository-default readiness. |

## Implementation

The public implementation is
`bayesfilter/inference/neutra_staged_training.py::train_neutra_five_stage`.
It accepts a transport, batched target log density, named masked variable
groups, stage specifications, a stateless latent-batch callback, a held-out
selection callback, and an untouched validation callback. It contains no
target names, funnel coordinates, known target coefficients, or target-specific
thresholds.

The default dense-IAF adapter implements:

1. shift-only affine location training;
2. affine location plus an explicit autoregressive linear scale path;
3. cumulative addition of IAF blocks;
4. full joint fine-tuning; and
5. untouched validation without an optimizer update.

Every phase tunes its learning rate from the same incoming checkpoint. The best
checkpoint is restored before the next phase, and update zero may be selected
when adding a block does not improve the held-out objective. The training step
is TensorFlow float64, batch-native, and XLA-compiled.

## Run manifest

| Field | Value |
|---|---|
| Git commit recorded by run | `3030d86df9cb00346df82c7c19f015c09c7c6e1f` |
| Campaign harness | `docs/benchmarks/run_neutra_generic_five_stage_campaign_2026_08_15.py` |
| Model harness | `docs/benchmarks/run_neutra_generic_five_stage_model_2026_08_15.py` |
| Environment | TensorFlow `2.20.0`, Python from the campaign supervisor environment |
| Device | GPU 0, repository TensorFlow/TFP GPU route |
| Memory policy | `TF_FORCE_GPU_ALLOW_GROWTH=true`; growth verified before logical-device initialization |
| Numeric mode | float64; XLA JIT on; TF32 off |
| Batch | 4,096, batch-native; no sample-wise loop or scalar fallback |
| Learning-rate grid | `2e-4`, `5e-4`, `1e-3`; phase-local piecewise multipliers `1`, `0.1`, `0.01` |
| Audit draws | 131,072 fresh proposal draws per cell |
| Seed policy | Seed 0 for every route/model; seed 1 only when either seed-0 route passed |
| Campaign wall time | 1,048.17 seconds |
| Plan | `docs/plans/bayesfilter-neutra-generic-five-stage-training-plan-2026-08-15.md` |
| Result | `docs/plans/artifacts/neutra-generic-five-stage-training-2026-08-15/campaign_result.json` |
| Integrity | `docs/plans/artifacts/neutra-generic-five-stage-training-2026-08-15/artifact_hashes.json` |

The per-cell manifests preserve the exact command, target definition, seeds,
allocator bytes, configuration, device, output paths, and wall time.

## Results

| Target | Route | Seeds passing | Importance ESS fraction | Log target/proposal-ratio SD | Verdict |
|---|---|---:|---:|---:|---|
| Reverse funnel, d=100 | staged | 2/2 | 0.99597, 0.99585 | 0.06334, 0.06426 | Viable on the untouched known-law gate |
| Reverse funnel, d=100 | cold joint | 0/2 | 0.53838, 0.94850 | 0.16549, 0.16171 | Rejected by root moment/tail screens |
| Correlated Gaussian, d=16 | staged | 0/2 | 0.96921, 0.97872 | 0.16773, 0.14285 | Rejected at matched budget |
| Correlated Gaussian, d=16 | cold joint | 2/2 | 0.99197, 0.99281 | 0.08923, 0.08503 | Viable on the untouched known-law gate |
| Banana, d=16 | staged | 0/1 | 0.61051 | 0.28689 | Undertrained at the cap |
| Banana, d=16 | cold joint | 0/1 | 0.72532 | 0.30656 | Undertrained at the cap |
| Three-mode mixture, d=4 | staged | 0/1 | 0.98407 | 0.12277 | Structural mode collapse |
| Three-mode mixture, d=4 | cold joint | 0/1 | 0.99339 | 0.09289 | Structural mode collapse |

The ESS and ratio-SD values are explanatory diagnostics. A high within-mode
ESS on the mixture did not rescue the candidate because the component-mass
screen exposed missing modes.

### Reverse funnel

The staged route passed root mean, root second moment, standardized child
residual mean/second moment, and both `|y| > 2` tail screens in two seeds. Seed
0 selected 4,500 path updates although 5,000 were available because the second
progressive block selected update zero:

| Phase | Selected LR | Selected update |
|---|---:|---:|
| Affine location | `1e-3` | 250 |
| Simple linear scale | `1e-3` | 2,000 |
| First nonlinear block | `2e-4` | 500 |
| Second nonlinear block | `2e-4` | 0 |
| Third nonlinear block | `2e-4` | 500 |
| Joint fine-tune | `5e-4` | 1,250 |

This isolates the main repair: the explicit linear conditional-scale stage
moved the held-out loss from 281.75 to 49.97 before nonlinear co-adaptation.
The implementation did not receive the known funnel coefficient one. The cold
route compressed the root variance and both tails; for seed 0 its root second
moment was 0.9412 instead of 1 and tail masses were 0.01665 and 0.01484 instead
of 0.02275.

### Correlated Gaussian

The staged route passed coordinate means but failed two coordinate second
moments and 12 of 15 adjacent cross moments. Its joint phase selected the
terminal 300th update at `1e-3`, and its terminal held-out losses remained above
the cold route: 5.9112 versus 5.9010 for seed 0 and 5.9074 versus 5.9008 for seed
1. The common architecture therefore had sufficient capacity, while the fixed
stage allocation and phase-local optimizer resets left too little joint
consolidation at this budget.

This rejects generalization of the fixed recipe at the matched budget. It does
not reject the generic controller or staged training with an adaptive budget.

### Banana

Both routes failed only the first two inverse-transformed latent second moments.
For seed 0, staged estimates were 0.8141 and 1.0903 and cold estimates were
0.8216 and 1.0848, against exact values of 1. Both routes selected terminal
checkpoints while the held-out objective was still improving. With one seed and
no plateau, the defensible classification is under-budgeted/undertrained. There
is no statistically supported ranking between the routes.

### Three-component mixture

The true component probabilities were `[0.5, 0.3, 0.2]`. The staged route's
proposal responsibilities were approximately `[0.999985, 0, 0.000015]`; the
cold route's were approximately `[0.00000023, 0.00000737, 0.9999924]`.
Each route found a different single mode. This is the expected mode-seeking
failure of cold reverse-KL training on a separated multimodal target. More
updates to the same objective are not a justified repair.

## Decision table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the generic controller API | Focused mechanics tests and valid known-law campaign execution | No controller, XLA, batching, finite-value, or artifact veto fired | Broader transport adapters are not tested | Retain as reusable experimental API | Repository default or universal recipe |
| Keep staged funnel route as a viable candidate | Untouched gate passed in 2/2 seeds | No known-law screen failed | Only two seeds and no HMC | Use it as the starting transport for a separately planned HMC test | Posterior/HMC correctness |
| Reject fixed recipe as generally transferable | Staged Gaussian failed while matched cold passed in 2/2 seeds | Gaussian cross-moment screens failed | Whether adaptive joint consolidation repairs it | Test adaptive/longer joint consolidation on Gaussian and banana | Staged training is intrinsically inferior |
| Classify banana as undertrained | Both routes ended at improving terminal checkpoints | Second-moment screens failed | Required training budget and schedule | Extend the budget with predeclared stopping diagnostics | Any route ranking |
| Reject reverse-KL continuation as mixture repair | Component masses collapsed to one mode for both routes | Component-mass and moment gates failed | A different mode-covering objective was not tested here | Use a mode-covering objective/evidence source | Failure of flow capacity in general |

## Inference status

| Evidence question | Status |
|---|---|
| Hard veto screen | Controller and run infrastructure passed; model-specific gate failures are recorded above. |
| Statistically supported ranking | Funnel and Gaussian pass/fail outcomes replicated in two seeds, but continuous-metric superiority is not statistically established. |
| Descriptive-only differences | ESS, ratio SD, loss values, runtime, clipping counts, and one-seed banana/mixture differences. |
| Default readiness | Not supported. |
| Next evidence needed | Adaptive joint-stage repair on Gaussian/banana; a different objective for multimodality; HMC only after the relevant known-law gate passes. |

## Engineering, numerical, and scientific ledgers

| Ledger | Verdict |
|---|---|
| Engineering correctness | Supported by focused tests for disjoint groups, masked updates, identical LR starts, checkpoint restoration, finite failure, full joint coverage, and non-mutating validation. |
| Numerical/training validity | GPU/XLA, batch-native execution and finite artifacts are supported. Model-level numerical adequacy is target-dependent. |
| Scientific interpretation | Funnel viability is supported for this known-law proposal test. Universal generalization, SSL-LSTM transfer, and HMC validity are unsupported. |

## Post-run red team

The strongest alternative explanation for the Gaussian failure is not that
staging is fundamentally harmful, but that fixed early-stage spending and
fresh optimizer moments starved the joint phase. A longer or adaptive joint
phase that still failed across seeds would weaken that explanation. The banana
evidence is weaker because only one seed ran and both routes were improving at
the cap. The mixture conclusion is stronger: component masses show mode
collapse directly, and high local ESS cannot establish global coverage.

The weakest evidence is any comparison based on continuous metrics rather than
the predeclared gates. No claim of overall superiority is made.
