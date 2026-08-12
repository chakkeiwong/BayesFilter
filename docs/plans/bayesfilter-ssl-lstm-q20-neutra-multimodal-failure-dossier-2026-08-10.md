# SSL-LSTM q=20 NeuTra multimodal failure dossier (2026-08-10)

## Direct verdict

The August 7 seed-B result was wrong relative to its implied global-posterior
interpretation.  It was a locally converged positive-mode chain, not a validated
posterior sample.  The failure is not explained by insufficient retained sample
count, a low training budget alone, or an incorrect R-hat implementation.  It is
the interaction of four established defects:

1. reverse-KL NeuTra training assigned almost no proposal probability to a known
   negative-observation-weight mode;
2. the learned coordinate system remained strongly multimodal and much more
   curved in the omitted region;
3. every HMC chain was initialized near the positive mode; and
4. HMC tuning examined that positive region only and selected a step size that
   is locally unusable in the negative mode.

The retained R-hat and ESS values were valid calculations for the states that were
sampled.  Treating those diagnostics as global convergence evidence was the
interpretation failure.  No exact posterior mode-weight authority currently exists.

## Claimed target and quantity actually computed

| Item | Classification |
|---|---|
| Claimed target | Draws representing the full seed-B SSL-LSTM q=20 posterior, suitable for posterior-predictive simulation. |
| Quantity actually computed | Four NeuTra-coordinate fixed-HMC chains initialized in one positive region, using a kernel tuned in that region, followed by within-sample R-hat/ESS. |
| Relation | Different.  Local stationary sampling in one basin does not establish discovery, transitions, or relative mass across basins. |
| Checked source | Historical commit `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`, archived checkpoint and tensor receipts, reconstructed exact transformed target, and the root-cause artifacts cited below. |
| Still unproved | Exhaustive mode discovery, exact mode weights, cross-mode mixing, full posterior correctness, NeuTra repair, and posterior-predictive validity. |

## Failure chain

| Stage | Intended role | Observed failure | Evidence | Classification |
|---|---|---|---|---|
| NeuTra objective | Learn a useful global coordinate map | Ordinary reverse KL learned essentially only the positive basin | Only `3/100,000` frozen proposal draws had negative observation weight | `transport_training_failure` |
| NeuTra inverse geometry | Put important posterior regions in accessible base space | Negative source MAP mapped to an extreme base tail | Positive inverse MAP standard-normal log density `-3.729`; negative `-96.704` | `mode_omission_mechanism` |
| Exact transformed target | Become close enough to unimodal for ordinary HMC | Exact pullback retained two separated stationary regions | Transformed stationary points were `23.707` latent units apart | `transformed_geometry_failure` |
| Local transformed curvature | Admit one globally valid HMC kernel | Negative region was about 79 times larger in maximum precision | Maximum precision eigenvalue positive `1.1633`, negative `91.7205` | `region_dependent_geometry` |
| Initialization | Exercise all material basins | All four starts mapped to positive observation weight | Starts were `0.326--1.270` from positive inverse MAP and `13.095--13.882` from negative inverse MAP | `initialization_failure` |
| HMC tuning | Select a target-valid fixed kernel | Tuning sampled only positive-region geometry | Frozen `epsilon=0.811521`, `L=3`, identity mass | `tuning_scope_failure` |
| Negative-region mechanics | Preserve finite, locally useful transitions | Frozen kernel accepted no local negative proposals | At exact transformed stationary starts: positive `23/32`, negative `0/32`; negative proposals reached the finite sentinel `-1e100` | `kernel_local_failure` |
| Causal control | Distinguish target invalidity from bad integration | Smaller step repaired local negative motion | With only `epsilon` changed to `0.1`: positive `32/32`, negative `31/32`; max absolute log-acceptance fell from `1e100` to `0.0566` | `step_size_cause_established` |
| Sequential sampling | Establish local chain readiness | All chains agreed inside the same positive region | 4,000 retained states, all with observation weight in `[0.128007,1.205762]`; no sign transitions | `local_screen_only` |
| R-hat/ESS interpretation | Establish global convergence | Diagnostics could not see a basin absent from all chains | Retained max modern R-hat `1.00724`; minimum bulk ESS `1380`; minimum tail ESS `533` | `interpretation_failure`, not a formula failure |
| Energy diagnostics | Check posterior-predictive output | Initial plug-in posterior-mean and fixed-MAP tests did not represent the posterior mixture | All five plug-in-mean and all ten positive/negative fixed-MAP equality tests rejected at `p=0.0001` | `wrong_or_incomplete_predictive_input` |
| Posterior-predictive mixture diagnostic | Test the correct empirical posterior-predictive law | Corrected diagnostic could not run because its input archive lacked resolved multimodal weights | Runner now fails closed on unresolved weights | `upstream_sampler_evidence_failure` |
| Provenance | Reproduce exact historical executable state | Original run used a dirty worktree and its complete transformed manifest cannot be reconstructed | `historical_identity_exact=false`; archived/current point parity passed at reviewed `5e-7` tolerance | `historical_provenance_limitation` |

## What did not fail

- The negative stationary point is not merely a non-finite target artifact.  Its
  exact transformed value and score are finite, and the small-step causal control
  moves locally with `31/32` acceptance.
- The retained-state R-hat and ESS computations are not shown to be numerically
  wrong.  They answer a local question and were over-interpreted.
- The TensorFlow/XLA batch target passed archived value/score compatibility at
  measured residuals `1.522e-7` and `2.079e-7` under the reviewed `5e-7` bound.
- The result does not reject NeuTra as a method family.  It rejects ordinary
  single-base reverse-KL training plus one-region tuning as a global solution for
  this target.
- The result does not prove that the two identified modes exhaust the posterior.

## Why more of the old computation is not the repair

More transitions with the old kernel would remain in the positive basin because the
negative basin is both approximately absent from initialization and locally unstable
under the selected step size.  More reverse-KL updates using the same single
standard-normal base and the same training distribution do not provide missing-mode
training signal.  Lowering the step to `0.1` repairs local integration but gives
trajectory length `0.3`, far too short to bridge latent separation `23.707`; it is a
causal diagnostic, not a selected global kernel.

## Evidence boundaries

The two-mode local Laplace fractions, positive `0.5112` and negative `0.4888`, are
explanatory only.  They are not exact posterior weights.  Sign of the observation
weight is a checked separator for the two known representatives, not a proof of a
complete basin partition.  Swap acceptance, future cross-mode transitions, and raw
replica occupancy are not by themselves estimates of posterior mode mass.

## Primary evidence

- Root-cause plan:
  `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-plan-2026-08-10.md`
- Root-cause result:
  `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-result-2026-08-10.md`
- Structured evidence:
  `docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/`
- Mode-coverage/predictive result:
  `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-result-2026-08-09.md`
- Correct posterior-predictive diagnostic and survey result:
  `docs/plans/bayesfilter-posterior-predictive-diagnostic-and-multimodal-hmc-survey-result-2026-08-09.md`

## Negative-result classification

| Question | Verdict |
|---|---|
| Was the target implementation invalidated? | No; checked finite/parity evidence does not support that conclusion. |
| Was the HMC implementation generally invalidated? | No; this selected kernel is invalid for the known negative region. |
| Was tuning invalid for its global claim? | Yes; it examined only one region. |
| Was the learned transport globally adequate? | No; proposal coverage and transformed geometry directly reject that claim. |
| Was the posterior archive globally valid? | No; it omitted a known competing region and has unresolved weights. |
| Was the NeuTra research direction rejected? | No; multimodal training/base designs remain candidates after an independent global authority exists. |

