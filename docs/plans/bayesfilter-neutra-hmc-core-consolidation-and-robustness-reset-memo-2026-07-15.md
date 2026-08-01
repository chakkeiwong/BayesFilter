# BayesFilter NeuTra HMC Consolidation And Robustness Reset Memo

Date: 2026-07-15  
Status: `PROGRAM_COMPLETE_WITH_DISCLOSED_COMMAND_TRANSCRIPT_CAVEAT`

## First-Read State

The program in
`docs/plans/bayesfilter-neutra-hmc-core-consolidation-and-robustness-program-2026-07-15.md`
is complete. C0-C2 consolidated sequential NeuTra HMC into
`bayesfilter/inference/neutra_hmc.py`, migrated the active LGSSM claim-bearing
route, and installed a discovery-complete route-policy guard. S1 validated one
fresh additional training seed on the original exact fixture. F0-F2 generated
one genuinely new fixture, admitted a new plain-HMC comparator, performed
target-specific GPU/XLA training, and validated the frozen NeuTra candidate.
Phase A audited omissions/drift and closed after Claude read-only convergence.

## What Is Established

- Canonical policy: `bayesfilter_neutra_sequential_hmc_v1`.
- Active claim-bearing LGSSM NeuTra HMC delegates to the shared TensorFlow/TFP
  sequential controller.
- Warm-up is retained, separately archived, and never pooled into posterior
  samples.
- Modern R-hat is max(rank-normalized split, folded rank-normalized split).
- Warm-up and retained sampling grow sequentially up to 10,000 per chain.
- S1 confirmation passed for new training seed `(20260715,1203)`.
- F2 confirmation passed on new fixture seed `(20260715,701)` after fresh F1
  training seed `(20260715,8201)`.

## Key Numerical Evidence

| Run | Max R-hat | Min bulk ESS | Min tail ESS | Comparator difference | Truth distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 confirmation | 1.0027965461 | 5394.2862 | 4381.2278 | 1.9669 combined MCSE | 1.6456 posterior SD |
| F0 plain HMC | 1.0053980349 | 1758.5945 | 3982.9331 | N/A | 1.3839 posterior SD |
| F2 confirmation | 1.0036022359 | 4073.1309 | 3154.1754 | 1.7740 combined MCSE | 1.3420 posterior SD |

All admitted kernels had no hard health/status/energy-error veto. F0 step 0.8
was rejected before a fresh bounded grid selected step 0.3.

## Claim Boundary

Correct conclusion: the tested shared NeuTra HMC procedure is viable for the
original fixture with one additional training seed and for one independently
generated fixture in the same 18D LGSSM family under the exact declared gates.

Unsupported conclusions: broad robustness, calibration, superiority, recipe or
sampler ranking, population reliability, production readiness, cross-model
transfer, or universal NeuTra reliability.

## Terminal Artifacts

- Phase A result:
  `docs/plans/bayesfilter-neutra-hmc-core-consolidation-phase-a-result-2026-07-15.md`
- Drift matrix:
  `docs/plans/bayesfilter-neutra-hmc-core-consolidation-phase-a-drift-matrix-2026-07-15.md`
- Repair record:
  `docs/plans/bayesfilter-neutra-hmc-core-consolidation-phase-a-missed-item-repair-record-2026-07-15.md`
- Local checks:
  `docs/plans/bayesfilter-neutra-hmc-core-consolidation-phase-a-local-check-record-2026-07-15.md`
- Claude review:
  `docs/plans/bayesfilter-neutra-hmc-core-consolidation-phase-a-claude-review-record-2026-07-15.md`
- F1 terminal run manifest:
  `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/phase-a/serious_run_manifest.json`

## Residual Caveat And Next Work

F1 command strings are exact reconstructions from the frozen CLI rather than
preserved contemporaneous shell transcripts. The manifest says so explicitly;
the hashed numerical/device/seed/wall-time evidence is contemporaneous.

No repair or continuation phase remains. A future broader robustness campaign
should predeclare multiple seeds/fixtures with uncertainty and include another
model family or dimension. It must not reinterpret this two-fixture evidence as
already supplying that broader result.
