# Generic NeuTra five-stage training reset memo (2026-08-15)

## Current state

The generic five-stage NeuTra controller, dense-IAF adapter, focused tests, GPU
campaign harnesses, plan, result note, and versioned artifacts are complete.
No campaign process is running.

Key paths:

- implementation: `bayesfilter/inference/neutra_staged_training.py`;
- public exports: `bayesfilter/inference/__init__.py`;
- tests: `tests/test_neutra_staged_training.py`,
  `tests/test_neutra_generic_five_stage_model_runner.py`, and
  `tests/test_neutra_generic_five_stage_campaign.py`;
- plan: `docs/plans/bayesfilter-neutra-generic-five-stage-training-plan-2026-08-15.md`;
- result: `docs/plans/bayesfilter-neutra-generic-five-stage-training-result-2026-08-15.md`;
- campaign result:
  `docs/plans/artifacts/neutra-generic-five-stage-training-2026-08-15/campaign_result.json`;
- artifact hashes:
  `docs/plans/artifacts/neutra-generic-five-stage-training-2026-08-15/artifact_hashes.json`.

## Scientific status

| Question | Answer |
|---|---|
| Is the controller generic and mechanically valid? | Supported by focused tests and target-independent API structure. |
| Does it repair the reverse funnel? | Yes for the untouched proposal-law gate in 2/2 tested seeds. |
| Is the fixed recipe generally better? | No. Matched cold joint training passed the Gaussian in 2/2 seeds while staging failed. |
| Is banana resolved? | No. Both routes were still improving and failed two second-moment screens at 1,000 updates. |
| Is separated multimodality resolved? | No. Both reverse-KL routes collapsed to one mode. |
| Is HMC or SSL-LSTM readiness established? | No. HMC was not run. |

## Next justified experiment

Preserve the generic controller. For Gaussian and banana only, plan a bounded
repair that makes the joint phase adaptive to held-out terminal slope and tests
optimizer-state carryover as an explicit tuned option. Compare against the same
cold baseline and distinguish matched selected-path budget from any expanded
budget. Do not spend more reverse-KL updates on the separated mixture; it needs
a mode-covering training objective or external mode-covering evidence.

Only after a target passes its untouched known-law proposal gate should NeuTra
HMC be tested under the repository sequential-HMC policy.

## Resume cautions

- Do not call the five-stage recipe universal or make it the default.
- Do not use held-out training loss as the distributional gate.
- Do not rank banana routes from the one-seed ESS values.
- Do not interpret high within-mode ESS on the mixture as global coverage.
- Preserve the concurrent dirty worktree; many unrelated files belong to other
  active lanes.
- Any new serious NeuTra run must remain GPU, TensorFlow/TFP, XLA-compiled,
  batch-native, and memory-growth compliant.
