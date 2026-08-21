# Generic NeuTra adaptive five-stage repair reset memo (2026-08-15)

## Current state

The adaptive stage-four scheduler and selected Adam-state carry mechanism are
implemented, tested, and exercised on the reviewed Gaussian/banana GPU/XLA
campaign. No campaign process remains.

Key paths:

- controller: `bayesfilter/inference/neutra_staged_training.py`;
- plan: `docs/plans/bayesfilter-neutra-generic-adaptive-five-stage-repair-plan-2026-08-15.md`;
- result: `docs/plans/bayesfilter-neutra-generic-adaptive-five-stage-repair-result-2026-08-15.md`;
- model runner: `docs/benchmarks/run_neutra_generic_adaptive_five_stage_model_2026_08_15.py`;
- campaign runner: `docs/benchmarks/run_neutra_generic_adaptive_five_stage_campaign_2026_08_15.py`;
- artifacts: `docs/plans/artifacts/neutra-generic-adaptive-five-stage-repair-2026-08-15/`.

## Scientific status

| Question | Answer |
|---|---|
| Is the generic adaptive/carry implementation mechanically valid? | Supported. |
| Did longer adaptive joint training repair staged Gaussian? | No. Both staged variants failed 0/2; cold passed 2/2. |
| Did carrying Adam moments repair either model? | No. It changed no pass/fail result and was slightly worse descriptively on Gaussian. |
| Is banana resolved? | No. All routes failed 0/2, mainly through first-two-coordinate variance distortion. |
| Is the fixed five-stage recipe generally transferable? | Rejected by the Gaussian and banana evidence. |
| Should HMC run from the banana candidates? | No; their untouched proposal-law veto fired. |

## Next justified direction

Do not run another universal five-stage campaign. Use target-specific training
protocols:

- Gaussian control: use cold joint training, which passed in both seeds.
- Reverse funnel: retain the previously successful staged continuation as a
  target-specific viable candidate.
- Banana: compare initialization and autoregressive ordering directly. Include
  an identity-biased single-block arm capable of representing
  `x1 = z1 + b*(z0^2-1)`, a full cold three-block arm, and only a minimal staged
  arm if its target-specific mechanism is stated. Tune each arm for banana and
  retain the same untouched exact-law screen.
- Separated mixtures: use a mode-covering objective/evidence source; reverse-KL
  staging is not the repair.

## Resume cautions

- `carry_selected` is experimental and must remain opt-in.
- Do not interpret the adaptive scheduler reaching its cap as convergence.
- Do not call small Gaussian errors acceptable after they failed the predeclared
  intervals in both staged routes and cold passed the same gate.
- Do not rank failed banana arms from ESS or loss alone.
- Do not infer architecture impossibility from optimization failure.
- Preserve unrelated dirty-worktree changes from other agents.
- New serious NeuTra training remains GPU, TensorFlow/TFP, batch-native, XLA,
  and memory-growth compliant.
