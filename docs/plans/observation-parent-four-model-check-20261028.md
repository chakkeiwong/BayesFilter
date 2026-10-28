# One-configuration likelihood and score checks

Owner request: commit the completed observation-parent work, then check one
configuration each of LGSSM, KSC SV, predator–prey, and range–bearing against
the available oracle. This is a bounded diagnostic, not a tuning or admission
campaign. The previous SIR campaign is closed.

## Question and evidence contract

Does the implemented filter estimate the same model log likelihood and every
parameter score component as an independent reference at one saved nominal
configuration? Compare original three-branch, uniform conditional parents, and
observation-weighted conditional parents where implemented. KSC's seven-mixture
likelihood and the nonlinear range–bearing likelihood currently reject the
conditional proposal. Record that rejection, and evaluate their explicitly
selected original three-branch route. Do not silently substitute a Gaussian
observation model or claim that those two runs validate the weighted default.

Primary quantities are log-likelihood error and componentwise score error/RMSE.
No scientific promotion or superiority ranking is sought from eight seeds.
Finite outputs, valid status, full horizon completion, exact target/data/parameter
agreement, and reference integrity are validity vetoes. Reference refinement and
Monte Carlo uncertainty qualify the interpretation; unresolved reference error
blocks an accuracy verdict. ESS and runtime explain behavior only. A successful
run does not establish general accuracy, HMC readiness, canonical LEDH admission,
cross-model default readiness, or uniform improvement.

## Fixed configurations and assumptions

- LGSSM: P44, dimension 3, nominal parameters, T=50; saved dataset 26100831.
- KSC SV: existing seven-mixture/q1/prior1 target, nominal parameters, T=50;
  saved dataset 26100831. The oracle must use this same mixture target.
- Predator–prey: canonical transition/observation target, nominal parameters,
  T=50; saved dataset 26100831.
- Range–bearing: M13 wrapped-bearing target, theta=(0,0), T=20; saved data 1704.
- N=1008, eight fresh paired design seeds 281301–281308, GPU FP64/XLA, TF32 off.
  FP64 is an explicit diagnostic exception to the production precision default.
  Memory growth is configured and verified before device initialization.
- Reuse the saved original-proposal controls and branch weights for each model
  as fixed diagnostic settings. These were tuned for the original proposal, not
  the new conditional arm; they are hypotheses, not new-route tuning evidence.
  Their possible failure is poor allocation for the changed proposal; report
  branch weights and ESS. Do not tune on these data.
- Preserve exact stored observations and parameter charts. LGSSM uses exact
  Kalman value/score; KSC uses independently refined grid integration;
  predator–prey uses two existing adapted Zhao–Cui reference fits with local
  score-radius refinement; range–bearing uses 16-replicate 16,384/65,536-particle
  Fisher references. The latter three are numerical references, not exact oracles.
  Verify their data identity and finite/refinement diagnostics before use.

## Research intent and skeptical audit

Mechanism: conditioning each child and weighting predictive parent probabilities
may reduce importance-weight imbalance; score accuracy is a separate question.
Expected failure: improved ESS can coexist with inaccurate recursive scores,
discrete parent boundaries, or poorly resolved reference derivatives.
Repair trigger: localized harness, environment, serialization, or shape failure.
Continuation veto: target mismatch, invalid reference that cannot be repaired
within budget, corrupt evidence, or exhausted compute budget. An unsupported
proposal is a model-specific implementation finding, not a reason to abandon
the other three requested comparisons.

Cheap adversaries: original proposal (established baseline), uniform conditional
parents (isolates weighting), and transition-only bootstrap proposal where the
shared implementation accepts it (removes flow/conditioning). Evaluate them in
each selected model, never by pooling models. Losing to a simple control vetoes
any later superiority claim. The available budget supports a diagnostic only;
there is no optimization or default promotion criterion in this plan.

Pre-run audit: PASS for this restricted question. The two unsupported observation
laws are disclosed; exact and numerical references are distinguished; score
coordinates and observation bytes will be checked; inherited settings remain
explicit hypotheses; ESS cannot decide accuracy; invalid runs are retained.
The tftwogpu environment is required: tf-gpu failed to load the local custom
Sylvester op because of an ABI mismatch. This is an environment failure, not
evidence against the numerical method.

## Execution and budget

1. Commit only observation-parent implementation, compatibility edits, tests,
   manuscript, and completed SIR evidence; preserve unrelated worktree changes.
2. Validate references and unsupported-model behavior; run a small call-chain
   smoke before full evaluation.
3. Run each model in a separate process with incremental JSON rows and logs.
4. Save per-model values, full score vectors, RMSE, bias, seed dispersion, ESS,
   and reference uncertainty/refinement. Record inference status and decisions.

Total new execution budget: 3,600 wall seconds across all attempts, at most
8 launches; each launch bounded by the remaining budget. No new reference
training, package installation, or scientific-method extension. Stop or report
unresolved rows when the budget is exhausted. Preserve each attempt separately.
Output: docs/plans/artifacts/observation-parent-four-model-check-20261028/.
Plan/result paths, exact commands, commit, source hashes, environment, seeds,
devices, memory policy, XLA/TF32, dataset hashes, and elapsed time go in manifests.
Execution command is recorded with the completed runner before its first launch.

Post-run alternative explanation to examine: a favorable nominal result could
be an easy-data or reference-fit effect; one configuration cannot resolve this.

## Frozen runner details before launch

Runner: `scripts/run_observation_parent_four_model_check.py`. Predator-prey
central reference is the arithmetic mean of the two independent fits at each
fit's smallest score radius (0.0003125). Both individual references, radius
refinement and conditional standard errors remain in the result. Range-bearing
uses the 65,536-particle reference centrally and retains the 16,384 comparison.
These choices precede new candidate results. Reference runs are diagnostic CPU
exceptions; the candidate runs use GPU FP64/XLA. Range-bearing input designs
preserve the earlier harness's FP32-rounded normals/clouds, then compute FP64.

Each launch uses:
`TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python -u scripts/run_observation_parent_four_model_check.py --model MODEL --output docs/plans/artifacts/observation-parent-four-model-check-20261028/ATTEMPT-MODEL`

The outer supervisor imposes the remaining campaign wall budget. Full stdout
and stderr are saved per attempt. The N=16,T=2 original-route smoke is only a
wiring/finite check, followed by the complete N=1008 run. Unchanged controls
and paired draws are used in all proposal arms; transition-only sets beta to
(1,0,0) in the same recurrence.
