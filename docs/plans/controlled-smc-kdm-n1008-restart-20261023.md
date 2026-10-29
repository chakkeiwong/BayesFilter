# Controlled SMC/KDM: fresh N1008 moment-training restart

## Active scope and authority

This plan refreshes the research work after the 23 October main/LEDH
synchronization. Work belongs on `ledh-refactor-with-policy-fix` in
`.localresources/worktrees/ledh-iapf-execution-20260922`. The current task
authorizes integration and planning. It does not launch a new training campaign
or replenish the time allowances consumed by earlier campaigns.

All previous 64-particle timing, training, tuning, moment and score results in
this investigation are disqualified. Do not load their weights, optimizer
states, calibration constants, selection decisions or simulated datasets. The
source implementations and mathematical derivations can be inspected and
reused after fresh checks. Small unit fixtures test mechanics only; all new
training, feasibility, performance and efficacy measurements must use N>=1008.

The N1008 runtime and dependency investigations are complete. They used no
trained guide and do not establish training feasibility or efficacy. Retain the
checked determinant and innovation logic. Further speed optimization is a
separate engineering task and is not a prerequisite for this research unless
the fresh consumer cannot fit the declared resource budget.

## Research question and evidence contract

Can a controlled-SMC-derived proposal or the SIR neural KDM adaptation reduce
the error of moments produced by the actual corrected LEDH consumer at N1008?
The first engineering scope is the existing 18-state, 9-observation SIR model
at T=5 and its declared theta0. It does not stand in for a longer horizon or a
parameter region. The planned primary numerical lane is GPU/FP32/TF32/XLA;
FP64 is an explicitly separate reference lane. Feasibility and validity must
be established in the actual lane before selecting a training budget.

For terminal moments, C=(theta0,Y[1:T]) and Phi contains first and raw-second
state features. Train the output moments of the final shared Contract-E cloud
with simulated Phi(X_T) labels and fixed calibration weights W. Particle
randomness U must be independent of X_T conditional on C. Then

    E ||mu_hat_eta(C,U)-Phi(X_T)||_W^2
      = E ||mu_hat_eta(C,U)-E[Phi(X_T)|C]||_W^2
        + E ||Phi(X_T)-E[Phi(X_T)|C]||_W^2.

The second term is common to methods. Paired losses on fresh simulated paths
therefore compare population moment MSE without posterior-moment or score
labels. One realized label is noisy: use independent paths and uncertainty
intervals, not an isolated label or a training-loss ranking. Local losses
conditioned on a parent and future observations address a different proposal
target and must be labeled separately. Intermediate future-informed clouds
must not be reported as filtering moments.

| Role | Requirement |
| --- | --- |
| Primary criterion | Predeclared paired heldout terminal-moment loss differences, with uncertainty over independent simulated datasets and independent training seeds, against the comparator ladder below. Freeze the interval method, multiplicity rule and practical effect threshold before opening the final holdout. |
| Promotion veto | Invalid weights, flow, covariance/reset, analytical derivatives, target-density correction or numerical flags; leakage; failure of the conditional heuristic comparisons. |
| Continuation veto | Wrong target or conditioning, contaminated holdout, irreproducible source/data, non-finite training with no identified repair, or exhausted declared compute budget. |
| Repair trigger | Disconnected gradients, unresolved gradient-check failure, ineffective guide action, loss regression, insufficient capacity/optimization or unaffordable N1008 consumer. Repair the localized cause and use fresh heldout partitions when required. |
| Explanation only | ESS, weight variance, gradient norms, component utilization, runtime, local proposal moments and training/validation loss. These cannot replace the final consumer criterion. |
| Not established | Better model scores, unbiased finite-N filtering, full original-author KDM replication, longer-horizon performance, canonical admission, a new default or HMC readiness. |

Score accuracy remains a separate downstream question. After a candidate clears
the moment comparison, compare analytical model scores to an independently
qualified reference if available. Never supply that reference as a training
label, infer score improvement from moment loss, or call an unqualified
Zhao-Cui computation an oracle.

## Comparator ladder and conditional checks

Construct these fresh competitors from the same SIR model and evaluation data:

1. Unchanged plain LEDH: establishes the cost and effect of adding a guide.
2. Current-observation conditional Gaussian proposal: exploits the current
   observation without a learned future message.
3. A simple quadratic future-message proposal fitted on training simulations:
   tests whether a small classical approximation captures the useful signal.
4. The existing controlled-SMC-derived proposal, refitted under the declared
   moment protocol: tests that mechanism independently of neural capacity.
5. The SIR neural KDM proposal, freshly initialized and trained: tests flexible
   locations and bandwidths against those simpler alternatives.

The first three also form the constructed heuristic adversary set. Evaluate
paired differences separately in low/high epidemic intensity and low/high
observation innovation regimes, with thresholds frozen from calibration data.
Report counts and uncertainty in each regime; an unresolved or losing regime
prevents a superiority claim. Do not train on the final heuristic comparisons.
Do not add a combined cSMC/KDM arm before their separate mechanisms are checked.

## Implementation sequence

1. **Scope and call-chain repair.** Replace hard-coded N64 in
   `neural_kdm_diagnostic_campaign.py`, `neural_kdm_study.py`,
   `moment_training_campaign.py` and their active callers with one explicit
   scope configuration. Propagate N through initial/noise/reset designs,
   objectives, chunk selection, manifests, continuation and evaluation. Reject
   N<1008 at campaign entry points. K=1008 follows the exact-divisor chunk
   policy. The same configuration must bind dtype/TF32/backend and distinguish
   the primary lane from the existing hard-coded FP64 reference choices.
   Historical readers must remain historical, not default launchers.
2. **Actual-consumer checks.** Trace both learned arms to
   `build_backward_guided_ledh_kernel` and the shared analytical LEDH executor,
   including UKF covariance evolution, OT, Contract E and the full correction.
   Check proposal sampling and full mixture density, the physical numerator,
   loss extracted from the returned final cloud, and nonzero neural/guide
   gradients against directional differences. Use a positive learned-proposal
   mixture weight when checking that the guide can affect the consumer.
   Training gradients concern guide parameters; model-score computations stay
   analytical. Fixed categorical probabilities remain an explicit restriction.
3. **N1008 feasibility.** With fresh inputs, measure one real forward/gradient
   update and one analytical-score evaluation at N1008, retaining validity,
   compilation, memory-growth/allocator, device, tracing and timing diagnostics.
   Separate compile cost from repeated work. The current neural trainer sums
   per-row gradients inside a TensorFlow loop; it is not a batch-native target
   implementation. Main's separate batched Gaussian optimization does not prove
   this guide endpoint uses that optimization. Record the actual execution lane.
4. **Fresh target-specific protocol.** Use feasibility costs to write one
   bounded campaign manifest and exact commands. Declare total wall and
   aggregate worker budgets, attempt limits, new seed namespaces, disjoint
   calibration/train/validation/final partitions, capacity and learning-rate
   search, cSMC fitting controls, training-seed policy and a budget ladder.
   Freeze objective scaling and comparison criteria before training. Preserve
   OT epsilon=1 and five Sinkhorn iterations by owner instruction. No new
   canonical/default claim follows from that fixed diagnostic setting.
5. **Execute and assess.** Run the reviewed ladder, repair localized failures
   within its budget, freeze selections, then evaluate untouched paired
   holdouts. Save decision and inference-status tables, conditional heuristic
   results and source/data provenance. Reject a failed candidate without
   treating it as a rejection of the research direction. Revisit score
   evaluation only after the moment question has an interpretable answer.

## Default audit, budget and skeptical review

N1008 and OT(1,5) are owner requirements. T5/theta0 define a bounded engineering
starting scope, not a general SIR setting. FP32/TF32/GPU/XLA follows repository
direction and requires fresh validity evidence; FP64 timing does not validate
it. Feature scaling comes only from fresh calibration simulations. Existing
network width, component count, optimizer settings, mixture weight and cSMC
fit controls are hypotheses to test, not transferred selections. Correction
caps, damping/ridge and safety settings require a recorded numerical rationale
and non-harm checks; do not silently import their old tuned values.

The active next step is step 1's bounded source/configuration repair and its
focused mechanical checks. No GPU or training time is assigned or spent by
this planning refresh. Before step 3, record its small feasibility cap and exact
command; before step 4 executes, specify the campaign's measured, adequate total
budget. Record elapsed and remaining time per attempt, without reusing old
campaign allowances. Write new evidence under a fresh versioned directory below
`docs/plans/artifacts/controlled-smc-kdm-n1008-restart-20261023/`.

Skeptical review: the old runnable commands fail the current particle-count
requirement; the guide trainer is not batch-native; guide-free runtime does
not price training; training loss is not final evidence; local proposal moments
and filtering moments have different conditioning; a mathematically valid loss
does not guarantee that this optimizer reaches a useful solution. The sequence
above addresses these issues before an expensive launch. The implementation
preparation is ready; the training campaign is deliberately not yet specified
well enough to launch. No efficacy, speed or scientific ranking is made by
this refresh.
