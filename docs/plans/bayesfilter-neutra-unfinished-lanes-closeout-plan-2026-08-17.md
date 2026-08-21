# NeuTra Unfinished-Lanes Closeout Plan, Revised Scope (2026-08-17)

> Historical scope revision: this active plan covers only the Banana, KSC-UKF,
> and German-credit NeuTra lanes. The prior executed result remains historical
> provenance; it is not an active promotion record for this revised scope.

## Scope

This campaign closes the remaining active non-SSL-LSTM NeuTra lanes without
promoting smoke, interface, or mean-only evidence:

1. **Banana feature/tail decomposition.** Reuse the frozen seed-15 `L=10`
   candidate and its 5,000-draw archive. Decompose the residual predictive MMD
   into raw-coordinate, latent-coordinate, and predeclared nonlinear feature
   families using overlapping windows and exact-vs-exact block calibration. No
   retraining or retuning.
2. **KSC-UKF sequential HMC completion.** Reuse a complete, hash-bound frozen
   transport and complete broad-grid handoff. Run only the shared sequential
   controller after validating the handoff; do not infer a candidate from a
   redacted summary. Stop on invalid handoff, finite/movement/energy veto,
   convergence cap, or missing target status.
3. **German-credit bounded repair.** Keep the source-bound target and data
   fixed while testing one richer reverse-KL capacity/long-budget hypothesis:
   `(128,128)` dense IAF, six stages, 3,000 updates, batch 1,024, float64
   GPU/XLA, with disjoint 8,192-row selection/audit proposal-support checks.
   HMC is conditional on the proposal ESS screen.

## Research Intent Ledger

| Lane | Main question | Promotion criterion | Continuation veto | Nonclaims |
|---|---|---|---|---|
| Banana | Does the residual MMD excess localize to a feature family? | Complete finite exact-control diagnostic | Candidate/archive/hash mismatch or nonfinite decomposition | No equality proof, retraining decision, or downstream transfer |
| KSC-UKF | Does the frozen transport pass sequential convergence and downstream gates? | Shared sequential HMC convergence, ESS, finite/status/movement/energy, and target-specific gate | Invalid handoff, no viable kernel, hard numerical veto, or cap | No universal filter result or default readiness |
| German | Can one richer target-specific reverse transport pass proposal support? | Global and median-batch ESS thresholds from the German plan; HMC only after support pass | Nonfinite training, invalid source binding, proposal ESS veto, or budget cap | No objective ranking or posterior claim |

## Default and Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| Banana projections | Derived from the analytic banana map and prior residual | Projection family can miss another discrepancy | Report full MMD and every declared family | Reviewed diagnostic hypothesis |
| KSC reuse | Existing frozen transport and broad-grid handoff | Missing private details or stale hashes | Validate both artifacts and target signature before launch | Conditional reuse |
| German `(128,128)`, six stages, 3,000 updates | Capacity/budget hypothesis from the prior support failure | Cost or optimization instability | 200-update GPU/XLA canary and finite support audit | Reviewed hypothesis |

## Evidence Contract

- Every serious lane has a fresh versioned artifact root, command, Git commit,
  environment, hardware, seeds, memory policy, XLA status, and hashes.
- Descriptive metrics cannot override a hard veto. Acceptance, loss, ESS
  fractions, MMD point values, and runtime remain explanatory unless the lane
  explicitly names them as a gate.
- The campaign does not rank candidates. A lane ends as `DIAGNOSTIC_COMPLETE`,
  `VIABLE_CANDIDATE`, `FAILED_SUPPORT_SCREEN`, or `INCOMPLETE_HANDOFF`.
- A failed repair weakens that implementation/tuning hypothesis; it does not
  by itself reject the NeuTra research direction.

## Skeptical Plan Audit

| Risk | Audit disposition |
|---|---|
| A public grid summary is treated as a sequential handoff | Rejected: require the complete private viable-set payload and exact hash. |
| German heldout loss is treated as proposal validity | Rejected: proposal support is the hard gate before HMC. |
| Banana feature pass is treated as full-law equality | Rejected: this lane is diagnostic only. |
| A failed repair is treated as evidence against NeuTra itself | Rejected: classify implementation, tuning, support, and sampler failures separately. |
| GPU memory/XLA provenance is absent | Hard launch veto; configure and record memory growth before TensorFlow initialization. |

Audit verdict: fit for bounded execution, subject to the three continuation
vetoes above. The prior executed result is historical and does not authorize a
new candidate or a new default.

## Execution Order and Budget

1. Run focused tests and compile checks.
2. Execute Banana decomposition; cap 30 minutes.
3. Validate the KSC handoff; run sequential HMC only if complete; cap 90 minutes.
4. Run the German 200-update canary; only if finite, run the 3,000-update repair
   and CPU proposal-support audit; cap 3 hours total.
5. Write the terminal result and reset memo for these three lanes.

The campaign stops each lane at its first continuation veto and preserves the
artifact. It does not consume another lane's budget to hide a failure.
