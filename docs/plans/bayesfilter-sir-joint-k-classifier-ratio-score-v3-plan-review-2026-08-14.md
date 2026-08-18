# Thorough Review: SIR Joint-k Classifier-Ratio Score V3

Date: 2026-08-14  
Reviewed plan: `bayesfilter-sir-joint-k-classifier-ratio-score-v3-plan-2026-08-14.md`  
Verdict: `PASS_WITH_REQUIRED_IMPLEMENTATION_GATES`

## Target And Derivation Review

The conditional binary identity is correct only because class balance holds at
each delta and the delta sampling law is identical between classes. The plan
requires both conditions and tests them. The constrained odd logit gives

`d z(y,delta)/d delta at zero = c1(y)/delta_scale`.

Since the central density log-ratio derivative is twice the observed-data
score, `c1/(2*delta_scale)` is the correct claimed estimate. This is not the V2
practice of estimating separate logits and dividing each by a small delta.

Degree five introduces approximation bias of order `delta^7` in the logit. The
maximum delta is reduced from V2's `0.08` to `0.04`, and exact log-scale cells
test the curvature lane. The plan must report higher-order contributions rather
than silently assuming them negligible.

## Skeptical Plan Audit

| Required risk | Review finding | Verdict |
|---|---|---|
| Wrong baseline | compares linear-quadratic odd head to nonlinear odd MLP and exact marginal score | pass |
| Proxy promoted to criterion | validation loss selects architecture; exact fixed-path score is primary | pass |
| Missing stop condition | any of nine exact cells stops SIR; no scientific repair authorized | pass |
| Unfair comparison | candidates use identical paths, perturbations, splits, batches, and budgets | pass |
| Hidden assumption | grid, polynomial degree, calibration, optimizer, precision, and replication assumptions are explicit | pass |
| Stale context | V2 failures motivate but do not supply data or controls; new seed domains are frozen | pass |
| Environment mismatch | direct `tftwogpu`, trusted GPU, XLA, memory growth, and TF32-off are required | pass |
| Artifact cannot answer question | per-k held-out rows, coefficient decomposition, fixed-path score, exact error, SE, and module audit directly answer the gate | pass |

## Material Corrections To The Initial Suggestion

1. Fit the odd logit slope directly rather than regress already divided
   `logit/(2*delta)` values. This avoids amplifying weak-delta noise.
2. Use only positive delta magnitudes as classifier conditions; the class sign
   already represents the symmetric parameter direction. Supplying signed
   delta in addition to the class would make the label trivial and invalidate
   the ratio task.
3. Permit one temperature scale but no calibration intercept, preserving
   oddness and the zero-delta identity.
4. Do not require every smallest-delta classifier to be individually
   significant. The estimator's purpose is pooling. Require pooled signal,
   non-inverted per-k behavior, at least two informative deltas, and exact-score
   admission instead.
5. Retain an upper separation veto and reduce the maximum perturbation to `.04`.

## Required Implementation Tests Before GPU Execution

1. Conditional datasets have every delta exactly balanced and no signed-delta
   label leakage.
2. Logits are exactly zero at delta zero and exactly antisymmetric under a
   synthetic sign reversal of the odd basis.
3. The only score conversion is `c1/(2*delta_scale)`.
4. Gaussian exact location and log-scale log-ratios have the declared odd
   expansion and finite-difference score.
5. Coefficient recovery succeeds on a small synthetic conditional logistic
   fixture without using an exact score as a training label.
6. Selection and final seed domains are disjoint; all horizons are paired
   prefixes.
7. Source and fresh-process runtime audits reject state-estimation, Fisher,
   likelihood, complete-data, and V1/V2 simulation-score dependencies.
8. Calibration has no intercept and preserves oddness.
9. Training records enough validation history to enforce the optimizer-
   completion gate.
10. The SIR command refuses to run without a hashed exact result whose terminal
    status is `PASSED`.

## Statistical Review

Three replicates provide bounded diagnostic uncertainty, not a publication-
grade sampling distribution. The precision and range gates can veto a reference
but do not prove unbiasedness. If SIR passes, estimates are approximate
classifier-ratio references with descriptive replicate SE. No algorithm ranking
is statistically authorized by this plan.

The ECE/AUC thresholds are diagnostics rather than theorem conditions. Their
role is only veto. The exact-score error and precision gates remain the decisive
calibration evidence. Removing the V2 Platt-slope interval is justified because
the scale correction itself is not an error when held-out calibrated behavior
and the exact fixed-path target pass.

## Review Verdict

The plan answers the user's proposed joint-k question without changing the
filter-independent target. It directly addresses V2's independent-fit variance
and small-epsilon division. The plan is executable after all ten implementation
tests pass. Exact-oracle failure is a terminal scientific result, not authority
to adjust the grid, degree, architecture, calibration, or gates post hoc.

## Smoke Harness Repair

The first smoke launch used the full-run batch size with the reduced smoke
dataset and stopped before fitting because pooled rows were not divisible by the
batch. This is an execution harness defect, not scientific evidence. The repair
binds batch size to the profile (`128` for smoke, `2048` for full), records the
pooled row count, and adds a focused divisibility test. No full-run data,
estimator, or gate changes.

## Full Exact-Oracle Result

The repaired smoke completed on trusted GPU/XLA and was diagnostic only. The
full exact run then completed under the reviewed V3 protocol. It failed the
exact gate: 25 of 27 final heads failed only the predeclared per-delta ECE
gate, leaving two admitted heads and no horizon/coordinate with three admitted
replicates. AUC increased monotonically with delta in the inspected rows,
temperatures were positive, and optimizer-completion gates passed, so this is a
valid methodological/calibration failure rather than a dependency or
truncation failure. SIR is vetoed. No post-run ECE relaxation is authorized.
