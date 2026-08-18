# NeuTra reverse-funnel root-cause diagnosis (2026-08-15)

Status: `DIAGNOSIS_COMPLETE_NO_REPAIR_RUN`

## Verdict

The remaining proposal failure is not caused by a wrong funnel density, wrong
log determinant, missing additive-scale gradient, insufficient Gaussian-base
support, XLA, or HMC. The corrected transport contains the exact funnel map and
its implementation passes forward, inverse, logdet, and autodiff score checks.

This diagnosis was superseded by the architecture-tuning and staged-training
campaign later on 2026-08-15. The frozen-state trace correctly located most of
that run's root distortion in stage 1, but it did not establish full reversal
as the primary cause. Full reversal is prescribed by the original NeuTra
experimental architecture, and a target-specific root-preserving alternative
also failed after independent learning-rate/schedule tuning.

The supported root cause is joint parameterization and co-adaptation under the
reverse-KL training path:

1. The identity proposal is extremely expensive under the funnel target.
   Reverse KL initially has a much steeper shortcut that shrinks the root
   coordinate `y` than the coordinated route that learns all 99 conditional
   scales.
2. In the full-reversal arm, the coordinate reversal moves `y` to coordinate 99 in
   the middle MADE stage. Its scale and shift can therefore condition on all 99
   other coordinates. Those coordinates contain information about `y` through
   their learned scale, so the middle stage learns to shrink and nonlinearly
   reshape `y` from the `x` cloud.
3. The three stages then co-adapt. Repairing `y` requires coordinated changes to
   the middle-stage root map and the 99 conditional scales; changing only the
   new additive matrix is uphill after this co-adaptation.
4. The fixed 5,000-update, constant-`1e-3` Adam protocol has no convergence
   rule and stopped with a nonzero large-batch gradient. The remaining root-tail
   improvement is small in the aggregate 100-dimensional reverse-KL objective,
   while minibatch gradients remain noisy. Aggregate loss is therefore a weak
   stopping/selection signal for the one-dimensional tail defect.

This is a joint parameterization and optimization-design failure for this
target, not evidence that reverse KL, full reversal, or the available
architecture cannot recover the funnel. A restricted root-scale warm-up passed
the exact proposal law, and two fresh tuned joint continuations preserved that
pass.

## Math trace

The unnormalized target used by the code is

```text
-log p_unnormalized(y,x)
  = 0.5 y^2 + 0.5 exp(-2y) sum_i x_i^2 + 99 y.
```

This matches `y~N(0,1)` and `x_i|y~N(0,exp(2y))`. At the identity proposal,

```text
E[L] = 0.5 + 49.5 exp(2) = 366.258...,
```

consistent with the measured independent-cloud value `366.112`.

For a root log-scale perturbation `y=exp(c)z_0`, `x_i=z_i`, the population
gradient at the identity is

```text
dL/dc at c=0 = 198 exp(2) = 1463.0... .
```

The implementation measured about `1480` for each of the three root-scale
biases. Gradient descent therefore immediately shrinks `y`.

In contrast, constrain the transport to the correct one-stage subfamily

```text
y=z_0,  x_i=exp(w_i y)z_i.
```

For each conditional coordinate,

```text
L_i(w_i) = 0.5 exp(2(w_i-1)^2),
dL_i/dw_i = 2(w_i-1) exp(2(w_i-1)^2).
```

This is strictly convex with its unique minimum at `w_i=1`. At zero the exact
gradient is `-2 exp(2)=-14.778`; the implementation measured a first-row mean
of `-14.950`. A scalar Adam replay with the campaign hyperparameters reaches
`w=0.9995` by update 5,000. Thus the target and additive-variable gradient are
correct, and reverse KL is well behaved in the root-preserving subproblem.

Raw gradient magnitudes do not translate directly into Adam step magnitudes,
so the approximately 100-fold magnitude gap is not by itself the verdict. It
does show the initial direction. Three independently trainable root-scale
biases and the zero-initialized final MADE layers make the shrinkage shortcut
available immediately, before the hidden nonlinear representation has learned
the coordinated conditional map.

## Code trace

- Target formula: `bayesfilter/inference/neutra_paper_d100_target.py`, lines
  179-192.
- Strict MADE mask: `bayesfilter/inference/neutra_weighted_training.py`, lines
  187-223. Output coordinate `i` can use inputs `j<i`.
- Full reversal after every nonterminal stage: the same module, lines 446-454.
- Reverse-KL loss and autodiff: the same module, lines 780-786.
- Global clipping and Adam update: the same module, lines 790-824.
- Fixed update cap and constant LR: the capacity runner, lines 379-407.

With three stages and two full reversals, the root ordering is

```text
stage 0: y is coordinate 0 and cannot condition on x
reverse: y becomes coordinate 99
stage 1: y scale/shift may condition on all 99 x coordinates
reverse: y returns to coordinate 0
stage 2: only an unconditional affine correction of y remains
```

The full reversal is an inherited paper-prescribed IAF mixing choice. The stage
trace below shows where one cold-start trained state distorted the root; it does
not prove that reversal is intrinsically wrong. A later matched root-preserving
arm failed the same exact-law gate, so ordering alone does not explain the
failure.

## Frozen-state stage evidence

One independent 65,536-row CPU diagnostic traced the root distribution through
the frozen corrected transport:

| Location | Variance of `y` | Kurtosis | `P(y<-2)` | `P(y>2)` |
|---|---:|---:|---:|---:|
| Base input | 0.99982 | 3.00893 | 0.02200 | 0.02373 |
| After stage 0 | 0.96989 | 3.00893 | 0.01978 | 0.02255 |
| After stage 1 | 0.85770 | 2.71032 | 0.01183 | 0.01340 |
| After stage 2 | 0.89883 | 2.71032 | 0.01324 | 0.01631 |

Stage 1 creates most of the tail failure. Its output root remains highly
correlated with the original `z_0` (`0.99936`), but the x-conditioned scale and
shift shrink the variance and reduce kurtosis. Stage 2 restores some variance
but is unable to restore the missing tail shape.

The standardized residual aggregate mean and second moment were nearly exact,
but that did not prove the full conditional law. Descriptive checks found
maximum coordinate mean `0.0165`, maximum coordinate variance error `0.0313`,
off-diagonal covariance RMS `0.00387`, and correlation `0.0268` between `y` and
the row residual second moment. These are explanatory diagnostics, not a new
joint hypothesis test. The proposal-law failure and importance-ratio standard
deviation already establish that the joint proposal is not exact.

## Co-adaptation evidence

At the frozen state, the independent 65,536-row gradient norm was `2.46`, so the
state is not stationary. The mean gradient of the stage-zero additive first row
was positive (`0.00595`), meaning a local descent step would decrease those
coefficients, not move them toward one. Forcing the first row uniformly to one
while freezing the nonlinear stages raised the diagnostic objective from about
`50.09` to `114.62`.

This corrects a statement in the August 14 result: coefficients equal to one
are exact only when later stages are identity. They are not the coordinatewise
target after the three stages have co-adapted.

A coordinated post-map correction

```text
y' = a y + b,
x'_i = x_i exp(y'-y)
```

preserves every standardized residual while correcting the root mean and
variance. It reduced the paired objective by `0.002963` with paired Monte Carlo
standard error `0.000269`. This proves that the frozen state has a coordinated
descent direction even though the single additive-block perturbation is uphill.
An affine correction cannot repair the observed kurtosis `2.71`, so a
root-preserving nonlinear repair is still needed.

## Optimization evidence

On 16 fresh batches of the actual training size 4,096:

- minibatch loss standard deviation was `0.1121`;
- global gradient norm averaged `3.32`;
- the stage-zero additive first-row mean gradient had signal-to-standard-
  deviation ratio `2.15`;
- the stage-one root-scale bias gradient averaged `-0.611` with standard
  deviation `0.243`;
- the stage-one root-shift bias gradient was noise-dominated.

The final checkpoint was the terminal update and its fixed-cloud selection loss
was still improving. Therefore `5,000 updates` was a budget cap, not a
convergence finding. Constant-rate Adam is still moving through a shallow
co-adapted valley. Global clipping fired on 140 of 5,000 updates and shaped the
early transient, but the final large-batch norm was below the clip threshold;
clipping is a contributing hypothesis, not the primary root cause.

## Root-cause classification

| Candidate cause | Verdict | Evidence |
|---|---|---|
| Funnel target or exact sampler wrong | Not supported | Formula, sampler, and exact-map tests agree |
| Forward/inverse/logdet implementation wrong | Not supported | Roundtrip, exact construction, and autodiff parity pass |
| Additive scale gradient wrong | Not supported | Measured initial gradient matches analytic value |
| Gaussian base inadequate | Wrong | The funnel is exactly a Gaussian pushforward |
| Corrected route lacks capacity | Not supported | Exact funnel construction exists in the route |
| Full reversal/root conditioning | Contributing hypothesis, not root cause | Stage 1 creates distortion in one frozen state, but tuned root-preserving arms also fail |
| Reverse-KL initialization shortcut | Primary training trigger | Analytic and measured root-shrink gradients agree |
| Joint co-adaptation | Primary persistence mechanism | Single-block exact coefficients are uphill; coordinated correction is downhill |
| 5,000-update constant-LR protocol | Contributing cause | Nonzero gradient, terminal selection, noisy minibatches, no convergence rule |
| Gradient clipping | Possible secondary contributor | Important early, inactive at frozen-state large-batch gradient |
| HMC | Not applicable | HMC was correctly vetoed and never run |

## Next discriminating repair

The executed repair used the known root structure during a restricted warm-up,
then returned to joint training:

1. Train the exact one-stage additive conditional-scale map first with the root
   map fixed to identity. The analytic subproblem gives a direct coefficient and
   proposal-law check.
2. Jointly fine-tune from the passed warm start using an architecture-specific
   decaying learning-rate schedule.
3. Use a decaying learning-rate/convergence protocol and checkpoint on the
   predeclared proposal-law diagnostics, not aggregate reverse-KL loss alone.
4. Strengthen explanatory conditional diagnostics beyond aggregate residual
   mean and second moment. They remain veto/explanation only; the untouched
   proposal-law audit remains the HMC nomination gate.

The repair is established only for this exact reverse-funnel diagnostic. It
does not establish a generally available warm-up for unknown targets or
SSL-LSTM posterior correctness.

## Diagnostic provenance

All new diagnostics were CPU-only, read-only with respect to frozen artifacts,
TensorFlow float64, TF32 irrelevant, and GPU devices intentionally hidden. No
optimizer update, HMC step, artifact overwrite, or package/environment change
occurred. The diagnosed state is
`reverse-funnel-capacity-r3/full-cap4-unbounded-linear/trainer_state.json`.

The commands used temporary scripts under `/tmp`; they are diagnostic-only and
not repository runtime paths.
