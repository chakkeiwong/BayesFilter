# Iterative factor-fit equivalence decision

Status: proposal only. No runtime setting, mandatory test tolerance, field, or
gate has changed. The original full-record D5 gate still fails.

The current all-field FP64 comparison uses `atol=rtol=1e-10`, while the original
factor fitter stops L-BFGS at a raw-gradient tolerance of `1e-9`. The latter
does not bound errors in fitted coefficients by `1e-10`. New direct evidence
shows that the original fitter itself exceeds the comparison gate after
one-ULP input changes, with the method, stopping rule, and decisions fixed.

## Evidence

Runs 02098/02099 CPU and 02100/02102 GPU3 capture every objective state, value,
and gradient on the original and current prepared arrays. All instrumented
results reproduce their uninstrumented complete records. Each fit uses 73
iterations and 217 evaluations. Replaying the current objective at every
original state gives maximum absolute gradient errors below `7.28e-17` CPU and
`4.16e-17` GPU. This covers the observed trajectories, not every possible input.

Runs 02101 CPU and 02103 GPU3 independently apply six declared one-ULP
perturbations to the original inputs and rerun the original fitter. All fits
retain status `usable`, the same anchors, 73 iterations, and 217 evaluations.

| Observation | CPU | GPU3 |
| --- | ---: | ---: |
| Original fields failing 1e-10 after one-ULP input changes | 1–50 | 9–53 |
| Largest original self-sensitivity, abs(delta)/(1+abs(original)) | 2.1403e-9 | 2.9149e-9 |
| Current/original difference on identical original inputs, same scaling | 4.3292e-10 | 1.7387e-9 |
| Original maximum covariance error versus analytic fixture | 9.1856e-7 | 9.1876e-7 |
| Current maximum covariance error versus analytic fixture | 9.1859e-7 | 9.1890e-7 |

The independent covariance is reconstructed from the declared latent-factor
identity and the actual changed coordinate scale. It reproduces the archived
training/holdout responses within `1.12e-16`. The finite stopped fits differ
from this exact covariance by about `9.19e-7`, precision by `1.12e-6`, and
loadings by `3.23e-6`. The existing independent factor tests already use
covariance tolerances `2e-5` (one factor) and `2e-4` (two factors).

In the full original D5 lifecycle, runs 01825 CPU / 01828 GPU have 66/53 failed
fields, all inside the first factor-fit record. Their largest scaled errors
are `3.4453e-9` / `2.8127e-9`. No endpoint, event-order, optimizer-count, target
value/score, rejection, or other controller field is among those failures.
This is evidence of amplified arithmetic sensitivity, not proof that all
compiler defects have been excluded.

## Proposed bounded change

For comparisons of **iterative factor-fit outputs only**, use
`abs(candidate-original) <= 1e-8 + 1e-8*abs(original)` for:

- `covariance_z`, `precision_z`, `projected_precision_z`,
  `marginal_standard_deviations`, and `loadings`;
- the associated covariance eigenvalues, covariance condition, prediction
  Jacobian condition, loading-row squared norms, and train/holdout score-fit
  error diagnostics.

This is an explicit 100-fold relaxation of those comparison tolerances. It is
an engineering equivalence allowance for a finite iterative solver, motivated
by measured original self-sensitivity and separate analytic accuracy checks;
it is not a theorem that all such fits are accurate to `1e-8`.

Keep every field present and checked. Preserve exact schemas, array shapes,
booleans, status strings, ranks, anchor order, iteration/evaluation counts, and
selection decisions. Keep `1e-10` for every other field, including initial and
returned target values/scores/positions, all filtering values/analytical
gradients, replay records, input/probe data, final loss, and threshold settings.
No optimizer tolerance, iteration limit, scientific model, seed, or production
decision rule changes. Keep the old 1e-10 discrepancies in result artifacts.

If approved, first qualify the comparator itself against deliberately changed
counts, decisions, missing fields, array shapes, and errors above 1e-8. Then
rerun complete original CPU/GPU D3/D5 lifecycles and their consumers. Retain the
independent covariance tests and add an analytic D5 check at absolute `1e-5`
for covariance, precision, and loadings, plus same-state objective/gradient
parity at the unchanged `1e-10` comparison. This extra D5 bound is about three
times the observed finite-fit coefficient error; it is a declared diagnostic
screen, not a new default accuracy guarantee. Public integration, memory/cost,
external watchdogs, final repeated tests, and merge gates remain mandatory.

The alternative is to retain the strict all-field gate and continue reproducing
backend rounding in every finite optimizer output. Under that alternative the
public sequential XLA integration remains blocked. Do not silently change the
optimizer stopping rule to obtain closer fitted coefficients.

## Review and decision

Primary-agent review: baseline closure and full records are preserved; both
CPU/GPU observations use identical original inputs; instrumentation validity
and original self-sensitivity are directly checked. The strongest alternative
explanation is a defect away from the sampled trajectories. Keeping independent
accuracy, full consumer, derivative, and downstream decision gates addresses
part of that risk; these tiny fixtures cannot prove general correctness. This
is not an independent review or a performance/scientific ranking.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Request a narrow comparison-contract decision | Original sensitivity and complete traces explain the numerical scale | Existing strict gate remains failed | Wider target/conditioning coverage | Owner decision, then full qualification if approved | No completed execution repair or merge readiness |

Approval is needed because this changes promotion criteria, under the
AGENTS.md academic campaign repair rule. It is separate from the pending
singular `design_condition` definition proposal; approval of either does not
implicitly authorize the other.
