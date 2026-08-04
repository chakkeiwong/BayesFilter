# Austria SIR Pairwise-Moment GenUT Score Trial Plan

Date: 2026-07-30  
Status: `AUTHORIZED_BOUNDED_FEASIBILITY_TRIAL`

## Research Intent

Test whether extending the current diagonal higher-moment GenUT/Contract-E
reset with deterministic pairwise co-skewness and co-kurtosis matching reduces
the recursive-score variance on the active Austria SIR target.

For whitened state coordinates

\[
z=L^{-1}(x-\mu),\qquad LL^\top=\Sigma,
\]

the candidate adds the ordered off-diagonal co-skewness moments

\[
C^{(3)}_{ij}=E[z_i^2z_j],\qquad i\ne j,
\]

and unordered co-kurtosis moments

\[
C^{(4)}_{ij}=E[z_i^2z_j^2],\qquad i<j.
\]

At `d=18`, this adds 306 co-skewness and 153 co-kurtosis constraints. It does
not construct the full third- or fourth-order tensors. Contract E continues to
restore the full mean and covariance matrix. Each pairwise iteration is also
restandardized so the executed correction retains zero standardized mean and
identity covariance up to numerical tolerance.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific question | Does pairwise higher-moment matching reduce Austria SIR recursive-score variance without damaging value stability or finite-program validity? |
| Baseline | Current diagonal-only candidate from `attempt05_final`: 4 diagonal steps, strength 0.2, epsilon 8, Sinkhorn/balance 16/16, ridge `1e-5` |
| Candidate | Same finite filter plus ordered pairwise co-skewness and unordered pairwise co-kurtosis correction in whitened coordinates |
| Tuning data | Fresh/disjoint Austria SIR calibration and validation trajectories plus tuning particle seeds; no claim observations or claim seeds read during selection |
| Claim data | Exact `austria_sir_T20` observations and the 16 common claim seeds `98201..98216`, `N=1008` |
| Primary promotion criterion | Candidate score SD is lower than diagonal-only on every coordinate, and the paired bootstrap/Student-t uncertainty for an aggregate standardized score-variance ratio supports a reduction |
| Promotion vetoes | Any numerical/reset/OT/score-additivity failure; value SD more than 25% above baseline; absolute mean value shift above one baseline standard error; pairwise residual fails to improve on validation; correction displacement above 2.0; any score-coordinate SD above baseline |
| Explanatory diagnostics | Pairwise and diagonal residuals, score mean/CI, SGQF inclusion, per-time score increments, runtime, allocator peak |
| Runtime score | Manual recursive forward sensitivity of the same finite value program; no runtime autodiff or finite differences |
| Nonclaims | No exact moment projection, exact nonlinear score, superiority, HMC/default readiness, Zhao-Cui result, or broad high-dimensional conclusion |
| Artifact root | `docs/benchmarks/artifacts/austria_sir_pairwise_moment_genut_score_20260730/` |

Because only 16 claim seeds are available, coordinatewise SD ratios and the
aggregate ratio remain limited stochastic evidence. The run may establish a
clear variance reduction or reject the candidate, but it cannot establish
general score accuracy without an exact Austria SIR oracle.

## Candidate Map

For a standardized equal-weight cloud `z`, form residual matrices

\[
R^{(3)}_{ij}=C^{(3),\star}_{ij}-C^{(3)}_{ij},\qquad i\ne j,
\]

and symmetric off-diagonal

\[
R^{(4)}_{ij}=C^{(4),\star}_{ij}-C^{(4)}_{ij}.
\]

Use the deterministic residual-gradient direction

\[
D_k^{(3)}=
2z_k\sum_{j\ne k}R^{(3)}_{kj}z_j
+\sum_{i\ne k}R^{(3)}_{ik}z_i^2,
\]

\[
D_k^{(4)}=
2z_k\sum_{j\ne k}R^{(4)}_{kj}z_j^2.
\]

Scale each family by `1/(d-1)`, project the combined direction out of the
sample mean/covariance tangent space, take a bounded strength step, and
restandardize. Implement the complete manual JVP of this exact map. Diagnostic
forward autodiff is allowed only in tests.

## Scope And Tuning Grid

The first bounded grid keeps the already repaired Austria OT settings fixed:

```text
epsilon=8
sinkhorn_steps=16
balance_steps=16
ridge=1e-5
diagonal_steps=4
diagonal_strength=0.2
pairwise_steps={0,1,2,4}
pairwise_strength={0.005,0.01,0.02,0.05}
pairwise_floor=1e-5
```

The zero-step arm is the diagonal-only baseline. This is a feasibility grid,
not an optimum claim. If every nonzero candidate fails the variance veto, stop
and retain diagonal-only. Do not expand the grid after viewing claim results.

The oracle-free selection order is:

1. hard numerical and mean/covariance gates;
2. lower validation pairwise residual than zero-step baseline;
3. score-variance veto on repeated validation branches;
4. value-stability veto;
5. lowest aggregate validation score variance;
6. lower correction displacement, strength, then step count.

SGQF is not read during selection and is used only as a post-run comparator.

## Skeptical Audit

| Risk | Finding and control |
| --- | --- |
| Wrong baseline | Use the exact diagonal-only controls and common seeds from `attempt05_final` |
| Proxy promoted to criterion | Pairwise residual improvement is necessary but not sufficient; score variance and value stability are promotion gates |
| Oracle leakage | SGQF and claim observations are unavailable to tuning |
| Full-tensor explosion | Candidate uses matrix contractions only: `O(Nd^2)` work and `O(d^2+Ndp)` auxiliary storage, not `d^3/d^4` tensors |
| Covariance regression | Project the update and exactly restandardize each iteration; full covariance tests are hard gates |
| JVP omission | Compare manual JVP against diagnostic forward autodiff on small fixtures and check score-increment additivity |
| Austria coupling ignored | Use every ordered/unordered pair, not only graph edges or random projections |
| Tuning saturation repeated | Pairwise grid includes an interior strength ladder and a zero-step baseline; boundary selection is reported, not called an optimum |
| Score accuracy overclaim | Austria has no exact score oracle; lower variance does not prove lower bias |
| GPU environment mismatch | Serious run uses escalated/trusted GPU, FP32, TF32, XLA, and verified memory growth |

Audit decision: `PASS_FOR_BOUNDED_FEASIBILITY_EXECUTION`.

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| Pairwise moment families | User hypothesis and current missing cross-moment analysis | candidate hypothesis | important triple/quadruple interactions remain omitted | pairwise residual and score-variance outcome |
| All coordinate pairs | deterministic alternative to random projections | candidate hypothesis | noisy irrelevant pairs dilute useful structure | validation variance and displacement |
| Existing diagonal controls | July 23 Austria selection | frozen baseline, not universal default | diagonal correction already causes variance | zero-step pairwise baseline reproduction |
| Pairwise strengths/steps | bounded engineering ladder | convenience grid | boundary optimum or instability | validation curve and eligibility |
| `N=1008`, 16 seeds | active same-target campaign | comparison scope | intervals remain broad | report CI and nonclaims |

## Implementation And Verification

1. Add pairwise moment/JVP helpers and opt-in controls with zero defaults.
2. Add actual pairwise residual diagnostics to the finite filter.
3. Unit-test pair moment formulas, mean/full-covariance restoration, zero-step
   parity, manual-JVP parity, and absence of NumPy/runtime autodiff.
4. Run CPU-hidden focused tests, including existing diagonal behavior.
5. Run a tiny trusted GPU/XLA Austria smoke.
6. Tune on disjoint calibration/validation data under the grid above.
7. Freeze controls and run the exact 16 common claim seeds.
8. Write result JSON/Markdown and a result/reset note with raw per-seed rows.

## Budget And Stop Conditions

Budget: one focused CPU test phase, one tiny GPU smoke, one 16-arm tuning phase,
and one 16-seed claim phase. Use fresh versioned attempt directories and do not
overwrite July 23 evidence.

Stop on target/hash mismatch, invalid mean/covariance, score/JVP failure,
repeated nonfinite candidates, GPU memory failure, or exhaustion of this grid.
A candidate failure rejects the current pairwise map/settings, not all
higher-order distribution matching.
