# GenUT Dual-Cap Default Algorithm: Documentation Integration Note

Date: 2026-08-07

Decision status: `OWNER_SELECTED_ALGORITHM_FAMILY_DEFAULT`

Intended reader: documentation agent integrating this material into the
BayesFilter LaTeX treatment of GenUT and score variance.

## Executive Decision

BayesFilter selects the **dual-cap GenUT algorithm family** as the default
GenUT choice when one common algorithm must be maintained across models. The
other implemented GenUT variants remain supported, user-selectable options:

1. diagonal higher-moment correction only;
2. diagonal plus pairwise correction, uncapped;
3. diagonal plus pairwise correction and coordinate cap only;
4. bounded-teacher variants where a separately validated bounded teacher
   exists; and
5. projected-cumulant and other research variants, explicitly labeled
   experimental.

The selected family is:

```text
diagonal third/fourth-moment correction
+ pairwise co-skewness/co-kurtosis correction
+ smooth rowwise radial cap on the pairwise correction direction
+ smooth coordinatewise cap on the final standardized cloud
+ affine restoration of the source mean and covariance
```

The currently selected defining controls are:

```text
pairwise_moment_correction_steps = 4
pairwise_particle_rms_cap = 2.0
coordinatewise_standardized_cap = 0.98
coordinatewise_standardized_cap_power = 8
```

The pairwise strength and all transport/reset controls remain scope-specific.
In the four-model evidence, pairwise strength was `0.02` for LGSSM, KSC SV,
and Austria SIR and `0.05` for predator-prey. Sinkhorn epsilon/iteration counts
were `2/8` for the first three scopes and `8/16` for Austria. These numbers are
evidence-scope settings, not universal constants.

This is a maintenance/default-selection decision. It is not a claim that the
dual route is statistically superior in every model, computes an exact
nonlinear likelihood score, is source-faithful Zhao-Cui, or is ready for HMC,
NeuTra, or posterior promotion.

## Current Implementation And Promotion Boundary

The active research workspace contains the mechanics as opt-in controls in:

- `bayesfilter/highdim/higher_moment_contract_e.py`;
- `bayesfilter/highdim/cubature_genut_filter.py`; and
- the scalar comparison wiring in
  `docs/benchmarks/run_moment_retuned_genut_whole_leaderboard.py`.

The evidence campaign used:

- `docs/benchmarks/run_genut_b098_radial2_four_model.py`.

This integration note and its evidence commit record the owner decision; they
do not commit the active mixed research implementation diff or silently alter
historical call behavior. The low-level function-signature defaults in that
research implementation remain zero/off so historical callers and explicit
options retain exact behavior. Thus the **algorithm-family policy default is
dual-cap**, while public factory/CLI selector wiring remains a separate bounded
implementation change. Documentation must not claim that every existing
constructor already executes dual-cap. A repository-owned selector should map
`algorithm="default"` to the dual family and expose the alternatives by name.

## Mathematical Definition

Let the weighted source cloud be

\[
  X=\{x_n,w_n\}_{n=1}^N,
  \qquad \sum_n w_n=1,
\]

with weighted mean and covariance

\[
  \mu=\sum_n w_n x_n,
  \qquad
  \Sigma=\sum_n w_n(x_n-\mu)(x_n-\mu)^\top,
  \qquad LL^\top=\Sigma.
\]

The reset cloud is standardized to an equal-weight cloud
\(Z=(z_{ni})\) with approximately zero mean and identity covariance. The
diagonal correction targets standardized skewness and kurtosis. The pairwise
correction additionally targets

\[
 C^{(3)}_{ij}=E[Z_i^2 Z_j],\quad i\ne j,
 \qquad
 C^{(4)}_{ij}=E[Z_i^2 Z_j^2],\quad i<j.
\]

For residual matrices

\[
 R^{(3)}_{ij}=C^{(3),\star}_{ij}-C^{(3)}_{ij},
 \qquad
 R^{(4)}_{ij}=C^{(4),\star}_{ij}-C^{(4)}_{ij},
\]

the implementation forms the raw residual-gradient direction

\[
 D_{nk}=\frac{1}{\max(d-1,1)}\left[
 2z_{nk}\sum_j R^{(3)}_{kj}z_{nj}
 +\sum_i R^{(3)}_{ik}z_{ni}^2
 +2z_{nk}\sum_j R^{(4)}_{kj}z_{nj}^2
 \right].
\]

It projects this direction away from the first-order mean/covariance tangent
space. Writing \(\bar D=N^{-1}\sum_nD_n\) and

\[
 A=\operatorname{sym}\left\{N^{-1}\sum_n
 z_n(D_n-\bar D)^\top\right\},
\]

the projected direction is

\[
 P_n=D_n-\bar D-Az_n.
\]

The global RMS normalization is

\[
 Q_n=\frac{P_n}{
 \sqrt{N^{-1}d^{-1}\sum_m\lVert P_m\rVert^2+\delta}},
\]

where \(\delta\) is the configured pairwise floor. The radial cap is applied
independently to each particle row using its coordinate RMS

\[
 r_n^2=d^{-1}\lVert Q_n\rVert^2,
 \qquad
 s_n=\left(1+\frac{r_n^2}{c_r^2}\right)^{-1/2},
 \qquad c_r=2,
\]

and therefore

\[
 \widetilde Q_n=s_nQ_n.
\]

One pairwise step is

\[
 Z\leftarrow\operatorname{standardize}
 \left(Z+\rho_{\rm pair}\widetilde Q\right),
\]

and the selected default family uses four such steps. For \(d=1\), there are
no off-diagonal pairwise moments, the pairwise loop is skipped exactly, and
the radial cap is consequently an exact no-op.

After the diagonal/pairwise iterations and their restandardizations, apply the
coordinatewise smooth cap

\[
 f_b(z)=\frac{z}{\{1+(z/b)^p\}^{1/p}},
 \qquad b=0.98,\quad p=8.
\]

For finite \(z\), \(|f_b(z)|<b\), and

\[
 f_b'(z)=\{1+(z/b)^p\}^{-1/p-1}.
\]

The complete manual tangent differentiates the moment residuals, affine
projection, RMS normalizations, radial scale, coordinate cap, and subsequent
affine restoration. Finally, the capped cloud is mapped back to the weighted
source mean and covariance:

\[
 X_n^+=\mu+L\widehat Z_n,
 \qquad
 \widehat Z=\operatorname{standardize}\{f_b(Z)\}.
\]

Consequently, the executed route restores the source mean and covariance but
does **not** preserve the previously matched third/fourth moments exactly after
the coordinate cap. The post-cap higher moments are part of the empirical
algorithm, not an exact moment-projection claim.

## Algorithm Order

The documentation should show the following order explicitly:

```text
weighted filtered source cloud
-> Contract-E mean/covariance reset
-> standardize equal-weight reset cloud
-> diagonal third/fourth-moment correction and restandardization
-> pairwise co-skewness/co-kurtosis correction
   -> affine-tangent projection
   -> global RMS normalization
   -> rowwise smooth radial RMS cap c_r=2
   -> strength step and restandardization
-> repeat pairwise step four times
-> coordinatewise smooth standardized cap b=.98, p=8
-> affine restoration to weighted source mean and covariance
-> propagate the complete manual tangent
```

The coordinate cap used here is the generic standardized-coordinate cap. It
must not be confused with the separate bounded-teacher chart cap used in the
strict Austria T1/T2 teacher experiments. The generic cap is a BayesFilter
`extension_or_invention`, not an operation claimed from Zhao and Cui.

## Four-Model Evidence

Primary artifact:

`docs/benchmarks/artifacts/genut_b098_radial2_four_model_20260807/attempt01/result.json`

Run scope:

- `N=1008`;
- 16 common claim seeds `98201..98216`;
- disjoint calibration trajectories and seeds;
- TensorFlow FP32, TF32 enabled, GPU/XLA;
- LGSSM `T=50`, KSC transformed SV `T=10`, predator-prey `T=20`, and
  Austria SIR `T=20`.

Every diagonal, coordinate-cap, and dual-cap arm passed finite/program,
transport/reset residual, and claim validity gates. The Austria uncapped
pairwise arm passed the claim observations but failed one disjoint calibration
score-additivity row, so it is not an eligible default candidate.

### Compact value comparison

Entries are the 16-seed GenUT mean with sample SD in parentheses. Reference
rows are deterministic approximations/oracles and have no particle-seed SD.

| Model | Reference value | Diagonal | Pairwise | Coordinate only | Dual default |
|---|---:|---:|---:|---:|---:|
| LGSSM T50 | Kalman `-136.07597` | `-136.33349 (0.46797)` | `-136.33247 (0.47084)` | `-136.33300 (0.46949)` | `-136.33006 (0.46809)` |
| KSC SV T10 | dense `-19.95628` | `-19.95395 (0.04760)` | identical to diagonal | `-19.95785 (0.04894)` | identical to coordinate |
| Predator-prey T20 | SGQF `-102.62270` | `-102.73954 (0.29234)` | `-102.74495 (0.30723)` | `-102.72680 (0.30635)` | `-102.72772 (0.30553)` |
| Austria SIR T20 | UKF `-681.68863`; SGQF `-682.34801` | `-683.36381 (0.63669)` | `-682.10394 (0.56474)` | `-681.74665 (0.55539)` | `-681.82315 (0.70390)` |

### Compact score comparison

| Model | Reference score | Diagonal mean | Pairwise mean | Coordinate-only mean | Dual-default mean |
|---|---|---|---|---|---|
| LGSSM | Kalman `[5.655,-3.835,0.302,-1.917,4.354]` | `[5.795,-4.050,0.240,-1.984,5.538]` | `[5.784,-4.032,0.221,-2.056,5.528]` | `[5.690,-3.989,0.198,-2.116,5.536]` | `[5.708,-3.990,0.203,-2.104,5.504]` |
| KSC SV | dense `[-0.706,0.635]` | `[-0.694,0.608]` | identical to diagonal | `[-0.707,0.575]` | identical to coordinate |
| Predator-prey | SGQF `[-27.641,.084,-.084,.856,17.526,-22.635]` | `[-27.775,.078,-.087,1.042,18.367,-23.651]` | `[-27.805,.072,-.088,1.041,18.434,-23.735]` | `[-27.769,.076,-.087,1.016,18.262,-23.525]` | `[-27.748,.075,-.087,1.015,18.244,-23.502]` |
| Austria SIR | UKF `[29.184,-106.963,9.327]`; SGQF `[28.739,-106.659,9.431]` | `[-865.923,170.885,114.981]` | `[-16.905,-108.702,15.152]` | `[39.318,-109.040,11.323]` | `[33.383,-106.884,10.230]` |

### Interpretation by model

**LGSSM.** The dual arm has the smallest tested GenUT value gap and score
Euclidean error against the exact Kalman reference (`1.180`, versus `1.213`
for coordinate-only and `1.214` for diagonal). The improvement is small and
mostly within paired uncertainty. Kalman remains the correct model-specific
method whenever its assumptions hold.

**KSC SV.** Pairwise and radial corrections are exact structural no-ops at
state dimension one. The dual arm therefore equals coordinate-only. The
uncapped diagonal route has better dense-reference score agreement, while the
capped route has a slightly closer value. This is the clearest counterexample
to a claim of per-model dominance, but it does not create an additional radial
maintenance path because radial is inert.

**Predator-prey.** Coordinate-only and dual are descriptively almost
indistinguishable. The dual score is marginally closer to the same-target SGQF
and Zhao-Cui diagnostics, but all paired radial-minus-coordinate intervals
cover zero and dual score SDs are slightly larger. No exact nonlinear score
authority exists.

**Austria SIR.** This is the main reason to prefer the dual family when one
common GenUT variant is required. The dual mean score is much closer to both
same-target Gaussian-closure diagnostics:

```text
dual minus UKF:  [+4.20, +0.08, +0.90]
dual minus SGQF: [+4.64, -0.23, +0.80]
coordinate minus UKF:  [+10.13, -2.08, +2.00]
coordinate minus SGQF: [+10.58, -2.38, +1.89]
```

This proximity is descriptive only. The dual arm increases value SD by 26.7%
and score-0 SD by 14.3% relative to coordinate-only, and its value is still
shifted by `+1.541` log units relative to the diagonal route, about 9.7
diagonal-baseline MCSEs. UKF and SGQF are approximate references, not exact
Austria score authorities.

## Why Dual Is The Single Default

No tested GenUT alternative is uniformly better than dual across the four
models:

- dual is descriptively best among tested GenUT variants against exact Kalman
  on the aggregate LGSSM value/score comparison;
- dual adds no behavior relative to coordinate-only in scalar KSC;
- dual is effectively tied with coordinate-only in predator-prey; and
- dual gives the strongest approximate-reference score agreement in Austria,
  where the simpler variants have the largest practical difficulty.

The radial cap therefore has an asymmetric maintenance payoff: material help
in the difficult Austria scope, negligible change in LGSSM and predator-prey,
and an exact no-op in scalar KSC. This supports choosing dual when maintaining
one GenUT algorithm family is more important than per-model specialization.

This reasoning is a default-selection judgment under the user's maintenance
constraint. It is not a statistically supported theorem that dual is superior.

## Known Issues And Required Warnings

### 1. The coordinate cap is not tail-only in the tested generic charts

Maximum cap-active fractions were approximately:

| Model | Active fraction | Mean coordinate displacement |
|---|---:|---:|
| LGSSM | `73.7%` | `0.2409` |
| KSC SV | `74.9%` | `0.2283` |
| Predator-prey | `70.8%` | `0.2314` |
| Austria SIR | `78.6%` | `0.1654` |

Thus the generic `b=.98` cap changes a majority of standardized coordinates.
Documentation must not call it a rare-tail-only operation. Its practical role
is bounded influence and finite-program regularization, followed by exact
first-two-moment restoration.

### 2. The cap changes the finite objective

The same finite GenUT value and its total derivative are computed together,
but the dual value program is different from the uncapped value program. A
smaller score variance or closer approximate-reference score does not imply a
smaller score bias for the underlying state-space likelihood.

### 3. Internal small-step FD sensitivity remains unresolved

At `h=1e-3`, maximum same-program finite-difference absolute residuals for the
dual arm were:

| Model | Maximum absolute residual |
|---|---:|
| LGSSM | `0.575` |
| KSC SV | `0.000511` |
| Predator-prey | `0.582` |
| Austria SIR | `201.85` |

Only KSC passed the prior small-step FD screen. The corresponding diagonal
baselines also failed for LGSSM, predator-prey, and Austria, so this is not an
isolated radial-cap failure; it is consistent with inherited FP32/TF32
small-step sensitivity and conditioning. Nevertheless, the result blocks any
claim that derivative admission or HMC readiness has been established.

### 4. Austria lacks an exact T20 score authority

No valid parameterized T20 observed-data Zhao-Cui score is available. The
fixed Zhao-Cui Austria example has no free parameter score; existing
parameterized Zhao-Cui artifacts are T1/T2 mechanics, local-complete-data, or
proposal diagnostics. UKF and SGQF are approximate Gaussian closures.

### 5. Scope-specific tuning remains mandatory

The default selects the dual **algorithm family**, not one universal numerical
configuration. Every claim-bearing LEDH/GenUT scope must retain its own offline
tuning artifact and exact target/event-order/data identity. Cross-model control
transfer is a warm start only.

### 6. No posterior or HMC claim

The four-model campaign tests finite values, recursive scores, seed dispersion,
and approximate-reference proximity. It does not test posterior agreement,
NeuTra training quality, HMC acceptance/convergence, or exact invariant-target
behavior.

## User-Selectable Algorithm Names

For future public wiring, use explicit stable names such as:

| Public name | Meaning |
|---|---|
| `dual_cap` | Default family: diagonal + pairwise + radial RMS cap 2 + standardized coordinate cap `.98/8` + affine restoration |
| `coordinate_cap` | Same pairwise correction and coordinate cap, radial cap disabled |
| `pairwise` | Diagonal and pairwise corrections, both caps disabled |
| `diagonal` | Diagonal third/fourth-moment correction only |
| `none` | No higher-moment correction; mechanics/reference option |
| `bounded_teacher` | Separately validated bounded-teacher route; never inferred from `dual_cap` |
| `projected_cumulant` | Experimental projected higher-cumulant route |

Do not change the semantics of an explicit historical option when changing the
default alias. Existing artifacts must remain replayable with their recorded
controls.

## Suggested LaTeX Integration

The documentation agent should integrate this note into
`docs/bayesfilter-genut-score-variance-problem-and-repair-note-2026-07-31.tex`
or the corresponding current chapter using the following structure:

1. Add a definition titled **Dual-cap GenUT reset** after the pairwise-moment
   repair definition.
2. State the pairwise residual-gradient direction, affine projection, global
   RMS normalization, rowwise radial cap, coordinate cap, and affine
   restoration equations above.
3. Add a proposition that both smooth caps are differentiable for finite
   inputs and that affine restoration recovers the weighted source mean and
   covariance on a valid Cholesky branch. Do not claim preservation of matched
   third/fourth moments after capping.
4. Add a short implementation-order algorithm block.
5. Add the compact four-model value/score tables, preferably separating exact
   Kalman evidence from approximate SGQF/UKF/Zhao-Cui diagnostics.
6. Add a boxed owner decision: dual-cap is the default GenUT family under the
   single-algorithm maintenance constraint; explicit alternatives remain
   selectable.
7. Add a limitations paragraph covering cap-active fractions, changed finite
   objective, T20 Austria reference limitations, FD sensitivity, and absence
   of posterior/HMC evidence.
8. Keep Zhao-Cui terminology precise: pairwise and cap operations are
   BayesFilter extensions/inventions unless a separate source-grounded bounded
   teacher route is explicitly in scope.

## Evidence And Source Paths

Primary comparison:

- `docs/benchmarks/artifacts/genut_b098_radial2_four_model_20260807/attempt01/result.json`
- `docs/plans/bayesfilter-genut-b098-radial2-four-model-plan-2026-08-07.md`
- `docs/plans/bayesfilter-genut-b098-radial2-four-model-result-2026-08-07.md`

The campaign runner and cap implementation remain in the active research
workspace and should be committed with the later public-selector wiring after
their mixed bounded-teacher dependency diff is split into a reviewed change.

Austria T20 arm ladder:

- `docs/benchmarks/artifacts/zhao_cui_genut_austria_t20_dual_cap_20260807/attempt01/result.json`
- `docs/plans/bayesfilter-zhao-cui-genut-austria-t20-dual-cap-result-2026-08-07.md`

Implementation locations inspected for this handoff:

- `bayesfilter/highdim/higher_moment_contract_e.py`
- `bayesfilter/highdim/cubature_genut_filter.py`
- `tests/highdim/test_higher_moment_contract_e.py`

The active workspace four-model campaign passed 29 focused
higher-moment/GenUT tests before execution. The campaign result JSON SHA-256 at
handoff is
`906592080bc00de939b01be7c87e50c2f1d533fe8700293f5021067f817a3acd`.

## Final Decision Statement For Reuse

> BayesFilter promotes dual-cap GenUT as the default GenUT algorithm family
> because it is the strongest single maintenance compromise across the tested
> LGSSM, KSC SV, predator-prey, and Austria SIR scopes. The default combines
> pairwise third/fourth cross-moment correction, a smooth rowwise radial RMS
> cap of 2, a smooth standardized coordinate cap with `b=0.98,p=8`, and affine
> mean/covariance restoration. Explicit diagonal, pairwise-only,
> coordinate-cap-only, no-correction, bounded-teacher, and experimental
> variants remain user-selectable. This promotion is an owner-directed
> algorithm/default policy under a maintenance constraint, not evidence of
> exact nonlinear score correctness, universal statistical superiority,
> posterior validity, or HMC/NeuTra readiness.
