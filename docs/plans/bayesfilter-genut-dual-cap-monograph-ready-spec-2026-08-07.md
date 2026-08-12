# GenUT dual-cap algorithm family — monograph-ready integration note

- **Date:** 2026-08-09
- **Purpose:** readable self-contained specification of the dual-cap GenUT default, suitable for direct integration into the BayesFilter monograph as a policy / method note.
- **Status:** documentation spec only; not yet integrated into the chapter text.

---

## 1. Why this algorithm exists

A diagonal higher-moment GenUT reset improves the skewness and kurtosis of each standardized coordinate separately, but it does not control off-diagonal third- and fourth-moment defects such as co-skewness and co-kurtosis. A pairwise higher-moment correction addresses those cross-coordinate defects, but in difficult nonlinear models it can produce large directional updates, unstable score behavior, and poor finite-program robustness.

The dual-cap route is BayesFilter’s answer to that stability problem when one common higher-moment algorithm family must be maintained across models. It augments the pairwise higher-moment correction with two smooth bounding operations:

1. a **rowwise radial cap** on the pairwise correction direction, which limits the RMS size of each particle’s pairwise update; and
2. a **coordinatewise cap** on the final standardized cloud, which limits standardized coordinate magnitude before the final affine restoration.

The result is a bounded-influence finite program that still restores the weighted source mean and covariance exactly on the declared affine-restoration branch, while no longer claiming exact preservation of the corrected higher moments after the coordinate cap.

This route is a **BayesFilter extension/invention** built from familiar components. It is not claimed to be source-faithful Zhao–Cui, not claimed to compute an exact nonlinear likelihood score, and not claimed to establish posterior, HMC, or NeuTra readiness.

---

## 2. Executive default-selection statement

BayesFilter selects the **dual-cap GenUT algorithm family** as the default GenUT family when one common algorithm must be maintained across models.

The selected family is:

\[
\text{diagonal third/fourth-moment correction}
+ \text{pairwise co-skewness/co-kurtosis correction}
+ \text{smooth rowwise radial cap}
+ \text{smooth coordinatewise cap}
+ \text{affine restoration of source mean and covariance}.
\]

The currently selected defining controls are:

- `pairwise_moment_correction_steps = 4`
- `pairwise_particle_rms_cap = 2.0`
- `coordinatewise_standardized_cap = 0.98`
- `coordinatewise_standardized_cap_power = 8`

These values define the default **algorithm family**. They do **not** remove the requirement for scope-specific tuning of pairwise strength and any route-specific transport or reset controls.

Other implemented GenUT families remain supported and user-selectable, including:

1. diagonal-only higher-moment correction;
2. diagonal plus pairwise correction without caps;
3. diagonal plus pairwise correction with coordinate cap only;
4. bounded-teacher variants where a separately validated bounded teacher exists; and
5. projected-cumulant and other explicitly experimental research variants.

This is a maintenance/default-selection decision, not a theorem of universal statistical superiority.

---

## 3. Objects and notation

Let the weighted source cloud be

\[
X = \{x_n,w_n\}_{n=1}^N,
\qquad \sum_{n=1}^N w_n = 1,
\]

with weighted mean and covariance

\[
\mu = \sum_n w_n x_n,
\qquad
\Sigma = \sum_n w_n (x_n-\mu)(x_n-\mu)^\top,
\qquad
LL^\top = \Sigma.
\]

The reset is represented through an equal-weight standardized cloud

\[
Z = (z_{ni}) \in \mathbb R^{N\times d},
\]

whose intended first-order target is zero mean and identity covariance. The diagonal correction targets standardized skewness and kurtosis coordinate by coordinate. The pairwise correction additionally targets the cross-coordinate moments

\[
C^{(3)}_{ij} = E[Z_i^2 Z_j], \quad i\neq j,
\qquad
C^{(4)}_{ij} = E[Z_i^2 Z_j^2], \quad i<j.
\]

Write the target-minus-current residuals as

\[
R^{(3)}_{ij} = C^{(3),\star}_{ij} - C^{(3)}_{ij},
\qquad
R^{(4)}_{ij} = C^{(4),\star}_{ij} - C^{(4)}_{ij}.
\]

These residuals define the pairwise correction direction.

---

## 4. Mathematical definition of the dual-cap route

### 4.1 Raw residual-gradient direction

The implementation forms the raw pairwise residual-gradient direction

\[
D_{nk} = \frac{1}{\max(d-1,1)}\left[
2 z_{nk}\sum_j R^{(3)}_{kj} z_{nj}
+ \sum_i R^{(3)}_{ik} z_{ni}^2
+ 2 z_{nk}\sum_j R^{(4)}_{kj} z_{nj}^2
\right].
\]

This is the unconstrained pairwise correction direction before projection and caps.

### 4.2 Affine-tangent projection

The pairwise direction must not move the cloud along first-order mean/covariance tangent directions. Writing

\[
\bar D = N^{-1}\sum_n D_n,
\]

and

\[
A = \operatorname{sym}\left\{N^{-1}\sum_n z_n (D_n-\bar D)^\top\right\},
\]

the projected direction is

\[
P_n = D_n - \bar D - A z_n.
\]

This step removes the affine-tangent component of the pairwise residual update.

### 4.3 Global RMS normalization

Normalize the projected direction globally by

\[
Q_n = \frac{P_n}{\sqrt{N^{-1} d^{-1} \sum_m \lVert P_m\rVert^2 + \delta}},
\]

where \(\delta\) is the configured pairwise floor. This produces a dimensionless pairwise update direction with controlled global scale.

### 4.4 Smooth rowwise radial cap

The first cap acts on each particle row through its RMS size:

\[
r_n^2 = d^{-1}\lVert Q_n\rVert^2,
\qquad
s_n = \left(1 + \frac{r_n^2}{c_r^2}\right)^{-1/2},
\qquad c_r = 2,
\]

so the radially capped pairwise direction is

\[
\widetilde Q_n = s_n Q_n.
\]

This is a smooth bounded-influence control on each particle’s pairwise correction magnitude.

### 4.5 One pairwise correction step

One pairwise step is

\[
Z \leftarrow \operatorname{standardize}\bigl(Z + \rho_{\rm pair}\,\widetilde Q\bigr),
\]

where \(\rho_{\rm pair}\) is the scope-specific pairwise strength.

The selected default family uses **four** such pairwise steps.

For \(d=1\), there are no off-diagonal pairwise moments. In that case the pairwise loop is skipped exactly, and the radial cap is an exact no-op.

### 4.6 Smooth coordinatewise cap

After the diagonal and pairwise iterations, apply the coordinatewise smooth cap

\[
f_b(z) = \frac{z}{\{1 + (z/b)^p\}^{1/p}},
\qquad b = 0.98,
\qquad p = 8.
\]

For finite \(z\), this satisfies

\[
|f_b(z)| < b,
\]

and its derivative is

\[
f_b'(z) = \{1 + (z/b)^p\}^{-1/p-1}.
\]

This is the second smooth bounding operation in the dual-cap family.

### 4.7 Affine restoration

Finally, map the capped standardized cloud back to the weighted source mean and covariance:

\[
\widehat Z = \operatorname{standardize}\{f_b(Z)\},
\qquad
X_n^+ = \mu + L\widehat Z_n.
\]

The executed route therefore restores the weighted source mean and covariance on the valid affine-restoration branch.

---

## 5. Exact and non-exact properties

### Proposition: exact first-two-moment restoration on the declared branch

On a valid Cholesky / affine-restoration branch, the executed route restores the weighted source mean and covariance exactly after the final standardization and affine map.

### Important nonclaim

The route does **not** preserve the previously matched third- and fourth-moment targets exactly after the coordinate cap. The post-cap higher moments are part of the empirical executed algorithm, not an exact higher-moment projection theorem.

That distinction must remain explicit anywhere this algorithm is documented.

---

## 6. Algorithm order

The documentation should present the executed finite program in the following order:

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

This order matters. In particular, the coordinate cap is applied **after** the pairwise iterations, and the final affine restoration is what closes the first-two-moment contract of the executed finite program.

---

## 7. Four-model evidence

### Scope
The current evidence campaign used:
- `N = 1008`
- 16 common claim seeds `98201..98216`
- disjoint calibration trajectories and seeds
- TensorFlow FP32 with TF32 enabled, GPU/XLA
- LGSSM `T=50`
- KSC transformed SV `T=10`
- predator-prey `T=20`
- Austria SIR `T=20`

### Compact value comparison

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

### How to read this evidence

This evidence supports a **default-selection judgment under a maintenance constraint**, not a theorem that the dual-cap family is universally superior.

- In LGSSM, the dual route is descriptively strongest among the tested GenUT variants, but Kalman remains the correct exact model-specific method.
- In scalar KSC, the radial cap is structurally inert, so dual and coordinate-only coincide.
- In predator-prey, dual and coordinate-only are descriptively very close.
- In Austria SIR, the dual route improves approximate-reference score proximity relative to simpler variants, which is the main reason to prefer it when one common family must be maintained.

This is enough to justify a default-family choice. It is **not** enough to claim exact nonlinear score correctness, universal superiority, or HMC readiness.

---

## 8. Why dual-cap is the single default family

No tested GenUT alternative is uniformly better than the dual family across the four-model comparison.

The dual family therefore has the right asymmetric maintenance payoff:
- material help in the hardest Austria scope,
- negligible or inert changes in simpler scopes,
- and one common maintained algorithm family rather than per-model fragmentation.

This is exactly the kind of owner-directed default decision the repository policy allows:
- a maintenance/default-selection choice,
- under bounded evidence,
- with explicit nonclaims.

---

## 9. Required warnings and limitations

The final integrated monograph text must keep all of the following warnings explicit.

### 9.1 The coordinate cap is not a rare-tail-only event
In the tested generic charts, cap-active fractions were large. The cap must therefore be described as a bounded-influence finite-program regularization, not as a near-never-triggered tail repair.

### 9.2 The cap changes the finite value program
The dual route computes the value and its total derivative consistently for the **same executed finite program**, but that finite program is not the same as the uncapped one. Reduced variance or improved approximate-reference proximity does not imply smaller score bias for the underlying state-space likelihood.

### 9.3 Small-step FD sensitivity remains unresolved in several scopes
The evidence still does not support derivative-admission or HMC-readiness claims for this route across the tested nonlinear scopes.

### 9.4 Austria T20 still lacks an exact score authority
UKF and SGQF remain approximate references. They are useful diagnostics, not exact observed-data score theorems for Austria T20.

### 9.5 Scope-specific tuning remains mandatory
The default chooses the **algorithm family**, not one universal numerical configuration.

### 9.6 No posterior or HMC claim
The campaign tests finite values, recursive scores, seed dispersion, and approximate-reference proximity. It does not establish posterior agreement, NeuTra quality, or exact HMC validity.

---

## 10. Stable public naming

If this is wired into public selectors, use explicit stable names such as:

| Public name | Meaning |
|---|---|
| `dual_cap` | Default family: diagonal + pairwise + radial RMS cap 2 + standardized coordinate cap `.98/8` + affine restoration |
| `coordinate_cap` | Same pairwise correction and coordinate cap, radial cap disabled |
| `pairwise` | Diagonal and pairwise corrections, both caps disabled |
| `diagonal` | Diagonal third/fourth-moment correction only |
| `none` | No higher-moment correction; mechanics/reference option |
| `bounded_teacher` | Separately validated bounded-teacher route |
| `projected_cumulant` | Experimental projected higher-cumulant route |

Changing the default alias must not silently change the meaning of an explicit historical option.

---

## 11. What should appear in the monograph after integration

The final monograph documentation should contain, in this order:

1. a short motivation section,
2. an object dictionary,
3. the mathematical definition of the dual-cap route,
4. the proposition about exact first-two-moment restoration and non-preservation of higher moments,
5. an algorithm block in the executed order,
6. compact four-model evidence tables,
7. a default-selection paragraph,
8. a limitations/nonclaims paragraph,
9. and (if needed) a short implementation-facing naming note.

The current integration note’s section titled “Suggested LaTeX Integration” should disappear once the content is actually integrated.

---

## 12. Final decision statement

> BayesFilter promotes the dual-cap GenUT family as the default GenUT algorithm family because it is the strongest single maintenance compromise across the tested LGSSM, KSC SV, predator-prey, and Austria SIR scopes. The family combines pairwise third/fourth cross-moment correction, a smooth rowwise radial RMS cap of 2, a smooth standardized coordinate cap with `b=0.98`, `p=8`, and affine restoration of the weighted source mean and covariance. Explicit diagonal, pairwise-only, coordinate-cap-only, no-correction, bounded-teacher, and experimental variants remain selectable. This is an owner-directed algorithm/default policy under a maintenance constraint, not evidence of exact nonlinear score correctness, universal statistical superiority, posterior validity, or HMC/NeuTra readiness.
