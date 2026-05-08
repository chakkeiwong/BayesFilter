# Mathematical audit of Chapter 18 and Chapter 18b

## Scope and method

This review audits the mathematical derivations and claim boundaries in:

- `docs/chapters/ch18_svd_sigma_point.tex`
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex`

I checked the chapter statements against their internal derivations, nearby BayesFilter contract chapters, and the local source PDFs:

- Julier and Uhlmann, *A General Method for Approximating Nonlinear Transformations of Probability Distributions* (1996)
- van der Merwe and Wan, *Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models* (2003)

I also spot-checked the worked numeric values in the first Chapter 18b structural UKF example and the `\phi=0` edge case. Those values are internally consistent with the stated point set and weights.

## Executive summary

The core law-level doctrine across both chapters is mathematically strong:

- the predictive law for a mixed structural transition is the pushforward of pre-transition uncertainty through the structural map;
- a sigma-point backend should place points on the variables that generate that law;
- SVD/square-root techniques address covariance factorization and derivative-path issues, not structural-law specification by themselves.

The main audit concerns are not basic derivation failures. They are mostly **claim-boundary and wording discipline**:

1. Chapter 18b sometimes overstates unscented-transform / UKF accuracy claims relative to what is actually derived.
2. Some uses of the word **exact** should be replaced by **same target law** or similarly narrower language.
3. Chapter 18 correctly derives branch-local spectral derivative formulas, but a few summary phrases are broader than the assumptions justify.
4. The worked numerical examples in Chapter 18b are plausible and mostly reproducible, but they should be labeled more explicitly as illustrative calculations under a specific UT parameter choice.

---

## Findings for Chapter 18b: structural deterministic dynamics

## 1. Strong points that look mathematically sound

### 1.1 Predictive pushforward formulation is correct

The chapter’s central law-level statement is sound: if
\[
A_t=(x_{t-1},\varepsilon_t), \qquad x_t = F_\theta(A_t),
\]
then the predictive law of `x_t` is the pushforward of the joint law of `(x_{t-1}, \varepsilon_t)` through `F_\theta`, under the stated independence assumptions.

This appears in:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:47-84`
- Proposition `prop:bf-structural-ukf-pushforward` at `docs/chapters/ch18b_structural_deterministic_dynamics.tex:622-697`
- the degenerate linear / nonlinear observation worked derivations at `docs/chapters/ch18b_structural_deterministic_dynamics.tex:1413-1498`

The proof strategy via bounded test functions is the right level of mathematical hygiene for the chapter.

### 1.2 Structural-vs-naive UKF doctrine is correct at the law level

The chapter’s claim that a naive full-state additive-noise construction generally targets a different predictive law if it injects direct perturbations into deterministic-completion coordinates is mathematically correct, provided the reference model is the declared structural transition.

This is well supported in:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:204-253`
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:505-589`
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:615-697`

The chapter is strongest when it states this as a **law mismatch relative to the declared model**, not as a generic condemnation of all approximate full-state filters.

### 1.3 Linear exactness vs nonlinear misuse is handled well

The distinction between:

- an exact linear-Gaussian collapsed representation that preserves the same conditional law, and
- a nonlinear sigma-point construction that perturbs the wrong variables,

is mathematically and conceptually sound.

The relevant discussion in `docs/chapters/ch18b_structural_deterministic_dynamics.tex:157-183` is one of the cleanest parts of the chapter.

### 1.4 The `\phi=0` edge-case interpretation is correct

The reviewer-response section around:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:1029-1076`

is mathematically right. Setting `\phi=0` removes the lagged-`k` memory channel but does **not** generally collapse the state to a point when `\gamma \neq 0` and `m_t` remains stochastic. The support becomes a lower-dimensional constraint set `k=\gamma m^2`, not a deterministic singleton.

The displayed numbers in that subsection are also consistent with the stated example.

---

## 2. High-priority issues in Chapter 18b

### 2.1 UT/UKF accuracy proposition is stronger in prose than in proof

The most important caution is the proposition:

- `prop:bf-structural-ukf-ut-accuracy`
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:708-795`

The actual proof establishes a **local small-uncertainty Taylor expansion statement**:

- transformed mean matched through the quadratic term;
- covariance matched in its leading linearized term;
- residuals controlled as `O(\eta^3)` under the scaling introduced in the proposition.

That is a legitimate local derivation. However, the surrounding wording risks sounding like a standard, broad unscented-transform theorem inherited directly from Julier (1996) or van der Merwe (2003). Those sources support the general sigma-point moment-matching picture, but they do not directly state this structural proposition in the chapter’s exact form.

In particular, the sentence at:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:752-756`

should be tightened. As written, “inherits the usual local second-order moment accuracy” can be overread as a general UT theorem rather than a chapter-local Taylor reduction on the correct structural input law.

#### Recommendation

Rewrite the proposition summary to say something like:

> Under the local small-uncertainty Taylor expansion used here, the structural UKF matches the transformed mean through the quadratic term and matches the leading covariance term for the Gaussian approximation on the correct pre-transition input variable.

Then add an explicit sentence:

> This is a local expansion statement for the selected structural input law, not a blanket exactness theorem for nonlinear structural transitions.

### 2.2 “Exact only if” is too strong / too ambiguous

In Proposition `prop:bf-structural-ukf-pushforward`, part (iii) currently says:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:641-644`

that a naive full-state additive-noise UKF “is exact only if” its induced predictive law coincides with the structural pushforward law.

The problem is not the law-comparison logic; that part is right. The problem is the word **exact**.

Law coincidence only removes the **structural target mismatch**. It does **not** imply exact UKF filtering for a nonlinear map unless additional conditions hold. So the sentence is mathematically easy to misread.

#### Recommendation

Replace:

> is exact only if

with something like:

> targets the same predictive law only if

or:

> avoids structural-law mismatch only if

This would align the claim with what the proposition actually proves.

### 2.3 Scaled-UT numeric example under-specifies the parameter choice

The first worked structural UKF example gives a fully numeric sigma-point table and then states:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:1092-1123`

that with “the scaled unscented choice `\lambda=0`” the central mean weight is zero and later that the covariance weight is `w_0^{(c)}=2`.

Those numbers are internally consistent, and I reproduced the example values. But `w_0^{(c)}` depends on `\alpha`, `\beta`, and `\kappa`, not only on `\lambda`. So the chapter should state the exact UT parameter triple if it wants the table to be reproducible from the prose alone.

#### Recommendation

Add the explicit parameter choice used for the example, or say clearly that the calculations adopt one specific scaled-UT convention with the stated weights.

### 2.4 Numerical examples are valid as illustrations, but their evidentiary status should be narrower

The numerical examples in:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:1081-1319`
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:1553-1788`

look mathematically coherent, and the first example plus the `\phi=0` edge case check out numerically. But the chapter should not leave them sounding like standalone empirical evidence. They are best treated as **worked illustrative calculations**.

#### Recommendation

Add one sentence near each example such as:

> These numbers are illustrative calculations under the stated sigma-point convention and should be read as worked examples rather than general performance evidence.

---

## 3. Medium-priority issues in Chapter 18b

### 3.1 Some “wrong model” wording should be relativized to the declared structural model

At several points, the chapter says or implies that a naive full-state perturbation yields a “different” or “wrong” model. The intended meaning is clear, but mathematically it is better to say:

- different **relative to the declared structural transition**;
- or a different **approximate state-space model**.

This matters especially around:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:227-235`
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:1220-1228`

The chapter already mostly knows this distinction; it just needs slightly more consistent wording.

### 3.2 Cross-covariance and update-law “failure” language should stay generic, not absolute

The worked examples convincingly show that changing the predictive law generally changes:

- `P_{xz,t}`
- `S_t`
- `K_t`
- posterior means and likelihood contributions

But some phrases in the example sections read almost like universal theorems rather than consequences of the class of naive constructions under discussion.

The relevant passages are around:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:927-988`
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:1714-1788`

#### Recommendation

Prefer phrases like:

- “generally changes”
- “in this class of constructions changes”
- “changes in the worked example because ...”

### 3.3 UT literature attribution should distinguish early UT from later scaled-UT conventions

The section:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:354-399`

correctly discusses sigma-point construction and weights, but it mixes the historical UT narrative with scaled-UKF parameters in a way that could be more precise. The scaled `(\alpha,\beta,\kappa)` rule is closer to later sigma-point / scaled UKF formulations than to Julier (1996) alone.

#### Recommendation

Attribute the scaled-weight formulas explicitly to the later sigma-point Kalman filter tradition, especially van der Merwe.

---

## 4. Findings for Chapter 18: SVD sigma-point filters

## 4.1 Strong points that look mathematically sound

### 4.1.1 The chapter correctly separates value-side factorization from derivative-side risk

The distinction between:

- value-side robustness from spectral factorization, and
- gradient / Hessian instability from spectral-gap denominators,

is mathematically sound and well framed.

Strong sections include:

- `docs/chapters/ch18_svd_sigma_point.tex:35-52`
- `docs/chapters/ch18_svd_sigma_point.tex:439-456`

These align with the standard eigenderivative picture.

### 4.1.2 The moment-derivative recursions are internally coherent

The chapter’s sigma-point derivative recursions, especially:

- point derivatives at `docs/chapters/ch18_svd_sigma_point.tex:101-108`
- map derivatives at `docs/chapters/ch18_svd_sigma_point.tex:122-138`
- transformed moment derivatives at `docs/chapters/ch18_svd_sigma_point.tex:142-188`
- update-object derivatives at `docs/chapters/ch18_svd_sigma_point.tex:190-323`

are mathematically well structured. The solve-form treatment of the Kalman gain derivative is also appropriate.

### 4.1.3 Reconstruction identities are a strong and correct validation target

The factor-reconstruction identities around:

- `docs/chapters/ch18_svd_sigma_point.tex:458-481`
- `docs/chapters/ch18_svd_sigma_point.tex:805-822`

are exactly the right kind of branch-aware derivative validation target. This is one of the strongest pieces of mathematical engineering discipline in the chapter.

### 4.1.4 The SVD-CUT affine exactness proposition is sound

The proposition:

- `prop:bf-svd-cut-affine-exactness`
- `docs/chapters/ch18_svd_sigma_point.tex:551-584`

is mathematically correct. If the standardized CUT rule is exact for Gaussian polynomials up to degree `d`, then exactness transfers to the possibly singular affine image `x = m + Cz` without needing an inverse of `C`.

This is a clean, well-stated result.

---

## 5. High-priority wording issue in Chapter 18

### 5.1 The summary sentence on SVD derivatives should be narrowed to branch-local eigenderivatives

The derivation in:

- `docs/chapters/ch18_svd_sigma_point.tex:361-456`

assumes:

- smooth dependence on parameters,
- simple spectrum,
- positive eigenvalues,
- no active hard floor,
- a chosen branch of the eigenbasis.

Under those assumptions, the displayed formulas are fine. But the sentence around:

- `docs/chapters/ch18_svd_sigma_point.tex:434-437`

is broader than ideal. What is actually derived is not a globally well-defined SVD derivative in all cases. It is a **branch-local eigenderivative formula for an eigen-based square root**.

#### Recommendation

Change the summary wording to something like:

> Equations ... are the branch-local eigenderivative formulas for an eigen-factor covariance square root on a smooth simple-spectrum branch.

That would align the prose exactly with the assumptions.

---

## 6. Medium-priority issue in Chapter 18

### 6.1 “Affine exactness” title could be more precise

The proposition title:

- `docs/chapters/ch18_svd_sigma_point.tex:551-552`

uses “affine exactness,” but the actual statement is about **preservation of polynomial exactness under affine pullback**.

The proposition itself is sound; this is only a naming issue.

#### Recommendation

Rename it to something like:

> Affine pullback preserves CUT polynomial exactness

or:

> Polynomial exactness under affine SVD placement

---

## 7. Notes on the degenerate-covariance discussion in Chapter 18b

The long proposition on degenerate covariance and formal sigma-point accuracy:

- `prop:bf-degenerate-covariance-formal-ut-accuracy`
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:1887-1946`

is mathematically acceptable as written **if read exactly as a local Taylor-order statement**. It correctly avoids relying on positive definiteness and only uses the first two moments plus remainder control.

The follow-up prose at:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex:1947-1955`

is mostly fine, but it would be safer to emphasize that this is a **formal local statement abstracting from finite arithmetic and factor-construction effects**, not a blanket guarantee that singular covariances are problem-free in implementation.

That caveat is especially important because the rest of the chapter correctly argues that implementation pathologies can still arise from:

- nugget injection,
- support leakage,
- factor branch choices,
- spectral-gap derivative issues.

---

## 8. Numeric spot-checks performed

I recomputed the first structural UKF toy example in Chapter 18b using the chapter’s stated sigma points and weights.

The following values match the text up to rounding:

- predictive mean `\hat x_{t|t-1} = (0, 0.11024)^\top`
- predictive covariance diagonal entries `0.27560` and `0.08657`
- observation variance `S_t = 0.61217`
- cross covariance `P_{xz,t} = (0.27560, 0.08657)^\top`
- gain approximately `(0.45020, 0.14141)^\top`
- posterior mean approximately `(0.08543, 0.13707)^\top`
- log-likelihood contribution approximately `-0.70298`

I also checked the `\phi=0` variant and confirmed the displayed `k`-variance remains positive and matches the chapter numerically.

So the first example is not only conceptually coherent; its arithmetic is also consistent with the stated setup.

---

## 9. Recommended revisions before treating the chapters as mathematically polished

## Must-fix wording changes

1. In Chapter 18b, replace “is exact only if” with “targets the same predictive law only if” in `prop:bf-structural-ukf-pushforward`.
2. In Chapter 18b, relabel the UT-accuracy proposition as a **local small-uncertainty Taylor statement** rather than letting it read like a generic UT theorem.
3. In Chapter 18, narrow the SVD derivative summary language to **branch-local eigenderivatives**.

## Strongly recommended

4. State the exact sigma-point parameter choice used in the worked Chapter 18b examples.
5. Add a one-line provenance statement that the worked numbers are illustrative calculations under that convention.
6. Replace bare “wrong model” wording with “different approximate model relative to the declared structural law” wherever the sentence could otherwise be read too categorically.

## Optional but helpful

7. Rename the SVD-CUT proposition title for precision.
8. Add one sentence after the degenerate-covariance proposition clarifying that value-side formal exactness does not remove finite-arithmetic or derivative-branch issues.

---

## Final assessment

### Chapter 18b

Mathematically, Chapter 18b is in good shape on its central argument. The pushforward law, sigma-point-variable doctrine, and structural-vs-naive distinction are all fundamentally correct. The main work remaining is **tightening the accuracy claims and reducing overstatement**.

### Chapter 18

Chapter 18 is also mathematically solid in its main structure. The derivative recursions, reconstruction identities, and SVD/CUT separation are sound. The remaining issues are mostly **precision of exposition**, especially around what kind of SVD derivative object is actually being derived.

Overall judgment: **no major derivation collapse found; several important wording and claim-boundary revisions still warranted before calling the chapters mathematically fully audited and polished.**
