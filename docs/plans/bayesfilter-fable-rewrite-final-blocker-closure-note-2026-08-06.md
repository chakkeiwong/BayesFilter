# Final blocker-closure note for the standalone rewritten BayesFilter monograph

- **Date:** 2026-08-06
- **Purpose:** record what remains open and what is sufficiently closed after the latest narrowed rewrite pass, so the next step can be a final bounded review rather than another broad rewrite.
- **Scope:** `docs/fable-rewrite/monograph/` and the associated review/ledger notes in `docs/plans/`.

---

## 1. What is now sufficiently closed for promotion-style evaluation

These items are now good enough to be treated as **closed or honestly narrowed** for the purposes of a final bounded release review:

1. **PF likelihood / differentiability boundary**
   - `ch19` now cleanly separates the true likelihood, unbiased estimator, and smooth surrogate scalar.

2. **LEDH canonical policy binding**
   - `ch32c2` explicitly binds the current canonical route identifiers and the historical-route demotion semantics.

3. **GenUT local attribution honesty**
   - `ch32c` now distinguishes the local whitened-moment variant from the broader external-paper claim.

4. **ICNN fixed-phi explanation**
   - `ch32e` now correctly states that the target-side expectation is constant in `theta` for fixed `varphi`.

5. **HAC passage status**
   - `ch28a` now explicitly demotes the theorem-level HAC statement to an implementation/audit direction until source support is fully closed.

6. **Source-support blocker statement**
   - `app_e` now functions as an honest blocker ledger instead of pretending source closure is already done.

7. **SR-UKF release posture language**
   - `ch17` / `ch12` / `app_c` now honestly say the signed derivative route is conditional unless fully derived or externally supplied and checked.

---

## 2. What is still open and must remain open until the next release gate

These are still live blockers and should not be silently downgraded:

### 2.1 Squared-TT retained-coordinate derivation
**Files:**
- `docs/fable-rewrite/monograph/chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`
- `docs/fable-rewrite/monograph/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`

**Open issue:**
The branch is now written as retained-first with right-side contractions, but the release record still lacks a fully explicit scalar/vector derivation certificate and numerical identity check.

**Status:**
- mathematically plausible and now much more coherent,
- but still **not fully certified** for final release.

### 2.2 SR-UKF signed derivative primitive
**Files:**
- `ch17`, `ch12`, `app_c`

**Open issue:**
The monograph still does not derive the ordered signed update/downdate primitive for all negative-weight branches.

**Status:**
- honest and acceptable as a development branch,
- but still a blocker if the final release is supposed to claim full analytical-score support.

### 2.3 Source-support and bibliography closure
**Files:**
- `references.bib`
- `ch28a`
- `ch32c`
- `app_e`

**Open issue:**
There are still theorem-level source-support and literature-completeness gaps that must be either inspected, downgraded, or placed into explicit nonclaims before publication-grade promotion.

**Status:**
- the rewrite is honest about the gaps,
- but the gaps are not yet fully closed.

### 2.4 Final release-quality typesetting debt
The rewrite still carries:
- one remaining foreign-command / `amsmath` warning,
- and substantial overfull / underfull box debt.

**Status:**
- build is fine,
- typography is not yet publication-polished.

---

## 3. What to do next

The next step should be **one bounded final review**, not another broad rewrite.

That review should focus only on:

1. the squared-TT derivation certificate,
2. the SR-UKF conditional release posture,
3. the remaining source-support boundary,
4. the LEDH policy subsection coherence,
5. the final typography debt.

If that bounded review passes, the rewrite branch is ready for a curated promotion into canonical `docs/`.

---

## 4. Honest status statement

At this point, the rewrite is best described as:

> a buildable, materially improved repair branch with narrowly concentrated remaining release blockers that are now explicitly ledgered and should be closed or preserved as nonclaims before canonical promotion.

That is the honest current state.
