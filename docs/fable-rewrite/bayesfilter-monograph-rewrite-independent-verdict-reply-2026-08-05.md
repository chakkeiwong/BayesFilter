# Reply to the Independent Verdict on the Fable Monograph Rewrite

- **Date:** 2026-08-05
- **Reviewed artifact:** `docs/fable-rewrite/bayesfilter-monograph-rewrite-independent-verdict-2026-08-05.md`
- **Decision on that verdict:** **Mostly agree. Keep the rewrite as a repair branch, not as a replacement candidate yet.**

---

## 1. Bottom line

I substantially agree with the independent verdict.

The rewrite branch is **useful, materially improved, and now mechanically buildable**, but it is **not yet strong enough to replace** the canonical `docs/` monograph tree wholesale. The verdict is right to distinguish:

- **mechanical success** from
- **mathematical closure**,
- **source-support closure**, and
- **replacement readiness**.

That distinction is the important one.

So my response is:

1. **Yes**, the rewrite should be kept.
2. **No**, it should not yet replace `docs/`.
3. **Yes**, the next pass should start from the rewrite branch rather than from the old tree.

---

## 2. What I agree with strongly

## 2.1 The rewrite is a partial repair, not a completed replacement

This is correct.

The rewrite tree was intentionally a bounded pass. It repaired a real subset of the confirmed defects, restored reproducible compilation of the copied monograph tree, improved several mathematically important explanations, and removed active-root citation/label failures. But it did **not** yet complete every known blocker from the review cycle.

So the verdict is right that this is a **repair branch** or **patch source**, not a completed replacement artifact.

## 2.2 The SR-UKF blocker family remains open

I agree.

If `ch17` still leaves the negative-weight filtered-factor identity unchanged, then the rewrite cannot be called a mathematically repaired replacement. Likewise, if Chapter 17 still promises a signed update/downdate derivative primitive that the book does not actually derive, that is still a blocker.

These were among the most important confirmed defects in the review cycle, and the verdict is right to keep them at the top of the replacement-risk stack.

## 2.3 The squared-TT lane remains replacement-blocking

I agree.

The verdict is right to separate:

- the two confirmed internal inconsistencies,
- from the third under-specified interface claim.

That matches the adjudicated review boundary and is the right standard for the next pass. If the rewritten branch leaves the `e^{-c_t}` scale mismatch and the retained-coordinate ordering contradiction untouched, then the branch is not yet mathematically closed enough for replacement.

## 2.4 Provisional bibliography entries cannot count as closed scholarly support

I agree strongly.

Mechanical BibTeX success is not the same thing as resolved source support. The independent verdict is right that records carrying notes like “provisional” or “verify before publication use” cannot remain as live replacement-grade citations in the monograph.

This is especially important in the survey-heavy and design-space-heavy chapters. A replacement candidate must either:

1. verify the record and inspect the supporting technical section,
2. replace it with an already checked source, or
3. remove/downgrade the claim.

That is exactly the right standard.

## 2.5 The LEDH policy binding is still incomplete

I agree.

The rewrite improved generic wording around chunk provenance and stop-gradient interpretation, but the verdict is right that the monograph still does not bind the **actual repository-owned canonical identities** and policy names tightly enough. If the canonical route and tuning semantics matter to the monograph's current claims, the replacement branch needs to say so explicitly.

## 2.6 The particle-filter algorithm improved more than its proof object

I agree.

The rewrite fixed the no-resampling algorithm in substance. The verdict is right that the proof language still needs a cleaner weighted post-decision measure. That is a very good catch: it preserves the important improvement while correctly refusing to over-credit the proof as fully closed.

## 2.7 The ICNN section is more honest but not yet fully specified

I agree.

The rewrite fixed the reviewed logical error about fixed-`\varphi` dependence, which was the primary mathematical defect. But if the pseudocode still does not define the target-side optimization object and schedule clearly enough, then calling it canonical remains too strong.

That is the right classification: **improved, but not fully replacement-ready**.

## 2.8 Buildability is real progress, but not the same as replacement quality

I agree strongly.

This was the right way to phrase the build result:

- the rewrite branch genuinely closes the fatal missing-figure build defect,
- but replacement readiness still depends on mathematical, scholarly, and typesetting gates.

That is the right separation of evidence classes.

---

## 3. What I would narrow or restate slightly

I agree with the verdict overall, but I would restate three points a little more carefully.

## 3.1 “Not correct as a whole” should be read as “not replacement-ready as a whole”

The verdict's summary table says:

> Is the rewrite correct? **No, not as a whole.**

I understand and mostly accept the intent, but I would phrase it more carefully as:

> The rewrite is not yet **replacement-ready as a whole**.

Why I would narrow that wording:

- many unchanged chapters simply inherit the current monograph state, neither newly repaired nor newly broken;
- many repaired sections are genuinely more correct than before;
- the replacement decision is blocked by a relatively concentrated set of mathematical and source-support issues, not by universal failure.

So I agree with the *decision* behind the sentence, but I would use more targeted wording.

## 3.2 The verdict should explicitly distinguish “unchanged known blockers” from “rewrite-introduced blockers”

Most of the serious blockers it lists are **not caused by the rewrite**. They are open findings that the bounded rewrite did not yet close.

That matters for the next pass because it means the rewrite branch should be treated as:

- a **preservation branch with incomplete closure**,
- not as a branch that introduced a new wave of high-risk mathematics.

The strongest repair instruction is therefore:

> preserve the good rewrite patches and close the remaining blocker families,

not

> distrust the rewrite branch globally.

## 3.3 Typesetting warnings need thresholding, not only counting

I agree that 190 overfull boxes and 769 underfull boxes are not replacement-grade by themselves. But the next pass should use a **severity threshold**, not just counts.

In other words:

- giant 100–400 pt overflows are real blockers,
- trivial underfull lines in longtables or narrow columns are not all equally important.

So I agree with the verdict's direction, but the next pass should classify the warnings by visual severity and location rather than treat every warning equally.

---

## 4. What this means for the rewrite branch right now

The verdict implies the following document status should be adopted explicitly:

### Current status of `docs/fable-rewrite/monograph/`

- **Buildable:** yes
- **Mechanically cleaner than `docs/`:** yes
- **Contains important high-confidence repairs worth preserving:** yes
- **Safe to replace canonical `docs/` immediately:** no
- **Best starting point for the next revision pass:** yes

That is the operational conclusion I would adopt.

---

## 5. Recommended next actions, based on the verdict

I agree with the verdict's repair order, and I would preserve it with minor sharpening.

## 5.1 First: close the two blocker families that remain mathematically central

1. **SR-UKF branch correctness**
   - negative-weight filtered-factor identity
   - signed update/downdate derivative promise

2. **Squared-TT branch correctness**
   - `e^{-c_t}` scale consistency
   - retained-coordinate ordering
   - route-audited handoff specification

Without those, replacement is still blocked even if the rest of the book improves.

## 5.2 Second: close scholarly-support blockers, not just BibTeX mechanics

- remove or verify provisional live records,
- resolve the HAC theorem support,
- resolve the Li–Coates support boundary,
- add the bounded source-support and omission ledgers that the rewrite design already promised.

## 5.3 Third: bind canonical LEDH policy text explicitly

Add the frozen route identifiers, historical-route classification, exact-divisor chunk selector language, and per-scope tuning semantics in one compact canonical-policy subsection.

## 5.4 Fourth: finish local repaired-but-incomplete arguments

- particle-filter proof object
- ICNN pseudocode status
- remaining target-contract wording gaps

## 5.5 Fifth: only then do the serious reader-facing/typesetting pass

This should include:
- accent normalization,
- large-overflow cleanup,
- remaining process-register trimming,
- and a visual pass over the worst pages.

---

## 6. The key strategic point

The verdict should **not** be read as a reason to discard the rewrite branch.

It should be read as a reason to change the branch's status from:

> maybe-ready replacement candidate

to:

> high-value intermediate repair branch with known replacement blockers.

That is a much more useful conclusion.

The most important preserved gains from the rewrite are real:

- the tree builds,
- the missing-figure defect is closed in the rewrite root,
- several target-definition passages are more honest,
- the particle-filter algorithm improved,
- the LEDH offset repair landed,
- the GenUT misattribution was corrected honestly,
- the ICNN fixed-`\varphi` explanation was repaired,
- the structural-UKF table was improved,
- active-root duplicate labels and unnumbered-display label failures were repaired,
- and several of the worst reader-facing review-register leaks were removed.

The next pass should **build on those gains**, not redo them from scratch.

---

## 7. Final verdict on the independent verdict

**I agree with the independent verdict overall.**

More precisely:

- I agree that the rewrite should **not** replace `docs/` yet.
- I agree that the blocker set it identifies is materially correct.
- I agree that build success and citation resolution are necessary but not sufficient.
- I agree that the rewrite branch should now be treated as a **repair branch**.

My only narrowing is presentational:

- I would describe the rewrite as **not yet replacement-ready as a whole**, rather than simply “not correct as a whole,” because many repaired sections are in fact better and some blocker families are inherited-open rather than rewrite-created.

That wording change does not change the execution decision.

---

## 8. Recommended status line for future notes

Use this status sentence going forward:

> The standalone rewrite under `docs/fable-rewrite/monograph/` is a buildable intermediate repair branch that preserves several high-confidence fixes, but it remains blocked from canonical replacement by open SR-UKF, squared-TT, source-support, canonical-LEDH-policy, and final-typesetting issues.

That is the most accurate summary of where things stand.
