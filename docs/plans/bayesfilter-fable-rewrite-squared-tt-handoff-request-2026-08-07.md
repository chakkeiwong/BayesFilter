# Handoff note: close the remaining squared-TT release blocker

- **Date:** 2026-08-07
- **Purpose:** request a future agent to finish the last open blocker in the standalone rewritten monograph.
- **Target artifact:** `docs/fable-rewrite/monograph/`
- **Scope:** only the squared-TT retained-coordinate release blocker. Do not broaden this into another general rewrite.
- **Resolution addendum, 2026-08-08:** completed under the amended plan in `bayesfilter-fable-rewrite-squared-tt-block-certificate-execution-plan-2026-08-08.md`; see the result note and documentation-agent amendment handoff. The scalar-only success criterion was strengthened because it could not detect the vector retained-prefix defect.

## Situation

The current rewrite branch now has an explicit retained-first / right-contraction story for the squared-TT lane, but the release ledger still says the lane is not fully certified. The missing item is a **scalar/vector derivation certificate** that ties the displayed retained-numerator contraction to a concrete numerical example and the corresponding derivative.

The authoritative notes to read first are:
- `docs/plans/bayesfilter-fable-rewrite-persistent-audit-ledger-2026-08-06.md`
- `docs/plans/bayesfilter-fable-rewrite-final-crosswalk-and-blocker-resolution-2026-08-07.md`
- `docs/plans/bayesfilter-fable-rewrite-squared-tt-derivation-note-2026-08-06.md`
- `docs/plans/bayesfilter-fable-rewrite-squared-tt-final-certificate-note-2026-08-07.md`
- the rewrite files:
  - `docs/fable-rewrite/monograph/chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`
  - `docs/fable-rewrite/monograph/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`

## Task request

Please complete the remaining squared-TT release blocker by producing a **fully explicit derivation certificate** for the retained-first branch.

### Required deliverable
Add a short, reader-facing derivation note and any required supporting edits so that the monograph explicitly shows:

1. the scalar two-coordinate retained-first case,
2. the right-side contraction that leaves the retained block explicit,
3. the corresponding retained numerator formula,
4. the corresponding derivative formula,
5. and a small deterministic numerical identity check (or a clearly marked equivalent certificate) that matches the derivation.

### The required mathematical claim
The note must show that, on the concrete retained-first branch:
- the displayed right contractions produce the same retained numerator as direct integration of the branch, and
- the same recursion produces the same first derivative as direct differentiation of that branch.

### Also required
The note must keep the interface/ownership story in `ch37` consistent:
- retained-block reference measure,
- physical-to-reference conversion ownership,
- Jacobian ownership,
- next-step query convention,
- stored evaluator convention.

## Constraints
- Do **not** rewrite unrelated chapters.
- Do **not** change the global monograph structure.
- Do **not** remove the explicit nonclaim if the certificate cannot be completed.
- If the certificate cannot be made exact, the correct fallback is to keep the lane explicitly as a branch contract and state the remaining gap honestly.

## Success criteria
The blocker is closed only if the final text contains:
- an explicit retained-first derivation certificate,
- a deterministic check or equivalent verifiable note,
- internally consistent coordinate/order notation across `ch36b` and `ch37`,
- and no hidden left/right contraction mismatch.

## Final note
If the exact certificate cannot be completed from the existing rewrite structure, please leave the branch honest, mark the gap clearly, and report the smallest remaining change needed rather than pretending closure.
