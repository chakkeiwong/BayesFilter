# Independent Verdict on the Fable Monograph Rewrite

- **Date:** 2026-08-05
- **Audience:** writing agent responsible for the next monograph pass
- **Compared artifacts:**
  - current monograph root: `docs/main.tex` and its input closure under `docs/`
  - rewrite root: `docs/fable-rewrite/monograph/main.tex` and its local input closure
- **Review mode:** read-only source comparison, targeted mathematical audit, local primary-source check where available, active-root citation/label audit, and clean-room LaTeX rebuild
- **Decision:** **REVISE. Do not replace `docs/` with the rewrite tree yet.**

## Executive verdict

The rewrite is a useful and materially better **partial repair**, but it is not a correct or complete replacement candidate.

It improves several important passages, restores reproducible compilation, removes active-root label/citation failures, and is more honest about approximate targets and source limitations. Those changes should be preserved. However, it leaves known confirmed mathematical defects unchanged in the SR-UKF and squared-TT lanes, leaves the signed factor-derivative promise unfulfilled, uses provisional bibliography records as live citations, does not encode current canonical LEDH policy identities, and still contains a few repaired-but-internally-incomplete arguments.

The answers to the owner's three questions are therefore:

| Question | Verdict |
|---|---|
| Is the rewrite correct? | **No, not as a whole.** Several repaired sections are correct or substantially more accurate, but known mathematical and source-support defects remain. |
| Is it better? | **Yes, locally and mechanically.** It is buildable, clearer about several targets, and fixes a meaningful subset of the prior findings. It is not yet demonstrably better as a complete monograph because most of the book is unchanged and the structural rewrite was not performed. |
| Should it replace the current version now? | **No.** Treat it as the patch source for the next revision, not as the new canonical tree. |

## Artifact boundary

The two `main.tex` files and the two `preamble.tex` files are byte-identical. The rewrite is in the copied chapters, appendices, bibliography, and bundled figure/artifact subtree.

Of the 62 chapter/appendix inputs after `preamble`, 25 differ and 37 are byte-identical. `references.bib` also differs. Thus this is a surgical repair pass over the current monograph, not a wholesale rewritten book. Any strength or defect in an unchanged file carries over unchanged.

This is consistent with `docs/fable-rewrite/execution-audit-2026-08-04.md`: that note explicitly chose a bounded pass, excluded complete signed-downdate calculus, full literature snowballing, and unresolved mathematical design decisions, and prioritized a successful PDF. That bounded scope was reasonable for producing an intermediate artifact. It is not sufficient for replacement.

## Findings

Findings are ordered by replacement risk.

### 1. Blocker: the negative-weight SR-UKF filtered-factor identity remains wrong

**Classification:** wrong relative to the chapter's stated covariance target

**Rewrite anchors:**

- `docs/fable-rewrite/monograph/chapters/ch17_square_root_sigma_point.tex:280`
- `docs/fable-rewrite/monograph/chapters/ch17_square_root_sigma_point.tex:286`
- `docs/fable-rewrite/monograph/chapters/ch17_square_root_sigma_point.tex:297`

The chapter first defines a signed covariance factor correctly as

```text
A_{u,*} = E_+ E_+^T - E_- E_-^T.
```

But the filtered-factor construction later calls

```text
SR_Bt(C_xx,+, C_KS)
```

and describes `C_xx,+` as the positive state-deviation stack, while the only stated downdate is the `K S K^T` contribution. When a covariance weight is negative, the negative state-deviation stack `E_-` is also part of `P_{xx,*}`. Omitting it makes the displayed factor compute a different covariance from

```text
P_{t|t,*} = P_{xx,*} - K S_* K^T.
```

The rewrite copy of `ch17` is byte-identical to the current chapter. This was the first confirmed mathematical finding in the prior audit and was not repaired.

**Required repair:** define the filtered signed factor from the complete positive and negative stacks. The negative stack must include both negative covariance-weight deviations and the update downdate contribution, with exact dimensions and ordering. Then prove or directly verify the reconstruction identity on positive- and negative-central-weight fixtures.

**Acceptance evidence:** a short derivation in the book's notation plus a deterministic numerical reconstruction fixture for at least one negative covariance-weight rule. A prose statement that the implementation uses a “declared branch” is not enough.

### 2. Blocker: the signed update/downdate derivative exposition is still falsely promised

**Classification:** unsupported and internally contradictory exposition

**Rewrite anchors:**

- `docs/fable-rewrite/monograph/chapters/ch17_square_root_sigma_point.tex:185`
- `docs/fable-rewrite/monograph/chapters/ch12_factor_derivatives.tex:110`
- `docs/fable-rewrite/monograph/appendices/app_c_factor_derivative_proofs.tex:4`

Chapter 17 says that Chapter 12 supplies the QR/Cholesky primitive calculus needed for the fixed update/downdate branch. Chapter 12 contains ordinary Cholesky and QR derivatives, but no derivative of the ordered signed rank-one update/downdate recurrence claimed by Chapter 17. Appendix C explicitly calls itself a placeholder for the missing signed update/downdate proof.

The rewrite made the placeholder more candid, which is better, but did not narrow the claim in Chapter 17. A placeholder cannot support an admitted multi-step analytical-score route.

**Required repair:** choose one of two honest routes.

1. Narrow Chapter 17 now: state that signed branch derivatives are not derived in the monograph and that the analytical-score route is conditional/incomplete on those branches.
2. Supply the actual ordered recurrence derivative, including the factor gauge, strict-positive-definite domain, downdate feasibility condition, branch boundaries, and equivalence between the recurrence output and the differentiated covariance factor.

Do not replace the missing recurrence derivative with differentiation of an abstract final Cholesky factor unless equivalence to the executed recurrence is proved.

### 3. Blocker: two confirmed squared-TT inconsistencies and one interface gap are unchanged

**Classification:** two wrong internal identities plus one under-specified interface

**Rewrite anchors:**

- `docs/fable-rewrite/monograph/chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:171`
- `docs/fable-rewrite/monograph/chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:192`
- `docs/fable-rewrite/monograph/chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:362`
- `docs/fable-rewrite/monograph/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex:323`

Both squared-TT chapters are byte-identical to the current monograph.

First, `ch36b` applies `e^{-c_t}` to the squared-TT contraction but not to the defensive term:

```text
a_t = e^{-c_t} [TT contraction] + tau_t integral(lambda_t).
```

Elsewhere the declared represented density convention is

```text
q_bar_t = e^{-c_t} (phi_t^2 + tau_t lambda_t),
```

so the defensive term also inherits `e^{-c_t}`. The displayed retained numerator is wrong relative to that stated target unless the text explicitly redefines `lambda_t` to absorb the factor, which it does not.

Second, the derivation assumes that the retained current state is the last coordinate (`z_D=z_t`), but the concrete branch orders coordinates as `(x_t,x_{t-1})`, making the current state the first block. Those conventions cannot both instantiate the same contraction without an explicit permutation.

Third, `ch37` defines a reference-coordinate retained evaluator and says that the next query is the previous-state block of the next fitting point. It does not assign ownership of physical-to-reference conversion, the density measure, or the Jacobian at that interface. The prior audit correctly classified this as under-specified, not as a proved missing-Jacobian algebra error.

**Required repair:**

1. Put the defensive term on the same declared scale and update its derivative consistently.
2. Freeze one coordinate convention, or introduce and propagate an explicit permutation through the TT cores and contractions.
3. Add a route/interface table stating map direction, query coordinate system, stored density measure, and Jacobian owner. Inspect the actual fixed-branch implementation before choosing the convention.

### 4. Blocker: live citations rely on provisional or unverified bibliography records

**Classification:** unsupported scholarly support; mechanically resolved citations are not verified citations

**Rewrite anchors:**

- `docs/fable-rewrite/monograph/references.bib:1204`
- `docs/fable-rewrite/monograph/references.bib:1212`
- `docs/fable-rewrite/monograph/references.bib:1220`
- `docs/fable-rewrite/monograph/references.bib:1228`
- `docs/fable-rewrite/monograph/references.bib:1236`
- `docs/fable-rewrite/monograph/references.bib:1244`
- `docs/fable-rewrite/monograph/references.bib:1260`
- `docs/fable-rewrite/monograph/chapters/ch32c_entropic_ot_sinkhorn.tex:905`
- `docs/fable-rewrite/monograph/appendices/app_e_researchassistant_workflows.tex:52`

The rewrite adds records whose notes say “provisional” or “verify before publication use.” Some lack author lists, DOI/arXiv identifiers, volume/pages, or exact source identity. Eight such keys are actively cited in the Contract E design-space paragraph. The chapter commendably admits that these sources were not all inspected, but inserting unresolved records to make BibTeX pass does not close the source-support finding.

The same problem remains in the HAC passage in `ch28a`: the book states a consistency result at `ch28a:475` while its own appendix still lists exact-source support for that theorem as a blocker. The rewrite also still records the Li-Coates durable-local-copy and bibliographic-resolution gap.

**Required repair:** for each live provisional citation, choose exactly one outcome:

1. verify the primary record and inspect the cited technical section;
2. replace it with an already inspected primary source that supports the narrower claim; or
3. remove the citation and downgrade/remove the claim.

Do not retain bibliography notes saying “verify before publication use” in a replacement monograph. Create the bounded source-support, claim-support, metadata, backward-snowball, forward-snowball, and omission-risk ledgers promised by the rewrite master document for the survey-heavy clusters.

### 5. Blocker: current canonical LEDH identities and tuning scope are not bound in the text

**Classification:** incomplete relative to current repository policy

**Rewrite anchors:**

- `docs/fable-rewrite/monograph/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex:672`
- `docs/fable-rewrite/monograph/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex:683`
- `docs/fable-rewrite/monograph/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex:2287`

The rewrite improves the generic wording: recorded chunk fields are no longer described as caller-free knobs, and stopped paths are correctly classified as partial derivatives unless the omitted chain contribution is zero.

However, the monograph contains none of the binding identifiers required by current policy:

- `contract_e_chol_v1`
- `contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1`
- `dpf_transport_exact_divisor_cap3000_v1`
- `historical_raw_barycentric_diagnostic_only`

It does not state the exact-divisor selector rule or its required examples, does not state the repository-issued per-scope tuning-artifact requirement, and still calls a measured `N=1024` witness the “repository `K=N` chunk policy.” `K=N` is true for that witness, but it is not the repository policy for all particle counts.

**Required repair:** add one concise canonical-policy subsection that binds route identity, total-gradient composition, raw-route historical status, exact chunk selection/validation, and per-scope tuning. Separate permanent policy from historical benchmark prose.

### 6. Major: the repaired particle-filter estimator has an inconsistent proof object

**Classification:** algorithm repaired; proof notation/conditional-expectation step still wrong as written

**Rewrite anchors:**

- `docs/fable-rewrite/monograph/chapters/ch19_particle_filters.tex:391`
- `docs/fable-rewrite/monograph/chapters/ch19_particle_filters.tex:488`
- `docs/fable-rewrite/monograph/chapters/ch19_particle_filters.tex:498`
- `docs/fable-rewrite/monograph/chapters/ch19_particle_filters.tex:504`

The algorithmic fix is correct in substance: on a no-resampling step it carries normalized old weights and uses

```text
tilde w_t^i = N w_{t-1}^{star,i} g(y_t | x_t^i),
```

so the incremental normalizer is the weighted predictive average.

The proof then calls `bar pi` the empirical measure of the particles from which propagation is drawn and claims that, without resampling, this unweighted empirical measure equals the weighted filter “by construction.” That is false unless all weights are equal. The later displayed conditional expectation is correct only if `bar pi` is redefined as the weighted post-decision measure or if the weighted sum is carried explicitly through the transition expectation.

**Required repair:** define separate resampled-unweighted and nonresampled-weighted post-decision measures, or use one weighted measure with post-decision weights throughout. Then rerun the induction without replacing a weighted measure by an unweighted one.

### 7. Major: the ICNN explanation is improved, but the pseudocode is not yet a defined trainer

**Classification:** original logical error fixed; executable specification remains under-specified

**Rewrite anchors:**

- `docs/fable-rewrite/monograph/chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex:287`
- `docs/fable-rewrite/monograph/chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex:309`
- `docs/fable-rewrite/monograph/chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex:348`

The rewrite now says plainly that `E_nu[varphi(Y)]` is constant in `theta` for fixed `varphi`; this correctly repairs the reviewed defect.

But the text calls the displayed form a “concrete canonical training objective” while also calling it schematic, and the algorithm tells the reader to update the target-side potential/conjugate estimate without declaring its parameterization, objective direction, optimizer, schedule, or return object. The `Require` and `Ensure` contracts mention only the source potential family.

**Required repair:** either define the joint/alternating objective and both optimization variables completely, with an exact source anchor, or demote the display and algorithm to a schematic design pattern and remove “canonical.”

### 8. Major: compilation is reproducible, but the PDF is not typeset to replacement quality

**Classification:** build gate passed; publication-quality typesetting gate not passed

A clean-room copy was cleaned with `latexmk -C` and rebuilt from source. The rebuild succeeded and produced a 491-page PDF. The final log had:

- zero undefined citations;
- zero undefined references;
- zero active-root multiply-defined labels;
- no extracted `[?]` markers;
- 190 overfull boxes;
- 769 underfull boxes;
- two `amsmath` foreign-command warnings.

For comparison, an out-of-tree build of the current `docs/main.tex` failed at
`ch28a:1002` because the referenced
`ssl-lstm-launch-traces-z.pdf` does not exist. Thus the rewrite genuinely closes
the current tree's fatal missing-figure build defect by using the available PNG
assets; the remaining objections are correctness, scholarship, and typesetting
objections rather than denial of that build improvement.

The largest overfull box is about 435 pt. Several 100--400 pt overflows are caused by long source paths, running headings, tables, and wide displays. These are not minor cosmetic warnings.

The bibliography and prose also use literal quote forms such as `T"odter`,
`B"urkner`, and `Nystr"om`; the clean rebuild confirms that at least the
bibliography form prints with a quote rather than an umlaut. Normalize these to
proper LaTeX accent forms such as `T{\"o}dter`, `B{\"u}rkner`, and
`Nystr{\"o}m`.

Also replace the old TeX `\over` expressions around `ch32c:1896` with `\frac` and remove the `\atopwithdelims` warning around `ch32c:2683`.

### 9. Moderate: the reader-facing structural rewrite remains largely undone

**Classification:** better navigation in places; no book-wide structural resolution

The new eleven-part reader map is accurate and should be retained. A few drafting phrases and duplicate sections were removed. But the main driver and chapter order are unchanged, and the size imbalance remains severe: multi-thousand-line research-record chapters sit beside chapters of only a few dozen lines. Reader-facing process language also remains, including phase, panel, artifact, ledger, promotion, and campaign language in expository chapters.

Not every use of “gate,” “artifact,” or “ledger” is inappropriate; many name legitimate evidence roles. The problem is book-wide register drift: historical project administration and benchmark narration often displace timeless explanation. Do a selective editorial pass, not a blind word replacement.

## Confirmed improvements to preserve

The next pass should start from these rewrite changes rather than redoing them from the current monograph:

1. The five SSL-LSTM figures use available PNGs, and a clean-room build now succeeds.
2. The reader map matches the actual eleven-part structure.
3. Finite invalid returns now distinguish exact support rejection from a declared modified target.
4. The HMC discussion correctly allows a deterministic approximate force under exact endpoint Metropolis correction when the proposal map remains reversible and volume preserving.
5. The particle-filter algorithm carries weights correctly on no-resampling steps, subject to the proof repair above.
6. The chapter-derived LEDH offset now includes the centered information term and reconciles it with the later source form.
7. GenUT is no longer falsely described: the chapter identifies its whitened-moment construction as a local GenUT-motivated variant, not the paper's physical-coordinate multivariate target.
8. The ICNN fixed-`varphi` explanation is now mathematically honest.
9. The structural-UKF comparison row is corrected.
10. Active-root duplicate labels and the mis-anchored unnumbered-display labels are repaired.
11. The duplicated entropic-OT subsection is removed.
12. The sparse-grid chapter gains useful explicit one- and three-dimensional calculations.
13. Li and Hu author names and the NeuTra record are improved.
14. The stop-gradient paragraph now distinguishes a partial derivative from the total derivative.

## What should be changed first

Use this order. Do not start with prose polish.

1. **Repair or narrow SR-UKF claims.** Fix the negative-weight filtered factor and either derive the signed recurrence derivative or remove the false promise/admission language.
2. **Repair the squared-TT scalar and interface.** Fix the missing shift factor, choose one retained-coordinate ordering, and freeze the physical/reference/Jacobian ownership after inspecting the actual route.
3. **Close source-support blockers.** Verify or remove every provisional live citation; resolve the HAC theorem and Li-Coates source gaps; preserve exact inspected anchors.
4. **Align the LEDH chapters with current canonical policy.** Add the frozen identities, total-gradient composition, exact-divisor chunk rule, historical-route classification, and per-scope tuning-artifact rule.
5. **Finish local mathematical cleanup.** Repair the particle-filter proof object and make the ICNN pseudocode either complete or explicitly schematic.
6. **Run the reader-facing and typesetting pass.** Remove genuine process-register leakage, shorten/path-wrap the worst material, fix accent syntax and old TeX fractions, and inspect every overflow above a declared threshold.
7. **Only then consider replacement.** Rebuild in a fresh tree and conduct one bounded independent review of the repaired blocker set.

## Replacement gate

Do not replace `docs/` until all of the following are true:

- the SR-UKF and squared-TT blocker rows have derivation or source evidence and are closed;
- no live bibliography record says provisional, metadata-to-verify, or verify-before-publication;
- the HAC and other theorem-level claims have inspected primary anchors or are downgraded;
- the canonical LEDH policy identifiers and per-scope tuning semantics are present and internally consistent;
- the particle-filter proof uses a correct weighted post-decision measure;
- the ICNN algorithm is either fully specified or explicitly non-canonical/schematic;
- a clean rebuild has zero undefined citations, references, and active-root duplicate labels;
- the two `amsmath` warnings are gone and severe page overflows have been repaired and visually checked;
- a finding-to-evidence matrix records every retained review finding as passed, narrowed with an explicit nonclaim, or still blocked;
- one final independent mathematical/source review returns no material blocker.

When that gate passes, merge the reviewed diffs into the canonical `docs/` tree rather than blindly replacing the directory. The rewrite tree bundles copied research artifacts and generated LaTeX files that should not automatically become canonical monograph source.

## Scholarship audit boundary

- **decision:** not scholarly-replacement-ready
- **metadata date:** 2026-08-05
- **seed papers directly checked in this review:** local Ebeigbe et al. GenUT full text; other literature conclusions use the repository's prior review record and the rewrite's own explicit source-gap statements
- **source-support summary:** the GenUT correction is consistent with the inspected primary technical text; several new transport/ensemble citations remain provisional or uninspected
- **citation/venue summary:** citation resolution was checked mechanically; live citation metadata was not treated as truth evidence
- **backward snowball:** not completed book-wide; required for the survey-heavy repair clusters
- **forward snowball:** not completed book-wide; required where metadata access is available
- **quarantined sources:** none established; provisional/unverified records are ineligible as claim support until checked
- **top omission risks:** second-order ensemble transforms, conditional/learned transport-map neighbors, exact HAC consistency support, and durable Li-Coates source support
- **claim-support gaps:** SR-UKF signed recurrence, HAC consistency, provisional Contract E ingredient citations, and the exact ICNN trainer
- **next required actions:** follow the numbered repair order above
- **what is not concluded:** this audit does not certify every unchanged equation or citation, does not establish book-wide literature completeness, and does not claim that the current canonical monograph is preferable to keeping the rewrite patches; it concludes only that immediate wholesale replacement is unsupported

## Final instruction to the writing agent

Preserve the rewrite's high-confidence patches, but change the document status from “replacement candidate” to “repair branch.” Close the two known mathematical blocker families first. A clean PDF and resolved BibTeX keys are necessary evidence, not correctness evidence.
