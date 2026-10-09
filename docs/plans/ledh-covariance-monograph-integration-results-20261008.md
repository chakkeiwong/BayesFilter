# Covariance proposal integrated into the monograph

Date: 2026-10-08. Branch: `sqmc-development`; base commit
`209223fdd061fa39726c69550f39e2ee324f489c`.

The proposal is now Chapter 27, **UKF-Guided Particle Proposals and
Covariance-Preserving Resets**, on printed pages 245–264 (PDF pages 265–284) of
[the compiled monograph](../main.pdf). It follows the existing LEDH reset and
custom-gradient chapter. [main.tex](../main.tex) loads the
[chapter wrapper](../chapters/ch32c3_ukf_covariance_guided_particle_proposal.tex),
which loads [one shared mathematical body](../chapters/ledh_covariance_proposal_body.tex).
The [standalone source](../papers/ledh_covariance_proposal_20261008.tex) loads that
same body and still produces a [19-page note](../papers/ledh_covariance_proposal_20261008.pdf).
The monograph has 642 pages.

## Preservation and scientific status

All 49 labelled equations, the covariance diagram, three pseudocode routines,
derivations, assumptions, and limitations are retained. An executable check
reversed the namespace, citation, and five formatting substitutions and recovered
the protected original body byte for byte. Its SHA-256 is
`5cc5d65c28d62d8e936ca986125b3b3bcc930443f09318fb11b1ddfa5a20c488`.
The protected original standalone source SHA-256 is
`c57a6e7bb30d714b9cd2388e01efe22b6ccc519083f5f28146c7a8d722c3eeb5`.
The substitutions consist of three shorter section running headings, permissible
line breaks in the pairwise feature list, and a two-line layout of the component
density derivative. No mathematical term was removed or changed.

The wrapper connects the proposal to the preceding chapter, explains its column
particle convention, and identifies the combined filter as a candidate.
The earlier [MathDevMCP audit](ledh-covariance-latex-mathdev-audit-results-20261008.md)
remains the mathematical audit record; it was not rerun for source integration.
Its limits remain: nine bounded symbolic identities were proved, two inputs were
not encodable, and the machine checks did not certify every proof or the complete
algorithm. No filter runtime, numerical default, empirical result, or HMC status
has changed.

## Build and reference checks

Both documents were built with pdflatex and BibTeX. The monograph used five
pdflatex passes: two failed passes identified local macro declarations, followed
by three successful passes with BibTeX. The standalone used six successful
passes across integration and layout changes. The final logs have stable
cross-references, no undefined citations or labels, and no duplicate labels.
All 52 new labels (49 equations, figure, score section, and chapter) occur exactly
once in the monograph auxiliary file. All seven proposal bibliography keys
resolve. Recorder files confirm both actual compiler input chains reach the
shared source. The exact commands and elapsed times are preserved in the build
records; no scientific framework or GPU was used. Compiler-pass limits were
respected. Total elapsed time through PDF delivery was 128.7 minutes, exceeding
the initial 45-minute estimate while completing routine integration repairs and
verification; this was not a scientific compute campaign.

The standalone is warning-free. The new chapter has no overflowing boxes or
LaTeX warnings; two vertical-spacing and one horizontal-spacing notices remain.
Complete chapter contact sheets and full-size diagram/algorithm pages were
inspected. The first algorithm spans pages; the two shorter algorithms remain
intact. Human readability acceptance remains pending.

The full monograph retains three pre-existing hyperref PDF-string warnings and
layout warnings in other chapters. Its final log contains 215 overfull hboxes,
compared with 213 in the protected prior log; the two additional warnings are in
unchanged `ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`
(lines 1256–1261 and 1368–1379), outside the new chapter. The protected prior
PDF/log is a saved build, not a fresh counterfactual build. These warnings were
recorded, not hidden or treated as mathematical failures.

## Bibliography corrections

Shared citations use the central BibTeX database. Primary records corrected
existing entries for Acevedo/de Wiljes/Reich (Walter Acevedo, *SIAM Journal on
Scientific Computing* 39(5), A1834–A1850) and Spantini/Baptista/Marzouk (*SIAM
Review* 64(4), 921–953). Evidence is preserved from the publisher DOI records:
[10.1137/16M1095184](https://doi.org/10.1137/16M1095184) and
[10.1137/20M1312204](https://doi.org/10.1137/20M1312204).
The added Popov/Subrahmanya/Sandu entry uses the published
[stochastic covariance shrinkage title](https://npg.copernicus.org/articles/29/241/2022/).
Entries were also added for Csuzdi/Toro/Becsi and the unscented particle filter;
existing Corenflos and generalized-unscented-transform entries were reused.

## Evidence and decision

Evidence root: `artifacts/ledh-covariance-monograph-integration-20261008-01/`.
`content-preservation.json` contains reversible substitutions and hashes;
`verification.json` records the input-chain, label, citation, warning, and visual
checks. `monograph-build/` and `standalone-build/` retain compiler, BibTeX,
auxiliary, recorder, PDF, and command records; `baseline/` preserves the previous
sources and PDFs. `manifest.json` records final source and PDF hashes.

| Decision | Primary criterion | Veto checks | Main limitation | Next action |
|---|---|---|---|---|
| Deliver integrated Chapter 27 | Full audited text included; both documents compile | No missing math, broken new references, duplicate labels, or unreadable new material | Machine proof coverage and human readership acceptance remain limited as recorded | Read the chapter; any implementation requires its own bounded plan and cross-model validation |

No empirical improvement, filter correctness certificate, or runtime promotion
is inferred from this documentation integration.
