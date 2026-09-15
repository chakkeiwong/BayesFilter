# Younis manuscript update

Status: complete. The requested LaTeX/PDF revision was completed on 2026-09-11.

## Reader and scientific purpose

Reader: a researcher familiar with state-space models and particle filtering,
who should be able to reconstruct the implementation, historical decisions,
negative results, and proposed model-score estimators from one document.
The writing remains a technical research note with explained equations and
source anchors. Human feedback on voice remains pending; the user has
explicitly requested the complete draft now.

The live document is
`docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex`.
Restore its latest source/bibliography from research commit `804616e3` into the
existing document directory, preserving the source and surviving PDF first.
This directory is ignored in the current checkout; that does not mean the
recovered source is absent from the research branch. Preserve a final source
archive in this plan's artifact directory for durable review.

Baseline and output root:
`docs/plans/artifacts/younis-manuscript-update-20260911/`.
Ordinary SHA-256 provenance is in `baseline-manifest.json`.

## Skeptical audit and scope

Proceed with documentary revision and builds. No new filter experiment,
implementation change, promotion, HMC run, or provider change is authorized
by this manuscript task. Use the latest implementation branch explicitly;
do not imply its changes are merged into the current checkout. The historical
Section 3.6 note and old campaigns remain historical, with their known
mathematical and scope limitations. Initial-law derivatives, SIR identity,
finite-score versus model-score bias, raw versus normalized IWSG, and lack of
new proposal results must remain explicit.

## Argument and coverage

Start with the underlying model score and the failure of derivative correctness
to establish unbiasedness. Define the executed program and source method,
then explain the sidecar, integrated observation, raw-IWSG, and support work.
Use the observed losses to motivate a shift in research question. Derive
model-corrected mixture proposals, ancestor-averaged Fisher scores, and
selective mixture derivatives where each becomes necessary. Retain the
existing support mathematics and explain why it is deferred for current model
testing. Conclude with a discriminating comparison, realistic limits, and
the actual implementation boundary.

Preserve or account for every existing labeled equation, citation, technical
derivation, numerical finding, assumption, and substantive qualification.
Replace stale directions and compressed administrative prose rather than
append a conflicting new summary. Check primary paper sections and author
code for new imported claims; retain local copies and an updated source ledger.

## Validation

Build both the protected source baseline and the revised PDF. Check references,
citations, overflow, table layout, and equation continuity. Inspect rendered
pages throughout the actual PDF, with extra attention to changed results,
new derivations, figures, and source/provenance material. Run exact algebra
checks appropriate to new examples and compare retained equation/citation
inventories. Save build logs, source archives, and a concise completion note.
Build success and self-review establish neither human acceptance nor scientific
performance of the proposed algorithms.

## Completion

The updated manuscript has 34 pages. All 51 original displayed mathematical
environments, all 53 labels, and all five citation keys are preserved. Exact
reference examples and static implementation wiring checks pass. The final
PDF builds without warnings and was visually inspected. Runtime parity of the
later shared correction dependency was not rerun. The new method remains an
explicit research proposal.

See `artifacts/younis-manuscript-update-20260911/completion.md` for the
coverage/preservation review, scientific decisions, validation limitations,
source/PDF archive, and compact continuation instructions.
