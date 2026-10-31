# Structural Model B documentation closeout — 25 October 2026

## Active task and reviewed scope
Update the LaTeX with the completed fresh-data Model B confirmation, retain the
dated original results, audit numerical statements, render, commit and push main.
Starting revision: 05a4db64a; the working tree was clean. No new experiment,
scientific default or target is introduced; no GPU budget is consumed.

## Skeptical audit before editing
The committed analysis.json, state.json, selections, training result and HMC
result under docs/plans/artifacts/structural-tanh-model-b-20261025-01/ govern.
The old running plan checkpoint is superseded by these terminal records.
Two narrative risks require correction: importing the original eight flow steps
into the fresh run (which selected 16), and confusing successful finite-target
sampling with exact posterior accuracy. Separate dated configurations and
results resolve these risks. Different datasets and a single fixed LEDH design
cannot support speed or superiority claims. The fourth chain's 76.4% acceptance
exceeds the preferred 65–75% band despite pooled acceptance of 73.575%.

## Verification plan
1. Update the structural chapter and five-model cross-reference.
2. Reconcile controls, training, likelihood/score references and uncertainty,
   posterior intervals, mapped R-hat, counts and acceptance against saved JSON.
3. Retain the existing equations, historical evidence and scientific limits.
4. Build the full document and inspect the changed rendered pages.
5. Commit the scoped files, reconcile remote main if necessary, and push.

## Completed checks
- Reconciled the fresh controls, training updates, HMC counts, mapped R-hat,
  physical posterior intervals, acceptance, MCSE, ESS and charged times against
  the committed JSON records. The likelihood/score tables also retain the
  reference refinement and replication uncertainty.
- Preserved the original experiment's equations and dated results. The fresh
  confirmation is a separate subsection; the five-model chapter points to it.
- Replaced the stale running checkpoint in the confirmation plan with the
  terminal decision. Sampling success is distinguished from unresolved exact
  nonlinear posterior accuracy; no superiority or coverage claim is added.
- Built the complete 783-page book with pdflatex, BibTeX and two further
  pdflatex passes. latexmk is unavailable in this environment. Final output:
  /tmp/bayesfilter-tanh-latex-20261025/main.pdf; build and bibliography logs
  are in the same directory. No undefined or multiply-defined references remain.
- Visually inspected PDF pages 145-148 and 711 (printed pages 123-126 and 689).
  The new tables, equations, cross-references and reproduction paths are readable.
  Corrected both new line overflows; the updated structural section has no
  overfull-box warning. Existing layout warnings elsewhere in the book remain.
- git diff --check passes. This documentation-only closeout runs no new
  scientific experiment and consumes no GPU budget.

## Checkpoint
Documentation and technical review complete; commit and remote synchronization
are the remaining delivery actions. Sampling is complete; exact nonlinear
posterior accuracy remains unresolved. Human reader feedback on the revised
prose is pending; rendering and self-review are technical checks.
