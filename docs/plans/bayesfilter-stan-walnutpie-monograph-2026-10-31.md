# Stan and Walnutpie monograph integration

Owner request: fully document the standalone Stan/NUTS ports, Walnutpie, related
implementation and experiments in the book built from `docs/main.tex`; then
commit, merge remote changes, resolve conflicts, and push.

## Scope and sequence

1. Audit the pinned source, tuning guides, implementation tests and result
   records. Cover early NUTS/fixed-HMC comparisons, Stan adaptation/direct/TFP
   routes, native bridge versus genuine TF GPU/XLA execution, Walnutpie
   initialization/control/R2P/metric support, author funnel replication,
   placement/optimization studies, q20 and the completed full80 comparison.
2. Add a source-grounded chapter to the existing book, with equations,
   implementation boundaries, reproducible result tables and explicit evidence
   qualifications. Use a separate bibliography file to preserve unrelated
   bibliography edits. Preserve earlier results and distinguish incomplete
   or invalid experiments from supported conclusions.
3. Review prose, cross-check numeric tables against saved artifacts, build
   the book and inspect the new chapter's rendered pages. Run focused tests
   for any newly committed full80 code; no new sampling campaign is needed.
4. Commit only this work and its necessary reproducibility files in an isolated
   checkout/index. Merge the intended remote branch without force-pushing,
   resolve conflicts while preserving both contributions, rebuild/check the
   integrated documentation, and push. Leave unrelated shared-worktree edits
   intact.

## Skeptical audit and evidence contract

This is a documentation and repository-integration task, not a new sampler
comparison. Success means the book accurately distinguishes source identity,
mathematical parity, numerical implementation, posterior evidence and measured
execution cost. Published claims require inspected method/source anchors;
local numeric tables require named result artifacts. Failed convergence,
incorrect device placement, stopped replications and source reload failures
must remain visible. Acceptance alone cannot establish mixing. Short timings
and unequal historical runtime totals cannot establish sampler superiority.

The working tree contains substantial unrelated edits, including `main.tex`
and `references.bib`, and the current preservation branch has no upstream.
Do not stage all changes, stash the shared tree, or infer a remote target
without checking prior Git integration history and remote references. Use an
isolated integration checkout if necessary. No public defaults or numerical
policies are changed. No new GPU compute is planned.

Pre-execution audit passes for this bounded scope: evidence already exists,
the chapter can be built without touching inference defaults, and scoped
Git integration protects concurrent work. Exact build/test commands and the
final commit/push outcome will be recorded at completion.

## Documentation and validation checkpoint

The chapter now has three source files (implementation/mathematics, results,
and reproducibility), a dedicated bibliography and a direct input from
`docs/main.tex`. It covers all scoped implementation families, source audit,
metric conventions, initialization/control, R2P, native packaging, optimization,
author-funnel replication, hybrid placement and both financial applications.
The completed million-run comparison is reported with its eligible analytic
discrepancies; the obsolete launch-only note is not mistaken for final evidence.

Preserved twelve compact result/manifest/check records (429,779 bytes), with
original paths and SHA-256 hashes, under
`docs/reference/stan-walnutpie-evidence-20261031/`. Large draws, local source
installations and financial data are not copied. Added the full80 application
harness and its existing focused tests needed to make the completed experiment
traceable. The shared tree's unrelated changes remain excluded.

Validation in `/tmp/bayesfilter-stan-walnutpie-docs-20261031`:

- `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true
  TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q
  tests/test_full80_walnutpie.py`: **13 passed in 7.08 seconds**.
  CPU mechanics/reference tests only; GPU devices intentionally hidden.
- `python docs/reference/check_stan_walnutpie_evidence.py`: all twelve
  hashes match; full80 terminal diagnostics and all four million-run summaries
  and eligible discrepancy classifications match their saved records. This
  verifies reporting, not an independent chain or target evaluation.
- From `docs`, `latexmk -pdf -interaction=nonstopmode -halt-on-error
  -outdir=/tmp/stan-walnutpie-latex-before-merge main.tex`: successful,
  **646 pages**, zero undefined references/citations. An initial unescaped
  identifier and new-chapter layout overflows were repaired. The final new
  chapter has **zero overfull boxes**; pre-existing book warnings remain.
- Rendered and visually reviewed all nineteen new chapter pages (PDF pages
  565--583, printed 545--563), including equations, all tables and API listings.
- `git diff --check`: passes.

Post-writing skeptical review: no universal fastest-device assertion, no
equivalence inference from a failed-to-reject mean test, no runtime ranking
from unequal historical time boundaries, and no conflation of successful
source checks with posterior validity. The strongest remaining limitations
are inherited experiment design and missing fresh full80 target replay after
source drift; both are explicit in the chapter. Human prose feedback remains
pending and does not block this requested documentation/integration delivery.

## Git integration checkpoint

Isolated branch `docs/stan-walnutpie-monograph-20261031` starts at
`8821e8c0c431071578084164c6915cb95fbb3d82`. The shared preservation branch
has no upstream and substantial unrelated dirty work. Remote target is
`origin/main`; the first fetch found 355 remote commits and no local-only
commits. Commit scoped deliverables, merge the fetched remote branch, resolve
any content conflict without discarding either contribution, rebuild/check,
and push without force. Final integration outcome is recorded below.
