# LEDH, controlled SMC and KDM manuscript audit — 14 October 2026

Completed shortly after midnight on 15 October 2026, Hong Kong time. The
manuscript revision retains the date of its edits and replication evidence.

## Question, scope and verdict

Is the linked manuscript current and mathematically correct? The target is
`docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex` in the shared
`surrogate-hmc` checkout, HEAD `89065bc6354801cb368d5e163ba49fd9c4372d10`.

The original document was out of date. Confirmed status and citation gaps
are corrected. The selected mathematical identities checked are correct under
their stated assumptions; this is not certification of every derivation or
every current implementation consumer. No filtering, training, tuning, GPU
experiment, branch change or commit was performed. Unrelated work is preserved.

The same-path manuscript in the recovered main worktree is substantially older.
It was not overwritten or silently synchronized. Also, `.gitignore:584` ignores
`docs/papers/`: an empty Git diff did not establish manuscript cleanliness.
Exact content guards and a protected backup were used instead.

## Evidence contract and skeptical audit

Checked equations must compute their stated targets under their assumptions.
Status statements must match dated evidence. Compilation, source replication,
or an unbiased likelihood identity cannot establish an unbiased normalized
model score, numerical superiority, or current consumer wiring.

The initial audit passed with these restrictions: distinguish controlled SMC
from original iAPF; do not turn bounded source checks into full-paper
replication; do not call (1,5) tuned; preserve the September results as history.
The owner-selected pair is a temporary development baseline, not a safety proof
or cross-model optimum. No numerical policy was changed in this document audit.

## Corrections

- Updated the date and October replication status; marked September commits
  `804616e3` and `5cc59cfa` as historical snapshots.
- Added Heng et al., Controlled Sequential Monte Carlo, Annals of Statistics
  48(5), 2904–2929, DOI 10.1214/19-AOS1914. Metadata and the backward regression/
  policy-refinement account were checked against the cached published paper.
- Distinguished accepted bounded controlled-SMC and released 25-particle
  bearings MDPF/A-MDPF source checks from original-iAPF and full-paper replication.
  The 283 closure checks are evidence/provenance/wiring checks, not new numerical
  trials; eight runtime-file matches are provenance evidence.
- Explained the proposed LEDH use of lookahead and evaluable mixtures, with the
  actual proposal law and its exact importance corrections retained. Integration
  and an improvement in LEDH score accuracy are not established by source closure.
- Set the written development comparator to epsilon=1 and five Sinkhorn
  iterations, retaining the need for scope-specific tuning before promotion.

## Mathematical findings and limits

| Checked target | Verdict | Remaining limitation |
|---|---|---|
| Fixed-proposal IWSG | Correct gradient identity for a mixture expectation | Not generally a marginal log-likelihood score; normalization remains |
| Importance correction | Correct physical-target/actual-proposal ratio | Requires support coverage and execution of that sampling law |
| Gaussian-product twist | Normalizers and telescoping corrected path law checked | Local branch derivatives do not automatically differentiate discrete branch probabilities |
| Fisher/ancestor recursion | Correct under regular-density/common-measure assumptions | Does not apply unchanged to degenerate transitions |
| Disturbance-coordinate pair | Unnormalized likelihood/derivative induction checked | Normalized ratio bias and long-horizon variance remain |
| Centered KDM density-score control | Zero mean with stated mixture law and independent coefficient fitting | Preserves baseline bias; useful covariance is not guaranteed |

These are bounded checks, not an audit of every preserved proof. No new
statistical ranking is supported. Historical numerical tables were preserved,
not rerun. The owner's pasted (1,5) table was not traced independently to raw
results and was not added as verified experimental evidence.

## Verification and artifacts

Protected source, bibliography and PDF:
`/tmp/ledh-kdm-manuscript-audit-20261014/baseline/`.

- Original source SHA-256: `4b9f315321c0f7a1668af6b97b7f42614f4cfea364ed405dd28e1f108d969042`.
- Revised source SHA-256: `d6bb57cc8b995ff1028c72bedb4478b1940bff1adc947b76d5a4a8ccaedfc5f9`.
- All 154 matched mathematical/proof blocks are unchanged. Original inline
  and unnumbered display mathematics remain in order; 184 labels preserved,
  one section label added. Reference targets and citation keys resolve.
- Rebuilt 59-page PDF published beside the source. PDF SHA-256:
  `b800617b8d4847418405e6893618de2853191b2f381deb0b532e668dcd9d2949`.
- Final LaTeX pass: no warnings, missing references/citations, or overfull/
  underfull boxes. `latexmk` was unavailable; installed pdflatex/BibTeX worked.
- Pages 4, 22–23, 50 and 55 rendered successfully. Visual inspection remains
  incomplete because `view_image` failed with the sandbox helper's
  `mountinfo path is not absolute` error. Successful rendering is not visual
  certification. Human visual review remains outstanding.
- Direct Update File patches had the same sandbox fault. Trusted `apply_patch`
  applied exact guarded replacements using full preserved Add File payloads.

Build commands: from the manuscript directory, run
`pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/ledh-kdm-manuscript-audit-20261014/build ledh_younis_kdm_score.tex`;
run `bibtex ledh_younis_kdm_score` from that build directory with
`BIBINPUTS=/home/chakwong/BayesFilter/docs/papers/ledh_younis_kdm_score:`,
then repeat the pdflatex command twice. An initial absolute-path BibTeX attempt
was rejected by `openout_any=p`; changing its working directory repaired it.
Complete build logs and rendered pages are in the same temporary audit root.

## Decision and next action

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Accept bounded status/citation correction | Inspected evidence agrees; original math preserved; build passes | No reference or compile failure | Full implementation and visual review not completed | Read revised PDF and explicitly reconcile the older main copy before integration | Whole-document certification or improved LEDH scores |

Research continuation remains the bounded matched-input (1,5) baseline/
integration plan named in the recovered checkpoint. Existing per-model
references have unequal certainty; previous results cannot be relabelled as
fresh (1,5) evidence. No long-run budget was used or new campaign launched.

## Sources

- `/tmp/bayesfilter-sqmc-main-integration-20260924/docs/plans/controlled-smc-kdm-source-closure-result-20261014.md`.
- `/tmp/bayesfilter-sqmc-main-integration-20260924/docs/plans/ledh-controlled-smc-kdm-integration-checkpoint-20261014.md`.
- `.localresources/papers/heng2020-controlled-smc-published.pdf`, Sections 4.2–4.3,
  equations (20)–(21), Algorithms 2–3; retained author manuscript also consulted.
- `.localresources/code/heng-controlled-smc/StateSpaceModels/Lorenz/cSMC_Resample.m`: inspected Gaussian twist/weight construction, not a current consumer wiring test.
- `.localresources/papers/younis-sudderth-2023-long-range-tracking.txt`, Section 4,
  equations (14)–(15), for the fixed-proposal IWSG identity.

The integration worktree is on `main`, base
`b28c25a6c749a08d191d05bd30a1ca5dfc0def69`. External searches returned no usable
results. Local primary sources support this bounded audit; current errata and
complete literature coverage are not certified.
