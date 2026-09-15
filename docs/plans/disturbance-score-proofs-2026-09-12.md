# Disturbance-coordinate score: derivation and MathDevMCP audit

Date: 2026-09-12. Stage: documents delivered; full MathDevMCP acceptance unresolved.

Question: can a disturbance-coordinate particle construction estimate the
original observed-data likelihood derivative without an ambient transition
density, and can a precisely centered control variate reduce its variance?
The user requests full LaTeX derivations, propositions/proofs, a MathDevMCP
audit, and a readable compiled document. This authorizes the document and
local mathematical checks; no algorithm implementation or sampling campaign
is part of this task.

Reader: a researcher familiar with likelihoods, conditional expectation, and
particle filtering. The added argument must teach the measure change, total
path derivative, sequential importance weights, unbiased unnormalized
derivative, normalization bias, and valid variance reduction in that order.
The existing manuscript and equations are protected by a copied baseline.
New formal results are project derivations, not claims of methodological novelty.

Skeptical audit: the preceding decision note incorrectly blurs direct
posterior-score estimation with differentiation of a random filtering
program. Do not add a second resampling likelihood-ratio term to an already
valid genealogical score estimator. Do not require fixed-ancestry finite
differences to match an expectation derivative. State fixed disturbance
support, domination, positivity, integrability, initial-law dependence, and
smooth-regime restrictions. Preserve the distinction between exact endpoint
Metropolis correctness and exact-force HMC efficiency. Conditional control
variates must be centered under the actual sampling law; future random
reweighting does not preserve a past zero-mean property automatically.

Evidence contract: the comparator is an independently derived exact
likelihood/score on a moving-line Gaussian model and an enumerated tiny PF.
Pass means each proposition has explicit assumptions and a complete proof,
all new proof/equation labels are submitted to MathDevMCP, substantive audit
findings are repaired, the PDF builds, and rendered additions are inspected.
MathDevMCP diagnostic success is not a Lean proof or empirical validation.
Missing mathematical assumptions and false equalities veto their claims;
tool unsupported/not-proved statuses must be reported honestly, not edited
away. Timing, appearance, and toy results cannot promote production behavior.

Budget: local text work, exact/symbolic CPU checks, PDF builds and at most four
full audit/revision rounds. No framework import, GPU, MCMC, or model/default
change. Sources: inspect local primary papers and try official metadata;
record service failures. No external agent delegation is needed.

Output root: docs/plans/artifacts/disturbance-score-proofs-20260912-01/.
Live manuscript: docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex.
Preserve the full baseline, commands, checksums, source-support notes, raw
MathDevMCP responses, compact findings, and final source/PDF. Other dirty files
are outside scope. Human readability remains for the user's review.

Checked findings: the manuscript now contains the full particle induction,
analytical tangent/score recursion, exact reference control, conditional
integration proof, and moving-line Gaussian calculation. The exact Fraction
enumeration and independent SymPy derivation pass. MathDevMCP's equation
audit covered all 30 new equation labels but returned formalization/parser
abstentions. A separate source-bound recursion audit reports
accepted_exact_source, zero mismatches, and
unverified:manual_formalization_required: the indexed stochastic notation
is outside its bounded algebraic backend. This is not a full audit pass.
The derivation-tree tool itself has document promotion disabled and omitted
14 multi-relation labels; preserve that coverage limitation.

Final outcome: the integrated manuscript is 43 pages and the extracted
proposal is 9 pages, with nine propositions and proofs, one assumption, and
one corollary. All added pages have been inspected. Round 4 binds the final
TeX and still reports partial coverage: three stochastic identities need
manual formalization. Five direct algebra checks are equivalent; two earlier
parser encodings were inconclusive and remain preserved. No complete
machine-proof certificate was produced. The DOI metadata correction concerns
the bibliography only. Full findings and document links are in
`docs/plans/artifacts/disturbance-score-proofs-20260912-01/result.md`.

Remaining audit budget: zero of four full rounds; no numerical campaign ran.
Exact next action for certification is formalization of the importance-pair,
genealogical recursion, and control-variate obligations with source-bound
backend support. Repeating the same parser audit cannot discharge them. The
derivations and compiled documents are delivered without a full-audit-pass
claim; no production or default admission follows from this document task.
