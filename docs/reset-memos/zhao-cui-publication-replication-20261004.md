# Active Zhao–Cui checkpoint

Question: reproduce the published predator–prey (PP) and SIR experiments with
released author code, then assess full-quadratic numerical scores at the fixed
T=20 targets. Stage: first campaign complete; additional 48-hour extension authorized;
independent marginal-weight merge review complete. Branch: sqmc-development. Base HEAD: 0b91a64f6. Current research
changes are uncommitted; preserve the unrelated dirty 20261003 documents.

Plan: `docs/plans/zhao-cui-publication-replication-and-score-20261004.md`.
Results: `docs/benchmarks/zhao-cui-publication-replication-status-20261004.md`.
Evidence root: `docs/plans/artifacts/zhao-cui-publication-replication-20261004/`.
Final score tables/figures: `report-score-final-01/` beneath that root.

## Checked findings

All five paper-profile and three released-driver runs completed. Exact paper
number replication remains unresolved: local datasets, settings and repetition
units differ. Actual ESS values and source anchors are in the result note.
Finite author-code outputs do not establish that the published results are
wrong. The original author source remains unchanged.

For PP, the three rank-20 fits with 100,000 paths each give, at h=0.0003125,
mean physical score (-33.051124, -0.55078735, 0.020627942, 4.4686647,
-8.7462792, 10.883935). Between-fit 95% t halfwidths are (0.187941,
0.0147903, 0.000094596, 0.0283026, 0.0265643, 0.0280803). Conditional mean
path-jackknife SEs are (0.0330587, 0.00174556, 0.0000323612, 0.00586542,
0.00661567, 0.00828419). Minimum ESS is 99,892; maximum held-out RMS is
4.60e-8. The quadratic precision diagnostics are favorable; common support
and finite-particle bias remain unresolved.

For SIR, rank-40 seed 2 completed in 859.06 s. At h=0.00015625 the score is
(112.685217, -65.165591, 5.576446), with conditional path SE (12.4285,
5.4720, 0.08618), minimum ESS 4,549 and held-out RMS 7.24e-7. One fitted
proposal supplies no between-fit interval. The rank-20 small-radius run
reached its 3,600 s cap after 8/9 estimates: preserve the partial evidence
and do not aggregate it. Wide rank-20 SIR neighborhoods have ESS near 1.2
and fail the reference-overlap criterion. That failure rejects this wide
candidate, not the numerical-score research direction.

These are independent CPU reference computations, with GPUs intentionally
hidden and two BLAS threads. No analytic or autodiff score was used to fit
the quadratic. The calculations estimate derivatives of finite importance
estimates; neither an exact oracle nor LEDH improvement, filter ranking,
default readiness or production validity is established.

## Completion and budget

Final queue snapshot: 2026-10-05T11:47:00.485970+00:00,
`conditional-reference-queue-05.status.json`. Aggregate campaign use is
169,213.099 seconds (47.004 of 48 job-hours), with 0.996 hours remaining.
The original queue is complete. Its budget remains closed and separately
accounted; the extension below authorizes new work.

The monograph contains the actual likelihood and score tables. Final manual
pdflatex/bibtex build: 612 pages, no undefined references/citations or rerun
warnings. Physical pages 401–402 and both final score figures were visually
inspected. Build record: `documentation-01/build-result.json`; PDF SHA-256:
`02276df5988f52b161f84a01814279bac2b24a619efdf97afe77ade958758090`.
Human readability review remains pending. All 22 focused quadratic, path-value
and exceptional-tail tests passed; git diff whitespace checks passed.

Post-run red-team: finite-particle/support bias could explain PP movement;
proposal rank and one-fit sampling variation could explain SIR movement.
Independent rank-40 fits or larger common-path calculations that move outside
reported uncertainty would overturn the favorable interpretation. The weakest
evidence is the single SIR rank-40 fit and incomplete rank-20 small-radius run.

## New user authorization and immediate review, 2026-10-06

The user granted 48 additional hours and requested review of the other agent's
range-bearing marginal-mixture repair before a main-branch merge. The new
serious-campaign budget is 172,800 aggregate job-seconds, currently unspent;
local review tests are engineering checks, not a resumed fit campaign.
Old usage remains 169,213.099 seconds in its own ledger.

Review outcome: optional merge of commit f5e69d716 is sensible; keep the ancestor
default. Five focused tests and a two-row CPU FP64 XLA parity/repeat check pass.
Remaining GPU/FP32 precision, bootstrap comparison and cross-dataset gaps block
default/HMC promotion. Full review:
`docs/benchmarks/marginal-weight-independent-review-2026-10-06.md`.

Current steering: the user requested a proper shared-correction merge plan,
including a PP/SIR canary. Plan and skeptical self-review:
`docs/plans/ledh-marginal-weight-merge-and-canary-2026-10-06.md`; checkpoint:
`docs/reset-memos/ledh-marginal-merge-20261006.md`. The source commit contains
173 files, so the plan requires scoped dependency extraction rather than a
wholesale merge. No integration or new canary has executed.

The next reference phase remains independent SIR rank-40 proposals/scores,
source-data tie-out and PP score-shift checks within the additional 48 hours.
Marginal weighting does not change that independent reference target.
