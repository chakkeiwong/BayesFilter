# Remaining replication gaps after the completed R campaign

Subsequent execution, 2026-09-21: the proposed Eq15, prefix-concentration and
timing actions have now run. See the [new result](../iapf-r-equation15-resolution-20260921-01/result.md).
The audit below preserves the earlier status; the new report closes the
diagnostic/timer tasks and records that the fitting-replication gap remains.

Checked 2026-09-21 against the completed study, current R consumers and the
locally archived arXiv v2 technical text. This is a status audit, not a new
experiment or a change to the frozen campaign criteria.

The first dimension experiment's coverage gap is closed: T=100, d=5,10,20,40,80,
1,000 independent filter replications on one fixed simulated dataset per
dimension, with BPF10,000, fully adapted APF5,000 and exact Kalman likelihoods.
All 25,000 method records are present; recorded fit and tail checks pass.
This reproduces the published experimental design using explicit reconstruction
choices. It does not reproduce the authors' exact numerical procedure.

## Open gaps and the evidence needed

| Gap | Checked finding | Closure evidence |
|---|---|---|
| Fitting objective | Both full-study variants execute diagonal log-quadratic QR fitting, not Eq.15's density least squares with free scale. Earlier constrained Eq.15 diagnostics do not constitute a successful five-dimension replication. | A fully specified implementation of the paper's fit, checked from the actual filter endpoint; original solver settings if recoverable, otherwise explicit sensitivity analysis and a qualified reconstruction claim. |
| Unspecified numerical choices | Available primary text specifies a positive constant added to the Gaussian but not its formula, and does not identify optimizer, starts, bounds or tolerances. Our floor uses a chi-square tail rule with power8; these are local choices. | Recover original settings or freeze justified alternatives before untouched validation. The mathematical ambiguity of the unrestricted density objective must be addressed explicitly. |
| Controller | Paper Algorithm4 uses indices l-k through l, hence six estimates for k=5. The qr arm preserves this window; short_qr deliberately uses five. Early doubling references undefined negative indices in the printed rule; our delayed convention is a reconstruction. | Resolve initial doubling semantics from author code/settings where possible. Keep the six-estimate paper arm separate from the five-estimate extension. A closer table entry cannot establish author identity. |
| Quantitative results and data | The authors' realized observations/RNG are unavailable. Our SD, resampling and particle patterns differ from the tables, even where our practical criteria pass. | Assess the specified method on independent datasets with uncertainty. Original observations are needed to reproduce the exact fixed-data experiment; identical RNG seeds are unnecessary for a statistical replication. Avoid tuning to published table entries. |
| Computational efficiency | Current wall_seconds times the full iAPF plus fit-table assembly/writes, tail diagnostics and saveRDS. Comparator branches omit that additional work. Two simultaneous workers also introduce contention. | Time equivalent algorithm boundaries, including all learning and the fresh final filter, while reporting diagnostic/I/O time separately. Check the paper's reported runtime relationship and, separately, a controlled equal-budget comparison. Current timings cannot establish algorithmic efficiency. |
| Other paper experiments | The fixed-alpha dimension study is complete at its intended scale. The alpha sensitivity experiment, linear-Gaussian PMMH and univariate/multivariate stochastic-volatility studies are not replicated by it. | Separate implementations and plans for those studies if whole-paper coverage is the goal. They were outside the authorized first-study scope. |

For Eq.15, if p is the vector of candidate normalized Gaussian densities and b
the positive backward targets, eliminating the scale gives
L*=||p||^2-(p'b)^2/||b||^2. Increasing the Gaussian variance without bound makes
p approach zero on a finite training cloud and drives this loss towards zero.
This local derivation explains why solver initialization and constraints can
matter materially. It does not prove the authors' local optimizer failed. QR
avoids that amplitude escape by changing the objective; its results cannot
establish replication of Eq.15.

Selected completed-study values illustrate the remaining numerical differences:

| Quantity | Paper | R qr, six-estimate window | R short_qr, five-estimate window |
|---|---:|---:|---:|
| d20 SD of likelihood ratio | 0.19 | 0.10338 | 0.10338 |
| d20 mean resampling count | 27.61 | 9.297 | 9.297 |
| d80 SD of likelihood ratio | 0.35 | 0.17948 | 0.24963 |
| d80 mean final particle count | 1142 | 1971 | 1068 |
| d80 mean resampling count | 71.88 | 37.190 | 37.348 |

These are descriptive comparisons on different realized data and different
fitting procedures. A smaller SD is not itself a replication failure or evidence
of superiority. The complete intervals are in
phase03-first-study-1000/analysis-v1/summary.json.

## Separate BayesFilter questions

The d80 prefix-error veto is an additional project diagnostic, not a criterion
reported in the paper. The actual endpoint removes the future twist before
comparison with the original model's Kalman prefix, so this is not a comparison
of the raw twisted normalizer with the wrong target. Both variants have larger
observed ordinary-innovation prefix squared errors than BPF/SIS; their paired
95% difference intervals include zero. The frozen conservative promotion veto
therefore remains, but statistical inferiority and failure of the published
terminal-likelihood result do not follow. Concentration in a few replicas/time
points remains to be examined before choosing a numerical repair.

Multivariate full-filter R/TensorFlow parity remains an integration gap for the
BayesFilter debugging objective, not a prerequisite for reproducing a paper in R.
The current TensorFlow execute_iapf consumer still rejects state/observation
dimensions other than one. No LEDH, KDM or HMC conclusion follows from this study.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the R code as an explicit independent reconstruction | Published run counts completed; exact method replication open | Frozen d80 promotion veto retained | Original numerical choices and realized data | Specify the Eq.15 numerical procedure, audit prefix-error concentration, and correct timing boundaries before a new comparison | Original-author replication or production readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | No missing method records or recorded fit/tail failures. d80 particle allowance fails for qr; extra prefix screen vetoes both. |
| Statistically supported ranking | No overall ranking; the vetoing d80 prefix difference intervals include zero. |
| Descriptive-only differences | Raw paper-table differences and current instrumented runtimes. |
| Default-readiness | No change. Neither reconstruction establishes paper Eq.15 replication. |
| Next evidence needed | Resolved/frozen fitting choices, disjoint-data validation, fair timing; other paper studies only under their own scope. |

Skeptical review: do not confuse our added prefix veto with the paper's research
question, or table closeness with method identity. Data dependence and rare
weights remain alternatives to implementation failure. More replications of
unchanged QR cannot resolve a different fitting objective or unknown settings.

## Checked evidence

- `phase03-first-study-1000/analysis-v1/summary.json`: completed coverage,
  per-method intervals, resampling/particle means and conditional comparisons.
- `docs/benchmarks/compare_iapf_r_fitting.R`: lines43-80 contain the timer and
  unequal diagnostic work; lines89-104 record untwisted prefixes and terminal values.
- `docs/benchmarks/reference_iapf_plausible_choices.R`: lines68-94 bind both
  full-study consumers to QR, floor8 and delayed doubling, with explicit window5.
- `docs/benchmarks/reference_iapf_paper.R`: lines108-160 implement the APF and
  untwisted prefix; lines305-308 the floor; lines362-399 the controller.
- `docs/plans/artifacts/iapf-r-source-reconciliation-20260920-01/sources/arxiv-extracted/iapf_arxiv.tex`:
  lines515-547 Algorithm4, 655-704 fitting/floor/resampling, 742-822 first
  study and tables, 837 onward PMMH, 920 and1018 stochastic-volatility studies.
- `docs/plans/artifacts/iapf-r-replication-gap-audit-20260921-01/source-and-math-audit.md`:
  prior derivation, source retrieval boundary and mechanics tests. Its missing
  d40 and 1,000-replication entries are superseded by this completed campaign.

The available primary manuscript remains the technical authority for this
audit. The final publisher technical text and verified original-author code
remain unrecovered. A fresh arXiv/DOI web recheck returned HTTP502; it supplied
no new source evidence. No external contact or new research run was made.
