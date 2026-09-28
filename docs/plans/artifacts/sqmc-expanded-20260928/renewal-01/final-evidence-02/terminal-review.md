# Terminal review — expanded SQMC comparison

Reviewed after numerical completion on 2026-09-28. This is a Codex terminal
review with executable checks, not an independent external review.

**Disposition: accept the completed descriptive comparison; withhold method,
scientific and default promotion.** All 32 route/scope units, 128 valid final
cells and 8,480 score coordinates are present. IID has smaller observed score
error in 27 of 96 SQMC route/data/design comparisons; these conditional losses
are promotion vetoes. They do not invalidate the saved experiment.

## Checked evidence

| Question | Finding | Evidence |
|---|---|---|
| Requested design complete? | Yes: eight scopes, four routes, N=1008/1020, all 4/17/157 coordinates, all final 2×2 pairs | [Audit](report/audit.json), [scores](report/scores.csv) |
| Exported values faithful to worker results? | Every score, exact score, signed/absolute error and likelihood verified; no mismatch | Audit plus terminal-checks.json |
| Data/oracle matched across routes? | Saved observations, Kalman outputs and seed partitions agree within each scope | Worker data hashes and report audit |
| Tuning frozen before final evaluation? | Saved controls match tuning artifacts; complete calibration/validation/final partitions preserved | Unit tuning paths in audit.json |
| Actual source preserved? | Every archived file matches its SHA-256 and current file; numerical source closure identical across all final units | provenance/source-sha256.json, source-final.tar.gz |
| Shared analytical evaluator called? | Campaign endpoint wiring tests passed; expanded worker calls evaluate_diagnostic, whose kernel calls canonical_value_and_analytical_score | run_sqmc_expanded_comparison.py worker; sqmc_campaign_tf.py lines 82–115; cpu-01.log |
| GPU and numerical mode? | Recorded GPU probe, FP64 reference mode, XLA on, TF32 off, verified memory growth before initialization | Each unit manifest, report audit |
| Invalid candidates hidden? | No invalid final cells. Original rejected calibrations preserved separately with sentinel labels and unavailable errors | original-calibration-rejections-01 |
| Heuristics evaluated against correct target? | Zero and first-only scores target the full-horizon Kalman score; matched IID loss labels independently recomputed | conditional tables and terminal-checks.json |
| Budget respected? | 21,377.330 GPU-process seconds charged; 21,822.670 seconds remain. Numerical work finished in 6.3825 elapsed hours | budget-at-completion.json |
| Figures readable? | Both final score and likelihood plots visually inspected; labels and legend fit; all eight scopes present | report/figures |

The call-chain claim is supported by executable wiring and finite-program
checks, not merely the existence of the shared function. Small-fixture checks
and graph/XLA parity do not prove every large finite-program gradient correct.
Full d=10 Kalman finite differences covered 11 selected coordinates; callbacks
covered every coordinate. No N=1020, T=120 all-coordinate finite-difference
accuracy proof is asserted.

## Findings and their disposition

1. **Conditional IID losses:** retained prominently in the report. They veto
   promotion for the observed situations; no statistical ranking follows from
   the count, because comparisons share two independent datasets per scope.
2. **Different selected controls:** full-d3 SQMC uses greater smoothing than
   IID, and full-d10 T=120 inverse CDF selects eight flow steps while its
   comparators select two. The report compares tuned configurations and does
   not attribute all differences to ancestry ordering.
3. **Stale generic pilot sentence:** repair-unit manifests inherited a sentence
   about previously inspected P44 pilot pairs. The actual full-model seed
   partitions and row labels are fresh and correct. Reports clarify the scope;
   historical manifests are not rewritten.
4. **Report-only defects:** a stale balancing description and a CSV field-list
   collision were fixed. Five deliberate CSV corruptions were rejected; clean
   interim and final audits passed. Original intermediate outputs are preserved.
5. **Inherited numerical protections:** the epsilon/flow calibration does not
   validate every protection or establish optimality. FP64 reference results
   cannot promote the FP32/TF32 production target.

## Scientific assessment

The target comparator is the exact marginal-likelihood Kalman score for the
matched predict-first Gaussian model. The candidate computes an analytical
score for a finite particle program. The saved nonzero errors show these are
different computed quantities in these runs. Finite particle count, integration,
smoothing and piecewise ancestry/order choices can contribute; the campaign
does not identify their individual causal shares.

All final configurations pass the implemented validity screen. Original mass-
guard rejections remain rejected candidates. The renewal required one fixture
repair retry; candidate rejections and a failed affordability projection were
recorded separately. No final repair unit suffered an infrastructure failure.

No statistically supported method ranking, posterior correctness, HMC readiness,
production admission or default readiness is established. The strongest
alternative explanation for favorable descriptive differences is the particular
data/design pair, compounded where selected controls differ. Replication on
independent datasets with predeclared paired uncertainty analysis could reverse
the observed ordering. Errors that persist under numerical refinement would
strengthen the case for a structural limitation. The narrow model regime and
two independent datasets are the weakest evidence.

The requested comparison is complete. Preserve the results and use the specific
conditional failures to design any separately authorized follow-up.
