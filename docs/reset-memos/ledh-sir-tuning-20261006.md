# SIR oracle-free tuning checkpoint

Active question: can offline controls reduce SIR value/score variability and reset distortion while preserving LGSSM, KSC SV and predator-prey? Branch: sqmc-development. Plan: docs/plans/ledh-sir-no-oracle-tuning-20261006.md.

Campaign complete: all 168 retained calibration/validation/confirmation rows are valid at T=10,20,40,50. No alternative passed all calibration screens, so baseline_marginal remains selected. Its two-seed heuristic comparisons veto promotion; no statistical ranking or new default is justified. Eight-design score SD remains about (5688,2518,2140) at T50. The early relative ESS can fall to 1/1008, while later reset-score errors are much smaller. This identifies a regime to investigate, not a proven error cause.

Verification: 54 focused tests pass; all 12 protected model/horizon values and every score coordinate replay exactly. The smaller-step score ladder agrees in all three coordinates at every horizon on the first confirmation design. Coarse finite differences were unreliable; curvature versus branch effects was not isolated. No analytical derivative code changed and no full SIR oracle claim is made.

Evidence: docs/benchmarks/ledh-sir-no-oracle-tuning-results-20261006.md and docs/plans/artifacts/ledh-sir-no-oracle-tuning-20261006-01/terminal_evidence.json. All attempts, including the interrupted T50 calibration attempt, are preserved. Main plus localization workers used 11777.555 of 28800 seconds; 17022.445 remain. No further sweep is implied by the unused budget.

Documentation and review complete: the 620-page monograph compiles with its bibliography and no undefined references. The new equations, algorithm and result table were inspected in the rendered PDF under documentation-02. Terminal self-review: docs/reviews/ledh-sir-no-oracle-tuning-result-review-20261007.md. Narrow command allow-list entries cover the campaign and bounded score diagnostic; no broad permission rule or package change was introduced.

Campaign closed: the final source/evidence consistency check passed, including all 168 retained rows, exact protected replays, the confirmation table and PDF/source hashes. The completed task is ready for its repository commit. A future scientific repair should focus on early weight concentration under fresh data and a new predeclared comparison; the current candidate's rejection does not reject the marginal-mixture research direction. There are no remaining campaign stages to execute.
