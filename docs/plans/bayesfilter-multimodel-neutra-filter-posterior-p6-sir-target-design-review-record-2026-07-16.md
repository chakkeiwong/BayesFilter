# P6 SIR Target-Design Review Record

Date: 2026-07-16

Reviewed path:
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p6-sir-target-design-subplan-2026-07-16.md`

## Findings And Repairs

| Finding | Severity | Repair |
| --- | --- | --- |
| Parent plan claimed P0 froze SIR data/prior/chart/filter settings, but P0 recorded them as missing. | material | Added a dedicated target-design rung before implementation/HMC. |
| Simulator emits `y0:y20`, while paper defines observations at `y1:y20`. | material | Froze `[1:21]` and required a negative hash for `[0:20]`. |
| Three-parameter inference could be mislabeled Zhao-Cui source-faithful. | material | Classified all three parameter posteriors as BayesFilter extensions; source ledger binds paper/author evidence. |
| Local complete-data scores could be substituted for observed-data likelihood. | material | Made this a hard veto and negative substitution. |
| Initial review incorrectly treated 18D level-2 SGQF as mathematically infeasible; the exact cloud has 37 points, while the generic merge code attempts a `3^18` neighbor search. | material | Reclassified this as a builder pathology; require an exact axis-cloud constructor, low-dimensional parity, and a frozen 37-point memory forecast. |
| Prior scale was an unexamined convenience. | material | Classified Normal(0,0.5^2) as a target-specific hypothesis with prior-predictive and information diagnostics. |
| PF/reference outputs might become promotion criteria. | moderate | Kept them explanatory or veto-only; target identity requires graph-native value/score and independent recomposition. |

Verdict: `AGREE_AFTER_VISIBLE_REPAIR`.

Claude review is advisory and private-workspace disclosure was already blocked
by the platform in P5. Repeating the same unavailable review is not a valid
reason to stop. Material source claims are instead bound to the inspected local
paper and author code; implementation still requires focused numerical tests.
