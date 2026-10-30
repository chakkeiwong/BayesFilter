# SIR FP32 score separation starts early
The saved FP32 and FP64 runs first exceed a one-percent cumulative-score
difference at observations 5 and 6 for the two checked parameter rows.
This is not merely an error in adding the final 50 score increments:
reconstructing the saved FP32 final scores from those increments changes them
by at most 8.25e-6, whereas final score component differences reach 27.75.
Both runs mark both rows valid throughout all 50 observations.

| Parameter row | First cumulative crossing | Scaled gap there | Final maximum scaled score gap |
|---|---:|---:|---:|
| Nominal | 5 | 9.20% | 48.69% |
| Offset | 6 | 18.13% | 8.75% |

The first crossings accompany modest primal differences: nominal ESS at
observation 5 is 800.760 versus 800.882, and offset ESS at observation 6 is
453.249 versus 453.154. FP32 reset covariance relative errors are around
1e-6; FP64 errors are around 1e-15. These saved measurements locate the
separation but do not identify its primitive or cause. The sequence score can
cross zero, so intermediate scaled ratios can be much larger than the terminal
ratio; they are localization diagnostics, not new acceptance thresholds.

The existing optional mixed-recursion tuning is the next discriminating
experiment. This analysis provides no reason to change the one-percent final
value/score criterion or to stop other models. A finite primal trajectory alone
does not establish numerical score accuracy.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain all-FP32 SIR rejection at these controls | Final score fails 1% | Arithmetic promotion veto; no continuation veto | Responsible primitive and sensitivity propagation | Finish current mixed-recursion scope and consumer checks | SIR or surrogate-HMC method failure |

| Inference status | Finding |
|---|---|
| Hard veto screen | All-FP32 score fails despite finite trajectories |
| Statistically supported ranking | None; deterministic saved-input comparison |
| Descriptive-only differences | Intermediate errors, ESS and covariance residuals |
| Default-readiness | No change to FP32 default; SIR scope remains unresolved in all-FP32 |
| Next evidence needed | Fresh mixed-scope precision/health checks, followed by actual-consumer pairing |

Command: tftwogpu Python inspect_saved_sir_precision.py in this directory.
Standard-library host analysis only; no framework import or GPU use.
Input hashes, exact command, reconstruction gaps and temporal rows are preserved
in sir-fp32-trace-diagnostic-01.json. Intent: sir-fp32-trace-intent.md.
Red-team: agreement with FP64 concerns the same finite filter, not the model
likelihood. Two rows cannot establish posterior-region accuracy. First visible
separation is not a causal localization of the underlying numerical operation.
