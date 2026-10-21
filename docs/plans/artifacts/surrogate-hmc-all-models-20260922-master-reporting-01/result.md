# Surrogate-HMC execution

Status: INCOMPLETE_IMPLEMENTATION. See decision.json, diagnostics and the complete per-job log. Posterior comparisons: {'B/R': 'INCONCLUSIVE', 'C/D': 'INCONCLUSIVE', 'D/R': 'DISCREPANCY', 'C/R': 'INCONCLUSIVE'}. All-model validation remains incomplete.

Gaussian mechanism comparisons: {'exact/analytic': 'INCONCLUSIVE', 'half/analytic': 'INCONCLUSIVE', 'rotational/analytic': 'INCONCLUSIVE', 'zero/analytic': 'TUNING_REJECTED'}. These controls do not establish DCTR model-inference performance.

KSC T2 posterior comparisons: {'B/R': 'INCONCLUSIVE', 'C/D': 'INCONCLUSIVE', 'C/R': 'INCONCLUSIVE', 'D/R': 'EQUIVALENT'}. Retained surrogate-HMC test recorded: True.

M10 T2 posterior comparisons: {'B/R': 'INCONCLUSIVE', 'C/D': 'NOT_RUN', 'D/R': 'NOT_RUN', 'C/R': 'NOT_RUN'}. Retained surrogate-HMC test recorded: True.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| P0 | {'B/R': 'INCONCLUSIVE', 'C/D': 'INCONCLUSIVE', 'D/R': 'DISCREPANCY', 'C/R': 'INCONCLUSIVE'} | See stage reports | Monte Carlo and reference precision | Assess LG1 posterior comparisons and continue the remaining model queue | All-model posterior recovery |

| Inference status | Evidence |
|---|---|
| Hard veto screen | See consumer checks, tuning verification and retained look diagnostics |
| Statistically supported ranking | None |
| Descriptive differences | Timing, acceptance and ESS alone do not rank methods |
| Default readiness | Not established |
| Next evidence | Resolve four posterior comparisons and heuristic checks |
