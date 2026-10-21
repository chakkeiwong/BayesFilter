# Surrogate-HMC execution

Status: INCOMPLETE_IMPLEMENTATION. See decision.json, diagnostics and the complete per-job log. Posterior comparisons: {'B/R': 'NOT_RUN', 'C/D': 'NOT_RUN', 'D/R': 'NOT_RUN', 'C/R': 'NOT_RUN'}. All-model validation remains incomplete.

Gaussian mechanism comparisons: {'exact/analytic': 'INCONCLUSIVE', 'half/analytic': 'INCONCLUSIVE', 'rotational/analytic': 'INCONCLUSIVE', 'zero/analytic': 'TUNING_REJECTED'}. These controls do not establish DCTR model-inference performance.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| all | {'B/R': 'NOT_RUN', 'C/D': 'NOT_RUN', 'D/R': 'NOT_RUN', 'C/R': 'NOT_RUN'} | See stage reports | Monte Carlo and reference precision | Preserve the completed C01 attempt and continue remaining model consumers | All-model posterior recovery |

| Inference status | Evidence |
|---|---|
| Hard veto screen | See consumer checks, tuning verification and retained look diagnostics |
| Statistically supported ranking | None |
| Descriptive differences | Timing, acceptance and ESS alone do not rank methods |
| Default readiness | Not established |
| Next evidence | Resolve four posterior comparisons and heuristic checks |
