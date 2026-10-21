# Surrogate-HMC execution

Status: INCOMPLETE_IMPLEMENTATION. See decision.json, diagnostics and the complete per-job log. Posterior comparisons: {'B/R': 'NOT_RUN', 'C/D': 'NOT_RUN', 'D/R': 'NOT_RUN', 'C/R': 'NOT_RUN'}. All-model validation remains incomplete.

Gaussian mechanism comparisons: {'exact/analytic': 'NOT_RUN', 'half/analytic': 'NOT_RUN', 'rotational/analytic': 'NOT_RUN', 'zero/analytic': 'NOT_RUN'}. These controls do not establish DCTR model-inference performance.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| P0 | {'B/R': 'NOT_RUN', 'C/D': 'NOT_RUN', 'D/R': 'NOT_RUN', 'C/R': 'NOT_RUN'} | See stage reports | Monte Carlo and reference precision | Resolve measured consumer/tuning prerequisites before posterior comparisons | All-model posterior recovery |

| Inference status | Evidence |
|---|---|
| Hard veto screen | See consumer checks, tuning verification and retained look diagnostics |
| Statistically supported ranking | None |
| Descriptive differences | Timing, acceptance and ESS alone do not rank methods |
| Default readiness | Not established |
| Next evidence | Resolve four posterior comparisons and heuristic checks |
