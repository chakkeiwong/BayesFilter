# SV20 surrogate-HMC result

SV20 completed an actual surrogate-HMC test: public HMC tuning followed by
2,000 retained draws per chain in four chains using the full canonical
analytical DCTR proposal field and independent model-likelihood endpoint
correction. **B/R is INCONCLUSIVE.** Both predeclared looks failed the
function-specific Monte Carlo uncertainty qualification. C/D, D/R and C/R
remain NOT_RUN. This does not reject surrogate HMC or establish equivalence.

The target is the original T20 actual stochastic-volatility model, with
physical gamma/beta parameters and independent Uniform(.1,.9) priors. The
endpoint uses the checked deterministic state quadrature; R is the independently
refined full-prior numerical posterior reference. Their numerical relation is
supported by empirical refinement checks, not a rigorous integration bound.
The force is an imperfect deterministic position-only DCTR field. Agreement
between its gradient and the endpoint gradient is not required by the test.

The [plan](surrogate-hmc-sv20-inference-2026-09-22.md) fixes the thirteen
physical-parameter and predictive functionals, margins, two looks, four
comparison families and MCSE qualification. The completed
[report](artifacts/surrogate-hmc-all-models-20260922-sv20-inference-01/report.json)
preserves the original calculations. Public tuning passed all twelve checks
with one leapfrog step and step size 0.8599962187 with its own adapted
metric. The worker fixed `max_leapfrog_steps=1`; longer trajectories were not
compared. The [code/literature audit](surrogate-hmc-algorithm-code-literature-audit-2026-09-22.md)
explains the resulting one-step Gaussian drift proposal. Its 250 discarded
warmup transitions and 1,000 fresh verification
transitions are excluded from retained posterior estimates.

| Retained draws per chain | Required MCSE failures | Largest parameter R-hat | Minimum bulk / tail ESS | Decision |
| --- | --- | --- | --- | --- |
| 1,000 | Standardized beta second moment: batch correlation −.376559 | 1.006049 | 1,037.75 / 666.94 | INCONCLUSIVE |
| 2,000 | Standardized gamma second moment −.382835; beta second moment −.348795; gamma 90%-CDF −.415436; beta 10%-CDF −.496898 | 1.001690 | 1,909.41 / 1,677.81 | INCONCLUSIVE |

These are lag-one batch correlations at the longest declared batch length;
the frozen absolute-correlation limit is .3. MCSE sensitivity ratios did not
cause these failures. Neither look records a discrepancy flag. Without valid
function-specific uncertainty, absence of a flag cannot establish agreement.
Parameter R-hat and ESS are descriptive here and do not override that veto.

All eight retained chunks passed the generic numerical health checks. The
separate [original-force audit](artifacts/surrogate-hmc-all-models-20260922-sv20-force-health-audit-01/report.json)
also checked warmup, verification and every retained chunk: all four chain
fallback counts were zero. It read existing telemetry and took no posterior
look. This matters because generic HMC health permits a qualified deterministic
field extension, while this scope requires the original field to remain
healthy. The worker now checks that condition before evaluating posterior
functionals, and the history reader verifies inline or archived-audit evidence.
Forty-nine focused regression tests passed, including actual worker execution
showing that a fallback stops posterior analysis.

The run preserved 104 source dependencies, all unchanged at completion. The
[manifest](artifacts/surrogate-hmc-all-models-20260922-sv20-inference-01/run-manifest.json)
records command, commit, source snapshots, data, seeds, GPU1, tftwogpu,
TF32/XLA and verified memory growth. Peak TensorFlow allocation was 545,516,032
bytes. Total inference charge was 9,976.426562 CPU-worker and 10,155.287274
GPU-wall seconds, including orchestration, inside the 12,000/12,600 reservation.
The independent force audit charged 4.074844 CPU seconds and no GPU time.
Archived numerical sources remain the authority after subsequent reporting
and guard repairs.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve B/R as INCONCLUSIVE | Joint equivalence not established | Functional MCSE qualification fails; numerical and original-force health pass | Adequacy of the short-chain uncertainty screen and attainable precision | Prospectively validate the diagnostic policy and fund a complete new design before further inference; qualify finite D independently for C/D | Neither posterior equivalence nor bias, efficiency superiority, default readiness or direction rejection |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | The declared uncertainty qualification fails at both looks; no implementation, non-finite, movement or original-force failure was observed |
| Statistically supported ranking | None |
| Descriptive-only differences | R-hat, ESS, acceptance, runtime and unqualified posterior differences |
| Default readiness | Not established; heuristic comparisons and wider scopes remain incomplete |
| Next evidence needed | Prospective validation across all actual functionals and comparison families, an adequately funded precision design, and independently qualified finite-D comparisons |

Post-run red team: negative batch correlations can reflect the variability of
the short-batch diagnostic rather than bad posterior inference. The
[precision audit](surrogate-hmc-inference-precision-result-2026-09-22.md)
also shows that this 2,000-draw schedule does not justify narrow joint
equivalence intervals even in a favorable IID median-CDF calculation. Neither
observation licenses relaxing a closed look. A prospective calibrated design
with valid uncertainty and tighter intervals could resolve this result in
either direction. The weakest evidence is the joint uncertainty qualification,
not an observed failure of the deterministic proposal or endpoint correction.
