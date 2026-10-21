# M10 endpoint qualification for the surrogate-HMC test

The M10 inference consumer passed all16 engineering checks on GPU1. This
qualifies the endpoint and canonical proposal-force wiring for a bounded
surrogate-HMC experiment. The research question remains whether HMC using
the imperfect full canonical analytical DCTR position-only force recovers the
declared model-parameter posterior after endpoint correction. Endpoint
qualification itself supplies no retained posterior evidence.

Evidence is preserved under
[endpoint01](artifacts/surrogate-hmc-all-models-20260922-m10-endpoint-01/P3-M10-endpoint/report.json),
with the exact command, source snapshot, input hashes, GPU memory policy and
timings in its run manifest and worker log. The child completed with exit0.
The master returned2 because the overall all-model implementation is still
incomplete. Master accounting charges102.805553 CPU-worker seconds and
91.271959 GPU seconds; the child's overlapping cost is not charged again.

M10 uses the sine transition and quadratic observation law at T=2 with
N=K=1024, the original nominal claim observations, and independent
Normal(0,log(2)^2) priors on log process and observation-noise multipliers.
Controls01 selected multiplier0.1 on separate calibration/validation data.
The consumer now binds the actual claim observations to the shared canonical
DCTR implementation. It does not relabel claim observations as calibration
data, substitute an autodiff score, or require its proposal force to equal
the reference endpoint gradient.

The B endpoint evaluates the independently qualified32/128 observation-root
quadrature in FP64 log arithmetic, with both roots and the full normal prior.
Positive quadrature terms are summed by log-sum-exp. Density-domain parity,
total reference derivatives, refinement and two underflow probes passed.
The derivative check concerns this independent reference, not DCTR score
admission. This is a numerical model-likelihood reference, not an exact
likelihood oracle or a proof of accuracy at every possible HMC position.

The remaining checks cover claim/reference binding, the canonical call chain,
force and endpoint replay, row independence, force health and reversibility.
Reversal error was1.1102230246251565e-16. All ten complete four-chain
transitions passed health checks: compilation plus four steady evaluations
for each of L1 and L2. The slowest steady L2 measurement was0.415736492 CPU
seconds and0.420167044 wall seconds. XLA, TF32 and GPU memory growth were
recorded and verified.

## Cost correction before inference

The archived endpoint report undercounted nomination candidates and did not
separately cost every retained predictive functional. Preserve that report;
its original forecast cannot authorize inference. Recompute from its measured
costs for both L1/L2 candidates, two mass windows, adaptation and nomination
verification,250 discarded warmup,1,000 fresh verification,2,000 retained
transitions,48 extra calls,240 compilation seconds,90 analysis seconds and
8,000 predictive evaluations. With the1.5 safety factor the corrected
forecast is3,005.918236 CPU seconds and3,028.145273 wall seconds.

The separately written
[M10 inference plan](surrogate-hmc-m10-inference-2026-09-22.md) funds one B/R
attempt by transferring2,800 CPU seconds from unused P5 to P3 within the
existing campaign cap. C/D, D/R and C/R require a separately qualified finite
D endpoint/reference and remain outside this B/R launch.

| Decision | Primary criterion | Veto checks | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Endpoint qualified | All16 checks pass | No observed endpoint/force/reversal failure | Sparse probes and empirical quadrature refinement do not prove global accuracy | Execute public tuning and the two frozen B/R looks | Posterior recovery, exact score, superiority or default readiness |

| Inference status | Evidence |
|---|---|
| Hard veto screen | No engineering veto in this bounded qualification |
| Statistically supported ranking | None |
| Descriptive-only differences | Transition and reference timings |
| Default readiness | Not established |
| Next evidence needed | Retained B/R functions with valid simultaneous uncertainty; independent C/D work and heuristic comparisons remain separate |

The strongest alternative explanation for success is that the probe set misses
regions where the quadrature or DCTR force fails. The actual claim chains and
per-call health checks can expose that failure. The weakest evidence is the
finite probe/refinement coverage; a successful smoke cannot resolve it.
