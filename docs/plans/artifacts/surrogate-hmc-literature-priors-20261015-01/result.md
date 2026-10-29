# Literature priors adopted for the surrogate-HMC tests

The five-model master and its preparation launcher now default to
`literature-model-priors-v1`. The scientific question remains whether the
filter's analytical score works as a surrogate HMC proposal force, with an
independent acceptance potential. The chosen prior defines the posterior and
is included consistently in both calculations.

The published Zhao–Cui predator-prey box is reproduced unchanged. Its empirical
SV prior is conditioned on the existing KSC fixture's fixed innovation variance
and positive persistence. Zhao–Cui fixes SIR's epidemic rates and has no
range-bearing example. Representative sources supply explicit adaptations for
SIR, range bearing, and the local LGSSM. Full texts, author-code evidence,
source discrepancies, mathematical transformations and the six literature
ledgers are in `docs/papers/surrogate-hmc-priors-2026-10-15/`.

The material local assumptions are the SIR calibration of 45 days per
benchmark time unit, static IG(1,1) covariance multipliers, the restricted
LGSSM AR support, and the transferred Normal prior on its initial-mean
coefficient. These are documented modeling choices, not claims that each
paper used the exact local model. IG(1,1) is proper with infinite mean.
The saved observations, likelihood models and filter kernels were not changed.

## Verification

Final CPU-only regression: **184 passed, 2 dependency deprecation warnings,
27.05 seconds**, exit 0. Log: `pytest-attempt-02.log`. GPU devices were
intentionally hidden before TensorFlow import. The earlier focused run passed
32 tests in 21.60 seconds and remains in `pytest-attempt-01.log`.

Checks cover normalized densities against independent TFP distributions,
analytic scores against diagnostic autodiff and central differences, stable
tail expressions, support, CPU-XLA scalar/batch parity with bounded tracing,
and the actual scalar force, batched force and independent endpoint consumers
using controlled likelihood fixtures. Orchestration regressions check prior
selection and worker forwarding, stale-scope rejection, benchmark-prior
rejection for scientific inference, and existing budget/continuation behavior.
No GPU filter sweep or posterior sampling was run in this phase.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Adopt the documented prior specification for this benchmark | Source mappings and transformed densities checked | No failed density, derivative or wiring checks | Scientific suitability of the declared adaptations and heavy tails | Check model/filter/endpoint numerical behavior over prior-supported values | These are not literal reproductions of all source experiments |
| Use the new master/preparation prior wiring | 184 regressions passed | Old prior scopes are rejected | Real GPU behavior under broader parameter support | Issue fresh scope-specific selections through the accepted-OT preparation route | Old tuning does not establish readiness for the new posterior |
| Keep the surrogate-HMC campaign unfinished | No new posterior samples | No new scientific candidate rejection in this phase | Convergence, posterior accuracy and remaining runtime | Adaptive warmup and cumulative sampling after required checks/tuning | Prior implementation is not HMC convergence evidence |

| Inference status | Finding |
|---|---|
| Hard veto screen | Correctness tests pass; unsupported prior IDs and stale benchmark scopes are rejected. Real prior-support filter validity is not checked here. |
| Statistically supported ranking | None; this was a deterministic correctness/source task, not a prior or sampler comparison. |
| Descriptive-only differences | CPU test elapsed times are operational records, not speed-comparison evidence. |
| Default readiness | Literature prior is wired as the campaign choice under owner direction. Broad scientific suitability and posterior readiness remain unevaluated. |
| Next evidence needed | Numerical support checks, fresh offline tuning, then cumulative HMC diagnostics against the declared endpoint. |

## Review and continuation

The strongest alternative explanation for future failures is that a correctly
implemented prior exposes numerically difficult regions of the fixed models,
particularly SIR's fixed-step ODE and large variance multipliers. The present
unit checks would not detect that. A source-unit mismatch, real endpoint
failure, or incorrect force behavior on that support would require repair;
it would not justify silently narrowing the prior. Source/code discrepancies
are retained, and the published settings control the choices identified in the
source README. No claim of universal or uniquely appropriate priors is made.

The preparation-launcher audit found and repaired an additional default-prior
gap before final verification. Both launchers now carry the chosen prior into
the Case, and the scope binds its settings, chart and source hash. The next
work remains part of the existing master: prior-support checks, fresh tuning,
adaptive warmup, and cumulative retained sampling. Keep the accepted epsilon
1/five-step OT choice, N4032/K2016, FP32 with TF32 off, and SIR's recorded mixed
precision exception. Do not restart the settled OT comparison or add a separate
terminal validation chain.

Campaign budget is unchanged: 38,550.877704 seconds used of 86,400; 47,849.122296
seconds (13.29 GPU hours) and 11 of 140 attempts remain. No worker is running.
Exact test commands, source snapshots and hashes are in `manifest.json`.
