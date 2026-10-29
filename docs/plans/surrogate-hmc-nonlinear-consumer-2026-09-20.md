# M10 full-DCTR consumer mechanics

The question is whether the actual M10 log-noise parameter task can reach the
complete canonical DCTR proposal field and an independent endpoint through
the shared HMC integrator. The purpose remains testing surrogate HMC, whose
force may differ from the endpoint gradient. This stage supplies mechanical
prerequisites; it does not select controls, tune HMC or retain posterior draws.

Use the M10-T2 model, prior, fixed coefficients and once-rounded calibration
observations from the [qualified local reference](surrogate-hmc-nonlinear-reference-result-2026-09-20.md).
The coordinates are lambda_Q/R, with independent Normal(0,log(2)^2) priors and
identity chart. No Jacobian is added. Full DCTR uses N1024/K1024, FP32/TF32,
the canonical analytical recursion, UKF particle covariance lifecycle,
Contract E--Chol and complete dual-cap/GenUT correction. Model callbacks
supply both noise covariance tangents; initialization is independent of these
two parameters. No finite-program autodiff or reduced filter is substituted.

The endpoint for this mechanics check is explicitly the FP64 finite state-grid
likelihood (513 nodes, radius9) plus the proper normal prior. The likelihood
is numerically qualified only at the declared local probes. Its gradient is
the analytical derivative of that finite-grid computation. Neither endpoint
nor gradient is called an exact nonlinear-model oracle. Outside the checked
local coordinate box, mark this diagnostic endpoint unqualified; this is not
permission to truncate the normal prior or use the local endpoint for retained
inference. Full-prior posterior reference and endpoint qualification remain
required before a model-posterior claim.

Primary mechanics criterion: all checks of actual shared-callable parity,
bitwise replay, row independence/permutation, endpoint independence, force
sign/status, finite-grid endpoint/gradient binding, identity separation,
proposal reversal, per-chain energy, cached endpoint, finite states and
endpoint status pass. FP32 reversal tolerance is32 ulps at observed scale;
FP64 energy/cache tolerance1e-10. Report every original force status and any
zero extension separately. Invalid original forces block control admission
but may motivate the planned scope-specific repair; they do not reject the
surrogate-HMC direction. Acceptance, speed and force error are explanatory.

Default/assumption audit: protection_controls(1) is an unselected hypothesis
from the post-invalidation shared implementation. Its nonzero damping, ridge
and cap choices have not been justified for M10; this diagnostic measures
their initial health only. Before HMC tuning, a dedicated M10 non-harm curve
with calibration/validation separation must justify or replace them. The
identity mass, step size0.02 and two leapfrogs are fixed small-step diagnostic
choices, never proposed tuned settings. Four probe chains lie near lambda0,
including one reduced-noise point. Fixed random inputs use20260920 and16220,
16221,16222 on CPU; this tiny input-generation exception performs no training.

Pre-mortem and skeptical audit: a shared reference/force bug, missing noise
tangent, wrong prior/chart, altered observations or reduced batch lane could
pass an apparent sampler smoke. Independent reference/linear checks already
pass; model-callback finite differences and actual N1024 call-chain parity
address the remaining early risks. Mechanical success cannot address posterior
tails or mixing. No heuristic dominance claim is made; downstream inference
must still compare to its declared model reference and simple estimators.

Run GPU1 with escalation, verified memory growth, TF32/XLA and stable graph
signatures. Budget600 GPU/450 CPU seconds across one run plus one localized
infrastructure retry; tests60 CPU seconds. These consume M10's initial
7200 GPU/1800 CPU allocation, including all earlier preparation/tests.
Enforce wall/CPU watchdogs, charge every attempt, and preserve source snapshots,
binding/input hashes, per-check output, timings, allocator diagnostics and
accounting in fresh roots. Initial root:
`artifacts/surrogate-hmc-all-models-20260920-m10-consumer-01`.
Required stop conditions are incorrect model/derivative/wiring, numerical or
artifact invalidity, missing checks or exhausted allocation. A failed control
hypothesis is a repair trigger under the remaining M10 budget.

The public artifact-authority tuning route, if later prerequisites pass, is
`tune_hmc_kernel` with `VerifiedTensorFlowHMCConfig` and a repository-issued
typed deterministic-field binding. This mechanics runner issues no tuning
artifact. Review consulted `docs/reference/hmc-tuning-interface.md` and the
active capability registry; the fixed-transport tuner is inapplicable.
