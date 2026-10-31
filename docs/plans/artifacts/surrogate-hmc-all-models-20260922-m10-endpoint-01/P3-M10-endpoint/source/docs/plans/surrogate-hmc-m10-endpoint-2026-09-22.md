# M10 inference endpoint and complete cost qualification

## Question and evidence contract

The campaign goal is testing surrogate HMC with the full canonical analytical
DCTR proposal force. The immediate question is whether M10 can bind that force
to the independently qualified model-reference likelihood at every HMC
endpoint, and whether a complete inference schedule is affordable. The old
513-point local mechanics endpoint is ineligible. B/R, C/D, D/R and C/R remain
separate; this qualification runs no tuning or retained posterior comparison.

Use the unchanged M10 sine-transition/quadratic-observation model, T=2,
N=K=1024, nominal claim data from reference01, selected multiplier0.1 from
controls01, and the full independent Normal(0,log(2)^2) priors on log noise
scales. The independent posterior reference is posterior05. The new endpoint
uses its finest state quadrature (Legendre32, normal128) and both observation
roots, in log arithmetic. It remains an approximate numerical model endpoint;
finite quadrature is not an exact likelihood oracle. Retain the existing
reference allowances and analytical omission bounds. Do not truncate the
parameter prior, floor the likelihood or alter canonical DCTR.

The pass criterion is executable numerical and consumer qualification:
agreement with the existing independent quadrature on representable values,
finite log densities in underflow cases, finite-difference derivative agreement
at interior points, refinement checks, deterministic replay, row independence,
shared canonical force parity, endpoint/force independence, proposal reversal,
and healthy complete transition probes. A failed check is a repair trigger and
blocks inference launch. Cost exceeding the remaining allocation is a launch
veto, not evidence against surrogate HMC. Force/reference gradient disagreement,
acceptance and timing variation are explanatory only. Passing cannot establish
posterior recovery, convergence, superiority, exact-score admission or default
readiness. Preserve report.json, run-manifest.json, all measured probes and
source snapshots in a fresh versioned output root.

## Mathematical and default audit

The likelihood integrals are positive weighted sums. Replace
log(sum_i w_i exp(l_i)) by logsumexp_i(log(w_i)+l_i), including the initial
convolution and both observation integrations. A zero quadrature weight has
log weight minus infinity; evaluate its unused logarithm at one to avoid a
spurious derivative of log(0). This is the same finite quadrature in real
arithmetic. Its total reference derivative includes moving integration nodes.
Autodiff here is an independent reference tool, never the canonical DCTR score.
Conditional predictive probabilities use normalized log weights. Compare all
quantities where the previous density-domain calculation is representable;
record the previously subnormal case (-3.3648781186133125,-3.192114385833288).

Refinement vetoes apply at the nominal, opposing one-SD and four three-SD
corner probes: relative change (difference divided by one plus the lower-level
absolute value) must be at most 1e-6 for log likelihood and predictive CDFs,
and 1e-5 for the score, following the earlier quadrature screen. Wider six/nine-SD
and preserved underflow probes must remain finite; their refinement differences
are explanatory tail diagnostics. Posterior05's separate tail allowances remain
necessary; finite tails alone do not establish accurate full-prior integration.

| Choice | Provenance and reason | Failure mode and early diagnostic | Status |
|---|---|---|---|
| 32/128 quadrature; both roots and existing partitions | Qualified posterior05 and nonlinear_quadrature.py | Missed narrow density or root-solver failure; old/new parity and 20/32 refinement | Reference hypothesis qualified on the frozen data |
| Log arithmetic | Positive-sum identity above | NaN derivatives through zero weights; finite differences and subnormal regression | Numerics-changing candidate; mandatory non-harm evaluation |
| Full normal log-noise priors | Existing M10 reference model | Accidental truncation or omitted prior gradient; direct prior identity | Fixed target definition |
| Full canonical DCTR, multiplier0.1 | Separate calibration/validation non-harm selection | Broken wiring or new status failures; direct batch parity and transition checks | Frozen proposal-field hypothesis, not exact score |
| FP64 endpoint/integration, FP32 TF32 DCTR, GPU XLA | Existing campaign policy | Backend or casting disagreement; explicit dtype/device and replay checks | Recorded mixed precision |
| Finite probe locations | Nominal, ±1/3 prior SD, wider ±6/9-SD tails and preserved underflow case | Sparse points miss narrow failures; refinement and eventual per-transition guards | Diagnostic design only |
| Relative parity 1e-9, derivative 2e-5, reversal 1e-8 | FP64 operation changes and central finite differences | Loose tolerance hides error; preserve raw differences, refinements and step-size sensitivity | Qualification tolerances, not posterior margins |
| Initial epsilon0.02, identity mass, L1/L2 | Earlier mechanics controls, used only to measure real complete transitions | Tiny acceptance misread as tuning success; no inference claim | Cost probe only |

The constructed heuristic set for eventual usefulness comparisons remains
prior-center prediction, prior-predictive integration, and a linearized Kalman
approximation, separately on nominal and small-noise/boundary situations.
No heuristic comparison is performed or passed by this endpoint qualification.

Skeptical pre-run audit: the old local endpoint would answer the wrong question;
the new endpoint explicitly replaces it. Density-domain underflow is a known
engineering failure and must be repaired before HMC. Qualification cannot make
the approximate reference exact or promote a short cost probe to inference.
Independent force/endpoint bindings may differ. Public tuning must use the
tested-supported, artifact-authority tune_hmc_kernel typed position-field
binding; the fixed-transport tuner is inapplicable. Review the new consumer's
actual call chain and preserve every original DCTR failure status.

## Budget and execution

Reserve at most 360 CPU-worker seconds and 480 GPU seconds for the first
qualification, plus at most 45 CPU seconds for focused tests. This fits M10's
remaining initial allocation (about690 CPU seconds); it does not reserve or
authorize an unaffordable inference launch. Recompute family/global spending
before launching. Global caps remain 54,000 CPU /226,800 GPU seconds; one serious
GPU task at a time. A localized repair may use the unspent reservation; any
additional stage allocation must be recorded prospectively under master §12.

Environment: /home/chakwong/anaconda3/envs/tftwogpu/bin/python; escalated GPU1
GPU-68251639-fe82-8f81-3ccc-2953c32e805b, verified memory growth, XLA and TF32.
Two TensorFlow intra/inter-op threads, OMP2, four CPU cores at most. CPU tests
set CUDA_VISIBLE_DEVICES=-1 before import. No package or backend changes.
New entry point: scripts/qualify_surrogate_hmc_nonlinear_endpoint.py, dispatched
by master stage P3-M10-endpoint. Exact command and arguments go in the manifest.
Output: docs/plans/artifacts/surrogate-hmc-all-models-20260922-m10-endpoint-01.

Measure compilation and at least four steady complete four-chain transitions
for L1 and L2. Forecast the complete proposed public tuning/verification and
two-look retained schedule from the slowest observed L2 cost with the master's
1.5 factor, plus explicit preparation/analysis/compilation charges. Reserve that
whole schedule before any later inference run; the optimistic local-endpoint
screen is superseded. Stop on a failed scientific check, source/data drift,
nonfinite accepted state, unverified GPU memory policy, or exhausted budget.
If the probe passes, write a separate prospective inference allocation and
retain the same target, public tuning authority, margins and closed-look rules.

Pre-mortem: this can pass sparse probes yet fail on a chain, or be numerically
faithful but too expensive. Per-transition validity remains mandatory and cost
must include the actual endpoint. A failure attributable to log arithmetic,
quadrature refinement or compilation is an implementation/reference repair
trigger, not a negative result about the DCTR surrogate force.
