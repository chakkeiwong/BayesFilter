# Nonlinear iAPF implementation inside master Phase 0E

The active question is whether the existing bounded density-fit iAPF procedure can evaluate the scalar sine-transition/quadratic-observation model through its shared fitting and twisted filtering kernels, with correct likelihood corrections and analytical frozen-fit derivatives. The adaptive-count slice at `1c12eefa` has completed. Nonlinear implementation is the next prerequisite, not an additional project outside the master.

## Skeptical audit and mathematical scope

The transition is `x_t = a*x_(t-1) + c*sin(x_(t-1)) + e_t`, with Gaussian variance Q, and the observation is `y_t = h*x_t + b*x_t^2 + v_t`, with Gaussian variance R. The existing six parameters control a, h, Q, R and the initial Gaussian law; c and b are fixed scope fields. This is a regular additive-Gaussian fixture. Degenerate DSGE transitions and smoothing-based scores remain outside this work.

Write the transition mean as u(x), and the frozen twist as psi(z)=N(z;m,V)+epsilon with V positive definite and epsilon positive. Direct integration gives

`M(x) = integral N(z;u(x),Q) psi(z) dz = N(m;u(x),Q+V)+epsilon`.

The normalized twisted transition is a mixture: the prior N(u,Q) has probability epsilon/M, and the Gaussian-product component has probability N(m;u,Q+V)/M, mean `u+Q(Q+V)^(-1)(m-u)` and covariance `Q-Q(Q+V)^(-1)Q`. This is exact for nonlinear u; no transition linearization occurs. The shared normalizer and sampler already implement this algebra.

The corrected observation potential is `g_t(y_t|x_t) M_(t+1)(x_t)/psi_t(x_t)`, with terminal M_(T+1)=1 and initial normalizer E_(x0) M_1(x0). Initial particles approximate the latter as in the existing local adaptation. Cancellation of the successive psi and M factors recovers the original unnormalized path density. The nonlinear observation must enter g itself and each backward fitting target `g_t M_(t+1)`; replacing it by a linear observation would be wrong for this target.

At fixed c,b and fitted coefficients, the transition-mean tangent is `du = da*x + (a+c*cos(x))*dx`, and the observation-mean tangent is `dh*x+(h+2*b*x)*dx`. The existing Gaussian-density and normalizer tangents then include Q/R dependence. Discrete ancestor/mixture labels, offline fitting and realized N are held fixed. This computes the derivative of that finite program almost everywhere, with no unbiased marginal-score or derivative-through-fitting claim.

Audit passed for this bounded implementation: Gaussian conditional densities make the normalizer tractable; the analytic chain rule covers both nonlinear means; the actual consumer must use the same kernels as the Gaussian lane. A distinct risk is data provenance: the nonlinear reference uses physical FP64 observations while GPU filtering uses their FP32 cast. Record both identities and validate fitting against the exact executed observations. Do not equate the two checksums or change the reference dataset silently.

## Evidence contract and research intent

| Role | Requirement |
|---|---|
| Mechanism | Shared Gaussian-plus-floor sampler/normalizer and bounded density fitter with nonlinear conditional means |
| Baseline | Frozen Gaussian implementation `1c12eefa`; zero-curvature parity; independently refined nonlinear grid reference |
| Engineering pass | Checked density identity and frozen-fit derivatives; zero-curve parity; real selection-to-final consumer on CPU and GPU; all fit/count/data/stream diagnostics valid |
| Scientific promotion | None in this slice; no quality default changes |
| Promotion veto | Invalid fit, grid refinement failure, scope mismatch, omitted offline cost, or observed underperformance against any applicable cheap heuristic |
| Continuation veto | Broken mathematical identity, source drift, incorrect score, unsupported backend, corrupt evidence or exhausted allocation |
| Repair trigger | Local numerical, fitting, provenance or harness failure; preserve failed attempts and repair within budget |
| Explanatory only | Fit objective/residual, optimizer boundary status, count trajectory, ESS, descriptive oracle error and compilation-inclusive timings |
| Not concluded | Superiority, unbiased score, fully tuned controls, high-dimensional generality, degenerate-transition applicability or matched-cost performance |

Conditional situations are weak curvature (c=0.12,b=0.04) and stronger curvature (c=0.35,b=0.12), at T=2 and initial N=16. Construct four heuristic adversaries for each: EKF uses inexpensive local derivatives; UKF estimates nonlinear Gaussian moments; bootstrap PF samples the physical transition; local-linear adapted PF uses the observation to choose a cheap Gaussian proposal. Evaluate each on the same final dataset as iAPF and report oracle score error. These are falsification checks, never tuning inputs. Their small sample sizes support descriptive promotion vetoes only, without candidate rankings. Gaussian zero-curvature checks use Kalman as an exact reference. The log-quadratic fitter receives nonlinear wiring and law/target tests; its positive-precision restriction may reject a nonlinear fit and cannot be silently ridged away.

## Default and assumption audit

| Choice and provenance | Justification and status | Failure mode and earliest diagnostic |
|---|---|---|
| Same six-parameter fixture, new fixed c,b values | Explicit curvature mechanics hypotheses; not representative target calibration | Reference domain misses mass; mesh/domain refinement and tail checks on every dataset |
| Additive Gaussian noise and Gaussian-plus-floor twist | Exact conditional convolution above | Accidental linearization; nonlinear normalizer integration and direct mixture-density identity |
| N=16, T=2; nominal theta `[.62,-.8,-.6,.9,.25,-.3]` from preceding mechanics slice | Small call-chain test, not accuracy evidence | Small clouds cause optimizer or resampling pathology; expose per-fit convergence, bounds and finite values |
| k in {1,2}, tau=100, iteration caps 4/5, maximum N=128 | Two inherited mechanics candidates; deliberately permissive stopping, not a calibrated default | Early stopping gives poor twists; report objective, iterations, all heldout errors, retain later full calibration |
| Mean bound 4, relative sd bounds [0.2,4], floor ratio 0.01; 2000 optimizer steps, 30 backtracks, tolerance 1e-7 | Explicit inherited bounded-family hypotheses; positive floor ensures full support; box constraints make optimization compact | Boundary solution or nonconvergence; recorded projected gradient, boundary status and cast margins veto invalid fits |
| Offline FP64, final FP32/TF32/XLA | Previous FP32 fit failed its declared tolerance; explicit higher-precision fitting, no automatic fallback | Cast changes or invalid coefficients; compare stored coefficients and require positive finite covariance |
| Physical FP64 data and CPU FP64 refined reference | Existing nonlinear reference design, not GPU timing evidence | Cast identity conflation; record physical and executed-data digests separately and compare same physical observations |

The numerical protections are explicit properties of the candidate family. No ridge, clipping, damping or new precision fallback is introduced. Their scope-specific scientific calibration remains open even if mechanics passes.

## Implementation and checks

Create a new mutable worktree from `1c12eefa`. Add shared conditional-mean value/tangent helpers used by the filter and both recursive fitters, with zero curvature as the exact preceding path. Pass explicit curvature settings through both adapters; admit nonlinear iAPF only through its selected procedure and the actual nonlinear consumer. Record exact executed observations and model coefficients in fitting evidence, while retaining physical-data provenance. Reject missing/mismatched curvature or fitting data.

Focused tests must check: nonlinear mixture normalization and density correction on an independently integrated one-dimensional fixture; full six-parameter directional finite differences with the same random inputs and frozen fit; zero-curvature value/score/fit parity; actual endpoints call the shared filter/fitter; wrong scope and cast-data failures; CPU selection and independent final replicates. Autodiff or NumPy, if used by a reference test, stays explicitly diagnostic. Runtime kernels use TensorFlow, stable signatures and native loops without pfor. Freeze the source only after these checks; never edit a snapshot after running it.

Then execute at most twenty numerical GPU rows: per curvature, four calibration/validation rows for the two procedures, two independent final replicates using selected controls, and four heuristic rows. Use disjoint datasets 900/910/920 (calibration/validation/final), seed 941; the fixed curvature is part of model scope. Select on calibration model-score error, use validation only as a validity check, and never use heuristic results to tune. Every recursive fit consumes an attempt. Controls are mechanics candidates and may not be promoted from this tiny selection.

## Execution, budget and stop conditions

Allocation within the existing 12 CPU process-hour / 8 GPU-hour master envelope: at most 90 CPU process-minutes, 30 GPU wall-minutes, three GPU launches, and 120 numerical row/recursive-fit attempts. Reconcile cumulative measured timings before GPU execution. The preceding conservative GPU ledger was 4433.874 seconds plus 52.191 seconds for the adaptive-scope slice; adding 1800 seconds remains below 28800 seconds. Every failed attempt consumes budget. No unchanged retry after three failures of one cause.

Use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, trusted RTX4080 SUPER access, verified memory growth and FP32/TF32/XLA filtering. Set `TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/younis-score-mpl`. CPU reference/test commands explicitly set `CUDA_VISIBLE_DEVICES=-1` before import. Probe trusted GPU load before launch; do not make comparative runtime claims under concurrent load. No package or environment changes.

Versioned output root: `docs/plans/artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/nonlinear-iapf-*`. Preserve full logs, source/driver checksums, commands, seeds, model/data identities, work counts, device/memory/XLA evidence and actual timing. The exact frozen command will be appended before launch. Fail closed before final samples if selection or the reference is invalid.

Pre-mortem: successful execution could conceal a linearized transition, linear observation fit target, wrong data digest, partial derivative, omitted fit cost or unexamined inherited tuning. The listed law, tangent, call-chain and provenance checks discriminate these before a comparison. A valid but inaccurate fitted proposal triggers later target-specific calibration, not abandonment of iAPF. At phase exit integrate checked files, record decision/inference tables, repair what is repairable and refresh the next master phase. Comprehensive control calibration and replicated nonlinear proposal comparisons remain the next scientific obligations.
