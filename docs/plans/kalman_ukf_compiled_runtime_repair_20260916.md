# Kalman and UKF compiled runtime repair

Owner request: 2026-09-16. Baseline: Git `21828174`, initially clean.
Input: MacroFinance-dz5-neutra `docs/plans/dz5_bayesfilter_rectangular_srukf_runtime_handoff_20260916.md`.

## Question and scope

Can the repository's active TensorFlow Kalman, UKF, square-root and related
sigma-point value/analytical derivative routes execute complete numerical
recurrences under XLA, without Python iteration over dates, parameters, chains,
points, or matrix coordinates? Audit public dispatch, numerical dependencies,
model callbacks, and the native HMC transition archive. Configuration, static
shape validation, host artifact I/O, and explicitly independent reference code
are separate roles; they must not hide a numerical runtime fallback.

Repair shared factor derivatives with tensor-batched algebra. Express genuinely
sequential numerical recurrences using TensorFlow control flow with fixed-shape
state. Preserve first and second analytical derivatives, QR sign/orientation,
fixed rank/pivot/support decisions, dtype, and existing likelihood measure.
Do not introduce jitter, change the UKF point rule, replace scores by autodiff,
use pfor, or silently fall back when compilation fails. Fix active callers as
well as helpers, and retain a discoverable audit/regression guard.

## Evidence contract and assumptions

- Engineering gates: inventory every relevant module and loop; no numerical
  Python loops in repaired runtime closure; no PyFunc/EagerPyFunc; bounded graph
  size with date/parameter counts; successful compilation of the actual target
  and complete HMC numerical archive block. A JIT flag alone is insufficient.
- Numerical gates: existing independent reference tests, new non-diagonal,
  off-origin multi-parameter and multi-chain finite differences, factor and
  moment parity, analytical Hessian checks where exposed, chain isolation,
  singular support and invalid-branch telemetry. Baseline parity alone is not
  an oracle. Use existing fixture tolerances and float64 QR differences at
  approximately 1e-5 relative / 1e-7 absolute (tighten for exact identities).
- Graph and CPU/XLA checks are explicit engineering/reference diagnostics.
  GPU qualification uses the existing tf-gpu environment, trusted access,
  TF_FORCE_GPU_ALLOW_GROWTH=true and verified memory growth before initialization.
  Record source SHA-256, exact command, hardware/environment, seed, timing,
  compilation/HLO evidence and artifact paths in versioned output directories.
- Toy HMC uses fixed seeds and short transitions solely to test composition,
  transport/mass, leapfrog, acceptance and numerical trace buffers; no tuning,
  retained DZ5 sampling, convergence or posterior claim is authorized here.
- Full-rank residual stacks and rectangular carried factors are distinct. Test
  their full-rank specialization on matched point orientation before advising
  integration. A different approximate target needs fresh consumer lineage.
- The saved R5 gamma-family synchronization diagnosis is historical explanatory
  context, not a baseline for new filter or HMC performance claims. Generic
  callback reuse may be repaired; CIR model changes belong to MacroFinance.

## Bounded execution and stop conditions

Routine editing/static checks and focused unit diagnostics proceed under the
owner request. Qualification budget: at most 90 aggregate CPU minutes and 20
aggregate GPU minutes, no training or long sampling; each test/diagnostic command
has a timeout of at most 10 minutes. At most three attempts per qualification
fixture before recording a blocker and choosing a smaller discriminating check.
Use `docs/plans/artifacts/kalman-ukf-runtime-20260916/<unique-attempt>/` for durable
qualification and `/tmp` for disposable logs. Never overwrite prior evidence.
Stop a run on nonfinite valid-branch results, changed target semantics, lost
support telemetry, uncontrolled memory, or budget exhaustion. An implementation
failure triggers repair within the unchanged contract, not scientific rejection.

## Skeptical pre-execution audit

Passed with limits: static Python loops may unroll during tracing rather than
execute per evaluation, so their removal is an engineering claim only. Date
recurrences cannot be parallelized. Reference loops must remain distinguishable
from runtime. XLA may ignore assertions, so test returned invalid-state signals.
Do not reuse pre-reset LEDH evidence. The baseline is current source, not an old
dirty snapshot. Numerical parity, compiled closure and independent differences
jointly answer the question; neither faster toy timings nor successful HMC
mechanics establish DZ5 scientific validity. Full DZ5 qualification and hardware
selection remain MacroFinance deliverables.

Mathematical anchors: `docs/chapters/ch12_factor_derivatives.tex`, equations
`eq:bf-factor-qr-r-first`, `eq:bf-factor-qr-q-first` and their second derivatives;
`ch17_square_root_sigma_point.tex` direct block-QR and likelihood/score equations.
MathDev CLI doctor succeeded; whole-doc extraction hit an unrelated malformed
LaTeX display. Narrow extraction and direct source inspection preserve the
equation anchors without changing the scientific method.

## Completion and recovery

The repair and its bounded engineering qualification are complete. See
[the result](kalman_ukf_compiled_runtime_result_20260916.md) for the audit scope,
durable tests, final CPU/GPU XLA receipts, failed attempts and remaining limits,
and [the MacroFinance handoff](kalman_ukf_macrofinance_handoff_20260916.md) for
the exact consumer contract. Machine/VS Code interruptions preserved the source;
temporary pre-crash logs were lost. Final-source qualification artifacts are
`cpu-v5` and `gpu-v4`, and the final audit is `audit-v3`. Earlier artifacts remain
preserved. No DZ5 tuning, retained sampling, device selection or scientific
promotion was performed.
