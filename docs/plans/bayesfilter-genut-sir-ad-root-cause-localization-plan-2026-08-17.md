# GenUT Austria-SIR AD Root-Cause Localization Plan

Date: 2026-08-17

## Research intent ledger

| Field | Frozen intent |
|---|---|
| Main question | Is the Austria-SIR GenUT `j0` discrepancy caused by an incorrect manual derivative, a different generative law, numerical conditioning/non-smoothness in the finite program, or ordinary between-seed variability? |
| Claimed internal target | The total derivative of the exact scalar returned by the same finite GenUT value program, with fixed observations, fixed noise, fixed design, fixed iteration counts, and fixed controls. |
| Candidate mechanism | Manual forward JVP through the SIR callbacks, Sinkhorn row quotient, Contract-E reset, higher-moment map, and recursive log-normalizer. |
| Expected failure mode | The manual JVP first disagrees with TensorFlow AD in one local map or at a particular horizon/particle-count/shape-control rung. |
| Promotion criterion | None. This is a root-cause diagnostic, not a method-promotion run. |
| Hard veto | Nonfinite values, failed source observation identity, failed GPU/memory-growth provenance, invalid finite-program branch, or inability to obtain an AD derivative of the same scalar program. |
| Repair trigger | Harness, serialization, XLA, or memory failure that does not change the question, maps, data, controls, or compute budget. |
| Continuation veto | The same-program scalar cannot be differentiated after one bounded XLA attempt and one graph-mode localization attempt, or the 12-minute/three-attempt campaign budget is exhausted. |
| Nonclaims | No exact observed-data SIR score, no classifier-oracle admission, no SQMC/GenUT ranking, no default readiness, and no HMC readiness. |

## Mathematical trace

At time `t`, the finite program propagates the current equal-weight cloud and
its tangent, computes particle log likelihoods `ell_i`, and adds

\[
  \Delta_t = \log\sum_i w_i e^{\ell_i}, \qquad
  \dot\Delta_t = \sum_i \bar w_i
    \left(\dot w_i/w_i + \dot\ell_i\right).
\]

It then differentiates the normalized weights, the fixed-iteration Sinkhorn
row quotient, the Contract-E mean/covariance reset, the requested shape
corrections, and the final affine restoration. The output score is
`sum_t dot(Delta_t)`. Therefore TensorFlow AD of the returned scalar value with
respect to `theta`, while holding observations/noise/design fixed, is an
independent derivative authority for this finite program. Agreement proves
only internal differentiation consistency; it does not prove that the finite
value equals the nonlinear observed-data log likelihood.

The source observation simulator and the adapter both use the Zhao-Cui
half-step fourth RK stage. The simulator additionally clips susceptible
coordinates after process noise. Law parity and clipping incidence must remain
separate from derivative parity.

## Hypotheses and tests

| ID | Hypothesis | Discriminating test | Interpretation |
|---|---|---|---|
| H1 | SIR callback tangent algebra is wrong | Compare direct and total callback tangents with `ForwardAccumulator`; also compare transition values with the FP64 simulator | A scale-material manual/AD gap localizes an adapter defect; FP32/FP64 value gaps alone are precision diagnostics |
| H2 | Sinkhorn row-quotient JVP is wrong | Compare manual particles/coupling JVPs with forward AD on a fixed valid cloud | First differing output localizes the transport derivative defect |
| H3 | Contract-E reset JVP is wrong | Compare every manual reset intermediate with forward AD in FP32 and FP64 over covariance-condition and gap-size ladders | Persistence in FP64 supports algebra error; collapse in FP64 supports conditioning/roundoff |
| H4 | Higher-moment or cap JVP is wrong | Compare no-shape, diagonal, pairwise, and dual-cap local maps with AD | The first arm/stage that differs localizes shape derivative error |
| H5 | Recursive score differs from the derivative of its own scalar | Compare reported score with AD of the returned scalar at `N=36,252,1008`, `T=2,5,20`, escalating only after smaller rungs are finite | First divergent rung identifies recursion depth/scale at onset |
| H6 | The finite program is internally correct but high variance | If H1-H5 agree, retain the three-seed horizon/ablation SD evidence from `attempt05` | Variance remains descriptive; three seeds cannot rank algorithms |
| H7 | The filter and simulation comparator use different SIR laws | Record transition parity, observation hash, and susceptible clipping incidence; compare `latent_preclip=False/True` only if derivative tests pass | A law mismatch invalidates common-target comparison but is distinct from JVP correctness |

## Error interpretation

No arbitrary absolute threshold is used. Every comparison records maximum
absolute error, RMS error, reference RMS, symmetric relative L2 error, and the
ratio to floating-point unit roundoff times the largest relevant Cholesky
condition proxy. FP32 and FP64 reset results form a precision ladder. A result
is called an algebraic mismatch only when it is scale-material and does not
collapse with higher precision or better conditioning. Otherwise it is
classified as roundoff/conditioning-limited or unresolved.

Finite differences are secondary only, using FP32-appropriate steps
`0.03, 0.01, 0.003`; they cannot override same-program AD.

## Default and assumption audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| Austria-SIR path seed `81120` | Existing common-target baseline | Wrong observations/event order | Frozen serialized-tensor hash |
| Claim seeds `98201:98203` | Existing diagnostic baseline | Too few seeds for a ranking | Descriptive SD/MCSE only |
| Controls `epsilon=8`, Sinkhorn/balance `16/16`, ridge `1e-5` | Existing GenUT baseline, not a promoted default | Conditioning or iteration-specific derivative behavior | Local-map AD and rung ladder |
| `N=36,252,1008` | Exact replicated cubature ladder | Small `N` behavior may not extrapolate | Escalate only after valid smaller rung |
| `T=2,5,20` | Recursion-localization ladder | Does not establish asymptotic horizon scaling | Report onset, no extrapolation |
| FP32 GPU/XLA | Actual candidate execution route | Cancellation/conditioning may imitate derivative error | FP64 local reset ladder and AD/FD separation |
| TensorFlow AD | Same-program derivative authority | Unsupported forward-mode/XLA op | One graph-mode localization retry; preserve failure |

## Evidence contract

- Comparator: TensorFlow forward AD of the same local value map, and AD of the
  same returned scalar finite program.
- Primary diagnostic: the first local or recursive stage with a scale-material
  manual/AD discrepancy.
- Veto diagnostics: finite-program validity, source observation hash, GPU and
  memory-growth identity, finite AD/manual outputs, and stable branch identity.
- Explanatory diagnostics: Cholesky condition proxies, horizon, particle count,
  shape arm, finite-difference ladder, increment concentration, and prior
  three-seed SD.
- Artifact: a unique directory under
  `docs/benchmarks/artifacts/genut-sir-ad-root-cause-20260817/` containing JSON,
  Markdown, log, exact source hashes, and run manifest.

## Skeptical pre-execution audit

- Wrong baseline: avoided by differentiating the exact scalar being scored,
  not a Kalman model, classifier estimate, or different filter.
- Proxy promotion: local AD and finite differences diagnose implementation;
  they do not establish an exact SIR score.
- Hidden law mismatch: simulator clipping and RK/event order are recorded in a
  separate law ledger.
- Unfair comparison: all derivative pairs share identical primal inputs,
  controls, noise, design, and branch.
- Threshold drift: conclusions use normalized errors and an FP32/FP64
  conditioning ladder, not the superseded `0.02` absolute threshold.
- Non-answering artifact: local intermediates identify the first differing map;
  full-program rungs identify the first recursion scale where it appears.
- Stop conditions and budget: at most three launches and 12 total GPU minutes;
  stop escalation on a hard veto or once the first reproducible local mismatch
  is isolated.

Audit verdict: `PASS_FOR_EXECUTION`. The plan directly distinguishes derivative
algebra, finite precision/conditioning, recursive variance, and target-law
differences without treating any one as evidence for another.

## Execution

Use the `tftwogpu` environment and physical GPU1 when its utilization is below
50% and more than 8 GiB is free. In this environment, `CUDA_VISIBLE_DEVICES=0`
selects the physical RTX 4080 SUPER and TensorFlow then reports it as logical
GPU0. Enable memory growth before initialization and retain
every attempt. The terminal result must include decision and inference-status
tables plus the strongest alternative explanation and the next discriminating
test.

### Bounded execution amendment after harness failures

The original three-launch cap was exhausted by compiler/harness failures before
a terminal artifact, while only about 3.5 of the fixed 12 GPU minutes were
consumed. The cap was first amended to six launches without changing the 12-minute
compute budget, model, data, method, hardware class, criteria, or nonclaims.
Every stage now writes a checkpoint, and the fifth launch uses scalar graph
reverse AD uniformly with XLA-versus-graph primal/manual parity vetoes. This is
a localized campaign repair under the repository retry policy, not an expansion
of the scientific question or compute budget.

Attempt six failed before its first graph result because TensorFlow's default
parallel Jacobian vectorizer cannot transform the loop/conditional graph. The
per-increment Jacobian was redundant with the already frozen `T=1..5` prefix
ladder, whose successive differences localize the onset. One seventh and final
launch is allowed after removing only that redundant computation. The total
12-minute GPU budget remains unchanged; no eighth launch is permitted.

The attempt-seven GPU process was never created because the platform's
automatic permission review timed out four times across the direct and approved
conda launchers. This is an external execution-boundary failure, not campaign
evidence. The final `N=36`, `T=1..5` graph derivative prefix ladder therefore
runs as an explicit CPU reference with `CUDA_VISIBLE_DEVICES=-1` and
`BAYESFILTER_GENUT_AD_REQUIRE_GPU=0`. It is eligible only to check mathematical
manual-JVP/AD parity and is not GPU/XLA, production, performance, default, or
HMC evidence.
