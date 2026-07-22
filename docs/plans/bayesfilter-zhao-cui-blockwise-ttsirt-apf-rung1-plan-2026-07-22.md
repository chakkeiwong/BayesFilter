# Zhao-Cui Blockwise TTSIRT-APF Rung-1 Plan

Date: 2026-07-22

Status: `COMPLETED_PASS_ENGINEERING_RUNG1`

## Research Intent

| Field | Contract |
| --- | --- |
| Main question | Can an actually fitted squared-TT and its conditional KR map supply a valid, non-collapsed proposal to the certified fixed-branch APF when replicated across 24 independent state blocks? |
| Mechanism | Fit scalar initial and bivariate adjacent Gaussian targets in algebraic reference coordinates; compile `(x_previous,x_current)` prefix-conditioned TTSIRT blocks; combine 24 blocks under one ancestor genealogy. |
| Expected failure mode | Poor density fit or numerical grid inverse compounds across 24 blocks, causing weight collapse despite correct APF plumbing. |
| Promotion criterion | Finite fitted transports and branch; reference-target quadrature mass error `<=0.03`; paired-core conditional formula tie-out `<=1e-10` in float64; inverse/CDF roundtrip error `<=1e-4`; same-scalar analytical score/FD max error `<=0.03`; minimum ESS fraction `>=0.5` at `d=24,T=3,N=256`; no support or measure mismatch. |
| Promotion veto | Bounded physical support, zero defensive mass, non-finite fit/map/value/score, wrong ancestor law, tensor-product suffix-grid conditional density, score/FD failure, or minimum ESS fraction below 0.5. |
| Continuation veto | The fitted TTSIRT cannot define a finite full-support conditional proposal even on the scalar Gaussian block. A failed 24D candidate alone is a repair trigger, not a direction veto. |
| Repair trigger | Candidate failure triggers a fresh degree/rank/scale/defensive-mass tuning scope or a larger structural block, within the attempt budget. |
| Explanatory diagnostics | Calibration and holdout sqrt-density residuals, ranks, conditional log-density error, ESS, log-weight spread, value/score difference from Kalman, compile and warmed time. |
| Must not be concluded | No source-faithful variable ordering or block factorization, Austria SIR, NAWM, nonlinear validity, HMC convergence, default readiness, or superiority. |

## Evidence Contract

The baseline ladder is evaluated at the same `d=24,T=3,N=256` scope: (1) an
exact fully adapted diagonal-Gaussian proposal with predictive auxiliary
probabilities, (2) the same exact conditional proposal with uniform auxiliary
probabilities, and (3) the fitted TTSIRT proposal with uniform auxiliary
probabilities. Arm 1 is an oracle ceiling; arm 2 isolates auxiliary-law loss;
arm 3 is the candidate. The prior rung-0 artifact remains engineering
provenance but is not a matched numerical comparator because it used
`T=10,N=1024`. The primary criterion is candidate downstream APF ESS plus
same-scalar analytical score identity. Fit residuals, cross-arm differences,
and Kalman differences are explanatory; they cannot replace the downstream
criterion or support a stochastic ranking. The artifact root is
`docs/benchmarks/artifacts/zhao_cui_blockwise_ttsirt_apf_rung1_20260722/`.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Independent scalar blocks | synthetic diagonal LGSSM oracle | Isolates fitted TT/KR mechanics at `d=24` | hides cross-coordinate rank needs | explicit nonclaim and later coupled fixture | diagnostic hypothesis |
| `(previous,current)` order | local prefix-conditioned TTSIRT API | Makes conditional suffix generation exact via Proposition-2 prefix marginal | differs from Zhao-Cui paper order | axis-order manifest and conditional-density tieout | `extension_or_invention` |
| Algebraic coordinate map | author `AlgebraicMapping` formula and local P85/P86 checks | Full physical support with finite reference domain | tail distortion or boundary instability | roundtrip/Jacobian tests and finite inverse | reviewed formula, scale to tune |
| Legendre degree/rank | local clean-room fitter | Small bounded candidate grid | underfit or ill-conditioning | disjoint holdout residual and condition veto | hypotheses, not defaults |
| Positive defensive mass | Zhao-Cui Eq. (13) | Enforces support | excessive defensive mass degrades ESS | candidate grid and downstream ESS | hypothesis |
| Grid CDF/bisection | local P83 diagnostic transport | Only implemented fixed TTSIRT inverse locally | numerical density/inverse mismatch | conditional log-density and roundtrip checks | `extension_or_invention`, diagnostic-only, nonproduction |
| Calibration design | 25-point Gauss--Legendre rule per active axis | Common deterministic design integrates the degree-10 fit objective without changing the sample set between candidates | quadrature misses algebraic-tail error | disjoint 32-point midpoint holdout per active axis | reviewed diagnostic choice |
| ALS controls | ridge `1e-10`; two sweeps; scalar order `(0,)`; adjacent order `(0,1,1,0)` | Matches the stabilized local clean-room fitter and stays inside the bounded CPU budget | under-convergence or ridge bias | per-update status, condition number, calibration/holdout residual | convenience hypotheses |
| KR numerical controls | grid size `129`; 24 bisection steps | Bounded first test of the existing P83 inverse | inverse bias compounds over 24 blocks | conditional density tieout and inverse/forward roundtrip | convenience hypothesis, nonproduction |
| Warmed repeatability | absolute repeated-XLA value difference `<=1e-5` | Float32 reductions need a numerical tolerance rather than bitwise identity | nondeterministic or unstable compiled scalar | two consecutive evaluations of each frozen arm | engineering gate |
| Auxiliary law | uniform, parameter-independent categorical law | Isolates proposal density and shared-genealogy mechanics before adding a predictive auxiliary compiler | high-dimensional predictive-weight variance can collapse ESS even with a good conditional proposal | downstream ESS and log-weight spread | deliberate baseline hypothesis |
| `d=24,T=3,N=256` | NAWM observable/shock count plus bounded mechanics budget | High-dimensional composition test without NAWM claim | too easy/short for general filtering | explicit scope and later ladder | convenience diagnostic |

## Skeptical Audit

The first audit incorrectly treated the differently scoped rung-0 oracle as a
fair comparator. The repaired ladder evaluates exact predictive-auxiliary and
exact uniform-auxiliary arms at the candidate's own scope, so TT approximation
loss is not confounded with auxiliary-law loss. Fit loss is not promoted to
the downstream criterion. The algebraic map prevents the bounded-support
error. The compiler uses paired-core prefix marginals, not the historical
suffix tensor grid. The 24D product target is intentionally easy and cannot
establish nonlinear or cross-coordinate scalability. The repaired audit
passes: the command and artifact answer only whether fitted scalar TTSIRT
mechanics survive 24-fold composition and how the two declared mechanisms
contribute descriptively.

The audit passes bounded diagnostic execution.

## Tuning And Holdout

Calibration candidates are the Cartesian product:

- degree in `{6, 10}`;
- adjacent rank in `{2, 4}`;
- algebraic scale in `{1.5, 2.5}`;
- defensive mass in `{1e-6, 1e-4}`.

Use a common deterministic 25-point Gauss--Legendre calibration rule per active
axis and a disjoint 32-point midpoint holdout per active axis. Fit `h` to the
square root of the exactly normalized target density relative to the uniform
reference measure. Candidate selection uses the heldout relative RMS between
the square roots of the full normalized proposal `h^2 + tau` and that target;
the raw fitter residual is explanatory only. This makes defensive mass an
actual evaluated candidate choice. Select by finite status then minimum of the
maximum heldout relative sqrt-density RMS across the initial and two adjacent
targets; ties within `1e-6` prefer lower degree, rank, and defensive mass.
Freeze the selected controls before a fresh stateless proposal branch with seed
`220723`.

The initial target is the exact scalar posterior for the first observation.
Each adjacent target, in compiler order `(x_previous,x_current)`, is the exact
normalized density

`p(x_previous | y_0:t-1) f(x_current | x_previous) g(y_t | x_current) / p(y_t | y_0:t-1)`.

Therefore its suffix conditional is the fully adapted scalar proposal. The
reference-measure fit target is exactly
`log target_physical + log|dx/dz| - log(reference_density)`. The fitted target,
normalizer, coordinate Jacobian, paired-core conditional density, and APF
importance correction are kept as separate checks.

The local grid inverse remains a numerical approximation to the fitted
TTSIRT map. Passing its roundtrip gate does not establish an exact randomized
proposal law or likelihood-estimator unbiasedness. This rung may establish only
the deterministic fixed-branch finite scalar and its same-program score.

These settings are scoped only to the synthetic diagonal model, `T=3`, scalar
blocks replicated to `d=24`, float64 offline fit, float32 GPU/XLA online
evaluation, and `N=256`. They are not transferable defaults.

## Budget And Stop Conditions

- At most 16 calibration candidates and 2 execution attempts.
- CPU fitting budget: 5 minutes total.
- Trusted GPU/XLA claim-branch budget: 2 minutes.
- Fresh versioned directory per attempt; preserve failures.
- Stop on non-finite target/fit/map, condition-number veto, missing defensive
  mass, same-scalar score failure, or exhausted budget.
- If ESS fails but the scalar block is valid, record candidate rejection and
  design the next rank/scale/block repair; do not reject the research direction.

### Authorized Continuation, 2026-07-22

The user authorized a repaired continuation after the two harness-failure
attempts. This adds exactly one canonical CPU precheck and one canonical
trusted GPU/XLA claim attempt without changing the target, data, candidate
grid, scope, promotion criteria, vetoes, seed, dtype, or hardware class.

The GPU is shared across several agents. The continuation therefore requires:

- `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import;
- repository memory-growth configuration and verification on every visible
  physical GPU before logical-device initialization;
- no fixed logical-device memory limit and no whole-device preallocation;
- a read-only `nvidia-smi` occupancy snapshot before launch;
- defer launch while the device is materially busy rather than competing with
  another active workload;
- record TensorFlow allocator current/peak bytes and the shared-device trust
  basis in the result manifest; and
- keep offline TT fitting and proposal compilation on CPU, using GPU only for
  the float32 XLA online APF arms.

Initial shared-device snapshot before the CPU precheck: RTX 4080 SUPER,
`16376 MiB` total, `2933 MiB` used, `13113 MiB` free, and `74%` utilization.
The GPU claim launch was deferred at that point.

## Executed Command

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONDONTWRITEBYTECODE=1 \
python docs/benchmarks/run_zhao_cui_blockwise_ttsirt_apf_rung1.py \
  --output-root docs/benchmarks/artifacts/zhao_cui_blockwise_ttsirt_apf_rung1_20260722/gpu_attempt01 \
  --dimension 24 --time-steps 3 --particle-count 256 --seed 220723
```

The canonical CPU precheck emitted `PASS_CPU_REFERENCE_PRECHECK`. After the
shared-device utilization fell from the initial `74%` to a stable `34-39%`
sample with `13510 MiB` free, the trusted GPU command emitted
`PASS_ENGINEERING_RUNG1`. The terminal interpretation is recorded in
`docs/plans/bayesfilter-zhao-cui-blockwise-ttsirt-apf-rung1-result-2026-07-22.md`.
