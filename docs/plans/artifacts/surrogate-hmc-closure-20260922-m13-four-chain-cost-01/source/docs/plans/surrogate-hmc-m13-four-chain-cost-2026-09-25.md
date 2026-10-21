# M13 four-chain public-kernel connection and cost

The question is whether the canonical analytical DCTR force can drive the
public tuner's four-chain TensorFlow kernel with the independently evaluated
endpoint, and what complete transitions cost. This is the next diagnostic
inside the surrogate-HMC closure master. It cannot issue a tuning artifact or
establish posterior convergence. LG1 remains closed; Claude review is deferred.

## Evidence contract and skeptical audit

Use data1704/T20, state dimension4, observation dimension2, parameters
lambdaQ/lambdaR with independent N(0,log(2)^2) priors. The force remains the
canonical native batch implementation, N4096, design20260923/partition9, FP32,
TF32, GPU/XLA and the previously declared unselected geometric .01 protection
candidate. Scope-specific control selection is still required before admission.
The endpoint is the same frozen full-support trajectory-importance construction,
4096 particles x8 replicates, seed(2026092402,9101), plus the prior. Its FP64
evaluation is compared directly with the scalar reference at each starting row.
The integrator/endpoint use FP64 as required by the existing public tuner;
expensive force calculations stay FP32. This removes the mechanics04 final
potential rounding and is explicitly a new interface precision scope.

The primary pass is executable wiring through
bind_neural_force_hmc_tuning_runner -> build_tensor_kernel -> the same native
compiled four-chain loop used by the public tuner, with valid finite endpoints,
force status, movement in all chains, and no divergence. Direct endpoint replay
and scalar parity must pass. Synthetic nonidentity-affine tests check the force
sign, coordinate transform and endpoint Jacobian convention. Scalar canonical
force versus native batch force is recorded as an explanatory FP32 diagnostic;
invalid native force or invalid endpoint is a hard harness veto. A nonzero
finite force discrepancy is reported, not converted into posterior harm.

Timing, acceptance and energy error are explanatory, not convergence evidence.
No R-hat is interpreted from the 8-step chunks. No statistical ranking, tuning
admission, all-model conclusion, global tail bound or posterior correctness
claim follows. Saved endpoint/grid/domain results support this diagnostic;
they do not make this force the gradient of the independent endpoint.

The cheap adversaries are the direct scalar endpoint (detects vectorization
mistakes), scalar canonical force (detects a native batch fork), and analytic
Gaussian/non-gradient synthetic bindings (detect coordinate/sign errors).
Evaluate all four starting quadrants separately. They are harness comparators;
posterior-method heuristic dominance remains untested by this diagnostic.

Skeptical audit: public capabilities identify tune_hmc_kernel as the authority
for this typed position-only field; this cost stage calls its binding/kernel,
not an authority-bearing tuner. No proxy test promotes the sampler. Distinct
starts expose row mixing. Fixed randomness preserves a deterministic endpoint.
Unselected controls and the short trajectory cannot establish efficiency. A
cost failure triggers a localized repair within remaining budget, not rejection
of surrogate HMC. Compilation/resource overrun stops the worker and is charged.

## Defaults, assumptions and pre-mortem

| Choice | Provenance and role | Risk and early check |
|---|---|---|
| FP32 canonical force, .01 protections, N4096 | mechanics04 warm-start candidate | Native batch numerical differences: scalar parity/status at all starts |
| FP64 coordinates/endpoint | public tuner contract; explicit reference precision exception | Accidental force promotion to FP64: dtype/wiring test and manifest |
| Identity affine geometry, step .002 | mechanics-only cost baseline, not tuned geometry | Poor exploration: no efficiency claim; later public adaptation required |
| Four starts (+/- .25,+/- .25) | bounded quadrants for wiring, not convergence initialization | Locality: later public verification needs independent dispersed starts |
| L=2 and4, eight steps each | cost slope probes, not selected trajectory length | Tiny measurements: report individual timings and conservative forecast |
| Global zero fallback on invalid force | existing deterministic proposal policy | Record fallback counts; any fallback prevents this connection screen passing |
| No force modifications during run | frozen proposal requirement | Hash numerical source closure, preserve before/after identity |

Pre-mortem: high acceptance at tiny steps can look convincing while chains barely
move; therefore there is no convergence or efficiency conclusion. Whole-loop
compilation may dominate and is separated from repeated calls. A scalar force
loop might appear batch-compatible; the implementation must call compile(4)
on the canonical target, checked both by unit wiring and actual native output.

## Budget, command and artifacts

At entry cumulative charge is 51745.458323292 CPU worker seconds and
40914.26400198684 GPU seconds. Remaining 2254.541676708 CPU and
185885.73599801316 GPU under the unchanged 54000/226800 ceilings. Reserve
1000 CPU/1400 GPU for this pass (worker850 CPU/1300 wall, review120 CPU/wall).
No budget increase is authorized. Abort rather than consume outside the cap.

Use the tftwogpu environment, escalated GPU access, memory growth, versioned
output, per-chunk serialization, git/source hashes and the existing master
accounting. Exact command:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-four-chain-cost --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-four-chain-cost-01
```

The worker records scope, binding identity, dtype, scalar/batch parity, separate
compile and repeat timings, raw four-chain trajectories and health diagnostics,
device/memory policy, and a conservative transition-cost forecast. Result note:
docs/plans/surrogate-hmc-m13-four-chain-cost-result-2026-09-25.md. If remaining
authorized time cannot cover scoped control selection, public tuning and
retained posterior comparison, finish their concrete plan and request the
additional measured budget at that real boundary.
