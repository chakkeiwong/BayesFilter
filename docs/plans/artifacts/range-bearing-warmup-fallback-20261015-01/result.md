# Range-bearing warmup failure: a nearly singular Euler flow proposal

Completed 2026-10-15 UTC (2026-10-16 Hong Kong). This is a bounded debugging
result within the five-model surrogate-HMC campaign, not a new posterior run.
The baseline is the actual literature-prior range-bearing attempt: T20,
four-dimensional state, two observations and two parameters, N4032/K2016,
FP32, TF32 off, eight flow steps and the frozen selected controls. The plan is
`docs/plans/surrogate-hmc-range-bearing-fallback-repair-2026-10-15.md`.

## What failed

The verified archive contains 250 warmup transitions per chain and four chains.
Two transitions in the first chain flag a surrogate-force fallback; both were
accepted. Acceptance is 95.2–97.2% across chains, with zero divergences. Endpoint
validity, finiteness, movement and the energy screen pass. No retained samples
exist. The failed screen is `no_force_fallback`, not R-hat or acceptance.

Transitions 20 and 21 share the position
q=(1.1239974267148416, -0.2243877888489681). The L=1 kernel caches the force
at an accepted endpoint and uses it at the start of the next proposal. Thus
these two flags are consistent with one invalid force position counted twice;
they are not evidence for two independent numerical failures. The saved input
before that position is healthy in replay.

The original FP32 full-filter replay reproduces the failure at observation 14.
The sole failed validity field is `proposal_maps`. At this observation the
particle ESS is 3846.73/4032, covariance-reset relative error is 1.064e-6,
and Sinkhorn row/column errors are 8.73e-10 and 1.19e-6. These checks pass.
The prior and Gaussian-guide proposal components are healthy. The flow
component for zero-based particle 436 has a covariance eigenvalue ratio
1.146e-7, below the FP32 four-state validity threshold 4*eps=4.768e-7.
The guard correctly rejects this covariance. Relaxing the covariance-reset
tolerance would not address this different check.

## The mechanism in the executed code

The checked call chain is the five-model force binding, the batch score filter,
and the shared `covariance_proposal_score_tf.build_maps`. The batch consumer
actually calls this shared function. It forms C=H' R^-1 H, holds this local
observation linearization fixed during the flow, and uses

    P(lambda) = (Q^-1 + lambda*C)^-1
    F(lambda) = -0.5 * P(lambda)*C
    A_next = (I + delta*F(lambda))*A

with delta=1/M and lambda=k/M, k=0,...,M-1. It checks the resulting proposal
covariance A Q A'. All parameter partials propagate through this recurrence.
The localization does not establish a new derivative-formula error: it
identifies an ill-conditioned finite value program whose score is rejected.

Write Q=Lq Lq' and S=Lq' C Lq. In whitened coordinates, the first Euler map
has eigenvalue a0(mu)=1-mu/(2M) along an eigenvector of S. At the failed input,
mu=16.01561546 and M=8, so a0=-0.00097597. The first step almost annihilates
that direction; forming A Q A' squares this small factor. This explains the
near-singular proposal and the precision-sensitive rejection. It is an
integration pathology, not evidence that the physical posterior covariance
should collapse. The covariance guard must remain active.

## What the bounded replays show

At the identical pre-observation-14 ancestor cloud, the minimum map covariance
ratio is 1.146e-7 at 8 FP32 steps, 0.033157 at 16, and 0.064475 at 32. The
eight-step FP64 map still has almost the same near-zero Euler factor, but
passes its much finer precision threshold. Precision alone does not remove
the discretization pathology.

Full recurrences were then rerun because altered early maps change all later
clouds. Each row below is at the actual failed warmup position. Scores are
likelihood derivatives with respect to theta, before the prior and q-chart
Jacobian; values are the filter's log likelihood, not the independent HMC
endpoint. The failed original output is only a prefix and is not comparable
to a complete log likelihood.

| Diagnostic configuration | All 20 observations valid | Log likelihood | Score in theta coordinates |
|---|---|---:|---|
| FP32, 8 flow steps | No: fails at observation 14 | Incomplete | Incomplete |
| FP64 recurrence / FP32 pairs, 8 steps | Yes | 37.1252698033 | (-9.4875145580, -4.7165761387) |
| FP32, 16 flow steps | Yes | 37.1211776733 | (-9.5296020508, -4.7273864746) |
| FP32, 32 flow steps | Yes | 37.1167945862 | (-9.5304117203, -4.7273559570) |

All candidates also pass the full recurrence at the preceding healthy input.
There, the original FP32 value/score are 38.8474197388 and
(-5.7894821167, -6.5408611298); 16 steps give 38.8387947083 and
(-5.7968273163, -6.5110387802); 32 give 38.8344039917 and
(-5.8051705360, -6.4942417145). These changes are reported, not treated as
accuracy against an oracle. On the failed input, 16 versus 32 steps change
log likelihood by 0.0043831 and score components by 0.0008097 and 0.0000305.
This is useful local agreement, not a uniform error bound or independent
validation. It says nothing about the independent endpoint's approximation
error.

## Repair and audit requirements

1. The smallest demonstrated candidate is 16 FP32 flow steps, with 32 as a
   refinement check. Keep N4032, TF32 off, current priors, endpoint and validity
   guards. Freeze a candidate only after fresh scope-specific calibration and
   validation, including independent reference checks. The failed warmup inputs
   are regression cases, never untouched validation data. Increasing M only
   moves the first-step resonance to mu=2M; it does not prove global safety.
2. A more structural candidate evaluates the coefficients at the right endpoint
   lambda_j=j/M. Its whitened multiplier is
   a_j(mu)=1-mu/[2(M+j*mu)]. For mu>=0 and j>=1,
   a_j(mu)>=1-1/(2j)>=1/2, so this particular zero-factor mechanism is absent
   in exact arithmetic. This proof concerns the frozen local PSD information
   matrix used here, not all finite-precision factorizations or the full filter.
   It does not establish value/score accuracy or posterior convergence.
   Li–Coates equation (15) and Algorithm 1 lines 12–18 also use lambda_j after
   advancing lambda; the current timing differs from that published rule.
   Author source has not been audited in this diagnostic, and the complete
   current frozen-linearization mixture route is not claimed source-faithful
   to their Algorithm 1.
3. If implementing that rule, update both the analytical-jet map and the
   independent directional/parity map, test the real batch consumer, and
   retain analytical score propagation. Check finite-difference score parity,
   small exact LGSSM cases, KSC references, and range-bearing references on
   independent inputs. More particles and looser guards do not repair the
   demonstrated Euler factor. A switch to FP64 is not the selected default.
4. After the fixed GPU/XLA candidate passes fresh tuning and health checks,
   start a new range-bearing warmup attempt and use the existing adaptive
   retained-sampling controller. Preserve the failed attempt. Do not add a
   separate terminal validation chain or reuse the old selection identity.

These are repair phases of the existing master. No live numerical/runtime file
was changed: active selections bind source hashes, and the other GPU models
are still running. Their remaining worker allocation is already reserved.
The current result therefore identifies a viable local repair, not an executed
GPU/HMC repair. A retry requires remaining campaign allocation, fresh tuning
and a new versioned output directory, not another procedural approval token.

## Decision and uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Original attempt fails its force-health screen | Actual failure reproduced and localized | Near-singular FP32 flow covariance | Full posterior-region frequency | Repair integration and retune | Failure of the range-bearing model or surrogate-HMC idea |
| 16/32 steps are viable repair candidates | Both saved inputs complete all 20 observations | No validity veto in these CPU replays | GPU/XLA parity and independent-region coverage | Fresh scope checks before sampler retry | General stability, improved inference or default readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Original FP32 map rejected; refinements pass the inspected inputs |
| Statistically supported ranking | None; these are paired deterministic debugging replays |
| Descriptive-only differences | Values, scores and CPU elapsed times |
| Default readiness | Not established; FP32 remains the user-selected default |
| Next evidence needed | Independent calibration/validation, GPU parity, reference checks and adaptive HMC |

Strongest alternative explanation: CPU graph arithmetic may not reproduce every
GPU/XLA failure. The actual archived GPU force flags plus matching CPU failure
support the localization, but a GPU replay is still required for promotion.
Other posterior-region covariances or the independent endpoint can fail for
different reasons. Discovery of a different first failing condition in GPU
replay would limit this diagnosis. Two saved inputs are the weakest part of
the repair evidence; refinement has not established broad robustness.

## Reproducibility

HEAD 10884528ca8b1a7463895256d30b9655d69c0955, branch
ledh-graceful-failure-execution-20260917. Successful replays use
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, deliberately hidden GPUs
before TensorFlow import, 4 intra-op/2 inter-op CPU threads, and an explicit
non-XLA diagnostic outer filter. Nested components can still compile with XLA;
this is not GPU/XLA performance or parity evidence. A preliminary attempt in
`tf-gpu` failed loading an incompatible custom-op ABI and did not run the
filter. Using the actual campaign environment resolved that setup error.

Commands are recorded in each JSON. The complete successful diagnostic set is
`inspect-warmup.py`, `replay-force.py uniform`,
`replay-force.py fp64_recursion_fp32_pairs`, `inspect-map.py`,
`replay-force.py uniform 16`, and `replay-force.py uniform 32`.
Full-filter elapsed times are respectively 64.04, 75.01, 87.27 and 91.49s;
map localization takes 56.73s. Concurrent CPU timings are not a fair runtime
comparison. No GPU campaign time was added by these diagnostic runs.

Evidence: `archive-summary.json`, `replay-uniform.json`,
`replay-fp64_recursion_fp32_pairs.json`, `map-diagnostic.json`,
`replay-uniform-steps16.json`, `replay-uniform-steps32.json`, and the saved
scripts/logs in this directory. The replays retain data and source hashes,
selected scope and exact coordinates. The first replay script version's hash
differs from the later version that adds the explicit flow-step argument;
each result records its executed version rather than restamping old output.
The source passage used for the timing comparison is the locally preserved
`.localresources/papers/m13-repair-20261004/li-coates-2017.txt` / PDF,
equation (15) and Algorithm 1. No author-code faithfulness verdict is issued.
