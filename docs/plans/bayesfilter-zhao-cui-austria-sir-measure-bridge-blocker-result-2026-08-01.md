# Zhao-Cui Austria SIR Measure-Bridge Blocker

Date: 2026-08-01

## Verdict

The current parent-preserving parameterized Zhao-Cui child cannot be admitted
for the exact Austria T1 observed-data score.  The independent conditional
reference value/score authority passed at `N=8,192` and `N=65,536`, but it also
establishes that the score target is an expectation under the physical
target-weighted law, not under the fitted fixed TT parent measure.

## Why this blocks the current path

The admitted parent is `q_0`, a fitted squared-TT density in a transformed
36-dimensional reference measure.  The exact physical target is `pi_theta`.
The centered residual family only enforces `q_theta=q_0` at `theta=0`; it does
not enforce `q_0=pi_0` or a Radon-Nikodym correction.  Therefore its origin
derivative is under `q_0`, while the desired likelihood derivative is under
`pi_0`.  These are different mathematical objects unless equality or the
correction is proved.

This is not an optimizer failure, rank failure, MCSE failure, or UKF failure.
The rank-12 representation branch is already closed, and both authority audits
passed.  The failure is a target/measure identity gap.

## Evidence anchors

- Zhao-Cui paper conditional proposal and weight correction: `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:807-924`, especially Eqs. (20)-(23) and Algorithm 3.
- Author fixed-route construction: `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:72-135`.
- Author marginalization implementation: `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25-85`.
- Independent T1 authority: `docs/plans/bayesfilter-zhao-cui-austria-sir-conditional-reference-t1-result-2026-08-01.md`.
- Sample-growth authority: `docs/plans/bayesfilter-zhao-cui-austria-sir-conditional-reference-sample-growth-result-2026-08-01.md`.
- Closed rank-12 child: `docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/rank12-minimax-v1/r12_retry02_rank12_gate_max_from_n01/result.json`.

## What remains possible

Only a newly reviewed Radon-Nikodym bridge may continue this line.  It would
need to define exact physical target density, parent support, transformed
Jacobian, finite normalizer, and total derivative in one program.  It would be
an extension/invention until source-mapped.  No existing child or proposal may
be relabeled to satisfy that definition.

The standing finite-program goal opens one bounded Bridge-B diagnostic under
the separately defined target
`L_ZC(theta)=L_parent(0)+log E_q0[pi_theta/pi_0]`.  This does not remove the
exact-physical blocker; it tests whether the working fixed parent is accurate
enough for that finite Zhao-Cui score to agree with the physical authority.

## Stop conditions

Outside the bounded ratio-bridge diagnostic, do not run another rank or
optimizer, off-origin density repair, T1 score claim, T2/T20 recursion,
GenUT/SGQF/UKF comparison, or HMC.

## Bridge-B execution veto

The exact parent KR sampler initially exceeded its configured working-set cap
(`1,224,081,408` estimated bytes versus `536,870,912`).  The sampler was
repaired to process the same uniforms in chunks of 1,024, preserving the
proposal law.  The retry then produced a nonfinite state in the Austria RK4
transition for a valid tail draw.  The run stopped before score comparison and
is preserved as a hard finite/resource veto.  No endpoint clipping, uniform
truncation, resampling, or alternate proposal was applied, because each would
define a different measure.

The current status is therefore `BLOCKED_RATIO_BRIDGE_PARENT_TAILS`.  The
fixed parent remains value-only and no total score is admitted.
