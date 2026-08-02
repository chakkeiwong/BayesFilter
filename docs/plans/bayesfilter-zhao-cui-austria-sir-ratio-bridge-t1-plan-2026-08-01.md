# Zhao-Cui Austria SIR Ratio-Bridge T1 Plan

Date: 2026-08-01

Status: `AUDITED_READY_FOR_BOUNDED_DIAGNOSTIC`

## Finite target

Let `q0(r)` be the normalized admitted fixed-parent TT density relative to its
reference measure, and let `pi_theta(r)` be the exact physical T1 joint target
expressed in the same coordinates.  Define

\[
 L_{\rm ZC}(\theta)=L_{\rm parent}(0)+
 \log E_{q_0}\left[\frac{\pi_\theta(r)}{\pi_0(r)}\right].
\]

This is a parent-preserving finite-program extension.  It satisfies
`L_ZC(0)=L_parent(0)` exactly.  Its origin score is

\[
 \nabla L_{\rm ZC}(0)=E_{q_0}[\nabla\log\pi_0].
\]

It is `extension_or_invention`; it is not the exact physical likelihood unless
`q0=pi0`, which is not claimed.

## Evidence contract

| Item | Gate |
|---|---|
| Parent value | Exact equality to `-31.1290512231882` within `2e-13`. |
| Measure conversion | Parent physical density includes affine, algebraic, and `2^36` reference-measure terms; estimated parent mass under `p0 f0` agrees with one within `3*MCSE`. |
| Finite score | Manual ratio score equals autodiff derivative of the same finite program within `1e-9`. |
| Physical comparison | Paired `q0`-weighted and physical `pi0`-weighted scores agree coordinatewise within `3*paired_MCSE + 1e-5` on both independent clouds. |
| ESS | Both `q0` and `pi0` normalized importance ESS are at least half the sample count. |
| Resource | Batch-native TensorFlow, no retained grid/history, peak below 6 GiB. |
| Nonclaims | Passing does not prove exact physical likelihood, source faithfulness, T2/T20, comparator superiority, or HMC readiness. |

## Execution

1. Implement the parent physical log-density and the finite ratio-bridge value,
   manual score, autodiff score, paired discrepancy, MCSE, and ESS.
2. Add focused CPU tests using the admitted parent and `N<=256` clouds.
3. Run two independent `N=8192` diagnostic clouds, seeds `92101` and `92102`.
4. If both pass and remain uncertainty-limited, run one predeclared
   `N=65536` confirmation pair.  If either fails materially, stop the bridge.
5. No optimizer or child fitting is part of this plan.

## Skeptical audit

The exact comparator is the physical conditional-reference authority; the
parent is not assumed exact.  Coordinate Jacobians are explicit.  The paired
score discrepancy is primary, while ESS and mass are vetoes.  Sample size,
seeds, thresholds, artifacts, and stop conditions are fixed.  A successful run
answers only whether this named finite Zhao-Cui target has an origin score
consistent with the physical authority.

Audit verdict: `PASS_FOR_BOUNDED_DIAGNOSTIC`.

Artifact root:
`docs/plans/artifacts/zhao-cui-austria-sir-ratio-bridge-t1-20260801/`.
