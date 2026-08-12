# Zhao-Cui Austria SIR Lane-B Parameter Child Zero-Slice Plan

Date: 2026-07-31

Status: `ACTIVE_EXECUTION`

## Entry Evidence

- T1 status: `PASS_NEW_FIXED_VARIANT_T1_VALUE_BASELINE`.
- T2 status: `PASS_NEW_FIXED_VARIANT_T1_T2_VALUE_BASELINE`.
- T1 identity:
  `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`.
- T2 identity:
  `f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e`.
- Cumulative origin value: `-66.28380350560136`.

This plan replaces the stale P88 reconstruction assumptions in the July 30
master plan. It preserves that plan's parameter semantics, source
classification, memory discipline, and derivative-ownership requirements, but
uses the newly admitted Lane-B T1/T2 artifacts as the actual parents.

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | Can compact external-theta channels preserve the admitted T1/T2 state TT slices exactly at theta zero while exposing correct manual derivatives of the resulting finite child scalar? |
| Candidate | Identity chart `u(theta)=theta`; conditioned state cores `A_k(theta)=A_k^0+sum_a theta_a D_{a,k}`; theta-independent frame, shift, tau and defensive reference. |
| Promotion criterion | Parent tensors remain immutable; conditioned cores, density, normalizer, retained marginal, increment and cumulative value reproduce the parents at zero; manual value and marginal derivatives match diagnostic GradientTape on the same child program. |
| Promotion veto | Parent mutation, theta integration/sampling, dense theta-state grid, changed defensive term/measure/shift, caller-stamped parent identity, missing T2 parent lineage, or runtime autodiff/finite-difference score. |
| Continuation veto | No compact derivative contraction exists, the exact origin slice changes the admitted scalar, or derivative storage exceeds the linear TT memory contract. |
| Repair trigger | Local contraction, identity, serialization, parity, or tolerance failure under the unchanged child definition. |
| Explanatory diagnostic | Random tangent-core GradientTape parity. It validates algebra only and does not train or promote a scientific score. |
| Must not be concluded | No trained parameter TT, correct model score, T5/T10/T20, HMC, production KR, posterior, or scientific validity. |

## Mathematical Program

For parent state core `A_k^0` and three compact tangent cores `D_{a,k}`, use

\[
 A_k(\theta)=A_k^0+\sum_{a=1}^3\theta_aD_{a,k}.
\]

Theta is an external conditioning input. It is never integrated or sampled.
The amplitude and density are

\[
 h_\theta(z)=A_1(\theta;z_1)\cdots A_d(\theta;z_d),
 \qquad
 \rho_\theta(z)=h_\theta(z)^2+\tau q_0(z).
\]

The shift, affine frame, `tau`, and `q0` are fixed parent fields in this first
child. Thus

\[
 Z(\theta)=\int h_\theta(z)^2d\nu(z)+\tau,
 \qquad
 L(\theta)=\log Z(\theta)-c.
\]

At zero, every conditioned core equals its parent tensor algebraically, so
`rho_0`, `Z(0)`, every source-style prefix marginal, and `L(0)` equal the
parent program.

For parameter `a`, each paired-core contraction has derivative

\[
 D_aM_k=(D_{a,k}\otimes A_k)+(A_k\otimes D_{a,k}),
\]

with the same basis mass/evaluation contraction as the value. Propagating
`(v,D_av)` left-to-right gives `D_aZ` and retained-marginal numerator
derivatives in memory

\[
 O\!\left(3\sum_k r_{k-1}b_kr_k\right),
\]

not a theta/state tensor grid. The normalized marginal derivative uses the
quotient rule. The child increment score is `D_aZ/Z` because `c`, `tau`, and
`q0` are frozen in this phase.

For T1 and T2 separately, tangent banks are independent compact child fields.
The cumulative origin score of the child mechanics is the sum of the two
increment scores. Later training must make the T2 tangent bank represent the
total previous-marginal dependence. An arbitrary mechanics tangent bank does
not establish that scientific ownership.

## Source And Classification Ledger

| Operation | Classification | Anchor |
|---|---|---|
| Squared-TT state mass and prefix marginal | `source_faithful` operation | Zhao-Cui Algorithm 2(b-c); author `@TTSIRT/marginalise.m:19-85` |
| Parent T1/T2 target/event order | `source_faithful` operation in a local extension | paper Eq. (15), Algorithm 2(a); author `models/full_sol.m:72-80,101-124` |
| Frozen parent artifacts and deterministic identities | `fixed_hmc_adaptation` | admitted Lane-B T1/T2 artifacts; no HMC authorization |
| External theta, centered tangent cores, child identity, manual derivative contractions | `extension_or_invention` | this plan |

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
|---|---|---|---|
| Identity chart | exact local theta coordinates; mechanics baseline | later domain conditioning may need scaling | zero-slice and tangent parity; revisit before training |
| Linear centered cores | simplest exact origin embedding; mechanics baseline | insufficient capacity away from zero | no capacity claim here; Phase-3 target-specific protocol |
| Frozen frame/shift/tau/q0 | required parent preservation | trained child may need explicit theta dependence | origin equality now; later design must add and differentiate any change |
| Three tangent banks | model parameter dimension | full cross-axis coupling may require rank expansion | Phase-3 heldout/downstream validation |
| FP64 CPU reference | zero-slice correctness gate | does not establish GPU/FP32 | later GPU/XLA phase only after trained score correctness |

## Skeptical Pre-Execution Audit

| Risk | Finding/correction |
|---|---|
| Wrong baseline | Uses the admitted Lane-B identities and value, not P88, APF, source replica, UKF, or retained-grid production route. |
| Proxy promotion | GradientTape parity checks the manual contraction only; no random tangent is promoted as a score. |
| Hidden assumption | Linear/identity conditioning is explicitly a mechanics baseline, not a trained default. |
| Stale context | Supersedes only the old plan's P88 reconstruction fields; source and derivative ownership remain binding. |
| Missing stop | Origin mismatch, parent mutation, noncompact storage, or derivative mismatch stop the phase. |
| Memory | Stores exactly three same-shaped tangent banks per artifact and streams parameter contractions. |
| Non-answering artifact | Exit establishes only the exact zero slice and derivative machinery required before parameter training. |

Audit verdict: `PASS_FOR_EXECUTION`.

## Execution Steps

1. Implement repository-owned conditioned-core identity and compact value,
   normalizer, point-density, and prefix-marginal derivative contractions.
2. Add parent mutation, shape, theta-state integration, and caller-identity
   guards.
3. Test zero-slice equality for the admitted T1 and T2 artifacts.
4. Test manual value/density/marginal derivatives against diagnostic
   GradientTape on bounded nonzero tangent fixtures.
5. Record core/tangent bytes and issue an origin child mechanics result.

Exit: `PASS_LANE_B_T1_T2_PARAMETER_CHILD_ZERO_SLICE`. No training, score claim,
GPU campaign, or HMC occurs in this phase.
