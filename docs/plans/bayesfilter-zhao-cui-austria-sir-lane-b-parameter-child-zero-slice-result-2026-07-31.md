# Zhao-Cui Austria SIR Lane-B Parameter Child Zero-Slice Result

Date: 2026-07-31

Status: `PASS_LANE_B_T1_T2_PARAMETER_CHILD_ZERO_SLICE`

## Result

The admitted T1 and T2 state TT parents can be embedded in a compact external-
theta child without changing their origin slices. The implemented child uses

`A_k(theta) = A_k^0 + sum_a theta_a D[a,k]`

and integrates only state axes. It stores exactly three tangent banks with the
same TT-linear core layout as each parent; no theta/state tensor grid or theta
particles are created.

At theta zero:

- every conditioned core equals the parent tensor exactly;
- T1 and T2 increment values match the admitted parents within `2e-13`;
- the T1 normalized retained-prefix marginal matches the parent within
  `2e-12`;
- parent core tensors remain unchanged; and
- child identities bind the parent identity, parent/tangent hashes, frame,
  shift, tau, measure, chart, and state-only normalization rule.

Manual paired-core derivatives for the normalizer, point log density, and
normalized prefix marginal match diagnostic GradientTape on the identical
child scalar. The focused and combined suites passed:

- zero-slice suite: 9 tests;
- combined B2/B3/B4/tail/child suite: 30 tests.

## Compatibility Finding

The T2 v1 identity embeds a recomputed parent T1 value in addition to the parent
identity. GPU issuance produced parent-value bits ending `...1a24`; CPU
recomputation produced `...1a26`. The canonical CPU loader therefore remains
fail-closed. A narrowly named parameter-parent compatibility decoder restores
only the claim-bound GPU scalar after verifying:

- selected T2 identity
  `f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e`;
- passed claim SHA-256
  `289565b59455a59e31190a5240ef98cbd885cfe4213677ecde1f22c31e206244`;
- parent identity and CPU value within four ULPs;
- all T2 tensor hashes and source closure; and
- CPU cumulative value agreement within `5e-13`.

This is a compatibility defect in v1 derived-float identity design. It is not a
license to restamp arbitrary artifacts or make the canonical loader permissive.

## Decision Table

| Field | Verdict |
|---|---|
| Decision | admit compact zero-slice mechanics and open target-specific parameter training design |
| Primary criterion | passed |
| Veto diagnostics | passed; canonical CPU identity drift remains explicit and fail-closed |
| Main uncertainty | no tangent channels have been trained against the SIR parameterized target |
| Next justified action | reviewed T1/T2 parameter-channel training protocol with disjoint data and downstream checks |
| Not concluded | no correct model score, parameter TT capacity, T5/T10/T20, HMC, posterior, or scientific validity |

## Inference Status

| Field | Status |
|---|---|
| Hard veto screen | passed for mechanics only |
| Statistically supported ranking | not applicable |
| Descriptive-only differences | synthetic tangent tests only |
| Default readiness | no |
| Next evidence | trained tangent/coupling channels and same-child total-score validation |

The test tangent banks are synthetic. Their returned derivatives are correct
for those mechanics children but are not the Austria SIR observed-data score.
No HMC was run.
