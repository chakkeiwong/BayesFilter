# Contract E--TP Phase 5 Austria SIR Target Semantics Result

metadata_date: 2026-07-15
status: BLOCKED_TARGET_MEASURE_MISMATCH

## Question And Verdict

Can the current Austria SIR simulator, transition density, and ordinary LEDH
Jacobian be treated as one filtering target for a Contract E--TP value/score
comparison?

No. They compute different probability laws. This is wrong relative to a
same-target value/score claim and is not a tolerance or finite-particle issue.

## Source Evidence

The pinned author source under
`third_party/audit/tensor-ssm-paper-demo/models/sir_austria/` states:

- `st_process.mlx`: apply `sir_step`, add full-rank Gaussian noise, then replace
  every negative susceptible coordinate by zero;
- `transition.mlx`: evaluate `mvnpdf` with identity-scaled covariance around
  `sir_step`, with no clipping, truncation normalization, or atomic mass;
- `priorpdf.mlx` and `priorsam.mlx`: both use a full-rank 18-dimensional normal;
- `setup.mlx`: process scale is one and observation scale is ten.

The BayesFilter fixture reproduces the same split: its process push clips but
its `transition_log_density` is Gaussian. Its process and initial covariance
matrices are both `I_18`.

## Mathematical Classification

For one susceptible coordinate, if `Z` has a continuous Gaussian law and
`X=max(Z,0)`, then

\[
  P(X=0)=P(Z\leq0)>0,
\]

while the positive half-line has the pushed Gaussian density. Thus the clipped
law has an atom at zero and is not represented by Lebesgue density `mvnpdf`.
In 18 dimensions the same issue occurs on coordinate hyperplanes. An ordinary
invertible LEDH flow and log-determinant correction cannot create or account
for those atoms.

The target actually computed by author `transition.mlx` is a valid full-rank
Gaussian density program. It may be implemented as a separately identified
source-density target, but it is not the clipped sampling law. Structural
deterministic completion is also inapplicable because the Gaussian covariance
is nonsingular.

## Decision Table

| Decision | Criterion status | Veto status | Uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| block clipped-law Contract E--TP score | push and density differ as measures | continuation veto for same-target claim | intended scientific target not selected | preserve as blocked until target binding | no claim that clipping is harmless |
| permit separately named Gaussian-density diagnostic | density program is mathematically complete and differentiable | requires new immutable identity and same-target tests | relationship to clipped synthetic data | test only after explicit binding in Phase 6 | not the clipped simulator law |
| retain P90/P91 evidence | established component/value-bridge scope unchanged | previous-marginal and transport derivative blockers remain | full observed-data score absent | reuse without promotion | not a filtering score comparison |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | failed for treating clipped push and Gaussian density as one law |
| Statistically supported ranking | none |
| Descriptive-only differences | none generated in this blocked lane |
| Default readiness | false |
| Next evidence | explicit target identity, matching proposal accounting, then `T=1,2,5` |

## Nonclaims

This result does not reject Contract E--TP for SIR, Zhao--Cui's fixed-TTSIRT
method, or structural methods for genuinely singular models. It blocks only the
mathematically inconsistent combination currently exposed by the Austria
simulator/density pair.
