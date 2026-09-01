# Phase 16 Result: Second Paired Representation Seed

Status: `PASS_HARD_GATES_ROLE_LIMITED_INTERACTION_REPLICATED`

Both identity and affine routes passed the Phase 16 hard gates on the same
audited N=300 bank. The second seed tuple was `20260825 8305`; the first paired
tuple was `20260825 7305`. Full-bank metrics are the primary diagnostic.

| Arm | Identity Frobenius (seed 1 / 2) | Affine Frobenius (seed 1 / 2) | Identity mean max (seed 1 / 2) | Affine mean max (seed 1 / 2) |
|---|---:|---:|---:|---:|
| compact | 0.4973 / 0.4988 | 0.8027 / 0.7799 | 0.1209 / 0.1210 | 0.2214 / 0.2081 |
| compact_low_lr | 4.0169 / 4.0100 | 0.3575 / 0.3675 | 1.0360 / 1.0324 | 0.1197 / 0.1183 |
| wider_mid_lr | 0.6035 / 0.5957 | 0.7467 / 0.8107 | 0.0926 / 0.0903 | 0.3320 / 0.3112 |

The interaction is directionally replicated: affine preconditioning repairs the
low-learning-rate compact arm, while identity remains descriptively better for
the compact and wider arms. The result is stable enough to reject a universal
"affine always fixes whitening" rule, but the two seeds are not enough for a
statistical ranking across stochastic candidates.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep coordinate choice arm-specific and role-limited | both paired receipts pass; interaction sign repeats | no hard veto | only two training seeds; one particle bank | audit/implement source-faithful modular mechanisms rather than selecting a universal preconditioner | no superiority, IID Gaussian law, posterior correctness, mode-discovery guarantee, HMC readiness, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for all four receipts |
| Statistically supported ranking | None; two seeds are descriptive |
| Descriptive-only differences | affine helps `compact_low_lr`; identity helps the other arms |
| Default-readiness | Not ready; selection is arm and scope dependent |
| Next evidence needed | source-faithful ETPF/GenUT/LEDH contract audit and, only if needed, a larger predeclared ladder |

The residual failure is not a real blocker. It is evidence that representation
conditioning and optimizer scale interact with the finite weighted measure; it
does not establish that the particle authority is invalid. No HMC was launched.
