# Phase 15 Result: Paired Identity-versus-Affine Full-Bank Adjudication

Status: `PASS_HARD_GATES_ROLE_LIMITED_PAIRED_DESCRIPTIVE`

The identity comparator completed on the same metadata-bound N=300 bank,
partitions, mode axis (`2`), profile, seed, 300-update budget, GPU memory
policy, XLA path, and batch-native trainer as the Phase 14 affine run. All
three arms passed finite-value, target/status, round-trip, audit, and artifact
gates. The full-bank metrics below are the primary representation diagnostic;
validation-subset metrics are explanatory only.

| Arm | Preconditioner | Full-bank mean max | Full-bank off-diagonal max | Full-bank diagonal max error | Covariance Frobenius residual | Loss |
|---|---|---:|---:|---:|---:|---:|
| compact | identity | 0.1209 | 0.2112 | 0.1591 | 0.4973 | 7.2337 |
| compact | affine | 0.2214 | 0.3414 | 0.3318 | 0.8027 | 3.1724 |
| compact_low_lr | identity | 1.0360 | 0.5764 | 3.0627 | 4.0169 | 10.3060 |
| compact_low_lr | affine | 0.1197 | 0.1243 | 0.1921 | 0.3575 | 5.5927 |
| wider_mid_lr | identity | 0.0926 | 0.2573 | 0.1553 | 0.6035 | 7.3111 |
| wider_mid_lr | affine | 0.3320 | 0.3016 | 0.3635 | 0.7467 | 3.6860 |

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain both coordinate routes as role-limited candidates | all hard gates pass; no uniform full-bank residual improvement | no hard veto | one paired seed and arm/preconditioner interaction | run a second paired seed, then select any follow-up by predeclared evidence | no universal affine benefit, IID law, posterior correctness, mode discovery, HMC readiness, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for both receipts |
| Statistically supported ranking | None; one seed and three interacting arms |
| Descriptive-only differences | Affine substantially repairs `compact_low_lr`; identity is descriptively better for the other arms |
| Default-readiness | Not ready; coordinate choice is scope/arm dependent |
| Next evidence needed | second paired seed, then a predeclared multi-seed comparison or a source-faithful modular mechanism test |

The result weakens the hypothesis that poor whitening is solely an affine
conditioning problem. It does not reject affine preconditioning: the strong
interaction is a repair trigger for replication, not a continuation veto.
No HMC was launched.

Run artifacts:

- identity: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase15-attempt1-identity-fullbank2401/result.json`
- affine comparator: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase14-attempt5-affine-fullbank2401/result.json`
