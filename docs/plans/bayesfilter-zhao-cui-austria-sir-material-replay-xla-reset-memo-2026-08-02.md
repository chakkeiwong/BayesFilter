# Zhao-Cui Austria SIR Material-Replay XLA Reset Memo

Date: 2026-08-02

Status: `T1_T2_CLOSED_NO_HMC`

## Current State

The active material-replay/XLA campaign completed T1 and T2. Exact-zero core
replay is no longer an admission gate. The frozen policy is the mixed
five-significant-digit functional rule `atol=5e-12`, `rtol=5e-6`.

Read first:

1. `docs/plans/bayesfilter-zhao-cui-austria-sir-material-replay-xla-repair-result-2026-08-02.md`.
2. `docs/plans/bayesfilter-zhao-cui-austria-sir-t2-scalar-consistency-repair-note-2026-08-02.md`.
3. T1 issuer `t1-material-fd-tangent-issuer-02/result.json`.
4. T2 issuer `t2-material-fd-tangent-issuer-02/result.json`.

## Strict Artifact Chain

- T1 issuer: `cc8460bffd737bcf682434c8ff49c9c52ceb8af45ec81fba92a5afcb4d1556d0`.
- T1 child: `5a006e8f55423cb08e6b3b1b08443c6ac8fb3af1c637ff48c20ed7941cae0603`.
- T2 issuer: `9b6dfaecdd311741facca0b31fb1e69c0accf82a79fd66ad76cc0481ca377313`.
- T2 child: `17e33778c558e62972eb5bfe342e297520ab1475b3722602aeab7827c60cf263`.

Always load T2 through `load_t2_training_jvp_child`, which strictly reloads T1
first. Do not caller-stamp, copy fields into an older schema, or edit any
source-bound plan/source file and then expect existing issuer identities to
remain current.

## Direct Limitations

- Both issuers use centered finite differences, not exact autodiff or JVP.
- T2 uses a first-core radial projection to enforce the direct scalar derivative.
- The route is `extension_or_invention`, not source-faithful Zhao-Cui parameter inference.
- Evidence is local to `theta=0` and the frozen T1/T2 finite programs.
- No T3+, arbitrary-theta, HMC, posterior-correctness, production-readiness, or scientific-validity claim is established.

## Stop Boundary

Do not run T3+, HMC, or later score work under this campaign. A fresh plan is
required with a new evidence contract, compute budget, output root, and
source-faithfulness classification.

The historical exact-zero reset memo remains preserved but is superseded by
the active owner direction and the completed material-replay result.
