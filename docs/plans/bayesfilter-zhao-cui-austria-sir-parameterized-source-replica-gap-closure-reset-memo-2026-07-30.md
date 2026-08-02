# Zhao-Cui Austria SIR Gap-Closure Reset Memo

Status: `SUPERSEDED_HISTORICAL_SOURCE_REPLICA_RESET_NOT_ACTIVE`

> Superseded by
> `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-reset-memo-2026-07-30.md`.
> Author-training reproduction is not the active next step.

Date: 2026-07-30

The historical plan summarized by this memo was
`docs/plans/bayesfilter-zhao-cui-austria-sir-parameterized-source-replica-gap-closure-2026-07-30.md`; it is not current authority.

Implemented:

- author-bound Austria SIR T1 source-replica spec and immutable serialized TT
  cores;
- correct finite-reference algebraic CDF/Jacobian handling;
- conservative no-full-grid KR memory checks;
- full-covariance block-upper author-order affine adapter;
- suffix-conditioned reverse KR inversion, forward map, numerical sampler
  density, and exact TT conditional comparator;
- bounded Algorithm-3 T1 importance correction and ESS diagnostic; and
- focused regression coverage (`41 passed, 2 warnings`).

The old blocker `BLOCK_T1_FORWARD_AUTHOR_CONDITIONAL_ADAPTER_MISSING` was
closed. The historical campaign ended at
`BLOCK_T1_SOURCE_REPLICA_FIT_OR_PROPOSAL_GATE`.

Attempt 7 is the historical terminal result. It passes finiteness, identity, memory,
roundtrip (`1.52e-5`), and numerical-vs-exact conditional consistency
(`7.61e-3` max log-density discrepancy). It fails proposal ESS: `1/8`, fraction
`0.125`, against the `0.5` continuation threshold. The transition-density
range, not numerical KR inconsistency, dominates the corrected weight spread.

Do not run T2/T20, parameter score, GPU/XLA, or HMC-facing work. The next phase
must address the fit route: reproduce/audit author TT-cross/ALS or create a
reviewed target-specific training protocol with disjoint validation and L1
tuning. Re-run an untouched T1 gate before continuing.

This is a candidate/fitter rejection, not a Zhao-Cui direction rejection.
Attempt 2 is superseded by the holdout-frame bug; attempt 3 is the corrected
same-frame historical baseline; attempt 6 is the matching correction-audit
predecessor. All attempt directories remain preserved.
