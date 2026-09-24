---
title: Generalized SV Oracle Resolution Record
date: 2026-09-14
status: durable-reference
---

## Summary

The generalized-SV oracle for LEDH score-lane work is a **Jacobian-transformed
version of the KSC-SV dense Kalman filter**. This resolution was reached after
multiple agent failures that attempted to treat generalized-SV as a distinct
model requiring its own oracle implementation.

This document preserves the mathematical relationship and implementation status
as a durable reference, extracted from the 2026-08-21 execution ledger that is
otherwise buried in historical artifacts.

## Mathematical Relationship

**KSC-SV model:**
```
x_t = φ x_{t-1} + σ_η η_t,    η_t ~ N(0,1)
y_t = β exp(x_t/2) ε_t,       ε_t ~ N(0,1)
```

**Generalized-SV model (Zhao & Cui 2023 notation):**
```
x_t = μ + φ(x_{t-1} - μ) + σ_η η_t
y_t = β exp(x_t/2) ε_t
```

The generalized form adds a **mean-reversion level μ**, making the latent state
a stationary AR(1) around μ rather than around zero.

## Oracle Construction

1. **Reparameterize to KSC form:** Define z_t = x_t - μ. Then:
   ```
   z_t = φ z_{t-1} + σ_η η_t
   y_t = β exp((z_t + μ)/2) ε_t = [β exp(μ/2)] exp(z_t/2) ε_t
   ```

2. **Run KSC-SV dense Kalman filter** on the z-process with modified observation
   scale β' = β exp(μ/2).

3. **Transform back to x-space:** The log-likelihood is invariant under the
   affine transformation, so no Jacobian correction is needed for the likelihood
   itself. The posterior moments require μ added back:
   ```
   E[x_t | y_{1:T}] = E[z_t | y_{1:T}] + μ
   Var[x_t | y_{1:T}] = Var[z_t | y_{1:T}]
   ```

## Implementation Status

**KSC-SV oracle:** COMPLETE. Located in
`bayesfilter.models.stochastic_volatility.ksc_sv_dense_kalman_oracle`.

**Generalized-SV oracle wrapper:** EXISTS but was last verified 2026-08-21. The
wrapper performs the reparameterization, calls the KSC oracle, and transforms
the output back to x-space.

**Gate status:** The ledger reports that a guard
`test_generalized_sv_oracle_matches_ksc_via_transform` existed and passed as of
2026-08-21. Location and current status unknown — needs verification.

## Failure Mode History

Agents repeatedly attempted to:

1. Implement a separate generalized-SV Kalman filter from scratch, duplicating
   the KSC filter logic with an added μ term.

2. Derive "generalized-SV-specific" recursions, producing longer code that was
   mathematically equivalent to the reparameterized KSC filter.

3. Ignore the KSC filter entirely and reach for generic nonlinear filters
   (EKF, UKF) despite the fact that both forms admit exact Kalman recursions.

The correct resolution is: **reparameterize and reuse the KSC filter**. This is
the minimal-code, mathematically correct approach.

## Source

Zhao, Y., & Cui, X. (2023). *Dynamic portfolio choice with return predictability
and stochastic volatility*. Working paper.

The paper frames the SV process with mean reversion and calls it "generalized
SV" to distinguish it from the canonical KSC zero-mean form. The distinction is
a single affine transformation.

## Next Action When Generalized-SV Oracle Is Needed

1. Verify the wrapper still exists and the gate still passes.
2. If missing, re-implement the wrapper as described above (20 lines, not 200).
3. Do NOT implement a separate generalized-SV Kalman filter from scratch.
4. Update this record with the verified location and gate path.

## Related

- [LEDH execution ledger 2026-08-21](../benchmarks/docs/execution_ledgers/) —
  original resolution record (needs exact path)
- [KSC-SV oracle](../../bayesfilter/models/stochastic_volatility/) — base filter
- Zhao & Cui (2023) paper — should be in `.localresources/` if it materially
  affects decisions
