# Corrected Parameter-Authority Phase 30 Result

Date: 2026-08-25  
Status: `PARAMETER_GENUT_GLOBAL_INFEASIBLE_SCOPE`

## Receipt

`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase30-theta-genut-scope/`

The GenUT construction was evaluated on the fresh Phase 28 M0 bank in the
declared parameter measure `theta in R^4`. Because the global construction was
infeasible, two sign-local theta-R4 scopes were evaluated as diagnostics. The
unmodified equations were used; no clipping, renormalization, or negative-
weight repair was applied.

| Scope | Feasible | Central weight `w0` | Discriminant/offsets |
|---|---:|---:|---|
| global theta-R4 | false | `-0.4751` | finite/positive |
| negative axis-2 theta-R4 | false | `-0.7192` | finite/positive |
| positive axis-2 theta-R4 | false | `-0.6253` | finite/positive |

The result is a scope-specific candidate failure. It is not evidence that all
GenUT variants, all theta proposals, or the particle-authority direction fail.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Do not promote unmodified GenUT for this finite theta bank | nonnegative-weight feasibility fails in all tested scopes | GenUT candidate veto only; continuation remains open | bank/proposal dependence and alternate moment designs | continue to independent NeuTra boundary audit; revisit GenUT only under a new reviewed design | no rejection of all GenUT, authority, posterior, IID whitening, LEDH, HMC, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | GenUT role veto: negative central weights; no infrastructure failure |
| Statistically supported ranking | none |
| Descriptive-only differences | discriminants, moment residuals, mode-local `w0` |
| Default-readiness | not ready |
| Next evidence needed | downstream target/transport boundary and, if desired, a separately reviewed alternative GenUT design |

## Red-team note

The negative weights may be caused by the finite cloud's skewness/kurtosis and
the chosen 2d+1 parameterization, not by a fundamental impossibility. A future
proposal-specific design could test another quadrature family, but silently
clipping these weights would change the method and invalidate the current
moment equations.

