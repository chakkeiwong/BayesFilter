# Phase 30 Repair and Refresh Note

| Attempt | Failure class | Repair | Result |
|---|---|---|---|
| 1 (`phase30-theta-genut-scope`) | candidate scope result: negative central weights | no clipping or weight substitution; preserve global and local receipts | `PARAMETER_GENUT_GLOBAL_INFEASIBLE_SCOPE` |

The receipt is
`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase30-theta-genut-scope/`.
Global theta-R4 GenUT had `w_zero=-0.4751`; negative and positive axis-2
subsets had `w_zero=-0.7192` and `-0.6253`. Discriminants and offsets were
finite and positive, but the nonnegative-weight condition failed in every
scope. No clipping or substitution was applied. This is a candidate/scope
finding, not a continuation veto; the fresh theta bank remains eligible for
the independent Phase 31 NeuTra boundary audit.
