# Phase 3 Result

Status: `PASS_GATE_ROLE_LIMITED`

The modular arm screen used the frozen N=100 M0 cloud. M1's finite moment
scaffold matched weighted mean/covariance (`2.8041e-10` maximum covariance
residual) and all 100 transformed rows had valid q20 target status. M2's
9-point sigma rule matched selected moments within `1.0e-8` and all nine rows
were valid. M3's affine scaffold recovered points to `1.78e-15`, but this is
not the canonical LEDH-PFPF route and carries no exact-flow admission. M4 is an
explicitly approximate comparator.

No arm was ranked. M1/M2 outputs remain auxiliary; M0 remains the only source
eligible for the next NeuTra screen.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Advance to Phase 4 | role contracts and target status pass | canonical M3 route not implemented, scoped to M3 | no statistical arm ranking | batch-native GPU/XLA NeuTra screen on M0 | no HMC/posterior claim |

## Repaired-bank propagation

The separate N=100 random-walk mutation bank was rechecked in
`phase3-mutation-revalidation-attempt1`. M1/M2 finite contracts and target
status passed again; M3 remained an affine scaffold and M4 remained
approximate. This is a consistency check only. The original identity-bank
results remain the frozen Phase 3 claim path, while the mutation bank remains a
candidate repair input.
