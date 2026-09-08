# Phase 22 Result: Per-Mode/Local GenUT Feasibility

Status: `LOCAL_GENUT_INFEASIBLE_SCOPE`

The same source-faithful GenUT equations were applied separately to the two
finite sign regions of the audited bank (`mode_axis=2`). Both regions remained
infeasible because the central sigma weight was negative, despite positive
discriminants and offsets.

| Region | Rows | Retained weight mass | Central weight `w0` | Feasible |
|---|---:|---:|---:|---|
| `theta[:,2] < 0` | 95 | 0.27495 | -0.26326 | false |
| `theta[:,2] >= 0` | 205 | 0.72505 | -0.11148 | false |

No sigma points were passed to the target because neither local rule satisfied
the paper's nonnegative-weight condition. No clipping, balancing, or invented
mode weights were applied.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close GenUT as a global/local q20 arm for this bank | both finite mode-conditioned feasibility checks fail | candidate-specific GenUT veto; no program blocker | other local charts or a different cloud could change feasibility | proceed to the source-faithful invertible LEDH-PFPF density fixture | no rejection of GenUT generally, no density/authority/IID/posterior/HMC/default claim |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard input/numerical screen | Passed |
| Global GenUT candidate | Infeasible in Phase 21 |
| Local GenUT candidate | Infeasible in both tested regions |
| Statistical ranking | None |
| Descriptive-only evidence | per-region central weights and skew/kurtosis |
| Default-readiness | Ineligible for this bank/scope |
| Next evidence needed | invertible LEDH-PFPF density/Jacobian fixture |

No HMC was launched.
