# Phase 21 Result: Actual q=20 GenUT Feasibility Probe

Status: `GENUT_Q20_INFEASIBLE_SCOPE`

The source-faithful GenUT equations were evaluated on all 300 weighted
authority rows. Input metadata, finite moments, and source hashes passed. All
four standardized marginal discriminants and offsets were positive, but the
single global central weight was negative (`w0 = -0.4377041`), so the global
`2d+1 = 41` point rule is infeasible under the paper's nonnegative-weight
condition. No target call was made for an infeasible sigma cloud.

| Quantity | Receipt |
|---|---:|
| standardized skewness | `[-0.9266, 1.1598, -0.9866, -0.9179]` |
| standardized kurtosis | `[3.3901, 3.7851, 3.5671, 4.8863]` |
| discriminants positive | true |
| offsets `u_i,v_i` positive | true |
| central weight | `-0.4377041` |
| GenUT feasibility | false |

This is a mathematical scope finding for one global rule on this empirical
measure. It does not prove that local/per-mode GenUT or another proposal route
cannot work, and it does not invalidate the particle authority.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not use one global GenUT rule on this bank | source feasibility condition fails before target evaluation | global-GenUT candidate veto only | local mode-conditioned rules may be feasible; empirical mode labels are finite-run diagnostics | test per-mode/local feasibility; otherwise shift effort to LEDH proposal contract | no rejection of GenUT generally, no density/authority/IID/posterior/HMC/default claim |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard input/numerical screen | Passed |
| Named global GenUT candidate | Rejected by its feasibility condition |
| Statistical ranking | None |
| Descriptive-only evidence | negative central weight and measured skew/kurtosis |
| Default-readiness | Not ready; global route is ineligible for this bank |
| Next evidence needed | per-mode/local feasibility or a different source-faithful proposal |

No HMC was launched.
