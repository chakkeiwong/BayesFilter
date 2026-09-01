# Phase 19 Result: Small q=20 ETPF Integration Probe

Status: `PASS_ETPF_Q20_PROBE_ROLE_LIMITED`

The source-faithful Sinkhorn/Riccati map was applied to a deterministic
32-row subset of the metadata-bound N=300 authority bank. Attempt 1 passed all
target/status and transport-marginal checks but exposed a scale-dependent
covariance residual (`0.01552`) at the source `1e-3` Riccati increment stop.
Attempt 2 tightened the increment tolerance to `1e-5` (repair hypothesis) and
passed the role-limited integration gates.

| Gate | Attempt 2 receipt |
|---|---:|
| finite transform | true |
| Riccati convergence | true, 59 iterations |
| row residual | `6.66e-16` |
| column residual | `5.55e-16` |
| mean residual | `4.44e-16` |
| covariance residual | `1.0013e-4` |
| target values finite | true |
| target statuses valid | 32/32 |
| rows below/above retained subset range | 7 / 10 |
| corrected negative-entry fraction | `0.50586` |
| transformed mode-axis-2 negative fraction | `0.34375` |

The target accepts the transformed rows, but the support excursions and
negative correction fraction are substantial. This is exactly why the arm is
role-limited: finite moments and target/status validity do not prove a density,
posterior authority, or mode discovery.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain source-faithful ETPF as a q20 auxiliary candidate | source constraints and 32/32 target/status gates pass after declared tolerance repair | no hard veto | deterministic small subset; no density correction for transformed empirical rows | audit a source-faithful GenUT fixture next; defer larger ETPF scale until a separate support/measure plan | no authority replacement, IID law, posterior correctness, mode guarantee, HMC, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed on attempt 2; attempt 1 preserved as scale diagnostic |
| Statistically supported ranking | None; no comparator |
| Descriptive-only differences | support excursions, negative correction, and mode fraction |
| Default-readiness | Not ready; auxiliary q20 probe only |
| Next evidence needed | GenUT source fixture and a separate density/measure contract before any larger arm |

No HMC was launched.
