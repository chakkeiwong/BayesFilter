# Neural angle conditioning result — 7 October 2026

The complete conditional neural screen **fails**: 5456 of 5472 checks pass.
All raw-proposal and geometric angle checks pass, including the preserved
angle-failing fixture and both new fixtures. But fresh fixture A produces
16 float32 gradient checks outside the unchanged bound: raw and clipped gradients
in four parameter tensors, repeated in MDPF and A-MDPF. Their largest scaled
error is 1.4453. These are repeated manifestations of the same first-update
discrepancy, not 16 independent failures.

All float64 checks pass. Float32 losses, gradient norms, Adam moments, parameter
updates, frozen parameters, device placement and single-trace checks pass.
The optimizer results do not erase the gradient veto. The original direct angle
screen still fails for the two preserved first updates; both failures remain
in legacy_particle_checks. This run observes 3.7432e-5 radians, slightly different
from the earlier run's 4.0531e-5, under the recorded thread configuration.

The derived bound correctly accounts for amplification by a short sine/cosine
proposal vector. Every error ball excludes the origin, both local angle maps
pass, and deliberate angle corruptions fail. The largest angle error uses less
than 4.85% of its conditional allowance. This establishes the stated conditional
angle result on these fixtures; it does not derive a universal floating-point
bound for the neural network or accept the full float32 consumer.

## Decision and inference

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Reject current complete neural screen | 16 gradient checks fail | Float32 gradient mismatch | Same-input gradient algebra versus rounding amplification | Replay the failing float32 inputs and parameters in both float64 implementations | Replication accepted |
| Retain conditional angle explanation | All raw and geometric checks pass | No origin/local-map failure | Conservative inherited raw-coordinate tolerance | Preserve explicit conditional scope | Universal numerical stability |
| Continue author training | Existing authorized development run remains healthy | No continuation veto | Completion and trained-model quality | Monitor A-MDPF stage | Numerical or stochastic qualification |

| Inference status | Finding |
|---|---|
| Hard veto screen | Float32 gradient comparisons fail; all computation stays finite |
| Statistically supported ranking | None |
| Descriptive-only differences | Error ratios, counts, runtime and allocator peak |
| Default-readiness | Not established |
| Next evidence needed | Same-input higher-precision localization, justified compatibility criteria and independent-trained-unit qualification |

Predeclared source fixtures: preserved two-row replay; new four-row fixtures
with seeds 202610171/202610181 and row offsets 256/288. Two actual Adam updates
per variant/dtype. Source draws are replayed; these are development inputs,
not qualification data. The actual TF consumer runs GPU/XLA, TF32 disabled,
verified memory growth. TensorFlow live allocator peak: 17,762,304 bytes.

Execution used 10.0999 CPU seconds and 153.0542 GPU seconds, one of three allowed
attempts. A separate conservative 60 CPU seconds covers implementation checks
and reporting. Full commands, environments, pinned-source hashes, input hashes,
captured data, logs and numerical checks are preserved under
artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-angle-20261007-01.
No runtime implementation or author method changed.

Post-run red team: the new gradient failures disprove the assumption that
previous passing optimizer fixtures settled neural derivative compatibility.
The existing float64 cases use their own draws, so they cannot yet establish
the high-precision answer on the failing float32 inputs. That exact-input
comparison is the next discriminating check. Both replications remain incomplete.
