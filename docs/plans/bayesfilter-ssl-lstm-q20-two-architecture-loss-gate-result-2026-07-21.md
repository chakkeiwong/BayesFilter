# q=20 SSL-LSTM NeuTra Two-Architecture Loss Gate Result

Date: 2026-07-21  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-two-architecture-loss-gate-plan-2026-07-21.md`  
Status: `COMPLETED_UNRESOLVED`

## Result

Both `(32,32)` and `(64,64)` completed two hard-valid loss-only training arms.
Saturation remained explanatory telemetry and did not veto a checkpoint, reduce
the learning rate, or affect selection. All learning-rate reductions were
triggered by the paired validation-loss plateau rule.

The predeclared architecture decision is `UNRESOLVED`. The `(64,64)` audit mean
was descriptively lower in both seeds, but only seed-a's paired 95% interval
excluded zero. Seed-b's interval included zero, so the required two-seed
consistency rule did not nominate either architecture.

| Architecture | Seed | Best step | Audit mean loss | Terminal saturation | LR reductions | Hard veto |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `(32,32)` | seed-a | 2,000 | 41.194448 | 0.093750 | 0 | none |
| `(32,32)` | seed-b | 2,000 | 40.977650 | 0.078125 | 1 | none |
| `(64,64)` | seed-a | 1,500 | 40.851793 | 0.101562 | 1 | none |
| `(64,64)` | seed-b | 1,250 | 40.870385 | 0.085938 | 1 | none |

Paired differences are `(64,64) - (32,32)` on the same 256 audit draws:

| Seed | Mean difference | Standard error | Paired 95% interval | Interpretation |
| --- | ---: | ---: | --- | --- |
| seed-a | -0.342655 | 0.051077 | [-0.443241, -0.242070] | interval favors `(64,64)` under this fixed protocol |
| seed-b | -0.107266 | 0.090307 | [-0.285108, 0.070577] | statistically unresolved |

These intervals quantify audit-cloud Monte Carlo uncertainty conditional on
each fitted transport. They do not quantify the full training-seed uncertainty
with only two independent training seeds.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not nominate an architecture | two intervals did not exclude zero in the same direction | all four arms finite; round trips at most `1.48e-14`; support radius about `4.0 < 4.3` | seed-b paired interval crosses zero; only two training seeds | architecture-specific loss tuning or additional prospectively planned seeds before a default/HMC decision | no superiority, posterior correctness, convergence, HMC readiness, or default promotion |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | `PASS_ALL_FOUR_ARMS` |
| Statistically supported ranking | none under the predeclared two-seed consistency rule |
| Descriptive-only differences | both `(64,64)` audit means are lower; training runtime and saturation differences are descriptive |
| Default readiness | not established |
| Next evidence needed | target-specific architecture/hyperparameter tuning with disjoint final audits, or more prospectively planned training seeds |

## Run Manifest

Both launches used commit `41f2aa4f263d96e5575a6448d89bdd93bb262035`,
the `tfgpu` environment, GPU 1, TensorFlow/TFP, XLA enabled, TF32 enabled,
batch size 100, 16 CPU target workers, the fixed parameter artifact hash
`236e179e027b226bd910ae0a6999b184f3d1f69c84ce4548510d043e2c3a5c45`,
and the owner-designated managed-session trust basis. The launches set
`TF_FORCE_GPU_ALLOW_GROWTH=true`; the runner established memory growth before
project imports and failed closed if verification was false. The original
terminal summaries did not serialize that verification field; the runner now
does so prospectively. They also did not record TensorFlow allocator current and
peak bytes, so no live-memory claim is made from these runs; the runner now
serializes those allocator counters prospectively.

The `(32,32)` first process ran from 16:15:12 to 17:04:54 (about 2,982 seconds)
and failed only during final export after persisting the step-2,000 checkpoint.
The recovery plus seed-b process recorded 2,988.414 seconds. Cumulative
`(32,32)` wall time was therefore about 5,970 seconds, below the 7,200-second
architecture cap. `(64,64)` recorded 5,949.607 seconds. Combined wall time was
about 11,920 seconds, below the 14,400-second program cap. These are wall-clock
resource accounts, not runtime-performance comparisons.

Artifacts:

- `(32,32)` summary SHA-256: `e87edbd7a048f16c786c0c81ef60d86b92a8e5f66b393b1ccdf4904e71095041`;
- `(64,64)` summary SHA-256: `b69e06b86620db9708c04dcbcd099fdacd5b2e324060ab294a939847772a1aa9`;
- combined summary: `docs/plans/artifacts/ssl-lstm-q20-two-architecture-loss-gate-2026-07-21/combined-gate-summary.json`.

## Implementation Failure And Repair

The first `(32,32)/seed-a` launch completed 2,000 optimizer steps, then failed
during final audit because the runner called `forward_and_logdet` on a frozen
transport. That API belongs to the trainable wrapper; frozen artifacts expose
`forward_batch` and `log_abs_det_jacobian_batch`. The repair uses the public
frozen batch API and recovers a verified stopped checkpoint without replaying
optimizer steps. This was an export implementation failure, not evidence
against the architecture, target, or loss criterion.

## Provenance Limitation

Another repository lane edited `neutra_training_control.py` at 19:08:52,
after the `(64,64)` process launched at about 18:07 and imported its controller.
The terminal filesystem snapshots therefore have different controller hashes
(`bd975e...` versus `d52591...`). This is not evidence that the loaded controller
changed inside the running Python process. However, because the old runner
captured hashes only when writing the terminal summary, the exact launch-time
imported hash for `(64,64)` was not independently preserved. The comparison is
therefore `VALID_WITH_SOURCE_PROVENANCE_LIMITATION`, not exact source-identity
evidence.

The executed artifacts still serialize identical decision-relevant controller
configs, the same loss-only policy, thresholds, schedules, target signatures,
seeds, audit definition, runner/trainer/target/pool hashes, and hard-veto
outcomes. The changed inverse-radius logic was not discriminating: every
observed radius was approximately `4.0`, below the common `4.3` threshold.
This supports using the run for the predeclared unresolved decision, but not
claiming independently proved launch-hash identity. The runner now captures
imported source hashes at process start so later concurrent edits cannot be
mislabeled as executed code.

## Post-Run Red Team

The strongest alternative explanation is optimization confounding: a shared
learning rate and initialization scale need not be equally suitable for both
widths. Thus the fixed-protocol outcome does not establish relative architecture
capacity. A prospectively tuned `(64,64)` and `(32,32)` comparison could overturn
the descriptive direction. The weakest evidence is training-seed replication:
two seeds are insufficient for a broad stochastic ranking. HMC remains withheld.
