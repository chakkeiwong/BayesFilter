# P2 Result: Exact Transformed SV

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Decision: `P2_COMPLETE_BOTH_CELLS_TARGET_BLOCKED_CONTINUE_P3`

## Outcome

P2 reached honest terminal states for both cells without issuing a posterior
identity or running HMC or NeuTra training.

| Cell | Terminal state | Reason | Next justified action |
| --- | --- | --- | --- |
| `SVX-SGQF` | `TARGET_BLOCKED` | The graph-native implementation passed engineering, score, batch, CPU-XLA, GPU-XLA, data-replay, and status checks, but no frozen SGQF level passed the numerical filter-admission rule. | Preserve the negative filter result; do not loosen thresholds after inspection. |
| `SVX-ZC` | `TARGET_BLOCKED` | The current factorized fixed-grid wrapper is `extension_or_invention`, and no production-admissible fixed Zhao-Cui source route exists. | A separately approved source-route implementation program would be required. |

Neither cell has a complete typed target identity. Therefore no plain-HMC
comparator, training screen, 5,000-step training, NeuTra HMC, or posterior claim
was admissible in P2.

## SVX-SGQF Evidence

The terminal attempt is:

`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p2/SVX-SGQF/target-admission/attempt-04-20260715T103649Z/`

The run used the preserved seed-81101 T=1000 raw trajectory, source prior and
two-probit chart, fixed audit points, levels 2/4/6/8, level-10 reference, and
predeclared thresholds. It completed on the trusted RTX 4080 SUPER GPU with XLA
and verified TensorFlow memory growth.

| Gate | Evidence | Status |
| --- | --- | --- |
| Data identity | observations `5e2423149e4f59eb588ccc7f16ec6d9ee984ccc4710a3ae07a3dbcf5c37db748`; states `338b3ba4ce18fe6ef758c216a679f0537729d032e2ae5ff795ed8ecbe1fed453` | Pass |
| Focused implementation suite | 29 CPU-hidden tests | Pass |
| Score correctness | maximum prefix posterior score/FD gaps were about `8.0e-10` to `1.1e-9` | Pass |
| Status validity | levels 6, 8, and 10 were finite with zero status code | Pass for those levels |
| Dense-prefix filter gate | threshold `1e-3` per observation; level 10 observed `0.0033869554` | Fail |
| Full value convergence | level 6 and 8 passed the `1e-4` per-observation value gap, but other mandatory gates failed | Partial only |
| Full score convergence | threshold `1e-3`; level 8 to level 10 observed maximum gap `0.5437891` | Fail |
| Typed posterior identity | no level passed all gates | Not issued |

The computed object is the fixed-SGQF approximate posterior recurrence. The
checked score is the derivative of that same finite recurrence and is supported
by finite-difference agreement. The recurrence is not equal to the dense KSC or
exact transformed-SV filter: the filter-admission discrepancies above establish
that the frozen candidate ladder did not meet the declared approximation
criterion. This is a filter-candidate admission failure, not an implementation
failure and not evidence against NeuTra on a different admitted target.

## Repairs And Attempts

Attempts 01-03 failed before any candidate level produced scientific evidence:

1. attempt 01: TensorFlow memory growth was configured too late;
2. attempt 02: visible-GPU dataset generation changed the stateful random stream;
3. attempt 03: a `float32` transformed-truth point was concatenated with the
   `float64` audit matrix.

Each was repaired locally without changing target, data, criteria, hardware
class, or budget. Attempt 04 is the only terminal ladder result. The repair
history is in
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p2-repair-record-2026-07-15.md`.

## Integrity And Budget

The recorded recursive hashes verify:

- `result.json`: `344a8f21bfc602f4b88649501003eceb811e02a37607948ceb2813a282513b43`;
- `run_manifest.json`: `e4bdc174092298ed4e08382959994365e3c9ebf07ceb94a73c51f41a2c2a1d91`.

Attempt 04 consumed 14.1 GPU wall-seconds. The failed attempts ended before the
level ladder and did not approach the four-GPU-hour admission bucket. No HMC or
training bucket was consumed. P2 remains far below its aggregate ceiling.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close P2 and continue P3 | Both P2 cells honestly classified | No program continuation veto; both cell-local admission vetoes fired | Higher SGQF levels were not tested because the frozen ladder ended at 10 and post-result expansion was forbidden | Enter independent KSC-UKF target repair | No exact-SV NeuTra, HMC, calibration, filter ranking, or scientific-validity claim |

## Inference Status

| Class | Status |
| --- | --- |
| Hard veto screen | `SVX-SGQF` failed its numerical filter-admission veto; `SVX-ZC` failed its production source-route veto. |
| Statistically supported ranking | None; no stochastic candidate comparison was run. |
| Descriptive-only differences | Per-level value and score gaps describe this one frozen ladder only. |
| Default readiness | Neither cell is ready for a posterior identity, comparator, training, or default. |
| Next evidence needed | A new predeclared filter candidate for SGQF, or a production-admissible source route for Zhao-Cui; neither is required for P3. |

## Post-Run Red Team

The strongest alternative explanation is that the dense-prefix threshold or
level-10 reference is too strict or too low-order for a useful approximate
posterior. That possibility does not authorize changing the frozen P2 rule
after seeing results. A new plan could test it with independent thresholds and
reference convergence, but this result remains a failure under the contract
actually run. The weakest evidence is the absence of a second independently
implemented SGQF recurrence; parity and FD tests establish implementation
consistency, not scientific adequacy.

P3 is independent: exact-SV observations, filter results, and transports cannot
be substituted for KSC evidence.
