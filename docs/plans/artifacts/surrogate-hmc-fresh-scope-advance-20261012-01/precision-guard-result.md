# Covariance guard result, 2026-10-13

The exact failed range-bearing input now completes all20 observations in all
four chains. Input tensor hashes are identical to the failed replay. Actual HMC
force and exposed B2 filter force agree exactly. The three previously healthy
forces are bitwise unchanged; all18 trace quantities through the old failure
(excluding its validity bit) are identical. All12 predicates pass at every step.

The old9.3706363e-14 covariance residual exceeded an8.8817842e-14 cutoff that
omitted particle-count accumulation. The new derived allowance is3.6717296e-12.
This is a false rejection repair, not a change in the likelihood or score.
FP32 retains its1% policy; all other vetoes remain. Focused CPU tests:56 passed,
including material/nonfinite covariance rejection and scalar/all-P call-chain
parity. Exact GPU replay02 used64.2102 worker seconds; remaining6282.4056 of
the authorized36000 seconds. Source snapshots, commands, memory policy,
device, seed and TensorFlow details are in replay result.json. Comparative
numbers and predicates are in precision-guard-comparison.json.

| Decision | Primary check | Veto status | Main uncertainty | Next action | Limit |
|---|---|---|---|---|---|
| Retain guard repair | Exact inputs healthy, force/trace invariance | All numerical predicates pass | Other parameter regions untested | Fresh tuning then mechanics through master | No posterior claim |

| Inference status | Finding |
|---|---|
| Hard veto screen | Old covariance-only false rejection removed; no other failing predicate |
| Statistically supported ranking | None; deterministic implementation check |
| Descriptive differences | First-call runtime is descriptive, not a speed ranking |
| Default readiness | FP32 default unchanged; optional mixed path remains scoped |
| Next evidence needed | Fresh calibration/validation and HMC mechanics, then budget-feasible actual sampling |

Post-run audit: the strongest alternative explanation would be a hidden
arithmetic change. Identical input hashes, unchanged healthy forces, unchanged
pre-failure traces and explicit predicate telemetry rule that out on this
replay. The allowance is an accumulation estimate, not a theorem for the entire
centering/factorization pipeline. The scientific algorithm and model-accuracy
questions remain separate. This repairs the master; it does not complete its
posterior phase.

Harness and telemetry follow-up:114 CPU tests passed; log precision-guard-harness-cpu.log.
