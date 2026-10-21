# Mixed-scope mechanics results, 2026-10-13 HKT

SIR exhausted its 1200-second worker cap before the pilot was archived. Its
independent endpoint audit passed (log value -1717.12544; four-design delta-method
MCSE 0.22198, explanatory only). There is no completed acceptance or chain
diagnostic. The timeout is a cost/harness limitation, not evidence against HMC.

Range bearing passed the endpoint audit, then the selected force was invalid
at one or more of the four prescribed starts q=(-.15,-.05,.05,.15). Sampling did
not begin. Fresh-data precision at nominal/nearby points was insufficient to
establish force validity over the starting region. Preserve the failed candidate.
The next discriminating diagnostic evaluates the full filter at those starts,
records completed steps and validity margins, and distinguishes small numerical
margins from a substantive proposal/reset failure. Do not narrow starts or
relax thresholds to hide this failure.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| SIR incomplete | No completed pilot | Worker time cap | Where time was spent | Incremental pilot records; shared cost repair | Sampling failure or convergence |
| Range needs repair | Initial-force screen failed | Invalid force | Failing row and operation | Per-row, per-step localization | Precision checks imply HMC readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Range initial force invalid; SIR time cap |
| Statistically supported ranking | None |
| Descriptive differences | Endpoint MCSE only; no complete sampler diagnostics |
| Default readiness | Unestablished |
| Next evidence | Valid force across initialization region; affordable complete pilot |

Paths: ../surrogate-hmc-mixed-mechanics-sir-20261012-01 and
../surrogate-hmc-mixed-mechanics-range-20261012-01. Commands, snapshots, hardware,
seeds and wall time are in their result.json files and the cumulative budget.
Post-run audit: a generic exception and terminal-only pilot save concealed the
localization needed for the next repair. Add observations of already-computed
forces and archive cold-run tensors immediately. This does not change the
sampler, data, endpoint or acceptance test. Numerical source changes still require
fresh tuning before another claim-bearing mechanics run.
