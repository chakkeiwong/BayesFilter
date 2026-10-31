# SIR NeuTra continuation — 2026-10-22

The training ladder and map checks completed; the HMC comparison is running. All
three fresh-seed replicas reached 512 accepted updates without a rejected
training batch. The nomination uses width 6, initial learning rate .01,
affine initialization scale .35 and seed 201. Nomination losses do not support
a superiority claim or establish posterior convergence.

The original freeze attempt stopped because TensorFlow's LogMatrixDeterminant
operation lacks a GPU-XLA kernel. This invalidated that check's execution, not
the saved fit. The localized repair evaluates the same 3x3 determinant using
its cofactor expansion. Fourteen focused CPU tests pass. On the actual saved
map, the charged GPU smoke passed: inverse error 8.88e-16; logdet consistency
error zero; determinant and independent slogdet difference 1.78e-15, against
the unchanged 1e-7 tolerance. The full freeze and short mechanics workers then
completed with `valid=true`. At 2026-10-22 13:54 Hong Kong,
`018-hmc-dense_affine` is live; no tuning or posterior result is saved yet.
Frozen-NeuTra HMC follows automatically in the existing master. Approximately
22 hours of the HMC budget remain, which is a budget limit rather than an ETA.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue HMC comparison | Geometry repair agrees with independent reference | Unsupported-op failure repaired; freeze and mechanics valid | Posterior convergence and comparison pending | Dense-affine HMC, then frozen-NeuTra HMC and planned continuation | Posterior convergence or method superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | Original freeze compilation failed; repaired smoke, full freeze and mechanics pass their checks |
| Statistically supported ranking | None |
| Descriptive differences | Replica losses 1730.116, 1730.171, 1730.125; individual SEs .171, .199, .199 |
| Default readiness | Not established |
| Next evidence needed | Same-endpoint adaptive HMC against dense affine; short mechanics checks do not establish heuristic dominance |

All prior attempts and charged times are preserved in `campaign.json`. The
repair does not reset the authorized 16-hour training or 24-hour HMC limits.
Prior evidence remains in `../sir-neutra-training-20261021-01/`. The reviewed
plan is `../../sir-neutra-training-execution-2026-10-21.md`. Worker manifests
record commands, source hashes, environment, GPU memory policy and wall time.
`preparation/` preserves the CPU log and repaired source snapshot.

Post-run review of the repair: finite geometry on 32 draws establishes only
that this implementation check executes and agrees locally. It cannot exclude
poor posterior coverage or disagreement between the LEDH training density and
independent MH endpoint. Those are questions for the remaining master phases;
this repair neither answers them nor changes their criteria. Use
`campaign.json` and `next-phase.json` for live progress.
