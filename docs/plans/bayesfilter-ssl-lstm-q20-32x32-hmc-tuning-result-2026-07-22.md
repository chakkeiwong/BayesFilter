# q=20 SSL-LSTM `(32,32)` NeuTra-HMC Tuning Result

Date: 2026-07-22  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-32x32-hmc-tuning-plan-2026-07-21.md`  
Run root: `docs/plans/artifacts/ssl-lstm-q20-32x32-hmc-tuning-2026-07-21/tuning-lock-20260722/`

## Decision

The repaired current-source preflight passed for both independent q=20
`(32,32)` frozen transports. The bounded tuning run completed with status
`TUNING_REPAIR_REQUIRED`; no retained HMC was launched.

This is a candidate-level tuning failure, not a continuation veto and not
evidence against the NeuTra research direction. The transformed target passed
the engineering validity gate. Chart A had a viable pilot scale and viable
short-trajectory pilots, but its 64-draw confirmation had two chains above the
confirmation upper band. Chart B's midpoint scale had finite telemetry but
failed the every-chain pilot gate (`0.3983` and `0.9075` in two chains), so no
trajectory ladder was admitted.

| Decision | Primary criterion | Veto status | Main limitation | Next action |
| --- | --- | --- | --- | --- |
| Transformed target validity | Both charts pass round-trip, value/score identities, and FD ladder | No hard veto | Engineering check only | Proceed to a declared tuning repair |
| Chart A pilot | Midpoint scale and `L=2,4` are viable | Pilot passes | 16-draw pilots are descriptive | Repair confirmation scale/trajectory jointly |
| Chart A confirmation | Every chain acceptance in `[0.60,0.80]` | Candidate veto: `[0.7983,0.7062,0.8481,0.8044]` | 64 draws per chain; native divergence telemetry unavailable | Use a predeclared adjacent step-size repair, not retained HMC |
| Chart B scale | Every chain pilot acceptance in `[0.50,0.90]` | Candidate veto at midpoint | Pooled mean `0.7505` hides chain dispersion | Add a bounded scale/initial-state robustness repair |
| HMC/posterior correctness | Not tested by this run | Not applicable | No retained samples or convergence evidence | Forbidden until fresh tuning and retained-chain gates pass |

## Evidence And Provenance

- Fresh preflight: `preflight-lock-20260722/summary.json`, SHA-256
  `582a3a966682321ca6fb466652db42e65e5d795576ff812326a7aeb4cc66bcdd`.
  It charged `298.72999228397384 s`; both charts selected FD step `0.001` and
  passed all hard checks.
- Fresh tuning summary: `tuning-lock-20260722/summary.json`, SHA-256
  `1e999b5181e60e74abd6a224832bb0bbbeca3048b6713aeddd00b27b6dcbabd6`.
  It charged `5040.527063304093 s` of the `23400 s` cap, with no resource or
  hard veto.
- Current execution-source signature: `4850cc7b9e7b3db9d9e67524bfd80695e370a1aaa8296260c81a516b41e78ac7`.
- GPU 1, TensorFlow 2.20.0, XLA and TF32 enabled, memory growth verified, and
  managed-session GPU trust recorded in the run manifest.
- All pilot/confirmation artifacts report
  `samples_retained_as_posterior_evidence=false` and
  `native_divergence_status=unavailable_not_zero`.

## Single-Writer Repair

The earlier `tuning-9303ed7/` root is invalid for cumulative accounting. A
mistaken concurrent `--resume` launch caused its checkpoint charged value to
decrease from `6499.714991202811` to `4977.73101590015` seconds. That root is
debugging/resource evidence only. The runner now owns an exclusive
`.material-run.lock` per material output root; the focused regression suite
passed (`17 passed`). The fresh run used a new root and exactly one writer.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Preflight passed; no tuning arm had a hard numerical/resource veto except the expected L=16 pilot nonfinite telemetry in chart A |
| Statistically supported ranking | None; no candidate ranking was attempted |
| Descriptive-only differences | Acceptance, movement, RMS jumps, and runtime |
| Default readiness | Not assessed |
| Next evidence needed | Review and execute a bounded repair for per-chain scale/confirmation robustness, then fresh retained-chain admission |

## Nonclaims

This result does not establish posterior convergence, posterior correctness,
transport quality, predictive validity, architecture superiority, sampler
superiority, or production/default readiness.
