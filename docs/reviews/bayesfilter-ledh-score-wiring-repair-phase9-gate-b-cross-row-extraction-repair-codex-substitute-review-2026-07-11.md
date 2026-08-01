# Codex Substitute Review: Phase 9 Gate B Cross-Row Extraction Repair

Date: 2026-07-11

## Scope And Limitation

Fresh local read-only review of both Gate B extraction failures, their archived
artifacts/logs, the fixed-SIR covariance repair, the predator-prey RK4 schedule
extraction repair, the all-five-row XLA regression, the cross-row repair result,
and the frozen Phase 9 command/artifact contract. Claude remains policy-blocked
as external repository disclosure. No GPU/CUDA command ran during this review.

This verdict authorizes the ten exact nonlinear Gate B commands only. Execute
score then FD for each row. A row's FD command requires its new score shard to
pass terminal, source/review hash, device, trust, precision, finite-output, and
memory checks. A row-local failure blocks that row and triggers diagnosis; it
does not reject another row unless it exposes shared-harness invalidity. Gate C,
Gate D, aggregation, and LGSSM remain blocked until the complete Gate B result
receives a separate review.

## Failure Classification

| Failure | Review classification |
| --- | --- |
| fixed-SIR attempt-1 FD | Harness/XLA extraction failure. The value route constructed `SpatialSIRSSM` during tracing solely to recover fixed covariance and failed before computing FD. It is not numerical FD evidence. |
| predator-prey attempt-1 score | Row-local score extraction failure. The JVP constructed `PredatorPreySSM` during tracing solely to recover the fixed RK4 schedule and failed before computing a score. Its FD process correctly did not run. |
| shared runner | Terminal artifacts, exact-command validation, prepared-input fingerprints, source/governance hashes, and device/trust controls remained functional. No shared continuation veto fired. |

## Repair Assessment

The fixed-SIR repair uses the already-prepared covariance whose builder obtains
and tiles the same fixed callback covariance. Cholesky of batch entry zero is
the same `[18,18]` factor expected by the existing process-noise `einsum`. No
state transition, covariance, parameter transform, target, or score equation
changed.

The predator-prey repair instantiates the established model once in eager
module initialization and freezes only `delta` and `_rk4_substeps`. All score,
value-only, and historical transition helpers continue to use the exact
schedule `2.0 / 20 = 0.1`; their RHS, RK4 stages, target density, prepared
particles/noise/covariances, transport, and sensitivity equations are
unchanged. The model class and its eager validation were not modified.

The successful intermediate fixed-SIR retry is correctly retained as evidence
that the first repair works on GPU/XLA, but it is excluded from the final Gate B
set because the runner's review identity and reachable source set changed. All
live score and FD paths must be regenerated under this review hash.

## Cross-Row Extraction Gate

The earlier eager-parity tests did not compile every row and therefore missed
both graph-time constructors. The new parameterized test invokes both
`_compiled_score` and `_compiled_value` for all five nonlinear rows under
`tf.function(jit_compile=True)` and compares every output with its eager
prepared-input adapter at `atol=rtol=1e-10`.

Actual-SV, generalized-SV, and KSC-SV also passed independent proactive
CPU-hidden score/value XLA probes before the combined test was added. No repair
was made to those rows.

## Verification

- New graph-safe plus all-row XLA tests:
  `6 passed, 79 deselected, 2 warnings in 73.21s`.
- Predator-prey model contract, including dynamics and same-scalar FD:
  `22 passed, 2 warnings in 128.77s`.
- Binding combined harness/cross-model/shared score contract:
  `158 passed, 2 warnings in 98.73s`.
- New-review-path combined rerun:
  `158 passed, 2 warnings in 98.95s`.
- Syntax, frozen exact-command currentness, and `git diff --check` pass.

CPU-hidden XLA is engineering/extraction evidence only. It does not establish
trusted GPU Gate B, full-row memory, HMC readiness, posterior correctness,
runtime superiority, statistical ranking, or scientific validity.

## Provenance And Stop Rules

- This exact review path is emitted in every run manifest and SHA-256 bound in
  the governance artifact set. Adversarial path/hash tests are present.
- The original literal argv, output/reference paths, targets, tiny shapes,
  seed, transport settings, precision, memory budget, FD steps, and tolerances
  remain unchanged.
- Score must precede FD for each row. No FD process may consume an archived or
  pre-repair score.
- Any nonterminal, nonfinite, wrong-device, wrong-trust, wrong-source,
  wrong-review, or over-budget score blocks that row's FD.
- Any terminal FD failure or frozen absolute-or-relative rule failure blocks
  that row pending diagnosis.
- Even a complete Gate B pass is tiny preflight evidence only and cannot
  authorize Gate C without its required result artifact and fresh review.

No material blocker remains for the ten nonlinear Gate B commands.

VERDICT: AGREE
