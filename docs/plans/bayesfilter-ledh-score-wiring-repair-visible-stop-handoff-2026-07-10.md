# LEDH Score Wiring Repair Visible Stop Handoff

Date: 2026-07-10

Last updated: 2026-07-11

## Status

`PHASE9_FD_POLICY_CORRECTED_PARTIAL_CONTINUATION_REVIEW_REQUIRED`

The inherited `0.005` absolute-or-relative FD decisions and the intervening
`2%` RSS/RMS correction are superseded. The owner clarified that the policy is
only for the finite-difference diagnostic and checks parameter directions
individually:

```text
r_j = abs(score_j - FD_j) / max(abs(score_j), abs(FD_j), 1e-12)
threshold = 0.05 * sqrt(p)
pass iff max_j(r_j) <= threshold
```

The `5%` constant was selected to mirror the conventional 95% threshold. The
calculation is not itself a calibrated confidence interval. It has no sampling
distribution, standard error, or coverage calculation.

Offline reclassification of all 11 completed Gate B/Gate C comparisons gives
9 passes and 2 failures:

- predator-prey fails Gate B `T=1,N=2`: `1.0 > 0.122474487139`;
- generalized-SV passes Gate B but fails Gate C `T=4,N=10000`:
  `0.442753962161 > 0.0866025403784`;
- fixed-SIR passes its historical terminal `T=20,N=10000` FD comparison:
  `0.0566700085587 <= 0.0866025403784`;
- Actual-SV (`p=2`) passes its historical terminal `T=4,N=10000` FD
  comparison: `0.0602924688125 <= 0.0707106781187`;
- KSC-SV passes its historical terminal `T=4,N=10000` FD comparison:
  `0.0369351492982 <= 0.0707106781187`.

No original trusted GPU/XLA score or FD shard was modified. No GPU command was
run for this correction.

## Decision Authority

- Correction plan:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-subplan-2026-07-11.md`.
- Correction result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-result-2026-07-11.md`.
- Reclassification JSON:
  `docs/plans/artifacts/ledh-score-wiring-repair-phase9-fd-policy-correction/phase9-fd-policy-reclassification-2026-07-11.json`.
- Review:
  `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-codex-review-2026-07-11.md`.

The final SHA-256 values and review verdict are recorded in the reset memo and
execution ledger after verification. The historical consolidated Phase 9
result remains available only as an execution narrative; its original FD
decisions are not current authority.

## Current Row State

| Row | Current FD-only state | Next ladder point | Runtime authority |
| --- | --- | --- | --- |
| predator-prey | Failed Gate B | None under current candidate | FD-blocked |
| generalized-SV | Passed Gate B; failed Gate C `T=4` | None under current candidate | FD-blocked |
| fixed-SIR | Passed full-time Gate C `T=20` | Gate D seeds `81121`-`81124`, then aggregation | New reviewed continuation manifest required |
| Actual-SV | Passed Gate C `T=4` | Gate C `T=50`, then `T=250`, `T=1000` if each gate passes | New reviewed continuation manifest required |
| KSC-SV | Passed Gate C `T=4` | Gate C `T=50`, then `T=250`, `T=1000` if each gate passes | New reviewed continuation manifest required |

Passing this FD diagnostic removes only that specific veto. It does not admit a
score or establish general score correctness, HMC readiness, posterior
correctness, full-time SV memory, or default readiness.

## Execution Boundary

Do not replay the historical exact-command manifest. It targets the superseded
v1 shard paths and predates the corrected v3 runner schema. Replaying it could
overwrite immutable source evidence and would not satisfy current governance
hashes.

Before any resumed GPU execution:

1. Write a narrow continuation subplan that preserves the original ladder
   shapes, seeds, transport, precision, memory gate, and row-local stop rules,
   while binding the corrected FD-only policy.
2. Generate a new exact-command manifest with new output/log paths and the v3
   runner.
3. Review that subplan and manifest before the first GPU command.

The correction itself authorizes no GPU run, Gate D run, aggregation, HMC run,
or Phase 10 execution.

## Remaining Blocks

- Predator-prey and generalized-SV are stopped by stored FD failures.
- Fixed-SIR, Actual-SV, and KSC-SV are not FD-blocked, but runtime continuation
  is blocked pending the reviewed manifest above.
- No nonlinear five-seed aggregate or score-admission artifact exists.
- The separate LGSSM lane remains outside this correction.
- Phase 10 has no scoped reviewed subplan and remains unauthorized.

## Nonclaims

The correction provides no statistical ranking or uncertainty-supported
superiority claim. It does not establish HMC behavior, posterior validity,
exact nonlinear likelihood correctness, native Actual-SV correctness for KSC,
runtime or memory superiority, full-time SV feasibility, or production/default
readiness.
