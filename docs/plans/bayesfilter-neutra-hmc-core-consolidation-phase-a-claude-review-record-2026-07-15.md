# NeuTra HMC Program Phase A Claude Review Record

Date: 2026-07-15  
Claude role: bounded read-only reviewer  
Codex role: supervisor and executor

## Availability And Prompt Recovery

- Trusted health probe returned exactly `CLAUDE_PROBE_OK`.
- The first broad one-path review produced no text within 60 seconds and was
  stopped. No repository action depended on that stalled process.
- The fixed-token read probe returned exactly `NEUTRA_PACKET_READ`, proving
  Claude and packet access were healthy.
- The retry was narrowed to a packet-only, at-most-three-finding gate. It
  returned after interruption; the full substantive output was preserved below.

## Round 1

```text
1. Gate-definition evidence gap: The packet says comparator agreement and truth
recovery are scientific gates, but it never states their pass/fail thresholds.

2. S1 requirement mismatch: The reviewed contract requires S1 to pass fresh
tuning plus admission/confirmation, but the S1 execution summary only states
fresh admission and confirmation.

3. Remaining provenance defect: F1 shell command strings were reconstructed
rather than preserved contemporaneously. This may be acceptable, but should be
resolved or explicitly downgraded from no material provenance defect remains.

VERDICT: REVISE
```

## Codex Classification And Repair

| Finding | Classification | Repair |
| --- | --- | --- |
| Missing gate thresholds in packet | material packet sufficiency omission, execution unaffected | Added exact R-hat, ESS, combined-MCSE, recovery, warm-up, and confirmation-minimum thresholds. |
| S1 fresh tuning omitted from packet summary | material packet sufficiency omission, execution unaffected | Added the fresh grid/tuning, selected step, and tuning/admission artifact hash to the S1 execution row. |
| No contemporaneous F1 shell strings | real but non-material residual provenance limitation | Explicitly classified terminal status as complete with a disclosed command-transcript caveat; explained why exact CLI/arguments and contemporaneous numerical/device/seed/hash evidence still identify and reproduce the run; forbade claiming a preserved shell transcript. |

Focused document checks are rerun after every packet repair. Round 2 asks only
whether these exact repairs resolve the material closeout objections.

## Round 2

```text
- The packet now states the exact predeclared admission/confirmation gates
  numerically. That resolves the missing-threshold blocker.

- S1 now explicitly documents fresh tuning evidence, including the selected
  step and bound tuning/admission artifact hash. That resolves the missing-S1-
  fresh-tuning blocker.

- The F1 command issue is now classified correctly and non-evasively. Given the
  contemporaneous hashed device/seed/steps/payload/path evidence, the explicit
  command-transcript caveat does not remain a material closeout blocker.

VERDICT: AGREE
```

Terminal review status: `CONVERGED_AFTER_ONE_REPAIR_ROUND`. Claude remained
advisory and read-only; Codex inspected, classified, patched, and rechecked each
finding.
