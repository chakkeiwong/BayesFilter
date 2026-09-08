# Inter-Phase Repair and Refresh Protocol

Program:
`docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`

Status: `ACTIVE_FOR_EVERY_PHASE`

This file is the common between-phases procedure. Each phase also has a
phase-specific repair note whose status and measured values override its
placeholders.

## Required handoff record

The outgoing phase result must state:

- exact command, git commit, Python/conda environment, device visibility,
  TensorFlow/XLA/TF32 and memory policy;
- input data/target signatures, seeds, controls, protocol hash, wall time, and
  unique output directory;
- every hard veto and its status;
- diagnostics classified as promotion criterion, promotion veto, continuation
  veto, repair trigger, or explanatory only;
- decision: pass, candidate-fail/repair, harness-fail/repair, or real blocker;
- remaining budget and the next smallest discriminating artifact.

## Repair loop

1. Reproduce the failure with the smallest exact command. Do not rerun a long
   campaign before localizing it.
2. Determine whether the computed quantity differs from the claimed quantity.
   If so, label the claim wrong relative to its target; do not soften it into
   an approximation without changing the target explicitly.
3. Inspect source anchors, tensor shapes, density terms, status values, and
   artifact hashes before changing code.
4. Make the narrowest in-scope repair with `apply_patch`.
5. Run a focused regression and preserve the failed artifact. Never overwrite
   it or silently reuse its controls.
6. If the repair changes a numeric control, target signature, partition, route,
   dtype, hardware class, or protocol, create a new tuning/attempt scope and
   label the old value as a warm-start hypothesis.
7. Update the phase result, this phase's repair note, and the next subplan.

At most three repeated attempts may be used for the same localized harness or
infrastructure failure. A changed failure mode is progress and gets a new
classification; an unchanged failure after three attempts is a real blocker.

## Refresh rules for the next subplan

The next subplan must be edited before it runs. It must carry forward:

- the actual artifact/hash and commit from the prior phase;
- gates passed and failed, with the failed candidate separated from a
  continuation veto;
- measured timing and remaining budget;
- controls that are frozen, controls that remain hypotheses, and any required
  disjoint partitions;
- the smallest repair or experiment that discriminates the leading failure
  explanations; and
- explicit nonclaims inherited from the prior phase.

If an entry gate fails, the next phase remains pending while its repair route is
run. If a true continuation veto fires, mark the phase and program blocked and
write why no in-scope artifact can answer the question. Reviewer delay alone is
not a blocker under the repository governance profile.

## Decision vocabulary

Use exactly one primary label per phase:

- `PASS_GATE`: entry conditions for the next phase are met;
- `CANDIDATE_FAIL_REPAIR`: mechanism failed, but the research direction and
  harness remain valid;
- `HARNESS_FAIL_REPAIR`: artifact or implementation failure is repairable;
- `DESCRIPTIVE_ONLY`: evidence is valid but cannot support the claimed role;
- `REAL_BLOCKER`: a stated continuation veto or exhausted budget stopped work.

