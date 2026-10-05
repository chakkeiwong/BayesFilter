# First-objective compiler-context localization result

CPU original04632 and candidate04633 pass in78.933 and79.941 seconds.
Each evaluates the objective once through the declared diagnostic replacement,
has one trace and no host callbacks, and saves callback arrays plus HLO. The
readback/policy group04634 passes161 checks in14.097 seconds. This unit used
three of six workers and172.971 CPU seconds, with no failed execution.

The first two callback positions, values and validity bytes are identical to
the frozen full-trajectory controls04584/04585. The original diagnostic scores
also match04584 exactly. The candidate diagnostic scores instead match the
original arm exactly: against04585 they differ by up to1.5276668818842154e-13
(1952 ULPs in the most sensitive coordinate). Thus the smaller dispatch does
not reproduce the historical first-score discrepancy.

This is a negative localization result, not a runtime repair. Removing the
L-BFGS program also changes the enclosing compiler context. That change is
sufficient to remove the observed discrepancy, but does not identify an
individual arithmetic operation, prove the candidate's full optimizer stable,
or establish convergence. The old121 strict full-record differences and all
failed comparisons remain preserved. No full474/504-callback trajectory is
repeated and no tolerance changes.

| Decision | Primary criterion | Veto/uncertainty | Next action | Not concluded |
|---|---|---|---|---|
| Close this truncated-dispatch diagnostic | Both arms and byte-level saved-input readback pass | Historical candidate score is not reproduced | Use a separately bounded real-optimizer one-iteration control | Runtime repair or precise compiler cause |
| Retain the numerical gap | Historical source/inputs remain pinned | Full optimizer context is still required by current evidence | Compare first callback bytes with a genuine bounded L-BFGS body | Equivalence, convergence or admission |

Skeptical post-run review: the negative result is expected if replacing the
optimizer changes fusion or other context, but this experiment cannot identify
that mechanism. Different input bytes would invalidate the localization;
the exact saved-input checks rule that explanation out here. The weakest
boundary is the replacement of L-BFGS itself. No independent reviewer was used.
