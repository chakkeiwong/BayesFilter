# Ordered-block capture and graph ownership checkpoint

The internal conditional sequential endpoint now takes the complete center and
scale as runtime operands. CPU runs03017--03021 pass85 checks: two complete
original3582b4ac comparisons with exact target counts, three ownership/HLO guards,
37 derivative/original-field checks and43 existing block consumers.03022 passes
129 policy checks. GPU qualification is pending; two GPU matrix preflights
declined before launching workers because neither non-desktop GPU remained
eligible across the bounded recheck. Desktop fallback conditions were not met.
The public ordered-block controller is unchanged.

The numerical question is preservation of the existing conditional algorithm,
including changed full centers/scales, exact decisions and complete records.
The ownership question is whether releasing the final compiled handle releases
the target and controller graphs, while a retained handle remains usable.
Neither question is answered by RSS alone. The original baseline and all
accepted-result tolerances remain unchanged.

Failures03009--03012 exposed real graph retention after callback/controller
collection.03010 identified cached helper graph ancestry.03013 then traced the
remaining path exactly: TensorFlow's global custom-gradient registry retained
the COD solve's result/matrix/RHS tensors, whose FuncGraph ancestry included the
terminal fit and outer conditional endpoint. The trace-only diagnostic executed
no numerical program and traversed all four new registry entries without
truncation. TensorFlow2.19.1 custom_gradient.py:512--515 registers these closures.

Capture-free eigenpair, affine-position, precision and score-COD functions now
trace under a fresh graph and eager construction context. The numerical code
and pullback formulas are unchanged. Shape-only precision reuse stays global:
scoping it per target would register repeated identical gradient closures.
Callback-dependent target, embedding and empty-schema factories use the existing
endpoint ownership scope.03019 confirms zero additional gradient registry
entries for a changed-target successor, distinct precision, continued retained
handle execution, and collection of every observed owner/graph after release.
The implementation depends on TensorFlow's private eager context; the focused
no-capture, one-trace, derivative and lifetime regressions guard that dependency.

03005/03006 matched original numerical records but failed raw HLO text identity.
The saved diff shows only dummy-source zero-node uniquifiers. The comparator
normalizes only that exact metadata pattern; negative checks retain sensitivity
to changed constants, operation identities and real source metadata. Raw HLO
exports remain preserved. No numerical or policy exception was added.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep internal capture/ownership repair | CPU complete records, counts and lifetime pass | GPU gate pending | GPU behavior and full outer composition | Renew GPU cases, then qualify ordered outer controller | Full repository compliance or merge readiness |
| Attribute graph retention | Registry-to-tensor-to-parent path observed; final lifetime checks pass | No forced cache/registry clearing | Other roots and signatures | Broader E4 audit/churn work | Native executable eviction or leak freedom |
| Preserve scientific comparisons | Original authority and derivative references pass | Strict decisions unchanged | Healthy initializer rounding remains elsewhere | Continue E2/E5/E6 | Canonical LEDH admission or HMC readiness |

The18 workers03005--03022 used542.4569561231183 seconds of this unit's24-worker /
3600-second ceiling. Cumulative charges are60190.454886148495 CPU and
59401.31348332534 GPU seconds, leaving15.28/35.50 process-hours under32/52-hour
caps. Exact commands, source hashes, environment, devices and logs are in each
numbered run. The receipt is
`artifacts/filter-gradient-repair-20260917/block-capture-checkpoint-03022.json`.
All runs use the stable campaign runner, source freeze and one numerical worker.
Focused Ruff and whitespace checks pass. No subagent review was used.

Post-run review: the strongest alternative explanation for lower retention is
merely moving leaked graphs into the global registry. The same-signature
successor guard directly checks that risk, while bounded shape caches and native
signature-churn behavior still need their separate evidence. CPU passes cannot
substitute for GPU or public outer-controller qualification.
