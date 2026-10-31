# Genuine one-iteration L-BFGS compiler-context control

The first-objective experiment04632--04634 does not reproduce the candidate's
historical score. Both reduced contexts return the original score on exactly
the old inputs. Its fake optimizer removed the component whose compiled
context may matter. The next engineering question is whether retaining the
real TFP optimizer but allowing only one iteration reproduces the first score
difference. This is a new bounded localization control, not graph bisection,
a full optimizer replay or a numerical repair.

Use the same r1 frozen snapshot, target, seed, initial point, instrumentation,
original04584/candidate04585 authorities and isolated CPU setup. Replace the
diagnostic optimizer dispatch with a wrapper around the installed real
`tfp.optimizer.lbfgs_minimize`, setting only `max_iterations=1`. Preserve all
other supplied settings, including line search controls. Save the dispatch
source/hash and complete callback arrays. The real optimizer may evaluate more
than one objective during that iteration; cap the artifact check at128 callback
batches and retain the300-second parent/280-second child deadline. Run original
and candidate separately, with one numerical worker at a time.

Primary controls: exact first two input rows against04584/04585, exact values
and status masks, saved HLO identity, one trace, no host callbacks, and the
declared dispatch. Compare all first-two-row score bytes and their ULP/absolute
differences to both historical arms. The localization result is positive only
if each arm reproduces its own historical score and the original discrepancy
survives. A collapse or third score is another negative result, not a repair.
No comparison threshold or original recorded failure changes. Do not rank
runtime, claim convergence, qualify a current source snapshot, or admit any
initializer from this intentionally truncated optimizer.

Allocate at most6 CPU workers/1800 CPU process-seconds as a separate unit,
inside the unchanged56 CPU/52 GPU hour caps. Three planned workers are the
original arm, candidate arm and saved-evidence/policy readback; the rest permit
at most two localized harness repairs. Any changed input/source, absent HLO,
unexpected callback count, timeout or missing output stops the affected arm
until classified. Use fresh numbered campaign directories. No GPU, training,
HMC, package mutation, flag change or external source edit is involved.

Skeptical pre-run review: reducing the iteration limit can itself change compiler
optimization, so a second negative result cannot locate the precise operation.
However, it removes the main confound of the previous fake dispatch while
keeping computation far below the474/504-callback trajectories. Exact initial
rows and the same recorded observer are essential controls. More instrumentation
inside the target would alter the disputed context and is deliberately avoided.
The primary agent reviewed this plan; no independent reviewer was used. Further
graph bisection remains gated on a smaller positive reproduction.

The positive readback04637 permits one bounded standard-library inspection
of saved HLO before choosing any numerical bisection. Charge at most120 CPU
seconds within this same1800-second unit. Compare corresponding computations
after renaming local SSA identifiers and computation references by declaration
order and removing source metadata; preserve shapes, constants, operand
relationships and operation attributes. Record full original hashes and any
unresolved reference instead of silently treating it as equal. Small text
fixtures must distinguish identifier-only changes from changed arithmetic.
The normalized comparison is a structural localization aid for unoptimized
HLO, not proof of identical optimized code, numerical equivalence or a runtime
repair. If correspondence is ambiguous, retain the limitation and inspect the
specific differing region. No new numerical worker is authorized by this
read-only extension. Preserve reader source, command, time and complete JSON
under `terminal-locator-hlo-20260928-r1`.

The first readback finishes in4.249 seconds but exposes declaration-order
ambiguity: some corresponding ordinals name different reductions/scatter
helpers. Its149 unequal ordinal pairs are not149 distinct arithmetic changes.
Refine the saved-HLO reader once under another120-second bound within the
same1800-second unit. Replace called-computation references by their already
computed normalized content hash and compare the resulting multiset of
computation hashes, preserving multiplicities. Stop on forward/recursive or
unresolved references. Check that changing a callee's constant changes its
caller's hash. This removes the observed ordering confound; instruction-order
differences still remain conservative leads, not semantic inequivalence.
Preserve r1 and write r2 separately. No numerical execution is added.
