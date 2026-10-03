# q20 nonlinear-learning mechanisms: continuation checkpoint

User requested a thorough LaTeX explanation, an implementation plan, review,
and tests of free global scale, RKL path gradients, and direct scalar shape.
Completed the documentation and optional implementation/mechanics phases in
the [plan](bayesfilter-q20-three-mechanisms-plan-2026-09-23.md).
Read the [result](bayesfilter-q20-three-mechanisms-results-2026-09-23.md).

New code is `bayesfilter/inference/neutra_training_mechanisms.py`; 16 new tests
and 46 existing regressions pass. The true loss is distinct from the path
gradient carrier. The generic small-Jacobian VJP engine is not an optimized
coupling-flow implementation. The scalar inverse checks residual AND bracket
width, reports failure, and has stopped gradients. The SGD update is a tiny
mechanics smoke, not the production Adam continuation. Parameter serialization
does not confer frozen-map HMC admission. Production defaults and unrelated
worktree changes remain preserved.

LaTeX section: `docs/chapters/ch26b_neutra_training_repair.tex`, included by the
NeuTra chapter. It documents measured failures, math, source boundaries,
efficiency/inversion, initialization degeneracy and the three test mechanisms.
The isolated chapter PDF builds and pages 5--11 were visually reviewed; no
overfull boxes, all new citations resolved, two external chapter references
unresolved only in the isolated build.

Next phase is concrete production integration and priced q20 GPU canaries, as
listed at the end of the plan. Bind the actual frozen depth-four checkpoint
and live budget ledger before launching. No q20 target evaluations or GPU work
were done in this implementation pass. The old scientific balances were not
expanded, reallocated or charged for routine unit tests. Do not claim a trained
q20 map or resume from the generic SGD fixture. Existing source context at the
start of this pass was main at e9fee584704a465fc6ab984e3a8cc53980f335d0.
