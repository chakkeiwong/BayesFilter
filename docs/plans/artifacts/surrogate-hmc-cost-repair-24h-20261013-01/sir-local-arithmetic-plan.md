# SIR arithmetic localization within the computation master

Question: which shared primitive amplifies FP32 roundoff before the SIR score
first differs materially from the same finite FP64 program? This is phase 2 of
the authorized cost repair, motivated by the cost of the current mixed route.

Keep N4032/K2016, TF32 off, saved data/design, the existing paired baseline
controls, two fixed parameter rows (nominal and chart offset 0.01), and analytical
all-parameter tangents. Inspect the first ten observations. These positions and
controls reproduce the saved arithmetic question; they are diagnostic baselines,
not new tuning authority. Horizon ten localizes the early discrepancy and cannot
establish the requested T50 accuracy. No runtime numerics change in this phase.

Compare each FP32 proposal, density, and reset output with that same primitive
in FP64 at inputs obtained by exact casting of its FP32 inputs. This separates
local arithmetic from propagated trajectory differences. Also retain an
independently propagated FP64 staged trajectory. Fused FP32/FP64 T10 runs check
whether staging preserves the behavior. A failed staging comparison forbids
attributing the fused failure to the staged trajectory. It does not invalidate
the identical-input primitive experiment. Preserve per-step value/score and
maximum absolute/scaled and norm errors; health failures veto interpretation.

At the largest local reset discrepancy, save the actual inputs and inspect
weighted moments, transport, transported moments, Cholesky recolouring and final
reset separately. If proposal/density dominates, follow that primitive instead.
Relative condition margins are explanatory. There is no tuning on these inputs.

Pass criterion is a reproducible localization result with identical-input hashes,
finite outputs, and an explicit staging comparison. Error size nominates a
repair; it does not promote one. Any repair must retain the finite target, pass
small analytical checks, all five full horizons and off-centre rows against FP64
within 1%, then actual consumer parity/cost and fresh scope tuning if retained.
FP64 is an arithmetic reference, not a model oracle. This phase establishes
neither model-score accuracy nor posterior convergence nor canonical LEDH status.

Budget: at most 3600 worker-wall seconds for the initial diagnostic, within the
existing 86400-second/120-attempt ledger, single GPU worker. Local retries consume
that ledger. Stop for corrupt inputs, non-finite primitive output, or exhausted
budget; a candidate arithmetic failure triggers the next localized repair.
Output: sir-local-arithmetic-01/sir_d18, including exact command/environment,
source snapshots, tensor checksums and stage-arithmetic.json. The master issues
a next-phase-plan after execution. Deliberate CPU unit checks hide the GPU.

Skeptical audit passed: identical primitive inputs avoid confounding roundoff
with trajectory differences; staged/fused comparisons guard instrumentation;
ten observations are localization only; controls cannot become tuning evidence;
no automatic default or safety-ridge change is authorized by this diagnostic.
No heuristic ranking is being made: the unchanged same-program FP64 comparator
answers arithmetic only. Earlier exact Kalman/KSC and particle references remain
the separate model-accuracy checks.

CPU preflight exposed a comparison hazard: casting an FP32 covariance to FP64
preserves FP32 antisymmetric roundoff, while FP64's symmetry guard is much
stricter. When this invalidates the direct density comparison, retain that mask
and compare both arithmetic modes at the same exactly symmetric FP32 covariance
and covariance tangent, (C+C^T)/2. This is the analytical identity for the
A Q A^T covariance, with a measured roundoff-sized input perturbation. Record
the input change and FP32 output change; do not alter either propagated
trajectory or weaken the runtime guard. A failed common-input health check
invalidates localization. This diagnostic projection cannot promote a runtime
numerics change. All other identical-input comparisons remain direct casts.

Reset follow-up: compare shared moments, transport, recolouring and restoration
at the saved FP32 inputs for observations1-6. Each FP64 local comparison casts
identical inputs. A candidate centered-residual moment expression is an algebraic
rearrangement only: r=x-x0, m=sum(w r), residual=r-m, mean=x0+m. The current
implementation forms x-(x0+m), which can lose small residual sensitivities in a
large common tangent. Compare both expressions to FP64 and preserve full reset
parity; do not change runtime based on this local nomination alone. Budget up to
1800s from the same ledger. This is the predeclared reset follow-up, no target
or tuning change. Candidate differences are not superiority evidence.


2026-10-14 execution refresh: the discrete-design defect is corrected and all
five full horizons were rerun. Four pass; SIR has a 4.51% score discrepancy.
Use the corrected source-matched saved inputs at
sir-local-arithmetic-fixed-design-01/sir_d18, observations1-6, for the previously
planned 1800-second reset decomposition. The centered-moment candidate was
checked on CPU and did not justify a runtime change. The new diagnostic calls
all shared primitives and validates saved tensor checksums, dimensions, controls,
and the scientific source hashes. A tiny CPU decomposition/corruption test and
master dispatch test precede launch. A failed local health check stops this
localization, while a large finite error nominates the next repair. Skeptical
re-audit passed: common-input precision is separated from trajectory propagation;
no numerical protection, tuning policy, posterior claim, or default is changed.
