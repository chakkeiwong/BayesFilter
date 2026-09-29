# SQMC correctness repair master program

Date: 2026-09-25. Branch: sqmc-development. Baseline: f3995a06a467f16574f96bbc8a68ccbbc4e30dad.
User authorization: create, review, and execute the program to repair the independent audit findings A1–A12. Historical numerical outputs remain preserved.

## Research question and evidence contract
Can the affected SQMC consumers evaluate the same specified LGSSM, return its finite-program analytical derivatives in every parameter direction, distinguish the four requested computations, and reject invalid or ineligible results?
The exact comparators are the Gaussian model equations, matched Kalman likelihood, independent finite differences at fixed random inputs and unchanged ancestry, and existing shared-core fixtures. Correctness requires agreement within scale-aware numerical tolerances, non-vacuous parameter checks, distinct cap-ablation wiring, and fail-closed validity and tuning checks. Kalman distance is reported separately from derivative parity: correct differentiation does not prove accurate filtering.
Hard vetoes: wrong model/timing/derivative, nonfinite accepted output, duplicate ablation, missing mandatory evidence, scope or partition mismatch. Such failures trigger localized repair; they stop dependent scientific interpretation, not the repair program. Explanatory diagnostics: Kalman error, ancestry changes, runtime, conditioning, compilation behavior. No method ranking, transferability, HMC readiness, asymptotic theorem, or production promotion follows from these repairs.

## Ordered implementation and acceptance
1. A1–A3: define shared TensorFlow LGSSM specifications (P44 covariance parameterization and diagonal-AR standard-deviation parameterization), transition-before-observation timing, initial law, simulation, analytical model callbacks, and matched Kalman value/score. Route every affected runner and tuner through them. Preserve original 3D tuning target as an explicit separate specification.
2. A4–A5,A7: consolidate route construction, full score assembly, ablation cap, initial tangents, finite/status checks and oracle error recording. Missing evidence must prevent successful accuracy claims.
3. A8: inspect the annealed recursion and its previous implementation. Restore the mathematically defined multi-stage computation with existing parity tests; if its claimed computation cannot be restored correctly, retire it explicitly and migrate affected consumers with honest capability tests, without silently replacing its target.
4. A6,A11: remove false Fisher/HMC interpretations from selection; use explicitly defined absolute/relative score errors, report undefined relative quantities as unavailable, require every seed valid, and bind all controls, target/execution scope and disjoint calibration/validation/claim partitions. Old artifacts are historical warm starts, never new-scope tuning authority.
5. A9–A10,A12: correct theorem/equivalence/performance claims, record missing historical evidence, preserve reproducible tuning inputs, eliminate touched NumPy runtime paths, and remove quadratic ancestry counting/search where an equivalent TensorFlow operation suffices.
6. Validate consumer wiring, analytical callbacks, finite-program scores, Kalman timing/model identity, validity rejection, tuning partitions, route distinction, and existing core tests. Run small deliberate CPU reference fixtures first, then bounded GPU graph/XLA compatibility diagnostics where supported. Save exact commands, environment, provenance, logs and a result/checkpoint. Finish with a skeptical review against every audit ID.

## Default and assumption audit
P44 physical quantities follow the checked P44 fixture: Q and R entries are variances, raw initial mean/covariance are retained. Diagonal AR parameters use standard deviations q,r as in the original canonical adapter. The executor predicts before every observation, so generators and the oracle do likewise. Existing numerical controls are diagnostic hypotheses only; cap .98 versus .97 reproduces the intended ablation and does not establish either as optimal or safe. Finite-difference step/tolerance will be checked for stability and ancestry changes. Small CPU float64 checks are explicit reference exceptions, not the default GPU/TF32 execution target. Multi-stage annealing requires checked recursion; deleting the rejection alone is not a repair.

## Skeptical review before execution
The original proposal to rerun the campaign with patched zero callbacks would still compare different targets, duplicate an ablation, and use false metrics. This program fixes definitions and deterministic evidence first. Old holdouts cannot become new untouched claims. Exact Kalman is the primary accuracy reference; particle approximation errors are not hidden by loose derivative tests. Simple filtering adversaries for any later method comparison are exact Kalman, bootstrap IID PF, and the plain uncorrected particle route, evaluated separately by dimension/horizon and informative/weak observation regimes. This repair phase establishes no stochastic ranking and does not tune on those adversaries. The plan is accepted for bounded implementation and verification; broad retuning/leaderboards require their own target-specific evidence, not an automatic promotion at program completion.

## Budget, artifacts and stop conditions
Artifacts: docs/plans/artifacts/sqmc-repair-20260925, unique numbered validation attempts. Maximum numerical execution wall time: 3600 seconds CPU diagnostics plus 1200 seconds GPU compatibility/reference checks; at most 20 grouped attempts. No full historical campaign, package changes, public release, or HMC run. Record actual use and stop execution at budget exhaustion, unresolved target ambiguity, or a failure that cannot be repaired without changing scientific scope. Continue source/documentation repair independently. GPU commands require escalation and verified memory growth; CPU commands hide GPUs before imports. Execution is authorized by the user's request; no additional procedural approval chain is required.

## Execution and terminal review — complete

Steps 1–6 completed within the stated correctness scope. Results, A1–A12
dispositions, mathematical corrections, and remaining coverage are recorded in
[the repair report](../benchmarks/sqmc-repair-results-20260925.md).
73 CPU tests passed with one deliberate production-screen deselection; four
historical runner tests remain unavailable because their imported file is
missing. Eight final GPU/XLA cases and the 32-cell diagnostic tuning workflow
passed. Failed attempts remain preserved. Historical conclusions were replaced
in the active documents and originals archived.

Terminal skeptical review: target and baseline identity are explicit; finite
program parity is not used as statistical accuracy; every required replication
must be valid; independent seed partitions are bound to the tuning artifact;
no performance, equivalence, theorem, default, or HMC promotion follows.
The original budget and scientific scope were maintained. Review was by Codex,
without an independent second reviewer.
