# Nonlinear LEDH master driver and monograph update — 2026-10-02

Owner request: update and compile the monograph, commit all current work,
synchronize `sqmc-development` and remote `main`, and create a master program
for predator–prey and Austria SIR with 18 state coordinates.

## Research intent and evidence contract

Question: how do the richer residual design, identity-region cap, pairwise
correction and optional safety guard change actual likelihoods and analytical
scores in these two nonlinear models? This delivery builds and checks the
driver; it does not execute or promote a full scientific campaign.

Use the shared `canonical_value_and_analytical_score` executor, its UKF
per-particle covariance lifecycle and total analytical derivatives. The old
`finite_value_standard_score_initial_rqmc` runner is not the comparator's
execution path. All arms use the current executor. Preserve the original
reset as a comparator configuration, not old numerical results.

The constructed simple comparator set is covariance-only reset, marginal-only
moment correction, and the original repeated-axis/capped reset. Compare these
with the richer marginal/pairwise reset and that same reset with the guard.
Report each model, dataset, parameter point, particle count and ancestry route
separately. Marginal-only versus pairwise isolates the need for mixed moments;
covariance-only measures whether higher-moment fitting earns its complexity.

Primary scientific evidence would be error against a verified same-target
likelihood/score reference, including uncertainty. The driver must accept and
validate independently supplied references against model identity, observation
hash, parameter coordinates, theta and horizon. Without such a reference it
reports raw values and paired changes, with oracle error explicitly unavailable;
agreement among filter arms cannot establish accuracy. Never relabel an
approximate, conditional, prior-inclusive, or differently timed score as an
exact observed-data likelihood oracle.

Numerical invalidity, mismatched directional values, mismatched references,
missing outputs and resource/time exhaustion are reported explicitly. Invalid
candidate output is a candidate failure; malformed reference/data invalidates
the comparison. Stop a worker at its time limit and the campaign at its total
budget. Failed workers consume budget and preserve logs. Moment residuals and
timing explain outcomes; they cannot promote a candidate. No default, HMC,
convergence or statistical-superiority claim follows from smokes or this driver.

## Defaults and skeptical audit

| Choice | Provenance and purpose | Failure mode and earliest check | Status |
|---|---|---|---|
| Current canonical nonlinear adapters | Existing analytical implementations; avoid a numerical fork | Wrong dimensions/timing or score wiring; execute both adapters and directional finite differences | Existing implementation, checked call chain required |
| GPU, float32/TF32, XLA, verified memory growth | Repository execution policy | Compilation/resource or roundoff failure; bounded GPU smoke, explicit FP64 reference option | Engineering default, not accuracy evidence |
| T=20, N=1008 | Existing nonlinear horizon and divisible by both 2d values | Settings may not transfer; record each complete scope and evaluate separately | Diagnostic starting scope |
| Synthetic observations from the selected canonical adapter | Same initial law, transition-before-observation and density as the tested model | Accidentally mixing historical y0-first data; preserve generator identity and observation hash | Explicit dataset choice |
| Fixed numerical controls and radius 8 | September diagnostic starting point; paired arms isolate reset changes | KSC settings need not suit d=18; record as untuned, inspect failures, require fresh scope tuning before claims | Warm-start hypotheses only |
| Independent datasets and matched design seeds across arms | Separate data sensitivity from particle-design variability | Pooling hides failures; report per dataset and per coordinate | Diagnostic design |
| External reference optional | Exact observed-data references are not established for these scopes | False oracle agreement; strict scope and parameter checks, explicit missing-reference fields | No oracle claim without checked reference |

Audit: the initial tempting reuse of the old nonlinear runner and its reference
functions was rejected. That runner uses another score path; its reference
helpers return unavailable results. The predator–prey UKF helper also uses a
different observation timing/parameter chart. The new driver must not use these
as same-target oracles. The existing common campaign kernel assumes observation
dimension equals state dimension; SIR has 9 observations and 18 states. Repair
that assumption in the shared kernel, and test it. The predator–prey canonical
adapter needs a dtype argument for the prescribed FP32 execution. Replace its
module's NumPy scalar constants by `math` constants without changing formulas.
This revised plan answers the question without transferring a KSC success claim.

## Implementation and bounded validation

1. Add a small shared nonlinear scope adapter and a master/worker CLI. Emit
   JSON, per-coordinate CSV, readable tables, source/data hashes, full controls,
   device policy, wall times and incomplete/failure states. Use fresh output
   directories; preserve every worker log. Support dry-run planning and explicit
   CPU reference smokes. Keep numerical computation in TensorFlow.
2. Test scope validation, reference alignment, budget/failure handling and the
   SIR 9-observation/18-state call chain. Use at most 1,800 GPU seconds and 600
   CPU seconds for implementation smokes, including compilation. No full T=20
   campaign is authorized by this implementation budget. Scientific execution
   remains a separate recorded invocation with an explicit total budget.
3. Add the T=120 actual-value tables and the likelihood-error/score relation to
   the existing monograph argument. Distinguish September accuracy evidence
   from October guard mechanics and explain the nonlinear next-model tests.
   Preserve existing derivations and synchronize the chapter mirror. Compile
   the full monograph (20-minute build allowance), inspect affected pages and
   check references. Human prose review remains pending.
4. Commit all work belonging to this task and the preceding repair. Fetch
   remote, merge development and remote main in the clean main worktree,
   resolve conflicts by inspecting both changes, run affected checks, commit
   and push main, then fast-forward development to the identical commit and
   push it if it has a remote branch. Verify local and remote commit identities.

Evidence root: `docs/plans/artifacts/ledh-nonlinear-master-20261002/`.
Result/checkpoint: `docs/reset-memos/ledh-nonlinear-master-20261002.md`.
No scientific experiment ranking is planned in this delivery. Focused executable
checks establish wiring and program behavior; broader validity remains open.
