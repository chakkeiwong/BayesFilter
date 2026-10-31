# Lorenz independent-repetition batching repair — 30 September 2026

The Figure6 pilot73 passed12/12 fresh pairs, but its predeclared conservative
forecast is134029.93 GPU seconds (37.23h), exceeding the remaining116809.51s
(32.447h). Warm four-refinement calls take12.8–15.9s; stable tracing counts are1.
The repair hypothesis is that four independent particle systems can share the
existing batch-native matrix kernels, reducing small-matrix execution overhead.
No stochastic or accuracy conclusion follows from these timings.

## Target, comparator and predeclared criteria

Generalize the existing LorenzRefinement controller to replicas=4, retaining
replicas=1 compatibility. Both must call the SAME existing
MultivariateGaussianControlledSMC.forward/.fit, never a reduced numerical copy.
Each repetition retains its own count and strict minESS>90% stopping decision.
Inactive rows are frozen. For subsequent kernel calls only, duplicate a still
active row into inactive work slots (including policy, cloud, weights,
observations and random inputs); discard all such work-slot outputs. This
prevents unnecessary inactive computations from triggering a global validity
failure. It does not change an accepted particle system or its stopping rule.

Before any performance nomination, compare4 simultaneous independent designs
against4 scalar systems at forcing2.5,4.8076923076923075,8.5,101 observations,
source512 particles and maximum4 refinements. Seeds base202609380, per-case
four distinct offsets. The first independent scalar lane must also replay69's
previously validated design at2.5; this guards the replicas=1 refactor itself.
Use unchanged forward tolerance1e-7+1e-9*abs(reference), exact ancestry and
stopping counts. Raw coefficients on separately rounded trajectories remain
explanatory only, as explicitly resolved by71–72. Verify every actually
executed batch fit on identical batch inputs against individual calls to the
same validated fit kernel at the original coefficient tolerance. All rows
must be finite/full rank/positive precision and all stable functions trace once.
The smallest active/inactive case must show differing stopping counts; if the
selected cases do not provide it, a synthetic ESS boundary wiring check is
required. No hidden extra refinements are accepted for stopped rows.

A failure of parity, independent stopping, validity or trace count is a
promotion veto and repair trigger; preserve partial data. Wrong data, corrupt
inputs or total budget exhaustion are continuation vetoes. No tolerance or
criterion relaxation. No pfor, damping, regularization, new precision default
or model change. This remains the explicit FP64/TF32-off source-reference lane.

After parity, measure4 warm repeated batch calls at the three source-grid
pilot points using fresh seed base202609390. Record actual per-system counts,
batch time divided by4, compile cost and memory. The APF kernels are unchanged;
use the worst measured warm APF time from73 and preserve that conservative
assumption. Forecast4000 systems using the largest warm per-system controlled
cost plus73's largest warm APF cost, add40 cold batch compilations and160 cold
APF compilations, and multiply by2 as before. This is a feasibility estimate,
not proof of algorithmic or statistical superiority.

Heuristic set for later science remains BPF512, APF512 and source count-matched
APF, evaluated at every forcing point. Neither parity nor cost can satisfy that
quality gate. Author AnalyseResults.m:32–37 computes var(logZ)/mean(logZ)^2 on
the source's unnormalized logZ. This statistic changes under an additive
likelihood constant; a later Figure6 run must retain BOTH source-convention and
normalized logZ and report convention-invariant raw logZ variance separately.

## Defaults, audit, budget

Batch4 is a bounded feasibility hypothesis: divides100 repetitions and limits
memory. Risk: cross-repetition stopping contamination or rounded ancestry
changes; matched-input fits and per-row scalar replay are the earliest check.
Inactive work duplication is an execution-only device, proven by active-mask
selection and tested by output parity. Existing solver/guards/source model
retain their earlier provenance; graph signatures are fixed. Seed choices are
reproducibility conveniences and cannot select a winning algorithm.

Allocate c2_lorenz_batch_validation450 GPU seconds total (max2 attempts), then
c2_lorenz_batch_pilot450 GPU seconds total (max2 attempts). Retain CPU500/GPU18000
tranche and inherited total balance; ceiling78 accommodates these bounded
repairs. Unique attempt roots and source/input snapshots are mandatory.
Any attempt failure consumes the same phase cap. No new overall grant.

Skeptical audit PASS: unchanged numerical kernels, explicit independent
stopping and identical-input comparison, numerical vetoes before timing,
conservative forecast, no quality promotion from a proxy. Full Figure6 still
requires a separate empirical plan after feasibility is established.

## Attempt74 infrastructure repair

The saved reference random arrays omit the singleton repetition axis. Attempt74
failed at input binding before scalar/batch comparison. Restore that axis in
the harness; no kernel or criterion changed. The failed attempt remains charged
and the retry uses the same450s aggregate validation cap. Syntax and fixture
shape inspection confirm the correction before retry.

## Attempt75 validity diagnosis and bounded retry audit

The scalar compatibility replay passed through all four source refinements, but
the first fresh batch returned valid=False. This is a candidate veto and repair
trigger, not evidence against controlled SMC. No timing pilot may follow yet.
The failed harness discarded fitted-stage diagnostics; preserve its rejected
inputs, per-row Gram ratios and residuals, then replay each row separately and
refit on the exact rejected batch inputs. This distinguishes random-design
failure from a batch implementation defect. Diagnostics cannot promote a run.

Increase the validation attempt allowance from2 to4 within the SAME aggregate
450 GPU seconds (33.015s already spent on74–75). One diagnostic retry is followed
by a repair-validation retry only if a concrete localized cause is found.
No tolerance, solver, stopping criterion, total campaign budget, or tranche
changes. Attempt ceiling78 remains. Syntax checks pass; skeptical audit passes
because the command now preserves evidence that identifies the invalid stage
and distinguishes the two hypotheses, before any performance use.
Metadata correction:73–75 inherited the old fixture middle forcing4.8801 in one
manifest field. The executed code and row summaries used grid index15, namely
4.8076923076923075. Preserve those artifacts and this erratum; new manifests use
the actual grid values and include the scalar-replay seed.

## Current hold after precision-repair attempts77–84

No safeguard passes independent correctness. Covariance projection was finite
in80 but fails11 of112 checks in82; rank guard83 now rejects the unresolved
fit. Batching validation and pilot remain blocked pending a newly validated
policy. See controlled-smc-lorenz-precision-result-20260930.md. Earlier
attempt-ceiling78 text is historical; current84 attempts are consumed.
