# Master program: systematic investigation of KDM, LEDH, and model-score estimators

Date: 2026-09-14

Execution refresh: 2026-09-17. The
[fresh fit-calibration phase](younis-score-iapf-fit-calibration-fresh-2026-09-17.md)
has executed from clean commit `418e5388`: 54 completed numerical rows,
14 rejected underflow fits, and 12 selected-candidate claim rows blocked by
incomplete source studies. Both conditional heuristic tables and all twelve
frozen-control baseline claim rows are complete. The baseline has larger
observed score error than EKF and UKF on every claim dataset; no candidate is
promoted and no statistical ranking is supported. See the
[result and diagnosis](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-fit-calibration-fresh-20260917-01/result-and-refresh.md).

Phase 0E continuation is now implementing and checking the
[relative-shape fitting repair](younis-score-iapf-relative-shape-repair-2026-09-17.md).
Its objective divides profiled residual squared norm by density squared norm;
it is an explicit Algorithm-3 adaptation, different from the published Eq. (15)
retained as comparator. The source anchors, analytical quotient derivative,
failure conditions and 32-row fresh-data experiment are written before execution.
The maximum 104 charges fit within the existing 109-charge remainder.
Fitted-moment integration into LEDH, wider model coverage and
powered replication remain open. The whole master is incomplete. Current
execution details and the exact restart action are in the
[active checkpoint](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/checkpoint.md).

## Purpose

This program investigates how to estimate the observed-data model score

\[
 S(\theta;y_{1:T})=\nabla_\theta\log p_\theta(y_{1:T})
\]

when a particle filter uses LEDH, KDM, GenUT, UKF, continuous resampling, or
optimal transport. The program is deliberately organised around the target
quantity. It does not treat a differentiable finite filter, a KDM expectation,
an unnormalised likelihood derivative, and the marginal score as interchangeable
objects.

The central research question is:

> Which proposal, score identity, resampling representation, and variance-
> reduction method gives the smallest scientifically relevant error for the
> model score at a declared compute budget, while remaining valid for the
> transition-support class of the model?

The applicability record distinguishes two non-interchangeable tracks:

1. **Regular-transition track.** Transitions admit densities on a common
   reference measure. This is the setting in which model-corrected mixture
   proposals, Fisher identities, backward pair estimates, Rao--Blackwellization,
   and Younis-style KDM calculus can be tested.
2. **Degenerate-transition track.** Transitions are deterministic or singular
   conditional on an ancestor, as in the DSGE target. Ordinary ambient-space
   Gaussian-mixture density ratios and regular-transition smoothing arguments
   are not assumed valid. A support-aware score identity must be derived before
   implementation or comparison.

No result from the regular track may be presented as evidence that the
degenerate track is solved.

**Active execution scope, amended 2026-09-14.** This program now executes
KDM/IWSG filtering and estimator combinations, proposal/covariance alternatives
(UKF, KDM, SGQF), twisting/iAPF, and coupled directional finite differences.
Regular models provide tractable reference tests for these mechanisms.
Regular-transition smoothing and degenerate/DSGE score derivations are assigned
to separate programs. Phases 5 and 6 and the corresponding literature rows
remain reference material; they are not generated as executable rows here and
are not prerequisites for the active KDM filtering study. Rhee--Glynn/JLS,
Nemeth, PaRIS, and other smoothing estimators are deferred with that work.
An imported result needs target, support, and provenance checks before use.

The changes below incorporate the mathematical corrections agreed in Claude's
follow-up. Claude's bounded Phase 4C review gave a REVISE verdict and marked
other sections unchecked; it did not approve those sections. The dependency,
tuning, and SGQF amendments are additional findings from this program review.

This master includes specification repair, implementation, testing, tuning,
experimental execution, analysis, and between-phase repair. Phases 0A--0G
below build the prerequisites for the scientific comparisons. Running the
master begins with those development tasks; it does not require a separately
executed implementation program. The former E0--E7 execution plan is
superseded by the phases and dependencies in this document.

The initial phases are performed by the supervising developer or agent with
the repository's existing development tools. They build the coordinator that
later runs numerical rows. They cannot require that coordinator to exist
before starting. All phases, including implementation and terminal review,
use the repair and next-phase refresh procedure in this master.

### Program milestones and starting status

| Milestone | Work inside this master | Completion evidence |
|---|---|---|
| Begin master execution | Read Phase 0's scientific requirements and start Phase 0A, followed by 0B and 0C. | Source/specification reconciliation and implementation work have begun; no pre-existing master CLI is required. |
| Baseline study runs | Complete 0A--0C and applicable Phase 1 admission checks. | Real Gaussian baseline, oracle, scoped tuning, report, and repair/resume tests pass. An oracle-only or fake-endpoint smoke is insufficient. |
| Active method matrix runs | Complete the applicable 0D--0G implementations and Phase 1 checks for every active family. | Actual method endpoints, controls, and reports execute under the required backend; missing integrations remain recorded as incomplete work. |
| Research program completed | Execute applicable Phases 2--4C and 7--9, repairs, uncertainty analysis, and final dispositions. | Evidence supports the reported positive, negative, or unresolved result for every planned row. Software availability alone is not scientific completion. |

Execution began on 2026-09-14. The coordinator, Gaussian providers, canonical
adapter, scoped selection/consumption, and diagnostic reporting now exist in
`bayesfilter/score_study/`, with a real CLI and versioned study specifications.
Seven baseline rows have run on CPU/XLA and GPU/XLA; full six-parameter
derivatives include the initial distribution. These are mechanics fixtures,
not evidence of score improvement. The active execution checkpoint is
[checkpoint.md](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/checkpoint.md).
KDM integration, persistent covariance alternatives, fixed and locally fitted
twists, finite differences, normalization reports, and nonlinear comparison
endpoints now execute. Bounded density-objective iAPF fitting and adaptive
iteration and adaptive-N selection/reporting now pass scalar Gaussian CPU/GPU
mechanics checks. The scalar nonlinear iAPF extension now passes twenty GPU
rows and analytical derivative/consumer checks. Comprehensive control calibration
and the broader research matrix remain work inside this master.
A missing method cannot be removed to declare the full matrix implemented.
The earlier [recursive fit-control calibration](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-fit-calibration-result-and-refresh.md)
closed partial at 64/64 charges; its invalid rows remain historical failure
evidence. The [underflow-guard repair](younis-score-iapf-underflow-guard-2026-09-16.md)
rejects zero density objective with positive shape residual; the isolated and
main fitter suites each pass 19 tests. This admission repair is complete.

The [fresh calibration result](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-fit-calibration-fresh-20260917-01/result-and-refresh.md)
now supplies both weak and curved heuristic tables and the frozen-control
baseline on all six untouched claim datasets with two independent final streams.
It used one GPU launch and 171/280 charges. Fourteen wider-candidate source
rows fail the guard; their solver-converged flags do not make them valid.
Both selected-candidate studies remain incomplete and cannot issue tuning
artifacts. Baseline-only selection succeeds, but baseline score errors exceed
EKF and UKF on every claim dataset, and fitting boundaries are active on ten
of twelve baseline claim rows. The source/result audit passes. No candidate
is promoted and no ranking is statistically supported. This is a fitting
repair trigger, not rejection of twisting or a harness failure. Reconcile
objective scaling, constraints and stopping with the source specification,
then execute fresh scope-specific calibration before iAPF-moment integration
into LEDH. Seven new campaign-driver tests pass in each checkout.
The earlier implementation slice is [nonlinear iAPF](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/nonlinear-iapf-result-and-refresh.md): twenty GPU rows, thirty frozen-source checks and the integrated consumers pass. Exact conditional twisting and frozen-fit derivatives are derived in manuscript Section 7.3. The observed score errors exceed EKF/UKF errors in both tested regimes; no ranking or promotion is supported. Its fitting controls now require the repair identified above. The preceding [adaptive-N selection/reporting](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-adaptive-scope-result-and-refresh.md) includes an actual 16-to-32 count change. The preceding [control diagnostics and safety screen](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/control-safety-result-and-refresh.md) completed:
108 comparison rows and eight GPU smoke rows complete after a validator-only
coordinate-cap repair. The inherited cap substantially compresses intermediate
coordinates; larger caps reduce that compression but do not rescue observed
score errors. All tested controls fail the descriptive heuristic screen in
weak and concentrated regimes. These two-dataset results select no default.
The diagnostics and repaired validator are integrated and seven shared-checkout
consumer tests pass. Complete control calibration remains open.
The preceding [bounded iAPF implementation and repair](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-result-and-refresh.md) records:
eight GPU rows and 12 directional checks pass after an explicit offline
FP64 fitting repair; final filtering and analytical scores remain FP32/TF32/XLA.
It establishes mechanics only. The preceding [1,440-row nonlinear covariance pilot](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/nonlinear-calibration-result-and-refresh.md)
passes numerical/reference and provenance checks, but all three covariance
candidates fail the observed conditional heuristic promotion screen. The
whole program remains incomplete; these are candidate-level results.

Concurrent edits to the shared canonical executor interrupted the first
numerical launch during provenance inspection. Subsequent runs use the isolated
checkout `/tmp/bayesfilter-younis-score-execution-20260914`, based on
`e7f2a88ecff49ced481b9615c0b1237b8cabe732` plus recorded repairs. Source closures
include package initializers; numerical evidence is tied to that checkout.
An XLA-elided assertion also required an explicit invalid-value propagation
guard at the final GenUT reset. Healthy and injected-invalid regressions pass.

## Project implementation constraints

The master program inherits the repository's active implementation contracts.
The claim-bearing LEDH baseline uses Contract E--Chol and its declared
analytical total-derivative composition. Autodiff, finite-program JVPs, and
GradientTape may be used for parity and debugging, but they are not a
claim-bearing score route. Active transport rows use the repository's exact
divisor chunk policy, and TensorFlow/TFP with the configured GPU, TF32, XLA,
and memory-growth settings is the default execution target. NumPy is confined
to independent references, tests, and post-run diagnostics. A candidate that
violates one of these contracts is either repaired or labelled diagnostic; it
does not enter the score leaderboard by changing a CLI flag.

These engineering constraints do not decide the scientific question. A route
can satisfy the implementation contract and still estimate the wrong target or
lose to a simple heuristic. Conversely, a mathematically interesting reference
route can remain diagnostic until its backend and derivative contracts are
implemented.

## Program architecture

The implementation should be a typed experiment system with one coordinator
and separate scientific components. This keeps a KDM expectation gradient from
being combined with a marginal score without detecting the target change.

### Registries

The coordinator reads five repository-owned registries:

1. **Model registry:** model equations, data generator, support class,
   parameterisation, initial law, transition representation, observation law,
   and oracle provider.
2. **Target registry:** exact marginal score, finite-program derivative, KDM
   expectation gradient, unnormalised derivative, or another explicitly named
   target. Each target declares its measure, normalisation, and valid support
   class.
3. **Proposal registry:** bootstrap, adapted, LEDH, KDM, forward/backward, OT,
   twisted PF, iAPF, SGQF-guided, exploration mixture, or coordinate proposal.
   Each proposal exposes both its sampler and its evaluable density or mass
   correction.
4. **Estimator registry:** Fisher, backward-pair, IWSG, pathwise, hybrid,
   Rao--Blackwell, ratio, or control-variate estimator. Each estimator declares
   the quantities it requires and the target it returns.
5. **Tuning registry:** scope identity, calibration partitions, candidate
   controls, selected controls, and the untouched claim partition.

Registry validation must reject an experiment before tracing or sampling when a
proposal, estimator, target, and support class are incompatible. In particular,
an ambient Gaussian KDM proposal cannot be registered for a degenerate model
under a regular-transition score identity.

Every registered row also has an execution status: active, deferred, or blocked
on a named prerequisite. The matrix generator expands active eligible rows
only. Listing a method in the registries, factorial axes, or literature table
does not authorize its execution.

### Matrix generator

The matrix generator expands a declarative study into rows over model, regime,
horizon, particle count, proposal, score estimator, variance-reduction method,
dtype/backend, and seed. It first generates the baseline ladder, then all
single-factor and pairwise interactions, and finally the full factorial of
surviving candidates. This gives exhaustive coverage without allowing an
unvalidated option to acquire promotion status merely because it appeared in a
large pooled sweep.

Each row is immutable after tuning. The claim runner consumes a repository-issued
tuning artifact and refuses a missing, stale, cross-model, or cross-horizon
artifact. Calibration and validation runs never write into the claim directory.

### Execution lanes

Use distinct lanes for:

- oracle construction;
- fixed-cloud estimator comparison;
- full-filter proposal comparison;
- long-horizon and particle-count scaling;
- externally supplied support/oracle evidence, imported read-only; and
- report assembly.

The fixed-cloud lane isolates conditional score-estimator variance. Repeat
over independent clouds to measure variation in the cloud itself; conditioning
on one cloud does not establish marginal-score accuracy. The full-filter lane
measures the combined effect of proposal, resampling, and score recursion. Their
outputs must not be merged into one score table without a lane label.

### Coordinator pseudocode

This is the numerical scheduler to implement in Phase 0B. It operates after
the initial development phases have created its services. Implementation and
mathematical repair remain supervised master tasks; their tests and resulting
capability status feed this scheduler.

```text
load model_registry, target_registry, proposal_registry,
     estimator_registry, tuning_registry
validate_registries_and_supports()

candidate_status = initialize_active_candidates_and_predeclared_repairs()
phase_state = load_current_plans_dependencies_and_budget()
while task := next_ready_task(active_tasks, candidate_status, phase_state):
    require_current_phase_plan_and_applicable_boundary_dispositions(task)
    if task.requires_development:
        complete_supervised_implementation_and_required_tests(task)
    matrix = generate_eligible_rows(task, candidate_status, declared_scopes)
    record_blocked_and_excluded_rows_with_reasons(task)
    for row in matrix:
        validate_target_measure_support(row)
        verify_named_prerequisites(row)
        if row.is_quality_claim:
            tuning = prepare_and_freeze_scope_on_allowed_partitions(row)
            validate_tuning_scope(tuning, row)
            require_repository_issued_tuning_artifact(tuning)
        for dataset in row.declared_partition:
            oracle = model_registry[row.model].oracle_provider.get(dataset, row.parameter)
            for seed in row.declared_seeds:
                result = execute_one_row_recording_success_or_failure(row, dataset, seed, oracle)
                write_result_to_new_attempt_directory(result)
    decision = aggregate_under_declared_evidence_role(task.results)
    apply_vetoes_and_heuristic_gate(decision)
    issues = classify_validity_defects_candidate_failures_and_uncertainty(decision)
    repairs = select_in_scope_repairs(issues, phase_state.remaining_budget)
    complete_available_repairs_and_impacted_regressions(repairs)
    invalidate_affected_evidence_and_recompute_decision(task)
    record_unresolved_issues_and_block_only_dependent_rows()
    update_candidate_status_from_verified_evidence()
    refresh_next_phase_plans_from_actual_results_or_terminal_disposition()
    audit_refreshed_dependencies_scopes_partitions_tests_commands_and_budget()
    write_phase_closeout_and_checkpoint()
    phase_state = reload_current_plans_dependencies_and_budget()
write_complete_program_row_accounting_including_unresolved_work()
```

The coordinator may schedule independent rows in parallel, but each row must
retain the exact command, environment, device, random seed, tuning scope, and
source revision needed to reproduce it. Parallel scheduling must not change
random-number coupling within a paired comparison.

Repair completion is a workflow obligation, not a promise that the coordinator
can invent or fix numerical methods automatically. The supervising developer
or agent performs source/mathematical repairs; the coordinator records source
changes, executes declared checks/retries, and refuses reuse of affected stale
evidence. A phase with remaining required repairs stays partial or blocked.

### Result schema

Every row writes machine-readable JSON plus a human-readable Markdown note with
at least:

```text
program_id, phase_id, phase_plan_version, row_id, attempt_id, git_commit
model_id, dataset_id, regime_id, parameter_id, horizon, particle_count
support_class, measure_id, target_id, score_kind, normalization_kind
proposal_id, resampling_id, estimator_id, variance_reduction_id
reset_id, derivative_id, tuning_scope_id, tuning_artifact_hash
dtype, backend, jit_compile, device, memory_policy, chunk_policy
seed, paired_seed_group, wall_time, peak_memory
oracle_id, oracle_status, score_error, value_error, diagnostics
hard_veto_status, heuristic_verdict, inference_status, artifact_paths
execution_status, blocked_prerequisites, repair_status, invalidated_evidence
```

The aggregation layer computes paired confidence intervals, bias/variance/MSE
decompositions, and the decision table. Diagnostic plots retain failed rows
with their status; rankings require passed vetoes, target compatibility, and
uncertainty evidence.

### Factorial axes

The active study matrix crosses the following axes, subject to compatibility
and prerequisite checks. The broader option table below retains deferred
literature coverage separately.

| Axis | Levels to enumerate |
|---|---|
| Target | exact marginal score; finite-program derivative; KDM expectation gradient; unnormalised value/derivative; declared approximation |
| Proposal | bootstrap transition; EKF; UKF; LEDH; physical transition mixture; KDM; fixed lookahead mixture; twisted PF; iAPF; SGQF-guided; defensive mixture |
| Resampling/coupling | multinomial/systematic discrete resampling with declared derivative target; continuous KDM mixture draw; OT with a derived correction or changed-target label; OT cloud plus explicit jitter; OT as coupling between replicas |
| Covariance | model (Q); UKF covariance; KDM weighted covariance; LEDH local covariance; GenUT-restored covariance; SGQF projected covariance; calibrated hybrid choices |
| Score identity | analytical finite scalar derivative; IWSG expectation gradient; analytical/IWSG hybrid; ratio of unnormalised estimates; finite-difference diagnostic |
| Variance reduction | none; conditional Rao--Blackwell; exact-mean control variate; independently calibrated estimator combination; declared coupling of finite differences |
| Support | regular common-density; bounded/constraint regular with checked support and differentiation conditions |
| Evaluation lane | fixed-cloud estimator with independent-cloud replication; full filtering recursion; directional diagnostics; long-horizon scaling |

The generator records why each combination is omitted. “Not applicable” and
“not yet derived” are distinct statuses; silently dropping an incompatible row
would make the final option inventory incomplete.

## Non-negotiable target definitions

Every implementation and result must label which of the following it computes.

### Exact model score

\[
 Z(\theta)=p_\theta(y_{1:T}),\qquad
 S(\theta)=\nabla_\theta\log Z(\theta).
\]

For a regular state-space model, Fisher's identity is

\[
 S(\theta)=\mathbb E_\theta\left[
 \nabla_\theta\log \mu_\theta(X_0)+
 \sum_{t=1}^T\{
 \nabla_\theta\log f_\theta(X_t\mid X_{t-1})+
 \nabla_\theta\log g_\theta(y_t\mid X_t)\}
 \mid y_{1:T}\right].
\]

The identity requires the stated support and differentiation conditions. For a
singular transition, its density notation is not used until a valid
ancestor/innovation base measure has been supplied.

### Finite-program derivative

\[
 L_N(\theta;\xi)=\text{the scalar produced by one declared finite program},
 \qquad
 S_N^{\mathrm{fin}}=\nabla_\theta L_N.
\]

This derivative is a valid derivative of the finite program only when all
declared dependencies are differentiated. It is not automatically an estimate
of \(S(\theta)\).

### KDM expectation gradient

For a mixture \(m_\theta\) and integrand \(F_\theta\),

\[
 \mathcal J(\theta)=\int F_\theta(z)m_\theta(z)\,dz.
\]

Assume a common support, differentiability, integrable domination permitting
differentiation under the integral, and a positive sampling density wherever
the derivative integrand contributes. With \(q_0=m_{\theta_0}\) held fixed
while differentiating,

\[
 \mathbb E_{q_0}\left[
 \nabla_\theta F_\theta(Z)|_{\theta_0}
 +F_{\theta_0}(Z)\nabla_\theta\log m_\theta(Z)|_{\theta_0}
 \right]
 =\nabla_\theta\mathcal J(\theta)|_{\theta_0}.
\]

This is the scope of the Younis IWSG identity. If \(F=\log g(y\mid z)\),
the target is an expected conditional log likelihood. In general,

\[
 \mathbb E_m[\log g(y\mid Z)]
 \neq \log\mathbb E_m[g(y\mid Z)].
\]

### Unnormalised pair and normalised score

The program must retain separately

\[
 \widehat Z_N,\qquad \widehat D_N\approx\nabla Z,
 \qquad \widehat S_N=\widehat D_N/\widehat Z_N.
\]

Unbiasedness of the first two does not imply unbiasedness of the third.

## Evidence contract

The primary comparison question is conditional model-score accuracy for a fixed
dataset and parameter, averaged over independent particle randomisations. The
exact comparator is an analytic or numerically certified score oracle. When no
oracle exists, the result is restricted to identity checks, convergence
evidence, and descriptive comparisons; it is not called unbiased or accurate
without a derivation.

The primary promotion criterion is a predeclared paired error measure against
the oracle, such as squared Euclidean score error or a fixed scale-normalised
version. Choose the metric before selection and account for oracle uncertainty
when it is nonzero. Confidence intervals respect independent datasets and the
particle replicates nested within them; the latter are not independent
datasets. Any claim of improvement needs its predeclared uncertainty criterion,
including an effect-size margin or precision target where relevant.
Required vetoes are:

- non-finite values, invalid supports, negative or zero proposal densities where
  the numerator has mass, failed derivative identities, or failed likelihood
  normalisation checks;
- a mismatch between the implemented target and the claimed target;
- missing initial-law, transition, observation, mixture, Jacobian, or total
  derivative terms;
- use of a regular-transition identity on a degenerate transition without a
  support derivation;
- data leakage from tuning or control-variate fitting into the claim run; and
- a failed heuristic-dominance check in any salient regime.

Explanatory diagnostics include bias/variance decomposition, ESS, weight tails,
ancestor entropy, mixture overlap, bandwidth sensitivity, covariance calibration,
OT marginal error, and runtime. They cannot replace the primary criterion.

Before each research run, record the main question, mechanism, expected
failure, promotion criterion, promotion veto, continuation veto, repair
trigger, explanatory diagnostics, and forbidden inference. Classify every
diagnostic by these roles. A failed candidate is not a continuation veto when
a planned repair addresses that failure. A model-score error claim and a
finite-program derivative claim require separate evidence even if they use
the same numerical estimates.

Passing this program does not establish posterior correctness for an untested
model, HMC readiness, asymptotic unbiasedness, or superiority outside the
declared scope.

## Campaign envelope and stop rules

### Initial implementation and validation allocation

The first tranche of this master is Phases 0A--0C plus Phase 0F's
deterministic mechanics. It includes implementation, tiny CPU reference
checks, and a bounded GPU integration check. The initial engineering
validation allocation is 12 CPU process-hours and 8 GPU device-hours,
including compilation and failed runs, with at most 12 GPU launches.
Reserve 2 of those GPU hours for repairs. These are planning allocations,
not measured costs or scientific precision thresholds; use the first timing
observation to reassess the remaining allocation and record exhaustion
instead of weakening a check.

Phase 0A pins the available interpreter/environment and exact commands for
the next phase. It does not guess a conda environment or install packages.
GPU checks use trusted/escalated access and verified memory growth; CPU
references explicitly set `CUDA_VISIBLE_DEVICES=-1`. Later phase refreshes
record their finite row counts, replications, compute caps, and attempt
budgets before serious launches. Establishing those values is a task inside
the master, not a request for a separate preparatory program.

### Scientific comparison allocation

“Budget and time are not a problem” permits exhaustive coverage, but it does
not remove the need for a finite, reproducible campaign definition. The initial
claiming envelope is:

- at least 100 independent observation datasets per regular model/regime;
- 64 independent particle randomisations per dataset for the first claim run,
  with a maximum of 256 if the paired interval remains too wide;
- particle counts \(N\in\{32,64,128,256,512,1024,2048\}\) and horizons
  \(T\in\{1,5,20,50,100\}\) where the model supports them;
- all baseline and identity rows, all single-factor ablations, and the complete
  factorial only for candidates that pass the preceding phase gates; and
- no replacement of a failed artifact: each repair or retry gets a new,
  versioned output directory.

These values are an explicit planning envelope, not inherited scientific
defaults. A row may stop early when a hard veto fires, when its target identity
fails, or when a repeated implementation failure has an established cause.
Stop unchanged retries and enter the phase-boundary repair process; this is
not a veto on a justified repair or an independent row. Three unsuccessful
repair trials for one cause require diagnostic reassessment and an explicit
remaining-budget disposition. A viable row may stop replication when its predeclared paired
interval is sufficiently narrow for the decision question; otherwise it is
reported as descriptive-only at the maximum replication count. A candidate
that fails a promotion criterion but has a planned repair continues to that
repair; it is not silently discarded as evidence against the whole method.

Before a serious launch, its concise run plan must fill in the finite number
of eligible rows, maximum attempts, total wall/GPU-hour budget, hardware,
uncertainty precision needed, and stop conditions. The large grid above is a
coverage proposal, not an instruction to launch its Cartesian product. A pilot
determines feasible allocation and power; its outcomes cannot select claim
data. Any optional replication increase uses a predeclared sequentially valid
interval/stopping rule or a fixed terminal sample size. Budget exhaustion
leaves an unresolved result; it does not relax a scientific criterion.

The campaign root is a unique versioned directory under
`docs/plans/artifacts/younis-kdm-score-master-20260914/`. Every launch records
the exact matrix manifest, git revision, environment, hardware, seeds, command,
wall time, and output hashes. The campaign is closed only after each planned
row is classified as promoted, viable-but-underranked, failed-with-repair,
failed-support, or unresolved-evidence.

## Skeptical pre-mortem

Assume the program produces an apparently favourable KDM or LEDH result. It
could still be misleading for the following reasons:

- the implementation differentiates the finite cloud while the claim is about
  the marginal likelihood; the earliest check is target metadata plus the
  analytic linear-Gaussian score;
- the KDM bandwidth or OT reset changes the propagated measure and the apparent
  gain is a target change; the earliest check is a physical-numerator,
  evaluable-denominator identity test;
- the score error is dominated by a random denominator, so a lower variance
  expectation gradient is irrelevant to the reported score; the earliest check
  is the separate \(\widehat Z_N\), \(\widehat D_N\), and ratio study;
- a missing initial or transition derivative is hidden by a fixed-cloud finite
  difference; the earliest check is a parameter-dependent initialization test
  and an independent exact Gaussian derivative fixture;
- tuning or control-variate fitting leaks information from the claim set; the
  earliest check is a scope-bound manifest with disjoint data hashes;
- a complex route loses to a cheap UKF, adapted PF, or bootstrap method in a
  salient regime while its pooled mean looks favourable; the earliest check is
  the conditional heuristic table; and
- a regular-transition success is reported as a DSGE result; the earliest check
  is the support-class field and a mandatory derivation reference for every
  degenerate run.

Any one of these findings blocks promotion until the corresponding repair is
completed. A failed candidate remains useful evidence about that candidate; it
does not by itself terminate the broader program.

## Option matrix

Every active candidate requires a complete estimator and target label.
In this inventory, Fisher/backward smoothing (including Nemeth, PaRIS,
Fearnhead, JLS), iterated filtering, continuous-likelihood methods, multilevel
debiasing, variational objectives, and degenerate coordinates are deferred
literature/reference rows. Their presence does not create executable tasks.
KDM/IWSG filtering, covariance/proposal alternatives, twisting/iAPF, and FD are
active subject to their named prerequisites.

| Family | Candidate | Intended role | Main question |
|---|---|---|---|
| Baseline | Bootstrap PF with a declared value/score route | classical reference | How much error comes from ordinary particle approximation? |
| Baseline | Locally adapted proposal with exact importance correction | tuned classical reference | Does simple observation adaptation explain any gain? |
| Baseline | EKF and UKF approximate-likelihood scores | cheap heuristic adversaries | Is low-order Gaussian moment information sufficient? |
| LEDH | Canonical LEDH-PF-PF-OT with analytical recursive score | project baseline | What does the existing finite algorithm achieve? |
| KDM | KDM proposal with physical transition numerator and evaluable proposal denominator | model-corrected proposal | Does KDM improve coverage without changing the target? |
| KDM | KDM expectation/IWSG gradient | declared expectation objective | What is its variance and bias for its own objective? |
| KDM/Rao--Blackwell | Nemeth--Fearnhead--Mihaylova KDE plus Rao--Blackwell score and observed information | direct regular-model score comparator | Does the published linear-cost kernel estimator improve horizon variance under its bandwidth and mixing assumptions? |
| Covariance | UKF per-particle covariance, KDM weighted covariance, and hybrid covariance | proposal-shape study | Does a better covariance estimate improve the score through proposal quality, or only change a finite approximation? |
| Fisher | Forward particle score with complete-data terms | direct score estimator | Can transition and observation score terms be propagated stably? |
| Smoothing | Backward pair/Fisher estimator | regular-transition only | Does ancestor averaging reduce score variance or finite-\(N\) bias? |
| Smoothing | Fearnhead--Wyncoll--Tawn sequential smoother | regular-transition only | How do linear- and quadratic-cost additive-functional smoothers compare? |
| Smoothing | PaRIS with explicit backward-draw count | regular-transition only | Does \(\widetilde N\ge2\) control path-degeneracy variance at the target horizon? |
| Debiasing | Coupled conditional particle filter with Rhee--Glynn correction | regular tractable-transition only | Can unbiased smoothing expectations and Fisher scores be obtained at acceptable random cost? |
| Rao--Blackwell | Analytically integrate tractable state blocks or ancestor indices | variance reduction | Which conditional integrations are exact and worthwhile? |
| Control variate | Exact-mean control variate for unnormalised derivative | variance reduction | Does known centering reduce variance while preserving expectation? |
| Control variate | Tractable reference plus residual | variance reduction | Does a separately corrected approximate model help? |
| Hybrid | Pathwise plus importance-weight gradient | gradient variance candidate | Can the two estimators be combined without changing the target? |
| Combination | Calibrated linear combination of two score estimators | bias--variance study | When do two biased but target-matched estimators have lower MSE? |
| Multilevel | Coupled \(N\)/bandwidth or randomized telescoping estimator | bias reduction | Is there a verified expansion supporting debiasing? |
| Proposal correction | Forward/backward mixture proposal with generative correction | proposal design | Can future-data coverage help while preserving the model measure? |
| Continuous likelihood | Malik--Pitt/DeJong-style continuous particle likelihood | continuous-resampling comparator | Which discontinuity is removed, and what finite-particle measure and derivative does the method target? |
| Simulation-only sensitivity | Ionides iterated filtering | unavailable-transition-density comparator | What bias, variance, and mixing cost arise as artificial parameter noise decreases? |
| Iterated APF | Guarniero--Johansen--Lee twisted/auxiliary filter | proposal-quality comparator | Does future-data twisting reduce normalizer and downstream score variance when the analytical score recursion is held fixed? |
| Proposal variance | Whiteley--Lee twisted particle filter | proposal-only variance control | Can normalizer variance be reduced without changing the physical target correction? |
| Proposal moments | Fixed structured Gaussian quadrature filter (SGQF) moments | high-dimensional proposal-design comparator | Do sparse-grid projected moments improve proposal coverage at an admissible cost without replacing the physical target? |
| Directional validation | Three-point, five-point fourth-order, and eleven-point cubic finite differences | oracle-calibrated score diagnostic | Which fixed positive \(h\)-ladder gives the best bias--variance tradeoff for each directional derivative? |
| Discretization | Jasra--Kamatani--Law--Zhou multilevel particle filter | coupled-level variance control | Does a score-level telescope exist for the chosen discretization and coupling? |
| Degenerate | Ancestor/innovation-coordinate estimator | unresolved support-aware route | Can a valid base measure and score identity be derived for DSGE? |

## Phase 0: specification and oracle ladder

This phase states the scientific specifications for the implementation phases
that follow. Phases 0A--0C build and verify the catalogue, oracles, baselines,
and tuning services specified here. Phase 0 does not assume those services
already exist, and Gate A is issued only after the applicable implementations
and tests pass.

### 0.1 Build a model catalogue

Use a minimum ladder of:

1. scalar and multivariate linear-Gaussian state-space models with exact Kalman
   value and score;
2. nonlinear regular-transition models with high-accuracy reference scores;
3. models with multimodal or strongly curved posteriors;
4. the existing repository models used by the LEDH adapters; and
5. the degenerate DSGE model as a deferred applicability record, without
   scheduling its support derivation or experiments in this program.

Record state dimension, parameter dimension, horizon, transition rank,
observation rank, support, analytic oracle availability, and the model's
parameter-dependent initial law.

### 0.2 Construct the oracle ladder

For each model, implement in order:

1. closed-form value and score where available;
2. deterministic quadrature or Rao--Blackwellised integration;
3. high-particle, replicated reference estimates with an error certificate;
4. an independent implementation using different code and random numbers; and
5. externally supplied support-specific references when applicable, with
   provenance and applicability checked before import.

Use the highest justified reference available; an unavailable closed form
does not block construction of a valid next rung. Replication at high particle
count bounds Monte Carlo uncertainty, not approximation bias by itself.
A numerical certificate must also bound relevant discretization/particle bias;
otherwise label the reference provisional and restrict the resulting claim.

An oracle is admitted only after value, directional derivative, finite-difference,
and parameter-coordinate checks agree within predeclared tolerances.

### 0.3 Freeze notation and target metadata

Every result records `target_id`, `measure_id`, `support_class`, `score_kind`,
`normalization_kind`, `proposal_id`, `reset_id`, `derivative_id`, and whether the
proposal is detached locally. A result with missing metadata is diagnostic only.

### 0.4 Prepare baselines and tuning before quality comparisons

Build and verify the exact Gaussian oracle and the applicable cheap baselines
before the first proposal or estimator comparison. For an active regular model,
the minimum ladder is bootstrap PF, an EKF/UKF approximation, a classically
adapted proposal where evaluable, canonical LEDH, and the proposed enhancement.
The Kalman oracle measures error in Gaussian fixtures; it is not a Monte Carlo
competitor that a particle method must beat. Define the heuristic adversaries
and salient regimes from Phase 7 now, but reserve their final evaluation as a
falsification check rather than a tuning objective.

Scope-specific tuning is a reusable prerequisite of every quality claim in
Phases 2, 3, 4, 4B, 4C, and 7; it does not wait until Phase 8. Bind model,
target, horizon, particle count, dimensions, dtype/TF32, backend, chunk policy,
proposal/estimator route, and all controls, including bandwidth, covariance
protection, twisting fit, finite-difference span/directions, and combination
coefficients. A horizon or particle-count change creates a new scope.
Partition calibration, validation/selection, and untouched claim observations
and particle streams in advance. Fit or choose controls on permitted partitions,
freeze them, and require the repository-issued scope artifact in every claim
row. Mechanics and debugging fixtures need no statistical tuning artifact but
cannot be used for quality claims.

For each material numerical default, record provenance, justification, likely
failure mode, smallest diagnostic, and status as baseline, hypothesis, or
reviewed choice. In particular, imported bandwidths, ridges, covariance floors,
lookahead families, step sizes, and conditioning limits are hypotheses until
justified for this scope. Phase 8 performs final replication and cost analysis.

### Implementation defaults and early diagnostics

Apply this audit while implementing Phases 0A--0G. The later experiment plan
refresh specifies target-specific values before the corresponding run;
unexamined inherited settings cannot become defaults through a passing smoke.

| Choice | Provenance and purpose | Failure mode and earliest check | Status |
|---|---|---|---|
| Regular Gaussian first | Phase 0; exact score separates target and implementation errors. | Apparent success hides missing initial-law derivatives; use a nonconstant-initialization fixture. | Reference baseline, not cross-model evidence. |
| Canonical analytical LEDH/GenUT/Contract E | Current owner policy. | A simplified lane appears to pass; test the actual consumer and reset composition. | Required baseline identity, with correctness still to verify. |
| Tiny smoke allocation and first-tranche caps | Master implementation allocation; make the first failure cheap and reserve repair time. | A smoke is misreported as a ranking, or compilation exhausts allocation; record no-ranking status and early timing. | Convenience allocation, not a scientific threshold. |
| Existing tuning services and kernels | Repository reuse candidates. | Their API exists but cannot issue/consume the required scope; exercise the complete path. | Unverified until tested. |
| Step sizes, bandwidths, ridges, twist families, and covariance controls | Selected derivations plus per-scope calibration. | Bias/robustness or target changes are concealed by numerical success; log realized choices and earliest validity/curve checks. | Hypotheses until justified, including off/zero settings. |
| Statistical precision and oracle tolerances | Exact identities, dtype/scale analysis, reference convergence, and powered pilot. | Arbitrary cutoffs or replication-driven selection; validate error scale and freeze a valid uncertainty rule. | Must be specified in the refreshed run plan, never inferred from a successful smoke. |

## Phase 0A: reconcile once, then build

Entry: Phase 0's scientific requirements and the current checkout. This is the first implementation task of the master and needs no master CLI.

1. Read the current canonical repair result and inspect the actual callable
   interfaces. Record each capability as verified, unverified, blocked, or
   deferred, with its first executable test. Refresh stale source claims.
2. Reconcile the manuscript's smoothness requirement, deterministic versus
   stochastic FD checks, and SGQF prediction/conditioning/reset description
   with the amended master. Preserve the existing derivations and source
   boundaries. Fix the malformed `,qquad`, build, and inspect the affected pages.
3. Define the exact numerical return contract: log likelihood/value target,
   derivative target, coordinate system, initial-law dependence, denominator,
   proposal law, and oracle precision. Specify which outputs are model-score
   estimates and which are derivatives of a different quantity.
4. Inventory reuse before adding modules. Identify real production/candidate
   baseline endpoints; diagnostic reference functions remain reference lanes.
   Verify paper equations and author code for method-faithfulness claims when
   their numerical implementation is selected. Do not wait for a new broad
   literature survey to build the coordinator.

Exit: no unresolved ambiguity about the first baseline's mathematical target
or the 0B interface. Remaining method gaps have named tests and block only
their dependent rows. Refresh 0B/0C using the current repair status.

## Phase 0B: implement a small coordinator that cannot hide failures

Entry: Phase 0A's target/return contract. This phase builds the infrastructure used by later numerical rows.

Proposed new locations, to be reconciled with existing services in 0A:

- `bayesfilter/score_study/`: typed study/row definitions, registry
  validation, adapters, phase state, and report assembly;
- `scripts/run_younis_score_master.py`: a thin CLI over those services;
- `docs/plans/configs/younis_score/`: versioned baseline and later study inputs;
- `tests/highdim/test_younis_score_master_*.py`: coordinator contract and
  integration tests.

These paths are planned deliverables, not existing executable commands.
Keep Python standard-library scheduling, JSON, and provenance outside the
numerical kernels. Use TensorFlow/TFP for candidate numerical calculations,
tuning, and admission decisions. No NumPy runtime dependency is introduced.

Execution reconciliation: the coordinator lives directly under `bayesfilter`
because importing `bayesfilter.highdim` eagerly loads numerical modules.
This preserves validation without TensorFlow/GPU initialization. Numerical
adapters may import the existing high-dimensional providers when run.

The CLI needs `validate`, `dry-run`, `run`, `resume`, and `report` actions.
Validation and dry-run must enumerate both runnable rows and excluded rows
with reasons without importing a GPU-initializing kernel. A run may not
silently reduce the requested matrix. Resume checks the saved scientific
specification, source dependencies, partitions, tuning scope, and completed
results before reusing work.

Code or mathematical repair remains work for the supervising developer or
agent. The CLI records the changed revision and issue disposition, schedules
declared regression/retry commands, and enforces the updated dependencies; it
does not generate arbitrary source fixes or silently choose a new method.

Implement separate fields for numerical validity, scientific candidate
decision, and execution completion. A low-MSE failure, unavailable kernel,
insufficient precision, and a deferred method are different states. Store
paired randomness by model, dataset, particle replicate, perturbation, and
coupling group, independent of scheduling order.

Required tests include incompatible support/target rejection; stale tuning;
missing initialization terms; missing active methods; deterministic row
generation; repeated/resumed execution; bounded attempts; incomplete output
rejection; preservation of failed attempts; and a blocked branch alongside an
independent runnable branch. Test the repair/refresh transitions explicitly.
A Phase 0B fake numerical endpoint can test scheduling, but its evidence must never
be presented as a real particle-filter run.

Exit: a synthetic failing task can be repaired and resumed with the right
dependencies and artifacts. This establishes coordinator behavior only.

Implement and test the following interface in this phase. The paths and
commands describe deliverables to create, not code already implemented:

```text
python scripts/run_younis_score_master.py --study <baseline.json> --action validate
python scripts/run_younis_score_master.py --study <baseline.json> --action dry-run
python scripts/run_younis_score_master.py --study <baseline.json> --action run --output <new-run-directory>
python scripts/run_younis_score_master.py --study <baseline.json> --action resume --output <existing-run-directory>
python scripts/run_younis_score_master.py --study <baseline.json> --action report --output <existing-run-directory>
python scripts/run_younis_score_master.py --study <tuning.json> --action select --output <tuning-run-directory> --selection <new-selection.json>
python scripts/run_younis_score_master.py --study <baseline.json> --action compare --output <existing-run-directory>
```

Replace placeholders with exact versioned paths when the phase creates its
fixtures and study configuration; save actual commands in the manifest.
Resume reads validated existing results and writes new attempts to fresh
subdirectories without overwriting evidence. Phase 0C exercises this
interface with real baseline endpoints. This CLI is an output of master
execution, not a precondition for beginning Phase 0A.

## Phase 0C: deliver a real baseline study before adding enhancements

Entry: Phases 0A--0B for orchestration; baseline-specific numerical repairs are part of this phase. Independent reference fixtures may be developed while the coordinator is being built.

Use scalar and multivariate regular LGSSMs. Include parameter effects on
initial mean/covariance, transition, and observation parameters rather than
only the existing transition-direction example. Verify the independent
analytical Kalman oracle and explicit total derivatives. Same-finite-program
parity and error against the exact model score are separate tests.

Construct the initial cheap comparator set: bootstrap PF, UKF Gaussian
approximation, and the analytically adapted Gaussian proposal where its
density is available. Kalman is the error oracle. Add the canonical
LEDH/GenUT/Contract E baseline only after its actual endpoint passes the
required lifecycle, reset, determinant, and analytical-sensitivity checks.
Keep approximation and finite-stream targets visible in every comparison.
Keep prior and adapted sequential importance sampling without resampling as
separate naive arms. The bootstrap/adapted PF arms resample explicitly. For
discrete resampling, a derivative with ancestor labels locally fixed is an
almost-everywhere finite-program derivative; it does not establish an unbiased
derivative of the program's expectation. Common uniforms preserve marginal
multinomial laws but do not remove this target distinction.

Build the tuning adapter and reports now. Each claim scope binds all master
fields, including proposal/estimator controls, bandwidth, FD choices, dtype,
TF32, horizon, particle count, and backend. Check the repository's actual
tuning issuer and reject a caller-stamped identity. Tune on calibration and
validation partitions; claim data and streams remain untouched until freeze.
Mechanics smokes do not need statistical tuning and cannot support a quality
claim.

The report preserves raw dataset/replicate results, paired oracle errors,
uncertainty, conditional heuristic tables, failures, oracle limitations, and
cost. Test aggregation on independently known small examples, including
shared versus independent particle noise and datasets with unequal numbers
of successful replicates. Do not silently discard failures or treat particle
replicates as independent observation datasets.

Run CPU-only reference checks with `CUDA_VISIBLE_DEVICES=-1`. Verify candidate
kernels separately on trusted GPU/XLA with stable signatures, analytical
derivatives, verified memory growth, TF32 recorded, and canonical chunking.
Autodiff is parity evidence only. Do not recover a missing capability through
a scalar fallback, pfor, or a reduced LEDH fork.

Exit: the same study specification drives a real baseline smoke, a scoped
tuning/claim plumbing check, artifact aggregation, and an interrupted-run
resume. The tiny claim-plumbing fixture carries explicit no-ranking status.
All required baseline rows either pass or keep 0C incomplete; running only
the oracle does not close this milestone. End with repair and refreshed 0D,
0E, and 0F plans. A statistical baseline claim requires its own sufficiently
powered run after this engineering exit.

## Phase 0D: KDM/IWSG, control variates, and estimator combinations

Entry: the Phase 0C baseline and the capabilities needed by the selected KDM row. Complete the shared executor repair here when it remains open; no external implementation program is required.

First test frozen-mixture IWSG identities against analytic integration or exact
small examples. Then vary independent clouds to separate conditional sampling
variance from cloud variation. Fix trace/callback support and required
analytical tangents in the shared canonical executor before integrated or
resampling KDM runs. Compare scalar and every claimed batch/device lane through
the consumer, including parameter-dependent initialization and bandwidth.

Implement the actual KDM proposal density, physical numerator, component
covariance lifecycle, and any OT/jitter law used. Check that samples and weight
denominators describe the same distribution. Do not add an OT determinant
without a corresponding density derivation or present a changed measure as
the original filter.

For an exact control variate, identify the sampling law and either a known
center or an independent unbiased center estimate. Test the coefficient and
center dependence conditions, account for center cost/variance, and report
the unchanged mean of the baseline estimator. For two biased estimates,
implement a separate calibration path using oracle error and held-out MSE;
variance minimization alone does not determine the better combination.

Exit: conditional identities and full-filter wiring pass independently; ratio
bias and target distinctions remain explicit. A variance reduction result
alone cannot close a model-score improvement claim. If centering is unknown,
that exact-CV row remains blocked while a correctly labeled calibrated
combination can still run.

## Phase 0E: covariance alternatives and corrected proposals

Execution update (2026-09-15): Phase 0D's two full KDM consumers, conditional
identities, known-center control and independent calibrated blend now execute.
See `artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/phase-0d-result-and-refresh.md`.
The 48-row CPU fixture and three GPU/XLA rows are mechanics evidence only.
The exact-center control had higher observed validation error; the small
calibrated blend was descriptively favorable. Neither result establishes a
ranking or removes the need for the later scientific comparison.

The [density-fit iAPF prerequisite](younis-score-iapf-implementation-2026-09-15.md) implements the density-scale objective and adaptive iteration controller alongside the log-quadratic comparator. Source `cf823241` passes the scalar Gaussian GPU consumer and derivative checks. The unrestricted objective need not attain its infimum; numerical bounds, local optimizer, stopping rule, offline fitting precision and realized particle count are explicit. [Adaptive-N selection/reporting](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-adaptive-scope-result-and-refresh.md) now executes at `1c12eefa`, binding the selected procedure and recording its realized count and full offline/final work. Ten GPU rows pass, including actual count growth; this is mechanics evidence, not scientific promotion. The [control-safety screen](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/control-safety-result-and-refresh.md) completes its bounded observability and domain repair; comprehensive control calibration remains open. The [nonlinear extension](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/nonlinear-iapf-result-and-refresh.md) at f5a4d411 now executes through shared kernels: twenty GPU rows pass, including independent final streams, actual adaptive counts, and physical/executed observation validation. The inherited fit controls fail the descriptive heuristic screen and often reach their declared bounds. Next audit and execute complete numerical-control calibration using fresh partitions; cap-only or k-only sweeps cannot close this obligation. Wider dimensional coverage and use of fitted iAPF moments inside the LEDH covariance lifecycle remain separate implementation work.

Entry: Phase 0C for the affected baseline. A KDM covariance provider also requires the relevant Phase 0D mixture-law tests; other providers need not wait for KDM.

Implement one provider interface for prior/predicted moments, observation
conditioning, persistent per-component state, reset/ancestry mapping, and
analytical sensitivities. Integrate UKF first, then KDM and SGQF. A
prediction-only replacement cannot stand in for a complete filtering
covariance lifecycle. Test the actual moments and derivatives consumed by
LEDH with linear-Gaussian exact moments and nonlinear stress examples.

SGQF requires separate checks for signed integration weights versus sampling
probabilities, projected covariance validity, sparse-grid point count,
dimension/level growth, and the real GPU/XLA call chain. Existing standalone
code does not certify that chain. Evaluate covariance safeguards explicitly;
record any ridge, clipping, or damping as a numerics-altering choice and test
its non-harm before promotion.

For twisting/iAPF, start with a positive fixed one-step lookahead whose
ancestor law, transition proposal, normalization integral, and weight
correction can be derived and checked. Test a finite discrete example and a
Gaussian example before fitted twists, including initialization, terminal
factors, telescoping, and recovery of the untwisted law when the lookahead is
constant. Inspect primary equations and author
code for source-faithfulness claims. Select hyperparameters and the fitting
procedure on calibration data. That frozen procedure may fit a proposal to each
new observation dataset using independent offline streams; freeze its fitted
coefficients and realized count before final sampling. Oracle scores, heldout
errors and final sampling noise must never enter fitting. Distinguish held-fixed learned controls from
their explicitly parameter-dependent evaluations and include the latter's
derivatives. An unknown required normalizer blocks that proposed correction.

Exit: each provider/proposal has density and total-derivative tests and a
consumer integration test. Better covariance or ESS alone is explanatory;
score-MSE improvement still requires untouched evidence. An unrepairable
within-scope proposal assumption rejects that row, not the rest of 0E.

## Phase 0F: finite differences and error attribution

Entry: Phase 0A for deterministic mechanics, implemented here on exact fixtures. Stochastic FD needs Phase 0C value endpoints; normalization and consistency studies need only the implemented estimator they assess.

Move the already checked exact three-, five-, and eleven-point examples into
maintained diagnostic tests. Include polynomial cancellation, smoothness
counterexamples, rectangular direction recovery, deficient rank, parameter
boundaries, and precision/conditioning checks. They may be developed before
KDM or SGQF integration is complete.

Implement the stochastic FD lane using verified value endpoints and an
explicit coupling of replicas. Check that each replica has the correct
marginal law; an OT coupling here is distinct from OT as a filter reset.
Measure the stencil covariance and the full bias-square term, including the
cross-term. Use a positive scale-aware step ladder and QR/SVD direction
reconstruction. Do not require a universal U-curve or a stochastic slope equal
to deterministic stencil order.

Once eligible estimators exist, implement the value/derivative/ratio study
and Phase 4B consistency diagnostics. Fit step sizes, combinations, and any
diagnostic-based decision rule on calibration/validation only. Measure oracle
error and diagnostic correlation with uncertainty on separate data. Unknown
bias attribution limits interpretation; it does not invalidate a correctly
measured held-out MSE comparison.

Exit: exact mechanics, coupling, reconstruction, normalization targets, and
partition checks pass. Approximate consistency remains evidence about the
tested relationship, not proof of unbiasedness.

## Phase 0G: implement model coverage, study configurations, and final reports

Entry: the applicable Phase 0C baseline and the specific 0D--0F capabilities
used by each study. Implement this phase in slices before each new model,
regime, dimension/level, or method interaction is measured. It is not a
requirement that every advanced method finish before any experiment begins.

1. Implement regular nonlinear model adapters, data generators, derivative
   inputs, and usable reference providers. Include the weakly nonlinear,
   strongly curved, multimodal, long-horizon, and high-dimensional regimes
   actually selected for the study. Check reference uncertainty and the full
   adapter-to-kernel call chain. A missing oracle limits the allowed conclusion
   and does not authorize inventing an accuracy certificate.
2. Implement explicit study configurations for the baseline ladder,
   single-factor changes, pairwise interactions, and the eligible larger
   matrix. Bind each row to its target, support, method implementation,
   tests, tuning scope, data/stream partitions, heuristic comparators, and
   evidence role. Include failed-candidate repairs that the later study is
   designed to discriminate; a failed single-factor arm cannot silently
   delete its predeclared repair experiment.
3. Implement capacity/timing measurement and bounded pilots before SGQF
   dimension/level expansion and factorial launches. Capture compile time,
   memory, point count, per-row costs, and failed attempts. The pilot updates
   the next run's finite budget and cannot select claim observations.
4. Implement report coverage for matched particle count and matched total
   cost, including tuning, twist fitting, center estimation, additional FD
   calls, compilation, and explicit amortization assumptions. Construct the
   conditional heuristic tables and nested/paired uncertainty calculations
   for each selected regime. Failed rows remain visible.
5. Implement final replication and audit commands and their integration
   tests before Phases 8--9. Test untouched-partition enforcement, complete
   accounting for requested methods, and invalidation after a terminal
   repair. Trace every scientific output back to its actual numerical
   endpoint and source revision.

Exit: each requested study has a validated configuration, actual model and
method endpoints, reference status, measured feasibility, and tested report
assembly. A tiny integration run establishes that this computation works;
it does not establish the subsequent scientific comparison. Missing
implementations keep their rows incomplete, with an explicit repair or
blocked disposition. Refresh the applicable Phases 2--4C and 7--9 after the
repair pass.

### Implementation coverage and initial capability inventory

The existing code below is input to Phase 0A's audit, not evidence that the
new integrations have passed. Verify actual calls and tests before reuse.

| Existing component | Implementation phase and required check |
|---|---|
| [Canonical scalar executor](../../bayesfilter/highdim/ledh_canonical_score_tf.py) and [ongoing loop repair](ledh-while-loop-regression-repair-plan-2026-09-14.md) | 0C/0D: preserve the shared canonical algorithm, repair initialization and required trace/callback capabilities, verify all affected consumer lanes. |
| [KDM primitives](../../bayesfilter/highdim/ledh_younis_kdm_tf.py), [integrated route](../../bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py), and [resampling route](../../bayesfilter/highdim/ledh_younis_kdm_resampling_tf.py) | 0D: verify mixture laws, identities, covariances, corrections, and analytical consumer derivatives. |
| [LGSSM reference](../../bayesfilter/highdim/ledh_younis_kdm_lgssm_reference_tf.py) | 0C: extend or select an oracle for all claimed parameter dependences; retain the bootstrap's fixed-stream finite-program target. |
| [SGQF values](../../bayesfilter/nonlinear/fixed_sgqf_tf.py) and [derivatives](../../bayesfilter/nonlinear/fixed_sgqf_derivatives_tf.py) | 0E/0G: implement the LEDH filtering lifecycle, consumer sensitivities, GPU/XLA capability, and scaling measurements. |
| [Tuning scopes](../../bayesfilter/highdim/ledh_tuning_scope.py) and [route registry](../../bayesfilter/highdim/ledh_tuning_registry.py) | 0B/0C: verify the complete issuer/consumer path; a scope type alone is not a tuner. |
| [GPU memory policy](../../bayesfilter/runtime/gpu_memory_policy.py) | 0C: configure and verify before device initialization, then record the actual backend policy. |
| [Previous Phase 4A runner](../benchmarks/run_ledh_younis_kdm_phase4a_campaign.py) | 0B/0G: audit reusable code against the amended targets, partition rules, and reports; it is not the new master merely because it is a CLI. |

Before a research row runs, its implementation record must identify the
producing phase, real callable, required capability tests and their results,
source dependencies, current tuning scope where applicable, and result/report
consumer. An absent field identifies unfinished master work. The record does
not authorize callers to self-attest canonical route identity.

## Phase 1: identity and call-chain verification

Phase 1 is admission of the executable evidence produced during Phases
0C--0G, evaluated per method and target. Its tests are implemented and run
inside those prerequisite phases and reused here at the checked revision.
Implementation does not wait for a later Phase 1 pass to create its own tests.
An admission failure returns to the responsible implementation phase's repair
step; it does not create a separate execution program.

Before measuring quality, choose the identity test from the estimator's
declared mathematical target:

| Target | Required evidence |
|---|---|
| Analytical derivative of a fixed finite scalar | Compare with that same scalar at fixed randomness using diagnostic FD/autodiff where differentiable; include initial and recursive dependencies. |
| IWSG gradient of a mixture expectation | Derive the locally fixed-proposal identity with support and differentiation/integrability assumptions; compare repeated estimates with a known expectation gradient. Samplewise equality with a different pathwise estimator is not required. |
| Unnormalised value/derivative pair | Verify the sampling law and the stated expectation/derivative identities separately; declare whether \(\widehat D\) differentiates this particular \(\widehat Z\). |
| Estimate of the model score | Verify its own derivation and implementation, then measure oracle error and uncertainty. Neither pathwise parity nor low MSE proves unbiasedness. |

All applicable initial, transition, observation, mixture-weight, covariance,
bandwidth, flow-Jacobian, OT, and GenUT dependencies must be covered.
Parameter-dependent initial means and covariances need explicit sensitivity
fixtures. First compare the same finite scalar; separately compare the
estimator statistically with Kalman. A finite particle realization need not
equal the exact model score.

Require executable consumer-to-provider tests for rank, batch, dtype,
graph/XLA/device, and analytical-sensitivity contracts. For a model-corrected
proposal, verify the actual sampler/denominator law and physical numerator.
Support and positivity checks apply to the relevant density or covariance;
signed quadrature integration weights are not sampling probabilities.

The protected execution snapshots have passed actual integrated-KDM,
resampling-IWSG, SGQF and persistent-mixture consumer tests. They also include
explicit initial-state and initial-covariance tangents, checked against the
six-parameter Gaussian oracle and fixed-program finite differences. This
resolves the original prerequisites for those snapshots. The main checkout's
native-loop integration subsequently passed its own endpoint regressions and
GPU mechanics checks, including the repaired callback interface. The versioned
integration result records those checks; passes remain tied to their actual
source revisions and do not imply scientific promotion.

## Phase 2: proposal and representation study

Execution update, 15 September: the first nonlinear settings failed the
conditional heuristic screen descriptively. The next repair is the
[nonlinear calibration and untouched pilot](younis-score-nonlinear-calibration-2026-09-15.md):
nine proposal/regime calibration scopes, frozen selections and fresh evaluation
against four constructed heuristics. Phase 0G now includes
`scripts/run_younis_score_campaign.py` and
`bayesfilter/score_study/heldout_reporting.py`, with executable tests for
selection-before-evaluation and dataset-level uncertainty. These are master
implementation tasks. The 1,440-row GPU pilot completed after these checks
passed; all nine selections preceded untouched evaluation. Its
[result and phase refresh](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/nonlinear-calibration-result-and-refresh.md)
records the heuristic promotion veto in all three regimes. Concentrated-regime
paired intervals include zero. Incomplete control calibration and modest
dataset count prohibit final ranking or default claims. Preserve the opened
evaluation data and continue to fresh calibration and the remaining method
prerequisites; this is not a continuation veto on the research direction.

Hold the score estimator fixed and vary only the proposal representation. The
factorial factors are proposal family, lookahead construction, covariance
source, particle count, horizon, observation regime, and compute budget:

1. bootstrap transition proposal;
2. UKF/EKF adapted proposal;
3. LEDH flow proposal;
4. KDM with explicit component covariances;
5. OT-resampled KDM cloud;
6. a fixed lookahead mixture proposal (backward-smoothing extensions deferred);
7. mixtures with a declared support-preserving exploration component; and
8. the Guarniero--Johansen--Lee iterated auxiliary particle filter (iAPF),
   where its learned \(\psi\)-twisting is used only to construct an auxiliary
   proposal.

The proposal family is crossed with the following covariance sources: physical
\(P^Q\), per-particle UKF, KDM weighted-mixture, LEDH/GenUT, SGQF projected,
and a calibrated convex hybrid. SGQF is included as a proposal-design arm
because the repository already exposes \texttt{tf\_fixed\_sgqf\_cloud} and
\texttt{tf\_fixed\_sgqf\_filter}; the current implementation is an
eager/Python-branch approximate lane, so it is not automatically a
claim-bearing high-dimensional filter. Its sparse-grid point count, memory,
moment positive-definiteness, and dimension/level scaling are recorded before
any score comparison. The SGQF moments parameterise \(q_t\), while the
physical transition and observation factors remain in the numerator and the
actual SGQF-induced proposal density remains in the denominator.

Twisted PF and iAPF are crossed as none, fixed one-step \(\psi\), and
iteratively fitted \(\psi\). The fitted lookahead is produced on a disjoint
pilot/calibration partition and frozen before validation. A twisted or iAPF
arm must report the ancestor law, state proposal, correction factor, support,
and whether the proposal density is joint or marginalized. It is not allowed
to substitute an approximate future likelihood for the physical target.

For regular transitions, compare both ordinary PF weights and full model-
corrected mixture weights. This separates proposal quality from target changes.
Measure conditional ESS, weight variance, coverage of high-posterior regions,
and score error at equal particle count and equal wall-clock cost.

The minimum proposal test set is: (i) linear-Gaussian/Kalman normalizer and
score with an exact model-correction identity; (ii) a nonlinear
regular-transition model with mode or tail stress; and (iii) a high-dimensional
SGQF scaling ladder. For every row, test proposal normalisation, support,
finite weights, target-preserving \(\widehat Z_t\), and the call chain from the
claim-bearing endpoint to the declared proposal. Proposal normalizer MSE is
the primary criterion for a normalizer-accuracy claim; variance alone suffices
only under a proved common expectation. ESS, coefficient of variation, coverage,
and weight tails explain or screen proposals under predeclared roles. A claim
of better score computation requires lower held-out oracle score MSE at the
declared budget. Proposal metrics cannot substitute for that criterion.
A paired uncertainty interval is required before a row is called better;
otherwise it is classified as viable or descriptive-only.

The focused proposal tests must cover (a) equality of the sampled proposal
density and the denominator evaluated at the sample, (b) normalization of the
twisted/iAPF correction on a finite discrete auxiliary example, (c) SGQF
cloud-weight and projected-moment invariants, (d) positive-definiteness and
conditioning of every covariance arm, and (e) a linear-Gaussian identity in
which every proposal recovers the same normalizer in expectation. These tests
precede the factorial campaign; a failed identity or support test excludes the
row rather than being averaged into its score result.

The iAPF is a proposal-quality experiment, not a replacement score identity.
Guarniero, Johansen, and Lee define a family of positive future-data twisting
functions \(\psi\), show that the corresponding \(\psi\)-auxiliary particle
filter preserves the marginal-likelihood normalizer after the change-of-measure
correction, and iteratively approximate the zero-variance \(\psi^*\) sequence
from particle output (their Sections 2--3 and Algorithms 2--4). In this
program, an iAPF arm changes the auxiliary ancestor and state proposal laws;
it may use UKF or LEDH moments within that construction. It is not merely a
replacement covariance filter. The physical target remains fixed, while the
analytical recursion must be derived for the changed sampling law and include derivatives of the
new ancestor probabilities, proposal density, twisting correction, and all
parameter-dependent normalizing integrals. Pilot-fitted controls may be frozen,
but parameter dependence in their declared evaluation is not silently detached.
The denominator and support record must describe the new sampling law.
If ancestor sampling is discrete, an expectation-gradient claim needs the
appropriate sampling-law terms; differentiating fixed ancestor labels is not
a replacement for them.
The arm is admissible only when the required transition integrals, proposal
sampling, and correction are evaluable. An ambient Gaussian implementation is
not a solution for a singular DSGE transition; a disturbance- or
manifold-coordinate version would require a separate support and derivative
derivation.

This is distinct from Ionides iterated filtering. Ionides augments the model
with an artificial parameter random walk and decreases its perturbation scale
to obtain a simulation-only likelihood-derivative approximation. It is a
separate comparator for implicit models, not a UKF replacement and not an
iAPF proposal module.

Twisting is also not synonymous with replacing the UKF. The UKF supplies
local moment geometry for a proposal (and, in canonical LEDH, the per-particle
covariance lifecycle). A twisted filter changes ancestor and transition
sampling through a positive look-ahead function \(\psi\), with a correction
that preserves the declared likelihood normalizer. The two mechanisms can be
combined: UKF/LEDH can provide local proposals while twisting supplies an
auxiliary look-ahead weight. A route that replaces the UKF with a twisted
transition must therefore be registered as a new proposal route and must
prove that its sampling law, normalizing integral, correction, and support are
evaluable. It remains a proposal-variance experiment; it does not change the
analytical score identity.

Every UKF/KDM/SGQF provider must specify the full time-indexed filtering
lifecycle: incoming component means, covariances, and weights; prediction
through the model; conditioning on the new observation; outgoing moments;
and reassignment or transformation of persistent component state at resampling
and Contract E reset. Physical process covariance, proposal covariance, and
KDM bandwidth have distinct roles. Record the actual moments consumed by LEDH
and their analytical sensitivities, not only the provider's standalone output.

SGQF's signed integration weights must not be used as categorical probabilities.
If projected moments define a Gaussian or mixture proposal, validate its density
parameters separately and declare any protection that changes those moments.
Before its scaling study, require multistep Gaussian moment fixtures and an
executable provider-to-LEDH test across batch dimensions, dtype, graph/XLA,
device, and analytical sensitivities. An existing eager standalone filter
does not establish this integrated capability.

The OT study must include exact marginal-balance error, coupling entropy,
transport cost, sensitivity to Sinkhorn iterations, and whether the transported
cloud remains compatible with the density used in the importance correction.
OT is never treated as a resampling theorem by itself.

Record where the existing LEDH flow Jacobian enters the likelihood weights.
OT moment restoration is not a proof of preservation of the filtering law;
retain its full moments/weights/transport dependence in the total derivative.
No extra OT log determinant is inserted without a derivation for the actual
sampling law. Clarifying the existing flow determinant and deriving an OT
density correction are different tasks.

OT has five separate experimental roles that must not be conflated:

1. a coupling for comparing two particle systems with common random numbers;
2. a deterministic cloud rearrangement used only as a proposal heuristic;
3. a continuous resampling law whose density can be evaluated and corrected;
4. an OT cloud followed by an explicit, evaluable jitter kernel; and
5. a model-changing reset whose score is the derivative of a new finite scalar.

Roles 1 and 2 do not by themselves define a model-corrected proposal. Role 3
requires a density and support proof. Role 4 requires the actual jitter-mixture
density. A change-of-variables Jacobian is included only when the sampling
construction requires it; jitter around fixed transported centers does not
create a new OT determinant. Role 5 is allowed only under the finite-program
target label and cannot be compared to the exact score as though the reset were
measure-preserving.

The Sen--Thiéry--Jasra coupling belongs specifically to role 1. It couples two
particle-filter replicas while preserving their marginal algorithms, usually to
estimate a difference between parameters, models, or discretization levels.
For score work this can reduce the variance of a finite-difference or
multilevel difference when the replicas are positively correlated. It does not
reduce the variance of one LEDH score by itself, and averaging two positively
coupled copies is not a variance-reduction argument. The Jacob--Lindsten--Schön
coupled conditional-particle construction is a different route: its meeting
and Rhee--Glynn correction can debias a smoothing expectation under regular,
tractable-transition assumptions. Neither coupling construction supplies a
singular-DSGE score identity.

### UKF, KDM, and the covariance question

The UKF already provides a Gaussian moment approximation, but it does not make
the KDM mixture identity true and it does not supply a model-score correction
for an arbitrary nonlinear or singular transition. The covariance study must
therefore compare three distinct objects:

1. UKF covariance used only to construct a proposal;
2. KDM weighted covariance computed from the continuous mixture; and
3. a hybrid proposal that uses UKF moments for local propagation and KDM
   moments for mixture placement.

For each object, hold the physical model numerator and score estimator fixed,
then measure proposal coverage, importance-weight variability, and score error.
This identifies whether a covariance improvement helps because it places
particles better, because it changes the target measure, or because it merely
changes the finite-program derivative. No covariance comparison is interpreted
as evidence that UKF has been replaced as the filter unless the complete
filtering recursion and its target are explicitly changed and audited.

For covariance interpretation, keep the score code and physical numerator
fixed while swapping only the proposal moments. Record eigenvalue margins,
condition numbers, weight tails, and proposal-density agreement. A covariance
arm fails closed if it is not positive definite, if a caller-supplied
covariance does not match the generated density, or if it is used to claim
replacement of the UKF without a complete filtering derivation.

## Phase 3: score-estimator study on a fixed particle law

For each proposal, compare the following estimators using the same particle
cloud whenever possible:

1. analytical derivative of the executed finite value program;
2. unnormalised likelihood derivative divided by the estimated normalizer;
3. KDM IWSG gradient of a declared mixture expectation;
4. hybrid analytical/IWSG gradients with an explicit expectation target;
5. Rao--Blackwellised conditional-integration variants; and
6. exact-mean control-variate and independently calibrated combination variants.

Fisher/backward-pair, Nemeth, Fearnhead, PaRIS, and coupled
conditional-particle/Rhee--Glynn rows are deferred to the separate smoothing
program. An available exact conditional expectation can still serve as a
small identity reference. Target-specific identity tests precede quality
comparisons; gradients of different objectives cannot be pooled.

Report vector bias, covariance, MSE, componentwise error, and correlation of
errors across coupled estimators. Do not call a lower MSE an unbiasedness result.

Control variates require a known or separately corrected centering constant.
Estimate coefficients on calibration data only, freeze them, and test whether
the empirical variance reduction survives on untouched data. A fitted baseline
whose centering expectation is unknown is labelled a heuristic variance
reduction candidate, not an unbiased control variate.

Exact centering preserves the expectation of the baseline estimator; a biased
baseline remains biased. Independent random centering also requires a proved
unbiased center and its variance contribution in the comparison. If
\(\widehat D=\nabla\widehat Z\) for the same positive finite scalar, then
\(\widehat D/\widehat Z=\nabla\log\widehat Z\) pathwise. Comparing these two
expressions tests derivative consistency, not ratio bias; compare their
expectation with \(\nabla\log Z\) to measure that bias.

Concretely, for a scalar LEDH score estimate \(B\), a KDM/IWSG statistic \(H\),
and its mean \(\mu_H=\mathbb E[H]\) under the actual joint experiment,
\(C_\beta=B-\beta(H-\mu_H)\) has \(\mathbb E[C_\beta]=\mathbb E[B]\)
for a frozen coefficient \(\beta\). Its variance is
\(\operatorname{Var}(B)+\beta^2\operatorname{Var}(H)
-2\beta\operatorname{Cov}(B,H)\). An independent unbiased estimated center
adds \(\beta^2\operatorname{Var}(\widehat\mu_H)\). A center derived conditional
on the incoming cloud must be valid for that cloud; repeat over independent
clouds to evaluate total error. The KDM mean identity, the centering mechanism,
and the correlation with LEDH error are three separate requirements.
If coefficient fitting reuses the random center, prove the required conditional
centering identity or use an independent center; unconditional unbiasedness of
the center alone does not justify multiplying it by a correlated coefficient.

### Combining two imperfect estimators

If (A) and (B) estimate the same score (S), a combined estimator

\[
 C_\alpha=\alpha A+(1-\alpha)B
\]

has error

\[
 \mathbb E[C_\alpha-S]
 =\alpha b_A+(1-\alpha)b_B,
\]

and, for a scalar score component, variance

\[
 \operatorname{Var}(C_\alpha)=
 \alpha^2V_A+(1-\alpha)^2V_B
 +2\alpha(1-\alpha)\operatorname{Cov}(A,B).
\]

The program must first establish that (A) and (B) target the same score.
Then estimate \(\alpha\) on independent calibration replications, using a
predeclared loss or an exact control-variate relation, freeze it, and evaluate
on untouched replications. With an oracle, the MSE-optimal scalar coefficient
can be estimated directly. Without an oracle, a variance-minimising coefficient
does not generally minimise MSE because the unknown bias remains. A combination
of two biased estimators is therefore a candidate for variance reduction, not
an automatic debiasing method. The study must report the two component biases,
their covariance, the selected coefficient, and the combined estimator's
untouched error.

For vector scores the cross term is
\(\alpha(1-\alpha)[\operatorname{Cov}(A,B)+\operatorname{Cov}(B,A)]\).
With a fixed positive-definite error metric \(W\), put
\(\Delta=A-B\). The oracle MSE quadratic has

\[
 \alpha^*=-\frac{\mathbb E[\Delta^\mathsf T W(B-S)]}
                   {\mathbb E[\Delta^\mathsf T W\Delta]},
\]

when the denominator is positive. If it is zero, the estimators agree in this
metric almost surely. A convex-combination restriction projects this value
onto \([0,1]\). Estimate the coefficient on calibration data, check the
denominator and oracle uncertainty, freeze it, and evaluate it on untouched
data. Without an oracle or a derived error identity, minimising variance is
a different objective and generally does not identify this MSE optimum.

The active comparison should include:

1. canonical LEDH and an eligible model-corrected KDM score estimate, each
   explicitly targeting the same model score;
2. analytical finite-program and coupled FD estimates of that same finite
   program, kept separate from model-score accuracy claims;
3. LEDH with a KDM/IWSG statistic whose centering under the actual joint
   sampling law is known or independently estimated without bias; and
4. IWSG plus an analytical or diagnostic pathwise gradient only when both have
   the same declared expectation target.

An unnormalised derivative cannot be added directly to a score without a
derived scaling and centering relation. An oracle-trained bias prediction from
Phase 4B is a heuristic residual-correction arm and must pass held-out
model-score MSE; it is not an exact-mean control variate.

If the component targets differ, the linear combination is a new target and
must be analysed as such. It cannot be described as a better estimate of the
original score merely because its numerical variance is smaller.

## Phase 4: normalisation and long-horizon study

This phase isolates the random-ratio problem.

1. Compare \(\widehat Z_N\), \(\widehat D_N\), and
   \(\widehat D_N/\widehat Z_N\) separately.
2. Vary particle count over a logarithmic ladder and retain paired random
   numbers across methods.
3. Vary horizon while holding model and dimension fixed, with a new tuning
   scope and frozen controls for each horizon.
4. Measure denominator coefficient of variation, reciprocal-weight tails,
   score bias, and variance.
5. Compare eligible analytical/model-corrected score estimators with ratio
   estimators at equal compute, without launching deferred smoothing rows.
6. Test coupled finite differences and consistency ladders after their own
   prerequisites; randomized debiasing remains outside the active scope.

The conclusion distinguishes established derivative or measure errors from
evidence about Monte Carlo and random-normalisation error. If attribution is
unresolved, say so. This limits causal explanation; it does not by itself
invalidate a correctly measured held-out MSE comparison.

## Phase 4B: approximate consistency diagnostics

When an exact score oracle is unavailable, the program may study diagnostics
that are plausibly correlated with bias, but it must keep them separate from
the score criterion. Candidate diagnostics include:

- score estimates at \(N,2N,4N\) under coupled random numbers;
- Richardson-style slopes in \(N\) or bandwidth, when a fitted expansion is
  supported by the data;
- agreement between eligible analytical, IWSG, and finite-difference estimates
  after checking that the claimed targets agree;
- conditional mixture/normalisation identities; backward/ancestor-pair
  residuals are deferred with smoothing;
- likelihood normalisation residuals and denominator stability;
- score directional identities under parameter perturbations; and
- replicate-to-replicate correlation between each diagnostic and measured score
  error on oracle-bearing models.

The diagnostic study must be calibrated on oracle-bearing models first. A
diagnostic can then be used as an explanatory warning or a predeclared
promotion veto. A continuation veto requires evidence of invalid execution or
a separately justified stop condition, not merely correlation with error.
The diagnostic cannot prove low bias on an oracle-free model merely because
its correlation is high elsewhere. The program must report the calibration
model, held-out model, correlation
uncertainty, and the strongest counterexample.

Central finite differences of a particle log-likelihood are useful for this
diagnostic, but they are not automatically a bias estimate. Writing
\(\ell(\theta)=\log Z(\theta)\) and
\(b_N(\theta)=\mathbb E[\log \widehat Z_N(\theta)]-\ell(\theta)\),

\[
\mathbb E[\widehat s_{\varepsilon,N}]-\ell'(\theta)
=
\underbrace{\frac{\ell(\theta+\varepsilon)-\ell(\theta-\varepsilon)}
 {2\varepsilon}-\ell'(\theta)}_{O(\varepsilon^2)\text{ under smoothness}}
+
\frac{b_N(\theta+\varepsilon)-b_N(\theta-\varepsilon)}{2\varepsilon}.
\]

Common random numbers or Sen-style coupled resampling can reduce the variance of
the difference, but they do not remove the log-normalisation bias, finite-
particle bias, or finite-difference truncation bias. If the same fixed random
design contains discrete resampling, the \(\varepsilon\to0\) limit may instead
differentiate a discontinuous finite program. Richardson extrapolation is
allowed only after a relevant truncation/error expansion has been justified
and tested on the quantity being extrapolated. A noisy MSE slope is not that
test; Phase 4C separates deterministic truncation from stochastic error.

A symmetric local-polynomial fit is a useful multi-point version of this
diagnostic, but the number of points is not itself an accuracy guarantee. For
points \(\theta+k h\), the derivative is the slope of the fitted polynomial at
zero. With a symmetric stencil, a linear fit has
\[
 D_h\ell=s+\frac{\ell^{(3)}(\theta)}{6}
   \frac{\sum_k k^4}{\sum_k k^2}h^2+O(h^4),
\]
for deterministic \(\ell\) with bounded fifth derivative locally. A quadratic
fit has the same leading order because the quadratic term is
even and is orthogonal to the slope on the symmetric design. To cancel the
third-derivative term one may fit a cubic or use the five-point fourth-order
stencil. These have different weights and error constants on different grids;
both give \(O(h^4)\) truncation with five bounded continuous derivatives locally.
More points can reduce observation-noise variance, but
they can also enlarge the stencil and increase the truncation constant or
amplify correlated particle noise. The implementation must therefore choose a
fixed symmetric grid, fit degree and span separately, and report both the
deterministic order in \(h\) separately from stochastic bias and variance.
The fitted derivative still
inherits the finite-particle/log-normalisation bias of the values being
regressed; regression does not turn it into an oracle.
An unconstrained normal draw for (h) is a poor default because draws near zero
divide particle noise by a small number; use a deterministic positive ladder
(or a distribution truncated away from zero), and use negative points as the
symmetric stencil rather than as independent random magnitudes.

A finite-difference statistic can be an exactly centered control variate when
its own expectation \(\mu_D\) is known (or independently estimated without
bias). An exact model score is not automatically the expectation of a finite
stencil applied to random log-likelihood estimates. When the center is valid, use
\(S_{\mathrm{cv}}=S-\beta(D-\mu_D)\), estimate \(\beta\) on calibration
replicates, and freeze it. If \(\mu_D\) is unknown and replaced by the
finite-difference estimate of the target score, the construction is a
calibrated heuristic or residual estimator, not an unbiased control variate.

## Phase 4C: fixed symmetric directional finite-difference ladder

This phase tests whether a coupled finite-difference calculation is a useful
directional score estimate or calibration diagnostic. It has two distinct
questions: does the deterministic stencil implement its stated approximation,
and how accurately does that stencil, applied to random particle log-likelihoods,
estimate the model score? Passing the first test does not answer the second.
Keep the model, proposal, target label, particle count, horizon, and coupling
fixed within a comparison; selected tuning controls stay frozen at all its
perturbed parameter points.

### 4C.1 Stencils, smoothness, and deterministic mechanics

Let \(f_v(t)=\ell(\theta+tv)\), where \(\ell=\log Z\) and \(v\) is a
declared unit direction in the chosen parameter coordinates. Test

\[
D_2(h;v)=\frac{\ell(\theta+hv)-\ell(\theta-hv)}{2h},
\]

\[
D_4(h;v)=\frac{\ell(\theta-2hv)-8\ell(\theta-hv)+8\ell(\theta+hv)-\ell(\theta+2hv)}{12h},
\]

and an unweighted eleven-point cubic least-squares fit at \(j=-5,\ldots,5\).
Write its derivative as \(D_{\rm cub}(h;v)=h^{-1}\sum_j c_j f_v(jh)\).
With \(S_r=\sum_{j=-5}^5 j^r\), its weights are

\[
 c_j=\frac{S_6 j-S_4 j^3}{S_2S_6-S_4^2}.
\]

These follow by solving the odd block of the cubic normal equations.
A fit against dimensionless \(j\) returns a linear coefficient that must be
divided by \(h\); a fit against \(jh\) already returns the derivative in
parameter units. Use QR or SVD for fitted coefficients, not explicit
normal-equation inversion. Centered even polynomial terms do not change the
unweighted slope. Linear/quadratic symmetric fits remain second order;
eleven points alone do not produce fourth order.

A sufficient local assumption is \(f_v\in C^3\) with bounded third derivative
for \(D_2-f'_v(0)=O(h^2)\), and \(f_v\in C^5\) with bounded fifth derivative
for the five-point and cubic fourth-order statements. Four continuous
derivatives alone are insufficient. Taylor substitution gives

\[
 D_4-f'_v(0)=-\frac{f_v^{(5)}(0)}{30}h^4+o(h^4),\qquad
 D_{\rm cub}-f'_v(0)=-\frac{143}{90}f_v^{(5)}(0)h^4+o(h^4).
\]

The eleven-point fit spans \(5h\) in each direction; the five-point formula
spans \(2h\). Compare node spacing and total span separately rather than
interpreting more points as an automatic accuracy improvement.

Mechanics tests use exact polynomials and noise-free functions with known
derivatives, followed by deterministic Kalman log-likelihoods. Check stencil
moments, fitted-slope units, and a function with nonzero leading truncation
coefficient. Measure order only in a range above the roundoff/evaluation-error
floor. Exact cancellation on a polynomial is a passing algebraic check, not a
failed slope test. Any numerical slope tolerance must follow from the fixture's
error budget; no universal tolerance or condition-number cutoff is assumed.

### 4C.2 Particle bias, covariance, and the actual MSE

For a fixed dataset and parameter, let
\(L_j=\log\widehat Z_N(\theta+jhv;\xi_j)\), with the correct marginal
particle law at every point and a declared coupling across the \(\xi_j\).
Assume positive likelihood estimates, finite second moments of \(L_j\), and
write \(b_N(\vartheta)=\mathbb E[L_N(\vartheta)]-\ell(\vartheta)\).
For any of the fixed stencils, set

\[
 \widehat D_h=\frac{1}{h}\sum_j c_j L_j,\quad
 T_h=\frac{1}{h}\sum_j c_j\ell(\theta+jhv)-s_v,\quad
 B_{N,h}=\frac{1}{h}\sum_j c_j b_N(\theta+jhv),
 \qquad s_v=v^\mathsf T S(\theta).
\]

Taking expectations and then using the variance of a linear combination yields

\[
 \mathbb E[\widehat D_h]-s_v=T_h+B_{N,h},\qquad
 \operatorname{Var}(\widehat D_h)=\frac{c^\mathsf T\Sigma(h)c}{h^2},
\]
\[
 \operatorname{MSE}(\widehat D_h)
 =\bigl(T_h+B_{N,h}\bigr)^2+
   \frac{c^\mathsf T\Sigma(h)c}{h^2},
 \qquad \Sigma_{jk}(h)=\operatorname{Cov}(L_j,L_k).
\]

The squared bias includes \(2T_hB_{N,h}\); truncation and finite-particle bias
can cancel. Summing their squares separately is wrong. On oracle-bearing
models, deterministic evaluations give \(T_h\); repeated particle evaluations
estimate total bias, \(B_{N,h}\), and covariance with uncertainty. Away from
such oracles these terms generally cannot be identified separately.

Estimate MSE directly from replicated squared oracle errors. A squared
replicate-mean error is not itself an unbiased estimate of squared bias;
retain its finite-replication uncertainty or apply a justified correction.

There is no universal stochastic slope or U-shaped MSE curve. If the same
additive noise enters all \(L_j\), it cancels because \(\sum_jc_j=0\). With a
mean-square differentiable coupled process, the numerator variance can be
\(O(h^2)\), so division by \(h^2\) leaves bounded variance. With independent
noise of nonvanishing variance it instead grows as \(h^{-2}\). Discrete
resampling and other couplings can behave differently. Record
\(c^\mathsf T\Sigma(h)c\), total bias, variance, and MSE; report a plateau,
boundary minimum, or irregular curve if that is observed. No noisy error
slope is required to match deterministic stencil order.

### 4C.3 Calibration, held-out comparison, and full-score reconstruction

Use a finite deterministic positive ladder, such as
\(\{h_0,h_0/2,h_0/4,h_0/8\}\), whose span, lower bound, and length are chosen
for the parameter scale, dtype, model domain, and evaluation accuracy.
Every perturbed point must be valid and distinguishable in the execution
dtype. If a log-likelihood evaluation has error bounded by \(\delta_j\), the
corresponding derivative error is bounded by
\(h^{-1}\sum_j |c_j|\delta_j\); use such bounds or measured reference errors
to justify the numerical range. No fixed universal range of \(h\) is imposed.
An unrestricted normal draw for \(h\) is excluded because it can approach zero.
A distribution truncated away from zero is an optional later comparison.

For constrained or differently scaled parameters, declare the coordinates
before choosing directions and steps. If \(\theta=g(\eta)\), the computed
likelihood score in these coordinates is
\(S_\eta=(\partial\theta/\partial\eta)^\mathsf T S_\theta\); reconstruct in
those coordinates and transform with the appropriate nonsingular Jacobian
when an original-coordinate score is required. This is likelihood
reparameterisation, not a posterior-density Jacobian term.

Calibrate each stencil and coupling on independent particle replications of
Gaussian oracle models, then nonlinear regular oracle models. Preserve each
replication's entire vector of perturbed values to estimate the full
within-stencil covariance, including correlations across directions and
stencils. Common random numbers must preserve each perturbed filter's marginal
algorithm. Independent noise is a useful covariance reference arm.

Predeclare the directional or scaled vector MSE, cost accounting, selection
rule, and uncertainty method. Use disjoint calibration, validation/selection,
and final claim data and streams; freeze the selected stencil, \(h\), directions,
and coupling before claims. The selected calibration minimum is optimistic.
Do not require validation MSE to fall inside a calibration confidence interval.
Use paired differences in squared oracle error on the untouched set, with
dataset-level or hierarchical uncertainty for particle replicates nested in
datasets. Report all predeclared comparisons or account for multiple selection;
do not select a winner on the claim set.

For \(m\) directions in \(d\) coordinates, define
\(V=[v_1,\ldots,v_m]\in\mathbb R^{d\times m}\) and let
\(b\in\mathbb R^m\) contain their derivative estimates. Then
\(b\approx V^\mathsf T s\). If \(V\) has full row rank, the least-squares
solution is

\[
 \widehat s=\arg\min_s\|V^\mathsf T s-b\|_2^2
           =(VV^\mathsf T)^{-1}Vb.
\]

Solve by QR/SVD without forming the displayed inverse. Coordinate directions
provide a simple reference; fewer than \(d\) independent directions identify
only a projection. Report singular values, reconstruction residual, and
propagated error: for \(A=(VV^\mathsf T)^{-1}V\), bias is \(A\,\operatorname{Bias}(b)\)
and covariance is \(A\,\operatorname{Cov}(b)A^\mathsf T\). Rank failure or error
amplification beyond the declared dtype/accuracy budget vetoes a full-score
claim. A small overdetermined residual alone does not establish accuracy.
Any generalized least-squares extension must freeze its independently
estimated weighting rule and account for its estimation uncertainty.

Compare equal-particle and equal-compute performance separately, charging all
nonzero-weight function evaluations, directions, pilot fitting, and replication.
Report both total campaign cost and amortized per-evaluation cost. MSE times
runtime is not a universal objective: averaging \(R\) replicas gives
bias squared plus variance divided by \(R\), leaving the bias unchanged.

### 4C.4 Pass criteria and saved evidence

Deterministic algebra/order failures, invalid support, non-finite values,
incorrect coupling marginals, missing sensitivities in a claimed derivative,
or insufficient reconstruction rank are repair triggers and relevant validity
vetoes. Noisy slopes and the absence of a U-curve are explanatory observations,
not vetoes. A valid candidate that fails held-out MSE improvement is rejected
for promotion at that scope; continue any planned repair within budget.
Without a model-score oracle or a certified reference-error bound, this is
a consistency/parity diagnostic, not a measured bias correction.

Focused tests cover exact weights and moments, cubic-fit units, invalid steps,
finite-precision point collapse, coupling-stream reuse and marginal laws,
rectangular direction reconstruction, and error propagation. Campaign artifacts
save perturbed parameter points, weights, coordinate transform, coupling,
deterministic errors, replicate vectors, covariance, MSE uncertainty,
selection partitions, reconstruction diagnostics, costs, and the criterion/
veto/explanatory role of each diagnostic. Store them under
`docs/plans/artifacts/younis-kdm-score-master-20260914/`.

## Phase 5 (deferred): regular-transition smoothing

The ordinary Younis forward/backward smoothing study belongs to the separate
regular-transition program. KDM filtering and IWSG remain active in Phases 2--4;
this deferred phase must not be a prerequisite for those experiments and no
smoothing row is generated by this master.

The external program should, in its own plan, implement the physical-transition
mixture before positive bandwidth/OT, add the auxiliary backward density,
compute adjacent-state expectations, and retune every method in its own scope.
It must keep proposal-placement gains separate from a score identity. Positive-
bandwidth KDM is a changed filtering approximation unless a limiting argument
and model correction are supplied.

## Phase 6 (deferred): degenerate-transition support program

This is a derivation program owned by the separate DSGE/support workstream,
before it is an implementation program. This master records applicability but
does not schedule these rows or transfer regular-track results to them.

1. Inspect the manifold/constrained-state particle-filter literature and
   classify each method by its state-space reference measure and target. A
   fixed-manifold intrinsic smoother is admissible only as a special-case
   control: it must supply (or permit deriving) a transition density and
   backward kernel with respect to a common, parameter-independent manifold
   measure. Structural DSGE equilibrium and policy manifolds are generally
   parameter-dependent, so this branch cannot be promoted to the DSGE route
   without a separate moving-support derivation.
2. Write the DSGE transition in explicit ancestor/innovation coordinates.
3. Identify the base measure on which the conditional transition law is
   represented and determine whether its support depends on parameters.
4. Derive the complete-data score including the deterministic map and any
   Jacobian or constraint terms.
5. Determine whether distinct ancestors induce mutually singular supports and
   whether backward ancestor probabilities collapse.
6. Prove or disprove a valid marginalisation identity for the proposed
   coordinate representation.
7. Build a tiny exact fixture with symbolic or numerical integration.
8. Only after those checks, implement an estimator and compare it with an
   independent support-aware oracle.

Full-rank Gaussian kernels, ordinary ambient-space KDM densities, and regular-
transition PaRIS formulas may be used as stress tests, but cannot establish
validity for DSGE. If the support derivation fails, the scientific conclusion
is that the route is unresolved, not that a tuned approximation solves it.

The manifold literature supplies useful state-space machinery, including
geodesic or tangent-space sampling and implicit-constraint proposals. It does
not, by itself, supply the observed-data parameter score. A future intrinsic
Poyiadjis/PaRIS-style estimator would be a new theorem on a fixed manifold;
for DSGE, the required object is instead a moving-support or
disturbance-coordinate derivation. Manifold score-matching papers, which
estimate $\nabla_x\log p(x)$, do not answer this parameter-score question.

## Phase 7: heuristic dominance and practical comparison

For each salient regime, construct and evaluate cheap adversaries:

- linear-Gaussian: bootstrap and locally adapted PF, with Kalman as the exact
  error oracle;
- weakly nonlinear: EKF and UKF score;
- highly nonlinear: bootstrap PF and locally adapted PF;
- long horizon: larger-particle bootstrap and a fixed-coupling central
  finite-difference reference of its likelihood;
- multimodal: mixture proposal without KDM score correction;
- no degenerate row is executed here; imported support-aware results require
  an applicability and provenance check.

A complex method losing to a heuristic in a salient regime is a promotion veto,
even if it beats another complex method on average. The heuristic table is not
a tuning target.

Construct three to seven concrete cheap competitors for each evaluated regime
from these families, with a rationale and the same target/budget accounting.
Oracle distance is the primary error measure; an exact oracle is not itself
a stochastic method to beat. Use paired uncertainty for loss comparisons.
A UKF evaluated on a linear-Gaussian model coincides with the Kalman oracle
when its moment rule is exact for affine maps. Such a row verifies correctness
and calibrates approximation error; it cannot establish particle-score
superiority over the exact Gaussian solution. Failure to improve that row
does not cancel the planned nonlinear comparison. An invalid identity or
implementation, in contrast, triggers repair before that method continues.
A statistically supported loss to a heuristic vetoes promotion; inconclusive
differences block a superiority claim until the planned precision is reached
or the budget ends. Do not tune against the final heuristic evaluation.

## Phase 8: final replication and compute fairness

Tuning is already a prerequisite of each claim row under Phase 0. Each route
receives a separate tuning scope binding model, target, horizon,
particle count, dimensions, dtype, backend, chunk policy, and all tunable
controls. Calibration, validation, and untouched claim data are disjoint.

The final comparison uses at least:

1. multiple independent observation datasets;
2. multiple particle randomisations per dataset;
3. paired seeds across methods where coupling is meaningful;
4. a particle-count and an equal-compute comparison;
5. uncertainty intervals for paired score error; and
6. preserved failed candidates and repair attempts.

One seed or one short chain is descriptive only. A candidate may be called
viable after passing hard screens, but a ranking requires uncertainty evidence.

## Phase 9: implementation and production audit

This is the terminal audit of implementations built in Phases 0A--0G and
repaired at phase boundaries. It is not the first implementation phase. If
the audit finds missing functionality or invalid evidence, return to its
implementation producer, repair it, and repeat the affected evaluation before
closing the program.

Before any default or HMC-facing recommendation, audit:

1. source-faithfulness to Younis and Sudderth where claimed;
2. support class and reference measure;
3. complete call chain from consumer to implementation;
4. analytical derivative route, with autodiff restricted to parity diagnostics;
5. TensorFlow/XLA, dtype, GPU memory-growth, and chunk-policy compliance;
6. tuning-artifact scope identity;
7. reproducibility manifest; and
8. rendered scientific documentation.

The canonical LEDH route remains governed by the repository's Contract E and
per-scope tuning policies. A new estimator can be scientifically promising
without replacing the canonical route until its own gates pass.

## Phase exit gates and dependency order

The phases are sequential where a later phase depends on a mathematical or
implementation identity, and parallel only where the dependency is explicit.
Every implementation phase ends with its stated executable checks and the
repair/refresh procedure. Completion of a plan paragraph is not completion
of the implementation it specifies.

| Task | Implementation producer and minimum prerequisite |
|---|---|
| Phase 0A specification/source reconciliation | Phase 0 scientific requirements and current checkout; developer tools only. |
| Phase 0B coordinator | 0A target/return contract; implement its own state, validation, and repair/resume tests. |
| Phase 0C baseline, oracle, tuning, and report integration | 0A--0B and baseline-specific repairs performed here. |
| Phase 0D KDM/IWSG/combinations | The applicable 0C baseline; repair required shared-executor capabilities here. |
| Phase 0E covariance/SGQF/twisting | The applicable 0C baseline; KDM-provider rows additionally use 0D law tests. |
| Phase 0F deterministic FD | 0A target specification; implement exact fixtures here, independently of KDM or the coordinator. |
| Phase 0F stochastic FD/normalization/consistency implementation | 0C value endpoints and only the implemented 0D/0E estimators actually used. |
| Phase 0G model/scale/report coverage | Applicable 0C--0F capabilities for each new study slice. |
| Phase 1 admission | Target-specific tests produced during 0C--0G at the actual consumer revision; failed tests return to their implementation producer. |
| Phase 2 proposal quality | 0C or 0D/0E proposal implementation; 0G when coverage expands; Gates A/B, actual correction law, and scope tuning. |
| Phase 3 estimator comparison | 0D or other applicable verified estimator implementation; Gates A/B, fixed-cloud law, and Gate C for each new proposal used. |
| Phase 4 normalisation study | 0F implementation and Phase 3 evidence for the verified value/derivative pairs compared. |
| Phase 4B consistency calibration | 0F implementation, oracle, and eligible estimators; deterministic FD checks when used, not completion of the whole normalization or FD campaign. |
| Phase 4C deterministic mechanics | Execute the exact scalar/direction tests implemented in 0F. |
| Phase 4C stochastic/full-score study | 0F implementation, eligible value/proposal law, deterministic mechanics, valid directions/coupling, and scope tuning. |
| Phases 7--9 comparisons, replication, and final audit | 0G coverage/report implementation, applicable mechanism gates, and frozen scopes for the candidates compared. |

These are dependencies for each row, not a requirement that every candidate
finish one numbered phase before any candidate enters another. A KDM wiring
block does not stop oracle FD mechanics or an already verified UKF comparison.
Neither Phase 4B nor Phase 4C requires the separately owned Phases 5 or 6.

1. **Gate A, after the applicable Phase 0C/0G implementation:** the model and
   regime being admitted have Phase 0's target record, executable baseline
   ladder, tuning service/scope, and checked oracle status. No score-quality
   comparison proceeds without a valid comparator or an explicitly diagnostic
   label. Later models need their own records and checks; they do not block
   the first verified Gaussian study.
2. **Gate B, after Phase 1:** each target-specific identity, support,
   positivity, sensitivity, and consumer-to-provider call-chain test passes;
   an IWSG identity is not replaced by finite-program parity. Failures return
   to implementation repair.
3. **Gate C, after Phase 2:** proposal comparisons show the physical numerator,
   proposal denominator, and resampling representation separately. Missing
   correction blocks a model-corrected claim. A separately defined, valid
   finite scalar may continue as a changed-target approximation; an invalid
   density or implementation cannot be repaired by relabeling it.
4. **Gate D, after Phase 3:** estimator-level conditional error is available on
   common clouds and total error over independent clouds; Phase 4 uses these
   verified estimators for normalisation comparisons.
5. **Gate E, after Phase 4:** \(\widehat Z,\widehat D,\widehat D/\widehat Z\)
   and their uncertainty are reported; derivative, measure, Monte Carlo, and
   ratio explanations are classified where evidence permits. Unresolved
   attribution limits causal language but does not erase a valid held-out MSE.
6. **Gate F, after Phase 4B:** consistency diagnostics have oracle calibration,
   held-out correlation uncertainty, and explicit explanatory/veto roles.
   Correlation is never a proof of low bias.
7. **Gate G, after Phase 4C:** deterministic mechanics, scale-aware stochastic
   calibration, coupling marginals, and full-rank reconstruction pass. A noisy
   MSE slope or absent U-curve is not a veto.
8. **Gate H, after Phase 7--9:** only candidates with statistical uncertainty,
   complete manifests, scope-specific tuning, and implementation audit may be
   considered for default or HMC-facing status.

Phases 0, 0A--0G, 1--4, 4B, and 4C form the active implementation and research
foundation. Phase 7 evaluates active regular
candidates against conditional heuristics. Phases 5 and 6 are deferred
external branches and cannot supply active claim evidence. Phases 8 and 9 are
terminal validation and audit, not sources of new tuning data.

### Between-phase repair and refresh

Every phase, including a failed or terminal phase, ends with an explicit
repair pass and a refresh of its successors. A scientific gate passing does
not waive unresolved defects in the computation that produced its evidence.
An issue is repairable when there is a concrete correction or discriminating
test within the current scientific scope and remaining resources. Record and
attempt those repairs; a promise that unspecified further tuning might help
is not a completed repair plan.

1. Reconcile planned, completed, failed, blocked, and omitted rows. State
   whether the finding invalidates the harness, mathematical target,
   implementation, numerical result, or evidence, or only rejects a candidate.
2. For each repairable issue, record its affected rows/callers, cause or
   smallest diagnostic, proposed correction, regression checks, scope and
   partition consequences, and remaining compute/attempt budget.
3. Perform the feasible corrections and applicable candidate repairs. Run
   focused regressions and affected consumer/integration checks, invalidate
   dependent evidence when necessary, and recompute the phase decision. An
   edit without verification does not close the issue. Do not relax the
   target, baseline, or scientific criterion to obtain a pass.
4. Give every unresolved issue a disposition. Missing validity prerequisites
   block their dependent rows; independent verified work can continue. An
   inconclusive comparison remains unresolved, and a valid negative result
   can close its experiment without making the candidate successful. Mark a
   phase partial when its required work remains unfinished.
5. Refresh each next-phase plan using the actual findings: question, eligible
   rows and repairs, prerequisites, baselines, evidence roles, controls and
   tuning scopes, data/stream partitions, tests, exact commands/environment,
   budget, and stop conditions. Record a brief skeptical audit and link the
   revised plan before the corresponding dependent work begins. For the last
   phase, refresh the final disposition and any justified follow-up instead.

Repairs informed by claim outcomes create a new candidate version with fresh
calibration, validation, and claim partitions. An infrastructure retry that
preserves the numerical program may reuse its streams in a fresh attempt
directory; it cannot select a favorable result or overwrite the failure.
Source or scope changes require checking which previous evidence and tuning
remain applicable. Unknown control-variate centers or proposal normalizers
cannot be fixed by changing a status label.

The closeout note and machine-readable issue dispositions are sufficient;
routine repairs and next-phase refreshes do not require a new external review
or renewed permission within an authorized campaign. Phase 0B implements
tests for failure followed by repair; a repair that invalidates a formerly
passing dependency; changed horizon/source with stale tuning; a shared-code
failure affecting two methods; a blocked method beside a runnable method;
exhausted repair budget; and a terminal repair requiring fresh evaluation.
The scheduler refuses dependent launches without the applicable disposition
and current next-phase plan. Phase 0C verifies this behavior with real
endpoints, and Phase 0G extends it to the final reports.

Use the phase note plus a machine-readable closeout record containing:

```text
phase_id, phase_plan_version, source_revision, source_changes
planned_rows, completed_rows, blocked_rows, omitted_rows_with_reasons
engineering_status, numerical_status, scientific_decision, inference_status
issues: [issue_id, affected_rows, classification, root_cause, repair,
         regression_evidence, scope_effect, disposition]
attempts, consumed_cpu_hours, consumed_gpu_hours, remaining_budget
invalidated_evidence, current_tuning_scopes, partition_usage
next_phase_plan, next_phase_plan_version, refreshed_dependencies
```

For developer phases before the coordinator exists, record the same
information in the phase note; Phase 0B adds the machine-readable support.
The process does not require its own unbuilt machinery to get started.

## Required artifacts

Each phase produces:

- a plan and skeptical pre-mortem;
- a model/target/measure ledger;
- an oracle or oracle-status report;
- executable identity and call-chain tests;
- a run manifest containing commit, command, environment, hardware, seeds,
  wall time, tuning scope, and artifact paths;
- raw structured results and uncertainty calculations;
- a decision table separating hard vetoes, descriptive evidence, statistical
  ranking, and default readiness;
- a repair record with verified fixes, unresolved issue dispositions, affected
  evidence, and remaining budget;
- a refreshed next-phase plan (or terminal disposition), linked to the actual
  results and checked prerequisites; and
- a red-team note stating the strongest alternative explanation, the result
  that would overturn the conclusion, and the weakest evidence.

The master result must include a matrix with one row per method, model, horizon,
particle count, and score kind. It must never combine KDM expectation gradients,
finite-program derivatives, and marginal scores in one unlabeled leaderboard.

## Decision rules

The program may promote a candidate only when:

1. the claimed target matches the computed quantity by derivation; a declared
   approximation retains its distinct target and reports its measured or
   bounded error against the model score;
2. all support, positivity, normalization, and derivative vetoes pass;
3. the candidate clears the conditional heuristic-dominance table;
4. paired uncertainty supports the stated comparison;
5. the result survives an untouched retest under its own tuning scope; and
6. no conclusion crosses from the regular-transition track to the degenerate
   track without a new support derivation.

If a candidate fails, classify the failure as implementation, tuning,
diagnostic, finite-normalisation, support, or evidence failure. Rejecting one
candidate does not reject the research direction; rejecting the direction
requires evidence that the repairable alternatives and their target identities
have also failed.

## First executable tranche

Executing this master starts with implementation. Follow this order within
this document:

1. Read Phase 0's target/oracle requirements and perform Phase 0A's source,
   capability, and manuscript reconciliation. Pin the existing environment
   and the initial validation commands.
2. Execute Phase 0B: implement the registries, CLI, dependency checks, result
   records, budget accounting, and repair/resume behavior. Develop Phase 0F's
   exact three/five/eleven-point and direction fixtures independently when
   their Phase 0A specifications are available.
3. Execute Phase 0C: implement and verify the independent Gaussian oracle,
   parameter-dependent initialization, actual bootstrap/UKF/adapted and
   canonical LEDH endpoints, tuning issuer/consumer path, and reports. Repair
   the applicable shared canonical code instead of adding a reduced lane.
4. Perform the repair/refresh step after each phase. Complete the real
   Gaussian baseline smoke, interruption/resume check, and scoped
   tuning/claim-plumbing check with explicit no-ranking status. This is the
   first executable software milestone produced by the master.
5. Execute the applicable 0D--0F branches for KDM/IWSG, centered control
   variates, biased combinations, covariance/SGQF providers, twisting, and
   stochastic FD. Finish Phase 1 admission for each method using the tests
   produced by its implementation phase. A blocked method does not stop
   independent verified branches.
6. Use Phase 0G to implement each additional study's model, configuration,
   reference, and reporting requirements before Phases 2--4C and 7 run it.
   Tune on disjoint partitions, use common clouds for conditional comparisons
   and independent clouds for total error, and preserve every replica's
   marginal law in paired full-filter comparisons.
7. Run the scientific studies under their refreshed finite plans, repair and
   update after each phase, then perform Phases 8--9 replication and final
   audit. Any outcome-informed repair uses fresh evaluation partitions.

No separate execution plan is needed before these tasks begin. The deferred
DSGE/support and smoothing programs remain separate assignments; this
tranche neither launches them nor inherits an unsupported numerical result
from them.

## Literature coverage addendum (2026-09-14)

This is a literature applicability inventory, not an expansion of the active
execution scope. Smoothing, iterated filtering, continuous-likelihood methods,
multilevel debiasing, and DSGE derivations remain deferred; their earlier
implementation requests are superseded by the active-scope amendment.

The literature gap audit in
`docs/plans/artifacts/particle-filter-score-literature-gap-audit-20260914/`
identified several named methods that must be made explicit before interpreting
the inventory as a complete score-computation study. One direct
regular-model comparator is the Nemeth--Fearnhead--Mihaylova KDE plus Rao--Blackwell score and
observed-information estimator. The existing `KDM` row is only a method label;
it does not yet specify Nemeth's kernel recursion, bandwidth, conditional
integration, or its regular-model asymptotic assumptions.

The smoothing branch must also name the PaRIS backward-draw parameter and the
Fearnhead--Wyncoll--Tawn linear-cost smoother. A generic backward-pair label
cannot expose the distinction between one backward draw, two or more draws,
mixing assumptions, path degeneracy, and O(N) versus O(N^2) computation. These
methods estimate additive complete-data functionals; their output is not
automatically the marginal score and their regular-HMM assumptions do not close
the degenerate DSGE branch.

The literature inventory also includes the coupled conditional-particle smoother of
Jacob, Lindsten, and Schön. Its Rhee--Glynn construction is a qualitatively
different answer from averaging two biased score estimates: under its meeting
and moment assumptions it debiases a finite-horizon smoothing expectation, and
the authors explicitly combine it with Fisher's identity to obtain an unbiased
score when the transition density is tractable. The estimator has random
runtime and its own variance/cost trade-off. It is a regular-model comparator,
not a singular-transition result.

The inventory records continuous-likelihood particle filters (Malik--Pitt and
DeJong et al., with Flury--Shephard as a related econometric source) and
iterated filtering (Ionides et al.) as deferred comparators. The first family
addresses discontinuous particle likelihood evaluation; it does not prove that
the derivative of a continuous finite particle program is the model score. The
second is designed for simulation-only models with unavailable transition
density and gives a vanishing-artificial-noise approximation with an explicit
bias, variance, and mixing trade-off. It is relevant only if a model's
transition can be simulated but its density or derivative cannot be evaluated
in the required base measure. When the transition density and its parameter
derivative are available, the plug-and-play motivation does not by itself
justify this extra arm. No claim of statistical dominance follows without a
comparison. If investigated elsewhere, iterated filtering must carry
a distinct target identifier from an exact fixed-parameter score.

Twisted particle filters should be an optional proposal-quality arm because
they preserve the likelihood through a change-of-measure correction while
optimising an asymptotic normalizer-variance criterion. Multilevel particle
filters remain a deferred discretization/coupling family: their telescoping result
does not by itself yield an unbiased score or an unbiased ratio derivative.
The variational particle-objective literature (FIVO/AESMC/VSMC) belongs in the
target-boundary table so that an ELBO gradient is not reported as an observed-
data score.

These additions do not change the central support split. The inspected sources
provide no generic ambient-density KDM or PaRIS theorem for deterministic or
singular DSGE transitions. Phase 6 still requires a base-measure or
disturbance-coordinate derivation and its own score estimator before any
regular-track result can be transferred.

## Open prerequisites and amendment review

This master specifies both the implementation and the experiments; it is not
evidence that either has passed. The following unfinished work is assigned to
phases in this master. It blocks only the dependent research rows, not the
start of the master or unrelated verified work.

| Obligation | Producing phase | Current evidence and required work |
|---|---|---|
| Coordinator, study inputs, and resume/report services | 0B/0C/0G | Implemented under score_study, with backend-free tests and real baseline run/resume. Research aggregation and later method coverage remain separate milestones. |
| KDM integrated/resampling endpoint | 0D, with shared baseline regression in 0C | Implemented and tested in the protected 0D snapshot; 48 CPU combination rows and three GPU consumer rows complete. Main integration preserves the concurrent native loops; both consumers pass main CPU direction checks and the focused GPU callback retry. |
| Parameter-dependent initialization | 0C and every affected 0D--0G adapter | Explicit initial-state/covariance tangents implemented; six-parameter canonical finite-difference regression passes. Extend the same contract to each later adapter. |
| UKF/KDM/SGQF moment lifecycle | 0C--0E | Shared UKF/SGQF and persistent Gaussian-mixture prediction, observation conditioning and reset carry execute with total analytical tangents. Provider snapshot 677e38a8 has CPU/GPU and independent selection/claim checks. Nonlinear snapshot a99a1c55 reaches the same shared consumers. The mixture construction is a local assumed-density candidate, not a reproduction of Younis's learned filter. |
| Twisting/iAPF | 0E | Scalar Gaussian and nonlinear consumers, adaptive N, frozen-fit analytical scores and the underflow guard execute. Fresh calibration at `418e5388` completes 54 numerical rows, both conditional heuristic tables and twelve frozen-baseline claim rows; fourteen source rows underflow and twelve selected claims remain blocked. The baseline fails the descriptive EKF/UKF screen on all six datasets; no promotion or supported ranking. Seven driver tests pass in each checkout, and eight study fingerprints / 54 result digests pass the saved-result audit. Next: source-grounded density-objective, constraint and stopping repair, then fresh calibration. Wider dimensions and iAPF-moment integration into LEDH remain open. |
| KDM as a LEDH control variate | 0D | Known-zero-center mixture density-score control and independent calibrated biased-score blend execute. The tiny exact-control fixture has worse descriptive validation error; no ranking. Other unknown centers cannot inherit this result. |
| Coupled finite differences | 0F | Three stencils, full directional reconstruction/covariance and frozen-design consumption execute. Forty-row pilot plus independent eight-row selection and two held-out mechanics rows complete. Conditional normalization, ratio covariance and N/2N/4N consistency reports also execute; 48 diagnostic rows complete. The eight-row pooled LEDH consistency association is only r=0.058, descriptive and insufficient for bias calibration. Larger replication and calibrated combinations remain open. See phase-0f-result-and-refresh.md and phase-0g-capacity-normalization-result.md in the active artifact root. |
| Companion manuscript | 0A, then relevant phase repairs | Synchronize FD smoothness, stochastic MSE, direction convention and covariance lifecycle before using it as the revised implementation specification. |
| Additional models, row matrix, and final reports | 0G | Scalar nonlinear transition/observation adapters, a refined numerical grid reference, EKF/UKF and corrected particle baselines execute at a99a1c55. All 96 rows across three regimes complete. At these untuned settings, every LEDH covariance candidate loses descriptively to EKF in every regime; this vetoes promotion, not the research direction. GPU capacity checks at d=4 and d=12, N=64, T=3, o=2 complete for Kalman and all three shared LEDH consumers. Scientific scope tuning, wider model coverage, replication and terminal reporting remain open. |
| Serious run | Applicable phase and preceding refresh | Fill the evidence contract, defaults audit, partitions, finite attempts/compute budget, exact environment/commands, and unique output root. |

The 15 September provider/nonlinear result and required next repair are recorded
in [phase-0g-provider-nonlinear-result.md](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/phase-0g-provider-nonlinear-result.md).
The [density-fit audit](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-density-fit-audit.md)
adds a required Phase0E specification: an unrestricted covariance/scale search
in the paper's density-scale criterion can have a zero infimum without a
finite minimizer. The implemented arm specifies its local optimizer and
termination semantics; a small absolute fitting loss alone cannot certify its
shape or downstream score quality. The existing local fitted comparator
remains available under its declared target and method identity.
The completed normalization/capacity allocation is recorded in
[phase-0g-capacity-normalization-result.md](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/phase-0g-capacity-normalization-result.md).
The completed consumers have been integrated into the main checkout while
preserving the concurrent native flow-substep loop. Protected/native-loop
parity, main consumer regressions and all nine planned GPU consumer smokes
passed, including a recorded repair of two initially omitted KDM callback
modules. The [integration result and next repair](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/phase-0g-main-integration-result.md)
records the versioned evidence. This establishes bounded implementation
coverage; scientific tuning and the remaining master phases are still open.

The companion manuscript at
[ledh_younis_kdm_score.tex](../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex)
now uses C5 smoothness, proves the finite-difference MSE decomposition with
its cross-term, separates stochastic error from deterministic order, and gives
the complete SGQF prediction/conditioning lifecycle. The PDF was rebuilt and
the amended pages inspected. Four MathDevMCP algebra checks passed; its two
equation-role audit abstentions are preserved and are not a whole-document
proof certificate. The existing ratio-bias witness remains valid, and
clarifying the flow determinant does not imply adding an OT determinant.

The [amendment review](../reviews/younis-score-master-program-amendment-review-2026-09-14.md)
records the mathematical checks, remaining implementation/documentation
obligations, and review scope. Historical reviews remain preserved. Their
procedural gates do not supersede the current repository policy, and a review
or document check is not a MathDevMCP audit result.
