# Authoritative master: controlled SMC and KDM/IWSG replication

**Replication-first revision, 3 October 2026.** The active execution contract
is [replication-first execution](controlled-smc-kdm-replication-first-execution-20261003.md).
The prior master revision is preserved at
`artifacts/controlled-smc-kdm-replication-20260928/master-restructure-20261003-01/master-before.md`.

The user's target is to finish the author baselines and establish a TensorFlow
implementation that is no worse numerically on source-valid cases. Additional
precision, projection, damping, covariance floors, joint-KL calibration and
other changes beyond the author method are **E0 enhancements**. They are not
part of the replication claim and must not block the source or port phases.
Source-prescribed safeguards and solver behavior remain part of replication;
correcting our implementation to match them is replication work.

**ACTIVE; BOTH REPLICATIONS INCOMPLETE.** Controlled-SMC source replication is
partial: six small Lorenz cases run successfully in explicitly labeled Octave,
but the selected full source matrices are unfinished. The 4,000-estimate
Figure 4 result is a TensorFlow GPU experiment, not an author-source result.
TensorFlow source-port parity is partial and fails in ill-conditioned fitting
cases. KDM/IWSG has primitive/reference evidence but no completed empirical
training replication. iAPF remains incomplete and deferred.

Dispatcher: `scripts/run_controlled_smc_kdm.py`. Active checkpoint:
[checkpoint](controlled-smc-kdm-checkpoint.md). The initial validation and
repair artifacts remain linked below as historical evidence; they do not alter
the new phase dependencies.



## Current execution state, 2026-10-03

The active order is source baseline, source-result assembly, frozen-input port
parity, and only then optional robustness. **No repair admission is a prerequisite
for replication.** The latest repair stage through attempt138 is retained as
historical diagnostic evidence. It repaired bookkeeping and exposed unresolved
arithmetic sensitivity, but it did not establish author replication or authorize
a changed numerical route.

The evidence is separated as follows:

- **Controlled-SMC author source:** six small Lorenz cases completed with the
  unchanged author files under explicitly labeled Octave CPU execution. This
  demonstrates source executability for those cases, not the full Figure 6,
  Table 1, neuroscience or multimodal matrices. MATLAB equivalence remains
  untested.
- **Controlled-SMC TensorFlow port:** fixed-input captures and solver crossover
  diagnostics exist. Raw normal-equation and centered-QR paths diverge in
  ill-conditioned cases. The port is incomplete under the no-worse contract;
  no projection, floor, damping or high-precision route is admitted as a repair.
- **Neuroscience Figure 4:** 4,000 estimates completed on the TensorFlow GPU
  implementation. They are a port experiment and do not count as an unchanged
  author-source replication.
- **KDM/IWSG:** exact mixture-Kalman and sequential consumer checks pass in their
  declared scopes. Source training settings and independent empirical learning
  remain incomplete.
- **iAPF:** incomplete and deferred; controlled-SMC or KDM progress cannot close
  the iAPF claim.

The next work is:

1. **C0 source lock:** freeze the selected source drivers, data provenance,
   runtime, RNG, solver behavior, particle counts, refinements, comparators and
   output statistics. Resolve the Lorenz data/driver and multimodal 100/101
   observation discrepancies before full source launches.
2. **C1 source baseline:** execute the unchanged author source in a declared
   MATLAB or Octave reference lane. Preserve every finite result, warning,
   early stop and failure. Complete the selected source matrices and APF/BPF
   comparators before judging the port.
3. **C2 source verdict:** compare source summaries with the paper using frozen
   uncertainty criteria. Mark missing settings or runtime differences as gaps;
   do not replace them with unlabelled reconstruction choices.
4. **P0/P1 port:** replay captured source inputs, correct translation or
   solver-compatibility mismatches while preserving the author method, then run
   the selected port matrix. Require fixed-input output/invariant agreement and
   no additional source-valid invalid outcomes; treat source failures as
   classified evidence.
5. **K0–K2:** recover KDM settings, finish normalized reference/consumer checks,
   and run the scoped empirical training replication independently of the C
   phases.
6. **E0:** only after the affected baseline and port records exist, evaluate
   numerical enhancements as separate methods with separate targets and
   non-harm criteria.

The live budget, attempt ledger and running-process state are authoritative in
[the checkpoint](controlled-smc-kdm-checkpoint.md). No source, port or repair
run is launched by this status update.

### Historical source-input localization record

The [matched-input localization plan](controlled-smc-lorenz-source-port-localization-20261001.md)
records the earlier fixed-input port diagnostic after native audit86. Two dispatcher phases capture
the author's actual regression inputs, verify unchanged source results, then
exercise the shared TensorFlow regress/weights/fit endpoints on those inputs.
The new phase names are `c1_lorenz_author_capture` and `c2_lorenz_source_port`.
Their aggregate caps are90 CPU and900 GPU seconds; at most four attempts total.
The attempt ceiling is90; the existing compute allocations are unchanged.

This is executable localization. It does not promise that an unknown repair,
full Figure6, source-M16 Table2 or KDM learning replication will automatically
finish. Diagnostic completion cannot unblock failed batch/repair validation.
The next implementation change must follow the first measured mismatch.
Use the linked plan for exact commands and the checkpoint for execution state.

Terminal: capture88 and comparison90 have completed; see the
[result and terminal review](controlled-smc-lorenz-source-port-result-20261001.md).
All six original source returns are unchanged. All22 source-stage weight,
log-normalizer and ESS comparisons pass; both TF regression solvers differ from
source coefficients on identical inputs. All11 numerical endpoints trace once
under GPU/XLA. No precision repair or full-replication claim follows. The next
bounded specification must distinguish Gram/RHS formation from solve behavior
at the first mismatch, then validate the chosen repair on full/failing consumers.
At that historical stage the diagnostic ceiling90 was consumed; stages91–95
are recorded above. Future repair
phases are conditional, not ready for an unattended end-to-end launch.

## Historical repair background, 2026-09-30

The [direct author-code audit](controlled-smc-lorenz-author-native-result-20260930.md)
now passes all six native adaptive cases, including four refinements at F2.5.
The unchanged source ran in13.04 CPU seconds for this small diagnostic.
Our earlier source reference used only one refinement at F4.8801; the later
four-refinement reference was a new SVD implementation. This was an unfinished
baseline, not evidence that the author code could not run. Before another
safeguard, compare the source and port on identical saved inputs. The native
success does not clear the different-seed GPU failure or complete Figure6.

The prior scoped neuroscience and Lorenz repairs finished, but fresh Lorenz
designs exposed another numerical failure. The latest
[precision-repair result](controlled-smc-lorenz-precision-result-20260930.md)
closes attempts77–84: three safeguards were rejected; a QR rank-validity
repair passed37 checks. The full empirical Lorenz lane remains blocked.
See the [checked repair result](controlled-smc-kdm-repair-execution-result-20260929.md).
The [guided plan](controlled-smc-guided-repair-20260929.md), [Lorenz consumer plan](controlled-smc-lorenz-consumer-repair-20260929.md),
[solver validation](controlled-smc-lorenz-solver-validation-20260929.md), and
[centered recursion repair](controlled-smc-lorenz-centered-recursion-20260929.md)
preserve the predeclared contracts and failed attempts.

| Problem | Checked finding | Next justified action |
|---|---|---|
| Wrong intermediate filtering target | Removed the remaining future-observation twist; 43 guide/identity checks and 10000 comparison rows completed | Preserve conditional underperformance as a promotion veto; do not tune on these outcomes |
| Lorenz full covariance and RK4 absent | All8 identical-input refinements and original forward/APF/stopping checks pass under revised72 contract;70 independent-trajectory coefficient FAIL retained | Scalar cost pilot73 forecasts37.23h, beyond remaining32.39h; validate batching and rerun cost before full matrix |
| Fresh Lorenz fits can have indefinite precision | SVD77 confirms eigenvalue−1730;78 precision/backtracking repairs overflow;80 covariance repair is finite but82 fails11/112 independent checks |86 unchanged-source cases all pass; first compare original and port calculations on identical clouds/draws, then decide whether a repair is necessary |
| Gram-matrix rank check falsely accepts an unresolved fit | Independent singular ratio8.11e-9; new QR-factor guard83 rejects it and preserves all8 healthy fits in37 checks | Retain conservative fail-closed guard and both diagnostics |
| Lorenz raw normal equations unstable | Raw Gram conditioning near machine precision; centered QR reduces the common-target residual; independent SVD validates the same objective | Use the explicitly named, scoped equivalent-solver repair; raw author coefficients remain numerically unmatched |
| Backward-integral cancellation | Exact Gaussian-ratio recursion repairs coefficient parity; attempt32 passes28 checks at unchanged tolerances | Preserve source and independent-reference checks during wider validation |
| XLA compile failure | Installed GPU backend does not support slogdet; checked Cholesky log determinant passes | Keep XLA enabled, stable signatures and numerical validity guards |
| Multimodal source ancestor bug | Source component probabilities use the wrong parent;98 affected fixture transitions; corrected TF forward passed18 checks | Validate NNLS fitting and full101-observation consumer before Table2; preserve corrected-source distinction |
| RBF coefficients and top16 retention are numerically sensitive | Full paired validation58 fails;59 finds matched targets/full predictions but different retained knots on both fresh seeds | Stop tolerance search; final4096-repeat diagnostic61 found a normalizer discrepancy in all four policies and poor filtering means; coverage62/63 identifies top16 truncation as the main severe-coverage mechanism; 66–68 completed: full recursion retained every positive coefficient; both refitted normalizer intervals pass, but both old saved-fit controls fail the fresh screen; filtering underperformance remains; preserve this scope and continue Lorenz |
| KDM empirical learning incomplete | Exact AppendixB coefficients and schedule remain unrecovered | Resolve provenance or label reconstruction hypotheses; validate a target-specific training protocol |
| Old neural bearings repetition cost | Ten unchanged trials would require85.89 GPU hours | Measure equivalent faster training before allocating repeats; this estimate does not apply to parametric AppendixB |

At that historical stage, the allocation was CPU500/GPU18000 seconds, with86 attempts recorded
through the native source audit (85 was a reporting failure;86 passed). No numerical job is
running at that stage. About32.38 inherited GPU hours remained then; subsequent
execution and current balances are recorded in the current-state section above. The CPU tranche was
expanded within its existing inherited balance after68. This draws on the inherited
campaign balance, not a fresh48-hour grant. Failed attempts remain charged.
Numerical and graph checks do not establish full empirical replication, general
default readiness, HMC readiness or stochastic superiority.

The [RBF consumer audit](controlled-smc-rbf-consumer-audit-20260929.md) is complete:
attempt34 confirmed the wrong ancestor index in the pinned author source;
attempt35 passed18 GPU/XLA corrected-kernel checks. This is a code defect
relative to published Algorithm1/eq11 and supplement9.1, not evidence that
the paper's mathematics is wrong. The active
[fitting validation](controlled-smc-rbf-fitting-validation-20260929.md)
tests the actual NNLS objective and full101-observation fit-to-forward consumer.
The source attempt exposed NNLS nonconvergence. Frozen GPU validation passed42
after termination calibration. A fresh-SVD independent reference passed all202
backward fits in50; full GPU fit-to-forward failed51 despite valid individual fits. Diagnosis53
locates amplification of coefficient differences by source top16 truncation;
test54 rejected all four factors against the fixed0.01 reference. Reference sensitivity55 and matched-setting comparison56 show factor0.1 reproduces both frozen retained policies. The [paired-stopping plan](controlled-smc-rbf-paired-stopping-validation-20260929.md) now tests that explicitly named finite procedure on fresh particle seeds, without relaxing criteria or rewriting the earlier failures. The [current result](controlled-smc-rbf-fitting-execution-result-20260929.md)
records the source defect, failed attempts and solver diagnosis. All phases
were executed inside their then-active CPU300/GPU18000 allocation; no empirical claim follows from forward parity.

## Decision and scope

Controlled SMC is now the first replication track; retain
MDPF/KDM-IWSG as the second. Controlled SMC is a direct methodological successor
to iAPF with an explicit backward fitting procedure and accessible author code.
Its neuroscience/Lorenz negative-log quadratic regression removes the particular nonlinear
Gaussian-plus-floor fitting problem in our reconstruction. It does not remove
regression conditioning, inadequate cloud coverage, approximation error or
proposal-validity risks.

Retain all iAPF source, tests, results and failure classifications. Under this successor, iAPF is **incomplete; further execution deferred**.
Reopen it if original implementation details are recovered or a concrete new
diagnostic resolves a remaining question. Controlled-SMC success must never be
reported as original-iAPF replication. This change is justified by a clearer
source specification and reference implementation, not by treating failure to
match finite BFGS endpoints as evidence that iAPF is scientifically invalid.

| Track | Replication target | Scope boundary |
|---|---|---|
| C: controlled SMC | Heng, Bishop, Deligiannidis and Doucet, *Controlled Sequential Monte Carlo*, Annals of Statistics (2020): backward policy fitting, multiplicative refinement, controlled proposals, weights and normalizing constants; quadratic neuroscience/Lorenz and separate NNLS/RBF multimodal examples | Initially reproduce the selected filtering experiments, not every static-model experiment in the paper. Recover and freeze the exact source-to-figure matrix before empirical runs. |
| K: MDPF/KDM-IWSG | Younis and Sudderth, *Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes*, NeurIPS (2023): kernel-density mixture filter, normalized predictive densities, IWSG learning, Appendix B and declared bearings experiments | A saved checkpoint, point-error result or working forward filter does not complete the training replication. KDM is the representation, not a third method. |

Controlled-SMC source experiments are NeuroScience, ToyMultimodal and
Lorenz. The [remaining-source audit](../reference/controlled-smc-remaining-source-audit-20260929.md)
now maps Lorenz Figure 6/Table 1 and manuscript Appendix F multimodal results;
it records missing normalization constants, dataset provenance and horizon
discrepancies that must be resolved explicitly in their executable phases. Their individual completeness must be reported. The static Gaussian/LQG
demo is a useful analytic control, **not a state-space filtering replication**.
A local linear-Gaussian Kalman fixture is also a correctness control; its success
cannot substitute for the selected paper experiments.

HMC, parameter-score extensions, LEDH, SIXO, annealed controlled SMC and
forward-only alternatives remain outside the initial replication scope.
The two tracks have independent dependencies: unfinished iAPF or controlled SMC
must not prevent valid KDM work.

## Checked starting point and sources

- [Previous master](iapf-kdm-replication-master-program-2026-09-24.md) and
  [checkpoint](iapf-kdm-replication-checkpoint.md) preserve the R4d outcome:
  paper-horizon coefficient parity fails; original iAPF settings remain partly
  unknown. Small consumer tests and frozen-twist value checks are reusable only
  for their tested functionality and source snapshots.
- [Follow-up source audit](../reference/iapf-followup-methods-20260928.md)
  records the inspected controlled-SMC paper sections, code anchors and limits.
  Author repository: [controlledSMC](https://github.com/jeremyhengjm/controlledSMC),
  local path `.localresources/code/heng-controlled-smc`,
  commit `2696f6d611aae172d10102cb7bfbfd314bb6b88f`.
- Controlled-SMC paper: [published DOI](https://doi.org/10.1214/19-AOS1914);
  local publisher PDF .localresources/papers/heng2020-controlled-smc-published.pdf.
  The original heng2020-controlled-smc.pdf contains arXiv v3 and appendices.
  The optimal recursion, regression/refinement equations and neuroscience
  settings have now been matched to the publisher version. The remaining-source
  audit maps Lorenz Figure 6/Table 1 and arXiv Appendix F; the final published
  multimodal supplement and several exact data/driver conventions remain unresolved.
- The unchanged author neuroscience code ran successfully under the explicitly
  declared Octave reference exception. The short and full-horizon fixed-input
  TensorFlow comparisons that belong to that scalar scope pass. The separate
  multivariate Lorenz port has ill-conditioned fitting mismatches. Scalar
  precision projection and normal equations remain explicit source-comparison
  settings, not universal defaults.
  No explicit source reuse license was found; the TF implementation is derived
  independently from the mathematical expressions, not copied source.
- KDM author code remains pinned at
  `.localresources/code/younis-mdpf-neurips-2023`,
  commit `b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18`.
  Carry forward the normalized-density, Appendix B, TensorFlow-port and
  independent-repetition gaps from the
  [whole repair plan](iapf-mdpf-whole-repair-plan-20260928.md).
  Its nonlinear Cappe example cannot stand in for Appendix B's linear-Gaussian
  dynamics with bimodal observations.

## Research intent and evidence contract

**Question:** can we first reproduce each declared source procedure and
empirical experiment, and then establish a TensorFlow/TFP implementation that
is no worse numerically on source-valid cases?

The source lane is authoritative for the replication claim. The TensorFlow lane
is a separate portability claim with its own fixed-input and repeated-run
criteria. High-precision solves, projections, damping, floors, joint-KL
calibration and other changed routes are enhancement evidence only; they cannot
be used to close a source or port gap.

**Mechanisms:** C learns and refines a backward twisting policy; K represents
multiple modes and trains through the source IWSG estimator. These mechanisms
are assessed against their own mathematical targets, not against one shared
convenient toy problem.

**Expected failures:** C may have ill-conditioned regressions, inadmissible
proposal precision or poor training-cloud coverage. K may have density or
gradient mismatches, insufficient training, unresolved source settings or
inadequate compute for repetitions.

| Diagnostic role | Predeclared meaning |
|---|---|
| Engineering pass | Actual reference and TF consumer call chains implement the declared operations; executable fixtures check fitting, proposals, normalizers, resampling/weighting and, for K, learning gradients. |
| Numerical validity / promotion veto | Nonfinite accepted states or weights, invalid covariance/density normalization, missing gradient terms or failed analytic/reference identities block a correctness claim for that route. |
| Scientific promotion | Complete the frozen experiment matrix and independent repetitions; use declared uncertainty comparisons for likelihood/density/state/training outcomes and account for adaptation/training cost. Execution alone is insufficient. |
| Heuristic-dominance veto | Conditional underperformance against a predeclared simple comparator blocks a superiority/default claim. A faithfully reproduced negative result can still establish replication; do not conflate these claims. |
| Continuation veto | Wrong target/data, corrupt outputs, invalid oracle, missing indispensable diagnostics or exhausted budget stop the affected run. A failed candidate normally triggers the planned diagnosis/repair. |
| Repair trigger | Source mismatch, solver conditioning, poor coverage, gradient discrepancy or learning failure selects the next smallest discriminating check. Do not blindly repeat a failed campaign. |
| Explanatory only | Regression loss, coefficient drift where the representation is nonunique, ESS, trace count, compilation time and individual-seed outcomes cannot establish empirical replication or superiority. |

Exact/near-exact references are certifying checks where computable: Kalman
likelihoods for a linear-Gaussian control, analytic Gaussian normalizers for
controlled proposals, and the correct mixture-Kalman reference for the actual
Appendix B model. Any approximate oracle must be converged or carry uncertainty.

Construct and freeze the heuristic sets before claim runs. For C, begin with
the bootstrap PF, an observation-guided EKF-proposal PF and an
observation-guided UKF-proposal PF, with normalized proposal corrections; include
the source APF where supplied. These respectively test whether learning adds
value beyond no proposal adaptation, local linearization and sigma-point
conditioning. Applicability and exact formulas must be checked for each model.
Evaluate separately by source noise/dimension regime and by prespecified
informative/extreme-observation regimes; never select methods on these checks.

For K, retain the planned normalized persistence, constant-motion, UKF and BPF
density comparators, using matched information and circular observation
semantics. Evaluate state error and normalized density scores conditionally on
source-defined difficult regimes, including ambiguity and abrupt motion where
present. Freeze the regime definitions without looking at claim outcomes.

Paper-level statistical criteria, repetition counts, intervals and tolerances
must be extracted and predeclared in each executable phase specification.
Separate calibration and untouched claim data. Where published results lack
enough uncertainty information for exact numerical equivalence, state that
limitation and use a separately justified criterion; do not invent precision.
Keep likelihood and log-likelihood bias distinct. No stochastic ranking follows
from a few seeds or a hard-screen pass.

## Current completion and remaining work

| Lane | Current status | Remaining requirement |
|---|---|---|
| G0 — contract and scope | Restructured on 3 October 2026 | Freeze source matrices, runtimes, data/RNG, solver conventions and predeclared port tolerances |
| C0 — controlled-SMC source lock | Partial | Lorenz Figure 6/Table 1 provenance and the multimodal driver/horizon discrepancy remain unresolved |
| C1 — unchanged author source | Six small Lorenz Octave cases pass; full selected source matrices incomplete | Run the declared source drivers and preserve all warnings, failures, costs and outputs; MATLAB equivalence remains a stated limitation |
| C2 — source replication verdict | Incomplete | Assemble paper-level summaries and uncertainty from the complete source ledger; do not substitute the TensorFlow matrix |
| P0/P1 — TensorFlow port | Partial fixed-input localization; ill-conditioned fitting parity fails | Correct source-semantics mismatches, pass fixed-input checks, then run the no-worse matrix; no new algorithmic safeguards |
| K0/K1 — KDM source/reference | Partial; primitive and exact mixture-Kalman controls pass | Recover Appendix B settings or label a reconstruction; finish normalized reference and estimator call-chain checks |
| K2 — KDM empirical replication | Incomplete | Run unchanged author training/filtering first, then the matched TensorFlow comparison; keep source and port verdicts separate, with independent seeds, costs and uncertainty |
| E0 — robustness | Deferred | Evaluate changed numerical routes only after the affected baseline and port records exist |
| J — terminal review | Historical tranche reviews complete | Review source faithfulness, port non-regression, KDM evidence and remaining gaps separately |

The completed 4,000-estimate Figure 4 result is TensorFlow GPU evidence, not an
unchanged author-source result. Figure 4 left/FFBS and Figure 5/PMMH remain
separate scopes. The Lorenz and multimodal source matrices require their own
source-native executable specifications. A primitive control, a short chain,
a fixed-sample gradient check, a heuristic comparison or a port pilot cannot
close an empirical replication claim.

## Phases and dependencies

| Phase | Work and required evidence | Completion condition |
|---|---|---|
| G0 — replication contract | Freeze source, port and enhancement claim boundaries; preserve old plans and live budget; distinguish source-valid failures from repairs | Every run declares one lane and cannot promote a changed route into source evidence |
| C0 — controlled-SMC source lock | Map paper equations, author files, driver, data, runtime, RNG, solver, particle count, refinement schedule, comparator and statistic for every selected experiment | Each experiment is executable or blocked by a named missing source item |
| C1 — author-source replication | Run unchanged author code in MATLAB or explicitly labeled Octave; preserve raw outputs, warnings, failures, timings and source costs | Complete source ledger and paper-level comparison for each selected C experiment, or an explicit partial result |
| C2 — source verdict | Aggregate source runs with declared uncertainty and separate replication from heuristic performance | Source claim is supported, partial, or unavailable with the exact remaining gap |
| P0 — fixed-input port parity | Replay source inputs and control decisions, trace the first mismatch and correct translation or solver-compatibility errors without changing the method | Required outputs and invariants pass frozen scaled tolerances; localization alone is partial |
| P1 — port non-regression | Run the matched matrix; compare validity, failure frequency and downstream likelihood/state accuracy under predeclared tolerances and uncertainty | No extra failure on source-valid fixed cases and non-inferiority is supported, or port remains incomplete/inconclusive |
| K0 — KDM source lock | Recover Appendix B/bearings settings, data, initialization, estimator and training schedule; classify facts versus reconstructions | Exact settings recovered or reconstruction is explicitly labeled and excluded from exact-author claims |
| K1 — KDM reference/consumer | Validate normalized mixture-Kalman reference and actual IWSG/KDM sampling, weighting, stop-gradient and optimizer call chain | Declared value/gradient/consumer checks pass; fixed-sample parity is not called training replication |
| K2a — KDM author empirical replication | Run unchanged author training/filtering with independent seeds, costs, comparators and uncertainty | Source empirical scope is complete or its source/compute gap is explicit |
| K2b — KDM TensorFlow empirical replication | After fixed-input value/gradient/update parity, run the matched TensorFlow scope and compare downstream density/state accuracy | Port scope is complete with a separate no-worse verdict, or remains incomplete/inconclusive |
| E0 — robustness extensions | Test high precision, projection, floors, damping, KL/model-adequacy controls and other changed routes | Separate target, separate result and non-harm evidence; no effect on C/P/K status |
| J — terminal review | Review source faithfulness, port non-regression, KDM evidence and scientific interpretation independently | One decision table per lane and no cross-lane promotion |

Dependencies are `G0 → C0 → C1 → C2`, `C1 → P0 → P1`, and `G0 → K0 → K1 → K2a → K2b`. Apply the source-to-port dependency per selected experiment: a completed source reference can feed its port without waiting for unrelated experiments. `C2 + P1 + K2a + K2b → J`. E0 follows replication and is never its prerequisite. K work may proceed independently of C work. iAPF remains a separate incomplete track.

## Defaults and assumption audit

| Choice | Provenance/status | Failure risk and earliest check |
|---|---|---|
| Log-quadratic C fit and multiplicative refinement | Paper §§4.2–4.3; source-faithful target within its declared approximation family | Wrong sign, cross-term convention or update composition; analytic quadratic fixture and source intermediates |
| Normal equations, precision projection and source margins | Inspected author implementation, not universal TF defaults | Conditioning or invalid full covariance; rank/condition diagnostics and eigenvalue/Cholesky checks before long runs |
| Stable solve or damping changes | Candidates, not silent source replacements | A changed fitted policy can change the method. Classify mathematical equivalence or explicit modification, and evaluate non-harm on healthy and boundary cases. |
| FP64 reference and GPU/XLA TF port | Existing reference-parity convention and repository execution policy | Arithmetic sensitivity or compilation problems; bounded fixed-input comparisons, peak-memory measurement and one trace per signature |
| K optimizer, architecture, coefficients and schedule | Exact recovered source values where available; unknowns remain hypotheses | Wrong experiment or transferred defaults; objective/scaling controls and target-specific calibration before claims |
| Seeds, horizons, particle counts and repetition count | Recover from each source experiment, then freeze | Convenient reductions can change the claim; source-to-run manifest check and explicit pilot labels |
| Existing iAPF code reuse | Engineering candidate only | An old twisted-filter implementation may have different policy algebra; test the new consumer, never infer C correctness from iAPF tests alone |

Repeated kernels use stable tf.function input signatures and native TensorFlow
loops, with XLA by default. No pfor, production NumPy or scalar-loop training
fallback. GPU commands require trusted/escalated access and verified memory
growth before initialization. Reference backend exceptions are named and kept
outside admitted runtime paths.

## Execution, budget and status

Active output root: docs/plans/artifacts/controlled-smc-kdm-replication-20260928/.
Active plan: [replication-first execution](controlled-smc-kdm-replication-first-execution-20261003.md).
The dispatcher reuses existing lock, process, budget, timeout, recovery and
manifest helpers. Each attempt preserves source snapshots, exact commands,
environment, input hashes, seed, device/memory policy, wall charges and results.
Original evidence is retained. Superseded comparisons are identified, not erased.
The old Lorenz repair phases remain diagnostic history; they are not an active
prerequisite and must not be launched as a substitute for C1, C2, P0 or P1.

Implemented commands, from the repository root:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py status
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py report --phase PHASE
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase PHASE
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py recover

The active plan field emitted by the dispatcher is the replication-first program; the implemented diagnostic phases and latest repair result are historical status fields.
The report action emits bounded evidence fields through the same approved command
prefix, avoiding separate ad hoc inspection commands and repeated approval prompts.
Historical diagnostic additions include c3_guided_validation, c3_guided_matrix,
c1_lorenz_consumer, c2_lorenz_consumer, c1_lorenz_solver_reference,
c1_lorenz_centered_reference and c2_lorenz_centered_validation.
c2_lorenz_solver_validation failed its coefficient tolerance; the later centered
recursion passes without changing thresholds. Its failure remains preserved.
The old c2_parity is superseded by c2_parity_fp64. A completed phase is not
silently rerun.

The wait-run action serializes a reviewed phase after its active dependencies.
The previously queued jobs have terminated. The runner refuses silent reruns of
completed phases. Remaining C/K empirical phases require implementation and
their own bounded specifications before launch; they are not launch-ready merely
because they appear in this master. Use trusted/escalated permissions for GPU
jobs. Existing prefix approval and owner campaign authorization persist.

The inherited opening balances were36.375361 GPU hours and753.166936
numerical CPU seconds. The executable dispatcher and
[checkpoint](controlled-smc-kdm-checkpoint.md) carry the live ledger, including
every failed attempt, remaining balances and active processes. Reporting-only
rendering is separate. This is not a fresh48-hour grant.

The prior repair tranche consumed its recorded allocation and remains in the
campaign ledger. Further C/K source or port work needs a concrete phase
specification under the remaining inherited balance. E0 enhancement work is
separately budgeted and cannot consume the replication lane by default.

The [terminal review](controlled-smc-kdm-terminal-execution-result-20260929.md)
records decisions and unresolved gaps. The old MDPF cost of 8.589 GPU hours per
trial implies 85.89 hours for ten further trials of that unchanged arm, which
exceeds the remaining balance. This measured cost concerns the old neural bearings arm; it is not a cost estimate for the short parametric AppendixB experiment. Measure a source-equivalent faster training
route before allocating repetitions. Reduced repetitions support a partial
claim, not full replication. Further phases draw from the existing inherited
campaign balance and need concrete contracts/cost checks, not new per-retry
approval ceremony.

Activation itself consumes no numerical campaign budget. Actual attempts and
remaining resources are recorded by the executable dispatcher; isolated reference
dependencies and numerical runs are covered by the approved execution scope. Ordinary localized repairs within an activated,
unchanged campaign remain covered by its authorization; no per-retry approval
tokens or review chains are introduced.

## Current review and historical execution verdict

**Historical proposal: not executable as written. Repaired validation tranche: PASS and executed.**
The 20260929 execution specification repaired the first-tranche defects and
recorded PASS for bounded reference and fixed-input validation. The historical
Figure 4 matrix is TensorFlow port evidence, not unchanged author-source
evidence. Full replication remains incomplete; source, port and KDM phase
specifications still require implementation.

The main benefit is a recoverable procedure and reference code, not a guarantee
that least squares is stable or that controlled SMC will outperform simpler
filters. The review identifies the following material boundaries:

1. Replacing a method changes the replication claim. Separate labels and the
   preserved iAPF failure record prevent false closure.
2. The static Gaussian demo is not the filtering experiment; analytic controls
   and paper experiments are separate completion requirements.
3. Source normal equations/precision handling may reproduce numerical problems.
   Test their actual call chain and boundaries before choosing a solver repair.
4. Octave compatibility is established for the executed source path. Independent
   TF derivation avoids source incorporation; other model/code/version mappings
   remain explicit work. Source availability alone does not establish replication.
5. KDM source gaps and empirical cost survive this change. Its progress is
   independent; no claim that replacing iAPF repairs KDM is supported.
6. Exact method identity, output correctness, stochastic performance and runtime
   are different evidence questions. Fixed-input fixtures precede untouched
   experiment comparisons with uncertainty and conditional baselines.
7. The bounded validation stage and subsequent batching checks passed. The
   Figure 4 matrix now has implemented commands, uncertainty calculations and
   bounded costs; other empirical phases remain incomplete. Budget exhaustion
   cannot count as success.

Historical sequence below records earlier decisions. The current-state section above determines the next work.

1. The [full backward-refit diagnostic](controlled-smc-rbf-full-refit-20260930.md)
   at66–68 is complete. All202 fits and3094 final execution checks passed.
   Both full refits passed the fresh normalizer screen; both old saved-fit
   controls failed it. The filtering heuristic veto remains. Preserve the
   [terminal assessment](controlled-smc-rbf-fitting-execution-result-20260929.md#full-backward-refit-terminal-assessment-attempts6668).
   No further repetition of this closed branch. Full retention is a repair
   setting, not source-M16 Table2 replication; exact coefficient equality is
   not necessary for numerically nonunique NNLS. Freeze supplied101 observations
   and disclose the100/101 source-text discrepancy. Wider RBF empirical work
   still requires independent learning runs and the source noise regimes.
2. Lorenz refinement/APF validation is complete under the explicit revised72
   contract;70 remains failed. See [result](controlled-smc-lorenz-refinement-result-20260930.md).
   The [Figure6 cost pilot](controlled-smc-lorenz-figure6-pilot-20260930.md)
   passed numerical checks but forecasts37.23h versus32.45h remaining.
   Batch validation76 exposed an indefinite third refit in both scalar and
   batch execution; SVD77 confirmed it in36 checks. The bounded
   [precision repair](controlled-smc-lorenz-precision-repair-20260930.md)
   rejected three safeguards.83 fixes the missing rank guard;84 localizes the
   discrepancy to the first projection. The subsequent
   [native-source audit86](controlled-smc-lorenz-author-native-result-20260930.md)
   passes all six unchanged-source adaptive cases. The subsequent
   [matched-input result88/90](controlled-smc-lorenz-source-port-result-20261001.md)
   captures1616 original regressions with unchanged source returns. All22 weight
   stages match, but regression coefficients differ for both TF solvers before
   a recursion difference is needed. Next specify the identical-Gram/RHS
   cross-solve to separate matrix formation from solve behavior at the first
   mismatch. Keep literal source replication separate from numerical repair.
   A repair, if required, still owes independent correctness and full-consumer
   validation under the [earlier contract](controlled-smc-lorenz-precision-result-20260930.md).
   Only then resume [batching validation](controlled-smc-lorenz-batch-repair-20260930.md)
   and cost measurement before budgeting the full empirical matrix. Recover Table1 data or label a fresh
   generative-design replication.
3. Resolve K AppendixB coefficients and training settings where possible; freeze
   explicit reconstruction hypotheses otherwise. Validate normalized bearings
   baselines and the full learning consumer. Measure cost before independent repeats.
4. Complete conditional heuristic and uncertainty assessments, then review each
   full scope separately. Primitive parity or one SGD update cannot complete
   empirical learning replication.

The [repair result](controlled-smc-kdm-repair-execution-result-20260929.md)
records the completed fixes and remaining scientific limits. The automatic
checkpoint identifies the next executable phase and remaining resources.
iAPF remains incomplete and deferred.
