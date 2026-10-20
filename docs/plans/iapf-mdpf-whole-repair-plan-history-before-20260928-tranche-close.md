# Historical repair plan — superseded at tranche close

Use [the current repair plan](iapf-mdpf-whole-repair-plan-20260928.md).
Original body SHA-256: 65793382a9aa7e05e23a16e25a79587209bf3bde99079caabda66849d392653a

---

# Whole repair plan: iAPF and MDPF/IWSG replication

Date: 28 September 2026, Hong Kong (27 September UTC).
Status: ACTIVE — owner instructed refresh, review and execution on 28 September 2026 (Hong Kong).
The first bounded execution specification is [here](artifacts/iapf-mdpf-whole-repair-20260928-01/execution-spec.md).
Live phase outcomes and budgets are issued by scripts/run_replication_master.py status.
Parent authority: [replication master](iapf-kdm-replication-master-program-2026-09-24.md).
This is its active repair sequence, not a competing master or an assertion that all phases are executable.

## 1. Objective and scope

Establish a defensible replication of (a) Guarniero–Johansen–Lee iAPF, starting with Section 5.2's linear-Gaussian experiment, and (b) Younis–Sudderth MDPF with IWSG, covering Appendix B and the bearings-only experiment. Then establish that the intended BayesFilter TensorFlow/TFP GPU/XLA consumers implement the same declared methods. KDM describes the kernel mixture representation; it is not a third independent filter.

Report four separate outcomes for each method: source identity, implementation correctness in checked scope, empirical replication, and BayesFilter GPU-port validation. Each may be matched, reconstructed, passed in scope, failed, incomplete, or inconclusive as appropriate. A completed negative replication is not method promotion; a successful port cannot substitute for missing experimental replication. Neither status permits later HMC, learned extensions or LEDH integration automatically.

The original scoped completion requirements remain: iAPF d=5,10,20,40,80, T=100 and 1,000 complete independent algorithm repetitions per fixed dataset/dimension, with the source comparator allocations; MDPF Appendix B and the specified bearings objectives/comparators/repetitions. Original data/seeds may be unavailable. In that event report new-data protocol replication separately from exact numerical reproduction. Other iAPF examples, DeepMind/House3D and A-MDPF remain outside this first scope.

## 2. Starting evidence and corrections

- Both scientific replications remain incomplete. The independent R iAPF reference is our reconstruction, not recovered author code. Exact-twist/Kalman and selected derivative/target checks passed, but do not certify every consumer.
- The publisher iAPF PDF is now local. Section 5.1 adds “a regularized version” before Eq. (15). Its complete reconciliation with the accepted manuscript and thesis is not done. The thesis describes implicit local-BFGS regularization with no explicit penalty for Part III. Do not assume those statements fully specify the final article's implementation.
- The objective has a checked vanishing-density/amplitude escape. Our 1e-6 stationarity prerequisite answered a numerical-solver question and must not remain a universal veto on a separately declared finite-optimizer filtering diagnostic. Preserve the historical failures; no retroactive promotion.
- The recent 96 BFGS fits concern four selected saved cases, not 96 independent replications. Some fits miss the numerical screen without collapse; others collapse substantially. Repaired boxes become stationary at bad boundaries.
- One full author-source MDPF NLL trial and untouched evaluation completed. The original worker exit was not observed after supervisor loss; recovered terminal evidence and successful evaluation remain valid within their stated limits. Do not retrain merely to repair bookkeeping.
- Previously recorded resources: 36.567 GPU hours, 1,590.902954 CPU numerical-campaign seconds. Reconcile the main and supplemental attempt ledgers before launching; do not infer the remaining attempt count simply by subtracting 17 from 60.

Evidence: [reassessment](iapf-replication-reassessment-20260928.md), [iAPF result](artifacts/iapf-thesis-reconstruction-20260927-01/result.md), [MDPF trial](../../.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/source-trial-result-20260925.md).

## 3. Sequence, deliverables and advancement rules

### R0 — Reconcile sources and freeze the actual replication targets

Read the complete relevant method/theory, experiment and appendix sections in the final iAPF article; compare Algorithms 3–5, Eq. (15)–(16), regularization wording, stopping/resampling and experiment settings against the accepted version and thesis. Trace each corresponding executed R endpoint. Record what is explicit, inferred, reconstructed, or still missing.

Time-box additional public author-code/supplement recovery to two focused search passes, with at most two hours of active source-search effort. The time box limits repeated searching, not the required reading of material already recovered. Check author repositories, publication supplements, version/correction records and directly relevant follow-ups. Do not require a broad citation survey or invent citation counts. External author contact is a separate action requiring explicit permission; it is not a prerequisite for disclosed reconstruction.

For MDPF, reconcile the local arXiv version with the proceedings, pinned author code and actual resolved configuration. Map Appendix B, bearings NLL, optional MSE refinement and each comparator to its target output. Inspect the real dataset/observation geometry before proposing a triangulation baseline: a two-sensor construction is ineligible if the consumed data provide only one bearing. Check state conventions, angle wrapping, available initial information, training labels, seed setup, checkpoint selection and test-time bandwidths.

Deliver one concise source/specification table plus call-chain map. It must identify original versus reconstructed fit details, floor definition and role, objective/checkpoint identity, data partitions, exact source revisions, paper figure/table target, and unresolved choices. Maintain separate source-support, claim-support and version/correction records in this source audit; unsupported claims remain explicitly open.

Exit: each proposed run has an unambiguous target and known source status. If original iAPF fitting settings remain missing, exact original-code reproduction stays open, but a disclosed reconstruction may proceed. A material ambiguity that changes the claimed method blocks that claim-bearing run, not unrelated MDPF work. Adding an explicit penalty or replacing the objective is not silently authorized as replication.

### R1 — Repair the master runner and evidence reporting

Keep one authoritative master in the main checkout. Extend the existing dispatcher instead of building another supervisor. Its phase registry must distinguish implemented/not-implemented, eligible/not-eligible, process state, scientific outcome and consumed budget. Read balances from ledgers rather than returning hard-coded remaining hours. A validation mode checks commands, input/source identities, data partitions and budgets without starting kernels.

Implement durable logs, bounded output, process-group ownership, timeout handling and terminal manifests. Resume only from source-supported checkpoints with the recorded state; if exact resumption is unsupported, a fresh attempt consumes budget. Never rerun a completed trial because its supervisor disappeared. Prevent duplicate launches, propagate child errors, preserve unknown exit codes, and stop only owned processes. Every attempt gets a new directory.

Focused tests must exercise actual child success, nonzero exit, timeout, supervisor loss/recovery, duplicate refusal and exhausted-budget refusal. Verify that a solver-screen failure can schedule an explicitly planned repair without marking the entire direction rejected. Do not build approval tokens, cryptographic launch gates or a review chain.

Exit: bounded fault-injection tests pass; status reports completed historical work honestly and identifies the exact next executable phase. Until implemented, this proposal's phase commands must not be advertised as working.

### R2 — Restore a complete, inspectable iAPF reference

Audit the actual filter call chain: initial X1 distribution and twist normalizer; Gaussian product/convolution; backward target; positive floor; Eq. (15) fit; ESS-adaptive accumulated weights; skipped-resampling normalization; particle doubling/window indexing; and independent final filtering draw. Reuse controls only when their executed dependencies match. R/FP64 remains an independent CPU reference, not the BayesFilter production implementation.

Use at most two source-motivated finite-local-BFGS reconstruction recipes if the author recipe is missing. R0 must freeze the initializer, covariance/amplitude coordinates, derivative method, objective scale, optimizer controls, floor, and finite work limit before new data are evaluated. A second recipe is allowed only for a named unresolved source choice, not because the first misses the paper's score. Do not silently substitute relative-shape loss, log-quadratic fitting, a penalty, adaptive clipping or a box constraint. Log-quadratic and moment fits can remain labeled diagnostics/baselines.

Record loss, gradient, curvature where feasible, mean/variance/amplitude trajectories, density/target scales, floor contribution and probability validity. Mathematical invariants and correct sampling/weighting are correctness requirements. Nonstationarity alone is explanatory for the finite-optimizer experiment. Density collapse rejects a useful-fit claim; if the full probability operations remain valid, preserve the downstream outcome rather than silently substituting a fallback or deleting the failure.

Run a small complete pilot: d=5,T=100,N0=1000, source k/tau/resampling settings; two fresh datasets, four independent adaptive/final-filter repetitions per recipe per dataset; maximum two recipes (16 complete iAPF replications). Baselines share each dataset but use separately recorded random streams. All repetitions and failures are retained. The pilot is for detecting execution/fitting pathologies and measuring cost, not ranking methods or proving unbiasedness. Freeze actual seeds and all numerical tolerances in the phase manifest before launch.

Bound reference controls to 300 cumulative CPU wall seconds and the pilot plus localized repairs to 900 seconds, subject to the remaining parent cap. These are ceilings, not expected costs or scientific stopping rules. Hitting a computational cap means INCOMPLETE, never convergence. Do not silently transfer the pilot to GPU to evade the CPU cap.

Exit: a fully specified reference completes end-to-end in checked cases, probability identities/controls pass, and failures have a localized explanation. Passing stationarity is neither sufficient nor universally necessary. If two justified recipes fail, preserve that result and stop the same-choice search. The next question is whether the source recipe, objective specification, floor or implementation is wrong; any new method-changing repair must be named separately.

### R3 — Close MDPF's reference gaps independently

Start from the completed trial and saved test trajectories/checkpoints. Reproduce its evaluation without retraining; verify the actual NLL-trained checkpoint, bandwidths, angle/state units, normalization and metric aggregation. Keep optional MSE refinement distinct. Verify effective RNG behavior; configured seeds are insufficient if the source ignores them. Source behavior and a corrected RNG implementation must be separately labeled.

Complete the Appendix B consumer from the paper's actual scalar generative model, trainable parameters, objective and optimization schedule. The similarly named author toy dataset is not automatically this experiment. Check the mixture-filter oracle against the specified generative model, not an arbitrary learned proposal.

Trace real IWSG consumers, including stop-gradient placement, numerator/denominator conditioning, component sampling, finite-sample self-normalization, density epsilons and truncated BPTT. Distinguish pointwise gradient parity, expected-gradient claims and learning behavior. Preserve source quirks for author-code replication and document any mathematical mismatch; do not silently “correct” the source and call it identical. Previously passed tests are reusable only for the same endpoints and source closures.

Construct and freeze eligible cheap baselines before interpreting learned quality:
- initial-state persistence density, with dispersion calibrated using training data;
- constant-motion predictive density estimated from the same training information;
- a fitted classical EKF/UKF with circular observation handling;
- a fitted generative BPF with the same observations and training information.
A true-parameter generative BPF or mixture Kalman calculation is an information-advantaged reference/oracle, reported separately. Observation-only triangulation is supplemental only when source geometry supports it.

For bearings, evaluate early (1–17), intermediate (18–50) and late (51–150) windows, and predeclared ambiguity/outlier strata. Set geometry/ambiguity boundaries from training/calibration data without examining learned test results. For Appendix B, condition on separated/overlapping modes and early/late times. All NLL baselines require normalized densities fitted without test labels; no arbitrary test-tuned KDE bandwidth.

Exit: source and mathematical identities are explicit; the completed trial has fair conditional comparisons; Appendix B runs the specified target. Learned underperformance is a headline and a promotion veto, not permission to alter the replication to win. No between-training-run ranking from one checkpoint.

### R4 — Validate the TF/TFP GPU/XLA consumers

Start this phase separately for each method only after its reference is sufficiently specified and passes applicable controls. Do not require iAPF and MDPF to wait for one another.

Trace from every claim-bearing public/runner endpoint to the shared implementation. Port values, state conventions, sampling/weighting, source-required finite fit or gradient estimator, and all relevant boundary behavior. A function that exists but is not reached by the benchmark/training endpoint does not count. Run wiring and parity tests on the actual endpoints.

Use TensorFlow/TFP for BayesFilter arrays, numerical routines, data/training paths and decisions. NumPy remains only in explicit independent references or diagnostic tests; author PyTorch stays an isolated comparison source. Preserve batches through learned-filter updates. Use TensorFlow control flow for repeated numerical time/particle work, stable explicit input signatures and bounded shape-specialized kernels. Python loops for orchestration, logging or finite optimizer setup are not automatically defects; identify which loops cause per-particle host work or graph unrolling. No unapproved pfor or scalar fallback.

Validate in order: independent reference; TF CPU/FP64 diagnostic where suitable; TF GPU graph diagnostic; GPU/XLA. Graph/no-JIT arms are explicit debugging exceptions. Confirm memory growth before device initialization, and run device probes/jobs with trusted GPU access. Use separate processes for mutually exclusive device/precision configurations. Record dtype and TF32; do not import an unrelated lane's precision policy as an iAPF default.

Measure compatibility, numerical discrepancies, branch/resampling effects, compile cost, host/device peak memory, trace counts and steady-state runtime. Bound expected trace variants before repeated calls. Keep strings, files, provenance, host materialization and unsupported validation outside pure kernels. Use matched random inputs for deterministic components; differing RNG/resampling outcomes require distributional tests rather than a fake elementwise equality requirement. For discrete optimizer decisions, detect trajectory divergence and evaluate full downstream behavior.

Exit: actual consumer wiring and relevant value/gradient/distributional parity pass, XLA has a recorded performance/memory assessment, and downstream scientific checks remain valid. JIT success alone never promotes a filter. If XLA fails, localize the graph incompatibility/shape/loop problem; a debug exception does not silently become the default.

### R5 — Execute staged replication with uncertainty

If both finite-fit recipes remain valid, use a source-based order frozen before the pilot to select a representative for bounded follow-up, or retain both without ranking them. Do not select a winner by descriptive pilot performance or heuristic scores. After freezing the surviving specification, use fresh validation data and then untouched confirmatory streams. Do not tune on the saved failure cases and describe their reuse as independent confirmation. Existing MDPF test evaluations remain recorded final evaluations; they cannot become training/selection feedback.

For iAPF, expand one scope at a time: d=5, then 10,20,40,80; retain T=100 and original algorithm/comparator settings. First measure complete-run cost, then start the source 1,000-repetition cell only if its worst-case allowance fits. Preserve the distinction between source-reported and supplemental high-dimensional comparator cells. Count adaptation cost, incomplete replicates and failures. Compare source particle allocations separately from any equal-cost study.

For MDPF, first resolve the target objective branch, then run the predeclared source comparators (TG-MDPF, IRG-MDPF, DIS-PF) with their appropriate source objectives and validation rules. Use the same eligible data and initial information; do not force identical hyperparameters when the source specifies different methods. Reconcile true independent training seeds before calling repetitions independent. The final bearings target retains the paper's 11 training runs per relevant arm. A partial set is a pilot, not the completed comparison.

Predeclare the inference unit. Particle draws, time points and 5,000 evaluation trajectories do not substitute for independent training seeds. Use paired trajectory analysis within a fixed checkpoint and hierarchical/run-level inference for multi-training-run comparisons. Fix the estimand, contrasts, multiplicity handling and uncertainty method before confirmation. Report Monte Carlo error and warn when heavy likelihood-ratio tails make empirical intervals unreliable.

The exact Kalman likelihood is the iAPF correctness reference. The sample mean of likelihood ratios is not the mean log likelihood; finite-sample confidence-interval coverage of one is not proof of unbiasedness or equivalence. Support unbiasedness through the checked construction and use Monte Carlo behavior to challenge its implementation. Scientific variance/runtime contrasts require uncertainty at the proper unit. Original table numbers from unknown datasets, without source uncertainty, are descriptive references—not exact constants or a “within twice the SD” pass rule.

A source-specific equivalence margin must have an external accuracy rationale and be fixed before untouched evaluation. If it cannot be justified, do not claim exact numerical-table equivalence; report protocol completion and supported/inconclusive contrasts. All pilot results remain descriptive.

Exit: each prespecified cell is complete or explicitly incomplete, every failure is accounted for, and each claimed paper conclusion has the required source and uncertainty support. Report negative replication, insufficient power/budget, and implementation invalidity as different outcomes.

### R6 — Close the program with a reproducible verdict

Assemble one report with per-method source identity, tested mathematics/call chains, complete protocol cells, conditional heuristic tables, uncertainty, GPU parity and unresolved limitations. Emit a machine-readable decision plus human-readable explanation. Include decision and inference-status tables. One terminal skeptical review is sufficient unless it finds a new material flaw.

Completion is evidence-based: no blanket green status from process completion, finite loss, optimizer success, checkpoint existence, checksum audit, XLA compilation or one favorable seed. Distinguish completion of the scoped replication from readiness for an extension or a broader deployment claim.

## 4. Evidence contract and diagnostic roles

Main question: can the declared original methods' behavior be reproduced, and preserved by the intended local consumers? Mechanisms under test: iAPF finite backward fitting within the twisted filter; MDPF continuous-mixture resampling and IWSG training. Expected failures include density escape, source/implementation mismatches, invalid gradients, metric/checkpoint confusion, insufficient statistical power and compiler/resource failures.

| Role | Criterion or diagnostic | Consequence |
| --- | --- | --- |
| Primary implementation criterion | Source/mathematical target identified; actual consumer passes applicable identities and parity | Supports correctness only within checked scope |
| Primary replication criterion | Complete declared protocol with appropriate downstream oracle/comparator evidence and uncertainty | Supports the stated scoped replication conclusion |
| Promotion veto | Invalid target/weights, nonfinite output, failed parity, leaked holdout, unresolved objective identity, observed conditional heuristic underperformance (descriptive when uncertainty is weak), inadequate evidence for claimed ranking | Withhold the affected claim; explain uncertainty and candidate status |
| Continuation veto | Corrupt/unusable evidence, invalid comparator/data, ambiguous target for a dependent run, exhausted resources, actual external/irreversible boundary | Stop affected dependent run; unrelated valid work can continue |
| Repair trigger | Local harness error, identified code mismatch, numerical pathology in a particular candidate, trace/memory regression | Apply the corresponding bounded repair under unchanged scope/budget |
| Explanatory only | Fit gradient/curvature for finite-fit experiments, raw loss, shape error, ESS, optimizer return code, compile/runtime data | Cannot independently establish or refute replication |
| Falsification instrument | Constructed conditional heuristic comparisons, separate from model selection | Do not tune the complex method to these checks |

No claim of universal correctness, superiority, HMC readiness or production readiness follows from a pilot. A finite-output test does not certify a learned filter. A source-code reproduction can expose a disagreement with a paper claim; record both rather than modifying the implementation to conceal it.

## 5. Material defaults and assumptions

| Choice | Provenance / justification | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- |
| BFGS/zero explicit penalty for iAPF Part III | Thesis, to be reconciled with final paper | Mistaking thesis detail for final executable recipe | R0 line-by-line specification comparison | Source-supported family; recipe unresolved |
| Initializer/coordinates/amplitude/scaling/stopping | Original choices not recovered | Different finite basin/path despite same minimized objective | Known-Gaussian recovery and trajectory record before fresh pilot | Reconstruction hypotheses; max two recipes |
| Floor c and any damping/bounds | Positive floor is source-required; formula missing | Tail failure, domination by floor, altered objective | Source recovery; calibration of shape/tail and probability effects | No inherited power silently promoted |
| Absence of explicit regularization | Thesis rationale is local optimization, not general well-posedness | Density escape in reconstructed coordinates | Saved failures plus instrumented fresh pilot | Hypothesis requiring confirmation; not a safety justification by itself |
| Pilot dimensions/counts | Source smallest model; deliberately bounded engineering diagnostic | Favorable small sample mistaken for quality | Report every replicate/failure and prohibit ranking | Convenience diagnostic, not final protocol |
| Source neural architecture/optimizer/schedule | Original experiment comparator, not cross-task transfer | Inadequate target-specific training or wrong objective | Resolved config, gradient/scaling checks, heldout curves and actual downstream tests | Replication baseline; any new trained variant needs its own protocol |
| Source RNG and epsilon behavior | Inspected original implementation | False independence or overclaimed unbiased gradients | Effective-seed and expected-gradient checks | Source behavior retained/labeled; correction is a separate arm |
| Precision/XLA and batch/shape choices | Local backend requirements plus measured parity | Numerical branch changes, retracing or memory blowup | Bounded cross-backend consumer check | Candidate until validated |
| Uncertainty/equivalence thresholds | Must be derived for the actual claim before confirmation | Post-hoc pass rule or underpowered equivalence | Contract validation and cost/power calculation | Unfrozen decision items; block confirmation until specified |

Numerics-altering protections, including choosing not to use one, need a mathematical/source rationale or dedicated calibration. If a new floor/bound/damping candidate is warranted, evaluate non-harm on healthy cases and flagged behavior on pathological cases before promotion. Do not optimize these protections against final heuristic or test scores. Ordinary observability and fail-closed invalidity checks are adopted with focused no-fire tests.

## 6. Budget and execution scope

The current allowance can support a bounded repair tranche, not a promise of the full final suite. Proposed GPU ceilings total 36.56 hours, slightly below the recorded balance; recalculate exact remaining seconds before launching:

| Allocation | GPU wall-hour ceiling |
| --- | ---: |
| Saved MDPF evaluation, conditional baselines and Appendix B diagnostics | 3.00 |
| TF/TFP consumer parity, GPU/XLA diagnostics and related repair | 8.00 |
| Eligible iAPF expansion/replication cells | 6.00 |
| One complete, predeclared MDPF comparison or independent repeat | 12.00 |
| Unassigned contingency, retained until a justified repair/reallocation | 7.56 |
| Total | 36.56 |

These are maxima, not runtime forecasts or instructions to consume all hours. Allowance may be reallocated within the existing total after recording why; do not launch a complete trial without sufficient remaining allowance. CPU numerical work remains under 1,590.902954 seconds: 300 controls + 900 pilot, leaving 390.902954 for other CPU diagnostics/reserve. Source reading and ordinary file inspection are not stochastic experiments; numerical child work is recorded honestly.

At most 12 new numerical launch attempts in the first repair tranche, subject to the reconciled parent attempt limit; no more than two localized retries for the same diagnosed infrastructure fault without revising the diagnosis. Count failed starts and compilation attempts. Historical closed phases remain closed. All budgets are shared ceilings, not separate extra grants.

The previous complete MDPF training took 8.589 hours. Ten more runs at that observed duration would take approximately 85.89 hours for that arm alone; real early-stop costs vary. This excludes comparators, refinement, Appendix B and port work. Therefore the current balance cannot credibly be promised to complete the paper-scale repeated suite. After R3/R4 yield actual cost and validity evidence, present a concrete remaining-compute estimate. Ask for expanded compute only at that real boundary; never lower repeat counts and retain the same completion claim.

## 7. Commands, permissions and artifacts

Current working executable (read-only status):

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter/scripts/run_replication_master.py status

Its present run-next only knows the closed I2b diagnostic sequence. The proposed R0–R6 sequence is not yet implemented. R1 must add explicit phase validation/dispatch and list the exact commands for every implemented phase. No future command or placeholder should be presented as runnable until verified. Each launch manifest binds the command, source checkout/dirty dependency hashes, plan, target, seeds, data/checkpoint hashes, environment, hardware/memory policy, precision/XLA, budgets, timing and result paths.

Use versioned evidence under docs/plans/artifacts/iapf-mdpf-whole-repair-20260928-01/<phase>/attemptNN. Preserve prior output directories. Keep one short checkpoint with the current question, checked findings, resources and exact next action.

Reuse existing approved direct script/interpreter invocations; avoid shell substitutions and compound commands that defeat prefix matching. Test permission behavior once when a capability is actually needed. GPU probes/jobs require escalation; deliberate CPU diagnostics hide GPUs before framework import. Do not weaken sandbox protections or request broad arbitrary-shell approvals. Normal local repairs within the already authorized scope/budget need no new scientific approval. External messages, additional compute, broad environment changes and material method changes remain real boundaries. This proposal makes no permission/configuration changes.

## 8. Skeptical review of this proposal

| Risk examined | Repair incorporated | Remaining condition |
| --- | --- | --- |
| Wrong baseline or source | Final/accepted/thesis and MDPF paper/code reconciliation precede dependent work | R0 must finish before specifying the final fit |
| Proxy promoted to success | Separate stationarity, likelihood behavior, replication and GPU parity | Phase manifests enforce roles |
| Infinite optimizer search | At most two justified recipes; bounded source search and CPU pilot | Failure leads to diagnosis, not another blind grid |
| Unfair comparison | Same information/data; source objectives separate; oracle labeled; adaptation cost counted | Freeze actual contrasts and geometry before evaluation |
| Hidden defaults | Audit above; no silent floor/regularizer/precision transfer | Unresolved choices block corresponding confirmation |
| Stale context/version | Publisher PDF recovered; previous failures remain scoped | Full version comparison still open |
| Environment mismatch | Author references isolated; explicit TF/TFP GPU port and actual consumer tests | Port not claimed from R/PyTorch success |
| Invalid statistical unit | Training runs separated from trajectories/time points | Multi-seed suite needs more compute |
| Missing stop conditions | Validity/budget continuation vetoes separate from failed candidates | Runner must implement this distinction |
| Artifacts answering wrong question | Complete finite filtering pilot before more local optimization; conditional MDPF tests | A trace/checksum/finite loss never closes replication |

Verdict: suitable as a whole-program repair proposal after correcting the prior solver-first dependency. Not yet launch-ready as one monolithic campaign: R0 must settle source choices; R1 must implement phase commands; confirmatory margins/inference and full-suite resources must be fixed before their dependent runs. These are genuine scientific/implementation dependencies, not a request for repeated human approvals. No independent reviewer or agent was invoked.

Premortem: a successful run could reproduce the wrong objective, tune to test data, confuse trajectory count with training seeds, or hide failure in averages. A failed run could be a source-choice, harness or compiler problem rather than evidence against the algorithm. Source mapping, frozen partitions, conditional baselines, actual-consumer parity and explicit failure classification are the earliest discriminating checks. Terminal review must state the strongest alternative explanation and what evidence would overturn the verdict.

## 9. Activation and concrete first tranche

The owner's execution instruction activates this repair sequence. The first
execution specification freezes two explicit finite-BFGS reconstructions;
missing original settings no longer create an indefinite wait for author code.
The final-paper/theory reconciliation supports that limited reconstruction
claim, not exact recovery. R2 controls precede its 16-run maximum pilot. R3 is
independent. Full scientific completion and port readiness remain separate.
See the active checkpoint and machine status for current outcomes.
