# Local review of the surrogate-HMC master program

2026-09-18. Reviewed document: [Testing surrogate HMC across BayesFilter
models](surrogate-hmc-all-models-master-program-2026-09-18.md). Checkout:
`/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917`, branch
`ledh-graceful-failure-execution-20260917`, source HEAD
`8a4cffa2fd085f43ab5710c24d6e7b48770aea0d`.

**Verdict: the revised document is a coherent master research program for
testing surrogate HMC, suitable for Claude's independent review. It is not an
implemented all-model runner or evidence that surrogate HMC has passed.** This
is Codex's local skeptical review of its own draft, not independent review.
The findings below were repaired in the master itself. The remaining scientific
and engineering work is stated as execution prerequisites or bounded research
questions, not silently treated as completed.

## Revision and execution review — 2026-09-19

September 20 C01 audit: four-arm qualification, public tuning, retained archives
and closed-attempt recovery have executable evidence. All 36 GPU checks and
54 CPU regressions passed. Three retained comparisons are inconclusive; zero
tuning failed fresh R-hat. Exact force also failed some MCSE batch-correlation
checks, motivating independent calibration of that rule. This is not authority
to relax closed criteria. See the
[terminal result](surrogate-hmc-gaussian-controls-result-2026-09-20.md).
Other model consumers remain incomplete; C01 is no DCTR state-space evidence.

September 20 terminal-attempt audit: the default master could rerun LG1
attempt 4 after its two planned looks had closed. That would add unplanned
retained evidence from the same claim data. Before another master launch,
discover prior retained reports for the declared model and attempt, preserve
their paths and hashes, and carry their comparisons forward only as historical
results. They cannot authorize fresh tuning or sampling. A partial archived
look also prohibits restarting the same attempt from scratch; infrastructure
failures before any look remain repairable. When both LG1 branches are closed,
the all-stage command should omit their preparation jobs and continue the
independent model queue. Test this behavior without numerical sampling.

September 20 continuation audit before wiring LG2: the independent reference
and GPU consumer scripts have executable evidence, but the master still has
only LG1 jobs. Merely adding LG2 to its register would be a false coverage
claim. Add its real reference and qualification jobs with LG2 scope cards and
independent prerequisites; preserve failure outcomes and the distinction
between engineering qualification and inference. Test that a failed LG1 branch
does not cancel LG2, and that merged C results cannot overwrite B/R with
`NOT_RUN`. A last-job summary must follow launch order, not sorted JSON keys.
No new HMC tuning or research criterion is authorized by these harness edits.

Continuation update, September20: the public verified TensorFlow tuner is
implemented and GPU qualified. LG1 references, scoped controls, A/B tuning and
retained inference have completed. A and B/R are INCONCLUSIVE after both planned
looks, with healthy chains. The finite endpoint and D quadrature qualified;
D/R fails one margin, which did not block C/D. Separate C tuning and retained
sampling completed. C/D and C/R remain INCONCLUSIVE at the planned terminal
limit; every retained chunk passed health checks. LG2's independent posterior
reference and actual full-consumer GPU qualification have also passed. Its
control calibration selected multiplier 0.01, while its measured B-only
inference forecast exceeds the initial allocation. Retention is deferred.
All 43 terminal-attempt/orchestration regressions passed; an actual CPU-only
master run preserved closed LG1 comparisons without launching a numerical
worker (`surrogate-hmc-all-models-20260920-master-closed-lg1-01`).
The [retained result](surrogate-hmc-lg1-inference-result-2026-09-20.md) and
[finite comparisons result](surrogate-hmc-lg1-finite-result-2026-09-20.md)
give current evidence. The following P0/P1 absence-of-artifact statements are
historical. The central objective remains testing surrogate HMC.

The September20 code review found and repaired four concrete issues: optional
exact-force A could incorrectly block B retention; scope cards confused data
replicate0 with inference attempt4; endpoint rounding flags were dropped before
archiving; and C required independent tuning plus a joint comparison-family
adjustment. Focused tests exercise these cases. The latest47-test core suite,
25-test master/equivalence/reference suite, and actual Gaussian GPU flag/archive
qualification passed. The finite endpoint's original physical value, canonical
force call chain, replay and row-independence qualification remain unchanged.
No new independent Claude review has been performed.

A subsequent call-chain review found that the master's `all` loop could stop
after a B-branch failure and thereby skip independent C work. It now executes
explicit prerequisites: a failed shared target/controls check blocks its
descendants, while failed B tuning or an unaffordable B retained forecast does
not cancel C. Every job outcome persists, partial execution remains visible,
and an interruption stops all further launches. Standalone forecast failures
also write a result and accounting record. The existing21 master tests passed;
all9 new dependency and master-endpoint regression tests passed. The initial
new-suite run had one wrong expected exit code in the test, corrected to the
existing failure exit-code contract. No scientific worker or qualified
endpoint/tuner source changed in this orchestration repair.

C's measured retained forecast fits the original LG1 GPU ceiling after the
master's explicit45-minute P0/P1 CPU reallocation. The15 CPU-hour whole-campaign
limit, target, margins, looks and attempt count remain unchanged. Launch root
`artifacts/surrogate-hmc-all-models-20260920-lg1-C-sample-01` preserves the actual
command, forecast and executed sources. C's terminal posterior verdict is still
pending; all-model execution and heuristic comparisons remain incomplete.

Remaining limitations are explicit: empirical quadrature allowances, assumed
stationary-chain CLT, two short retained looks, one inference dataset, unresolved
heuristic comparisons and missing consumers for other model scopes. These are
limits on inference and coverage, not evidence that the force must be exact.

The Opus reply correctly withdraws F1. I incorporated independent frozen
force/endpoint bindings, explicit per-arm precision, proper synthetic-prior
interpretation, and the owner's execution authorization into the master. The
goal and the four posterior comparisons are unchanged. No equality of the two
bindings is required. Each binding must be replayable and internally consistent.

I then implemented and reviewed the P0/P1 driver, full LG1 diagnostic consumer,
independent TensorFlow Kalman endpoint, and actual GPU probe. Checked the call
chain from the CLI through the consumer and linear fixture to
`canonical_batch_fused_value_score`, and from the typed imperfect-field binding
through the public TensorFlow tuner. The force uses the full September 17
canonical route and explicit position-field semantics. The prior, stationary
initialization, transition timing, chart/Jacobian and analytical reference
derivative have independent dense-law TensorFlow checks.

**Readiness verdict: P0/P1 mechanics are partly implemented and tested;
retained inference remains blocked by the missing admitted TensorFlow tuner.**
The earlier generic registry check established binding support, not eligibility
of the TensorFlow tuning configuration. The interface explicitly distinguishes
these. The full-consumer GPU witness completed diagnostic tuning with finite
states and health passing, then received the expected retained-runner error:
`TensorFlow diagnostic tuning is not admission-capable and cannot build a retained runner`.
The older admitted path uses NumPy numerical operations. Neither is an eligible
substitute for the required TensorFlow public tuning artifact. This is a shared
implementation continuation veto, not a failed DCTR candidate.

| Review finding | Repair or disposition | Executable evidence |
|---|---|---|
| New launches initially reset spent cost | Carry prior incremental charges forward; reject invalid or over-budget accounting | Budget regression; second GPU run retains first-run and CPU-test charges |
| Raw invalid force could fail reporting before deterministic fallback | Preserve raw status/nonfinite diagnostics and check the shared effective field; no new force repair introduced | Kernel invalid-force regression; GPU effective-force check |
| Process creation/error paths could leave a RUNNING record | Finalize accounting and logs on launch error, timeout and interruption | Launch-error and failed/timeout regressions |
| Declaring the whole LG family ready would overstate LG1 evidence | Per-variant status; unimplemented scopes remain visible | Register regression; machine-readable coverage remains false |
| CPU test command included an implicit-pfor Jacobian | Exclude that test; no pfor approval inferred | 47 focused tests pass, 2 excluded |
| Generic tuner capability was insufficient readiness evidence | Record actual TF configuration and retained-handoff refusal; no flag changes | Full N1024/T10 GPU consumer through public tuner |

The [execution result](surrogate-hmc-all-models-execution-result-2026-09-19.md)
records 47 passing focused CPU tests and 13 passing GPU mechanics checks,
including scalar/batch equality at the checked point. It also retains the first
failed dry-run bootstrap test and its repair. No posterior draw, convergence
claim, candidate ranking, qualified finite-DCTR reference, scoped control
selection, all-model coverage closure or admission artifact has been produced.
The GPU tests cover the recorded positions and settings; they do not qualify
all fallback/support branches or floating-point tolerances for inference.

The September 18 design audit below remains useful history. Its statements
about missing consumers and unexecuted checks describe that earlier stage and
are superseded by the bounded implementation evidence above.

## Question reviewed

Can surrogate HMC driven by the full DCTR algorithm's imperfect force provide
reliable parameter inference across the repository models, and at what cost?
The intended test concerns posterior inference. Disagreement between the force
and a reference gradient is part of the premise and is not, by itself, a reason
to abandon the experiment. The program must still establish that it evaluates
its declared endpoint density and executes an eligible proposal/acceptance map.

I checked goal alignment, probability targets, the proposal argument, failure
semantics, model coverage, source/API assumptions, reference independence,
uncertainty and stopping rules, cheap comparators, tuning boundaries, cost and
attempt accounting, and the next executable task. No model inference, training,
GPU probe or new numerical test was run for this document review.

## Material findings and repairs

| Finding in the inherited approach or draft | Why it matters | Repair now in the master |
|---|---|---|
| Filtering or score agreement could prevent the intended inference test. | That substitutes an accuracy requirement on the force for the research question about an imperfect force. | Sections 1, 4 and 10 distinguish explanatory force/filter errors, promotion vetoes, dependency-specific continuation vetoes and planned repairs. B runs when its own reference/consumer is ready. |
| Model posterior, finite-DCTR posterior and proposal force could be conflated. | Correct sampling of a biased finite likelihood is not necessarily correct model inference. | Section 3 defines all three, the chart and prior terms, the potential-gradient sign, frozen randomness and separate value/force controls. An approximate reference retains its own density notation. |
| Two individually acceptable errors could be mistaken for acceptable total error. | C versus D and D versus R can each differ by the allowed margin while C versus R differs by twice that margin; errors can also cancel. | Section 4 requires four separately reported comparisons: B/R, C/D, D/R and C/R. Section 7 requires common physical functions and direct uncertainty calculations. |
| Expensive finite-DCTR quadrature could block the first mechanism test. | An available Kalman endpoint can already test whether the imperfect force works. | Sections 4 and 10.1 schedule reference-endpoint B first. C/D remain obligations for a complete practical result, with their own measured allocation. |
| A failed conventional-HMC comparator could become a global veto. | A hard target can defeat that comparator while another reference and surrogate sampler remain valid. | Section 4 makes the veto follow the failed dependency. A matching-score HMC success is not a universal prerequisite. |
| Numerical endpoint failures could be relabeled as support boundaries. | Rejecting those points silently changes the density; a valid reference endpoint can coexist with a failed force. | Section 3.2 separates endpoint, force, support and arithmetic status. Sections 6.3 and 9 require typed status propagation and failure-injection tests. |
| Excluded posterior mass was insufficient to justify moment accuracy. | A small missing tail can still change an unbounded mean or second moment substantially. | Section 3.2 derives the total-variation/event bound and the bounded-function bound, and requires separate tail-moment bounds under the particular affected endpoint. |
| The force wrapper's default semantics claim a scalar-potential gradient. | Using that default would misstate this experiment and could contaminate admission metadata. | Section 6.3 explicitly requires `DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS`; section 9 requires executable checks that it cannot issue exact-score status. |
| Batch permutation checks alone would miss symmetric cross-chain coupling. | Independent per-chain MH is invalid if one chain's field or potential depends on other chain states. | Section 9 also requires row independence under appending/replacing other chains, along with shape, sign, dtype, coordinate and source-route checks. |
| The inventory omitted concrete families and controls. | A restricted LG/SV/PP ladder cannot answer the all-model request. | Section 5 adds range–bearing, structural quadratic completion, banana and mixture controls; retains distinct SIR/SV/structural laws and higher-parameter obligations; requires a source-mapped closure register. |
| Predictive comparisons could combine parameter error with a different filtering approximation. | Candidate and reference would then be evaluating different functions of the parameters. | Section 7 uses the same qualified model conditional predictive probability for both posterior expectations and includes its uncertainty. LG1 gives an explicit Kalman predictive event. |
| The first LG1 numerical setup and later production scope could be confused. | An FP32 field inside an FP64 integrator is a distinct computation; its result does not establish all-FP32 sampler behavior. | Sections 9 and 10.1 specify FP64 HMC/Kalman reference with explicit full-DCTR FP32/TF32/GPU/XLA casts, proper prior/chart and stationary initial law. Separate precision scopes cannot share admission. |
| A bounded campaign could be described as sufficient for all models without timings. | Tuning, reference qualification and replication may exhaust the budget before useful full-scale evidence exists. | Section 12 treats 63 GPU-hours / 15 CPU-worker-hours as proposed ceilings, requires measured complete-cost forecasts and fair family scheduling, and permits an explicit incomplete outcome. |
| Used attempts could disappear on changing the plan. | Restarting counters would hide expenditure and effectively expand the campaign. | The old three LG1 filtering attempts remain used; two additional inference attempts are proposed explicitly. No experiment is launched by this writing task. |

## Mathematical and implementation assessment

The section 3.1 argument is correct under its stated ideal-arithmetic
assumptions: a finite deterministic position-only force gives momentum kicks
that are translations; drifts are translations in position. Their symmetric
composition with momentum reversal is a volume-preserving involution.
Metropolis correction using the declared potential **and kinetic energy**
therefore preserves the corresponding extended target. This derivation does
not require the field to be that potential's gradient. It also does not prove
irreducibility, floating-point reversibility, adequate mixing or finite-run
accuracy. The master now maintains these distinctions.

The numerical-failure policy remains a real implementation obligation. A finite
deterministic force extension can satisfy the kick argument; an arbitrary early
return, overflow sanitization or rejection branch does not acquire a proof by
being called a safeguard. Reverse replay and executed-branch checks are required.
The absence of observed failures is empirical qualification, not a proof about
every possible state. The exact support and target cannot change without a
separate comparison.

The following source checks support the design, with no claim of completed
consumer integration:

| Inspected source | Checked fact and its limit |
|---|---|
| [HMC tuning interface](../reference/hmc-tuning-interface.md), and [capability registry](../../bayesfilter/inference/tuning_contract.py), definition at line 850 | A typed endpoint-corrected position-field binding is supported. This does not establish TF admission: that configuration is diagnostic-only, while the legacy authority uses NumPy. A runner/stage helper cannot issue the canonical tuning artifact, and a fixed-transport tuner is not a general-force tuner. |
| [neural_force_hmc.py](../../bayesfilter/inference/neural_force_hmc.py), semantics/status definitions near lines 30–66 and force binding near line 104 | The generic field semantic exists, but the force wrapper defaults to a potential gradient. The bare-value endpoint default interprets positive infinity as outside support. The planned consumer must supply explicit semantics and status. |
| Same kernel, `neural_force_proposal` at line 467 and `neural_force_hmc_transition` at line 603 | The shared implementation contains fixed-step kicks/drifts, momentum reversal, per-chain full-energy acceptance and cached endpoint handling. Its actual DCTR call chain and failure branches remain to be qualified by executable tests. |
| [Kernel tests](../../tests/test_neural_force_hmc.py), typed binding at line 129, involution at 218, full energy at 250, omitted-kinetic counterexample at 277, batch checks at 409 | Relevant mechanics tests exist. The recovery ran five focused CPU checks. That historical result does not cover a new full-DCTR model consumer or GPU/XLA inference. No new test run is claimed here. |
| [LG fixtures](../../bayesfilter/highdim/ledh_program_fixtures_tf.py), fixture definitions at line 28 and bounded target/route factory below | LG1 is a one-parameter problem with its actual prior/chart and initialization obligations. The old five-coordinate example is not an eligible replacement. Consumer readiness and scoped control selection are still open. |
| [Common model suite](../../experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py), [quadratic structural law](../../experiments/dpf_implementation/tf_tfp/fixtures/structural_ar1_quadratic_tf.py), [generic density controls](../benchmarks/run_neutra_generic_five_stage_model_2026_08_15.py), and [NeuTra model registry](../../bayesfilter/testing/neutra_model_registry_tf.py) | Concrete additional laws and control targets exist. Source definitions can identify coverage; historical inadmissible LEDH results cannot supply evidence for a new run. |

**Call-chain verdict: not checked end to end for the proposed all-model
consumer, because that consumer has not been implemented.** Existing component
capabilities are supporting evidence for feasibility, not proof that a future
model endpoint calls them correctly. Section 9 names the required wiring and
parity artifacts rather than claiming this audit supplied them.

## Statistical and feasibility assessment

The primary decisions concern prespecified posterior functions with reference
error and autocorrelation-aware Monte Carlo uncertainty. R-hat, ESS, movement,
mode checks and numerical validity can veto interpretation, but acceptance,
force loss, runtime or filtering moments cannot confer posterior correctness.
The two-look adjustment handles the declared opportunities to claim equivalence
only under the stated uncertainty assumptions. Rare events, missed modes,
uncertain references and unaffordable precision remain inconclusive.

The 0.10 standardized-moment, 0.02 probability and 0.03 predictive tolerances
are proposed scientific choices, not validated defaults. Each scope must
translate them into physical consequences before retained results are opened.
Agreement on this finite summary vector is not equality of joint distributions.
The required predictive, joint and mode checks constrain stronger claims, and
the planned small dataset count cannot establish frequentist interval coverage.

The constructed heuristic set tests conditional performance in the situations
each model is meant to handle. An uncertainty-supported loss vetoes a practical
advantage claim; it does not erase a separately supported posterior-sampling
result. Timing rankings require a declared uncertainty analysis and comparable
tuning access. One-run ESS/time values remain descriptive.

The budget arithmetic fits inside the conservatively reconciled remainder, but
all-model feasibility is **not established**. Some references and analytical
callbacks do not yet exist. A proper prior for a filtering-only fixture creates
a declared new inference task, and a restricted noise-scale task cannot complete
an existing full-parameter model obligation. Model cards, source-law mapping,
measured cost and actual commands must be completed before their dependent runs.

## Residual questions for Claude and execution

1. Does any required model or distinct observation/initialization law remain
   missing from the minimum register? P0 must still close that question by a
   source-based census; “all models” has not been certified by this review.
2. Are the proposed error tolerances meaningful for every physical model
   quantity? Freeze justified scope-specific choices before claim runs.
3. Can the full DCTR consumer and offline artifact issuer represent an imperfect
   proposal force honestly while preserving all canonical algorithm capabilities?
   Implement and test this before inference; no route label is sufficient.
4. Can each larger model obtain a qualified reference and enough sampling
   precision within its allocation? Measure this, preserve unavailable cases,
   and do not infer full-scale validity from a small model.
5. Does the executed GPU/XLA map preserve the stated status, row-independence
   and numerical-reversal contracts, including fallback branches? The document
   proof and prior CPU component tests do not answer this.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Master ready for independent design review | Research question, target distinctions, comparisons and outcome rules specified | No unresolved contradiction found in the revised design; consumer/reference/budget readiness remains conditional | Model closure, physical tolerances and implementation feasibility | Claude reviews the master; execution subsequently begins with P0/P1 and the specified LG1 task | Runner correctness, posterior success, all-model completion or production readiness |

Strongest alternative explanation of a future failure: the adapter, reference,
tuning or budget fails before the force mechanism is genuinely tested. Strongest
alternative explanation of apparent success: biased references, missed modes or
cancelling finite-likelihood and sampling errors. The four comparisons and
independent checks are designed to expose those explanations; they remain work
to execute, not results of this review.

Documentation consistency and source-symbol checks are recorded in
[validation.json](artifacts/surrogate-hmc-master-20260918-01/validation.json).
They verify document mechanics only. The [Claude handoff](surrogate-hmc-all-models-claude-handoff-2026-09-18.md)
keeps the independent review focused on the actual research goal.
