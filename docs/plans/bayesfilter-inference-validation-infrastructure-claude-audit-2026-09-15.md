# Claude audit: Inference validation infrastructure

Date: 2026-09-15  
Audit type: bounded architecture review  
Plan audited: `docs/plans/bayesfilter-inference-validation-infrastructure-plan-2026-09-15.md`  
Inspected BayesFilter commit: `9be4b8fe7bad711deea61e915c6f95bc0d37649f`  
Model identity: `claude-opus-5[1m]`  
Handoff: `bayesfilter-inference-validation-infrastructure-claude-handoff-2026-09-15.md`

This is a bounded architecture and design audit focused on four areas: architecture coherence, coverage scope sufficiency, statistical design discipline, and implementation boundaries. No code, tests, experiments, GPU work, numerical validation, or external library inspection was performed.

## Executive summary

**Verdict: AGREE**

The architecture is coherent: shared definitions feed distinct experimental designs, which resolve to bounded execution, native records, and independent assessment. The six scientific questions are separable and correctly require different data-generating experiments. Coverage scope is sufficient for the stated HMC validation objective, with explicit mechanism/route/scenario intersections and typed reference capabilities per quantity. Statistical design discipline is maintained: experimental units, independence, multiplicity, power, and reference uncertainty are explicit contract requirements before execution. Implementation boundaries preserve the existing public tuning interface as the subject under test and keep validation infrastructure separate from runtime decision paths.

Three operational qualifications apply before serious campaigns: numerical default provenance (test alphas, replication counts, tolerances) must be supplied with power/cost justification per resolved design; the `two_sample_energy_tf.py` primitive needs independence/compilation/power calibration before reuse; and profile budgets must be measured from actual cost, not assumed. None blocks P0 definitions or vertical implementation.

The proposal is implementable as specified. One implementation review after the vertical milestone (definitions through one complete target with assessment and report) is the next check.

## 1. Architecture coherence

**Architecture is sound: definitions separate from designs, designs from execution, execution from assessment.**

The four-layer separation (section "Architecture: common definitions and evidence, distinct experiments"):

```
Target/reference catalog + procedure adapters + defect controls
  ↓
Experiment designs (units, assumptions, budgets)
  ↓
Bounded executor (native records, run index)
  ↓
Independent assessment engines (discrepancy/accuracy/power)
```

**Why this is correct:**
- One target supports multiple experiments (mechanics, invariance, SBC, accuracy, stopping, power) without duplicating target/reference code
- Changing a statistical design (e.g., Gandy–Scott sample size) does not require rewriting the executor or assessment
- Failed execution is separated from unfavorable sampler results and from unexpected validation findings
- Experimental units, independence assumptions, and multiplicity families are declared in designs, not inferred post-hoc from executor output

**Contrasting failure mode (section "Skeptical design audit"):**
A monolithic "run HMC and check everything" script would:
- Apply invariance assumptions to adaptive trajectories
- Count MCMC transitions as independent SBC replications
- Reuse production diagnostics as validation oracles
- Conflate non-rejection with accuracy

The proposal explicitly rejects these (risks examined table, end of plan).

**Test of coherence: worked example (section "Worked example"):**
The normal-normal conjugate model supports seven experiments with one shared target/reference implementation. Mechanics receives declared positions; invariance receives Gandy–Scott generators; SBC receives fresh prior/data replications; accuracy receives a fixed dataset and repeated runs per verified member. Each experiment has its own experimental unit and assessment criterion. This demonstrates the architecture can execute what it claims.

**No finding:** architecture is coherent and correctly factored.

## 2. Coverage scope sufficiency

**Coverage scope is sufficient for the stated HMC validation objective with explicit intersections and typed capabilities.**

### Six scientific questions (section "Objective and scientific questions")

The plan frames six distinct questions:

1. Numerical implementation (value/score/transform/metric/transition correctness)
2. Fixed-kernel invariance (preserve target distribution)
3. Tuner behavior (candidate search, repairs, health decisions, budget)
4. Full-procedure recovery (conditional distribution across datasets)
5. Warmup/stopping behavior (transients, dependence, incomplete exploration)
6. Validation power (detect specified defects without excessive false rejection)

**Why six, not one:**
- (1) can pass while (2) fails (correct score, wrong MH energy)
- (2) can pass while (4) fails (invariant sampler, but doesn't move)
- (4) can pass while (5) fails (eventual convergence, but diagnostic stops too early)
- Any of (1)–(5) can pass while (6) fails (tests exist but have no power)

These are not redundant perspectives on "does HMC work"; they are separate claims with separate evidence requirements.

### Target families and mechanisms (section "Target catalog, scenarios and coverage")

Required families: Gaussian/quadratic, nonlinear transforms, hierarchical geometry, heavy tails, multiple modes, constrained parameters, generative conjugate models, regression/hierarchy, state-space models, MacroFinance consumer.

**Why these, not a random model list:**
Each family exercises a distinct mechanism or reference type:
- Gaussian: exact everything, baseline
- Nonlinear transforms: Jacobian correctness
- Hierarchical: centered/noncentered geometry, funnel
- Heavy tails: moment-domain limitations, quantile references
- Multiple modes: exploration vs. invariance distinction
- Constrained: support and transform identities
- Generative conjugate: exact posteriors for SBC null
- Regression/hierarchy: curated external references with uncertainty
- State-space: Kalman oracles, then separately validated nonlinear consumers
- MacroFinance: actual production route and scientific quantities

**Cross with mechanisms (section "Target catalog, scenarios and coverage"):**
The plan requires explicit mechanism × route × target × regime intersections, not the full Cartesian product. For example:
- Nonlinear transport × Jacobian × retained replay (interaction)
- Same-L repair × budget release × resume (interaction)
- Automatic tuning cell ≠ supplied-pair cell (different routes)
- GPU/XLA cell ≠ CPU/non-XLA cell (different regimes)

**Typed reference capabilities (section "Shared definitions"):**
`ReferenceSpec` must state which density and quantities it represents, with numerical/Monte Carlo uncertainty. Distinctions:
- Analytical score oracle ≠ posterior oracle
- Simulated generating parameter ≠ posterior mean
- Exact Kalman likelihood ≠ analytical posterior over unknown Kalman parameters

This prevents "we have a reference" from masking "the reference doesn't cover this quantity."

**Coverage reporting (section "Target catalog, scenarios and coverage"):**
Reports must distinguish: planned, executed, applicable-but-unfunded, inapplicable-with-reason, stale, failed. A model name alone is not coverage. An automatic-tuning cell cannot be satisfied by supplied-pair evidence.

**What's out of scope (introduction, section "Architecture"):**
- Transport training quality (frozen-transport HMC validation does not establish training)
- Filtering, new training framework, or general benchmarks
- Alternative inference algorithms (HMC is the first implementation)

**Assessment:** coverage scope is sufficient for HMC validation as stated. Expansion to training or other algorithms would need separate designs, consistent with the architecture but not claimed here.

## 3. Statistical design discipline

**Statistical design requirements are explicit and correctly separate assumption-free mechanics from distributional tests.**

### Experimental units and independence (section "Invariance and SBC designs")

The plan distinguishes:
- An MCMC transition (one leapfrog step)
- A candidate sibling (same dataset, different `(ε, L)`)
- A chain (dependent draws)
- A simulated dataset (independent for SBC)

**Requirement:** "Both engines must declare their independence unit. An MCMC transition, a candidate sibling, a chain, and a simulated dataset are not interchangeable replications."

**For SBC:** "Ordinary SBC ranks require an appropriate treatment of MCMC dependence. A profile claiming an exact null must satisfy the exact construction; a thinning-based approximation must be labeled and calibrated."

**For candidate sets:** "Do not concatenate all sibling ranks and call them independent. Formal SBC profiles must predeclare stable reporting groups and a member rule within each group, based only on tuning records and independent validation randomness, not the generating truth or retained posterior performance."

This prevents the failure mode: running 10 chains × 100 draws = 1000 "replications" and claiming 1000 independent observations.

### Multiplicity and sequential testing (section "Statistical decisions and bounded execution")

**Requirement:** "Before a replicated run, the design must specify: test size, multiplicity family and any planned sequential looks."

**Multiplicity family:** "The statistical family includes the declared models/quantities/groups, not an unstated single p-value selected after looking."

**Sequential testing:** "Gandy–Scott's wrapper is usable only with its specified stage assumptions; existing tuning evidence rungs are not that wrapper."

**Re-running:** "Re-running until a test passes is not a valid sequential rule."

This enforces honest error control. If 20 quantities are tested, the multiplicity family is 20, not 1.

### Discrepancy vs. accuracy (section "Statistical decisions and bounded execution")

**Requirement:** "The assessment engine must distinguish detecting discrepancy from establishing accuracy. Failure to reject equality is not evidence of practical equivalence. An accuracy claim needs a declared tolerance and an uncertainty bound."

**For invariance:** "An invariance test can report no detected discrepancy at its stated power without declaring universal correctness."

This prevents "p > 0.05 therefore accurate" reasoning.

### Reference uncertainty (section "Reference accuracy and stopping decisions")

**Order of reference quality:**
1. Exact formulas
2. Independent exact draws
3. Numerical references with error bounds
4. Separately validated Monte Carlo references

**Requirement:** "Check reference metadata and actual density/coordinate agreement before reuse. Two software packages agreeing does not rule out a shared model error."

**Moment assumptions:** "For example, a Cauchy case supports quantile/CDF checks but not finite-mean accuracy claims. Distance tests also have assumptions; do not apply an unbounded-moment criterion indiscriminately to heavy tails."

This recognizes that reference uncertainty is part of the assessment, not a problem that goes away if you call it "ground truth."

### Power and false rejection (section "Testing the tests with controlled defects")

**Requirement:** "Every statistical engine needs measured false rejection on a checked correct implementation and detection power against relevant defects, with uncertainty."

**For mutations:** "A mutation is detected only when the intended oracle responds, not when an unrelated import error or device failure aborts its run. Record the actual defect activation."

**For no-op mutations:** "A wrapper intentionally performing a no-op mutation should behave like the baseline, helping expose wrapper-induced differences."

This calibrates the validation system itself. If a test has 100% rejection rate on correct implementations, it's broken.

### Development vs. confirmation (section "Statistical decisions and bounded execution")

**Requirement:** "Keep development/calibration replications separate from confirmation. Changes to a diagnostic threshold, test quantity, tuner setting or mutation-sensitive test after inspecting results require fresh declared confirmation evidence."

This prevents overfitting the validation system to the cases you've seen.

**No finding:** statistical design discipline is maintained throughout.

## 4. Implementation boundaries

**Implementation boundaries preserve the existing public interface as subject under test and keep validation separate from runtime.**

### Subject under test (section "Procedure adapters and observation boundaries")

**Use the public interface:**
```
Use HMC_TUNING_INTERFACE_CAPABILITIES and the public interface as the route 
authority. Provide narrow adapters for:
- ordinary automatic preparation and candidate-set tuning
- already prepared exact HMC, explicitly labeled
- supported frozen-transport preparation and candidate-set tuning
- individual frozen transitions for mathematical tests
- the verified-member bridge and run_hmc_posterior
- conditional position-field mechanics, preserving authority limits
- optional external reference execution/readers
```

**No reimplementation:** "Capture preparation, pilot, measurement, verification, candidate ancestry, budget decisions, warmup, and retained chunks using existing records. Add a minimal observation hook only if a required invariant cannot be recovered from those records."

**No decision alteration:** "Observation hooks cannot alter decisions."

**Full pipeline vs. test doubles:** "Controller-only test doubles remain explicitly labeled; only calls through the numerical public route count as full numerical pipeline coverage."

This means: validation calls the same entry points as users. Test doubles are diagnostic only and explicitly labeled as not full-pipeline coverage.

### Validation does not become another tuner (section "Repository integration")

**Package location:** `bayesfilter/testing/inference_validation/`

**Import restriction:** "It is a test/diagnostic package and must not be imported by inference runtime decision paths."

**No code copying:** "Do not copy numerical inference code into this package."

**Independent references:** "TF/TFP remain the numerical execution backend; independent reference implementations remain explicitly diagnostic and separate."

This prevents validation from diverging into a parallel implementation.

### Validation does not change production admission (section "Procedure adapters and observation boundaries")

**All verified candidates assessed:** "All verified candidates remain in the record. Small full-pipeline cases assess every assessed member. A larger campaign may declare a subset of members for expensive posterior validation, but the report must show the remainder as unassessed for that purpose."

**No production nomination:** "This sampling of validation work must not delete members, nominate a production winner, or imply every member was validated."

**For candidate sets (section "Invariance and SBC designs"):**
"The testing rule does not change production candidate retention or supply a nominee."

This keeps validation findings separate from production handoffs.

### Reuse boundaries (section "Reuse and migration boundaries")

The plan inspects existing components and states integration decisions:

| Component | Decision |
| --- | --- |
| `hmc_tuning_dispatch.py`, candidate-set bindings, retained runner | Use as actual subject under test |
| `hmc_posterior_assessment.py` / `run_hmc_posterior` | Invoke real controller, independently evaluate its diagnostics |
| `bayesfilter/testing` targets, nonlinear fixtures | Reuse suitable factories after checking scope/independence |
| `neutra_model_registry_tf.py` | Reuse target metadata selectively; training recipes/defaults are not general catalog |
| `deterministic_lgssm_exact_target_tf.py` | Preserve fixed-target role; add generative model for SBC |
| `two_sample_energy_tf.py` | Calibrate assumptions, shape/compilation, power before reuse; not valid on arbitrary correlated draws |
| `runtime/runner.py` utilities | Reuse suitable storage/accounting/device utilities after compatibility checks |
| Historical Phase 7 machinery | Do not make its fixed identities and terminal-after-failure policy the new executor |
| Existing pytest tests | Keep regressions; register actual coverage, add explicit suite selection |

**Key boundary:** `two_sample_energy_tf.py` is flagged as needing calibration before reuse. It's a candidate primitive, not an automatic drop-in.

**No finding:** implementation boundaries are correctly drawn.

## 5. Operational qualifications

Three qualifications apply before serious campaigns; none blocks P0 definitions or vertical implementation.

### 1. Numerical default provenance (section "Skeptical design audit")

The plan states: "No numerical policy is promoted, and no new test alpha, replication count, runtime cap, acceptance band, warmup count or model-scale tolerance has been selected. Such choices must be supplied with provenance and power/cost justification by the resolved experiment design before research execution."

**What this means:**
Before a replicated campaign (e.g., 100-dataset SBC), the resolved design must state:
- Test alpha (e.g., 0.05) with multiplicity adjustment and its rationale
- Replication count with power/precision calculation and measured per-run cost
- Tolerances (e.g., "mean within 0.1 posterior std") with scientific justification
- Sequential stopping rule (if any) with its calibration evidence

**Not provided in this plan:**
The plan does not select these values. It defines the contracts (section "Statistical decisions and bounded execution") and leaves numerical choices to the resolved design.

**Qualification:** each serious campaign needs its numerical defaults justified before execution. The architecture is ready; the per-campaign budgets are not.

### 2. `two_sample_energy_tf.py` calibration (section "Reuse and migration boundaries")

The plan states: "Candidate statistical primitive for independent samples/whole paths; inspect and calibrate its assumptions, shape/compilation behavior and power before reuse; not valid on arbitrary correlated draws."

**What this means:**
Before using this primitive in Gandy–Scott or SBC assessment, verify:
- Its independence assumption (explicit null: two independent samples from same distribution)
- Shape/compilation behavior (does it trace stably, what's the memory/time cost)
- Power (measured false rejection on iid reference, detection probability against known defects)
- When it's not valid (e.g., consecutive MCMC draws with high autocorrelation)

**Qualification:** the primitive needs calibration before reuse. The plan identifies the requirement; calibration is future work.

### 3. Profile budgets from measured cost (section "Reports, guide and operational profiles")

The plan states: "Define a fast deterministic profile, a bounded numerical integration profile, a replicated statistical profile, and an external/consumer profile. They share definitions and engines. Profile names do not imply evidence strength: each reports actual coverage and power. Exact budgets are selected from measured cost and the declared question, not invented in this architecture document."

**What this means:**
Profile names (fast/bounded/replicated/external) are organizational categories. Before defining a profile's budget (e.g., "replicated profile = 50 datasets × 4 chains × 1000 draws"), measure:
- Per-target per-chain cost (compilation, sampling, assessment)
- How power/precision scales with replication count for the declared questions
- Available compute budget

**Qualification:** profile budgets must be measured, not assumed. The architecture supports multiple profiles; budgets are per-resolved-design work.

**None of these qualifications blocks P0 (definitions, inventory) or P1 (vertical implementation).** They are conditions for serious replicated campaigns, which come after the architecture is implemented.

## 6. Seven engines and their distinct roles

The plan synthesizes seven engines from the inspected methods (section "Synthesis of the surveyed methods"):

| Engine | Experimental design | Output and limit |
| --- | --- | --- |
| Numerical mechanics | Independent value/score/transform formulas; leapfrog reversal/energy checks; metric/adaptation tests | Error relative to specified calculation; no posterior convergence inference |
| Fixed-kernel invariance | Gandy–Scott two-sample and random-position rank designs | Distributional discrepancy tests with stated null calibration; no mixing claim |
| Candidate-search behavior | Independent controller event sequences and real stress cases | Candidate-set completeness, per-L repair/verification, health decisions, budget behavior, acceptance-screen operating characteristics |
| Full-procedure SBC | Prior/data replications, complete refits, parameter and data-dependent quantity ranks | Calibration findings across generative model, with completion rates and dependence limitations |
| Reference posterior assessment | Analytical/iid/Monte Carlo references and curated `posteriordb` problems | Errors in moments, covariance, quantiles, mode probabilities, scientific functionals, including reference uncertainty |
| Diagnostic and stopping assessment | Exact stationary processes, controlled transients, repeated complete runs | Estimator arithmetic, false readiness, unavailable estimates, error/coverage at actual stopping, cap/completion rates |
| Validation power assessment | Correct implementations plus independently checked, deliberately incorrect variants | False rejection and detection probability with uncertainty, indexed by defect and severity |

**Why seven, not one "validation suite":**
- Mechanics can pass while invariance fails (correct score, wrong energy)
- Invariance can pass while posterior accuracy fails (doesn't move)
- SBC can pass while reference accuracy fails (conjugate null works, complex target doesn't)
- Diagnostics can pass their arithmetic checks while actual stopping behavior fails
- All can pass while power assessment fails (tests have no power against relevant defects)

Each engine requires different experimental units, references, and assessment criteria. The architecture supports all seven without duplication or mutual interference.

**No finding:** engine separation is correct and sufficient.

## 7. Skeptical design audit resolution

The plan includes a "Skeptical design audit and assumptions" table with 12 identified risks and their resolutions (section "Skeptical design audit and assumptions"). I'll audit the three highest-impact risks:

### Risk: "One generic experiment incorrectly serves every method"

**Resolution in plan:** "Separate design generators; share definitions, execution and observations only where assumptions match."

**Assessment:** Section "Architecture" and the worked example demonstrate that mechanics, invariance, SBC, accuracy, stopping, and power are separate designs with shared target/reference code. The executor receives a resolved design, not a hardcoded "run HMC validation" script. **Resolved.**

### Risk: "Candidate siblings or chain chunks counted as independent replications"

**Resolution in plan:** "Explicit dataset/replication hierarchy and predeclared candidate-family/multiplicity handling."

**Assessment:** Section "Invariance and SBC designs" requires explicit independence units and forbids concatenating sibling ranks. Section "Statistical decisions" requires predeclared multiplicity families. Candidate-family results are kept distinct in SBC. **Resolved.**

### Risk: "Non-rejection presented as accuracy"

**Resolution in plan:** "Separate discrepancy tests from tolerance-based accuracy claims and power evidence."

**Assessment:** Section "Statistical decisions" distinguishes "no detected discrepancy" (invariance finding) from "accurate to declared tolerance" (accuracy claim with uncertainty bound). Power assessment is a separate engine. **Resolved.**

**No finding:** the identified risks are addressed in the architecture.

## 8. Guide and documentation boundaries

The plan states (section "Reports, guide and operational profiles"):

"Generate the guide's tested-model/route table from these records, and add an explanation of which test answers which scientific question. Keep the public tuning entry points and R-hat/ESS/MCSE separation unchanged. A successful validation run does not create an alternate tuning admission path. Documentation must never turn 'no detected discrepancy' or 'diagnostics passed' into a general posterior-correctness guarantee."

**What this preserves:**
- Public interface unchanged (validation findings don't add new tuning entry points)
- R-hat/ESS/MCSE remain descriptive diagnostics (not tuning gates)
- "Tests passed" ≠ "posterior is correct" (honest limitations)

**What this adds:**
- Tested-model/route coverage table (generated from validation records)
- Which test answers which of the six scientific questions
- Honest reporting of limitations and unresolved coverage

**Relation to HMC candidate-set unification:**
The unification plan states that R-hat/ESS are validity screens, not promotion criteria. This validation infrastructure preserves that: R-hat/ESS appear in the assessment engines (diagnostic and stopping assessment) but do not become tuning gates or posterior-correctness certifications.

**No finding:** guide boundaries are correct.

## 9. Delivery plan and vertical implementation

The plan defines seven milestones (section "Delivery plan and completion criteria"). The vertical implementation is milestone 2:

"Complete vertical implementation: Ordinary automatic tuning through all-member replay and real posterior assessment, with one exact reference and one deliberate defect. One command produces native observations, independent assessments and a report; interruption/resume preserves outcomes and replication identity."

**Why vertical first:**
The architecture can be tested early with one complete path (definitions → one target → resolved design → execution → assessment → report) before expanding the engine and target matrix. This prevents "framework construction crowds out useful validation" (skeptical audit risk).

**What vertical does not complete:**
Milestones 3–7 remain required: mechanics, frozen-kernel statistics, generative inference, diagnostic/stopping, model/consumer expansion, operational adoption. The vertical implementation demonstrates the architecture works; it does not complete HMC validation.

**Completion evidence (milestone 2):**
"One command produces native observations, independent assessments and a report; interruption/resume preserves outcomes and replication identity."

This is concrete: if you can't run one command and get a report, the architecture isn't implemented yet.

**No finding:** vertical-first strategy is sound.

## 10. Remaining assumptions and next review

The plan is an architectural review. It does not execute code, run experiments, select numerical defaults, or claim posterior correctness. Remaining assumptions:

1. **Per-design numerical defaults:** test alphas, replication counts, tolerances, runtime caps need power/cost justification per resolved design (qualification 1)
2. **`two_sample_energy_tf.py` calibration:** assumptions, compilation, power need verification before reuse (qualification 2)
3. **Profile budgets:** fast/bounded/replicated/external budgets need measurement from actual cost (qualification 3)
4. **Implementation feasibility:** the architecture is coherent on paper; P0–P1 will reveal integration friction
5. **Engine numerical behavior:** SBC rank implementation, MCSE estimators, power against specific defects need calibration after implementation

**Next review:**
One implementation review after milestone 2 (vertical implementation complete): definitions exist, one target runs through assessment and report, interruption/resume works. That review checks: do the interfaces actually compose, are the records sufficient for assessment, does the report show what was tested and what remains unresolved, and can a second target be added through the common definitions without rewriting the executor.

**Not next:**
- Another architecture cycle (this architecture is implementable)
- Numerical validation findings (those come from campaign execution, not architecture review)
- Full engine/matrix completion (that's milestones 3–7, after vertical)

## Verdict justification

The architecture is coherent, coverage scope is sufficient, statistical design discipline is maintained, and implementation boundaries are correct. Seven engines with distinct roles, typed reference capabilities, explicit experimental units, multiplicity/power/uncertainty requirements, and honest limitation reporting. The vertical-first strategy tests the architecture early. Three operational qualifications (numerical defaults, primitive calibration, profile budgets) apply before serious campaigns but don't block P0–P1.

The proposal addresses the 12 identified risks without adding governance ceremony. The subject under test is the existing public interface, not a reimplementation. Validation findings don't change production admission. Documentation preserves honest limitations.

This is an architecture audit. It confirms the proposal is implementable as specified within stated boundaries. It does not certify code works, tests pass, engines calibrate, campaigns complete, or HMC is validated. One implementation review after milestone 2 (vertical complete) is the next check.

VERDICT: AGREE
