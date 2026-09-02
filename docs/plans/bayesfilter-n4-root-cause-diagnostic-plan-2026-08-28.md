# BayesFilter C2 n=4 Root-Cause Diagnostic Plan

**Date:** 2026-08-28  
**Status:** Executed through Stage 2; stopped on the predeclared QMC continuation veto  
**Audience:** Claude Code and the BayesFilter experiment owner  
**Owner request:** Correct the attempt05 diagnosis and trace the implementation
to define tests that can identify the root cause  
**Supersedes:** bayesfilter-n4-failure-fix-plan-2026-08-27.md  
**Corrected analysis:**
docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.pdf
**Terminal handoff:**
docs/benchmarks/artifacts/c2_n4_root_cause_20260828/attempt04/n4_root_cause_execution_handoff_for_claude_20260828.md

## Instructions to Claude

Treat the corrected PDF and this file as the active handoff. Do not execute the
retired 2026-08-27 fix plan and do not implement a branch-count or R_gram
normalization change. If asked to execute this plan, begin with Stage 0 only;
do not launch a research run until its shared-call-chain and no-forward-change
tests pass. Record any proposed deviation in the result note before running it,
and stop for owner direction if it changes the target, method, hardware class,
budget, or a numerical default.

## 0. Execution Record

Stages 0 through 2 were executed on 2026-08-28. Stage 0 passed nine focused
CPU-only mechanics and call-chain tests. Stage 1 ran the unchanged n=4,
degree-6, rank-6, 32-sweep configuration on the RTX 4080 SUPER with XLA and
verified TensorFlow memory growth; its one-scramble t=3 result nominated the
fit term. Stage 2 captured t=2, t=3, and t=4, added a fresh ten-replicate
800,000-particle per-step reference with covariance, and evaluated four
scrambles across the 8192/16384/32768 row ladder plus standard-normal and
Student-t mixture arms.

The frozen decomposition closed to at most `1.34e-15` nat. Only t=3 met the
predeclared `0.00125`-nat QMC half-width limit. At t=3 the fit term was
`+1.524930 +/- 0.001131` nat and the conservative state term was
`-0.012090 +/- 0.002958` nat, so both were material at that frozen step and the
fit term dominated. The t=2 half-width was `0.001281` and the t=4 half-width
was `0.400877`; those steps are formally unresolved. The generated Stage 2
artifact correctly set the campaign status to `STAGE_2_UNRESOLVED`, but its
component labels at t=2 and t=4 failed to apply the precision gate and are
superseded by the attempt04 terminal correction.

The plan therefore stopped under the named `christoffel_qmc_uncertainty`
continuation veto. Stage 3 and later stages were not executed. Attempt02
contains the original serious-run provenance but its result JSON is invalid
because the serializer emitted 17 bare non-finite explanatory bounds.
Attempt03 is the strict-JSON, no-recompute repair. Attempt04 records the
terminal interpretation and consumes no additional fitted-target budget.

## 1. Decision Before Execution

The attempt05 n=4, degree-6, rank-6, seed-42 cell is wrong relative to its
screened particle-filter comparator: corrected log evidence is +36.9423 rather
than -66.6980. The result rejects that cell. It does not establish a capacity
limit or a unique implementation defect.

Do not implement any branch-count, R_gram, or shift-denominator correction.
The code and mathematics show that:

- the discrete branch axis is a square-root component index and must use
  counting measure;
- summing its squared components reconstructs the adjacent target;
- the Gram normalizer is a coefficient contraction and has no sampled row
  denominator;
- the emitted RMS is branch-averaged training error, so multiplying RMS squared
  by branch count is meaningful only as a residual diagnostic; and
- changing tau to consume that diagnostic would be a Class-C numerical change,
  not a root-cause test.

The next work is diagnostic implementation and focused testing. No production
default, tau policy, floor, ridge, likelihood, or target may change in Stages
0-4.

## 2. Skeptical Plan Audit

### 2.1 Audit findings

The retired plan had material defects:

- It assumed the wrong branch normalization and proposed repairing a quantity
  that is already mathematically correct.
- It promoted same-row RMS into fit-quality evidence.
- It treated a Gaussian-oracle sweep budget as an SV default.
- It inferred a tau/sign correlation contradicted by the raw data.
- It proposed a likelihood clamp before identifying the first non-finite
  operation.
- It suggested LGSSM success would establish the SV issue as capacity or hint
  drift; LGSSM can reject a general bookkeeping bug but cannot choose among
  nonlinear fit, state, hint, and tail mechanisms.
- It did not preserve the full fitted TT or frozen pre-update state, so its
  proposed held-out test was not executable from current artifacts.

This revised plan removes those assumptions and begins with the smallest test
that directly measures generalization on an already fitted target.

### 2.2 Why this plan can answer the question

For each frozen step, it estimates the normalizer of the exact branch target
independently of the fit and compares it with the fitted TT Gram. This gives the
identity

    engine_error_t
      = [log Z_H,t - log Z_T,t]
      + [s_t + log Z_T,t - log Z_c,t-1 - PF_increment_t].

The first term is contemporaneous fit/normalization error. The second is error
already present in the approximate retained state, coordinate hint, or target
construction relative to the true filter. The terms cannot be separated from
the existing total, training RMS, or sign pattern alone.

**Audit verdict:** PASS after the revisions above. The proposed commands create
fresh artifacts, preserve attempt05, have bounded budgets and stop conditions,
and produce quantities that answer the stated question.

### 2.3 Execution erratum: reverse-triangle measures

Stage 1 exposed one reporting defect in the proposed residual bound. A
held-out residual is an empirical integral and cannot be divided by the exact
TT Gram to obtain a reverse-triangle bound: that mixes two measures. For each
row arm, compute

    Z_H,QMC = sum_i weight_i * sum_b H_i,b^2

and use `rho_H,QMC = residual_QMC / Z_H,QMC` in the reverse-triangle bound for
`log Z_H,QMC - log Z_T,QMC`. Compare the exact TT Gram separately through

    log Z_H - log Z_T
      = [log Z_H - log Z_H,QMC]
      + [log Z_H,QMC - log Z_T].

The first term detects exact-Gram mass not resolved by that integration arm;
the second is the fitted-versus-target discrepancy under the arm's empirical
measure. The primary fit term remains `log Z_H - log Z_T`. This erratum does
not change the target, fit, state, normalizer or likelihood.

## 3. Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | At the first divergent n=4 SV steps, is the evidence error produced by the current TT fit, inherited from the pre-update state/hint, or both? |
| Candidate under test | Existing degree-6, rank-6, 32-sweep, seed-42 C2 route with no numerical-policy change |
| Expected failure mode | Training RMS generalizes poorly; an earlier fit error is recursively amplified; or GH9 coordinates miss important n=4 mass |
| Primary diagnostic criterion | Frozen-state decomposition of each corrected engine-minus-PF increment into fit and state terms |
| Promotion criterion | None in Stages 0-5; these are diagnostic stages |
| Promotion veto | Any proposal to call a repaired candidate default-ready without scope-specific tuning and untouched valid-reference runs |
| Continuation veto | Broken snapshot identity, lane-forked target assembly, invalid comparator, non-finite diagnostic, unresolved decomposition identity, excessive QMC uncertainty, or exhausted budget |
| Repair trigger | A component is localized with uncertainty small enough to separate it from zero and the 0.0025-nat per-step bar |
| Explanatory diagnostics | Training/held-out RMS, row ESS, ALS condition, Gram spectrum, tau, shell residuals, u_old maximum, floor-dominance rate, sweep history |
| Must not be concluded | Minimal rank, route failure at n=4, default readiness, HMC readiness, statistical superiority, or low-degree cause |

The attempt05 candidate failed. That is not a continuation veto for this
diagnostic because the planned decomposition is specifically designed to
explain that failure.

## 4. Evidence Contract

### 4.1 Exact baseline and comparator

- Candidate forward program:
  bayesfilter/highdim/squared_tt_engine_gaussian_xla_tf.py,
  run_value_filter_branch_axis_gaussian_xla.
- Frozen candidate scope: n=4, degree=6, rank=6, observation seed 42, model
  seed 52, horizon 20, 8192 rows, 32 sweeps, ridge 1e-10, current tau/floor
  rules, float64, current XLA route.
- Candidate artifact:
  docs/benchmarks/artifacts/c2_completion_20260824/attempt05/cell_n4_d6_r6_s42_w32.json.
- Comparator:
  the screened 800000-particle, 10-replicate arm in reference_n4_s42.json.
- First diagnostic steps: t=2, t=3, and t=4, because the corrected discrepancy
  grows from 0.0420 to 1.5128 to 7.1481 nats.

### 4.2 Pass/fail and uncertainty rules

Stage 0 passes only when target closure, evaluator parity, call-chain wiring,
snapshot identity, and no-forward-change checks pass.

The one-scramble t=3 probe in Stage 1 is a gross-failure nomination test only.
It may trigger the formal decomposition but cannot establish a cause.

For the formal decomposition:

- use at least four independent scrambles per row law and step;
- compute uncertainty on log Z_T and on both decomposition terms;
- require the 95% half-width for log Z_T to be at most 0.00125 nat, or at most
  20% of the smaller material component needed for classification, whichever
  is stricter;
- if that criterion is not reached at the maximum row ladder, classify the
  step unresolved rather than choosing a mechanism;
- preserve PF per-step replicate uncertainty. The current artifact has means
  but not per-step SE, so a formal near-threshold classification requires a
  fresh versioned reference artifact that records replicate increments and
  their covariance.

For a gross multi-nat discrepancy, the existing screened PF mean can nominate
the responsible component while the per-step uncertainty artifact is being
completed. It cannot certify a near-bar result.

### 4.3 Vetoes

- production and diagnostic routes assemble different targets;
- capture changes the original forward value beyond 5e-12 absolute or
  5e-12 relative in float64;
- missing or mismatched model, observation, config, step, state, hint, or
  basis identity;
- training and held-out rows overlap or share the same scramble;
- held-out target recomputes the shift instead of reusing the frozen training
  shift;
- non-finite target, core, Gram, independent integral, or log ratio;
- invalid PF comparator or missing per-step uncertainty for a threshold-level
  conclusion;
- GPU run without trusted/escalated access, XLA, memory growth, device
  provenance, and a fresh versioned output directory;
- a Class-C arm silently changes the accepted forward path; or
- prior evidence is overwritten.

### 4.4 Explanatory-only quantities

Row ESS, tau, increment sign, training RMS, held-out RMS, maximum whitened
coordinate, shell residuals, floor-dominance fraction, ALS condition and sweep
history explain behavior. None alone establishes filter correctness or a rank
ordering.

## 5. Traced Implementation Boundary

The transition call chain currently does the following:

1. Factors the previous retained suffix Gram with a relative 1e-12 ridge.
2. Evaluates the old retained prefix and forms its square-root amplitudes.
3. Appends the defensive square-root component.
4. Computes log_g, sum_sq, log_f, and the physical-row shift.
5. Expands each physical row over branch codes and repeats its weight.
6. Fits a full 2n+1-axis TT by XLA-compiled ALS.
7. Integrates the branch and previous-state suffix into new_gram.
8. Returns only the current-state prefix, new_gram, Z_H, and summary scalars.
9. Discards the full fitted suffix cores and does not preserve the pre-update
   state, full target identity, or shift in the result artifact.

The current JSON therefore cannot support Claude's held-out test. Reconstructing
the target later in an independent benchmark implementation would violate the
call-chain rule and could manufacture a false diagnosis.

## 6. Stage 0: Shared Diagnostic Observability

### 6.1 Implementation shape

Refactor the existing transition kernel without changing its public default
behavior:

1. Extract one private pure-TensorFlow target assembler from the current lines
   95-165. It must return physical rows, expanded branch rows, scalar branch
   targets, repeated fit weights, shift, and finite/min/max summaries.
2. Call that helper from the existing production transition fit.
3. Add a diagnostic capture variant selected by a setup-static Python option.
   It executes the same numerical body but also returns all fitted core values
   at requested steps. Do not copy or simplify the fit.
   Each cached production/capture wrapper must use an explicit TensorSpec input
   signature derived from the configuration-time shapes, with the number of
   traces bounded and tested.
4. Preserve a host-side diagnostic snapshot containing:
   - model/config/observation/step identity;
   - pre-update prefix cores, suffix Gram, absolute floor mass, Z_c, coordinate
     map, and defensive nu;
   - joint hint mean and Cholesky, observation, training row seed, rows and
     weights;
   - frozen training shift, branch count, mixed-basis identity, all fitted
     cores, Z_H, raw/corrected increment, and finite summaries.
5. Evaluate the captured full TT with the repository TT contraction used by
   prefix_row_vectors over all axes; because the terminal rank is one, select
   its single output column.
6. Keep strings, JSON writing, hashes, manifests and validation outside XLA.

Do not add NumPy to the runtime call chain. TensorFlow/TFP remains the
diagnostic numerical backend; the existing NumPy PF fixture is an explicitly
diagnostic independent reference.

Suggested runtime edit:
bayesfilter/highdim/squared_tt_engine_gaussian_xla_tf.py.

Suggested diagnostic harness:
docs/benchmarks/diagnose_c2_n4_frozen_target_20260828.py.

Suggested tests:
tests/highdim/test_c2_gaussian_frozen_target_diagnostics.py.

### 6.2 Focused tests before a research run

1. Branch target energy closure at random and tail rows.
2. Expanded-row weight and RMS identity:
   counting residual equals branch count times emitted RMS squared.
3. Full-TT evaluator parity with the terminal ALS design prediction.
4. Direct full-TT Gram equals prefix/suffix contraction.
5. Frozen shift reuse: adding a known shift offset rescales T and Z_T by the
   derived factors.
6. Snapshot round trip preserves dtype, shapes, values and identity.
7. Diagnostic capture off versus on produces the same forward total and
   existing diagnostics.
8. Wiring test proves the production fit and held-out evaluator resolve to the
   shared assembler.
9. t=0 is explicitly classified as a separate n-axis fit and is not assigned
   fabricated transition-Gram fields.

Run focused tests deliberately on CPU:

    CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run \
      -n tftwogpu python -m pytest -q \
      tests/highdim/test_c2_gaussian_frozen_target_diagnostics.py

The CPU choice is for small mechanics tests only and must be stated in their
artifact. It is not production-target evidence.

## 7. Stage 1: Smallest Held-Out Probe

Run only through t=3 for the frozen attempt05 configuration. Capture the
already fitted t=3 TT and pre-update state, then evaluate it without refitting
on one disjoint 8192-row Christoffel scramble.

Record:

- training and held-out branch-counting residuals;
- empirical `Z_H,QMC`, `rho_H,QMC = residual / Z_H,QMC`, and the corresponding
  reverse-triangle log bound under the same row measure;
- the separate exact-Gram integration gap `log Z_H - log Z_H,QMC`;
- direct held-out Z_T and log Z_H - log Z_T;
- central and tail shell residual contributions;
- maximum and minimum log_g, log_f and shifted target;
- fraction of rows where the defensive branch has the largest squared
  amplitude; and
- exact design, initialization and held-out seeds.

Decision:

- gross held-out deterioration or a multi-nat fitted-normalizer discrepancy
  nominates contemporaneous t=3 fit/generalization failure and proceeds to the
  formal multi-scramble decomposition;
- a small t=3 fit term moves the formal test backward to t=2 because earlier
  fit error may already be inherited at t=3;
- a broken closure/parity identity stops the campaign and returns to Stage 0.

This probe must not be reported as proof from one scramble.

## 8. Stage 2: Formal Frozen-State Decomposition

### 8.1 Steps and row ladder

Capture states immediately before t=2, t=3 and t=4. For each step evaluate:

- four independent Christoffel scrambles at 8192 rows;
- if needed, 16384 and 32768 rows with the same four scramble identities;
- an independent standard-normal scrambled Sobol arm with equal reference
  weights; and
- a TensorFlow/TFP tail-enriched Student-t mixture importance arm.

The training shift stays frozen in every arm. No fit is repeated during
normalizer evaluation.

### 8.2 Quantities

For every arm and scramble:

    Z_T = sum_i weight_i * sum_b T_i,b^2
    Z_H = exact fitted TT Gram
    e_fit = log(Z_H) - log(Z_T)
    delta_state = shift + log(Z_T) - log(Z_c_previous)
    e_state = delta_state - PF_increment
    closure = engine_corrected_increment - PF_increment - e_fit - e_state

Require closure at float64/QMC precision. Report the mean, standard error,
95% interval and scramble-level values. Preserve PF per-step covariance rather
than pretending the three step errors are independent.

### 8.3 Interpretation table

| Fit term | State term | Interpretation | Next action |
|---|---|---|---|
| Material | Small | Contemporaneous fit/generalization dominates | Stage 3 frozen-target fitter study |
| Small | Material | Error is inherited or lies in hint/target state | Stage 4 state and hint validation |
| Material | Material | Recursive combination | Stage 3, then Stage 4 with repaired fit held fixed |
| Small | Small while observed error is large | Diagnostic identity, reporting convention or comparator is wrong | Stop; audit Stage 0 and PF convention |
| Uncertain | Any | QMC/PF evidence inadequate | Increase rows/replicates within budget or report unresolved |

Material means the relevant uncertainty interval excludes zero and its
magnitude exceeds the 0.0025-nat per-step scientific bar. This classification
is about the tested frozen step only; it is not a route-wide cause claim.

## 9. Heuristic Adversary and Conditional Checks

Construct these independent checks:

1. Fitted-TT direct Gram, the algebraic authority for Z_H.
2. Disjoint Christoffel integration, which checks generalization under the
   training row law.
3. Standard-normal scrambled integration, which targets the declared reference
   measure without Christoffel reweighting.
4. Tail-enriched importance integration, which tests the Student-t defensive
   regime.
5. Screened particle-filter per-step increments, which compare the approximate
   retained-state target with the true filtering computation.

Evaluate the residual conditionally on fixed infinity-norm shells
[0,2], (2,4], and (4,infinity), and separately at t=2, t=3 and t=4. Shell
statistics are explanatory veto diagnostics, never tuning targets. Losing to
any integration adversary is the headline for that step. Passing all weak
adversaries does not certify the filter.

## 10. Stage 3: Frozen-Target Fitter Study

Run this stage only when Stage 2 assigns material error to e_fit.

Hold fixed:

- frozen pre-update state and observation;
- hint, floor, shift and target;
- physical training rows for rank comparisons;
- held-out scrambles; and
- all non-fitter numerical settings.

Decouple the row-design seed from the core-initialization seed. Use at least
three initialization seeds for each compared arm. To remain inside the fit
budget, first run the sweep ladder 16, 32, 64 and 128 at rank 6. Then nominate
one sweep count from the frozen diagnostic validation results and compare ranks
3, 4 and 6 at that count on common physical designs. This is at most 18 unique
rank/sweep/initialization arms after accounting for the overlapping rank-6
arm; it is not a full 3-by-4 grid.

Add observability after each complete sweep:

- training objective;
- held-out counting residual;
- log Z_H - log Z_T;
- central/tail shell contributions;
- local solve condition;
- core norms and gauge-sensitive scale summaries; and
- final versus best held-out iterate.

Do not silently deploy best-iterate selection. Record it diagnostically first.
Paired uncertainty across common designs and initializations is required
before saying one fitter setting is better. If rank or sweeps are changed for a
later claim run, perform fresh scope-specific tuning on calibration data.

## 11. Stage 4: State and Hint Validation

Run this stage when e_state is material, or after a material e_fit term has
been repaired and the state term remains.

### 11.1 Backward state localization

Start at the earliest measured material state term and repeat the decomposition
one step earlier until:

- the first state discrepancy is localized;
- t=0 is reached; or
- uncertainty prevents classification.

Compare retained-state moments and normalization against an independent
high-particle diagnostic at each boundary. This is not optional if a later
held-out fit looks healthy: a correct fit to a wrong inherited state still
produces wrong evidence.

### 11.2 n=4 hint audit

On calibration observations disjoint from any later untouched claim set,
compare GH9, GH11 and GH15:

- filtered and predictive means;
- covariance and lag-one cross-covariance;
- coordinate-map condition and Jacobian;
- whitened mean/covariance residuals;
- central and tail coverage; and
- downstream frozen-target Z_T and fit difficulty.

Use a screened particle reference with per-step replicate uncertainty.
Hint metrics explain coordinate quality; they are not the primary filter
criterion. Freeze any selected hint order before a later untouched claim run.

## 12. Stage 5: Paired Mechanism Probes

Only after Stages 2-4 localize the responsible component, run the one paired
mechanism family nominated by that evidence on one frozen target. Do not run
the following as an automatic full matrix:

- artificial shift offsets -10, 0 and +10;
- previous-Gram ridge 0 versus 1e-12;
- current Student-t floor versus constant Gaussian-reference floor;
- a clearly labeled diagnostic-only floor-free target; and
- if justified by the fitted scale, a predeclared relative local-ridge ladder.

The shift arm tests solver homogeneity, not evidence bookkeeping.
defensive_nu=None is the Gaussian-reference floor arm, not a floor-free arm.
Ridge, floor and shift changes are Class-C numerical changes. They may be
evaluated under this plan but cannot become defaults from a primary-metric
improvement. Acceptance requires non-harm on healthy frozen targets and a
bounded, flagged response on the failing target.

## 13. Stage 6: Low-Degree Non-Finite Localization

Treat this as a separate diagnostic. Use a short foreground XLA run that
returns the first failing step and first failing stage. Record finite flags and
minima/maxima for:

1. transition log density;
2. observation log density;
3. reference conversion;
4. amplitude sum of squares and floor ratio;
5. log_f and shift;
6. shifted scalar branch targets;
7. each ALS core update and local system;
8. fitted cores;
9. prefix and suffix Gram eigenvalues;
10. Z_H and Z_c; and
11. final increment.

Do not dump large row tensors by default. Preserve the fail-closed increment
guard. Do not clamp the likelihood, change the density, or declare degrees
unsupported until the first non-finite operation is identified and reproduced
by a focused test.

## 14. Stage 7: Repair, Tuning and Untouched Evidence

This stage requires a localized mechanism and a reviewed repair. Then:

1. Add a focused regression that fails on the old path and passes on the
   repaired path.
2. Run healthy Gaussian and n=2 no-harm checks.
3. Perform target-specific tuning on calibration observations for every
   changed rank, sweep, row, hint, ridge, shift, tau or floor control.
4. Freeze the selected scope in a repository-issued tuning artifact.
5. Run untouched observations with valid references.
6. Rerun n=2 if any shared numerical or fitting policy changed.
7. Report hard vetoes first, statistical uncertainty second, and descriptive
   performance only after both.

This plan cannot promote a default. A separate result review must decide
whether the evidence justifies an optional repair, a new tuning scope, or a
default proposal.

## 15. Default and Assumption Audit

| Choice | Provenance | Status | Failure mode | Earliest diagnostic |
|---|---|---|---|---|
| 32 sweeps | Gaussian A3 oracle | transferred baseline | target-specific nonconvergence | held-out sweep trace |
| Rank at most 6 | attempt05 grid | hypothesis | inadequate capacity or worse optimization | common-design multi-init ladder |
| Ridge 1e-10 | runner constant | convenience choice | scale/gauge-dependent local bias | frozen shift/ridge pairs |
| Previous-Gram ridge 1e-12 | numerical guard | low-probability hypothesis | target perturbation or nonlinear amplification | zero-ridge pair |
| GH9 hint | attempt05 runner | unvalidated n=4 hypothesis | poor whitening and tail placement | n=4 GH ladder vs PF moments |
| 8192 rows | A3 row-law evidence | baseline | unseen tail mass | disjoint row ladder and tail arm |
| Student-t nu near 27.62 | domination criterion | safety hypothesis | pointwise tail amplification | floor pairs and shell residuals |
| Rank-dependent seed | runner formula | defective comparison design | confounds rank, rows and initialization | decoupled seeds |
| Training RMS | current output | explanatory only | false fit confidence | held-out residual and Z_T |
| PF per-step mean without SE | current artifact | incomplete formal comparator | overconfident step classification | versioned replicate increments |

No inherited setting becomes a default through this diagnostic.

## 16. Budget, Artifacts and Stop Conditions

### 16.1 Budget

- At most 6 GPU-hours on one repository-approved GPU.
- At most 4 CPU-hours for PF/reference assembly and reporting.
- At most 30 fitted-target attempts across Stages 1-5.
- One full frozen-target configuration counts as one fitted-target arm; its
  sequential time steps do not count separately.
- Independent integral evaluations without refitting are not fitted-target
  attempts but remain inside the wall-time budgets.

### 16.2 Fresh artifact root

    docs/benchmarks/artifacts/c2_n4_root_cause_20260828/attempt01/

Every retry uses attempt02, attempt03, and so on. Never overwrite attempt05 or
an earlier diagnostic attempt.

### 16.3 Serious-run manifest

Record:

- Git commit and dirty diff hash;
- exact command and conda environment;
- TensorFlow/TFP versions;
- GPU identity, XLA/TF32 status, dtype and verified memory growth;
- logical-device limit if used;
- model, observation, design, initialization and scramble seeds;
- all frozen config and state identities;
- wall time and peak TensorFlow allocator memory;
- input/output hashes and paths;
- this plan and the result note; and
- trust basis owner_designated_managed_session_visible_gpu_trusted when its
  requirements are met.

### 16.4 GPU command template

Run with trusted/escalated GPU access:

    TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=1 /home/chakwong/anaconda3/bin/conda run \
      -n tftwogpu python \
      docs/benchmarks/diagnose_c2_n4_frozen_target_20260828.py \
      --output-root \
      docs/benchmarks/artifacts/c2_n4_root_cause_20260828/attempt01 \
      --stage heldout-t3 --jit-compile

The harness must default to XLA JIT and fail closed if GPU memory growth is not
configured before device initialization.

### 16.5 Stop conditions

Stop and write a result note when:

- a component is localized and the next action is a Class-C/default change;
- the shared target/evaluator checks fail;
- PF or QMC uncertainty cannot meet the contract within budget;
- the frozen-state identity cannot be reproduced;
- an unplanned target, data, method, hardware or budget change is required; or
- the total budget is exhausted.

A failed candidate alone is not a stop condition. If the next planned stage is
designed to distinguish exactly that failure and no continuation veto fired,
continue.

## 17. Pre-Mortem

The campaign could pass while misleading us if the diagnostic target differs
from production, the held-out shift is recomputed, the same scramble leaks
between fit and validation, tail mass is missed by every row law, or PF
per-step uncertainty is omitted. Stage 0 wiring and closure tests, fixed-shift
identity, independent adversary rows, shell reporting and replicate PF
increments address these risks.

It could fail for engineering rather than scientific reasons through XLA
return-size pressure, snapshot serialization, GPU memory reservation, or an
overly large row ladder. Use capture only at requested steps, keep row-level
data out of default artifacts, verify memory growth first, and lower integral
chunk size without changing the row set or mathematical target.

The strongest alternative explanation after any apparent H1 result is that a
poor inherited state makes the frozen target unusually difficult; after any
H2 result, it is that an earlier H1 error created the state discrepancy. That
is why the decomposition must be applied backward rather than stopping at one
step.

## 18. Required Result Note

The terminal note must include:

- exact commands and manifest;
- Stage 0 check results;
- scramble-level Z_T, Z_H, e_fit, e_state and closure;
- PF and QMC uncertainty;
- shell/adversary table;
- first localized step and component, or unresolved status;
- hard vetoes;
- viable hypotheses and rejected hypotheses;
- whether any ranking is statistically supported;
- which differences are descriptive only;
- next justified action;
- what is not concluded; and
- a post-run red-team section naming the strongest alternative explanation
  and the evidence that would overturn the interpretation.

## 19. Plan Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Stage 0 shared observability | Shared target and no-forward-change tests | Passed | Focused CPU mechanics only | Retain capture/evaluator path | Root cause |
| Stage 1 nomination | Held-out t=3 gross check | Passed | One scramble | Stage 2 completed | Causality |
| Stage 2 decomposition | Fit/state identity with uncertainty | Failed continuation screen | t=2/t=4 QMC precision | Stop; exact t=3 coefficient-energy localization needs a new plan | Repair/default |
| Low-degree lane | First non-finite stage | Pending | No stage-local evidence yet | Focused foreground diagnostic | Underflow cause |
| Capacity/default status | Not eligible | Vetoed by unresolved cause/tuning | Confounded attempt05 | Untouched post-repair campaign | Minimal rank or readiness |

## 20. MathDevMCP Scope

The corrected algebra has been checked with the local MathDevMCP CLI for:

- finite branch-energy factorization;
- compensating shift and normalizer scaling;
- evidence telescoping;
- the reverse-triangle endpoint expression;
- the additive fit/state decomposition; and
- cancellation of the explicit per-step tau term.

Those symbolic checks do not replace the Stage 0 call-chain tests or the
empirical decomposition. MathDevMCP cannot establish that TensorFlow executes
the intended lane, that held-out rows cover the population, or that a
hypothesis is causal.
