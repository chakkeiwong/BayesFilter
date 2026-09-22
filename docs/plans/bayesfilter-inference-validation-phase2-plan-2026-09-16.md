# Inference validation continuation: prerequisites and statistical evidence

Active continuation: [HMC repair master program](bayesfilter-hmc-repair-master-program-2026-09-16.md).
The records below preserve the preceding campaign and its historical budgets.

Date: 2026-09-16. Status: active continuation of the
[infrastructure plan](bayesfilter-inference-validation-infrastructure-plan-2026-09-15.md).
The [first execution result](bayesfilter-inference-validation-infrastructure-execution-result-2026-09-16.md)
is completed development work, not completion of the infrastructure program.
The owner's request to refresh and execute covers the work within the existing
scope and allowance. This document does not increase that allowance.

Update: the owner subsequently allocated **24 CPU hours and 24 GPU hours**.
The [funded campaign](bayesfilter-inference-validation-24h-campaign-2026-09-16.md)
completed within both allowances and governs that implementation and numerical
evidence. The unfunded status below
describes the earlier checkpoint, not the current authority to proceed.
The work-package table reflects that campaign; the original budget and terminal
review below remain historical. Current findings and remaining repairs are in
the [funded result](bayesfilter-inference-validation-24h-result-2026-09-16.md).

## What changed in the plan

The original plan described the right architecture but did not turn its final
list of missing evidence into a dependency-ordered continuation. Its fixed-size
diagnostic fixtures do not implement a stopped-versus-fixed comparison. Its
full-procedure SBC has incomplete datasets, so choosing a larger dataset count
alone will not repair it. The current simplex posterior boundary requires equal
active and reported dimensions; marking Dirichlet unavailable was honest
accounting, not an implemented adapter.

The immediate executable step is an offline readiness check: reconcile all
attempt costs and incomplete full-SBC datasets from the saved run indexes,
verify their recorded result and tensor hashes, resolve existing suite inputs,
and document what remains necessary before scheduling statistical work. A small
reusable script and focused regressions make that check repeatable. It must not
import TensorFlow, initialize a GPU, rerun tuning, or modify prior evidence.

## Intent and evidence contract

The research question remains whether the complete public inference procedure
recovers its declared posterior and whether its stopping decisions have useful
error and coverage behavior. The next engineering question is whether the
prerequisites, independent units and resources for testing that claim are ready.

| Role | Continuation requirement |
| --- | --- |
| Baseline | Existing public tuners and retained bridge; independent analytical/generative references; fixed-length comparator must be defined before stopping comparisons |
| Engineering pass | Saved costs, failures, availability and hashes reproduce; a missing or capped fit cannot count as a complete SBC dataset |
| Statistical criterion | Predeclared discrepancy/power tests for SBC; uncertainty on availability, coverage and defect detection; quantity-specific reference error for posterior assessment |
| Promotion veto | Incomplete required fits, reference disagreement, invalid numerical execution, uncalibrated test assumptions or insufficient uncertainty evidence |
| Continuation veto | Broken target/reference, corrupted observations, invalid device launch, or insufficient remaining budget for the next declared job |
| Repair trigger | Missing candidate group, missing retained output, worker timeout, unavailable diagnostic or excessive construction/compilation cost |
| Explanatory | Acceptance, local R-hat, ESS, MCSE and timing except where explicitly declared as posterior criteria or cost accounting |
| Unsupported conclusions | Adequate burn-in from local diagnostics; universal correctness; superiority; external parity; default readiness; statistical power inferred from pytest counts |

Every verified candidate remains retained. R-hat, ESS and MCSE do not become
tuning requirements. Replicated fits are independent conditional on a dataset;
the dataset is the independent SBC rank unit. A retry with the same seed is not
a new statistical replication. No success-only replacement of failed datasets
is permitted.

## Budget and executable work

The inherited campaign ceilings are 1,800 CPU and 3,600 GPU worker-seconds. The
terminal ledger records 1,673.6557522671064 CPU and 3,107.1110839610337 GPU seconds,
leaving 126.34424773289356 and 492.8889160389663 seconds respectively. These are
local engineering ceilings, not statistically justified sample sizes.

The continuation's immediate scope is routine offline implementation, planning
and focused tests, excluded from numerical-worker allowances in the original
execution specification. It schedules zero new campaign workers. The last GPU
full-SBC attempt used 800.98 seconds without completing its two-dataset design;
that observation does not support fitting a larger replication campaign into
the remaining 492.89 seconds. Earlier shorter runs had missing outputs and
cannot supply a complete-fit cost estimate. No adequate confirmation allocation
or power calculation has yet been established.

Keep the remaining allowance unspent. Statistical campaigns below remain
unfunded until measured complete-fit costs, power requirements and a sufficient
bounded allocation are recorded. Routine implementation of their prerequisites
does not itself require a new research budget. No new package installation,
external message, transport training or GPU run is part of this immediate step.

## Work packages and dependencies

| Order | Work | Required completion evidence | Current status |
| --- | --- | --- | --- |
| A | Reconcile evidence and resources; produce a fresh continuation preflight | Every indexed attempt charged, hashes checked, complete/missing/unstarted SBC datasets distinguished; available suites resolve without sampling | Completed; evidence below |
| B | Diagnose complete-fit availability and construction cost | Saved native records classify missing L groups, safety-bound failures, zero retained output and unassessed members separately; instrument preparation, first-call compilation, steady execution and assessment without inferring cost from warnings | Implemented timing and cost pilots; native ordinary search can exhaust the prepared ceiling with no member; bound-requalification repair remains open |
| C | Add a stopped-versus-fixed experiment to the existing engine | Matched target, tuning artifact, candidate rule, starts and coupling policy; fixed counts chosen before results; both arms accounted for even when either fails; independent reference quantities and paired replication summaries | Implemented and tested; eight Gaussian and eight mixture replications per device completed; GPU mixture exposed one local-readiness/global-error counterexample |
| D | Extend reported quantities for constrained models | Separately represent active dimension and reported dimension; preserve active checkpoints and replay identity; test all three simplex probabilities from two coordinates, empty chunks, continuation and covariance singularity | Implemented and tested; all 23 CPU simplex members assessed; two selected GPU members assessed with all 44 qualified members retained |
| E | Supply independently checked realistic references | Regression, centered/noncentered eight-schools and MacroFinance bundles with exact data/prior/coordinate/quantity identity, source version, dependence and reference uncertainty | Inputs unavailable |
| F | Run fresh full-procedure SBC and stopped/fixed replications | Independent datasets, declared fit groups and output rules, complete denominators, development/confirmation separation, uncertainty and calibrated power at meaningful defect sizes | Ordinary SBC incomplete; separate prepared confirmation completed 224/224 fits and 32/32 datasets without a detected rank discrepancy; no ordinary calibration or small-error guarantee |
| G | Broaden mutation severity, acceptance-screen and sequential tests | Baseline/no-op controls, activation records, null rejection/detection intervals, difficult boundary cases; inspected Gandy--Scott sequential source before implementing the optional wrapper | Rank primitive and CPU/GPU actual-HMC severity calibration completed; subtle-defect power weak; acceptance-screen calibration and optional sequential wrapper remain open |

B starts by assessing whether requiring a particular L group is a usable
procedure, not by changing groups after seeing which candidates survived. A
missing group is an availability outcome. Any revised predeclared output rule
defines a new experiment and requires fresh data/randomness. Likewise,
insufficient warmup or retained allocations may justify a new development
design, but cannot be repaired by discarding unsuccessful fits or conditioning
SBC on favorable posterior diagnostics.

The saved-fit triage below also makes the next diagnostic concrete: inspect
bounded interior epsilon proposals for an L family whose directional repair
reaches the safe upper bound with excessive acceptance. The observed curve is
nonmonotone. Preserve the existing safety bound, acceptance criteria and all
verified members. Establish the response under independent measurements before
claiming that interpolation, a wider search budget or another proposal strategy
repairs the missing-family problem. An absent family need not mean no valid
pair exists, and no evidence here authorizes raising the safe upper bound.

C must measure global reference quantities. For mixtures include mode
probability and a bounded global functional; for heavy tails avoid undefined
means. Use paired repeated experiments when the declared coupling is valid.
Compare error and work jointly, report unpaired/missing outcomes, and do not
select the fixed endpoint to match the observed stopping time. Shared numerical
prefixes require identical chunk/seed semantics; otherwise use independently
seeded arms and say so. A fixed-size normal MCSE interval is not automatically
valid under optional stopping. Coverage is measured at the actual stopping time.

D needs a narrow shared posterior API repair rather than dropping the third
simplex coordinate or making a second sampler in the validation package. The
Dirichlet density/score and frozen-kernel coverage remain distinct from this
missing posterior capability. Do not set `pipeline_available=True` until the
actual retained bridge and posterior controller support the transform.

E accepts recorded external bundles only after target and uncertainty checks;
two packages agreeing is not proof of the model. Externally supplied MCMC rows
must not inherit an iid standard error. Approximate-filter posteriors and the
scientific model posterior require separate reference identities.

G also retains the broader plan's high-dimensional, training-inclusive transport
and consumer coverage as explicit future work. Frozen synthetic transports
cannot fill training-inclusive cells. The optional sequential test is not a
prerequisite for fixed-design SBC or stopping comparisons.

## Replication sizing and numeric provenance

Before F, choose a meaningful discrepancy for each quantity, then calibrate the
test's null rejection and power using independent pilot data. Freeze the test,
multiplicity family, stopping rule, seed policy, rank resolution and allocation
before fresh confirmation. The old two fits per dataset give only three rank
bins and have no demonstrated power against subtle defects.

For a separately estimated Bernoulli rate, independence gives
`Var(p_hat) = p*(1-p)/n <= 1/(4*n)`. Thus 100 independent units bound the rate
standard error by 0.05, and 400 bound it by 0.025. These are derived *precision
illustrations*, not confidence-interval guarantees, SBC power calculations or
selected campaign sizes. The preflight reports these explicitly labeled
alternatives. Multiple quantities and comparisons still need their declared
multiplicity/uncertainty treatment. Failed or missing outputs stay in their
denominator with separate availability reporting.

| Choice | Provenance and use | Failure mode / earliest check | Status |
| --- | --- | --- | --- |
| Existing campaign caps | Inherited convenience ceilings in the execution plan | Confusing remaining time with an adequate design; reconcile indexes first | Binding resource limit |
| Native tuner, acceptance policy and all-member retention | Current public registry and guide | Validation becomes another tuner; inspect route and record identities | Baseline under test |
| Existing small posterior caps and broad L grid | Development fixtures and native search policy | Missing fits or early caps dominate; inspect native outcomes before scaling | Hypotheses, not promoted defaults |
| Four chains and fixed L reporting group | Existing development designs | Shared within-fit observations or missing groups create biased rank selection; count whole fits and missing groups | Experiment-specific baseline |
| 0.05 / 0.025 rate standard errors | Illustrative precision choices plus the variance derivation above | Mistaken for statistical power or nominal interval coverage | Planning alternatives only |
| CPU-only preflight | Standard-library saved-record inspection | Accidental framework import/device initialization; CLI import regression | Offline exception, no sampler evidence |
| New campaign replication count and budget | Not yet measured or chosen | Underpowered or unaffordable run; complete-fit pilot and power calculation first | Unresolved, not a launchable design |

## Skeptical pre-execution audit

The audit rejects scaling the earlier SBC design immediately: it is incomplete,
its fit cost is censored by failures, and no powered design fits the remaining
budget. It also rejects describing the existing stopping engine as a comparison
against fixed-length sampling. Saved posterior diagnostics are subjects under
test and cannot supply their own reference. Retracing warnings alone do not
prove unbounded traces, compilation cost or a numerical defect.

The revised immediate step passes: its comparator is the immutable saved
indexes/results, its outputs answer readiness and accounting questions, and it
performs no statistical promotion. CPU/GPU budgets remain separate, failed
attempts are charged, running reservations are treated conservatively, and
historical sources are not relabeled. Independent synthetic corrupt/missing
record fixtures will check the auditor. No unrelated worktree changes are
included. The test and preflight output below are its terminal evidence.

The preflight also found that the shared numerical source has changed since
the terminal audit: among the changed files are `hmc_candidate_set_execution.py`
and `fixed_transport_hmc_mechanics_tf.py`. Therefore every saved numerical
profile is historical relative to this checkout. A focused existing prepared
member/reload/interval regression and the planning regressions will check the
current integration on an isolated CPU fixture. This is routine regression
work under the original test exclusion, not another statistical campaign or
permission to relabel the historical profiles current.

## Commands and artifacts

Use the existing `tfgpu` interpreter from the repository root. Prefix tests with
`CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true`; GPUs are intentionally
hidden. The preflight script uses only the standard library and does not import
the inference implementation.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/audit_inference_validation_campaign.py \
  docs/plans/artifacts/inference-validation-2026-09-15 \
  --output docs/plans/artifacts/inference-validation-phase2-2026-09-16/preflight-r1.json

CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/inference_validation/test_campaign_audit.py
```

Also resolve every `docs/validation/*.json` suite using the existing `plan`
command; it performs no numerical execution. Preserve a machine-readable suite
inventory alongside the preflight. Record command, source state, wall time,
results and remaining work in this document after execution. Use a new output
filename for a changed preflight; do not overwrite the original campaign.

## Execution record

The immediate continuation is complete. The reusable
[`audit_inference_validation_campaign.py`](../../scripts/audit_inference_validation_campaign.py)
reader and its ten focused regressions were added. The
[preflight](artifacts/inference-validation-phase2-2026-09-16/preflight-r1.json)
reproduces all **107** indexed result checks and **364** tensor checksums with
zero invalid artifacts. It charges failed attempts and unresolved running
reservations, reports source staleness and distinguishes complete, missing,
partially started and unstarted SBC datasets. The `1e-6` second ledger comparison
tolerance is only a floating-point summation allowance, not a compute extension.

All **nine suites** resolve into **73 jobs**, of which **70** have the required
inputs and **three** lack external reference/observation bundles. The
[suite inventory](artifacts/inference-validation-phase2-2026-09-16/suite-inventory-r1.json)
records each actual command and resolved plan. Input availability is not a
numerical pass, a funding decision or permission to execute all those jobs.

The new auditor tests passed **10/10**; current-source planning and real prepared
member/reload/interval tests passed **16/16**. Results are preserved in
[auditor JUnit](artifacts/inference-validation-phase2-2026-09-16/preflight-tests-r1.xml)
and [integration JUnit](artifacts/inference-validation-phase2-2026-09-16/current-source-tests-r1.xml).
The second command used the same CPU-hiding, memory-growth and two intra-op/one
inter-op thread environment as the first campaign:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/inference_validation/test_definitions.py \
  tests/inference_validation/test_public_pipeline.py \
  --junitxml=docs/plans/artifacts/inference-validation-phase2-2026-09-16/current-source-tests-r1.xml
```

The tested checkout was `d31e1b7618f613b9f7a7ec7ace7f11556870fb4e` plus the
preserved worktree. The new auditor tests took 0.30 seconds and the integration
tests 10.62 seconds as reported by pytest. Framework deprecation warnings do not
change those results. No production source or numerical default was changed by
this continuation, and no numerical campaign worker was launched. The remaining
allowances are still **126.34 CPU seconds** and **492.89 GPU seconds**.

### What the saved failures mean

The two earlier complete-process SBC attempts each have one complete dataset
out of two. The final attempt has zero complete datasets: one is missing a
required fit output and one started but lacks its terminal dataset record.
Those three attempts are separate evidence versions, not six independent
datasets that can be pooled to improve a rate.

The [native fit triage](artifacts/inference-validation-phase2-2026-09-16/saved-fit-triage-r1.json)
shows that all four final-attempt tuners completed. Dataset 0/fit 0 retained two
verified `L=5` members and no `L=3` member. Both `L=5` members produced 256
retained draws per chain. The predeclared SBC group was `L=3`, so that fit has no
eligible output for its rank. Dataset 1/fit 1 retained three verified members,
but its posterior pipeline had not completed before the worker cap.

For the missing `L=3` family, the recorded epsilon/mean-acceptance observations
are `(0.6, 0.96690)`, `(1.2, 0.82682)` and `(1.58600, 0.97907)`. The last epsilon
is the declared safe upper bound. These descriptive observations explain the
directional-repair boundary and motivate the interior-search diagnostic in B;
they do not establish a valid untested pair, numerical resonance, or the
superiority of another search strategy. Every observation labels R-hat as
reporting-only. This missing-family outcome does not show candidate loss or
R-hat reentering tuning admission.

### Decision and terminal review

| Decision | Primary criterion | Veto evidence | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the offline continuation checks | Ledger, integrity and availability tests pass | No corrupt indexed artifacts found | Byte integrity does not prove scientific validity | Use the recorded baseline for B | Inference calibrated |
| Preserve the failed SBC outcomes | Complete-fit and dataset denominators reconstructed | Missing L group and capped posterior output | Untested interior epsilon behavior and complete-fit cost | Diagnose B, then freeze a fresh design | Tuner or HMC universally fails |
| Defer the larger statistical campaign | No adequate measured allocation/power design exists | Remaining allowance is insufficient for the proposed expansion | Replication requirement and full cost | Complete implementation prerequisites and cost/power sizing before an expanded allocation | Broad plan finished |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Missing required SBC outputs and worker caps remain explicit |
| Statistically supported ranking | None |
| Descriptive-only differences | Saved acceptance values, timings and tiny SBC availability counts |
| Default-readiness | Not evaluated; tuning and posterior defaults unchanged |
| Next evidence needed | Complete-fit cost/availability, actual stopped/fixed comparator, realistic references and powered fresh replications |

The strongest alternative explanation for apparent readiness is that small
fixtures and coarse defects miss realistic failure mechanisms. Neither a clean
saved-record audit nor 26 focused regression passes establishes statistical
power or sufficient burn-in. Current-source numerical campaigns remain absent;
source changes have made all earlier campaign profiles historical under the
existing conservative source identity rule. The program remains open at B--G.
The resource boundary limits the next statistical experiment; it does not
invalidate the target, HMC, the references or the validation research direction.
