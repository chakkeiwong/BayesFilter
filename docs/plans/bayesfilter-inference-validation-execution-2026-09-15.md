# Inference validation: bounded implementation and execution

Date: 2026-09-15. Owner authorized revision, thorough review and execution.
Baseline HEAD: `5139f151237e9764ebf34b4e8cf6ceee1e74a38d`, plus the preserved
existing worktree. Architecture: [revised plan](bayesfilter-inference-validation-infrastructure-plan-2026-09-15.md).
Claude's review is architectural and did not inspect numerical code.

## Intent and evidence contract

Question: can one reusable system execute the distinct surveyed experiments,
exercise the real public HMC procedure, preserve every candidate/failure and
produce independent, uncertainty-aware findings and honest coverage?

Comparators are independently derived densities/scores/moments and exact draws,
known stationary processes, correctly constructed rank nulls, and named
deliberately incorrect variants. Existing runtime diagnostics are subjects under
test, not their own reference. No sampler or estimator superiority is sought.

| Role | Criterion |
| --- | --- |
| Engineering pass | Declared engines run with validated inputs; independent assertions, checkpoint/replay, scope, coverage and CLI tests pass |
| Statistical finding | Fixed-design tests report calibrated-null assumptions, p-values/effect errors, multiplicity and availability; null/power replications carry binomial uncertainty |
| Promotion veto | Incorrect numerical identity or unresolved statistical assumptions prevents that claim; absence of rejection never grants practical accuracy |
| Continuation veto | Broken reference/target, corrupted artifacts, invalid GPU launch or total budget exhaustion stops the affected experiment |
| Repair trigger | Local harness failure or detected implementation defect; repair, regression and fresh evidence under remaining budget |
| Explanatory | Acceptance, R-hat, ESS, runtime and MCSE when not explicit posterior criteria; none alters tuning qualification |
| Not concluded | Universal mixing, sufficient burn-in, exact sequential confidence, target-scale default readiness, external parity without checked external evidence |

Every completed test must distinguish execution validity, sampler finding and
expected test response. Defect-induced import/device errors are not detection.
Catalog scope includes all families in the architecture; missing external data
or target adapters remains explicit missing coverage, never a fabricated pass.

## Execution limits and assumptions

Total campaign allowance: 3,600 GPU worker-seconds and 1,800 CPU worker-seconds
for numerical demonstrations/calibration, excluding routine implementation and
pytest regressions. These are convenience ceilings chosen to bound this local
engineering task, not expected runtimes or scientific allocations. Measure a
small complete-path pilot before allocating replicated complete fits. Stop new
jobs when budget is insufficient; record actual cost and unfunded cells.

Output root: `docs/plans/artifacts/inference-validation-2026-09-15/`, with fresh
versioned run directories. Environment: existing `tfgpu` Python; no environment
mutation. CPU reference checks deliberately hide GPUs. GPU workers require
trusted access, verified growth, and TF/TFP XLA. Source hashes, versions, seeds,
commands, device policy and elapsed times accompany each execution.

Numerical/statistical allocation audit:

- Float64 and independent formulas are reference choices to isolate arithmetic.
  Scale-aware finite-difference steps follow cube-root machine epsilon; their
  error is a diagnostic, not an exact gradient proof.
- Existing preparation presets and broad `(3,5,9,13,18,25)` L coverage are
  inherited procedure choices under test. Small evidence/posterior allocations
  are explicitly fixture profiles, not new runtime defaults.
- Statistical primitive calibration uses 256 independent replications when
  affordable: worst-case binomial standard error is `1/(2*sqrt(256))=0.03125`.
  This measures coarse defect sensitivity, not tight calibration of rare errors.
- Test-family alpha 0.05 is a conventional diagnostic hypothesis, with explicit
  within-design Bonferroni correction. All quantities and fit groups count.
  Fixed looks only; no rerun-until-pass or optional peeking.
- Monte Carlo rank-null comparisons use a predeclared simulation count and
  plus-one p-value. Its resolution must be finer than the adjusted threshold.
  Uniform integer ranks are checked independently of the sampler implementation.
- Full SBC uses independent complete fits per dataset, one fixed output per fit.
  Tiny complete-fit demonstrations validate plumbing only; insufficient dataset
  replication is labeled underpowered rather than promoted to calibration.
- Accuracy tolerances on analytical fixtures are engineering hypotheses in
  posterior units. Report observed error and uncertainty; do not call a wide
  tolerance a scientific validation or adjust it to obtain a pass.
- Each seed is a reproducibility choice, separated by experiment, dataset, fit,
  candidate, phase and attempt. Replayed records are never extra replications.

## Skeptical pre-execution audit

Revisions address incomplete SBC units and success conditioning; prior tuning
and the test oracle cannot share generating truth. The data generator and
reference provider must be separate from the fitted target. Settings, devices,
automatic/prepared routes and candidate subsets are recorded rather than inferred.
The prior Gaussian example is not used as evidence of complete automatic coverage.
Reviewers did not prove code correctness. No uncalibrated two-sample energy
primitive is silently reused. A test failure rejects its candidate/claim, not
the whole program. The first complete path is followed by the remaining engines
and catalog tests; it is not declared program completion. This audit passes for
implementation and bounded engineering execution under the stated limits.

## Implementation sequence

Implement records/catalog, designs and scope validation; independent references
and controls; public/frozen procedure adapters; distinct assessment engines;
bounded execution/resume and reports; numerical/null/mutation tests; then actual
automatic pipeline, replicated primitives and funded full-fit demonstrations.
Add generated coverage to the reference and guide. Preserve all unrelated work.
Record commands, results, failures, cost, decision/inference-status tables and
terminal red-team findings in a separate execution result.

## Implementation audit before numerical profiles

The draft compiles but is not execution evidence. The source-level audit checked
the public capability registry, current candidate receipts, retained-runner
coordinate mapping, and Gandy--Scott Algorithm 2 / Proposition 2.3 alongside
the authors' `mcunit` implementation. For a reversible position kernel, applying
the same kernel independently from the anchor and reversing the left arm is
correct; no negative-time HMC kernel is required. The Dirichlet reference's
`sum(alpha * log(p))` already includes the additive-logistic Jacobian.

Actual repairs required: explicit float64 step constants in the independent
leapfrog comparison; missing-receipt/stream checks; invalid route/control
combinations that would silently do nothing; failed-fit denominators; truthful
empty-member and incomplete-work findings; interrupted attempt accounting;
reference-domain checks; and calibrated power for both rank and two-sample
primitives. These are harness repairs, not changes to HMC qualification.
The first CPU commands are debugging checks only. Funded profiles start after
these changes pass focused regressions, and source changes start fresh evidence.

The resolved profiles and numeric-provenance table are in
[`docs/validation/README.md`](../validation/README.md). Their maximum CPU
reservations total 1,790 seconds; the initial GPU profile reserves 2,900 seconds.
These preserve the original total allowances. The standard preparation preset
is necessary for the complete ordinary route: the tiny smoke allocation failed
the existing start-bank dispersion requirement even on a Gaussian. This is a
recorded preparation limitation, not evidence against HMC. The bridge regression
uses a deliberately broad centered acceptance region (.5,.9), with repair region
(.45,.95), to exercise persistence independently of near-boundary stochastic
qualification. The numerical profiles retain the native default acceptance policy.

The trusted readiness helper reported no device satisfying its conservative idle
policy. A subsequent trusted NVIDIA inventory showed GPU 0 at 1% utilization with
750 MiB used out of 32,760 MiB; GPU 1 was busy. GPU 0 may be shared for these small
validation kernels with verified memory growth. No process is stopped, no memory
is preallocated, and these runs cannot support throughput comparisons. The worker
records the actual visible-device and memory policy. Recheck if resource use
changes materially before the numerical profile starts.

## Cost-informed local continuation

The original CPU automatic job completed tuning (48 candidates, 15 verified) in
about 90 seconds and assessed 12 members before its 180-second worker cap. The
prepared two-replication job also hit its 90-second cap. Those failed attempts
remain intact. They show insufficient worker allocation for every-member
assessment, not failure to retain qualified tuning candidates.

An additional fresh automatic CPU attempt reserves 360 seconds using
`docs/validation/automatic-cpu-recovery.json`. It preserves the method, target,
acceptance policy and scientific interpretation; it is an engineering retry,
not another independent statistical replication. Fast CPU and statistical
profiles consumed about 192 seconds together, leaving sufficient unused time
within the 1,800-second campaign allowance even after reserving all unfinished
original CPU jobs and this retry. No stopping threshold is changed to make a
candidate pass.

`docs/validation/nonlinear-transport-gpu.json` reserves another 400 seconds for
the repository's frozen dense-IAF codec with fixed synthetic weights, in addition
to the diagonal-affine case. It adds no training or transport-quality claim and
keeps total initial GPU reservations at 3,300 seconds, below 3,600. Its target is
the same exact Gaussian in model coordinates. Existing proposal-field mechanics
and native retained-bridge regressions are tested separately; they do not supply
exact-score retained authority for an arbitrary proposal field.

After the original CPU profiles and automatic retry ended, their indexed worker
time totaled 1,097.19 seconds, leaving 702.81 seconds. The automatic retry
completed all 15 verified members in 207.48 seconds. Reserve another 580 seconds
for fresh prepared (140), actual stopping (270), and missed-mode stopping (170)
attempts in `pipeline-cpu-recovery.json`, preserving their original criteria and
all-member policy. The higher allocations respond to measured assessment cost;
they are not threshold changes or additional independent comparison replications.
The unchanged complete-fit SBC already produced a fully recorded incomplete
calibration result: one dataset lacked a selected L-group output and encountered
an epsilon proposal above the final-metric safety bound; another produced both
outputs despite unfavorable posterior diagnostics. Those failures are scientific
availability evidence for that explicit configuration, not a reason to discard
datasets or loosen the safety bound.

## Terminal source audit and bounded completion (2026-09-16)

The second source audit found that a completed posterior-controller call could
contain zero retained draws, yet the aggregate called every such member
assessed. It also found that all-missing stopping quantities disappeared from
the summary, reference-array diagnostics always reported completion even after
an early deadline, and external MCMC reference rows inherited an iid reference
standard error. Repair these reporting/availability errors, explicitly separate
unavailable mutation outcomes from defect detection, and test the boundary
cases. Do not change tuning membership, scientific thresholds, or the sampler.
Mark the unsupported simplex3 posterior transform at planning time; its valid
density/score/frozen-kernel tests remain usable.

The skeptical audit passes for this repair: the baseline is the original saved
native record, the independent references remain unchanged, and no empty output
or favorable diagnostic can satisfy posterior coverage. External correlation
must disable an iid standard-error claim. Source hashes will keep earlier runs
historical; do not relabel them current or rerun the entire matrix just to make
the coverage table look complete. The full-book build is documentation checking,
not statistical evidence.

Indexed consumption before terminal repairs is 1,673.66 CPU worker-seconds and
1,509.15 GPU worker-seconds. The two post-repair CPU reruns used 81.81 and 114.36
seconds of the original allowance; they are implementation retries, not new
statistical replications. Remaining allowances are 126.34 CPU and 2,090.85 GPU
seconds. After focused regression tests, reserve at most 1,990 GPU seconds for
fresh automatic tuning (600), dense frozen transport (500), full SBC (800),
Gaussian mechanics (45), and wrong-energy invariance (45). These caps reflect
the earlier measured costs; unused reservations are not extra authority.
The dense-transport retry changes only search work allocation from 40 to 100
units because the first run had five candidates still awaiting verification;
its law, transform, epsilon proposals, thresholds and posterior caps stay fixed.
Use a new JSON suite and output root. Stop jobs at their caps and preserve any
remaining failures or missing coverage; no rerun-until-pass or claim of confirmed
calibration is authorized by these development allocations.

Final indexed spending is 1,673.66 CPU and 3,107.11 GPU worker-seconds. The
terminal dense-transport retry completed; automatic posterior assessment and
full SBC hit their individual caps. The detailed terminal result, incomplete
coverage and next evidence are in
[the execution audit](bayesfilter-inference-validation-infrastructure-execution-result-2026-09-16.md).

The next authorized work and the unresolved statistical allocation are specified
in [the continuation plan](bayesfilter-inference-validation-phase2-plan-2026-09-16.md).
It preserves these ceilings and prior attempts; the first campaign is not
silently restarted with a fresh allowance.
