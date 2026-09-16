# Inference validation suites

These suites exercise distinct questions through shared target definitions,
independent references, the public HMC tuners, saved observations, and common
reports. Tuning retains every verified candidate; R-hat, ESS, and MCSE remain
posterior diagnostics. The package cannot issue a new tuning authority.

The architecture and source survey are linked from the
[execution specification](../plans/bayesfilter-inference-validation-execution-2026-09-15.md).
Each resolved design declares development or confirmation status. A completed
process is not necessarily a complete statistical experiment, and neither
status grants default readiness. A profile name describes execution scope, not
strength of scientific evidence.

Current completion, missing evidence and the next execution order are recorded
in the [continuation plan](../plans/bayesfilter-inference-validation-phase2-plan-2026-09-16.md).
The completed [funded campaign](../plans/bayesfilter-inference-validation-24h-campaign-2026-09-16.md)
used 21.80 charged CPU hours and 11.17 GPU worker-hours within separate 24-hour
allowances. Its [results](../plans/bayesfilter-inference-validation-24h-result-2026-09-16.md)
include complete prepared-route rank evidence, ordinary-search availability
failures and a missed-mode counterexample despite passing local posterior checks.
The offline command
`python scripts/audit_inference_validation_campaign.py <campaign-root> --output <new-file.json>`
reconciles saved attempt costs, result/tensor hashes and incomplete full-SBC
datasets without launching a sampler. A clean audit establishes consistency of
the saved records; it does not establish statistical calibration or fund another
campaign.

## Commands

Run from the repository using the existing `tfgpu` environment:

```bash
python -m bayesfilter.testing.inference_validation list
python -m bayesfilter.testing.inference_validation plan docs/validation/fast-cpu.json
python -m bayesfilter.testing.inference_validation run docs/validation/fast-cpu.json --output /tmp/my-new-validation-run
python -m bayesfilter.testing.inference_validation run docs/validation/fast-cpu.json --output /tmp/my-new-validation-run --resume
python -m bayesfilter.testing.inference_validation report /tmp/my-new-validation-run
python -m bayesfilter.testing.inference_validation assess /tmp/my-new-validation-run --output /tmp/my-new-rank-assessment.json
```

Use a fresh versioned output directory for each evidence version. Resume checks
the suite and source identity, skips completed jobs, preserves failed attempts,
and charges consumed worker time. Native numerical checkpoints retain completed
tuning and posterior chunks. A coordinator interruption with unknown worker
completion charges the reserved budget conservatively. An ordinary advisory lock
prevents accidental concurrent coordinators. GPU runs require trusted execution;
workers verify memory growth before initialization. CPU profiles deliberately
hide GPU devices and use a documented non-XLA diagnostic exception.

`--max-workers N` supervises up to N separate worker processes with one index
writer. Budgets count summed worker wall time, including failed and cancelled
attempts, rather than elapsed calendar time. Predeclared SBC groups can combine
independently seeded dataset shards from the same suite and frozen source.
Aggregation verifies membership, numerical settings, source and result checksums;
missing groups and fits stay in their original denominator. Pilot, changed-source
and repaired confirmation experiments are not pooled.

For ongoing accounting use `scripts/audit_inference_validation_campaign.py`
with `--live --cpu-seconds 86400 --gpu-seconds 86400`. Add `--verify-artifacts`
for result and tensor checksum checks. Its live ledger includes running and
queued reservations; an explicitly cancelled suite releases unstarted work.
Use `--overhead-cpu-seconds` and `--overhead-gpu-seconds` to include separately
accounted test/report costs or conservative charges in the allowance check.
Unstarted jobs retain their planned statistical denominator and zero worker cost.
This is accounting, not an assessment of statistical validity.

`assess` recalculates rank and permutation tests from preserved observations,
without launching a sampler. Other engines report `reassessment_not_supported`
and link their original assessments. Reassessment is exploratory and cannot
silently turn development evidence into confirmation.

## Profiles and interpretation

| Suite | Implemented exercise | Limit |
| --- | --- | --- |
| `fast-cpu.json` | All available target densities/scores; dense-metric leapfrog oracle; controller repair, retention, stream and restart checks; known faults | Numerical and controller checks, no convergence claim |
| `pipeline-cpu.json` | Automatic and prepared tuning, frozen transport, every-member posterior assessment, actual stopping, and independent complete-fit SBC units | Small CPU development runs; SBC and stopping estimates have wide uncertainty |
| `statistical-cpu.json` | Frozen-kernel random-position and two-sample tests across difficult target families, reference SBC, ignored-data controls, MCSE/R-hat arithmetic and primitive power | Primitive calibration does not establish full-pipeline power; no global all-model correctness test |
| `numerical-gpu.json` | GPU/XLA mechanics, frozen-kernel defects, ordinary and frozen-transport tuning, full-procedure SBC and repeated real-HMC defect tests | GPU coverage is only for the executed shapes, targets, settings and source version |
| `external-consumer.json` | Matched external observations and references for regression, eight schools and MacroFinance | Unavailable until actual bundles are supplied; no external fit or consumer numerical evidence is fabricated |

Execution validity, sampler findings and test response occupy separate fields.
A known wrong-energy kernel being rejected is successful defect detection and
an unfavorable sampler result. An exception is never counted as detection.
Identity and sign-flip two-cycle kernels preserve the symmetric Gaussian law;
their lack of exploration illustrates why invariance cannot establish mixing.
A position-only wrong score can still give an invariant Metropolis-corrected
kernel; the density/score oracle, rather than invariance alone, must detect it.

The reference SBC engine uses proper generative normal-normal, beta-binomial
and LGSSM-location models. The last model has one unknown location with a normal
prior and a known latent AR(1) covariance. Its reference uses a separate scalar
Kalman recursion. It is not a reference for arbitrary unknown state-space
parameters. Parameter, bounded-radius, dependency and data-dependent likelihood
quantities are evaluated where applicable. Truth is assessor-only: numerical
fits start from fixed data-independent positions.

Full SBC uses independent complete fits conditional on each dataset and takes
one fixed output per fit. It never calls chain draws or candidate siblings
independent replications. `declared_l_first` chooses the first ID within a
predeclared L group; `first_verified` chooses the first ID across the returned
set. Selection precedes posterior sampling and uses no truth or posterior
diagnostic. `posterior_members: selected` leaves sibling records explicitly
unassessed; `all` assesses every verified member. Missing fits stay in the planned denominator. Conditional ranks from
completed datasets cannot establish unconditional calibration when fits fail.

Ordinary designs with `native_search: true` pass no search override to the
public tuner. Its own preparation chooses the starting epsilon and broad search
configuration. Designs supplying `search` and `step_size` exercise a different,
explicitly configured procedure. An epsilon supplied before preparation can
exceed the final-metric bound, and native bounded search can return no verified
member. These are separate findings; neither is repaired by changing acceptance
thresholds after inspecting results.

Actual-stopping rows compare the final controller's means and medians with
available exact functionals. Nominal normal intervals formed from its MCSE are
evaluated at the actual stop, including cap outcomes. Binomial intervals use
complete pipeline replications and one declared candidate group, never sibling
counts. Failure to obtain a posterior output remains unavailable. These measured
intervals have no presupposed sequential coverage guarantee.

Designs declaring `fixed_comparator` additionally use the same verified member
for separately seeded, fixed discarded warmup and retained blocks. Both arms
are archived through the existing member runner. Exact-reference mean, median
and mixture mode-probability errors remain available even when runtime readiness
fails. A comparator exception preserves the stopped arm and records a missing
pair. Pointwise paired intervals and small-replication coverage estimates are
descriptive; they do not prove one stopping rule superior.

Reports distinguish controller calls from usable posterior outputs. A completed
call with no retained draws is unavailable for posterior accuracy; every
declared stopping quantity remains in the denominator even if all intervals
are missing. The generated table shows current-source executions, historical
executions and planned designs separately. Affine and dense nonlinear transport
evidence occupy different cells. Repeated executions of one design are repairs,
not additional independent replications.

## Allocation and assumption audit

| Choice | Provenance and justification | Failure diagnostic / interpretation |
| --- | --- | --- |
| Float64 densities and gradients | Reference regime; independent SciPy formulas and scale-aware centered differences | Relative error; GPU/XLA coverage is separately executed |
| FD step `eps_machine^(1/3) * max(1, abs(q))`; relative tolerance `100 * eps_machine^(2/3)` | Balances central-difference truncation and rounding; factor 100 is an engineering margin | Raw probes, values, scores and errors retained; not an exact-gradient theorem |
| Dense momentum mass `[[2,.3],[.3,.8]]` | Positive definite nonidentity diagnostic fixture, chosen to expose covariance/precision inversion | Compare TFP leapfrog with independently derived block-matrix powers and reversal |
| Density tolerance `1e-10`, leapfrog tolerance `1e-9` | Conservative float64 engineering margins on order-one fixtures | Quantitative errors retained; no transfer to ill-scaled consumers |
| R-hat arithmetic tolerance `1e-12` | Existing independent-rank regression tolerance, reused for the same float64 array experiment | Arithmetic mismatch is distinct from an unfavorable R-hat value |
| Target parameters | Named laws: Gaussian scales, banana bend, funnel scale, t degrees of freedom, mixture separation/weight and proper conjugate priors; complete values are in target/reference source and scenario JSON | Independently checked density/score laws; difficult settings are stress hypotheses, not tuning defaults |
| Initial offsets `[-1,-.3,.4,1]`, remote offset 8, mode offset -5 | Fixed, reproducible data-independent start hypotheses | Preserve actual preparation bank; these do not establish overdispersion or global mode coverage |
| Standard automatic preparation | Existing public preset under test | Smoke preparation failed to produce a dispersed Gaussian bank in development; standard is used for complete-path scope, preserving that failed smoke observation |
| Broad L grid `[3,5,9,13,18,25]`; supplied-pair `[3,5]` profiles | Existing broad policy versus explicit narrower prepared-route coverage | Do not label supplied-pair evidence as broad automatic coverage |
| Default search pilot/refinement and finite budgets | Existing controller with declared engineering allocations; no winner selection | Zero-member, capped, inconclusive and incomplete outcomes are reported |
| Prepared epsilon hypotheses 1.1/1.3/1.5; initial .4/.6/.8; wrong-energy 1.7 | Convenient Gaussian/reference exploration and strong named defect sizes, not transferred defaults | Every pair receives its own real acceptance and health checks; a failed hypothesis is not repaired by relaxing acceptance |
| Eight discarded evidence warmup transitions; at least 64 measured draws | Small engineering allocations; native acceptance policy minimum retained | Weak evidence can be inconclusive; no convergence inference |
| Prepared epsilon domain [.005,3], repair factor 2, maximum five repairs per family | Explicit finite engineering search hypotheses; automatic/frozen routes preserve their native domain derivation | Boundary or repair exhaustion stays a failure to find a member, not proof no member exists |
| Posterior chunks/caps and MCSE tolerance .1, bounded-function error tolerance .25 | Small development allocations in each JSON, chosen to exercise stopping and persistence | Report actual counts, error and uncertainty; tolerance comparison remains descriptive |
| Per-design alpha .05 with within-design Bonferroni multiplicity | Conventional diagnostic error allocation; each design is its own stated scientific question | No family-wide all-model correctness claim; negative-control designs are reported separately |
| Monte Carlo null counts 399 or 1999 | Plus-one p-values, with preflight resolution check against every tested quantity | Discrete resolution is preserved in each p-value |
| 128/256 invariance or reference-SBC replications; 256 primitive-power trials | Coarse development allocation; binomial SE is at most .03125 for 256 trials | Exact binomial bounds reported; continuous differences remain descriptive |
| Two datasets × two full fits for SBC; eight real-HMC power trials | Bounded complete-path demonstrations based on local measured preparation/compile cost | Too small for strong calibration or power claims; failed fits are explicit |
| AR(1) rho .8; transient offset `8 exp(-t/(n/3))`; constant and missed-mode controls | Known stationary finite-sample covariance and deliberately severe diagnostic stress | Compare runtime arithmetic with independent ranks and exact stationary variance; transient/missed-mode outputs do not inherit stationary CLT validity |
| Seed 20260916 with separately hashed phase/dataset/fit/member streams | Reproducibility choice, not a scientific default | Restart retains identity; no seed search or rerun-until-pass |

The historical CPU profile maximum worker reservations sum to 1,790 seconds. Its initial GPU
profile reserves 2,900 seconds, below the 3,600-second allowance. These are hard
engineering ceilings, not precision-based scientific allocations. Local failures
and retries consume the same total allowance. Primitive tests measure observed
cost before any larger confirmation allocation is proposed.

## Extending coverage

Register a target with its coordinate dimension, model quantity names, moment
conditions, generative capability and independent reference. Add target and
reference implementations separately, and test density/score/transform agreement.
Create scenarios and JSON designs using an existing engine. Add a named mutation
and a no-op control when a new mechanism needs a new oracle. Declare required
coverage cells explicitly; a controller double cannot satisfy numerical coverage.

The external reference reader accepts portable saved JSON observations with
target/data/prior/coordinate identity, quantity order, source/version/method,
uncertainty, dependency declaration, and sample checksum. This is an ingestion
boundary, not a claim that arbitrary external output is correct. A real consumer
still needs matched density/coordinate evidence and a reviewed reference.
Only a bundle explicitly declaring `sampling_structure: "iid"` receives an iid
reference standard error; MCMC or unknown dependence leaves that combined error
unavailable. Mean reporting additionally requires `finite_variance: true`.

Current limits that remain visible in coverage include external/consumer fits,
complete large replicated automatic full-procedure SBC, nonlinear
learned-transport preparation, and conditional position-field mechanics beyond
existing repository tests. Dirichlet posterior reporting now supports three
named probabilities from its two active coordinates. The CPU campaign pilot
assessed all 23 verified simplex members; a capped all-member GPU pilot remains
incomplete and cannot receive the CPU result's credit. A separate GPU subset
experiment assessed two selected members across two replications and retained
all 44 verified members; the other 42 remain unassessed for posterior accuracy.
The fixed-look rank design does not implement Gandy--Scott's optional sequential
testing wrapper. None of these cells is satisfied by counting additional pytest
cases or by an architectural review verdict.
