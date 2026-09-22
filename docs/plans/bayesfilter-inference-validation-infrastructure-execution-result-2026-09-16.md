# Inference validation infrastructure: execution and terminal audit

Date: 2026-09-16. This implements the bounded campaign in
[the execution specification](bayesfilter-inference-validation-execution-2026-09-15.md)
and the architecture in
[the infrastructure plan](bayesfilter-inference-validation-infrastructure-plan-2026-09-15.md).
The [terminal campaign audit](artifacts/inference-validation-2026-09-15/terminal_audit.json)
and [final GPU report](artifacts/inference-validation-2026-09-15/gpu-terminal-r1/report.md)
preserve the completed accounting. Unrelated worktree changes have been preserved.

## Verdict on the review and the implemented program

Claude's architectural agreement was reasonable. Its audit explicitly did not
inspect numerical execution and therefore did not establish implementation
correctness. The architecture needed no replacement. Implementation required
repairs to statistical units, numerical references, failure accounting and
reporting before it could answer the proposed questions honestly.

The reusable package now supplies target/reference definitions, validated
experiment designs, public-procedure adapters, seven experiment engines,
bounded workers, native checkpoints, observation checksums, assessment and
coverage reports. The common commands are `list`, `plan`, `run`, `assess`, and
`report`; suite definitions and extension instructions are in
[docs/validation](../validation/README.md). The implementation calls the real
BayesFilter tuners, retained-member bridge and posterior controller.

This delivery provides executable development infrastructure. It does not
complete broad statistical calibration, certify every supported sampler case,
or establish sufficient burn-in. Required missing cells remain visible.

## Repairs found by source and execution review

| Finding | Repair and verification |
| --- | --- |
| A one-pair or controller-only test could be mistaken for complete tuning coverage | Ordinary automatic, prepared, frozen, controller and external routes have distinct declarations; numerical tests use the public tuners |
| Dense-mass leapfrog reference and scalar dtype needed independent checking | Independent block-matrix oracle, explicit float64 constants, reversal and wrong-metric regression |
| Candidate records alone did not establish correct verification or retention | Check exact L/epsilon, candidate hashes, mass, fresh streams/seeds, receipts, ancestry, work observations and retained membership |
| SBC could incorrectly count chain draws or sibling candidates as independent | Independent complete fits within each dataset, one predeclared output per fit, fixed L group, generating truth restricted to the assessor |
| Parameter ranks could miss ignored data | Include data-dependent likelihood quantities and an independent ignored-data control |
| Invariance could be mistaken for exploration | Identity and two-cycle controls, explicit no-mixing conclusion, separate posterior and stopping experiments |
| Completed controller calls could contain no retained draws | Separate attempted assessments, usable outputs, missing members and full assessment completion; also interpret older saved records conservatively |
| All-missing stopping quantities disappeared | Initialize every declared mean/median denominator, include unavailable and capped outcomes, assess intervals at the actual stopping time |
| A diagnostic-array deadline could be reported as complete | Record planned/completed replications and return incomplete when the deadline prevents completion |
| Invalid evidence could be mistaken for mutation detection | Separate invalid/unavailable execution from intended oracle discrepancy and from estimated detection probability |
| External MCMC rows inherited an iid standard error | Require an explicit iid declaration for that calculation; otherwise leave combined reference uncertainty unavailable |
| Simplex dimensionality failed late in posterior execution | Plan the unsupported three-quantity/two-state posterior transform as unavailable; retain valid Dirichlet density/score/invariance coverage |
| Restarts could duplicate evidence or lose budget charges | Native checkpoint reload, completed-result consistency, ordinary coordinator lock and conservative interrupted-reservation accounting |
| Earlier source or affine evidence could silently fill a current nonlinear cell | Source staleness, result/tensor hashes and separate affine/dense-IAF coverage; current and earlier-source counts shown separately |

The package remains diagnostic. Inference runtime code does not import it.
R-hat, ESS and MCSE do not reject, rank, repair or delay tuning candidates.
Every verified member stays in the tuning record even when its separate
posterior controller fails or remains unfinished.

## Engineering verification

The final validation package tests pass: **72 tests**, including independent
references, numerical mechanisms, mutation controls, missing-output accounting,
serialization, job resume and real retained-member replay. The existing neural
force and candidate execution regressions pass: **60 tests**. The tuning
documentation contract passes: **15 tests**. These counts are regression
coverage, not independent stochastic replications.

Commands used the existing `tfgpu` interpreter, with deliberate CPU hiding and
memory-growth settings before TensorFlow import:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
tests/inference_validation --disable-warnings

# Same environment for the existing regressions:
python -m pytest -q tests/test_neural_force_hmc.py tests/test_hmc_candidate_set_execution.py --disable-warnings
python -m pytest -q tests/test_hmc_tuning_documentation_contract.py --disable-warnings
```

Compile checks and whitespace checks passed. GPU worker manifests record TF
2.20.0, TFP 0.25.0, float64 fixtures, XLA, TF32 state and verified memory growth.
GPU 0 was shared; its changing utilization prevents throughput comparisons.
Full command arguments, source hashes, root seeds, design values, native records
and elapsed times are preserved per attempt beneath
`docs/plans/artifacts/inference-validation-2026-09-15/`.

The guide's validation explanation and generated model/route coverage were
updated. An isolated full-book LaTeX build resolved the new validation and
precision citations. Three unrelated existing citations in the discontinuous
gradient chapter remain undefined: `Afshar2015`, `Gorinova2020`, `Pakman2014`.
Their absence is not caused by this validation change.
The [560-page build](artifacts/inference-validation-2026-09-15/guide-build-r1/main.pdf)
and build log are preserved; the validation prose and coverage pages were
visually inspected, with table spacing repaired before the final build.

All **107** indexed completed results and **364** saved tensor checksums matched.
This verifies the recorded result/tensor files; it is not a new numerical or
statistical validation of their contents. The final report regenerates source
staleness and incomplete-posterior status from the saved records.
All 16 recorded GPU attempt manifests confirm memory growth before logical
device initialization and the declared XLA execution setting.

## Campaign cost and completion

| Execution profile | Indexed worker-seconds | Process outcomes |
| --- | ---: | --- |
| CPU fast, first and second versions | 82.07 + 81.81 | 20 + 20 completed |
| CPU statistical, first and second versions | 110.68 + 114.36 | 23 + 23 completed |
| CPU pipeline initial | 696.96 | 3 completed, 4 capped |
| CPU automatic recovery | 207.48 | 1 completed |
| CPU pipeline recovery | 380.29 | 3 completed |
| Initial GPU matrix | 1,401.94 | 10 completed |
| First dense-IAF GPU search | 107.21 | Process completed; no verified members |
| Terminal GPU repair | 1,597.97 | 3 completed, automatic and SBC capped |
| External inventories | 0 | Three unavailable targets in each inventory |

CPU use totals **1,673.66 / 1,800 seconds**; GPU use totals
**3,107.11 / 3,600 seconds**. Routine pytest and document builds are outside
these numerical-worker allowances, as declared before execution. The remaining
126.34 CPU and 492.89 GPU seconds do not turn capped jobs into completed ones.
No additional retry is claimed or counted. The terminal source version has
fresh dense-transport, Gaussian mechanics and wrong-energy evidence plus the
partial ordinary/SBC records. Earlier complete runs remain historical evidence
for their saved source, including reruns with identical seeds.

## What the numerical runs show

The fast profiles exercised 13 available target laws: Gaussian, rotated
Gaussian, banana, funnel, Student-t, Cauchy, mixture, Gamma, Beta, Dirichlet,
normal-normal, beta-binomial and the scalar LGSSM-location model. Density/score
mechanics passed. Named score, metric, Jacobian, candidate-retention,
cross-L and missing-observation defects were detected. These experiments do
not establish full tuning coverage for all 13 laws.

Frozen-kernel profiles tested separate rank and independent-sample experiments.
Identity and two-cycle controls preserved the intended invariant law while
demonstrating absent exploration. The independent primitive calibration had
14/256 null rank rejections (95% binomial interval approximately .030--.090),
4/256 null KS rejections (.004--.040), and 256/256 detections for each strong
ignored-data/location control (.986--1). Eight real-HMC wrong-energy
replications detected the defect eight times, with interval .631--1; baseline
rejections were 0/8, with interval 0--.369. This is coarse development evidence,
especially for actual sampler power. Repeated profiles with the same seed
are implementation retries, not extra replications.

Earlier complete automatic runs assessed all 15 verified CPU members and all
13 verified GPU members. Some completed CPU assessments had no retained draws;
the repaired report distinguishes that missing output. In the final automatic
GPU attempt, tuning completed with 46 candidates and 12 verified members, but
the worker cap stopped posterior execution after nine member calls. Their
draws and every tuning record remain saved. The three remaining members are
unassessed in that attempt.

The first dense-IAF transport experiment exhausted its 40-unit search allocation
with five candidates still validating. A fresh 100-unit allocation, preserving
target, transform, epsilon proposals and thresholds, completed with 25 candidate
records and two verified members. Both produced posterior outputs within the
declared descriptive reference tolerance. Neither passed all posterior stopping
criteria. This demonstrates functioning transport/replay integration without
turning a tuning pass into a convergence assertion. No transport training was
performed or evaluated.

Both earlier full SBC jobs completed as processes but had only one complete
dataset out of two; their findings were `calibration_incomplete`. Missing L-group
outputs and, in the CPU case, an epsilon proposal beyond the prepared safety
bound remained in the record. Even complete two-dataset runs would be inadequate
for a calibration claim.

The final SBC attempt reached its 800-second cap. Dataset 0 again lacked an
output in the declared L group, and dataset 1 was unfinished at termination.
It therefore supplies partial execution/failure evidence only. The report does
not substitute ranks from the completed fits or treat process timeout as a
detected statistical defect.

The missed-mode stopping experiment found favorable local posterior diagnostics
despite reference disagreement in both independent replications. The selected
group's first-coordinate mean and median intervals covered the global reference
in 0/2 cases. The ordinary Gaussian stopping experiment had mixed coverage in
four replications. These are deliberate small development experiments; they
establish neither a reliable error rate nor universal failure. They show why
MCSE and local chain diagnostics cannot prove exploration of an unseen mode.

## Decision and inference status

| Decision | Primary criterion | Veto evidence | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Use the infrastructure for development | Definition, numerical, persistence and documentation regressions pass | Incomplete/invalid evidence stays visible | Coverage and reference availability remain limited | Add declared targets/defects through existing engines | Universal sampler correctness |
| Preserve candidate-set tuning semantics | Exact-pair inventory and retained bridge checked | No losing of verified members to posterior diagnostics | Expensive all-member assessment can exceed worker caps | Allocate measured assessment cost explicitly | All verified members have converged |
| Keep the stopping policy empirically testable | Actual stopped quantities compared with independent references | Missed-mode disagreement and unavailable outputs | Tiny replicated sample and sequential interval coverage | Larger fresh stopped-versus-fixed experiments with global references | Sufficient burn-in from a local diagnostic |
| Keep external cases unresolved | Matched bundles required | No usable regression/eight-schools/MacroFinance bundle | Target/prior/coordinate and reference error unknown | Supply and independently check those inputs | Consumer pipeline validation |

| Inference category | Status |
| --- | --- |
| Hard veto screen | Numerical mutations, invalid/missing evidence, reference mismatch and timeouts remain explicit |
| Statistically supported ranking | None sought or established |
| Descriptive-only differences | Posterior errors, acceptance, timing and small-run stopping comparisons |
| Default-readiness | Not established; no runtime default changed |
| Next evidence needed | Fresh adequately powered full-procedure SBC, stopped-versus-fixed comparisons and target-specific external references |

## Remaining coverage and terminal red-team assessment

External regression, eight-schools and MacroFinance targets remain unavailable.
The simplex posterior adapter, Gandy--Scott optional sequential wrapper,
training-inclusive transport validation, high-dimensional target regimes,
acceptance-screen operating characteristics, broad mutation severity/power
ladders and a replicated stopped-versus-fixed comparison remain open. Generic
power infrastructure can repeat supported engines, but its existence does not
supply power evidence for every engine or defect. Reference-posterior errors
currently use bounded marginals and descriptive means/medians; full covariance,
tail/mode and consumer scientific-function comparisons need declared extensions.
Historical source results cannot fill current-source coverage silently.

Native runner construction emits TensorFlow retracing warnings. Frozen mechanics
and transition checks report bounded traces and the native runner has explicit
signatures. The observed warnings arise while constructing multiple static
runners for different contracts; this is an unresolved compilation-cost issue,
not measured proof of unbounded retracing or a numerical defect. Compilation,
sampling and assessment costs are not yet uniformly separated across all
engines; worker elapsed time is the dependable campaign accounting quantity.

The strongest alternative explanation for favorable results is that the small
fixtures and severe mutations are easier than realistic inference problems.
The weakest evidence is full-pipeline statistical power and stopping coverage.
An independently checked realistic posterior reference, a subtle undetected
mutation, or a calibrated multi-dataset discrepancy would overturn a stronger
correctness claim. No such stronger claim is made here. Failed configurations
motivate targeted repair or additional evidence; they do not reject HMC or the
validation research direction.
