# KSC T2: actual surrogate-HMC test

The full canonical DCTR surrogate-force arm completed independent public
tuning and 2,000 retained draws per chain, across four chains. **B/R is
INCONCLUSIVE.** Every retained chunk passed endpoint, finite-value, energy,
movement and divergence checks. Three functionals failed the predeclared
batch-dependence screen at the final look. These results do not establish
posterior equivalence or a posterior discrepancy. They do establish execution
of an actual surrogate-HMC test on the declared KSC scope.

The research question remains whether endpoint-corrected HMC with an imperfect
full canonical DCTR position-only proposal field can recover model-parameter
posteriors, and at what computational cost. Gradient disagreement is not a
continuation veto. The independent C/D sampler question remains open.

## Target and experiment

This is model M03's existing KSC Gaussian-mixture observation law at T=2,
with stationary AR(1) state, innovation scale 1, parameters gamma and beta,
independent Uniform(0.1,0.9) priors, and the probit chart
theta=0.1+0.8 Phi(q). It is a distinct model from the exact transformed
actual-SV observation law. The exact likelihood sums all 49 mixture paths.
An independent FP64 reference integrates the full proper parameter prior.
Its 13 functionals cover five standardized moments, six marginal CDF values,
and two probabilities for the next observation. Four parameter-grid
refinements passed the frozen precision and contraction screens. The numerical
allowances are empirical refinement estimates, not rigorous global bounds.

A uses the exact mixture force and B uses the full canonical analytical DCTR
force. Both use the same exact-mixture endpoint density, prior, Jacobian and
FP64 integrator. DCTR uses N=1024, K=1024, FP32/TF32, Contract E–Chol,
GenUT, both trust-region caps and the UKF per-particle covariance lifecycle.
The consumer calls the shared canonical batch implementation; 18 GPU mechanics
checks include call-chain parity, replay, reversal and endpoint/force
independence. The DCTR field is not admitted as an exact score.

Disjoint calibration and validation selected the protection multiplier 0.1.
The untouched nominal claim data and all controls were frozen before inference.
Each arm independently used the public typed position-field tuner, followed
by 250 discarded transitions and 1,000 fresh verification transitions per
chain. All 12 verification checks passed for both arms. Warmup and verification
draws are archived and excluded from the eight retained 250-draw chunks.
The only posterior looks were 1,000 and 2,000 retained draws per chain.

| Arm | Nominated epsilon | Leapfrog steps | Final parameter R-hat maximum | Minimum bulk / tail ESS | Final uncertainty screen | Posterior verdict |
|---|---:|---:|---:|---:|---|---|
| A, exact force | 1.1691828003 | 1 | 1.001649 | 3274.50 / 4123.51 | Passed | INCONCLUSIVE |
| B, full DCTR force | 1.1768948616 | 1 | 1.000464 | 3324.45 / 4359.50 | Failed batch-dependence screen | INCONCLUSIVE |

These diagnostics are descriptive. At B's final look the largest-batch lag
correlations were −0.454877 for standardized gamma, −0.430211 for gamma's 10th
percentile CDF, and −0.442377 for its median CDF, exceeding the absolute 0.3
limit. MCSE sensitivity and R-hat screens passed. The first look also failed
the dependence screen, then on beta's 90th-percentile CDF. No screen was
relaxed after observing these results. Nine B intervals lie within their
margins arithmetically, but the unqualified uncertainty estimates cannot
establish equivalence. A's final uncertainty screen passed; seven of its 13
intervals still did not fit the margins. No qualified interval demonstrated
a discrepancy.

## Decisions and inference status

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| B/R remains inconclusive | Simultaneous equivalence not established | All health checks passed; three function MCSE dependence screens failed | Reliable interval coverage and residual interval width | Preserve both looks; diagnose the dependence screen under a prospective repair plan before any new inference attempt | Neither posterior equivalence nor rejection of surrogate HMC |
| A/R comparator remains inconclusive | All13 equivalence intervals not inside margins | Final health/mixing/uncertainty screens passed | Monte Carlo interval width | Preserve this completed comparator attempt | No comparative ranking |
| C/D remains NOT_RUN | Independent finite-D reference and C tuning absent | No C continuation veto evaluated | Finite-target sampler behavior | Qualify the finite endpoint and independent D reference, forecast the complete C schedule, then test C/D within the campaign allocation | D/R accuracy cannot decide C/D |
| D/R and C/R remain NOT_RUN | Required finite-target evidence absent | Not evaluated | Finite-target bias and combined inference | Keep these separate when D and C become available | No finite-DCTR posterior accuracy claim |

| Inference question | Status |
|---|---|
| Hard veto screen | No state/target/energy/movement/divergence failure. B's MCSE qualification failed at both looks; this vetoes an equivalence conclusion, not the research direction. |
| Statistically supported ranking | None. |
| Descriptive-only differences | R-hat, ESS, interval positions, selected kernels and costs from this one nominal dataset. |
| Default-readiness | Not established; all-model validation is false. |
| Next evidence needed | Qualified uncertainty under a prospective protocol, independent C/D testing, then untouched replication, longer horizons and conditional heuristic comparisons. |

Engineering correctness, numerical validity and scientific inference have
different evidence. Focused reference/consumer tests and mechanics checks
support this execution path. The archived chains support healthy numerical
execution. The posterior inference verdict remains inconclusive.

The constructed heuristic set is the prior-center plug-in prediction, prior
predictive integration and the existing moment-collapsed mixture UKF posterior,
evaluated separately in nominal and stress situations. Those comparisons are
NOT_RUN. This result supports no usefulness or default claim.

## Execution, recovery and budget

The [prospective plan](surrogate-hmc-ksc-t2-2026-09-21.md) contains the evidence
contract, assumptions, audit, fixed looks and reservation. The
[run manifest](artifacts/surrogate-hmc-all-models-20260921-ksc-t2-inference-01/run-manifest.json)
records the exact command, git commit, all 78 source hashes, data and random
seeds, memory-growth verification and hardware. The
[report](artifacts/surrogate-hmc-all-models-20260921-ksc-t2-inference-01/report.json)
contains all look diagnostics. Source snapshots, tuning archives and every
retained chunk are preserved under that same versioned root.

The executed public command was:

```bash
CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b \
TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py \
  --stage P3-KSC-inference --scope KSC \
  --qualification-root docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-qualify-01 \
  --selection docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-controls-01/selection.json \
  --posterior-root docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-posterior-01 \
  --cost-root docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-cost-01 \
  --output docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-inference-01
```

Execution used escalated access to GPU1, RTX4080SUPER, with verified memory
growth, XLA and TF32. The process used four allowed CPU cores. This first KSC
inference attempt consumed 1,124.504461 CPU-worker seconds and 1,169.190838 GPU-wall
seconds against a 3,450/3,600 reservation. TensorFlow's recorded allocator peak
was 19,724,032 bytes; this is allocator telemetry, not total device reservation.

Before inference, controls01 completed all 16 nominal/stress calibration and
validation evaluations and issued the checked selection, then exceeded its
250-second CPU limit while compiling the transition-cost check. Its actual
269.619530 CPU seconds are charged and the SIGXCPU failure is preserved.
Recovery verified all selection/source/data/curve hashes, measured cost in a
separate bounded worker, and added four-core affinity plus a live CPU/wall
watchdog. Cost-only recovery and posterior preparation passed. This was an
infrastructure repair, not a rejected posterior candidate.

The predeclared 2,200-second CPU redistribution increases P3 from 10,800 to 13,000
and reduces P5 to 8,600, leaving the 54,000 CPU-worker / 226,800 GPU-wall campaign
caps unchanged. KSC's family ceilings are 4,000/7,200 seconds. The original
execution plan's 12,800 P3 subtotal was an arithmetic typo, corrected after the
run. It did not affect the enforced family, campaign or job limits. Through
inference, KSC used 1,641.962543 CPU / 1,619.123895 GPU seconds; subsequent
reporting and regression checks are charged separately in the campaign ledger.

## Program review and follow-up

The review found that the standalone public KSC command executed correctly but
the generic all-model report omitted its completed evidence. The master now
reads the preserved KSC attempt, checks archived sources/data/samples and
continuation links, and retains each comparison's actual verdict. Reusing a
closed attempt never authorizes a new sampling look. The model register marks
KSC coverage as partial and preserves the unresolved exact-transformed variant.

Other repairs reject an incorrectly named selection artifact and incompatible
per-stage CLI overrides, forward and deduplicate additional accounting records,
check all 13 reference precision records and the four refinement grids, validate
timing records, distinguish retained execution from a tuning-only attempt,
and report the active sampling phase accurately. These changes were made
after the frozen inference run. They do not rescore its draws or change any
kernel, endpoint, margin or scientific verdict.

The final reporting check also found that P0 discovered the completed LG1 and
C01 attempts but omitted their conclusions. The reader now loads those records
and verifies the linked finite-reference checksum. A changed reference fails
closed. The first reporting output is preserved. The corrected
[all-model report](artifacts/surrogate-hmc-all-models-20260921-ksc-t2-master-02/decision.json)
retains LG1's D/R discrepancy, all inconclusive conclusions, and the incomplete
model coverage. It launched no workers and created no new posterior looks.

The earlier 81 focused regressions passed; after the reporting repair,
[59 affected regressions](artifacts/surrogate-hmc-all-models-20260921-ksc-t2-tests-09/pytest.log)
passed. Intermediate test runs exposed two fixture mistakes, which were fixed;
all attempts remain charged. Through the corrected master report, campaign use
is 12,696.313770 CPU-worker / 11,029.423012 GPU-wall seconds. KSC use is
1,744.545861 / 1,619.123895, leaving 2,255.454139 / 5,580.876105 in its allocation.

The strongest alternative explanation for B's inconclusiveness is a noisy
fixed batch-correlation screen: only ten largest batches per chain enter it,
and marginal rank/ESS diagnostics were healthy. That explanation is not
established by these chains. Conversely, healthy marginal diagnostics cannot
exclude residual dependence or missed mixture behavior. A prospective
calibration using both false-veto and false-pass controls, followed by untouched
inference, could distinguish these explanations. Existing diagnostic-repair
work is not permission to retrospectively replace the frozen screen.

The weakest scientific evidence is one short nominal dataset and an empirical
reference error allowance. The next independent sampler comparison is C/D;
its preparation is not cancelled by B's uncertainty failure, nor would a
future D/R discrepancy cancel it. Any new inference attempt must retain the
campaign's attempt accounting, independent data policy and full-look forecast.
