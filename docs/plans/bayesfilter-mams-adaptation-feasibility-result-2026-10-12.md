# Bounded generic MAMS adaptation: terminal feasibility result

Date: 12 October 2026. This closes the bounded D4R3 feasibility execution in
[the validation plan](bayesfilter-mams-adaptation-bounded-validation-2026-10-11.md),
not D4R4 state-space confirmation. The question was whether the same discarded
MAMS proposer reaches the public tuner, retains and reloads every verified
member, and assesses posterior evidence separately across contrasting targets.
The frozen-grid SGQF cohort is development context, not a cost-matched
comparator. NUTS is excluded.

## Result and provenance

The GPU/XLA run completed eight targets with **25/25 verified members exported,
reloaded and posterior-assessed**. Twenty-one passed the deliberately short
local diagnostic policy; four did not complete that policy. No member was
removed from tuning for R-hat, ESS, MCSE or reference behavior, and no
numerical hard veto was reported. One seed per target supports no ranking.

| Target | Verified / created | Short local checks passed | Reference or incomplete check |
| --- | ---: | ---: | --- |
| Isotropic Gaussian | 4 / 4 | 4 | Exact moments reported descriptively. |
| Rotated Gaussian | 2 / 2 | 0 | Both warm-up checks inconclusive at 1,024 transitions per chain; no retained draws. |
| Student-t(5) | 2 / 2 | 2 | Finite exact moments reported; tail delivery not established. |
| Exact noncentered funnel | 4 / 4 | 2 | One warm-up-inconclusive member; another met precision but not final R-hat at the retained cap. Model-coordinate tails remain open. |
| Analytic residual funnel | 2 / 2 | 2 | Local checks passed under partial whitening; no global guarantee. |
| Unequal separated mixture | 4 / 4 | 4 | **All four missed the left mode:** observed mass 0 versus exact mass about 0.3000001. Local checks falsely reassure about global coverage. |
| LGSSM filter adapter | 3 / 3 | 3 | Actual filter/score public handoff exercised; no independent posterior reference here. |
| Nonlinear accumulation filter adapter | 4 / 4 | 4 | Actual approximate filter/score route exercised; no independent posterior reference or exact nonlinear-likelihood claim. |

The short policy used 512–1,024 warm-up and 256–1,024 retained transitions
per chain. The two rotated-Gaussian and one exact-funnel cases hit the warm-up
cap. The other exact-funnel case met lugsail precision but had final maximum
R-hat about 1.01498 against the diagnostic retained threshold 1.01. These
are posterior outcomes, not failed tuning qualification. The mixture's mode
discrepancy is independent reference evidence even though local checks passed.

The exact launch was:

```bash
env CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/validate_mams_adaptation.py \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/gpu-feasibility-01 \
  --targets gaussian rotated_gaussian student_t funnel_noncentered residual_funnel mixture LGSSM NL-ACC \
  --seed 411 --max-wall-seconds 7200
```

It ran from immutable `source-0abad7abe` at commit `0abad7abe` using Python
3.13.13, TensorFlow 2.20.0 and TFP 0.25.0. Trusted GPU 1 was an RTX 4080
SUPER with memory growth verified before initialization; XLA and TF32 were
enabled. Terminal audit checked 174 source/operator SHA-256 values, the Git
tree, every candidate/member/receipt and 325 recorded seed pairs for reuse.
All eight cases and the supervisor finished normally. The pilot used 148,468
actual value/score/status row evaluations; qualification conservatively
charged 5,210,112. These are distinct accounting measures. The worker elapsed
719.775 seconds; the enclosing supervisor elapsed 722.673 seconds, the campaign
charge. The [manifest](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/gpu-feasibility-01/manifest.json),
[aggregate result](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/gpu-feasibility-01/result.json)
and [terminal audit](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/gpu-feasibility-01/terminal-audit.json)
link to versioned per-target tuning and posterior records.

## Checks, budget and decision

The isolated committed-source regression suite passed **99 tests**. A
separate CPU-only suite passed **42 tests** in 35.40 seconds: independent
LGSSM and nonlinear adapter references, corrected capability assertions, and
seven positive/mutated MAMS oracle and stream-isolation checks. GPUs were
hidden before TensorFlow import. Earlier committed tests had four stale
capability expectations; the corrected suite has zero failures. An optional
uncommitted generic full-chain fixture could not import its separate untracked
`ssm_targets.py`; it produced no numerical result and is not evidence for
this source commit. The actual LGSSM and nonlinear accumulation GPU/XLA
public pipeline completed in the eight-target run. Reports under
`N5/generic-adaptation/checks-01/` preserve both test attempts.

The previous settled campaign balance was 158,333.764659 aggregate
worker-seconds. Charge the full 3,600-second local-work reserve and
722.673105 seconds of supervisor time, including its worker; release unused
time from the 7,200-second stage reservation. The settled balance is
**154,011.091554 seconds**, about 42.78 aggregate worker-hours, with the
prior overhead reserve unchanged. This is no new allocation. The mixture
result is a reference/coverage failure and next-design trigger, not an
invalid artifact or continuation veto. The optional fixture import failure
is test-assembly debt, not an observed MAMS numerical defect.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept bounded engineering feasibility; keep MAMS experimental | Public handoff and denominators passed on all eight targets | No shared numerical/source/stream defect; mixture vetoes global-coverage claim | One seed, short chains, missing state-space posterior references | Price D4R4 against independent references and matched classical baselines | Robust tuning, better mixing, posterior correctness, release/default readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No numerical hard veto in 25 members; mixture's mode reference vetoes a global-coverage claim. |
| Statistically supported ranking | None; no paired comparison or uncertainty interval. |
| Descriptive-only differences | Local check counts, IACT hypotheses, costs and exact-moment differences. |
| Default-readiness | No; optional experimental route only. |
| Next evidence needed | Independent model/observable references, full posterior allocations, searched fixed/randomized/SNAPER comparators, fresh paired replications and uncertainty analysis. |

The strongest alternative explanation for 21 local passes is local-basin
confinement; the mixture demonstrates it. A fresh dispersed-start study with
independent references could overturn apparent local success. The current
result does not invalidate the implementation, target, source snapshot or
research direction. It shows that acceptance and local convergence screens
cannot establish global coverage. D4R4 must keep that discrepancy visible
without tuning on this negative-control result and calling the same data
untouched confirmation.

## Refreshed next phase

The full nonlinear comparison cannot be promoted from these short timings.
Read-only D4R4 preflight on 12 October found that all four saved full-model
posterior-reference loaders reject the current checkout: each archived source
closure includes five changed inference/tuning files (`__init__.py`,
`hmc_candidate_set_artifacts.py`, `hmc_candidate_set_execution.py`,
`hmc_candidate_set_tuning.py`, `tuning_contract.py`). Their recorded input
files and target-source files still match. Three saved local-geometry source
closures also include a changed geometry diagnostic script. Inspection of the
five common diffs found new MAMS exports, repair records and controller rules,
not an altered reference integrand. Four current CPU parity regressions passed
against independent full-horizon value/score calculations (PP-UKF, PP-SGQF,
SIR and structural UKF), but finite probe points alone cannot reissue the
old reference artifact. The prior reference directories remain historical
until a versioned source-compatibility bridge with full dependency audit is
checked, or new references are computed. This is a provenance gate for
posterior comparison, not a reason to reject cost-only pilot pricing.

The old full-model MAMS driver also uses a nine-point frozen grid, an inherited
epsilon upper bound of 1 and zero repair descendants. It cannot be relabeled
as execution of the new adaptation. A new full-model consumer must call the
shared discarded proposer and public candidate-set tuner, while treating any
saved local factor only as a named warm-start geometry hypothesis.

Before another serious launch, D4R4 must (1) assemble a committed source
closure for SGQF, UKF, SIR and structural adapters; (2) check each finite
target and score against its independent reference; (3) specify transformed
parameters, tails, predictions and global-coverage checks; (4) price pilot,
qualification, warm-up and retained work on GPU/XLA for adaptive MAMS and
adequately searched fixed/randomized/SNAPER baselines; and (5) freeze a
paired-seed matrix, uncertainty procedure, complete denominators and total
suballocation fitting the settled 42.78-hour balance. K1/K7 state-space
references and rotated/funnel/mixture negative controls remain in scope. If
the full matrix does not fit, narrow the claim and record unfunded cells
before launch. Historical grid scripts are archival, not adaptive evidence.
