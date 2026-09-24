# Pinned iAPF score confirmation and fresh-data fitting audit

2026-09-19. Owner authorization: “I approve the budget and continue the
execution.” This opens a new bounded campaign; the previous 8000-call campaign
remains closed. Codex executes and self-reviews; no independent review is claimed.

## Question and sequence

Does the Gaussian-innovation correction retain its physical model-score
accuracy on the intended RTX5080, and does it survive fresh observations and
fresh proposal/control fitting? The target is the observed-data score, estimated
by a normalized complete-data Fisher statistic. The finite fixed-label
likelihood derivative remains a separate diagnostic. No HMC force is created.

1. **Pinned confirmation:** keep datasets 1490/1500/1501/1510/1511 and their
   frozen proposal fits, but use 96 fresh calibration and 64 fresh final streams
   per dataset. Retain matched ancestor-only controls and the earlier frozen
   192-calibration ancestor coefficients. This repeats the conditional scientific
   comparison on the intended device; it is not a cross-device parity test.
2. **Fresh data:** use previously unused dataset IDs 1900/1901 (weak) and
   1910/1911 (curved). Fit each proposal independently of all score calibration
   and final streams. Use the existing `floor_001` configuration only as an
   explicitly inherited baseline. Run the same 96/64 paired comparison with
   matched ancestor controls, EKF, UKF and no-resampling scores. Record every
   fitting bound contact, convergence diagnostic, realized particle count,
   and backward-reference shape discrepancy before final comparisons.
3. **Bound sensitivity, if any fresh baseline fit touches a bound:** on all four
   fresh datasets, evaluate the predeclared nested standardized fitting box:
   mean bound 4 -> 8, SD bounds [0.2,4] -> [0.1,8]. Reuse the same offline fit
   streams for a paired sensitivity check, but use fresh control calibration
   and final streams. Freeze this action based only on offline bound contacts,
   before inspecting score comparisons. Retain the original results unchanged.
   Interior baseline fits are non-harm controls; bound-active fits test whether
   an arbitrary search boundary caused the restriction. No further widening,
   floor adjustment or final-data tuning is authorized by this plan.

The particle model is scalar sine transition/quadratic observation, T=2 and
N=4096. Physical parameters, reference refinement, analytical particle kernels,
ancestor sampling law, and FP32/TF32/XLA scope stay fixed. This is a local
diagnostic extension, not author-code equivalence or canonical LEDH admission.

## Evidence contract and intent ledger

Primary comparison in each stage: corrected Fisher score with 18 controls
versus the matched 12 ancestor controls, using paired final squared error to
the refined FP64 quadrature score. Each stage separately uses four 99.75%
percentile bootstrap intervals (40000 resamples; approximate Bonferroni 99%
family). A negative upper endpoint in all four comparisons passes that stage.
Stage families are separate questions; no campaign-wide multiplicity claim.
Intervals condition on each realized proposal fit, coefficient fit and dataset;
they do not integrate calibration or population uncertainty.

* Promotion vetoes: any observed loss to the constructed heuristic set within
  a dataset; any offline bound contact; failed mean-bias screen; unresolved
  hardware identity. Failure of these screens blocks promotion, not the next
  discriminating repair. A non-significant interval is inconclusive.
* Continuation vetoes: invalid/unconverged fit that cannot yield a valid score,
  nonfinite numerical results, failed reference refinement, source/input drift,
  seed overlap except declared paired fitting, wrong GPU, missing artifacts,
  or an exhausted resource ceiling. Local infrastructure failures permit a
  recorded repair/retry under the unchanged ceiling.
* Repair trigger: an offline fitting bound contact triggers stage 3 regardless
  of score MSE. A failed score comparison does not authorize changing bounds
  or tuning controls on final streams.
* Explanatory diagnostics: variance, likelihood errors, ESS, ranks/singular
  values, coefficient size, shape residuals, runtime and device allocation.
  They cannot replace final score MSE.
* Heuristic adversaries: EKF (local linearization), UKF (nonlinear moment
  propagation), and no resampling (removes ancestor randomness). On the affine
  calibration dataset add exact Kalman, whose superiority remains a veto to
  any general superiority claim. Evaluate each fixed weak/curved dataset
  separately and preserve every failure prominently.
* Not concluded: unbiased finite-N scores, general iAPF/KDM superiority,
  population performance, long-horizon/multidimensional performance, tuning
  admission, a new default, LEDH conformance or HMC readiness.

The correction and its same-law reference centering are unchanged from the
[derivation](younis-iapf-innovation-control-2026-09-18.md): six first/second
Gaussian moment contrasts, 16 independent reference clouds, and a frozen
linear projection outside the normalized ratio. It preserves the underlying
finite-N bias, subject to the stated sampling and arithmetic assumptions.

## Assumptions and skeptical audit

| Choice and status | Provenance and justification | Failure risk and early check |
|---|---|---|
| Fixed scalar T2/N4096; diagnostic scope | Previous stage enables controlled replication | Narrow scope; fresh observations precede horizon expansion |
| 96/64 streams; bounded replication | Previous design permits a direct repeat | Noisy calibration; separate paired intervals and stored coefficients |
| 16 reference clouds; derived precision choice | Adds 1/16 to control-moment variance | Shared randomness; reserve disjoint generator seeds |
| FP64 control fit with FP32 rank cutoff; checked safeguard | Previous rank regression and finite-support tests | Spurious null directions; ranks, singular values and affine control |
| Proposal fit N16, cap128, k1, tau100, four iterations; inherited baseline | Existing explicit iAPF adapter configuration | Small cloud/early stopping can fit poorly; report cloud count, all optimizer diagnostics and oracle shape errors; no tuned-fit claim |
| Relative-shape objective, floor .001; inherited hypothesis | Prior diagnostic nomination, not an author/default claim | Floor/shape misspecification; preserve floor fractions and downstream error |
| Original fitting box; inherited baseline | Bounds are on cloud-standardized means and log SDs | Artificially constrained optimum; inspect every offline contact |
| Nested box; conditional safety candidate | Factor-two expansion in standardized scales tests boundary sensitivity while preserving a finite domain | Different optimizer basin/overfit; paired fitting seeds, interior non-harm controls and untouched final checks |
| No extra ridge/damping | Frozen least-squares projection with input-precision rank handling | Calibration overfit; independently frozen application and intervals |
| RTX5080 UUID pin; required scope | Earlier correct run manifest identifies physical device | Ordinal reordering; pin before TF import, assert one GPU and matching details, save output devices |

Bound expansion is not accepted merely for lowering MSE. Its safety question is
non-harm on interior fits (relative coefficient discrepancy <=2e-4) and valid,
flagged behavior where a baseline contacts a bound. The tolerance is a
diagnostic comparison allowance above FP32 cast precision, not an admission
threshold. Persistent contacts or unconverged fits reject it as a complete
bound repair. A larger cloud or a different proposal family would need its own
subplan. Finite-box sensitivity cannot prove an unconstrained optimum.

Pre-mortem: a repeated fixed-data success could hide dataset-specific fitting;
fresh observations address that next. A relaxed box could simply overfit a tiny
cloud; interior invariance, reference-shape errors and new final streams expose
that. Error-screen failures are evidence about this candidate, not invalidity
of Fisher's identity or rejection of the iAPF/LEDH research direction.

Skeptical audit PASS before implementation: exact target and matched baselines
are preserved; no variance or fit objective substitutes for final accuracy;
device identity is now explicit; all post-selection uncertainty limits are
stated; bound sensitivity is triggered by fitting information alone. Existing
historical final streams are excluded. Implementation must retain the shared
numerical kernels and snapshot the exact source used by each attempt.

## Environment, budget and artifacts

Checkout `/home/chakwong/BayesFilter`, branch `surrogate-hmc`, starting commit
`6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef`; preserve unrelated changes.
Python `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`. All GPU commands
use escalation, `TF_FORCE_GPU_ALLOW_GROWTH=true`, and physical UUID
`GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`. Verify memory growth before device
initialization; record TF32/XLA, actual tensor devices and allocator peak.
CPU-only focused tests explicitly set `CUDA_VISIBLE_DEVICES=-1`.

New campaign ceiling: **4 launches, 8000 charged filter calls, 8 adaptive fits,
1800 cumulative driver seconds, 600 CPU test/probe seconds**. Stage 1 needs
1116 calls; each four-dataset stage needs 944 plus at most 32 reserved fitting
calls, totaling at most 3068 for three stages. The fourth launch and remaining
calls are available for localized repair, not an unplanned sweep. Earlier
campaign charges are archived separately. Unspent resources are not a reason
to run unnecessary comparisons.

Output root: `docs/plans/artifacts/younis-iapf-pinned-continuation-20260919-01/`.
Every attempt gets a new directory, manifest, seed inventory, frozen calibration,
raw results, decision, source snapshot, full log and post-run interpretation.
Commands use `docs/benchmarks/diagnose_younis_iapf_innovation_control.py`
with `--continuation-stage pinned|fresh|wide` and `--output <unique-directory>`.
Preflight and focused CPU checks precede launch. At each completed stage update
the master/checkpoint; after the campaign write decision/inference tables and
the next smallest discriminating task. No repeated human approval is needed
within these limits.

## Terminal review, 2026-09-19

All three planned stages executed. [Result and audit](artifacts/younis-iapf-pinned-continuation-20260919-01/result.md)
confirm the intended single RTX5080 scope. Pinned fixed-data confirmation passes
4/4 primary comparisons; fresh observations and wider-box sensitivity each pass
3/4. Both fresh weak datasets lose to UKF; one fit still contacts a bound.
Widening is rejected as a complete fitting repair, while passing the narrower
interior non-harm check (three fitted proposals exactly unchanged).

Terminal code/math review found that inherited k=1/tau=100 cannot establish
convergence: a two-value sample CV is bounded by sqrt(2). Six independent
analytic cases reproduce this at the actual controller. The pre-run audit
flagged early stopping but missed this exact range defect. No settings were
changed after examining final errors; the finding motivates the next protocol.
See [the derivation and proposal diagnosis](artifacts/younis-iapf-pinned-continuation-20260919-01/fitting-diagnosis.md).

Use: 3/4 launches, 3068/8000 charged calls, 8/8 fits, 137.859234/1800 driver
seconds; conservative 60/600 CPU test/probe seconds. Eight focused CPU tests,
source snapshots, frozen-calibration checks, independent score arithmetic,
refined references and device checks pass. This campaign is closed because
its planned stages are complete and its fitting allowance is exhausted.
Unused launch/call capacity is not a new fitting allocation. The next task is
a larger-cloud, informative-stopping protocol with independent calibration and
untouched validation, retaining UKF and preserving all current failed finals.
No numerical/default/HMC/LEDH promotion; whole-master work remains.
