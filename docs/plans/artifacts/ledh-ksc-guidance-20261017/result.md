# KSC future-guidance diagnostic result — 17 October 2026

The shared optional consumer passed the tested KSC likelihood, representation,
reference and analytical finite-program derivative checks. Future guidance did
not establish score improvement: every guided arm failed at least one
conditional descriptive heuristic screen. This rejects promotion of these
candidates, not continuation of the research program.

Plan: [bounded KSC diagnostic](../../ledh-ksc-guidance-audit-20261017.md).
Numerical evidence: [attempt02 result](attempt02/result.json),
[run manifest](attempt02/manifest.json), and
[independent terminal audit](terminal-audit.json).
Branch: `ledh-refactor-with-policy-fix`; numerical base:
`ea1018de1da55ed02a29a8d60f08a6827fce8a42` plus the 72 verified source files.

## Computed target and limits

The target is `sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1`, with
theta=(Phi^-1(gamma), log beta)=(1.5,0), prior N(0,1), process variance one,
and the seven-component observation mixture. This is not the native
stochastic-volatility observation likelihood. T=2, N=32; each of the ordinary,
negative-future-tail and positive-future-tail fixtures has eight paired clouds.

The first guide is the exact two-observation future likelihood, represented
as a Gaussian mixture in state. Its comparator is a moment-matched Gaussian;
terminal guidance is constant. Mixture precision is positive and
responsibility averaged; it is not the negative Hessian of the log mixture.
Guide coefficients remain frozen at the base parameter during differentiation.
The analytical score is the derivative of this same finite frozen-guide value
program, including moving-anchor dependence. It is not a derivative through
refitting, and finite-difference agreement does not identify it with the model
likelihood score.

All eight arms use the same consumer: four unguided identity/global/local/equal
mixture heuristics and four Gaussian/mixture future-guide global/equal arms.
Global/local arms retain 10% predictive density. OT remains epsilon=1 with five
Sinkhorn iterations and chunk=N. Flow/correction controls are untuned KSC
baseline hypotheses. This optional moment-restoring consumer does not establish
canonical UKF/dual-cap/Contract-E admission.

## Checked findings

- 192/192 GPU/XLA evaluations were valid; 16 focused tests passed.
- All 24 centered-difference checks passed at both declared step sizes.
  Maximum analytical derivative absolute error: 6.403e-9, or 0.004906 of the
  applicable tolerance.
- Grid versus exact 49-branch reference differences were at most 1.777e-15
  in value and 1.166e-15 in score, below the predeclared 1e-6/1e-5 bounds.
- The independent standard-library auditor passed pairing, coverage, actual
  reference agreement, derivative checks, all 120 heuristic comparisons and
  all 72 before/after/current source-hash checks. No roundoff warning fired.
- Four guided arms incurred respectively 7, 10, 6 and 11 descriptive heuristic
  losses: Gaussian/global, Gaussian/equal, mixture/global, mixture/equal.
  These are 34 losses across 120 comparisons, not independent statistical tests.

The following selected MSEs show the mechanism problem. They are descriptive
eight-cloud estimates, not a ranking. The first coordinate is Phi^-1(gamma);
the second is log beta.

| Fixed observation pair | Unguided global MSE, first / second coordinate | Exact-mixture future global MSE, first / second coordinate |
| --- | --- | --- |
| Ordinary (0.2,-0.4) | 0.000301535 / 0.0198806 | 0.000175944 / 0.0328146 |
| Negative future tail (0.2,-8) | 0.000511440 / 0.0351947 | 0.000311425 / 0.0477582 |
| Positive future tail (0.2,5) | 0.0265584 / 0.552090 | 0.0250924 / 0.668544 |

In these three global-arm cells, the exact future message has lower observed
first-coordinate MSE and higher second-coordinate MSE. Better representation
alone is therefore not an established repair. The eight-cloud diagnostic
provides no statistically supported ranking or population conclusion.

## Attempts, environment and accounting

The first CPU test run took 29.73 seconds: 14 passed and two new fixtures used
inadmissible zero defensive mass. The fixtures were corrected to the existing
(.1,.9,0)/(.1,0,.9) convention; the guard was unchanged. The rerun passed all
16 checks in 39.50 seconds. Both logs are retained in this directory.

[Attempt01](attempt01/) stopped after 4.2101 GPU wall seconds, before filtering:
lists were passed to a cached reference API requiring hashable arguments.
Both call sites were repaired to use tuples, with no scientific control change.
Attempt02 completed in 40.9831 seconds. Total GPU wall usage is
45.1932/900 seconds and both of the two permitted launches are consumed.
Unused seconds do not authorize a third launch.

The completed run used escalated local GPU access, FP64 GPU/XLA with TF32 off,
and verified memory growth on both physical GPUs. This is a reference exception,
not FP32/TF32 production evidence. The manifest records the exact command,
environment, seeds, data identity, devices, settings and source hashes.
Reported allocator current/peak bytes were 19,712/480,768.

The command was:

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_ledh_ksc_guidance_diagnostic.py --output docs/plans/artifacts/ledh-ksc-guidance-20261017/attempt02 --budget-seconds 450
```

CPU tests, references and terminal audit used approximately 70.12/180 seconds;
test timings are rounded. The terminal auditor itself took 0.02324 seconds.
No process remains running. No additional numerical rerun is required for
closure.

## Decision and inference status

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept tested engineering screen | Likelihood/reference/representation and finite-program derivative checks pass | No continuation veto | Tiny T=2, N=32 reference scope | Preserve executable evidence | General nonlinear or canonical correctness |
| Reject guided-arm promotion | No supported model-score improvement | Every guided arm has a descriptive heuristic loss | Eight clouds; fixed observation fixtures | Diagnose consumer mechanism before attributing failure to fitting capacity | Rejection of all future guidance or KDM |
| Close bounded stage | Two launches used, artifacts audited | No corrupted artifact or numerical failure | Remaining scientific work needs its own protocol | Follow the target-specific protocol below | Automatic transfer of unused GPU allowance |

| Inference item | Status |
| --- | --- |
| Hard veto screen | No numerical/implementation continuation veto; descriptive heuristic promotion vetoes supported |
| Statistically supported ranking | None |
| Descriptive-only differences | All reported MSE and heuristic comparisons |
| Default-readiness | Not established; controls are untuned and consumer is not canonical admission |
| Next evidence needed | Target-specific fitting and consumer-mechanism checks, then disjoint conditional score comparisons with paired uncertainty |

## Mathematical review and next step

The MathDevMCP structural math-to-code audit returned
`structural_mismatch`. Its identifier matcher treated prose words as required
code symbols, so that return is inconclusive as a semantic audit; it does not
prove either correctness or a mathematical mismatch. The original response is
preserved in [the structural audit record](mathdev-structural-audit.json).
The checked convolution derivation, executable representation tests, actual
consumer wiring and analytical finite differences are the supporting evidence
for this bounded result. No formal proof certification is claimed.

The next [target-specific protocol](../../ledh-ksc-fitting-consumer-protocol-20261017.md)
separates consumer failure from representation/fitting failure. It must not
assume that fitting a more flexible message improves the score when even the
exact future message failed the current screen. Fitted cSMC/KDM nonlinear
comparison, full canonical lifecycle, scope tuning and untouched claim runs
remain open.

Post-run red team: the strongest alternative explanation is finite-particle,
reset or proposal-geometry error in this untuned consumer, rather than a bad
future message. Fresh conditional comparisons with adequate uncertainty could
overturn the descriptive losses. The weakest evidence is the eight-cloud
score comparison; the strongest evidence is finite-program and reference
consistency on the declared fixtures.
