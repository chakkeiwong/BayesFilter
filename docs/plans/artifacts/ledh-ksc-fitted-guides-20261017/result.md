# KSC fitted future guides: completed comparison

17 October 2026. Worktree ledh-iapf-execution-20260922, branch
ledh-refactor-with-policy-fix, HEAD ea1018de1da55ed02a29a8d60f08a6827fce8a42
plus the source changes frozen in the two launch manifests.

## Question and answer

Does target-specific future-message fitting improve likelihood-score estimation
in the existing LEDH proposal-mixture consumer? Neither tested fitted family
qualifies for promotion. No quadratic configuration passed eligibility in all
development conditions. Both fitted-kernel arms failed the conditional
heuristic screen; no prespecified interval supports score-MSE improvement.
Three simultaneous intervals support increased MSE for the global kernel arm.
These are failures of the tested candidates, not a rejection of backward
guidance as a research direction.

The target is sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1, not the native
stochastic-volatility observation likelihood. The parameters are
(Phi^-1(gamma), log(beta))=(1.5,0), Q=1, prior N(0,1), T=2, N=32. Ordinary,
negative-future-tail and positive-future-tail datasets are respectively
[0.2,-0.4], [0.2,-8] and [0.2,5]. The proposal, denominator, OT (1,5), reset and
analytical derivative remain the shared diagnostic consumer. A local positive
kernel fit is not original KDM training; using a fitted controlled-SMC
potential here is not running the full twisted-SMC recursion.

## Development and untouched final comparison

Development evaluated 120 fits using disjoint calibration and validation seeds.
No quadratic setting had twelve eligible fits: sixteen otherwise valid
controlled-SMC fits had nonpositive guide precision, and four fits were invalid
(NaN). No clipping or replacement seeds were used. The kernel setting with
256 design points, sixteen transition replicates and bandwidth factor 0.5 was
selected by validation cross-entropy, then frozen. Its mean excess
cross-entropy was 0.0384557; this is a fitting criterion, not score evidence.

Final-02 preserves all 2,304 cells: 1,920 valid evaluations and 384 explicitly
ineligible quadratic cells. There are eight independent fitting clusters, each
with eight paired clouds per condition. The 24 planned contrasts retain their
Bonferroni multiplicity even though twelve quadratic contrasts are unavailable.
Intervals use Student t with seven degrees of freedom on cluster-level paired
MSE differences. They are conditional on the frozen development selection and
three fixed datasets; with eight clusters, approximate normality is a material
limitation.

Positive differences below mean worse MSE than unguided/global:

| Condition | Coordinate | Kernel/global MSE difference | Simultaneous 95% interval |
|---|---|---:|---:|
| Ordinary | log beta | 0.16952125 | [0.05189683, 0.28714567] |
| Negative future tail | Phi^-1(gamma) | 0.00274671 | [0.00032723, 0.00516620] |
| Negative future tail | log beta | 0.26121763 | [0.10898194, 0.41345332] |

All nine other available intervals include zero. The global and equal-mixture
kernel arms respectively lose 23 and 20 descriptive comparisons against the
constructed conditional heuristic set. Those losses veto promotion under the
predeclared screen; they are not individually established statistical rankings.
The complete conditional table and contrasts are in final-02/analysis.json.

## Decision table

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not promote quadratic settings | Unavailable | No configuration eligible across development conditions | Fit failure versus positive-information restriction | Diagnose the rejected precision and the consumer restriction | Quadratic control is impossible |
| Do not promote kernel/global | Fails all-condition improvement requirement | 23 descriptive heuristic losses; three supported MSE increases | Eight clusters and three fixed datasets | Inspect how mixture information enters proposal covariance | Original KDM is ineffective |
| Do not promote kernel/equal | No supported improvement | 20 descriptive heuristic losses | Intervals include zero | Same consumer diagnosis, without final-data tuning | Equivalence to unguided proposals |
| Continue mechanism investigation | Numerical/reference audit passes | No continuation veto | Exact-message controls also have heuristic losses | Derive and check the information approximation before another campaign | Default, canonical LEDH or HMC readiness |

## Inference status

| Evidence | Status |
|---|---|
| Hard veto screen | Ineligible quadratic cells preserved; accepted evaluations pass reference, derivative, trace and provenance checks |
| Statistically supported ranking | Three conditional kernel/global MSE increases; none supports improvement under the specified simultaneous intervals |
| Descriptive-only differences | Heuristic losses, exact-message comparisons, fit loss, ESS and runtime |
| Default-readiness | Not evaluated; OT and all other controls are untuned development settings |
| Next evidence needed | A justified consumer repair, fresh scope-specific tuning and untouched conditional score comparisons, then canonical validation |

## Engineering and numerical verification

The independent terminal audit passes: 11,520 trace checks, twelve
frozen-coefficient finite-difference checks, 72 unchanged source hashes, 192
unique paired clouds, and independent selection/reference checks. Maximum
finite-difference error is 1.3918e-8; reference likelihood and score differences
are at most 1.7764e-15 and 1.1657e-15. These checks establish agreement with
the finite-program analytical derivative, not equality to the exact model score.
Focused adapter tests pass 25/25, analysis tests 7/7 and CPU-XLA smoke tests 2/2.

Four engineering attempts repaired callback mixture dimensions, premature
float32 scalar conversion, and the test/diagnostic cumulative-logZ shape.
Their failures and regressions are preserved in focused-tests-01 through -04.
No scientific threshold was relaxed. See terminal-audit.json and its log for
the independent final checks.

## Execution and budget

Both runs used the tftwogpu environment, trusted GPU 0 (RTX 4080 SUPER), FP64,
XLA, TF32 disabled and verified memory growth. This is a reference/diagnostic
exception, not production TF32 evidence. Each manifest preserves commands,
environment, settings, seeds, source hashes, hardware, timing and output paths.

- Development: scripts/run_ledh_ksc_fitted_guides.py --stage development
  --output docs/plans/artifacts/ledh-ksc-fitted-guides-20261017/development-01
  --budget-seconds 450; 24.596142773 GPU-process wall seconds.
- Final: the same runner with --stage final, output final-02, and
  --frozen development-01/frozen-selection.json (full paths in the
  manifest); 212.319763199 GPU-process wall seconds.
- Analysis: scripts/analyze_ledh_ksc_fitted_guides.py; independent audit:
  scripts/audit_ledh_ksc_fitted_guides.py. Full invocations and timing are in
  analysis-01.log and terminal-audit-01.log.

Total GPU-process wall time is 236.915905972/900 seconds. Both permitted
launches were used; the campaign is closed even though time remains. The CPU
reference/analysis/smoke ledger remains below 20/180 seconds. No third launch,
final-data retuning, canonical admission, default change, commit or push follows.

## Post-run red team and next action

The strongest alternative explanation to inadequate fitting capacity is the
consumer's positive-information approximation, rather than the fitted density
itself. Accurate density fitting does not by itself establish an accurate local
quadratic approximation. Exact future-message controls also fail heuristic
screens, so another capacity ladder is not yet justified. Inspect the actual
guide-to-global/local-map call chain and derive the difference between its
information matrix and the log-message Hessian. A deterministic saved-fit
diagnostic can localize that difference without rerunning or retuning final
clouds. It cannot establish that this mechanism caused all observed score error.
Fresh independent downstream comparisons could overturn the candidate rejection.
