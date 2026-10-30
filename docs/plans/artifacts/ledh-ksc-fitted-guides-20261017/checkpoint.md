# KSC fitted-guide checkpoint — 2026-10-17

- Checkout: ledh-iapf-execution-20260922, branch ledh-refactor-with-policy-fix,
  HEAD ea1018de1da55ed02a29a8d60f08a6827fce8a42. Preserve unrelated dirty work.
- Active question: why does a fitted future-message representation fail to
  improve scores in the existing LEDH proposal? Separate fitting error from
  the positive-information approximation used by that consumer.
- Completed: development-01 contains 120 fits. Kernel n256 / bandwidth factor
  0.5 was selected on independent validation and frozen before final-02. No
  quadratic configuration was eligible in all twelve development fits: sixteen
  otherwise valid fits had nonpositive guide precision and four were invalid.
  Do not clip, replace seeds, or tune on the final score results.
- Final-02: 2,304 cells: 1,920 valid evaluations and 384 explicitly ineligible
  quadratic cells. Eight independent fit clusters each have eight paired clouds
  per condition. Both fitted-kernel arms fail the conditional heuristic screen.
  Of twelve available prespecified simultaneous contrasts, none supports
  improvement and three support increased MSE.
- Independent terminal audit PASS: 11,520 trace checks, twelve frozen-guide
  finite-difference checks, 72 unchanged source hashes, and reference/pairing/
  selection checks. Maximum finite-difference error is 1.392e-8. This checks
  the finite-program derivative, not equality to the model score. Evidence:
  terminal-audit.json, final-02/analysis.json and final-02/analysis.md.
- Budget CLOSED: 236.915905972/900 GPU-process wall seconds; both permitted
  launches used. CPU reference/analysis/smoke ledger remains below 20/180
  seconds. No process is running and no third launch is available under this
  campaign. Engineering retries remain recorded in focused-test logs.
- Scope: seven-mixture KSC approximation, T=2, N=32, FP64 GPU/XLA with TF32
  disabled and verified memory growth. OT (1,5) is an untuned development
  comparator. Local positive kernel fitting is not original KDM training;
  fitted controlled-SMC potentials are not a full twisted-SMC consumer. No
  canonical LEDH, default, HMC or cross-model promotion follows.
- Subsequent diagnosis is complete: all 48 saved-fit/control rows pass. The
  consumer omits between-component uncertainty; this explains a mathematical
  difference, not the downstream score failure by itself. See
  ../ledh-ksc-information-20261017/result.md and its active checkpoint.md.
  Candidate rejection is not a continuation veto. Do not reopen this campaign.
