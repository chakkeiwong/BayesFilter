# Reply to MacroFinance: bootstrap probability and diagnostic consistency

Both central findings were correct and remained unfixed in main
`2c2419c096ce744e23a37e0ab19e65840334119d`. The earlier ESS repair reached
`hmc_posterior_diagnostics`, but bootstrap and the public convergence endpoint
were separate omissions. This repair changes both call chains and their tests.
The [reviewed repair plan](bayesfilter-macrofinance-bootstrap-diagnostics-repair-2026-09-22.md)
records scope, limits and the skeptical audit.

## Findings and implemented corrections

| Issue | Checked verdict | Repair |
| --- | --- | --- |
| Bootstrap acceptance statistic | Real defect: the classifier and directional repair used the binary runner acceptance field. | Compute `mean_acceptance_probability` from every recorded proposal log ratio; exclude burnin, include rejections, and validate recorded count/shape/finiteness. Binary fields remain explicitly explanatory. |
| Reported 0.6846376853 versus 13/16 case | Reproduced with a controlled trace against preserved pre-repair source. Old decision: increase epsilon. New decision: pass the acceptance screen. | The complete three-round fixture stops at the third screen, epsilon 1.154700538379251 and clamped L=3; the unsafe fourth screen is never requested. |
| Unsafe trial handling | Previously every bootstrap exception ended preparation. The separate startup and metric-boundary searches do not automatically repair this call site. | Only a narrowly adapter-declared `InvalidArgumentError`, attributed by the repository recorder to a proposal target callback after a finite pre-state, may nominate a smaller fresh trial. Preserve original failure, use the existing repair cap and multiplier, replace terminal runners, and require a completed new screen before handoff. |
| Public convergence bulk/tail ESS | Real integration defect: older TFP arithmetic and inclusive upper-tail indicator remained reachable. | Delegate to repaired posterior bulk/tail and split/folded R-hat implementations, converting the public axis order explicitly. Preserve thresholds, schemas and split conventions; record ESS method identity. |
| Constant-draw ESS | Additional regression found during this repair: XLA mean rounding could leave a tiny positive variance for exactly constant draws. | Check constancy directly on the draws; retain the existing rule that constant input is nonpromotable. |

The independent antithetic fixture makes the diagnostic defect concrete. The
old public API returned bulk ESS `[278247.0989091855, -8269.061153451126]`;
the repaired route returns `[3082.547155599167, 3082.547155599167]`, matching
ArviZ 0.21 within the declared tolerance. This is an engineering reference
check, not an empirical claim about the MacroFinance posterior or HMC efficiency.

The failure domain bound is separate from measured acceptance endpoints. It
never supplies a fictitious zero acceptance or changes the target, its score,
assertions, proposal state or Metropolis correction. An interior logarithmic
midpoint is a new trial, not evidence that all smaller steps are safe. Every
trial records recomputed L and clamping. Exhaustion returns no repaired pair.
Failed-round vetoes remain in history; only an attributed discarded failure
followed by a successful new screen stops vetoing subsequent preparation.

## Consumer implications

No public tuning entry point changes. Ordinary exact-value/score consumers
continue through `tune_hmc_kernel` and its preparation helper. The historical
`hmc_kernel_tuning` bootstrap symbols alias `hmc_bootstrap`.

Consumers that inspect bootstrap decisions should read
`mean_acceptance_probability`. `acceptance_rate` remains the historical binary
field, also exposed as `binary_acceptance_rate`. Probability evidence that is
missing or nonfinite does not fall back to either binary field.

For optional exception recovery, the supplied adapter must implement
`classify_target_exception(error) -> bool`, recognizing only its own known
domain assertion failures. Classifying every TensorFlow error is wrong.
The repair also requires repository first-failure attribution, currently
available for `tf_function` without XLA. Other modes or missing attribution
remain fail-closed. No consumer classifier or exact MacroFinance primitive was
invented in BayesFilter; the first overflowing primitive in the supplied report
remains unidentified.

The public convergence API still takes `[draw, chain, parameter]`; the
posterior diagnostic API still takes `[chain, draw, parameter]`. Existing calls
need no threshold or shape migration. Recompute old reports from draws before
comparing diagnostic thresholds. The separately named TFP mean/quantile
precision estimator and covariance-adaptation ESS heuristic are unchanged.
The fixed-transport reporting helper also names its shared repaired mean-ESS
method explicitly, retaining its existing split convention.

R-hat, ESS and MCSE remain outside tuning-candidate membership, ranking and
repair. The official book sources, chapters 21b and 26b, and
`docs/reference/hmc-tuning-interface.md` now explain these distinctions.

## Validation and limits

Tests cover both directions of probability/binary disagreement, the supplied
sixteen-proposal case, rejected proposals, absent/nonfinite/empty/misaligned
traces, strict result counts, exact public call-chain wiring, L clamping,
preserved exceptions, unrelated errors, finite retry caps, no unscreened
handoff, and replacement of a terminal traced runner. A real bounded Gaussian
target exercises TensorFlow assertion capture and fresh-runner recovery.
Independent public-API reference tests cover ties, antithetic chains, odd
lengths, constant draws and the preserved precision-estimator convention.

The test receipts, actual commands, environment, source hashes, elapsed worker
times and XML results are under `artifacts/hmc-repair-master-2026-09-16/m24-r2/`.
The final combined receipt is `tests-final-r1/result.json`; prior failed test
attempts remain preserved. The book was rebuilt, the two changed PDF pages
were inspected, and citation resolution was checked. GPU devices were
intentionally hidden during these CPU diagnostic/reference checks.

Final result: **413 passed, two skipped**. The absent optional DZ5 fixture and
the existing tiny Gaussian smoke whose bootstrap exhausted its repairs are the
two skips; they are not passing posterior evidence. Total measured CPU charge
including intermediate attempts is 444.9127681890968 seconds. No GPU budget was
used. The before/after reproduction is saved in
`m24-r2/tests-final-r1/baseline-reproduction.json`.

The supplied MacroFinance run directory is unavailable in this checkout.
Its claimed source archive and raw run tensors therefore were not independently
replayed. The old run remains consumed. MacroFinance must qualify its current
source/target and use a fresh versioned output root before rerunning preparation.
The 22/180/22 schedule, six-candidate inventory and subsequent posterior
validation have not been tested by these fixtures. Four remaining campaign
slots versus six candidates remains a separate consumer budget issue. Missing
native divergence telemetry remains unavailable, never zero.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Unsupported conclusion |
| --- | --- | --- | --- | --- | --- |
| Repair the two reported call chains | Old behavior reproduced; corrected decision and independent ESS parity demonstrated | Original assertion retained; unrelated/initial/retained failures remain terminal | Exact consumer run artifacts unavailable | Fresh consumer qualification and preparation under its existing budget | Old third round is a verified final kernel, or the 180-transition window now succeeds |
| Preserve separate estimator contexts | Named TFP precision and metric heuristics retain their computations | No new tuning R-hat/ESS gate | Full scientific calibration remains in the master program | Continue source-bound confirmation and report method identities | A local reference test establishes posterior convergence or a new default |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Controlled known-domain proposal failures are repairable only under explicit attribution; unrelated and malformed evidence remain terminal. |
| Statistically supported ranking | None sought or established. |
| Descriptive-only differences | The supplied short-screen acceptance values describe that screen; test runtime is not a performance comparison. |
| Default-readiness | This corrects existing reporting/decision semantics; it does not establish broad model readiness. |
| Next evidence needed | Fresh MacroFinance preparation, unchanged candidate verification, then separate posterior assessment. |

Post-run red-team: the strongest alternative explanation of future failure is
genuinely unsafe geometry or insufficient adaptation evidence after bootstrap.
These repairs remove the demonstrated wrong decision and reporting omissions;
they do not guarantee every target will tune successfully. The live M21/M22
campaign keeps its frozen pre-repair source and cannot certify the new code.
