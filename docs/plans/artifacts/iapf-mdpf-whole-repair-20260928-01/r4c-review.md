# Skeptical review of refreshed master and R4c execution

28 September 2026. **PASS for the bounded implementation/diagnostic scope;
OPEN/REVISE for general R-solver equivalence and full scientific/default
promotion.** This is self-review, not independent terminal acceptance.

Reviewed actual sources: shared finite fitter, shared paper value kernel,
R exporter, parity/replay/impact consumers, R 4.1 vmmin, dispatcher, contracts
and selected structured results. The complete actual consumer executed in
attempt 28; this is call-chain evidence. Attempt source snapshots, not current
unstaged filenames alone, identify what ran.

| Risk | Finding and resolution |
| --- | --- |
| Wrong baseline/target |Legacy six-parameter/initial-state convention was not reused as the paper target; new explicit-matrix value route is the consumer. Appendix B nonlinear toy substitution remains prohibited. |
| Hidden finite default |A_raw/B_scaled,100 iterations, initializer, floor, FP64 and full-rank restriction remain explicit hypotheses/restrictions. No unexamined regularizer or damping was promoted. |
| Incomplete R semantics |Invalid trial handling differs, and full versus symmetric-triangle Hessian arithmetic needs checking. These are R4d tasks; general finite-solver equivalence is not certified. |
| Proxy replacing promotion |Small checks and a close final logZ cannot close replication. Paper parameter parity remains FAIL after actual-filter completion. |
| Stale program |“Fitter/controller not implemented” was wrong after R4c. Master and executable phase ledger now distinguish implemented consumer, failed longer-run parity and future work. |
| Serialization/comparator defect |15-digit R CSV loss proved, then repaired with 17-digit strings plus 3,142 exact matrix roundtrips. Failure persists; repair not overclaimed. |
| XLA/loop explanation |GPU graph also differs; same-backend perturbations amplify. Native TF loops and one trace/signature are observed. No evidence of a NumPy/per-particle Python path in the tested consumer. |
| Environment/memory |Escalated GPU visibility, memory growth, FP64/TF32-off recorded. CPU references hide GPUs; author PyTorch remains an independent reference. |
| Unfair comparison |Same full-precision model/entropy arrays. MDPF baseline initialization and mixed-state metric requirements remain explicit; normalized-density and training-seed gaps are open. |
| Statistical overstatement |No ranking; single paper-horizon run descriptive. 5,000 MDPF trajectories do not become 5,000 training replicates. |
| Wrong stopping rule |Strict parameter failure vetoes its promotion, not the research direction. The explicit impact diagnostic continued without erasing that failure. |
| Budget/commands |All 28 attempts preserved and charged; next work is not advertised as runnable. A full repeated MDPF suite exceeds available compute at measured cost. |
| Evidence version |Small129-check runs predate a localized source edge repair; current version repeated eight isolated controls, paper backward checks and actual consumer. Versions are not silently conflated. |

Fourteen dispatcher tests PASS (2.837 seconds reported by unittest).
They cover dependency enforcement, failures/timeouts, budget/lock/recovery,
source syntax preflight, scientific nonpromotion, and the new failed-parity
preservation guard. Routine test logs are saved in regression-r4c.log.

Premortem remained relevant: commands can run the wrong experiment or reproduce
a convenient numerical proxy. Post-run evidence exposed finite sensitivity and
a fixture defect without blaming the paper. The main unresolved causal
alternative is source-semantic discrepancy versus arithmetic amplification.
R4d must distinguish them; it must not fit a tolerance to the failure.

Permission review: no new broad allow-list, magic approval token, package
mutation or external message was needed. Existing approved direct commands
and escalated GPU execution handled the work. The subsequent numerical scope
needs a real implementation and predeclared contract, not a ceremonial
reapproval of this unchanged local campaign.
