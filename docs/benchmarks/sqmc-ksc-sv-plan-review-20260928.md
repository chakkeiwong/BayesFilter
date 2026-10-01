# KSC SV plan review — 2026-09-28

The earlier plan existed, but was wrong about its exact oracle. It cannot be
executed unchanged. The corrected plan is
`docs/plans/sqmc-ksc-sv-comparison-20260928.md`.

## Material findings and repairs

1. **Wrong reference.** The existing KSC factory prices observations with a
   seven-component mixture and uses Gaussian moments only for the proposal.
   Kalman on those moments computes a different marginal likelihood and score.
   An observation-transform Jacobian does not remove this density mismatch.
   The new grid reference integrates the declared mixture and propagates both
   derivatives analytically. Exact mixture enumeration checks T1/T2; a separate
   test shows the Gaussian Kalman difference. Long-horizon convergence remains
   empirical, checked on each dataset rather than asserted from the small test.
2. **Missing derivative.** The flow observation tangent omitted 2*d(log_beta).
   The canonical factory now includes it. Both coordinate directions passed
   callback finite differences including moving particle locations. Static
   inspection follows KSCSpec.model -> shared campaign kernel -> canonical
   analytical executor -> observation tangent in the flow and mixture-density
   tangent in the weights. Executable parity for this full GPU consumer chain
   is NOT CHECKED yet and is required before the comparison.
3. **Unfair/stale assumptions.** The plan declares the initial law, timing,
   fixed Q, mixture density, parameter point, precision and N explicitly.
   Every route/horizon receives its own calibration and disjoint validation.
   Final data/design pairs are shared between routes, independent between
   replications, and excluded from selection. Eight pairs replace the older
   sixteen as a bounded initial stage; N2016 and broader parameter regimes
   are explicitly deferred. No broad SV or default claim is licensed.
4. **Numerical defaults.** The plan records inherited finite-program settings
   as hypotheses and schedules dedicated ridge/damping/trust sensitivity on
   calibration data only. These observations cannot establish optimal tuning
   or non-harm. Invalid perturbations remain evidence, not reasons to silently
   turn off safeguards or expand the control search.
5. **Execution/accounting.** The new supervisor shares accounting across all
   KSC output directories, retains original source snapshots, requires passing
   matching-source GPU checks, records pending attempts, and reserves shutdown
   time. It distinguishes reference nonconvergence from harness failures and
   stores finite candidate errors without treating them as infrastructure
   failures. At most two infrastructure retries are available per unit.
   Runtime enforcement itself still needs the bounded execution check.

## Evidence already obtained

`docs/plans/artifacts/sqmc-ksc-sv-20260928/attempt-01/cpu-tests.log` reports
**5 passed, 2 dependency deprecation warnings in 4.10 seconds**. GPU devices
were intentionally hidden with CUDA_VISIBLE_DEVICES=-1; the launcher wall
charge is 6.486 seconds in cpu-tests.json. Checks cover T1/T2 exact mixture
agreement, both grid-score finite differences, the Gaussian approximation
counterexample, and both total callback derivative directions.

Subsequent changes add explicit reference-failure records, runner guards,
source snapshots and sensitivity diagnostics; they have received syntax-only
inspection, saved in static-review.json. Their numerical execution is pending.
The callback numerical formula that passed the tests has not changed again.

The completed Kalman work and retention decision are committed as **479a4616**.
New KSC files and repairs remain uncommitted pending final executable checks.
The commit hook passed three tests in 106.30 seconds. It defaults to CPU in
`tests/conftest.py`, but its historical device placement was not captured;
300 seconds are provisionally reserved rather than claiming measured GPU use.

## Decisions and limits

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain all four routes | Owner decision; Kalman differences descriptive | No retained route rejected | Small Kalman replication | Test KSC under corrected reference | No overall winner |
| Reject old KSC oracle claim | Mixture and Gaussian densities differ | Scientific reference veto repaired in new plan | Long-T quadrature error | Per-dataset grid convergence | Kalman is not exact here |
| Accept CPU checks for their bounded role | Five focused tests pass | No CPU fixture veto | Full GPU call chain untested | Renew time, run bounded GPU checks | No full score-accuracy claim |
| Hold numerical execution | Elapsed deadline passed | Time continuation veto active | No KSC GPU evidence | Owner elapsed-window renewal | No implicit compute increase |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Old oracle assertion wrong; elapsed budget expired; KSC final candidates untested |
| Statistically supported ranking | None for KSC |
| Descriptive-only differences | No KSC final observations yet |
| Default readiness | Not established; frozen controls remain hypotheses |
| Next evidence | Passing full call-chain GPU checks, converged final references, independent pairs and paired uncertainty |

Strongest alternative explanation for a future apparent SQMC advantage is the
single parameter regime, limited tuning, or fortuitous data/design draws. A
ranking could reverse with broader regimes or further independent pairs.
The weakest present evidence is the unexecuted full GPU call chain; callback
and reference tests cannot substitute for it. The existing mixture model also
approximates native SV, so agreement here cannot certify a native-SV score.

Review verdict: **REVISE applied to the older plan; corrected plan reviewed,
execution conditional on elapsed renewal and passing bounded checks**.
This was a local Codex review with executable CPU evidence, not an independent
review. No reviewer was launched and no external publication was performed.

## Execution update — 2026-09-29

The owner renewed elapsed execution by 48 hours; the shared ledger records
its start and deadline. The original 12 GPU-hour aggregate cap and retry cap
remain unchanged. The new CPU check passed all five cases. The matching-source
GPU check passed all four consumer routes: maximum coordinate finite-difference
discrepancy 4.159741995302113e-11; maximum graph/XLA discrepancy
2.942091015256665e-15; every N1008 check finite in both coordinates. Evidence:
`docs/plans/artifacts/sqmc-ksc-sv-20260928/attempt-02/check-attempt-01/checks.json`.
These close the pre-run execution questions above. They do not prove score
accuracy; that is measured by the independent held-out comparison.

The final campaign completed all 16 scopes under its frozen source snapshot.
The terminal review is docs/benchmarks/sqmc-ksc-sv-results-20260929.md; it reports
actual accuracy, reference convergence, uncertainty, heuristic losses and
safeguard sensitivity. The earlier oracle documents now
carry correction notices while preserving their historical text.
