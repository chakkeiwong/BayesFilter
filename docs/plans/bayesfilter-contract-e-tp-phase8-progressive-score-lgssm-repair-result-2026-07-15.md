# Contract E--TP Phase 8 Progressive-Score LGSSM Repair Result

metadata_date: 2026-07-15
phase: 8A
status: LGSSM_CONTINUATION_ORACLE_PASS_COMPACT_PROGRESSIVE_CANDIDATE_FAIL
plan: `docs/plans/bayesfilter-contract-e-tp-phase8-progressive-score-lgssm-repair-plan-2026-07-15.md`
master_plan: `docs/plans/bayesfilter-contract-e-tp-all-model-gradient-comparison-master-plan-2026-07-15.md`

## Outcome

The recursive LGSSM failure was caused by feature insufficiency, not by the
finite-scalar derivative wiring. The initial feature system retained only the
next corrected-LEDH contribution and therefore lost information after its
one-step guarantee.

Two repairs were tested:

1. a compact 16-row feature system adding five
   `next_predictive * centered_target_model_score` rows; and
2. an 11-row LGSSM oracle diagnostic replacing the one-step row with the exact
   remaining-horizon likelihood
   `p_theta(y_(t+1):T | X_t=x) / p_theta(y_(t+1):T | X_t=0)`.

The compact candidate failed at `T=10`. The exact continuation candidate passed
the frozen center value and score screens at `T=2,10,50`. This proves that a
small sufficient continuation statistic can repair the LGSSM recursion. It does
not prove that the compact transferable score-mark feature is sufficient, and
it does not supply a nonlinear continuation feature.

## Mathematics Implemented

The chapter now defines the target-model conditional additive score field,
proves its tangent-measure covariance representation for dominated filters,
states the pairwise backward recursion, and explicitly separates that theorem
from the moving atomic Contract E--TP finite program. Target-model score marks
are diagnostic primal features; the total derivative of the executed finite
scalar remains owned by TensorFlow autodiff/JVP and independently checked by
FD.

The mark recursion has an additive gauge. Common mark offsets pass unchanged
through row-stochastic backward weights and disappear after centering. The
implemented compact chart therefore constrains only the five score-bearing
interactions and re-centers selected marks explicitly.

For the exact LGSSM continuation arm, a reverse-time Gaussian information
recursion computes natural parameters `(J_t,h_t)` satisfying

```text
R_t(x) / R_t(0) = exp(-0.5*x'J_t*x + h_t'x).
```

Tests verify its values and all parameter derivatives against the direct
conditional Kalman program.

## Evidence

### T=2 quadrature control

Order 3 failed the inherited center screens before projection error could be
interpreted. Both feature modes produced the same result, and their discrepancy
started at the first increment. This reproduced the known teacher quadrature
error.

At order 5, both arms passed:

| Quantity | Result |
| --- | ---: |
| Value difference to Kalman | `0.0002046289013` |
| Largest score relative error | `0.0025024215` |
| Sign reversal | none |
| Largest same-scalar FD relative error | below `1.0e-9` |

### T=10 discriminating result

| Candidate | Value difference | Largest score relative error | Sign reversal | Same-scalar derivative |
| --- | ---: | ---: | --- | --- |
| Compact score interaction | `0.0214014927` | `1.677698129` (`phi2`) | `phi2` | pass, max about `9.49e-7` |
| Exact continuation | `-0.0004812718` | `0.002768626` | none | pass, max about `2.95e-9` |

Thus the five one-step score interactions are insufficient, while the exact
remaining-horizon statistic repairs the observed recursive information loss.

### T=50 continuation result

The controlling aggregate is
`docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8_continuation_information_v2_lgssm_t50_order5_attempt1_20260715/result_aggregate_v2.json`.

| Quantity | Result |
| --- | --- |
| Contract E--TP value | `-136.07645802675307` |
| Kalman value | `-136.0759748579247` |
| Value difference | `-0.0004831688284`, pass |
| Contract E--TP score | `(5.649234189,-3.834215952,0.302395966,-1.905359063,4.362908728)` |
| Kalman score | `(5.655446880,-3.835056459,0.302361684,-1.917180271,4.354265919)` |
| Componentwise relative errors | `(0.001099,0.000219,0.000113,0.006166,0.001985)` |
| Sign reversals | none |
| Same-scalar FD relative errors | maximum `1.98e-8`, pass |
| Chart count | `49` |
| Minimum chart weight | `0.0001778712494` |
| Maximum scaled condition number | `2231.870337` |

The five score directions were executed in separate CPU-hidden float64 JVP
shards to bound memory. The aggregator requires one preparation hash, one
feature mode, all five direction indices, matching primal/Kalman values, valid
charts, and per-direction same-scalar FD passes. It records every shard hash,
command, and wall time.

## Failure And Repair Ledger

| Attempt | Classification | Root cause | Repair |
| --- | --- | --- | --- |
| Progressive `T=2`, 21 rows | mathematical feature-rank failure | centered observation-scale score row duplicated the quadratic moment span | use gauge-fixed 16-row chart and explicit re-centering |
| First result serialization | harness failure | result writer rejected an existing preparation directory | preserve output-file veto but allow existing parent directory |
| Parallel/monolithic `T=10` score | harness memory failure | persistent history Jacobian and repeated large score-mark derivatives | total reverse AD plus FD history; compact result then completed |
| Direct continuation `T=50` | implementation scaling failure | conditional Kalman continuation rebuilt at every reset, creating an `O(T^2)` graph | exact `O(T)` backward-information recursion |
| Monolithic continuation `T=50` score | harness memory failure | five directions accumulated in one TensorFlow process | five independent scalar-JVP shards plus strict aggregate |

No threshold, target, feature identity, data, or promotion criterion was changed
after seeing a scientific result.

## Verification

- focused Phase 1--5 plus Phase 8 suite: `49 passed`;
- direct information-feature values and total derivatives match the conditional
  Kalman implementation;
- Python compilation passed for all touched modules and harnesses;
- scoped `git diff --check` passed;
- existing `docs/main.log` has unrelated bibliography warnings but no new
  undefined control sequence or LaTeX error in the edited section;
- Claude Opus health probe returned `CLAUDE_PROBE_OK`, but three bounded
  substantive reviews returned no output. This is recorded as an advisory
  review limitation; source equations and local proofs were checked directly.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept LGSSM exact-continuation diagnostic | `T=2,10,50` value and every score component pass | no engineering or center scientific veto | off-center chart validity absent | retain as oracle/reference arm | general nonlinear algorithm or HMC readiness |
| Reject compact one-step score-interaction candidate | `T=10` value/score fail with `phi2` sign reversal | candidate promotion veto fired | which non-oracle continuation basis is sufficient | run look-ahead/continuation-basis ladder | research direction rejected |
| Do not hand off unchanged compact features to nonlinear full horizons | general-progressive handoff condition not met | handoff blocked, campaign continuation allowed | target-specific continuation design | prepare nonlinear short-prefix feature candidates and LGSSM non-oracle basis repair | all-model completion |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | continuation candidate passes at frozen center; compact candidate fails scientific promotion |
| Statistically supported ranking | none; deterministic center diagnostic only |
| Descriptive-only differences | chart condition, runtime, and candidate gap magnitudes |
| Default readiness | blocked |
| Next evidence needed | non-oracle continuation basis that passes LGSSM, target-specific nonlinear chart protocols, off-center audit, Zhao--Cui certification, paired comparisons, and GPU/XLA scaling |

## Post-Run Red Team

The strongest alternative explanation is that exact future observations make
the continuation arm nearly tautological. That is correct and is why the arm is
labeled an oracle diagnostic, not a deployable algorithm. Its value is causal:
it shows the recursive failure is repairable by retaining the right future
functional and is not forced by LEDH value/score wiring.

The weakest evidence is transferability. The compact score-mark candidate did
not pass, so no nonlinear-model success may be inferred. A nonlinear
continuation feature must be target-specific, fixed before audit, and tested at
short prefixes before any full horizon.

Phase 8A gate: `LGSSM_ORACLE_CERTIFICATE_PASS_GENERAL_PROGRESSIVE_HANDOFF_BLOCKED`.

