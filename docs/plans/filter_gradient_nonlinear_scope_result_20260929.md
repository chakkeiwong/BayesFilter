# Nonlinear analytical-score qualification on a fresh calibrated scope

The enclosing TensorFlow/XLA analytical-direction implementation passes all
three nonlinear LEDH-family providers on CPU and GPU, with and without control
diagnostics. Both untouched cases per provider retain the original finite
value program and analytical derivatives. Maximum nested output difference is
2.22e-16, maximum two-step five-point error is7.10e-12 (gate2e-6), and maximum
CPU/GPU score difference is1.23e-15 (gate1e-9). This qualifies the tested
execution behavior; it does not establish canonical LEDH or filtering accuracy.

The original seed9292027/four-four fixture remains refused and excluded from
calibration, validation, timing and acceptance. Its preserved failure was
insufficient reset balance, not demonstrated ill-conditioning. See
filter_gradient_nonlinear_directions_result_20260929.md for that separate unit.

Under filter_gradient_nonlinear_scope_calibration_20260929.md, each provider
independently passed the first predeclared reset-count pair, eight/eight, on
three calibration and three validation inputs. No higher count was evaluated.
The resulting nominations were frozen before untouched evaluation. Exact
partition hashes, seeds9293027/9294027/9295027 and generation rules remain in
the plan and fixtures. These nominations are test-only; they cannot satisfy a
canonical tuning artifact requirement or set production controls.

| Provider | Calibration/validation | Diagnostics CPU/GPU | Ordinary CPU/GPU |
|---|---|---|---|
| LEDH | 04806 | 04809 / 04812 | 04815 / 04818 |
| SGQF | 04807 | 04810 / 04813 | 04816 / 04819 |
| Mixture covariance | 04808 | 04811 / 04814 | 04817 / 04820 |

The qualification checks complete nested output parity, changed parameters,
observations and directions, two-step five-point derivatives, real endpoint
wiring, owner reuse, one trace, enclosing HLO without pfor/host callbacks, and
actual host rejection on invalid input. Real independent CPU grid refinement
checks pass. Ordinary and diagnostic endpoints also agree with each other.
All finite derivatives refer to each filter's own finite approximation.

Continuation review identified that the first untouched harness checked only
the diagnostics-enabled signature. Six ordinary-endpoint workers were added
using the same frozen counts/arrays and unchanged gates; no tuning followed
inspection of untouched data. No production source changed in this phase.
Run04821 passes162 combined readback/policy checks, including the earlier
refusal cohort and current source hashes. The guard remains309 sources and
1448 exact configuration/schema/reference exceptions, with no new numerical
loop exception. Harness/runner Ruff and whitespace checks pass. Full runtime
Ruff reports five import-order findings, all reproduced from the committed
baseline (three in nonlinear_tf, two in nonlinear_adapter); this repair adds
no lint diagnostic. Frozen numerical sources were not reformatted after tests.

All16 workers pass, using249.909228 CPU/223.021712 GPU process-seconds within
the24-worker/2400-second-per-device allocation. Numbered manifests in
docs/plans/artifacts/filter-gradient-repair-20260917 preserve exact commands,
Git/source hashes, software, device placement and wall times. GPU3 runs use
trusted access, verified preinitialization growth and recorded TF32. Other
retained GPU contexts prevent uncontended performance/capacity conclusions.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Qualify this nonlinear execution scope | Full parity, derivatives, endpoint, graph and source checks pass | No new numerical veto | Tiny d1/N8/T2 scope; costs and broader consumers | Matched fresh-process cost/accounting screen | Canonical LEDH, exact nonlinear likelihood or production readiness |
| Preserve original failure | Refusal behavior retained in both implementations | Old scope remains invalid | Controls outside the freshly calibrated scope | Keep failed evidence and its regression checks | Failed original has become healthy |
| Keep master open | Scoped implementation progress recorded | Performance, source applicability, GPU capacity and DZ5 gates remain | Broader coverage and accepted costs | Follow master queue | Main merge or whole-program completion |

Terminal self-review: independent finite differences check the derivative of
the implemented approximation, not its model accuracy. The strongest coverage
limitation is the scalar, two-observation fixture; broad inference claims
remain unsupported. No stochastic ranking is made, and no failing numerical
gate was weakened. The cost plan now explicitly binds the newly validated
LEDH fixture, avoiding the refused original scalar.
