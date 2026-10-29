# Nonlinear direction execution and refused-fixture result

Nonlinear EKF/UKF and LEDH/SGQF/mixture consumers now call the shared fixed
TensorFlow direction owner once per full analytical score. Six numerical
tangent contractions in the moment filter are explicit tensor operations.
The existing mathematical recurrences, covariance providers, data/reference
models, controls and thresholds are unchanged. Native status and value-invariance
flags are enforced at the host; optional diagnostic payload ordering is retained.

Runs04783–04805 complete23 workers in336.628358 CPU and225.752586 GPU
process-seconds, within the24-worker3600/3600-second allocation. The final
04805 readback passes161 checks. The guard covers309 source scopes with1448
exact configuration/schema/reference exceptions; no numerical filtering or
score loop is newly allowlisted. GPU3 used trusted access, verified growth,
TensorFlow2.19.1 and TF32 on. Its other retained context excludes uncontended
timing claims. Exact manifests preserve commands, sources, environment and time.

The new fixture was frozen before execution at seed9292027,
d=o=1,N8,T2,float64,c=.17,b=.09, exact K=N. Its SHA256 is
5738c6889b51838bfae174db33fb43d2d954c0ad0a047b1e9b8e2e307820d98f.
No failed-case controls, arrays or tolerance were changed.

| Route | CPU run | GPU run | Outcome |
|---|---:|---:|---|
| EKF | 04783 | 04787 | Healthy full-score qualification |
| UKF | 04784 | 04788 | Healthy full-score qualification |
| LEDH | 04792 | 04799 | Original/enclosing refusal preserved |
| LEDH diagnostics | 04793 | 04800 | Original/enclosing refusal and diagnostic payload preserved |
| SGQF | 04794 | 04801 | Original/enclosing refusal preserved |
| SGQF diagnostics | 04796 | 04802 | Original/enclosing refusal and diagnostic payload preserved |
| Mixture covariance | 04797 | 04803 | Original/enclosing refusal preserved |
| Mixture diagnostics | 04798 | 04804 | Original/enclosing refusal and diagnostic payload preserved |

EKF/UKF complete output comparisons are exact, including changed theta, data
and directions. Five-point checks at2e-4/1e-4 have maximum error4.36e-12 against
the unchanged2e-6 gate. Both verify actual endpoint wiring, one trace, enclosing
HLO, deliberate invalid-input host rejection and the real independent CPU grid
reference/refinement checks. The reference is numerical, not an exact nonlinear
likelihood. These derivatives are of each filter's own finite approximation.

The first LEDH test04785 failed its healthy-fixture assumption. Diagnostic04786
shows the original six-direction authority also returns -inf with zero score
sentinels. Those zeros accompany refusal and are not usable analytical scores.
Exact-TV diagnostics04789–04791 identify the cause without changing controls:

| Provider | First reset column TV | Second reset column TV | Existing gate |
|---|---:|---:|---:|
| LEDH | 7.081277e-8 | 2.188536e-4 | <=1e-4 |
| SGQF | 7.403000e-8 | 2.148714e-4 | <=1e-4 |
| Mixture covariance | 1.013192e-7 | 1.994853e-4 | <=1e-4 |

All three first resets are valid; the second reset fails the existing balance
gate under four Sinkhorn/four terminal balance steps. Both correction flags and
the recorded covariance eigenvalue checks are healthy. Ill-conditioning is not
established. The exact shared route at ledh_canonical_score_tf.py:569–580 checks
this TV before combining overall program validity. Original/enclosing sentinels,
discrete fields and nonfinite locations agree; finite auxiliary comparisons
use the original1e-9 rule. Actual CPU/GPU endpoints raise their validity veto.
These are refusal-only results; healthy T2 derivatives and costs remain open.

04795 preserves a harness failure: a new rejection branch accidentally required
bitwise floating-diagnostic equality and failed on a5.55e-17 SGQF diagnostic
difference. It was corrected to the already declared1e-9 finite-field gate,
retaining exact status/sentinel comparisons. No scientific tolerance changed.
A premature launch while04797 was terminating was refused by the runner lock
before a worker/artifact was allocated; no numerical processes overlapped.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Qualify EKF/UKF execution | Healthy parity/derivatives and real CPU/GPU endpoint checks pass | None in this fixture | Cost/scale and broader inference | Matched cost screen | Exact nonlinear likelihood or whole-repository readiness |
| Preserve refused LEDH-family scope | Original/enclosing validity and refusal agree | Healthy T2 evidence blocked by reset balance | New scope's calibrated controls and untouched validity | Execute fresh disjoint calibration plan | A zero refusal score is a correct usable score |
| Retain full-program gaps | No threshold/algorithm/RNG changes | Memory/cost, streaming, GPU and DZ5 gates remain | Applicability beyond these consumers | Continue master queue | Main merge or canonical LEDH admission |

Fresh calibration, validation and untouched partitions are prepared separately
under `filter_gradient_nonlinear_scope_calibration_20260929.md`. They must not
use the failed seed for nomination or validation. Each provider nominates from
calibration only; validation can veto, and untouched checks follow a frozen
validated nomination. This is a test-scope mechanics exercise, not a canonical
tuning artifact or production default. Its costs remain conditional on healthy
qualification. The earlier cost plan is not authority to time a refused scalar.

Skeptical review: refusal parity proves preservation of error behavior, not
successful filtering. The explicit distinction prevents a green unit-test
summary from hiding the original failed numerical scope. Scalar nonlinear
fixtures cannot establish multidimensional performance or scientific quality.
The shared assembly has no new mathematical fork, but healthy LEDH-family
T2 evidence remains a real gap until the separate untouched checks pass.
