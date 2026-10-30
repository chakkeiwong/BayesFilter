# AR2 finite-likelihood investigation — 2026-09-23

The primary surrogate-HMC test B/R remains healthy: four chains of 8,000
retained draws, maximum R-hat 1.001053, acceptance 0.7304–0.7379, and no
retained force fallback. All 13 estimated posterior differences lie inside the
pilot margins; formal equivalence is inconclusive, with ten intervals inside
the margins and three unresolved. Do not rerun B/R or its A/R comparator.
See surrogate-hmc-ar2-posterior-result-2026-09-23.md.

The separate finite-likelihood posterior D is not ready for comparison with
the model posterior R. The preflight found seven numerical endpoint failures
among 40 outer probes; all nine posterior-bulk and nine center probes passed.
A numerical failure is not a support boundary and cannot be replaced by zero
density. Small R density at these points does not bound D probability there.
Dense quadrature also failed its cost forecast: about 17,306 CPU seconds
against the finite pass's 6,000-second allocation.

The matched localization used the same frozen controls and FP32-rounded
particle, noise, observation and coordinate design in both precision arms.
FP64 recovered four of the seven failed points; three still failed with
RESET_CORRECTION at CORRECTION. Every failed point had ESS equal to one at its
first failed step, with a zero or extremely small weighted covariance.
This supports collapse/conditioning as a mechanism beyond FP32 arithmetic.
The exact internal correction subguard has not been isolated.

The instrumentation reproduced current FP32 consumer values bitwise and
matched their status; all three healthy controls passed in both precisions.
Its scientific qualification remains false: several cap diagnostics contain
nonfinite values at failed steps. Those values are preserved, not converted
into successful diagnostics. The saved-tensor inspection verified every
tensor hash and records first-failure covariance and ESS. FP64 here is a
diagnostic and does not change the admitted FP32 scope.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep completed B/R evidence | Healthy sampling; point differences inside margins | No retained force fallback | Three formal intervals unresolved | Preserve results; no extra looks | Formal equivalence or speed superiority |
| Do not launch D/R or C/D,C/R yet | Finite endpoint domain qualification failed | Seven FP32 and three FP64 correction failures; nonfinite observables | D mass and behavior near invalid regions unknown | A fresh scope-specific correction/collapse repair and cheaper integration design would be required | Failure of surrogate-force HMC with exact endpoints |
| Continue the existing master on M13 | Wrapped observation callback already repaired | Integrated model reference not yet qualified | Nonlinear integration and reference cost | Define and check the independent reference scope before tuning/sampling | All-model completion |

| Inference status | Finding |
|---|---|
| Hard veto screen | Finite endpoint numerical failures block the affected finite-posterior comparison. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Four rescued points under FP64; runtimes and ESS/covariance traces. |
| Default readiness | No new numerical default or DCTR scope admitted. |
| Next evidence needed | Valid endpoints with scoped tuning, qualified D integration, then C/D and C/R. |

Attempts and commands are in run-manifest.json and accounting.json under
artifacts/surrogate-hmc-closure-20260922-ar2-finite-01 through -04.
Attempt 01 was an adapter-status harness KeyError; attempt 03 was a dtype
boundary error. Both were repaired and charged before fresh retries.
Attempt 04 completed with source hashes unchanged. Its review tests and full
logs are retained there. The inspection is
artifacts/surrogate-hmc-ar2-finite-inspection-20260923.json.

The four finite attempts consumed 250.743113 CPU-worker seconds and
116.384288 GPU-wall seconds. Recorded campaign charges are 35,338.766601 CPU
and 33,138.502698 GPU seconds. Remaining global balances are 18,661.233399 CPU
and 193,661.497302 GPU seconds; finite-pass balances are 5,749.256887 CPU and
7,383.615712 GPU seconds. Manual recovery/inspection overhead is unmetered and
is not represented as measured computation.

Post-run review: high precision recovery weakens a precision-independent
failure explanation for four points, while three persistent failures rule out
a purely FP32 explanation for this probe set. A local numerical repair may
rescue the finite target; its controls would need fresh scope-specific
calibration and holdout validation. These probes do not establish a posterior
bias magnitude or invalidate the separate completed exact-endpoint test.
