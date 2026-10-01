# Fixed fitted-APF execution repair

The fixed-iteration fitted-twist adapter now executes compatible random draws,
each filter/fit iteration, failure stopping and the final analytical score in
one cached TensorFlow/XLA owner with a stable signature. It calls the existing
filter and recursive fitter; no numerical authority or derivative target was
replaced. Quadratic features use tensor operations with the original column
order. FP64 random conversion reuses the shared stateless authority; FP32
conversion extends that authority using the same TensorFlow Philox/Box–Muller
definition. No seed labels or fitted/final stream separation changed.

The nominal fit remains frozen for the final analytical score. The owner
returns coefficient/error/value histories, attempted count and first invalid
iteration. On failure it skips subsequent fits and the final filter; the host
raises the existing iteration-specific rank/precision error. Existing public
subkernel-call accounting remains five for the two-iteration fixture, with an
additional field recording one enclosing execution.

The frozen baseline is4f0dfeb3d, with original sources and fixture hashes under
`tests/fixtures/filter_repair_fitted_apf_20260929/`. The execution plan records
the seed, dimensions, numerical gates, defaults/nonclaims and exact runner
groups. Runs04840–04857 preserve commands, sources, environment, hardware,
memory-growth provenance, complete records and enclosing HLO.

| Model | Dtype | CPU maximum full-record difference | GPU maximum full-record difference |
|---|---|---:|---:|
| Gaussian | FP64 |8.881784197e-16|0|
| Gaussian | FP32 |4.768371582e-7|5.960464478e-7|
| Nonlinear scalar | FP64 |4.440892099e-16|0|
| Nonlinear scalar | FP32 |4.768371582e-7|4.768371582e-7|

Every row includes original and changed theta, nominal theta, observations and
seeds; full fit histories/coefficients, final clouds/value/score/status and
exact replay. Each owner has one trace and enclosing HLO without host callbacks.
FP64 two-step fixed-fit finite differences have maximum error1.31628819e-10.
Ordinary Gaussian/nonlinear public endpoints pass on CPU/GPU using a frozen
physical dataset and live fit/final random streams. Independent normalized-
proposal and positive/negative-precision tests pass. Injecting identical
particles in the second fit triggers the real QR rank veto on both devices;
the third history slot and final outputs remain unexecuted NaNs, and the host
reports iteration1. No ridge, clipping, retuning or tolerance change was used.

04842 failed an exact GPU FP64 comparison against eager feature evaluation by
3.47e-18. The original fitter already compiled its features, so that comparator
was wrong for the vectorization question. The corrected test retains eager
arrays as explanation and requires exact equality to the archived original
compiled features.04843 passes all five GPU primitive/independent checks. The
failure remains archived; no runtime change followed from it.

04857 passes161 combined readback/policy checks. The guard now covers312 sources
and1457 exact exceptions; the nine additional exceptions cover index/seed-label
configuration, seed-disjointness validation and completed-result formatting.
There is no numerical-loop or NumPy exception. Harness/new-source Ruff checks
and whitespace checks pass. The two pre-existing I001 import-order findings
also occur in the archived baseline; no new lint category was introduced.
Eighteen workers consumed117.377552582 CPU and192.454147875 GPU seconds. Remaining
global budget is25.029981 CPU/24.776033 GPU process-hours.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Qualify this fixed-fit numerical repair | All recorded CPU/GPU gates pass | Rank/precision errors preserved | Compilation/residency cost unmeasured | Matched original/candidate cost screen | No cost acceptance |
| Keep F07/F19 and terminal closure open | Adaptive iAPF remains untouched | Its Python recurrence remains debt | Variable particle shapes and stopping | Separate native controller repair | No whole-repo completion |

| Inference status | Finding |
|---|---|
| Hard veto screen | Healthy mechanics pass; deliberately invalid fit rejected |
| Statistically supported ranking | None |
| Descriptive-only differences | Numerical errors above for bounded fixtures |
| Default readiness | Numerical controller qualified in this scope; costs remain open |
| Next evidence | Matched fresh-process costs/memory and adaptive-controller repair |

Post-run review: this qualifies the fitted controller reached by the endpoints,
not every operation in the surrounding score-study harness. Public endpoints
still perform separate data/oracle and eager preparation/reporting operations;
those require their own applicable dispositions. The returned kernel is used
only for trace accounting by the two repository callers; both were exercised.
No HMC, training, canonical LEDH admission, posterior-quality or broad scientific
claim follows. CPU is reference/debug; GPU remains the default target, and
shared numerical checks do not establish uncontended capacity.
