# Skeptical review of the Stan warm-up transplant

Review performed by the implementing agent; no independent reviewer or agent
was used. This supplements the pre-execution review in the plan and records
findings from the pinned release and first focused checks.

| Question | Finding | Disposition |
| --- | --- | --- |
| Is Stan a real installed baseline? | CmdStan 2.37.0 release downloaded and built locally; stanc reports 2.37.0. | Pass; preserve tarball hash and build log. |
| Does source support the adaptation recurrences? | Actual Stan headers agree with XLA dual averaging, a restart, smoothed epsilon, dense/diagonal covariance regularization, and schedules for 150/200/1000 transitions. | Four focused tests pass, including enclosing warm-up and sampling XLA. |
| Is inverse metric confused with momentum covariance? | q=A z, C=A A^T implies momentum covariance C^-1; dense kinetic and position-step checks pass. | Keep derivation and test; target-gradient checks against native model remain required. |
| Is short-budget scheduling actually identical? | The release's underfilled-buffer path differs from the simple inferred schedule. | Explicitly reject underfilled schedules for warm-up >=20; do not claim fallback parity. Main 1000-step schedule unaffected. |
| Does adaptation accidentally pool chains? | Per-chain epsilon, means and covariance; scalar execution has separate stateless seeds. | Distinct adapted metrics tested. Batched streams/layout may differ and remain calibration alternatives. |
| Can transformed caches survive a metric change? | Reusing them would be wrong. | Rebuild bootstrap state in new coordinates while retaining physical position. |
| Is native timing comparable? | CmdStan's printed warm-up excludes initial heuristic; CSV and process startup add cost. | Report process wall and printed phases; add TF serialization and report common diagnostics separately. |
| Could thread execution silently move GPU calls to CPU? | TensorFlow device context is thread-local. | Set explicit device context inside each worker callback before GPU timing. |
| Are reference and target data identical? | Matrix regression must reuse the exact design; rotated target needs correctly oriented inverse transform. | Serialize shared TF-generated data; compare native and TF densities/gradients before sampling comparisons. |
| Does success imply a fastest sampler? | Three repetitions and fixed budgets do not identify optimal sequential time to precision. | Conditional descriptive timing, fixed-budget precision screen and model-dependent interval only. |
| Does a failed funnel invalidate the transplant? | Could reflect geometry and short warm-up rather than faulty recurrence. | Preserve failed cells; use noncentered control, target parity and numerical checks to distinguish causes. |
| Does this answer the earlier fixed-HMC comparison? | Not yet: public candidate-set tuning is a different integration task. | Explicitly restrict this campaign to the requested native-Stan versus transplanted-NUTS validation; never reuse the old grid as canonical HMC. |

Plan survives this review for the stated engineering and diagnostic scope.
No posterior/default promotion follows from tests. Remaining decisive checks
are native target parity, full-run health, CPU/GPU calibration, all scheduled
target/seed cells and common posterior diagnostics. The budget and stop
conditions remain unchanged. No independent review is being claimed.

## Execution findings and localized repairs

1. GPU warm-up and retained sampling compiled and executed, but the first
   tensor artifact write inherited a GPU device context. Tensor serialization
   has only a CPU kernel. Place that artifact boundary explicitly on CPU; the
   subsequent GPU batch and scalar runs completed with verified memory growth.
   The failed attempt remains preserved and charged.
2. The native Gaussian value/gradient check failed before comparison sampling.
   CmdStan 2.37's get_uparams_r reads column-major flattened multi-point JSON
   and then reshapes sequentially as rows. Its gradients matched interleaved
   input coordinates, identifying the input boundary rather than target math.
   Four independent single-vector calls remove the mismatch. A focused test
   then passed for every target's value and gradient.
3. The first complete comparison used Python subprocess timeout polling.
   Its increasing sleeps quantized native process-wall observations and
   penalized short native calls. Keep its samples and diagnostics, but use a
   fresh comparison with blocking waitpid on an external timeout process for
   the timing report. Method, data, seeds, validity criteria and budget are
   unchanged. This is a measurement repair, not retuning on evaluation data.

The finite-budget centered funnel failed posterior screens in both engines.
This rejects the configured centered-funnel estimates; it does not invalidate
the checked adaptation arithmetic or the successful noncentered control.
The planned comparison proceeds through all targets rather than stopping at
that expected geometry failure. No inference of step-size repair is made
from native divergence flags.
