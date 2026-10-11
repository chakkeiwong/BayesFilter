# D4R4a adaptive MAMS full-model pricing execution

Plan: [full-model pricing](bayesfilter-mams-d4r4a-full-model-pricing-2026-10-12.md).
Date: 12 October 2026. Status: **source and CPU preflight complete; trusted
GPU/XLA cost-only launch pending the scoped source snapshot**. No posterior
or reference promotion follows from this phase.

## Skeptical implementation audit before launch

The four target factories and current full-horizon score paths are the intended
baseline. The saved local factors are numeric warm-start proposals, not a
validated global map. `geometry_source_policy='strict'` still rejects changed
source; only an explicit `warm_start_only` binding can consume old factors,
and it records every mismatch, current data identity, factor, starts and
binding hash. The new price consumer calls the same MAMS discarded proposer
as D4R3 and the existing native execution binding; it never issues an HMC
tuning artifact or interprets R-hat. Its 32-transition prices describe cost
and health, not qualification, posterior delivery or performance superiority.

All four saved reference loaders reject current hashes. Their inputs and
target-source files match, and four independent current value/score regressions
pass, but their broad source-closure records cannot be edited or waived in
place. The price consumer never loads those references. This prevents a false
posterior claim while retaining the ability to measure full-model work.
Target-specific numerical differences remain descriptive; the same count is
not a fair cost comparison across likelihoods. One fresh seed block, a local
start bank, inherited 0.2 epsilon and sqrt(d) duration are declared pilot
hypotheses, not calibrated defaults. All generated, omitted, failed and
unstarted proposals remain visible. Shared target/score/source defects stop
affected execution; an individual veto or exhausted resource allocation does
not reject the research direction. The total parent deadline is 7,200 seconds
including its shutdown allowance, with no per-target timeout.

The full-model source snapshot must contain the two new runtime files, the
four tracked target implementations, the shared proposer and all dynamic
dependencies. A static closure computed in the dirty working tree includes
unrelated untracked validation modules. The launch therefore archives the
scoped commit first and computes source hashes **inside that immutable
snapshot**. Saved geometry design and factor files remain outside it as
separately hashed inputs. This resolves the source mismatch without staging
unrelated shared-worktree code.

Review verdict: the revised cost-only plan answers its engineering question
and does not elevate pilot proxies or old references. It is suitable to launch
after the scoped source commit, snapshot import/compile checks, trusted GPU
probe and memory-growth check. A launch from the dirty checkout would not
meet this plan.

## Checks before the source commit

- The direct filter-adapter source and old reference-input inventory found the
  same five changed inference/tuning files in all four reference archives;
  three geometry archives also include a changed geometry driver. None of the
  target-source or observation-input hashes changed.
- Four current full-horizon independent value/score tests passed in 33.18
  seconds (PP-UKF, PP-SGQF, SIR and structural UKF). These are finite-point
  parity diagnostics, not reference-artifact requalification.
- Five preparation tests passed in 154.48 seconds: the four local/prior
  mapping/status cases and the explicit stale-source rejection/warm-start
  MAMS case. Three new cost-consumer tests passed in 46.38 seconds, including
  a real full-horizon PP-UKF tiny-pilot/native-price smoke and interrupted
  candidate-denominator preservation. CPU devices were deliberately hidden.
- Separate read-only preparation of all four actual saved factors on the
  current target confirmed finite starts, finite value/score and valid status
  at full horizons (20, 20, 20, 100). The saved factors remained labeled
  `warm_start_only`; no reference or posterior inference was made.

The new source commit and source snapshot are the next actions. Reserve at
most 1,800 seconds for this local phase and 7,200 seconds for one bounded
GPU attempt and localized recovery under the unchanged scientific contract.
The pre-reservation settled balance is 154,011.091554 seconds; the provisional
unallocated remainder is 145,011.091554 seconds. Charge actual supervisor
time once and settle/release unused reservations at terminal.

The staged-source tree `6a879950bee14ce43f921c3994969ce1442a8408`
was archived independently at `/tmp/bayesfilter-mams-d4r4a-index-20261012-r2`.
Its three focused cost-consumer tests passed again in 46.86 seconds with GPUs
hidden, including the actual full-horizon PP-UKF tiny-pilot/native-price
path. The trusted TensorFlow probe at 2026-10-11 17:05:49 UTC selected idle
host GPU 1, executed a device operation, and verified memory growth on its
visible device. This probe is readiness evidence only. A fresh committed
archive, not this staged `/tmp` directory, is required for the serious launch.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Proceed to bounded cost-only GPU pricing | Focused wiring and full-target start checks pass | Reference provenance blocks posterior comparison, not price; no shared preflight numerical defect | GPU/XLA elapsed and pilot stability unmeasured | Commit source, snapshot, probe GPU, launch four-target price | Posterior validity, robust adaptation, method ranking |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Prelaunch source/target checks pass for cost-only role. |
| Statistically supported ranking | None planned or claimed. |
| Descriptive-only differences | CPU smoke, local factors, old grid timings and future short-path prices. |
| Default-readiness | No; experimental MAMS route only. |
| Next evidence needed | Trusted GPU price, then independent references and paired full posterior study. |
