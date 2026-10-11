# D4R4a adaptive MAMS full-model pricing execution

Plan: [full-model pricing](bayesfilter-mams-d4r4a-full-model-pricing-2026-10-12.md).
Date: 12 October 2026. Status: **both GPU attempts terminal; four-target
cost and short-path audit passed after scoped SIR repair**. No posterior
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

The source commit and source snapshot are complete. Reserve at
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

## Committed launch and resumed review

Source commit: `ca3c26034565aabb6eaab329714edb7bf0d8101c`; tree:
`a66c30783705570992e50b2e833928399f114a1e`. The immutable source is
`N5/generic-adaptation/source-ca3c26034/`, archived from that commit with
`bayesfilter`, `scripts` and `tests`. The separately copied compiled operator
hash is `2feb93135f2359e60aca446f52f8468b104199b4d398d4fe6057030df20f23d6`.
Snapshot compilation checks pass; it is not modified during execution.

The resumed skeptical review checked the actual supervisor and worker: the
7,200-second parent cap includes the 60-second cooperative allowance, all four
target rows survive interruption, each generated proposal is listed before
pricing, and old posterior reference means are never loaded. The review
passes for cost and short-path health only. The strict reference mismatch
still prevents a posterior comparison.

A fresh trusted GPU probe passed at 2026-10-11 17:09:53 UTC (12 October
01:09:53 Asia/Shanghai); host GPU 1 was idle and TensorFlow 2.20.0 performed
a device operation with memory growth verified. Worker PID 1927342, parent
PID 1927340, began at 17:11:41 UTC / 01:11:41 Asia/Shanghai. The manifest
records GPU 1, XLA enabled, TF 2.20.0, TFP 0.25.0, and verified preinitialization
memory growth. The first XLA cluster compiled. Initial preparation is still
work in progress, not a terminal pilot failure. Session: `70657`.

Actual command, from the immutable snapshot directory:

```bash
env CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/price_mams_adaptation_full_models.py \
  --geometry-root /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/full-model-pricing-01 \
  --targets NL-PP-SGQF NL-PP-UKF NL-SIR NL-STR --seed 5200 --max-wall-seconds 7200
```

Standard output and error go to `N5/generic-adaptation/full-model-pricing-01.log`.
The actual command includes this redirection. The terminal audit must charge
supervisor elapsed once, inspect every target and proposal, compare source,
operator and geometry hashes, and refresh the next phase from observed costs.

Attempt 01 terminated after 652.0493955072016 supervisor seconds. PP-SGQF
completed in 380.479 seconds with two healthy 32-transition short paths;
PP-UKF completed in 245.900 seconds with two healthy short paths. SIR failed
during preparation after 19.060 seconds:
`target_status_telemetry conditioning fields must be both present or both absent`.
Structural remained `not_started_repair_required`. The SIR archived source
contains the companion condition statistic, but the committed cost-run
snapshot omitted its still-uncommitted implementation. Thus the prior CPU
checks, which ran from the dirty working tree, were not an adequate isolated
four-model preflight. This is an infrastructure/source-integration defect,
not a failed MAMS pilot or evidence against SIR sampling.

The localized repair is described in the plan. It includes the existing SIR
status-only condition-number recurrence and expands the full-horizon tiny
proposer/native-price regression to all four models. A new isolated source
commit and fresh `full-model-pricing-02` root will run SIR and structural
only, with original per-target seeds. The remaining GPU allowance is
`floor(7200 - 652.0493955072016) = 6547` seconds. Attempt 01 and its
successful PP rows remain intact. No changed math target, data, method,
promotion criterion, hardware class or privacy boundary is authorized by
this repair; failure to establish the status-only property would veto retry.

The isolated staged tree `16ab3cac2d8186b5be8f05d0bb216deb592698ae`
included the SIR telemetry repair, all four true-horizon price smokes, the
unchanged MAMS implementation and the copied compiled operator. With GPUs
hidden, nine focused tests passed in 459.66 seconds: one warm-start guard,
four real full-horizon tiny proposer/native-price paths, one interrupted
inventory regression, two SIR value/score checks and one paired-field
validator check. JUnit evidence is at
`N5/generic-adaptation/checks-01/full-model-sir-repair-isolated.xml`.
The original and repaired SIR programs were also run in separate CPU-only
processes at two fixed full-horizon parameter points. Their value, score,
status code and validity mask were exactly equal; the repaired result added
only `innovation_condition_estimate`. The recorded comparison is
`N5/generic-adaptation/checks-01/sir-status-repair-value-score-parity.json`.
Finite-point equality and the inspected status-only source diff support this
localized repair; they do not requalify the historical posterior references.

The repair commit is `9c33936693d3f96816bd84fd723cd7dfa65e0bd6`,
tree `33aa725c438c025913c0ee3d8e5b71dab3e2a41a`. Attempt 02 runs from its
immutable archive at `N5/generic-adaptation/source-9c3393669/`, with the same
compiled operator SHA-256 as Attempt 01. The archived source compiles. A fresh
trusted GPU 1 TensorFlow probe passed at 2026-10-11 17:38:39 UTC, confirming
device execution and memory growth. The retry began at 17:48:54 UTC; parent
PID 1955033, worker PID 1955035, session `40374`. Its manifest records XLA,
the source commit/tree, GPU 1 and preinitialization memory growth. It entered
SIR adaptation, so the previously failing source-validation boundary is
cleared. No SIR result or structural cost claim is made before terminal audit.

Actual retry command, from the immutable `source-9c3393669` directory:

```bash
env CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/price_mams_adaptation_full_models.py \
  --geometry-root /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/full-model-pricing-02 \
  --targets NL-SIR NL-STR --seed 25200 --max-wall-seconds 6547
```

The retry log is `N5/generic-adaptation/full-model-pricing-02.log`. The
remaining allowance is shared with Attempt 01, not a new 6547-second campaign.

## Terminal outcome and budget settlement

Attempt 02 completed normally with a 539.426640639082-second supervisor
charge. SIR and structural each completed two candidate prices, with no
hard short-path veto. The independent terminal auditor verified all 177
source hashes, operator and geometry inputs, pilot work and seed accounting,
all four price bundles, GPU/XLA execution metadata and the per-target
candidate denominator. Attempt 01's terminal audit checked its 177 source
hashes and all four recorded PP bundles while preserving the SIR preparation
failure and unstarted structural row. The combined audit checked 80 distinct
seed pairs across the selected four completed target rows and explicitly
records the two source commits. It does not combine those rows into a
same-source method comparison.

Both attempts charged 1,191.476036146283 GPU supervisor seconds; the unused
6,008.523963853717 seconds of the original 7,200-second allocation are
released. Conservatively charging all 1,800 reserved local-check seconds
leaves 151,019.61551755294 aggregate worker-seconds in the existing campaign,
after its preexisting overhead reserve. No new budget was granted or needed.
The actual 32-transition calls ranged from 3.018 to 14.838 seconds; the
complete per-target cost and work table is in the
[result note](bayesfilter-mams-d4r4a-full-model-pricing-result-2026-10-12.md).

All eight final short paths were healthy, but every source and slow-observable
length estimate was unusable under the configured two 64-draw windows and
maximum lag 16. The only returned durations came from the declared
`sqrt(d)`/factor-two neighborhood. This pilot cannot establish successful
length learning. The old full-model posterior references still fail their
strict source checks against either committed price snapshot. They remain
excluded from this cost result and must receive a versioned compatibility
audit or replacement before dependent posterior comparisons.

The reference audit was repeated against the **actual committed launch
snapshot**, rather than the earlier dirty working tree. The saved PP-SGQF,
PP-UKF and structural references have nine changed source files each; SIR
has ten. The changes include other agents' uncommitted NeuTra and nonlinear
files that are present in the working tree but absent from this launch commit.
Four or five untracked diagnostic reference/driver files in each old source
closure are also absent from the cost-only snapshot. Observation input hashes
still match for all four. The SIR target-source difference adds an innovation
condition-number diagnostic without changing its returned value or score, but
the full source-compatibility bridge has **not** been established. These
differences do not invalidate this declared cost-only snapshot, whose exact
target source and geometry-input hashes are in its manifest. They prevent
reuse of those old references for a posterior claim until a versioned
dependency audit or fresh references pass on the chosen D4R4b source.

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
