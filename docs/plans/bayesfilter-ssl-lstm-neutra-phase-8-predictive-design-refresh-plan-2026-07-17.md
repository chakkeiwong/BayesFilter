# SSL-LSTM NeuTra Phase 8 Predictive Design Refresh Plan

Date: 2026-07-17

Status: `CONTROLLED_POWER_REPAIR_LADDER_EXHAUSTED_PHASE9_CLOSED`

## Objective And Entry

Resolve the valid no-oracle comparator and calibrate an immutable one-to-ten-
step predictive comparison design without opening confirmatory G/H forecast
outcomes.

Entry authority:

- Phase 7 decision `PHASE7_RETAINED_ADMISSION_PASSED_PHASE8_HANDOFF`;
- Phase 7 public receipt SHA-256
  `b79e5f6041e284de40bbd3834cc909fd12f45d012f172e570acccaa62dbe31a5`;
- G/H independently admitted at `512` retained draws per chain;
- no posterior oracle and no admitted ordinary-HMC comparator; and
- the historical A3 LGSSM oracle/statistics implementation is engineering
  evidence only, with provisional margins and known power gaps.

This plan authorizes read-only audit, focused CPU-hidden checks, and design
implementation/tests. It does not yet authorize a material calibration run,
GPU forecast run, opening confirmatory G/H forecast banks, or a predictive-
equivalence decision. A resource and exact-command amendment follows the
design audit and a small timing canary.

## Research Intent And Evidence Contract

| Field | Contract |
| --- | --- |
| Main question | Can we freeze a statistically valid and sufficiently powered design for comparing the predictive laws induced by admitted G/H without treating either as posterior truth? |
| Comparator | G and H are independently trained/sampled peer replications on the common locked target; analytic LGSSM and controlled synthetic alternatives validate machinery/power only |
| Primary design pass | Under predeclared null and material controlled alternatives, all required validity checks and null/power criteria pass at the frozen compute budget |
| Promotion vetoes | Underpowered required mean/variance/dependence alternatives, invalid simultaneous intervals, invalid cluster bootstrap, unstable/singular covariance after the frozen ridge rule, invalid omnibus statistic, or comparator ambiguity |
| Hard continuation vetoes | Phase 7/source/hash drift, confirmatory G/H forecast leakage into calibration, wrong forecast equation/horizon identity, nonfinite computation, invalid random hierarchy, GPU/XLA route failure for serious execution, or resource exhaustion |
| Explanatory only | Raw power curves, bandwidth/weight diagnostics, third/fourth moments, quantiles, parameter summaries, runtime, and individual G/H predictive differences before Phase 9 |
| Nonclaims | No posterior truth, predictive equivalence, parameter correctness, superiority, model adequacy, or default readiness from Phase 8 calibration |
| Result artifact | One design-audit/calibration receipt under `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/` plus a Phase 8 result or blocker |

## Skeptical Pre-Execution Audit

The first action is a source-and-artifact audit, not a calibration run. It must
answer:

1. Which A3 components are production-reusable TensorFlow/TFP code versus
   LGSSM-only fixtures or reporting adapters?
2. Do the SSL-LSTM forecast operator and retained archive reader preserve the
   chain/draw/forecast/horizon hierarchy required by cluster inference?
3. Can calibration use analytic/controlled fixtures and fresh generated
   pseudo-draws without reading any G/H confirmatory forecast outcome?
4. Which A3 margins, weights, bandwidths, block length, bootstrap count,
   sample counts, and seeds were explicitly provisional and therefore must not
   be inherited?
5. Does every proposed artifact answer design validity/power rather than
   silently answer the future G/H comparison?

The audit must reject weak-baseline-only comparison, proxy promotion,
post-outcome tuning, environment mismatch, and commands whose artifacts cannot
distinguish underpower from scientific similarity.

Audit disposition, 2026-07-17:

| Surface | Disposition |
| --- | --- |
| `ssl_lstm_predictive_tf.py` | Reusable: locked horizon 10, `float64`, TensorFlow/XLA path, terminal covariance checks, Philox innovation families, role/arm separation, tensor hashes, complete-path cluster identity, and device provenance |
| `predictive_equivalence.py` summaries | Reusable: mean/log-variance plus explanatory moments/quantiles/cross-horizon covariance with strict shape/finite contracts |
| Dependence-aware resampling | Reusable: fixed-chain hierarchical draw-block and forecast-replication indices; chain population is not resampled |
| Simultaneous intervals and bounded MMD | Reusable mechanics and fail-closed classification; inference still requires freshly calibrated inputs and admissibility claims |
| A3 LGSSM oracle | Reusable engineering/machinery control only; not an SSL-LSTM comparator or posterior oracle |
| A3 fixture constants | Prohibited as Phase 8 defaults: margins, MMD tolerance, bandwidths, mixture weights, block length, bootstrap/coverage counts, draw/forecast counts, perturbation sizes, alphas, and seeds are explicitly `A3_TEST_FIXTURE_ONLY_NOT_A4_FROZEN` |
| Long-run covariance | Gap found: no production-owned chain-aware spectral covariance plus deterministic SPD ridge policy existed |

The long-run covariance gap is being repaired in
`bayesfilter/inference/predictive_equivalence.py`. For chain/draw/feature input
`X[c,d,f]`, contiguous within-chain batch means estimate spectral covariance;
division by total chain-draw count gives pooled-mean covariance. The prospective
ridge ladder is `(0,1e-12,1e-10,1e-8,1e-6)` times
`max(mean(diag(covariance)),1)`. Select the first finite positive-definite
candidate with condition number `<=1e8`; otherwise fail closed without a
precision matrix. This is an engineering policy for the upcoming calibration,
not evidence that a final predictive weighting is calibrated.

The audit found no confirmatory G/H forecast bank or predictive comparison
artifact. Phase 7 private shards have sampler evidence only.

## Comparator Resolution

Phase 8 freezes this hierarchy prospectively:

1. Primary peer comparison: independently admitted G versus H predictive laws.
2. Machinery control: exact scalar-LGSSM forecast oracle and direct equation
   simulation.
3. Power controls: identical-law null pairs plus controlled mean, variance,
   skew, and cross-horizon-dependence alternatives.
4. Ordinary HMC is excluded because its retained archive is not admitted.

G and H are not truth/reference arms. Phase 9 may conclude only bounded
predictive stability/equivalence under the calibrated design, never posterior
correctness.

## Design To Calibrate

- Horizons remain `1..10` unless the audit finds an equation/identity defect.
- Co-primary horizon features are predictive mean and log variance.
- Start with equal scientific horizon weights.
- Estimate chain-aware long-run covariance and freeze a deterministic positive-
  definite ridge escalation/failure policy before calibration outcomes.
- Use bounded characteristic-function or RBF-MMD features for the joint path
  omnibus check. Raw empirical MGF weighting remains motivation only and is
  not admissible because of tail instability.
- Treat third/fourth central moments, quantiles, and cross-horizon covariance
  as explanatory unless the calibration prospectively demonstrates power.
- Preserve both common-random-number and independent forecast-bank designs;
  neither alone may support Phase 9.

## Calibration And Leakage Boundary

Calibration inputs may include the exact LGSSM oracle, controlled alternatives,
and fresh synthetic/noise banks with their own seed family. It must not read:

- any future Phase 9 G/H forecast bank;
- any Phase 9 feature, interval, or omnibus result; or
- G/H predictive differences used to select margins, weights, bandwidths,
  block length, bootstrap count, or forecast replication count.

The admitted Phase 7 samples may be inspected only for shape, lineage, and
cost planning during Phase 8. If scale calibration requires target-relevant
values, the plan must freeze a blinded/pilot split before computing them and
exclude pilot draws from Phase 9 confirmation.

## Required Checks

- exact Phase 7 receipt and private-shard hash replay;
- exact SSL-LSTM terminal-state/forecast equation and horizon-order tests;
- A3 LGSSM oracle replay on CPU-hidden reference and trusted GPU/XLA canary;
- identical-law null calibration with uncertainty on achieved error rate;
- controlled mean, variance, skew, and dependence alternatives;
- chain/draw/forecast cluster-bootstrap reconstruction;
- common versus independent bank identity and seed separation;
- singular/near-singular covariance and ridge escalation/failure fixtures;
- characteristic-function/RBF omnibus symmetry, identity, and material-
  alternative fixtures;
- proof that failure to reject equality cannot emit equivalence;
- strict no-confirmation-leakage guard; and
- one focused statistical/native review before any material calibration run.

## Resource And Sequential Design

Start with a CPU-hidden import/shape/replay audit and one tiny trusted GPU/XLA
forecast timing canary. These are engineering-only. From measured timing,
freeze:

- calibration seed count and replication ladder;
- forecast replications per chain/draw;
- bootstrap count and block/cluster policy;
- CPU sample-generation worker count;
- GPU/XLA calibration cap; and
- exact commands and output paths.

The material calibration ladder continues after a missed power target when the
next rung was designed to repair Monte Carlo underpower. It stops for invalid
machinery, leakage, exhausted cap, or failure at the maximum prospective rung.

The engineering canary uses the four fixed A0 start points, not retained G/H
draws, with horizon 10 and two forecast replications. It runs both a shared
Philox bank and two independent-arm banks, checks exact role/seed/tensor
separation, compiles terminal/forecast/statistics/covariance programs under
trusted GPU/XLA, and records output placement and timing. Its forecasts are
mechanics-only and cannot calibrate or decide Phase 9.

## Engineering Canary Freeze

Skeptical audit disposition: `PASS_CANARY_ONLY`. The canary uses the correct
Phase 7 handoff and A0 mechanics points, does not read retained samples or a
confirmatory forecast bank, cannot promote a design from proxy metrics, has a
resource stop, and writes telemetry that answers only GPU/XLA mechanics and
timing. It fails closed unless terminal, forecast, summary, and long-run-
covariance numerical outputs are GPU resident and all four fixed-shape XLA
surfaces have exactly one trace.

Frozen command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-cuda timeout 1260s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_predictive_design_canary_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/engineering-canary.json --wall-cap-seconds 1200
```

Resource contract:

- physical GPU 1 only; do not use or interrupt occupied GPU 0;
- one invocation and at most `1200` seconds of runner wall time;
- outer `timeout` adds only `60` seconds for cancellation overhead;
- stop without retry for receipt/source drift, nonfinite values, failed terminal
  covariance, GPU placement failure, XLA retracing, innovation-bank overlap, or
  cap exhaustion; and
- a passing canary authorizes only audit of its receipt and prospective design
  of the material calibration ladder. It does not authorize that ladder.

Frozen output:

`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/engineering-canary.json`

Pre-canary focused review:

`docs/reviews/bayesfilter-ssl-lstm-neutra-phase-8-pre-canary-native-review-2026-07-17.md`

## Engineering Canary Failure And Repair 01

The frozen first invocation completed the GPU/XLA numerical path and reached
receipt construction, then exited `1` before writing any file because a scalar
TensorFlow string status became Python `bytes`, which the strict JSON adapter
did not decode. No partial receipt exists. This is an artifact-serialization
implementation failure: it invalidates the first invocation as evidence but
does not identify a forecast, covariance, GPU, XLA, or scientific failure.

Repair 01 changes only `_json_safe` to decode `bytes` as UTF-8 and adds a
focused raw-bytes plus `tf.string` round-trip regression. It does not change
points, forecast equations, seeds, banks, numerical policies, gates, or caps.
The complete canary must be rerun because console completion without an
immutable receipt is inadmissible.

Repair-01 skeptical audit disposition: `PASS_REPAIR_CANARY_ONLY`. The failure
cause is reproduced by the focused serializer contract, the patch is narrower
than the scientific computation, the original output remains absent, and the
new path preserves visible attempt identity. No material calibration is
authorized.

Frozen repair-01 command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-pyc-repair01 CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-cuda-repair01 timeout 1260s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_predictive_design_canary_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/engineering-canary-repair-01.json --wall-cap-seconds 1200
```

The repair stops under the same hard gates as the original canary. No further
automatic canary retry is authorized if repair 01 fails.

Repair 01 passed with decision
`PHASE8_ENGINEERING_CANARY_PASSED_RESOURCE_FREEZE_REQUIRED` in
`194.53392069903202` seconds. The immutable receipt is
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/engineering-canary-repair-01.json`,
SHA-256
`5924b550b1ca5b18d276bd8ea3a3a15cd27b28f95d25f4a7669bd3804f5a9127`.
All four original compiled surfaces traced once and all terminal, forecast,
summary, and covariance outputs were GPU resident. The canary selected ridge
`1e-6` for its deliberately singular four-point fixture at condition number
`2411219.6469682734`; this is mechanics evidence only, not a calibrated weight.

## Target-Pilot Split And Evidence Contract

The A2 evidence forecast contract fixes two forecast replications. Phase 8
therefore gains serious precision from retained draws rather than changing the
accepted forecast configuration. The Phase 7 `512` retained draws per chain
are prospectively partitioned for each chart:

- pilot prefix: indices `0..63`, `64` draws per chain, permanently excluded
  from Phase 9;
- confirmation suffix: indices `64..511`, `448` draws per chain; and
- block length candidate `16`, giving four pilot and 28 confirmation blocks
  per chain. Its final inferential use still requires controlled calibration.

TensorFlow's serialized tensor format requires deserializing the full first
`256`-draw shard to select indices `0..63`. The pilot runner records this
explicitly. It never selects, maps, forecasts, summarizes, emits, or decides on
indices `64..255`; the second `256`-draw shard is hash-verified without tensor
deserialization. This is computational exclusion, not a claim that no bytes
outside the prefix were decoded.

Target-pilot question: can a label-pooled, excluded G/H prefix provide finite
target-specific predictive center/scales, a nondegenerate median path-distance
bandwidth ladder, and an admissible chain-aware covariance surface without
computing an arm difference or opening a confirmation forecast bank?

Target-pilot primary pass:

- all four Phase 7 manifest/shard/sidecar hash bindings replay;
- only the fixed prefix enters mapping, forecast, or statistics;
- transport mapping and forecast execution are finite, trusted GPU/XLA, and
  single-trace;
- all terminal covariance statuses are valid;
- independent pilot innovation families are disjoint;
- pooled horizon scales exceed their prospective numerical floors;
- the pooled positive off-diagonal Euclidean path-distance median is finite and
  positive;
- bandwidth factors remain `(0.25,0.5,1,2,4)` around that median;
- the block-16 influence covariance admits the prospective ridge ladder under
  condition cap `1e8`; and
- no arm-specific predictive summary, G/H difference, or Phase 9 decision is
  emitted.

Pilot vetoes are source/receipt/hash drift, wrong split, use of excluded suffix
values, nonfinite mapping/forecast/statistics, terminal covariance failure,
innovation reuse, zero/floored scales, degenerate path distances, covariance
ridge exhaustion, GPU/XLA/trace failure, serialization failure, or resource
exhaustion. Timing, pooled scale values, bandwidths, and the pilot ridge are
explanatory/calibration inputs only. A pass does not establish calibrated
power, predictive equivalence, posterior truth, ranking, model adequacy, or
Phase 9 readiness.

Skeptical audit disposition: `PASS_TARGET_PILOT_ONLY`. The pilot answers a
target-specific scale/feasibility question that synthetic controls cannot
answer, but label pooling and the no-difference guard prevent it from deciding
the G/H comparison. It preserves a much larger contiguous confirmation suffix
and does not inherit A3 fixture constants. The exact command has a resource
stop and a fresh output.

Focused review:

`docs/reviews/bayesfilter-ssl-lstm-neutra-phase-8-target-pilot-native-review-2026-07-17.md`

Frozen target-pilot command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-pilot-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-pilot-cuda timeout 1260s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_target_pilot_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-pilot.json --wall-cap-seconds 1200
```

Resource contract: one invocation on physical GPU 1, at most `1200` runner
seconds plus `60` seconds cancellation margin. Stop without automatic retry on
any veto. A pass authorizes receipt audit and implementation/freeze of the
controlled null/power calibration only; it does not itself authorize that
material calibration run.

## Target-Pilot Failure And Terminal Diagnostic

The target pilot exited `1` before writing a receipt. G mapping succeeded, but
the terminal-state gate returned status `8` (`STATUS_PROJECTION`) for six of
the `256` selected G prefix points. All reported terminal fields were finite;
no status indicated nonfiniteness, asymmetry, material indefiniteness, factor
reconstruction, filter parity, or target parity. No forecast tensor and no H
mapping/terminal/forecast computation was produced. The failed pilot remains
failed and no automatic pilot retry is authorized.

This failure invalidates the current forecast input surface for the pilot; it
does not reject the admitted sampler, G/H transports, predictive-validation
idea, or NeuTra research direction. Status `8` may represent either an overly
tight eigendecomposition reconstruction guard or a real projection defect.
The next smallest discriminating action is therefore terminal-only
localization, not another pilot.

Diagnostic question: on the exact G prefix points, do GPU/XLA and CPU
TensorFlow audits of the same raw `3x3` terminal covariance identify projection
residual alone, with symmetry, minimum eigenvalue, and factor reconstruction
inside their existing bounds?

The diagnostic runs the G transport map and terminal compiled program only,
records per-point residual-to-`tau` ratios, and re-audits the returned raw
covariances with TensorFlow on CPU. It performs no forecast, predictive
summary, G/H difference, power calibration, or suffix selection. A
projection-tolerance repair may be nominated only if all GPU failures are
projection-only, negative-eigenvalue and symmetry ratios are at most one, and
factor-reconstruction ratios remain at most 16. The diagnostic cannot itself
authorize a tolerance change or pilot rerun.

Skeptical audit disposition: `PASS_TERMINAL_LOCALIZATION_ONLY`. The exact
failure is reproduced rather than inferred from a different point set; the
command directly distinguishes material covariance invalidity from a
projection-reconstruction guard; output is fresh and capped; no proxy can
promote the design.

Frozen diagnostic command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-terminal-diag-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-terminal-diag-cuda timeout 660s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/diagnose_ssl_lstm_neutra_phase8_terminal_projection_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/terminal-projection-diagnostic.json --wall-cap-seconds 600
```

Resource contract: one terminal-only invocation on physical GPU 1, at most
`600` runner seconds plus `60` seconds cancellation margin. Stop without retry
on non-reproduction, any non-projection terminal status, placement/trace/hash
failure, serialization failure, or cap exhaustion.

The diagnostic passed in `184.58174790698104` seconds. Receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/terminal-projection-diagnostic.json`,
SHA-256
`ea2ded5e9e3321c18048a4306606c3d2dcd12fbc728050152ebef0ba521c0bcc`.
It reproduced exactly six G failures at indices
`[33,68,144,189,200,201]`, all status `8`; CPU TensorFlow re-audit of the
same raw covariance tensors had zero failures. GPU minimum eigenvalues were
strictly positive, symmetry residuals were zero, and maximum factor-
reconstruction residual was `0.031251*tau`, but projection ratios were
`2.01e6` to `1.41e7`. CPU projection ratios were below `0.047`.

This pattern cannot be repaired by changing `8*tau` to another roundoff-scale
constant. It identifies an eigenvector convention mismatch: the XLA CUDA
batched path supplies eigenvectors in the equivalent row orientation for the
failing points, while the implementation unconditionally reconstructs as if
vectors were columns. The factor reconstructs its own incorrectly oriented
covariance, explaining why factor residuals remained tiny.

The numerical repair evaluates both `Q diag(lambda) Q^T` and
`Q^T diag(lambda) Q`, selects the orientation with smaller Frobenius residual
against the symmetrized raw covariance, and constructs the principal square
root using the same selected orientation. It does not change eigenvalues,
roundoff clipping, `tau`, the material-negative policy, projection multiplier
`8`, factor multiplier `16`, target/filter equations, forecast equation, or
randomness. The historical A2 semantic contract already requires the
principal eigen-square-root; this corrects its backend implementation rather
than changing that semantic contract.

Before any pilot retry, a new terminal-only GPU validation must evaluate both
G and H excluded prefixes and require all `512` terminal statuses to be zero,
all fields finite, all outputs GPU resident, one trace per fixed program, and
projection/factor residuals within the unchanged gates. It must also retain the
original diagnostic receipt and failed-pilot record. A pass authorizes a
separately named target-pilot repair attempt; it cannot itself authorize
forecasts, calibration, or Phase 9.

Skeptical audit disposition for exact-prefix validation:
`PASS_TERMINAL_REPAIR_VALIDATION_ONLY`. It binds the diagnostic receipt,
evaluates the exact excluded G/H prefixes rather than a proxy fixture, leaves
all numerical gates unchanged, cannot execute a forecast, and has a fresh
receipt plus bounded resource stop.

Frozen validation command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-terminal-validation-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-terminal-validation-cuda timeout 660s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/validate_ssl_lstm_neutra_phase8_terminal_orientation_repair_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/terminal-orientation-validation.json --wall-cap-seconds 600
```

Resource contract: one invocation on physical GPU 1, at most `600` runner
seconds plus `60` seconds cancellation margin. Stop without repair/retry for
any nonzero terminal status, nonfinite field, residual-gate failure, GPU/XLA or
trace failure, receipt/hash drift, serialization failure, or cap exhaustion.

The exact-prefix validation exited `1` before writing a receipt. The same six
G indices retained status `8` with numerically identical projection residuals;
H was not evaluated. Therefore the row-versus-column eigenvector hypothesis is
falsified, the candidate repair is rejected, and its production source change
has been reverted. No tolerance changed and no pilot retry occurred.

The next discriminating diagnostic operates on the exact raw G terminal
covariances and compares, inside trusted GPU/XLA:

- column and row eigen reconstruction residuals;
- eigen-equation residuals;
- TensorFlow SVD reconstruction residuals under both orientations;
- an SVD-derived symmetric PSD covariance/root reconstruction; and
- Cholesky reconstruction for the strictly-positive subset.

This is decomposition localization only. It may nominate a backend-stable
implementation route but cannot change production code, execute forecasts, or
retry the target pilot.

Skeptical audit disposition: `PASS_RAW_DECOMPOSITION_LOCALIZATION_ONLY`. The
diagnostic binds the exact prior failure receipt and exact G prefix, compares
candidate decompositions on the same raw covariance tensors, keeps all current
residual thresholds, has no production write or forecast path, and stops under
a fresh bounded receipt.

Frozen command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-decomp-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-decomp-cuda timeout 660s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/diagnose_ssl_lstm_neutra_phase8_raw_covariance_decompositions_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/raw-covariance-decomposition-diagnostic.json --wall-cap-seconds 600
```

Resource contract: one invocation on physical GPU 1, at most `600` runner
seconds plus `60` seconds cancellation margin. Stop without retry for failure
non-reproduction, nonfinite decomposition output, GPU/XLA/trace failure,
receipt/hash drift, serialization failure, or cap exhaustion.

The raw-decomposition diagnostic passed in `174.81534266693052` seconds with
decision `PHASE8_NO_PRINCIPAL_DECOMPOSITION_REPAIR_IDENTIFIED`. Receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/raw-covariance-decomposition-diagnostic.json`,
SHA-256
`32d1051b667df210c7eae4d174731a28fb7d8b9b26c13d590d5742d15f082fbd`.

The key localization is structural, not a new decomposition choice. On the
exact failing raw covariances, a separate batched XLA `eigh` reconstructed all
six within `0.27*tau`, and Cholesky reconstruction was below `0.004*tau`.
The eigensolver inside the per-point `tf.map_fn` terminal core alone emitted
the million-`tau` projection residual. SVD was inconsistent on some points and
is not nominated.

The repair therefore stages covariance audit after terminal filtering:

1. the per-point XLA filter returns raw covariance and all target/filter parity
   fields;
2. a separate fixed-shape batched XLA program symmetrizes, eigendecomposes,
   applies the same permitted roundoff clipping, constructs the same symmetric
   principal root, and applies the unchanged gates; and
3. only covariance-derived tensors and covariance status bits are replaced.
   Nonfinite, filter-parity, and target-parity bits from the filter stage are
   preserved.

No target, filter, forecast, eigenvalue, tolerance, clipping, material-negative,
or random-number policy changes. Focused CPU-hidden checks passed: `3` staged
audit/status tests and `66` Phase 8 predictive tests.

Skeptical audit disposition: `PASS_STAGED_AUDIT_EXACT_PREFIX_VALIDATION_ONLY`.
The validation binds the decomposition receipt, uses the exact G/H excluded
prefixes, calls the public terminal extraction API, requires both compiled
stages to trace once, retains every prior numerical gate, executes no forecast,
and writes a fresh bounded receipt.

Frozen validation command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-staged-validation-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-staged-validation-cuda timeout 660s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/validate_ssl_lstm_neutra_phase8_terminal_orientation_repair_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/terminal-staged-audit-validation.json --wall-cap-seconds 600
```

Resource contract: one invocation on physical GPU 1, at most `600` runner
seconds plus `60` seconds cancellation margin. Stop without retry for any
nonzero terminal status, nonfinite field, unchanged residual-gate failure,
either-stage trace/placement failure, receipt/hash drift, serialization
failure, or cap exhaustion.

The exact-prefix staged validation passed in `180.77984058496077` seconds.
Receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/terminal-staged-audit-validation.json`,
SHA-256
`5aa08e674130c2a8b8e5fd7bf47989c68595f350a6a4d399ed1394f8883fad68`.
All `512` G/H terminal points had zero status. Maximum projection ratios were
`0.9492089905630805` for G and `0.8388212516109781` for H; maximum factor
ratios were `0.031250002430169746` and `0.02343750863413478`. Both terminal
filter and staged covariance programs traced once.

Target-pilot repair 01 is now eligible. It changes no pilot input or
statistical choice: the `64/448` split, pilot root seed `(12001,12002)`, arm
IDs, two forecast replications, block length `16`, bandwidth-factor ladder,
ridge ladder, condition cap, output fields, no-difference guard, and
`1200`-second cap remain identical. The only implementation change is the
validated staged covariance audit inside public terminal extraction. The
runner now requires and records both terminal-stage trace counts.

Skeptical audit disposition: `PASS_TARGET_PILOT_REPAIR_01_ONLY`. The original
failure is understood and independently reproduced; a rejected repair was
preserved and reverted; the nominated repair passed both charts under exact
inputs and unchanged gates; the new pilot path is fresh and cannot emit a G/H
difference or open confirmation forecasts.

Frozen target-pilot repair-01 command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-pilot-repair01-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-pilot-repair01-cuda timeout 1260s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_target_pilot_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-pilot-repair-01.json --wall-cap-seconds 1200
```

Resource contract: one invocation on physical GPU 1, at most `1200` runner
seconds plus `60` seconds cancellation margin. Stop without further automatic
pilot retry on any original pilot veto, either terminal-stage trace failure,
serialization failure, or cap exhaustion. A pass authorizes only receipt audit
and prospective controlled null/power calibration planning.

## Target-Pilot Repair 01 Timeout And Forecast Chunk Repair

Target-pilot repair 01 reached the external `1260`-second timeout with exit
code `124`; no pilot receipt was written. The last observed stage was XLA
compilation of the static `256`-draw forecast program. The attempt therefore
cannot support a pilot, calibration, equivalence, or scientific conclusion.
Its supervisor close record is
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-pilot-repair-01-timeout-record.json`.

This is a resource veto for the frozen attempt, not a recurrence of the staged
covariance defect. The exact G/H terminal validation had already passed all
`512` prefix points, and the timed-out attempt emitted no new covariance
failure. Source audit localized the compile cost: `_forecast_batch_core`
iterates over the static draw dimension in Python, so the `256`-draw graph
unrolls `256 * 10` forecast steps into one XLA module.

The bounded repair adds optional draw-axis chunking to the compiled forecast
API. The default remains unchunked. Terminal extraction and the staged
covariance audit still execute once on the full draw matrix; only the forecast
recursion is split. Each chunk slices the already materialized draw, terminal,
and innovation tensors, invokes the same fixed-shape XLA program, and outputs
are concatenated in original draw order. The pilot freezes chunk size `16`.
No forecast equation, horizon, replication count, terminal rule, covariance
gate, transport, source draw, innovation byte, seed, or statistical choice is
changed.

Focused CPU-hidden validation passed:

- exact chunk-size-1 versus unchunked-size-2 forecast tensor equality and
  unchanged full-bank hashes;
- fail-closed chunk-size API validation;
- `8` chunk-canary and pilot contract tests;
- `58` predictive-statistics tests; and
- Python compilation and `git diff --check`.

Focused review:
`docs/reviews/bayesfilter-ssl-lstm-neutra-phase-8-forecast-chunk-repair-native-review-2026-07-17.md`,
verdict `AGREE_COMPILE_CANARY_ONLY`.

The next smallest discriminating action is an engineering-only trusted GPU/XLA
compile canary. It uses `32` tiled A0 start-derived points, a static chunk size
of `16`, two chunks, fresh independent-arm seed `(13001,13002)`, two forecast
replications, and horizon ten. It reads no Phase 7 retained samples and cannot
emit a G/H difference, target scale, bandwidth, covariance weight, calibration
result, or Phase 9 outcome.

Canary pass requires finite correctly shaped forecast output on GPU, zero
terminal statuses, recorded chunk size `16`, and trace count one for the
terminal-32, staged-covariance-32, and forecast-chunk-16 programs. Source or
terminal-validation drift, nonfinite output, wrong shape/order, status failure,
non-GPU placement, trace failure, serialization failure, or resource exhaustion
is a hard canary veto.

Skeptical audit disposition: `PASS_FORECAST_CHUNK_COMPILE_CANARY_ONLY`. The
canary directly tests the compile shape that repairs the observed resource
failure, uses the original unchunked output as the already-passed parity
baseline, cannot promote a design from proxy evidence, and has a fresh output
plus finite resource stop. A pass authorizes only an exact G/H excluded-prefix
chunk validation. It does not authorize another target-pilot attempt,
controlled calibration, or Phase 9.

Frozen command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-chunk-canary-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-chunk-canary-cuda timeout 660s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/validate_ssl_lstm_neutra_phase8_forecast_chunk_repair_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/forecast-chunk-repair-canary.json --wall-cap-seconds 600
```

Resource contract: one invocation on physical GPU 1, at most `600` runner
seconds plus `60` seconds cancellation margin. Stop without retry on any veto.

The trusted GPU canary passed in `220.9772220539162` seconds. Receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/forecast-chunk-repair-canary.json`,
SHA-256
`e78e76203278548183f7974562249e3a292ae4f21e315cd137b955131e342587`.
The terminal-32, staged-covariance-32, and forecast-chunk-16 programs each
traced once; all terminal statuses were zero and all forecast outputs were GPU
resident. The preliminary sandboxed launch saw no CUDA device and wrote no
receipt; per GPU trust policy it is sandbox evidence only. The same frozen
command produced the receipt in the trusted context.

The exact-prefix validation now evaluates both frozen G/H `0..63` prefixes,
their original transport mappings, pilot seed `(12001,12002)`, independent arm
IDs, two forecast replications, horizon ten, and draw chunk size `16`. It emits
only archive/mapping/bank/output hashes, shapes, devices, statuses, traces, and
timing. It cannot call predictive summaries, compute scales/bandwidths/weights,
or compare G with H.

Exact-prefix pass requires both `256`-point charts to produce finite
`[256,2,10,1]` observations on GPU with zero terminal statuses; all six
innovation-family tensor hashes must be distinct; chunk provenance must equal
`16`; and terminal-256, staged-covariance-256, and forecast-chunk-16 programs
must each trace once. Any archive/source/canary/timeout-record drift,
confirmation suffix selection, nonfinite output, status/shape/order/device/hash
failure, trace failure, serialization failure, or resource exhaustion is a
hard veto.

Skeptical audit disposition:
`PASS_FORECAST_CHUNK_EXACT_PREFIX_VALIDATION_ONLY`. This is the exact target
shape and excluded prefix that timed out, not a smaller proxy. It preserves the
failed-attempt record and passed compile canary, forbids statistical output,
has a fresh receipt, and directly determines whether target-pilot repair 02 may
be considered. A pass does not itself authorize that pilot repair,
calibration, or Phase 9.

Frozen command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-chunk-exact-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-chunk-exact-cuda timeout 960s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/validate_ssl_lstm_neutra_phase8_forecast_chunk_exact_prefix_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/forecast-chunk-exact-prefix-validation.json --wall-cap-seconds 900
```

Resource contract: one invocation on physical GPU 1, at most `900` runner
seconds plus `60` seconds cancellation margin. Stop without retry on any veto.

The exact-prefix validation passed in `239.67237085592933` seconds. Receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/forecast-chunk-exact-prefix-validation.json`,
SHA-256
`f272d6eb407e8d3dbe11ebac2f1dcfac8a16cacd6ce5c92ca074c6bb050cb0d1`.
Both G and H produced finite `[256,2,10,1]` observations with zero terminal
statuses, chunk size `16`, and disjoint innovation tensors. Terminal-256,
staged-covariance-256, and forecast-chunk-16 each traced once. G paid
`229.85232187097427` seconds of compile/first-execution cost; H reused the
compiled programs in `4.960395403089933` seconds. No predictive summary or
G/H difference was computed and no confirmation suffix was selected.

Target-pilot repair 02 is therefore eligible. Its runner now fails closed
unless the exact chunk-canary and exact-prefix receipt hashes and decisions
replay. It preserves the original `64/448` split, pilot seed `(12001,12002)`,
arm IDs, two forecast replications, horizon ten, block length `16`, bandwidth
factors `(0.25,0.5,1,2,4)`, ridge ladder, condition cap, no-difference output
contract, and `1200`-second runner cap. The sole execution-shape change from
repair 01 is the exact-validated forecast chunk size `16`.

Skeptical audit disposition: `PASS_TARGET_PILOT_REPAIR_02_ONLY`. The repair
directly addresses the observed static-unroll timeout, has tensor-exact
unchunked parity on a focused fixture, passed a trusted GPU compile canary and
the exact G/H excluded prefixes, preserves all scientific/statistical inputs,
binds the failed-attempt history, and writes a fresh receipt. It cannot emit a
G/H difference or open confirmation forecasts. A pass authorizes receipt audit
and prospective controlled null/power calibration planning only.

Frozen command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-pilot-repair02-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-pilot-repair02-cuda timeout 1260s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_target_pilot_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-pilot-repair-02.json --wall-cap-seconds 1200
```

Resource contract: one invocation on physical GPU 1, at most `1200` runner
seconds plus `60` seconds cancellation margin. Stop without any further
automatic target-pilot retry on an original pilot veto, chunk-receipt drift,
trace/placement/serialization failure, or cap exhaustion.

## Target-Pilot Repair 02 Failure And Pairwise-Distance Repair

Repair 02 cleared both chart forecasts but failed before any predictive receipt
or G/H difference. XLA raised `CustomCall failed: Buffers have different size
at runtime` in `pooled_pairwise_distance_scale` on the pooled standardized
shape `[8,64,2,10]`. The implementation used `tf.boolean_mask` twice, creating
data-dependent tensor lengths inside a compiled program. Supervisor close
record:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-pilot-repair-02-failure-record.json`,
SHA-256
`f4de95bf6fc540b32d7d2ba7e06002600af5b3fb90512ef1ba5cb6a05bed4abc`.

This is an implementation/kernel-shape failure. It does not reject the target
pilot, bandwidth idea, predictive validation, G/H samplers, or NeuTra direction.
It also does not invalidate forecast chunking: repair 02 reached the pooled
statistics stage only after both chart forecasts completed.

The repaired pairwise primitive preserves the exact original statistic while
using a fixed XLA shape. It computes the full distance matrix, retains only
strict-upper-triangle positive entries by replacing all other entries with
positive infinity, sorts the fixed `N*N` vector, and gathers the positive-entry
median from its scalar count. This is identical to sorting the dynamically
masked positive upper triangle because every finite candidate precedes
infinity. Exact duplicate pairs remain excluded; an all-duplicate cloud still
fails closed.

Focused review:
`docs/reviews/bayesfilter-ssl-lstm-neutra-phase-8-pairwise-distance-repair-native-review-2026-07-17.md`,
verdict `AGREE_EXACT_SHAPE_CANARY_ONLY`. `62` focused distance/statistics/canary
tests passed, plus Python compilation and `git diff --check`.

The next action is a trusted GPU/XLA canary at the exact failed shape
`[8,64,2,10]`. It uses a deterministic TensorFlow range/sine fixture, compares
compiled and eager medians and counts under a scale-aware roundoff tolerance,
requires the compiled median on GPU and trace count one, and reads no retained
sample or forecast artifact.

Skeptical audit disposition: `PASS_PAIRWISE_DISTANCE_EXACT_SHAPE_CANARY_ONLY`.
The command tests the exact failed shape and operation rather than a small
proxy; its eager comparator implements the same statistic without XLA; it binds
the failed-attempt record; and its output cannot promote the target pilot or
calibration. A pass authorizes planning of a separately recorded pilot repair
03 only.

Frozen command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-distance-shape-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-distance-shape-cuda timeout 360s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/validate_ssl_lstm_neutra_phase8_pairwise_distance_shape_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/pairwise-distance-exact-shape-canary.json --wall-cap-seconds 300
```

Resource contract: one invocation on physical GPU 1, at most `300` runner
seconds plus `60` seconds cancellation margin. Stop without retry on any
binding, parity, count, placement, trace, finite, serialization, or resource
veto.

The exact-shape distance canary passed in `3.411637287004851` seconds. Receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/pairwise-distance-exact-shape-canary.json`,
SHA-256
`fd8489ec557c49a0169af8a656d6619535ab41ca85c07b9ba167b593e29c0871`.
At `[8,64,2,10]`, compiled and eager medians were both
`2.288481844607227`, with zero residual, `523776` positive/total
upper-triangle pairs, one XLA trace, and GPU output.

Target-pilot repair 03 is eligible. The pilot runner now hard-binds the distance
canary in addition to the Phase 7 receipt, original engineering canary, chunk
canary, and exact-prefix chunk validation. Repair 03 changes no target-pilot
input or statistical choice from repairs 01/02. It retains the `64/448` split,
pilot seed, arm IDs, two replications, horizon ten, chunk size `16`, block
length `16`, bandwidth factors, ridge ladder, condition cap, output/no-
difference contract, and `1200`-second cap. Its sole new implementation is the
exact-shape-validated fixed-size median calculation.

Skeptical audit disposition: `PASS_TARGET_PILOT_REPAIR_03_ONLY`. The new failure
was localized after forecasts, repaired without changing the statistic, tested
against manual/eager formulas, and validated at the exact failed XLA shape.
All prior failed attempts remain preserved. A pass authorizes only receipt audit
and prospective controlled null/power calibration planning. Any failure stops
the target-pilot lane without another automatic repair attempt.

Frozen command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-pilot-repair03-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-pilot-repair03-cuda timeout 1260s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_target_pilot_2026_07_17.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-pilot-repair-03.json --wall-cap-seconds 1200
```

Resource contract: one invocation on physical GPU 1, at most `1200` runner
seconds plus `60` seconds cancellation margin. Stop without further automatic
target-pilot retry on any source/receipt/split/finite/status/seed/distance/
ridge/GPU/XLA/trace/serialization/resource veto.

Repair 03 passed in `240.57270147104282` seconds with decision
`PHASE8_TARGET_PILOT_PASSED_CONTROL_CALIBRATION_REQUIRED`. Receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-pilot-repair-03.json`,
SHA-256
`5ae511c248e222edf14660c91c4a48412706c6f452298ebe5e144bbe8f01c098`.

Receipt audit passed every prospective gate:

- prefix `0..63` only; suffix `64..511` remains unopened and unused;
- no arm-specific predictive summary or G/H difference;
- all six compiled surfaces traced once and all evidence outputs were GPU
  resident;
- both charts had 256 zero terminal statuses and disjoint innovation tensors;
- all horizon scales were finite and strictly above floor;
- median positive path distance was `4.291580961404927`, with bandwidths
  `[1.0728952403512317,2.1457904807024635,4.291580961404927,
  8.583161922809854,17.166323845619708]`;
- all `523776` upper-triangle pairs were positive; and
- block-16 influence covariance selected ridge `0`, condition number
  `1125.3010946853005`, under the `1e8` cap.

## Controlled Calibration Smoke And Nomination Freeze

The controlled design uses the exact confirmation evidence shape per arm:
`[4 chains,448 draws,2 forecast replications,10 horizons]`. It generates fresh
synthetic TensorFlow/Philox controls only; it does not read retained G/H values,
forecast banks, summaries, or differences.

Frozen co-primary design:

- mean margin `0.15` predictive SD, strictly below material `0.20`;
- log-variance margin `log(1.15) = 0.13976194237515863`, strictly below
  `|log(1.25)| = 0.22314355131420976`;
- total alpha `0.05`, feature alpha `0.03`, MMD alpha `0.02`;
- Bonferroni/studentized simultaneous intervals for 20 features;
- block length `16`, giving 28 blocks per chain;
- five target-pilot bandwidths with equal weights `0.2`;
- MMD tolerance candidate ladder `(0.005,0.01,0.02,0.04,0.08,0.16)`;
- ridge ladder `(0,1e-12,1e-10,1e-8,1e-6)` and condition cap `1e8`; and
- chain pairs `[(0,1),(2,3)]`.

Required families include iid and AR identical-law nulls, true-equivalent
persistent mean `0.05` and variance `1.05`, persistent and horizon-1 mean
`+/-0.20`, and persistent/horizon-1 variance ratios `1.25` and `0.80` where
applicable. Skew and changed cross-horizon dependence remain explanatory.

The first action is a two-family full-shape smoke: one iid null and one
persistent `+0.20` mean alternative, one replication each. It exercises every
generation, covariance, interval, MMD, decision, placement, trace, and artifact
surface but always returns `PHASE8_CONTROLLED_CALIBRATION_SMOKE_PASSED_NOMINATION_REQUIRED`.
It cannot nominate or validate a tolerance.

Skeptical audit disposition: `PASS_CONTROLLED_CALIBRATION_SMOKE_ONLY`. The smoke
uses the exact confirmation shape rather than a small proxy; the null and
material arm catch reversed or inert decisions; all scientific/statistical
settings are frozen before outcomes; and output cannot select a design. A pass
authorizes the separately frozen 20-replication nomination only.

Focused review:
`docs/reviews/bayesfilter-ssl-lstm-neutra-phase-8-controlled-calibration-native-review-2026-07-17.md`,
verdict `AGREE_SMOKE_ONLY`.

Frozen smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-calibration-smoke-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-calibration-smoke-cuda timeout 660s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_controlled_calibration_2026_07_17.py --mode smoke --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/controlled-calibration-smoke.json --wall-cap-seconds 600
```

Resource contract: one invocation on physical GPU 1, at most `600` runner
seconds plus `60` seconds cancellation margin. Stop without retry on source/
pilot drift, nonfinite output, invalid covariance/MMD/decision, non-GPU output,
retracing, serialization failure, or cap exhaustion.

### Controlled Calibration Smoke Failure And Repair 01

The original smoke initialized trusted GPU 1 and compiled its first XLA
cluster, then exited `1` before writing a receipt. The runner called
`tf.concat` without its required `axis` argument while constructing the fixed
20-feature margin vector. The intended vector was already prospectively frozen
as ten mean margins followed by ten log-variance margins, but that exact
orchestration line had not been executed by the CPU-hidden tests.

Failure record:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/controlled-calibration-smoke-failure-record.json`.
The original output remains absent. This is an implementation failure that
invalidates the invocation as smoke evidence. It is not covariance, MMD,
GPU/XLA, calibration-power, or scientific-direction evidence.

Repair 01 adds `axis=0`, factors the construction into `_feature_margins`, and
adds a direct regression requiring shape `[20]`, with the first ten values
equal to `0.15` and the last ten equal to `log(1.15)`. It changes no data,
family, seed, margin value, alpha, bandwidth, MMD tolerance candidate, ridge,
condition cap, shape, trace/placement gate, decision rule, or resource cap.
Focused repair checks passed: `9` controlled-calibration tests; Python
compilation; and `git diff --check`.

Skeptical audit disposition: `PASS_CONTROLLED_CALIBRATION_SMOKE_REPAIR_01_ONLY`.
The failure is deterministic and directly reproduced by the missing required
argument; the patch implements the already-frozen vector rather than modifying
the design; the regression executes the repaired surface; and the rerun has a
fresh output path. A repair pass may authorize nomination planning only. Any
repair failure stops before nomination.

Frozen repair-01 command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-calibration-smoke-repair01-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-calibration-smoke-repair01-cuda timeout 660s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_controlled_calibration_2026_07_17.py --mode smoke --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/controlled-calibration-smoke-repair-01.json --wall-cap-seconds 600
```

Resource contract: one repair invocation on physical GPU 1, at most `600`
runner seconds plus `60` seconds cancellation margin. Stop without another
automatic smoke retry on source/pilot drift, nonfinite output, invalid
covariance/MMD/decision, non-GPU output, retracing, serialization failure, or
cap exhaustion.

Repair 01 passed in `14.477921013021842` seconds with decision
`PHASE8_CONTROLLED_CALIBRATION_SMOKE_PASSED_NOMINATION_REQUIRED`. Receipt:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/controlled-calibration-smoke-repair-01.json`,
SHA-256
`35201abd756a39a0c16a61477ad58b58c77d11ba00b3ef362484c42828f75c66`.
All six compiled surfaces traced once; GPU/XLA/TF32 provenance, pilot/source
bindings, admissible covariance and MMD intervals, and the two-family smoke
scope passed. Both covariance surfaces selected ridge zero with condition
numbers `34.5642` and `40.6607`. The selected tolerance and validation bounds
are null. Both single-replication decisions were inconclusive; that is
descriptive smoke output, not a hard veto or a power estimate.

Prospective nomination after a smoke pass: 20 replications per family with
root seed `(14001,14002)`. A tolerance is nominated only if every required
family has at least `18/20` simultaneous feature coverage; every equivalence
family has at least `16/20` PASS and at most `1/20` material-difference; and
every material family has at least `16/20` material-difference and at most
`1/20` false PASS. Select the smallest passing tolerance. Skew/dependence do
not veto nomination.

Prospective fresh validation after nomination: exactly 60 replications per
family with root seed `(15001,15002)`, binding the immutable nomination receipt
and selected tolerance. Required one-sided 95% exact-binomial bounds remain:
coverage lower `>=0.85`, true-equivalence PASS lower `>=0.70`, null material-
difference upper `<=0.10`, material false-PASS upper `<=0.10`, and material-
difference lower `>=0.70`. At 60 replications this requires at least `56`
coverage successes, at least `49` required power successes, and at most one
rare false event. Validation is not yet authorized; its exact nomination
binding and command must be added after nomination exists.

### Controlled Calibration Nomination Freeze

Nomination hard-binds the passing repair-01 smoke receipt and verifies its
decision, two-family scope, one-trace counts, GPU/XLA trust manifest, null
selected tolerance, null validation bounds, pilot binding, and historical
smoke-runner hash. The current runner also rejects `--mode validation`
unconditionally until a future patch hard-binds the exact nomination receipt
and selected tolerance.

The maximum run remains 20 independent synthetic replications for all 13
families at the fixed confirmation shape. A prospective futility rule may stop
strictly before replication 20 only if every tolerance candidate is unable to
meet all frozen nomination thresholds even when every remaining outcome is
assigned favorably. A successful nomination always requires all 20
replications. The rule cannot nominate early, relax a threshold, rank viable
tolerances, or use explanatory families as vetoes.

Skeptical audit disposition: `PASS_CONTROLLED_CALIBRATION_NOMINATION_ONLY`.
The exact baseline is the frozen identical-law/equivalent family set; material
mean/variance alternatives are not proxies for G/H and answer calibration
power only. The smoke's interval widths make underpower plausible but one
replicate cannot estimate power, so the repeated controlled stage remains the
smallest valid discriminating artifact. The runner binds the smoke, preserves
the random hierarchy, stops on invalid computation or resource exhaustion,
and records per-replication evidence. No G/H confirmation input is accessible.

Focused nomination checks: `72` controlled/predictive tests passed; Python
compilation and `git diff --check` passed. Review:
`docs/reviews/bayesfilter-ssl-lstm-neutra-phase-8-controlled-calibration-native-review-2026-07-17.md`,
verdict `AGREE_NOMINATION_ONLY`.

Frozen nomination command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-calibration-nomination-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-calibration-nomination-cuda timeout 1860s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_controlled_calibration_2026_07_17.py --mode nomination --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/controlled-calibration-nomination.json --wall-cap-seconds 1800
```

Resource contract: one invocation on physical GPU 1, at most `1800` runner
seconds (`0.5` GPU-hour) plus `60` seconds cancellation margin. Stop without
automatic nomination retry for source/pilot/smoke drift, invalid covariance or
MMD, nonfinite output, hard-veto classification, non-GPU output, retracing,
serialization failure, cap exhaustion, or no mathematically viable tolerance.
A nomination pass opens only receipt audit and prospective hard-bound fresh
validation planning; an underpowered decision triggers design repair, not G/H
confirmation or rejection of predictive validation as a research direction.

## Decision And Handoff

Possible decisions:

- `PHASE8_PREDICTIVE_DESIGN_FROZEN`;
- `PHASE8_DESIGN_UNDERPOWERED_REPAIR_REQUIRED`;
- `PHASE8_INVALID_PREDICTIVE_MACHINERY_BLOCKER`;
- `PHASE8_COMPARATOR_OR_LEAKAGE_BLOCKER`; or
- `PHASE8_RESOURCE_CAP_EXHAUSTED_VALID_INCOMPLETE_EVIDENCE`.

Only `PHASE8_PREDICTIVE_DESIGN_FROZEN` opens Phase 9. The handoff must contain
the immutable configuration/hash, comparator statement, fresh confirmation
seed banks, exact cluster hierarchy, margins/weights/bandwidth/ridge policy,
and Phase 9 commands. It must not contain an already observed G/H confirmatory
forecast result.
