# BayesFilter SSL-LSTM Predictive Validation Live Plan

Date: 2026-07-13

Status: `PASSED_FOR_A4_DESIGN_ONLY`

## Question And Scope

Can the current TensorFlow/TFP scalar-LGSSM oracle and predictive-statistics
machinery correctly and reproducibly compute joint 1-to-10-step forecast-law
diagnostics on CPU and trusted GPU/XLA?

This plan validates engineering and statistical machinery. It does not yet
compare an estimated SSL-LSTM model with a reference model and does not decide
predictive equivalence.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Exact baseline | Analytic scalar LGSSM forecast mean/covariance plus direct equation simulation from materialized innovations |
| Candidate mechanism | TensorFlow `float64` oracle, forecast summaries, standardized paths, descriptive quadratic MMD, cross-chain linear MMD, hierarchical intervals, and fail-closed decision logic |
| Primary pass criterion | Focused tests pass; CPU artifact independently agrees with analytic/direct baselines under predeclared numerical and uncertainty checks; trusted GPU/XLA artifact has correct placement and agrees with persisted CPU tensors within the predeclared scale-aware tolerance |
| Promotion vetoes | Formula or direct-replay mismatch; nonfinite values; invalid covariance; wrong tensor hierarchy; invalid chain/block/cluster resampling; non-admissible evidence emitting `PASS`; missing XLA/GPU placement; CPU/GPU parity failure; incomplete artifact |
| Continuation vetoes | Broken analytic assumptions, corrupted inputs/artifacts, unavailable trusted GPU route after a trusted check, or a repair that changes the estimand without review |
| Repair triggers | Localized implementation, fixture, serialization, shape, XLA compatibility, or resource defect |
| Explanatory only | Runtime, HLO size, trace count, sub-threshold residuals, descriptive quadratic MMD, high moments, quantiles, and one-fixture controlled-alternative power |
| Preservation artifact | CPU and GPU JSON artifacts plus `docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-result-2026-07-11.md` |

## Final Evidence

- A2 terminal-state forecast machinery previously passed its CPU and trusted
  GPU/XLA engineering checks.
- A3 production code lives in
  `bayesfilter/testing/scalar_lgssm_forecast_oracle.py` and
  `bayesfilter/inference/predictive_equivalence.py`.
- The focused CPU-hidden A3 suite passed `65/65` on 2026-07-13.
- The CPU-hidden artifact passed all 21 checks and independent numerical replay:
  SHA-256 `f8252b9a0f6bba1bc5350b0516ceaddca04006bfe489acc74ac7f13d7846d82b`.
- The trusted GPU/XLA artifact passed all 21 checks, independent replay, and
  persisted-input CPU/GPU parity: SHA-256
  `5c31b26fbf20a10b754ad3e99bb8dc1481b12c74669c3b60e8e7cae8e080b693`.
- The actual identical-law fixture was `INCONCLUSIVE_UNDERPOWERED`; variance,
  skew, and dependence alternatives were also underpowered. These are A4
  calibration repair signals, not A3 engineering or continuation vetoes.
- The full result is recorded in
  `docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-result-2026-07-11.md`.
- The design-only A4 handoff is
  `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-subplan-2026-07-11.md`.

Historical hashes and traces may identify old evidence, but they are not
runtime authorization gates. Normal Git status inspection is sufficient to
preserve unrelated concurrent work.

## Execution Plan

1. Replace the legacy hash/trace-gated benchmark entry point with a compact
   runner that preserves the existing numerical calculations and emits a
   self-contained run manifest. Do not change production mathematical logic.
2. Run in-memory compilation and the two focused test files with the GPU hidden.
3. Generate a CPU-hidden reference artifact and independently replay its
   analytic formulas, materialized innovations, statistical decisions, and
   artifact schema.
4. Generate a trusted managed-session GPU/XLA artifact from persisted CPU
   inputs and verify device placement, XLA compilation, finite outputs, and
   scale-aware CPU/GPU parity.
5. Write one A3 result note with exact commands, environment, seeds, wall time,
   hard-veto status, uncertainties, decision table, and post-run red team.
6. If A3 passes, draft the A4 calibration design in this same live-plan style.
   Do not execute calibration until its evidence contract is written.

## Commands And Environment

Focused CPU-hidden regression:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a3-pycache \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest \
  -p no:cacheprovider -q \
  tests/test_scalar_lgssm_forecast_oracle.py \
  tests/test_predictive_equivalence.py
```

The compact runner will use this interface:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a3-pycache \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_predictive_validation_a3_2026_07_13.py \
  --mode cpu-reference \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-cpu-reference.json

PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a3-pycache \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_predictive_validation_a3_2026_07_13.py \
  --mode gpu-xla \
  --cpu-reference docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-cpu-reference.json \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-gpu-xla-canary.json
```

The runner must record the Git commit, dirty state, command, conda environment,
TensorFlow/TFP versions, CPU/GPU visibility, device rows, TF32/XLA settings,
seeds, wall time, inputs, and output paths. CPU mode must explicitly record that
GPU devices were hidden. GPU mode must record trust basis
`owner_designated_managed_session_visible_gpu_trusted`.

Resource stop: stop an individual evidence command after 15 minutes, on
nonfinite output, device-placement failure, repeated XLA compilation failure,
or obvious memory pressure. Diagnose with the smallest focused check before
retrying. Do not launch a sweep or overnight run under this plan.

## Skeptical Pre-Execution Audit

| Challenge | Disposition |
| --- | --- |
| Wrong baseline | Cleared: the baseline is the derived analytic LGSSM and direct equation simulation, not another sampler or the implementation under test alone. |
| Proxy promoted | Cleared: tests, replay residuals, runtime, HLO, high moments, quantiles, and descriptive MMD cannot establish predictive equivalence or scientific validity. |
| Missing stop conditions | Cleared: formula, finiteness, hierarchy, decision, device/XLA, parity, artifact, time, and memory vetoes are explicit. |
| Unfair comparison | Cleared: GPU consumes persisted CPU inputs; it does not regenerate floating random draws from seed metadata. |
| Hidden assumptions | Cleared for A3: scalar LGSSM equations, horizon 10, `float64`, materialized innovations, chain/draw/forecast/horizon hierarchy, and inferential roles are fixed by code and tests. |
| Stale context | Cleared: current source and focused tests are checked directly; mutable document hashes and an exact historical `HEAD` are not treated as scientific evidence. |
| Environment mismatch | Cleared prospectively: CPU deliberately hides GPU; serious parity evidence uses the trusted managed-session GPU/XLA route and records placement. |
| Commands answer the question | Cleared: focused tests, analytic/direct CPU replay, and persisted-input GPU parity directly test the A3 engineering question. |

Audit decision: `PASS_TO_IMPLEMENT_COMPACT_RUNNER_AND_RUN_FOCUSED_CHECKS`.

## Execution Note: First CPU Artifact Attempt

The focused CPU-hidden regression passed `65/65` in 13.92 seconds. The first
full CPU artifact attempt then stopped after approximately three minutes at the
legacy aggregate `controlled_alternatives` check.

Classification: localized evidence-role defect in the reused benchmark core,
not a failure of the analytic oracle, direct simulation, statistics
implementation, target, data, or research direction. The aggregate described
valid underpowered alternatives as repair triggers but simultaneously required
the provisional mean and variance alternatives to emit `MATERIAL_DIFFERENCE`.
That silently promoted one-fixture detection power from explanatory evidence to
a hard veto, contrary to this plan's evidence contract.

Repair: the Tier 2 adapter retains exact alternative construction identities,
positive variance/skew/dependence mechanics, finite/admissible intervals, and
absence of `INVALID_HARD_VETO` as required checks. Whether each provisional
alternative emits `MATERIAL_DIFFERENCE` remains recorded as explanatory power.
No production numerical function or fixture constant changed. A separate
provenance defect was also localized: this environment exposes TFP's module
version without `tensorflow-probability` distribution metadata, so manifests
now read `tensorflow_probability.__version__`.

Continuation decision: rerun one fresh CPU artifact. Stop if alternative
mechanics or inferential validity fails; continue if only provisional detection
power is limited.

The repaired CPU generation then passed, but its first independent replay
stopped before statistical recomputation because the historical verifier
expected only three resampling rows while generation correctly persisted four:
chain indices, draw indices, forecast-replication indices, and the arm seed.
The verifier also substituted `[0, 0]` for that persisted seed. Classification:
independent-verifier schema defect, not invalid generated indices or a failure
of the resampling algorithm. Repair: require the exact four-row materialized
schema, validate the seed shape, reconstruct metadata with the persisted seed,
and retain fresh seed-based replay as a diagnostic cross-check. Because runtime
artifacts bind the verifier source, regenerate the CPU artifact after this
repair rather than accepting a stale source binding.

The next independent replay passed the index schema and deeper numerical
recomputation, then stopped because the reused seed-attestation helper compared
the artifact role to its historical label. Classification: adapter
compatibility defect, not a seed, tensor, device, or numerical mismatch.
Repair: validate the Tier 2 artifact role and CPU-hidden manifest directly,
then provide the historical label only in a shallow compatibility view passed
to the unchanged seed-attestation helper. The persisted artifact and its Tier 2
role are not modified. Regenerate once more because the verifier source binding
changed.

That replay reached the controlled-alternative formulas and found the persisted
dependence path differed from a fresh reconstruction by at most
`4.440892098500626e-16` at scale `1.9920662530876396`; the predeclared
`8192 * eps * scale` tolerance is `3.6235474055277148e-12`. Classification:
floating evaluation-order difference, not a formula mismatch. Repair: the Tier
2 verifier now independently reconstructs all four alternatives, applies the
same scale-aware tolerance to continuous paths/features/intervals, and keeps
names, statuses, decisions, validity flags, repair triggers, shapes, dtypes, and
integer schedules exact. It independently enforces all mechanics and validity
vetoes while leaving provisional detection power explanatory.

The configuration binding was also narrowed to the numeric fixture projection
and live-plan path. It no longer hashes the mutable live plan or governance-only
fixture fields, so recording execution notes cannot invalidate valid numerical
artifacts.

The next replay cleared all numerical and statistical sections, then found raw
simulation HLO text differed only in TensorFlow's process-local function ID
(`simulation_kernel_5393` versus `simulation_kernel_278`). After replacing only
those process-local IDs, the complete HLO text matched exactly, including all
284 lines and entry layout. Repair: retain raw HLO/hash/byte-count
self-consistency in the artifact, require one concrete trace and the exact
device class, and compare fresh versus persisted HLO after narrowly normalizing
only TensorFlow process-local function IDs.

Generation artifacts now bind only production code, focused tests, the
numerical generation core, and the Tier 2 runner. Verification receipts bind
the independent replay core and verifier separately. A verifier-only repair
therefore requires a fresh independent receipt, not regeneration of unchanged
numerical tensors.

The first trusted GPU/XLA attempt initialized both RTX 4080 SUPER devices,
loaded cuDNN, and compiled on the CUDA XLA backend, then stopped before
numerical evaluation because the historical GPU input loader repeated the old
three-row resampling schema. Classification: GPU adapter schema defect, not a
GPU, XLA, CPU-reference, resampling, or numerical failure. Repair: require and
consume the exact four persisted rows, including each arm's materialized seed.
The verified CPU artifact and receipt remain the GPU input authority; no random
index is regenerated from seed metadata and no CPU rerun is required.

## Interpretation And Nonclaims

A passing A3 result establishes only that the oracle/statistics machinery works
for the reviewed scalar-LGSSM fixtures on the tested CPU and GPU/XLA paths. It
does not establish:

- SSL-LSTM predictive equivalence or calibration;
- posterior or parameter correctness;
- HMC or NeuTra validity, readiness, or superiority;
- model adequacy, identification, or scientific validity;
- production, public API, package, release, or default readiness; or
- a statistically supported ranking of methods.

No ranking is expected in A3. Continuous stochastic differences remain
descriptive unless a predeclared uncertainty analysis supports a comparison.
