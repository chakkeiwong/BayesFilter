# SSL-LSTM NeuTra Phase 5 Exact Transformed-Target Preflight Plan

Date: 2026-07-16

Status: `PHASE5_PASSED_R2_PHASE6_PLAN_READY_EXECUTION_NOT_AUTHORIZED`

## Objective And Entry Conditions

Prove that the independently confirmed, immutable trial-0 G/H dense-IAF
payloads define the intended locked SSL-LSTM posterior in `z` coordinates after
reload. This is an exact change-of-variables engineering preflight before any
HMC tuning or retained sampling.

Entry conditions now pass:

- Phase 4 trial-0 G/H confirmation decision is
  `FRESH_CONFIRMATION_PASSED`;
- G best payload SHA-256 is
  `6e147d5b33d003e0c895f294fc6b33523dcf97dc24af794d26a677886dedc354`;
- H best payload SHA-256 is
  `ed0e42602aa39788ca1ea8d3c881d8bf85e15b91a687ef9adbe00a7b2c9120fb`;
- both payloads bind locked target semantic SHA-256
  `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e`;
  and
- the current authorization ended with H training confirmation. It does not
  authorize an HMC mechanics transition, tuning, or retained sampling.

The historical affine-control plan is superseded as the main Phase 5 handoff.
Affine fixtures remain implementation controls only.

## Evidence Contract

| Field | Prospective contract |
| --- | --- |
| Question | Do the reloaded G/H dense-IAF transports implement the exact declared change of variables for the locked target, including Jacobian value and score terms, with no mutable trainer state reachable? |
| Exact comparator | Direct locked target at `theta=T(z)` versus `FixedTransportValueScoreAdapter` in `z`, plus pre-freeze/reloaded transport parity where preserved artifacts permit it |
| Primary pass | Both G and H pass payload identity, direction, forward/inverse, logdet, exact transformed-value identity, full transformed-score finite differences, scalar/batch/permutation parity, and original-start roundtrip gates |
| Promotion veto | Any payload/target/signature drift, mutable state reachable from the frozen binding, wrong logdet sign, missing logdet score, score finite-difference failure, start-map failure, nonfinite value/score, or scalar/batch/permutation mismatch |
| Continuation veto | Shared target or adapter mathematical failure, corrupt G/H artifact, unavailable required implementation surface, or unauthorized GPU/HMC action |
| Explanatory only | Mapped start radii, graph size, continuous residuals below thresholds, runtime, and descriptive G/H differences |
| Nonclaims | No sampler admission, HMC readiness, posterior correctness, complete support or mode coverage, predictive validity, superiority, or default readiness |
| Result artifact | `docs/plans/bayesfilter-ssl-lstm-neutra-phase-5-exact-transformed-target-preflight-result-2026-07-16.md` plus structured receipts under `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-5-trial0-gh/` |

## Skeptical Audit And Pre-Mortem

Audit status: `PASSED_FOR_EXACT_VALUE_SCORE_PREFLIGHT_NO_HMC`.

- Wrong baseline: repaired. The comparator is the exact locked target, not
  ordinary HMC or a fictitious posterior oracle.
- Proxy promotion: prohibited. Training loss and support screens admitted the
  payloads to preflight but cannot establish transformed-target correctness.
- Unfair selection: prohibited. Both independently confirmed G and H payloads
  must pass; do not use only the more convenient candidate.
- Missing stop: any shared math/target failure stops Phase 5; a candidate-local
  artifact failure rejects that binding and triggers focused diagnosis.
- Misleading pass: value identity could pass with an omitted logdet score; full
  directional score finite differences are required.
- Misleading locality: `z=0` alone is insufficient; fixed shells, tails, and
  inverse-mapped original starts are required.
- Serialization drift: canonical payload hashes and reload parity are binding.
- Environment mismatch: CPU-hidden checks are reference/debug only. Any later
  GPU/XLA receipt requires a separately authorized exact command and budget.
- API mismatch: repaired prospectively. The locked target exposes scalar
  `value_and_score` and batch-native `batch_value_and_score`; the preflight
  uses a read-only bridge that dispatches by rank and preserves the locked
  target signature/capability rather than weakening either API.
- Stale mechanics requirement: the master program's one-transition HMC canary
  is deferred to Phase 6 because the current authorization excludes HMC. The
  Phase 5 trusted receipt compiles and evaluates only exact transformed
  values/scores under GPU/XLA.

## Required Work And Checks

1. Load G and H with `load_frozen_neutra_artifact`, requiring the exact target
   signature and canonical payload hash.
2. Verify the loaded objects expose no optimizer, trainer, or trainable
   `tf.Variable` surface.
3. Bind each transport through `FixedTransportValueScoreAdapter` to the locked
   target using the declared direction:

```text
theta = forward_z_to_theta(z)
log_pi_z(z) = log_pi_theta(theta) + log_abs_det_dtheta_dz(z)
score_z(z) = J_forward(z)^T score_theta(theta)
             + grad_z log_abs_det_dtheta_dz(z)
```

4. At fixed reference, coordinate-shell, far-tail, and inverse-mapped original
   start points, check transformed value identity in `float64`.
5. Check every transformed-score component against central directional finite
   differences. Require absolute error `<=1e-6` or relative error `<=1e-5` for
   every finite component; record both errors without favorable selection.
6. Inverse-map all four historical starts and require `theta -> z -> theta`
   maximum absolute residual `<=1e-9`. Do not replace starts after seeing their
   radii.
7. Require scalar/batch parity and chain-row permutation parity within `1e-10`.
8. Retain non-diagonal affine and independent dense-IAF fixtures as
   implementation controls. They validate generic math, not the G/H target.
9. Add focused tests for wrong direction, wrong logdet sign, omitted logdet
   score, corrupted tensor hash, target mismatch, and mutable-state reachability.
10. Run one focused transformed-density/score review before declaring Phase 5
    pass.

The prospective exactness probe bank is fixed before execution. For each G/H
transport it contains 21 `theta` points: the locked prior center, the eight A0
historical-geometry coordinate points at latent radius 2, the eight equivalent
points at radius 4, and the unchanged four original dispersed A4 starts mapped
through the immutable A0 geometry. Each point is inverse-mapped to `z`; all four
coordinate directions are checked by central finite differences with step
`1e-5`. The broader Phase 4 neighborhood and seeded-prior probe bank remains
support evidence and is not silently promoted into this exactness criterion.

## Commands And Resource Boundary

Implementation and reference tests use CPU-hidden TensorFlow:

```text
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_ssl_lstm_neutra_phase5_exact_preflight.py tests/test_dense_iaf_neutra_artifact_loader.py tests/test_batched_value_score.py
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase5_exact_preflight_2026_07_16.py --mode cpu-reference --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-5-trial0-gh/cpu-reference-r2.json
```

The user authorized the suggested Phase 5 work on 2026-07-16. The bounded
trusted GPU/XLA value-score command is:

```text
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase5_exact_preflight_2026_07_16.py --mode gpu-xla --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-5-trial0-gh/gpu-xla-r2.json --wall-cap-seconds 1800
```

This command used physical GPU 1 only, TensorFlow/TFP `float64`, XLA JIT on,
TF32 enabled, and no randomness. The 1,800-second cap was a sequential launch
stop checked between G and H, not asynchronous termination inside one target
call. It evaluated
value/score programs only: no HMC kernel, transition, tuning, or retained sample
surface is imported or called. The expected output directory is fresh before
execution and receipts record source/payload hashes, environment, device,
compile metadata, wall time, and all gate residuals.

## Forbidden Actions And Handoff

- Do not train or modify either transport.
- Do not tune step size, mass, or trajectory length.
- Do not run an HMC transition under Phase 5 authority.
- Do not retain or diagnose chains as posterior samples.
- Do not replace historical starts with standard-normal starts.
- Do not label unavailable divergence telemetry as zero divergences.
- Do not rank G and H from preflight residuals.

If both G and H pass exact transformed-target preflight, draft a separate Phase
6 transformed-HMC tuning plan with its own resource authorization. A Phase 5
pass only establishes exact frozen binding readiness for that later test.

## Closeout

Both G and H passed CPU-hidden reference and trusted GPU/XLA exact preflight.
The authoritative corrected trusted receipt completed in `417.3270` seconds;
no HMC was launched. The
decision and run manifest are recorded in
`docs/plans/bayesfilter-ssl-lstm-neutra-phase-5-exact-transformed-target-preflight-result-2026-07-16.md`.
Focused review returned `AGREE_R2_NO_BLOCKING_FINDING` in
`docs/reviews/bayesfilter-ssl-lstm-neutra-phase-5-exact-preflight-native-review-2026-07-16.md`.

The Phase 6 design draft is
`docs/plans/bayesfilter-ssl-lstm-neutra-phase-6-transformed-hmc-tuning-plan-2026-07-16.md`.
Its execution remains unauthorized.

Final contract audit found that the first executable score-parity gate reused
the `1e-6` finite-difference absolute tolerance rather than the plan's `1e-10`
scalar/batch/permutation threshold. Observed residuals passed `1e-10`, but the
first `cpu-reference.json` and `gpu-xla.json` receipts are superseded because
their executable gate was too loose. The corrected source uses
`PARITY_ATOL=1e-10`; the fresh `*-r2.json` receipts pass and are the only
authoritative Phase 5 receipts.
