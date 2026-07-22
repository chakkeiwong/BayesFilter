# Phase 4 Attempt 02 Prelaunch Audit

Status: `PASS_FOR_LAUNCH`

## Research Intent

Question: does a second sequential sampling seed, using the exact same
preserved LGSSM transport and admitted public-tuner kernel as Attempt 01, pass
sampler validity and the owner's truth-tail diagnostic?

Baseline: Phase 4 Attempt 01 final-kernel hash
`e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc`,
public tuner seed `(20260621, 8)`, fixed identity mass in NeuTra coordinates,
target acceptance `0.70`, and acceptance band `[0.65, 0.75]`.

Promotion criterion: exact kernel-hash equality before sampling; valid adaptive
warm-up; retained rank-normalized/folded split R-hat and ESS admission; no
health veto; and interpretation of the truth-tail result under the owner's
two-seed rule.

Continuation veto: transport/target drift, tuning replay hash mismatch,
nonfinite target/gradient, mass mutation, missing required diagnostics,
GPU/XLA or memory-growth failure, or exhausted Phase 4 attempt budget.

Explanatory only: runtime, step size, leapfrog count, acceptance trajectory,
and posterior descriptive summaries beyond the declared gates.

Nonclaims: no universal NeuTra validity, distributional equivalence, sampler
superiority, or default-readiness conclusion.

## Skeptical Audit

| Risk | Audit result |
| --- | --- |
| Wrong baseline | Passed: Attempt 02 is bound to Attempt 01's exact final-kernel hash, not merely the same tuning policy. |
| Proxy promoted to criterion | Passed: engineering tests and acceptance telemetry do not replace sequential convergence, health, ESS, or truth-tail checks. |
| Hidden changed default | Passed: only sequential seeds change through `seed_offset=1000`; tuner seed, target, transport, mass policy, thresholds, GPU/XLA, and hardware class remain fixed. |
| Unfair replication | Passed conditionally: exact hash equality is checked after tuning and before sampling. A mismatch returns `TUNING_REPLAY_HASH_MISMATCH` and is not sampling evidence. |
| Missing stop condition | Passed: the subplan names target, artifact, health, replay, resource, and budget vetoes. |
| Artifact cannot answer question | Passed: the fresh root records transport identity, tuning result, replay-hash gate, sequential chunks, diagnostics, truth-tail table, and run manifest. |
| Environment mismatch | Passed: trusted `nvidia-smi` saw the RTX 4080 SUPER; the BayesFilter helper reported memory growth enabled; a tiny TensorFlow 2.19.1 GPU operation compiled with XLA. |

## Local And Independent Review

- frozen-validation focused suite: `22 passed`;
- public tuner/fixed-mass/replay/NeuTra/public-API regression: `134 passed, 1 skipped`;
- `py_compile` and `git diff --check`: passed;
- bounded Claude review of `phase4-attempt01-result.md`: `VERDICT: AGREE`.

## Exact Command

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py \
  --action validate-frozen \
  --cell LGSSM-EXACT \
  --output-root docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt02 \
  --frozen-transport docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-attempt-02/LGSSM-EXACT/final/segments/steps-004001-005000/frozen_transport.json \
  --frozen-transport-sha256 b0b89656b2503146556f50b4e5e3e0e6b9b63daf0673380043ccb046dd14877e \
  --expected-tuning-final-kernel-hash e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc \
  --seed-offset 1000
```

The output root is fresh. The Phase 4 second-attempt budget remains available.
