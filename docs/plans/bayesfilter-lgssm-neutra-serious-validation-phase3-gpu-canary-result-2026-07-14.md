# BayesFilter LGSSM NeuTra Phase 3 GPU Canary Result

Date: 2026-07-14  
Status: `PASS_PHASE3_TRUSTED_EXACT_TARGET_GPU_XLA_CANARY`

## Result

The exact fixture-bound 18D LGSSM reverse-KL training path executed on the
trusted RTX 4080 SUPER with TensorFlow 2.19.1, float64, XLA JIT, TF32 enabled,
GPU memory growth enabled, and soft placement disabled. The first invocation
stopped after one step; the second resumed the same immutable configuration to
eight steps and froze the transport.

Artifact:
`docs/benchmarks/artifacts/lgssm_neutra_serious_validation_2026_07_13/phase3/result.json`

Artifact hash:
`sha256:3aefb59c202f0859b5ef30a862f1f4848b6fec0e25bf627ccd4730e673e68a84`

Score-parity addendum:
`docs/benchmarks/artifacts/lgssm_neutra_serious_validation_2026_07_13/phase3/score_parity_addendum.json`

Addendum artifact hash:
`sha256:bcd386c2fb5584709f3d564ba975314efed1c7af06bbf45d1896765fda0c3dc3`

## Checks

| Check | Result |
| --- | --- |
| Target signature | `f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30` |
| Partial/resume steps | `1 -> 8`; resume flag true |
| Training-state hash | `a567fa39fafe39b4191c1a3981eef306e6c7feab7e40e57e28b42c279139cdeb` |
| Trainable/frozen forward parity | max absolute error `0.0` |
| Trainable/frozen logdet parity | max absolute error `0.0` |
| Training target status | valid at every recorded step; zero floors |
| Held-out target status | valid on 64 independent base draws |
| Compiled fixed value/score | finite; all outputs on `/GPU:0` |
| Exact-artifact frozen score parity | restored trainable/reference versus frozen explicit score max absolute error `1.4210854715202004e-14` across 20 GPU/XLA probes; gate `1e-8` |
| Exact-artifact value/forward/logdet parity | max absolute error `0.0` for each quantity on the same probes |
| XLA repeat | second compiled call exactly reproduced the first |
| GPU fallback | none; all trainable, Adam, objective, and probe outputs on GPU |

The JSON result contains the git commit, dirty-worktree disclosure, Python and
TensorFlow environment, device policy, seeds, commands, elapsed time, and
artifact hashes under `run_manifest`; the addendum binds the unchanged
checkpoint and frozen-payload file hashes.

The observed reverse-KL losses and force norms are explanatory only. They do
not nominate a transport or establish HMC readiness, convergence, posterior
correctness, or superiority.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Proceed to Phase 4 candidate freezing | Exact-target GPU/XLA training, checkpoint/resume, status telemetry, freeze/reload, and compiled fixed-score checks passed | No Phase 3 continuation veto fired | Eight steps are only an engineering canary | Materialize the truth-centered affine control and both predeclared 1,000-step dense seeds | Transport quality, tuned HMC convergence, posterior validity, robustness, production/default readiness |

Before Phase 3, the static campaign gate required exact array equality between
the affine center in
`docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/mass.json`
(file SHA-256
`54549c9156821536bc4780f0406a7716b0d3fa39a5b5900fa2893cbef2968a95`)
and `raw_truth` in the fixture bound by target signature
`f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30`.
That check passed. Therefore the affine initialization is favorably
truth-centered for this fixture. This note does not characterize the full
affine factor as oracle or near-oracle geometry. Any later pass is conditional
on the truth-centered initialization and cannot establish calibration,
robustness, or generalization.
