# Deterministic LGSSM HMC Phase 7 Preflight Blocker Result

Date: 2026-07-11

Status: `BLOCKED_BEFORE_PHASE7_SAMPLING`

## Scope And Direct Verdict

Claimed Phase 7 input: the exact executable frozen HMC kernel selected by the
pinned Phase 6AA artifact.

Quantity produced by the authorized refresh: a valid executable private replay
whose target/config/fixture/XLA/geometry/mass/adapter and selected-step
identities match, but whose selected-trajectory, private-loop-kernel, and
public-kernel hashes differ from the Phase 6AA pins.

Verdict: different under the plan's stated exact-identity contract. The Phase
7 smoke and serious sampler must not run.

The inspected committed and refreshed private events agree on every selected
HMC mechanics field available in both records. Current
`handoff_screen_policy` provenance changes the hashed stage lineage and
downstream hashes. This is an engineering hash-contract/baseline-migration
failure. It is not a target failure, implementation-invalidity result for the
sampler, convergence failure, or evidence against the scientific direction.
The old full private replay was not persisted, so complete private-payload
identity remains not checked.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the exact pinned Phase 6AA kernel be replayed into Phase 7? |
| Exact comparator | Phase 6AA public kernel, private-loop kernel, selected-step, and selected-trajectory hashes plus fixed input hashes. |
| Primary criterion | Failed: all exact kernel and trajectory hashes did not match. |
| Continuation veto | Fired: refreshed kernel/input hash mismatch. |
| Explanatory only | Equal inspected mechanics fields, equal verification acceptance, elapsed time, and policy-provenance diff. |
| What is not concluded | No Phase 7 convergence, posterior recovery, sampler ranking, production/default readiness, GPU readiness, or scientific-validity conclusion. |
| Preserving artifact | This note, structured blocker JSON, refreshed public artifact, ignored private replay, public progress/result, and refresh log. |

## Hash Gate

| Identity | Expected | Observed | Status |
| --- | --- | --- | --- |
| Source config | `sha256:1b5683e2f210e3976fca712ec2970f8327831596c0b67776316efbd0b6b46729` | Same | Pass |
| Fixture | `sha256:5b8f4ae78e00b69fb4b75deb1ccd3facfd7869f5d9fc0c7cb87eafdad8c8793e` | Same | Pass |
| XLA gate | `sha256:8941b369f6280ebc3c124220a9bab21f6889228deb92121d63f2fefba3ea6842` | Same | Pass |
| Geometry | `sha256:e2b9531e86f85a662c4da26595e0ab082dd8a1a29d2dbb83b31b076bbf7683ac` | Same | Pass |
| Mass | `sha256:92536fbd13e1ba89c53bfcc874355194b8d2d097ea498d22b5ccd7c318490d8e` | Same | Pass |
| Adapter | `3a71b33479f6eb3681584d3a7a31550a17a5731116253131e9a21a9b5d21af08` | Same | Pass |
| Selected step | `ec7db59e51465eee95658167e1f7596e21d9ab0efdac11f54c2d397aa270ab40` | Same | Pass |
| Selected trajectory | `6eaf7a563353b278a71dcfbe2515fda6d46c47ab2e38996b6b61fab1bbbd13b3` | `3f4b33680ed1e8365670772afe313e479a3a43a4a1c3f2ac2a77c49795aeb04b` | Veto |
| Private-loop kernel | `391558a9b5f4cdc1b9dff9a5e9bceba668dedded7298c1d8c76daea42f42039a` | `2823e20048c0969b79931604462ba142a34aed06fd8cfab3baf03eab89c0168f` | Veto |
| Public final kernel | `8ddf25a3b572893e19e814fad5ca5b6150718e36f760c159b47db1231d92ffff` | `07910941750ad6b882d357411c8ed9a1faa36b886f6125e78af8306ccdae7fbf` | Veto |

`validate_phase7_inputs` failed closed with
`DeterministicLGSSMPhase7Error: public final kernel hash mismatch`.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Worktree | Dirty; in-scope Phase 7 files plus concurrent user-owned LEDH/QR changes preserved and not edited by this lane. |
| Command | `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase7-refresh python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning` |
| Environment | Conda `tf-gpu`; Python `3.11.14`; TensorFlow `2.19.1`; TFP `0.25.0` |
| Device | CPU only by deliberate `CUDA_VISIBLE_DEVICES=-1`; logical device check reported CPU only. |
| XLA/JIT | Host XLA compile observed; refreshed artifact has `xla_confirmed=true`, `jit_compile=true`, and no non-JIT runtime. |
| Data | Deterministic `T=120` fixture, hash shown above; no external data version. |
| Seeds | Fixture root `(20260709, 301)`; kernel-tuning root `(20260709, 501)`; derived seeds retained in the private replay. |
| Start/end | Approximately `2026-07-11T03:22:43+08:00` to `2026-07-11T03:44:13+08:00` from the log and artifact timestamps. |
| Reported stage wall time | `1283.6328543908894` seconds |
| Public kernel artifact | Embedded hash `sha256:3db5c7711d885e6d1e5bbcb05d675976beee88a837ae5c3a7657984d12d00b7c`; file SHA-256 `e95f1197862192ce8436ffe21ec9926519de3810d6bff4d4397a8b9caa590f43` |
| Private replay | Embedded hash `sha256:ce878e2a28c49256ac2d75f3b3d8e2207a5a106e0c9e0175dfcf43020799b867`; file SHA-256 `40c6a8d2cea8c55b5e923dace2bc3500a274b29a4abdd3ceed5b83c773f9ac0c`; 4,210,339 bytes |
| Public tuning result | File SHA-256 `6fb7b1daf8e966f0f02e3fac2fb40e2a7f6d8626a025124e81f7cd92e93d2767` |
| Public tuning progress | File SHA-256 `e4cf7f42f7a2b921159d92b6279bceb0edc72c4fb0b5dc4610e8829fa6d1afe0` |
| Log | `/tmp/bayesfilter-phase7-phase6-refresh.log`; SHA-256 `703e9e198e2be803d4243ae2654a9056ed6b9a7acaac770765e2453d5b80196a` |
| Plan | `docs/plans/bayesfilter-deterministic-lgssm-hmc-phase7-repair-and-execution-plan-2026-07-11.md` |
| Reviews | Substitute plan, implementation, and blocker-result reviews under `docs/reviews`; all `VERDICT: AGREE`. Claude was unavailable due managed external-disclosure rejection. |

## Decision Table

| Decision | Status |
| --- | --- |
| Phase 7 decision | `BLOCK_BEFORE_SMOKE_AND_SERIOUS_SAMPLING` |
| Primary criterion | Failed exact refresh-hash equality. |
| Promotion-veto status | Not evaluated; Phase 7 diagnostics did not run. |
| Continuation-veto status | Fired on selected-trajectory and kernel hash mismatches. |
| Main uncertainty | Full old private replay is unavailable, so complete private-payload semantic identity cannot be proved from the inspected fields alone. |
| Next justified action | Review a versioned semantic-mechanics identity and baseline-migration repair; do not silently repin. |
| What is not concluded | No convergence, recovery, ranking, production/default, GPU, DSGE, NeuTra, or broad scientific conclusion. |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard-veto screen | Engineering continuation veto supported by exact hash inequality. Refreshed Phase 6 itself had no HMC hard veto. |
| Viable candidate | The refreshed Phase 6 kernel passed its own screen, but is not admitted to Phase 7 under the pinned identity contract. |
| Statistically supported ranking | Not applicable; no candidate ranking was attempted. |
| Descriptive-only differences | Verification acceptance and runtime are explanatory only; equal mechanics fields do not supersede the hash veto. |
| Default-readiness | Not evaluated and not supported. |
| Next evidence needed | A reviewed semantic identity derivation plus replay reconstruction and artifact-integrity checks. |

## Three Ledgers

| Ledger | Result |
| --- | --- |
| Engineering correctness | Phase 7 implementation tests passed; replay refresh and serialization succeeded; preflight correctly failed closed on stale pins. |
| Numerical/sampler validity | Refreshed Phase 6 passed its existing fixed-kernel verification, but Phase 7 R-hat and ESS were never computed. |
| Scientific interpretation | No scientific conclusion. The blocker concerns artifact identity and provenance. |

## Closeout Verification

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase7-closeout \
python -m pytest -q \
  tests/test_deterministic_lgssm_hmc_phase7_tf.py \
  tests/test_hmc_convergence.py \
  tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `34 passed, 2 warnings`. Both warnings are existing TFP
`distutils.version` deprecations.

The structured blocker JSON parsed successfully, its embedded stable hash
recomputed exactly, `git diff --check` passed on the Phase 7 lane, and no stale
pre-execution status token remained in the active plan, runbook, handoff,
result, or blocker artifact.

## Negative-Result Classification

- Implementation failure: no failure in the Phase 6 tuning run; the Phase 7
  preflight correctly enforced its contract.
- Tuning failure: not supported; the refreshed Phase 6 artifact passed.
- Diagnostic failure: Phase 7 diagnostics were not run.
- Evidence against the idea: none. The result only weakens the claim that the
  committed Phase 6AA hashes remain a replayable identity under current code.
- Rescue condition: a reviewed, versioned identity migration proves the same
  executable mechanics or restores the original hash schema.

## Post-Run Red Team

Strongest alternative explanation: the equal inspected mechanics fields may
hide a private payload difference not represented in the old event artifact.
That possibility is why the run remains blocked despite the apparent
provenance-only diff.

What would overturn the blocker: a reviewed derivation showing that all fields
affecting replayed HMC transitions are identical and that the differing hashes
encode only non-mechanical policy/provenance, followed by successful replay
reconstruction and tamper checks under a versioned identity contract.

Weakest evidence: there is no old full private replay for direct comparison.

Next smallest justified action: create and review the narrow baseline-migration
repair described in the stop handoff. Phase 7 smoke, serious sampling, Phase 8,
and NeuTra remain unexecuted.
