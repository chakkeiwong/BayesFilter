# Phase 21 Result: HMC Readiness Decision Gate

Date: 2026-07-08

## Scope

Phase 21 classified the Phase 17-20 evidence for the LGSSM-first fixed
transport NeuTra-HMC path. It did not run new training, HMC sampling, HMC
tuning, GPU jobs, DSGE/c603 work, nonlinear SSM expansion, or default-policy
changes.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | `LGSSM_REFERENCE_HMC_READY` |
| Decision scope | Static QR LGSSM fixture and exact Phase 17-20 artifacts only |
| Primary criterion status | Passed |
| Veto diagnostic status | No hard veto fired |
| Main uncertainty | Short chains, no R-hat/ESS, covariance residual passed by a tight margin |
| Next justified action | New reviewed program for longer LGSSM replication or first non-LGSSM target |
| What is not concluded | Production readiness, default readiness, sampler superiority, nonlinear SSM validity, DSGE/c603 validity, broad NeuTra validity, or scientific validity |

## Inference-Status Table

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed |
| Statistically supported ranking | None; no method ranking was tested or supported |
| Descriptive-only differences | Acceptance rate `1.0` and posterior residual magnitudes are descriptive |
| Default-readiness | Not supported |
| Next evidence needed | Longer LGSSM replication or first non-LGSSM target under a new reviewed program |

## Artifacts Inspected

| Artifact | Decision/hash |
| --- | --- |
| Phase 17 payload | file SHA-256 `18992dface97aa8142d714b2fe99b89ee4717c0e5f01a06c7f6e4a868b220aa1`; stable hash `sha256:27f2c4364db13d1be14d7ad48b3257bd3f8418c091ad4d075db8504917bdb1c3` |
| Phase 18 mechanics | `PASS_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE`; file SHA-256 `6d1cb178f1e7b91170450acf8843cb7e7062d9fe6c125d21217489fdf4ade2a5`; stable hash `sha256:5059f900fe92df20ef73b326aa5fdaac7f0264a475949c83b0405d994d9b1269` |
| Phase 19 CPU harness | `PASS_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS`; file SHA-256 `38c0400b04ce7438f3bc70236ccfd42916c11c1b9c05d95bd76260b64f8c10b4`; stable hash `sha256:aaa7893d6a6313db74350eb1efac57f4188165170d42056df1b079eef2c313f6` |
| Phase 20 validation | `PASS_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION`; file SHA-256 `094962f3fd8dbd5002ef5d92e42e23ae34cc52dc234ad836d78fe4edd768e188`; stable hash `sha256:07404fd92a4b5e69449088c8852392241fdbbc9cb61eea91ceb4b2b235f6d553` |
| Phase 21 decision JSON | `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase21_readiness_decision_seed20260707.json`; stable hash `sha256:3be31c6ae57721d47de7b9b55a46115aafdef53d2e17ec100bbe2af97a1da82e` |

## Veto Screen

| Diagnostic | Status |
| --- | --- |
| `jit_compile=true` evidence | Passed |
| `jit_compile=false` runtime | Not run |
| Training after Phase 16 | Not run |
| GPU sample generation | Not run |
| CPU-hidden chain generation | Passed |
| Phase 20 reference posterior | Present |
| Phase 20 posterior residuals | Passed |
| Worker errors | Absent |
| Unsupported product/default/scientific claims | Absent |

## Phase 20 Evidence Used

Phase 20 retained 256 total samples across 2 workers and 4 chains. It compared
sample summaries against a deterministic 2D quadrature reference posterior over
the exact LGSSM likelihood target.

Key diagnostics:

- mean max absolute residual: `0.23550553833312193` versus tolerance `0.35`;
- covariance max absolute residual: `0.6497737415124094` versus tolerance
  `0.65`;
- worker acceptance rates: `[1.0, 1.0]`;
- R-hat/ESS: unavailable in the bounded helper.

The covariance residual passed by a very narrow margin. That supports the
predeclared Phase 20 fixture gate, but it is not convergence, optimal tuning,
or default-readiness evidence.

## Local Checks

- Phase 21 consistency parser over Phase 17-20 artifacts: passed with
  `failed=[]`.
- `git diff --check` on Phase 21 docs and decision artifact: pending final
  closeout command.

## Handoff

The 17-21 LGSSM-first proper-HMC gap-closure runbook is complete at the narrow
fixture-local level.

Recommended next program:

- either longer LGSSM replication with R-hat/ESS and uncertainty reporting; or
- first non-LGSSM target under the same GPU-training / CPU-multicore
  sample-generation / `jit_compile=True` policy.

Both require a new reviewed plan before runtime execution.
