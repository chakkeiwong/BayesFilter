# LGSSM NeuTra Gap Closure Phase 3 Result - Frozen Objective

Date: 2026-07-15  
Decision: `PASS_PHASE3_FROZEN_OBJECTIVE_VALIDATION`

Both fresh 5,000-step candidates passed deterministic GPU/XLA and CPU-hidden
XLA probes on the exact transformed LGSSM objective. Identities, target status,
finite values, requested devices, and second-call determinism all passed.

| Candidate | max theta diff | max logdet diff | max value diff | max score diff |
| --- | ---: | ---: | ---: | ---: |
| `dense_seed1201` | `4.44e-16` | `3.55e-15` | `1.07e-13` | `5.11e-15` |
| `dense_seed1202` | `2.22e-16` | `3.55e-15` | `1.07e-13` | `7.22e-15` |

All are below the respective `1e-12`, `1e-12`, `1e-8`, and `1e-8`
tolerances. The first GPU attempt exposed a harness ordering defect: it loaded
TensorFlow transport tensors before configuring memory growth. No artifact or
scientific computation was produced. Memory configuration was moved ahead of
candidate loading, a focused order regression passed (`14 passed` with the
memory-policy suite), and both unchanged probes then passed.

Primary result:
`docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/phase3/result.json`
with artifact hash
`sha256:6f6896ff257362231627beb87053273598e8ef27c97386eb55113a298136636e`.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| advance both candidates | exact frozen objective identity and cross-device parity pass | no identity, finite, status, device, XLA, or parity veto | HMC geometry and chain mixing remain untested | tune each candidate independently | no convergence, posterior, superiority, or default claim |

The HMC seed ledger is frozen in the result artifact. No serious sampling was
executed in this phase.
