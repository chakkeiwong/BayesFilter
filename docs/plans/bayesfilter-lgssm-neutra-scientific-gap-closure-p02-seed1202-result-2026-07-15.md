# LGSSM NeuTra Gap Closure Phase 2 Result - Seed 1202 And Training Close

Date: 2026-07-15  
Decision: `PASS_PHASE2_TWO_FRESH_FROZEN_CANDIDATES`

## Outcome

`dense_seed1202=(20260713,1202)` completed and passed the strict graph-native
training harness normally after the runtime-package import repair. Both
predeclared 5,000-step seeds are now immutable candidates for frozen objective
validation; neither was selected or rejected by training loss.

## Two-Seed Evidence

| Field | Seed 1201 | Seed 1202 |
| --- | ---: | ---: |
| Program / total seconds | `813.4168 / 818.8255` | `900.7402 / 906.3906` |
| Terminal loss | `43.096647` | `42.424706` |
| Terminal gradient norm | `2.458150` | `2.859010` |
| Terminal clipping | no | no |
| Target status / floors | valid / 0 | valid / 0 |
| Frozen parity max differences | all `0.0` | all `0.0` |
| Checkpoint SHA-256 | `a5519a74e02b259cc0558223384714da7c8ee4a71148b70eb4436ce3083a8384` | `704e5ac5750f7d4e412ccc3516b0aba7dd9edb69a4a8c1816fc7ecced1ad7257` |
| Payload SHA-256 | `6429977ba1754ce5f36248104c82fa18639311a0727298bc3ed436b4a670a745` | `92e5ca376fd9660be138e8badc2ff871deff09ca97784ad247add54692352e31` |
| Transport hash | `bcbe925f2ca77996bfe05cd5b951d1a66f540327789093d0ade8fecdf0773363` | `ff53fa204d65c6b0a3e816134081f942c9f52368c2816c109595e8bdef62c355` |

Both used RTX 4080 SUPER, TensorFlow float64, memory growth, XLA JIT, batch
128, one compiled `tf.while_loop`, and no checkpoint or state reuse.

Aggregate strict invocation time was `1725.2160 s` (`28.75 min`), within the
60-minute wall budget. Terminal losses and gradients are descriptive only;
there is no uncertainty analysis supporting a seed ranking.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Pass both seeds to frozen validation | two independent exact 5,000-step jobs pass identity, device, finite/status, closure, and parity gates | no surviving engineering veto | training proxy does not establish usable HMC geometry | run identical GPU/CPU frozen objective probes | no posterior, convergence, seed-ranking, superiority, or default claim |

## Inference Status

| Status | Verdict |
| --- | --- |
| Hard veto screen | both training candidates pass |
| Statistically supported ranking | none |
| Descriptive-only differences | seed1202 terminal loss is `0.672` lower and runtime `87.6 s` longer |
| Default readiness | not evaluated |
| Next evidence needed | frozen objective identity followed by independently tuned and confirmatory HMC |

## Handoff

Continue with the Phase 3 frozen-transport subplan. The original seed1201
rejection and its separate post-validation artifact remain preserved; seed1202
uses its normal strict result directly.
