# SIR Remaining-Gap Closure Phase 4 Result

Date: 2026-07-16

Status: `PASS_FACTORY_ISSUED_IDENTITY_NOT_SCIENTIFICALLY_ADMITTED`

Plan: `docs/plans/bayesfilter-sir-remaining-gap-closure-master-plan-2026-07-16.md`

Artifact:
`docs/benchmarks/artifacts/sir_remaining_gap_closure_20260716/phase4_identity_attempt01/identity.json`

## Result

The repository production Contract E identity factory now registers the exact
Austria latent-SIR route specification
`contract_e_chol_latent_preclip_sir_austria_v1`. A dedicated repository-owned
issuer selects the reset, value, and total-gradient callables internally; callers
cannot self-attest alternative callables through that API.

The issued identity binds:

- `contract_e_chol_v1` reset semantics;
- total direct moment/weight plus streaming-transport derivative composition;
- positive streaming row-mass quotient normalization;
- XLA-compiled value, gradient, and reset wrappers;
- fixed transport settings through source globals;
- all eight realized prepared fields, including residual design and ridge;
- the frozen Austria static model specification; and
- 52 transitive BayesFilter/FilterFlow source dependencies plus TensorFlow
  provenance.

Key digests:

| Item | SHA-256 |
| --- | --- |
| identity | `ee3a39ee64bddb934acfc1524f573d35de7e7dd86b0cc6835885e35644056af0` |
| source dependency closure | `7687f08d441d399a431856f5ffbb0d41513bae0fa11b64e8986eb8de8227c893` |
| prepared input | `d17bca002517922617f40e9873ccc89efd29e6dd354dc4a016893b4e0d1c02b4` |
| identity certificate artifact | `d229036ed9d25eaca85d05d2b0b695df735651eb92775813d278a599ae7ad8c8` |

The identity has production factory scope but remains
`factory_bound_identity_candidate_not_admitted_phase2` with `admitted=false`.
Identity is route provenance, not scientific admission.

## Focused Checks

- generic identity/schema suite: `29 passed`;
- SIR identity, substitution, mutation, wrong-JIT, forgery, and equality suite:
  combined `35 passed`;
- reset/streaming/latent-SIR regression suite: `32 passed`;
- omitted prepared fields fail closed;
- prepared ridge mutation changes prepared and identity digests;
- substituted, local, wrong-JIT, or monkeypatched callables fail closed;
- scalar prepared fields preserve rank zero in identity serialization; and
- the top-level registered callable matches the former candidate core on an
  identical full Austria fixture.

## Handoff Boundary

This identity is exact for the Austria `d=18` static model and its frozen
prepared inputs. It must not be transferred to a distinct `J=2` target. Phase 5
requires a separate two-node spatial route identity and exact-route GPU/XLA
preflight before its comparison intervals can carry claim-bearing status.

