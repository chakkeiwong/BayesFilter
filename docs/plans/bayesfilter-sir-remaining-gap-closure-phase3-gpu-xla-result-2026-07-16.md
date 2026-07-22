# SIR Remaining-Gap Closure Phase 3 Result

Date: 2026-07-16

Status: `PASS_EXACT_REGISTERED_ROUTE_GPU_XLA_ENGINEERING_CERTIFICATE`

Plan: `docs/plans/bayesfilter-sir-remaining-gap-closure-master-plan-2026-07-16.md`

## Result

The exact registered Austria latent-SIR callable
`bayesfilter.highdim.ledh_contract_e_latent_sir_tf:latent_sir_contract_e_canonical_value_and_score_tf`
was executed on the trusted NVIDIA GeForce RTX 4080 SUPER with TensorFlow 2.19.1
and XLA JIT. TensorFlow logged CUDA XLA initialization, cuDNN loading, and
`Compiled cluster using XLA!`.

The GPU run consumed the same frozen `T=2,N=32` prepared tensor artifact as the
current-source CPU float64 reference. All eight prepared tensor hashes match.

| Check | Result |
| --- | --- |
| CPU status | `PASS_BOUNDED_DIAGNOSTIC` |
| GPU status | `PASS_BOUNDED_DIAGNOSTIC` |
| GPU output device | `/GPU:0` |
| absolute value delta | `5.684341886080802e-14` |
| maximum score-coordinate delta | `9.059419880941277e-14` |
| valid chart | pass |
| reset validity | pass at both observations |
| clipping-boundary chart | pass |
| GPU same-scalar FD maximum relative error | `5.382500928451606e-10` |

Artifacts:

- CPU reference:
  `docs/benchmarks/artifacts/sir_remaining_gap_closure_20260716/phase3_cpu_reference_attempt01/result.json`
- exact registered-route GPU/XLA result:
  `docs/benchmarks/artifacts/sir_remaining_gap_closure_20260716/phase3_gpu_xla_registered_route_attempt02/result.json`
- GPU artifact SHA-256:
  `542313f84219b0cdb4e339891d6c54644afed1af09b6cbdf0936d90c3cb7843d`

Attempt 1 executed the pre-registration local bound factory and passed
numerically. It is preserved as historical engineering evidence. Attempt 2 is
the claim-bearing Phase 3 artifact because it executes the exact symbol later
bound by the identity factory.

## Evidence Boundary

This phase certifies exact-source GPU/XLA execution and CPU/GPU numerical
agreement for the frozen finite route. Same-scalar FD remains an engineering
check only. This phase does not establish target accuracy, teacher equivalence,
HMC readiness, leaderboard readiness, or Zhao--Cui source-route closure.

