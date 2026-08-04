# Zhao-Cui Austria SIR Observed-Data Score Result

Date: 2026-07-30
Status: `SUPERSEDED_HISTORICAL_APF_RESULT_NOT_ACTIVE`
Historical terminal status: `BLOCK_T1_PROPOSAL_QUALITY`
Historical plan: `docs/plans/bayesfilter-zhao-cui-austria-sir-observed-data-score-active-implementation-plan-2026-07-30.md`

> Superseded by
> `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`.
> Preserve the measurements below as APF-candidate evidence. Its repair action
> is not the current continuation path.

## Outcome

The exact Austria target, latent-preclip FP32 value/score model, shared
source-order APF recursion, immutable T1 squared-TT artifact, KR memory gate,
inverse microbatching, exact Gaussian initial proposal, and complete T1 fitted
proposal branch are implemented and tested.

Execution stops before T2. The current full 36D coupled proposal is finite and
has accurate grid-map mechanics, but it fails the downstream proposal-quality
gate. More rank, training batches, and positive L1 did not repair the failure
within the bounded diagnostic ladder.

## Evidence

| Decision field | Result |
| --- | --- |
| Decision | Stop before T2; repair proposal representation/compiler. |
| Primary criterion status | Not reached: no `T=20,N=1008` branch or claim run. |
| T1 mechanics | Pass: finite value/score, immutable replayable artifact, complete proposal correction. |
| T1 proposal-quality veto | Fail: final ESS fraction `0.125013`, required `>=0.5`. |
| KR validity | Pass: inverse/forward maximum error `5.95e-8`, required `<=1e-4`. |
| Initial proposal | Pass: exact Gaussian prior; initial ESS `8/8`. |
| Main uncertainty | Whether parent-dependent innovation coordinates or a reviewed block factorization can make the conditional learnable. |
| Historical next action within this campaign | Extend the compiler for a parent-dependent current-state coordinate map, or write a reviewed compartment/neighborhood repair plan. |
| Not concluded | No T2/T20 feasibility, production KR closure, HMC readiness, posterior correctness, scientific validity, or superiority. |

Terminal structured artifact:
`docs/benchmarks/artifacts/zhao_cui_austria_sir_observed_data_score_20260730/attempt11_proposal_t1_smoke_cpu_ukf_gaussian_frame/result.json`

## Attempt Ledger

| Attempt | Classification | Result |
| --- | --- | --- |
| `attempt01_target_cpu` | target seal | Passed exact target hashes and dimensions. |
| `attempt02_mechanics_cpu` | harness failure | Wrong result key; repaired. |
| `attempt04_mechanics_cpu` | CPU mechanics | Passed bootstrap value/score mechanics. |
| `attempt05_proposal_preflight` | historical blocker | Correct at creation; superseded after T1 bridge implementation. |
| `attempt06_mechanics_gpu_xla` | harness failure | TensorFlow initialized before memory growth; repaired. |
| `attempt07_mechanics_gpu_xla` | GPU/XLA mechanics | Passed bootstrap mechanics with verified memory growth. |
| `attempt08_proposal_t1_smoke_cpu` | invalid pass classification | Coarse KR grid gave round-trip error `0.1136`; preserved and superseded. |
| `attempt09_proposal_t1_smoke_cpu_repaired_kr` | KR repair diagnostic | Round-trip passed, but initial algebraic proposal caused ESS collapse. |
| `attempt10_proposal_t1_smoke_cpu_exact_prior` | exact-prior diagnostic | Initial ESS repaired to `8/8`; T1 conditional ESS remained `1.15/8`. |
| `attempt11_proposal_t1_smoke_cpu_ukf_gaussian_frame` | terminal structured smoke | Mechanics passed; proposal-quality gate failed at ESS fraction `0.125`. |

Additional bounded diagnostics compared degree/rank/training/L1 candidates.
Observed T1 ESS fractions stayed approximately `0.126`, `0.134`, `0.257`,
`0.151`, and `0.126`; these are descriptive diagnostics, not a statistically
supported ranking. The consistent collapse is evidence against the current
unconditional coordinate-map/trainer assembly, not against all coupled or
factorized Zhao-Cui proposal research.

## Inference Status

| Inference field | Status |
| --- | --- |
| Hard veto screen | T1 proposal-quality veto fired; no nonfinite, target, memory, or KR round-trip veto. |
| Statistically supported ranking | None; diagnostics are short deterministic candidate screens. |
| Descriptive-only differences | Rank-2/longer training improved some held-out metrics but did not remove ESS collapse. |
| Default-readiness | No. |
| Next evidence needed | A parent-conditioned innovation compiler or reviewed factorized proposal that passes T1 ESS and round-trip gates before T2. |

## Red Team

Strongest alternative explanation: the stochastic trainer and unconditional
UKF coordinate frame are poorly matched to the narrow observation-conditioned
joint, so the negative result may be tuning/representation failure rather than
insufficient TT expressiveness. A parent-dependent innovation map would test
that explanation directly. The conclusion would be overturned by a fresh T1
artifact that preserves exact initial density, passes round-trip `<=1e-4`, and
reaches ESS fraction `>=0.5` on a frozen validation branch.
