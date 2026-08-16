# SSL-LSTM q=20 physical AIS repair reset memo (2026-08-10)

## State

The bridge-correct physical AIS campaign is complete.

- Full per-bridge HMC passed mechanics but failed the compute budget: `778.0 s`
  for four paths and 16 bridges, implying about `28,008 s` for the original
  material design.
- Sparse rejuvenation (HMC every eighth bridge) passed known-law and real-target
  canaries and completed the 1,000-path material campaign in `3391.4 s`.
- All 1,000 target paths were finite and status-valid.
- The central 64-bridge lane passed ESS (`0.451`), maximum weight (`0.0166`), and
  interval half-width (`0.0427`) gates.  Its descriptive negative-region estimate
  was `0.5253`.
- It failed movement: `0/800` central paths changed sign.
- The 32-bridge estimate was `0.3598`, ESS fraction `0.101`, maximum weight
  `0.210`, and difference from the central estimate `0.1655`.  Schedule stability
  failed.
- Do not use `0.5253`, `0.3598`, or the earlier direct-IS `0.4683` as posterior
  mode weights.
- No posterior archive exists.  Predictive validation and NeuTra retraining remain
  blocked.

Terminal result:

`docs/plans/bayesfilter-ssl-lstm-q20-physical-ais-repair-result-2026-08-10.md`

Artifacts:

- `r1/`: full-rejuvenation canary.
- `r2/`: sparse-rejuvenation canary.
- `r3/`: material batches, aggregate receipts, and `material.json`.

## Next repair

Plan annealed SMC, not another AIS point estimate.

1. Use the same exact target, physical chart, normalized two-local-Gaussian initial
   law, XLA batch-4 workers, and versioned detached execution.
2. Place temperatures adaptively using a predeclared conditional-ESS target and a
   bounded bisection rule.  Record every beta and incremental ESS.
3. Perform global systematic or stratified resampling when the declared ESS trigger
   fires.  Preserve parent indices and component/sign ancestry at every resampling.
4. Apply freshly bootstrapped fixed HMC after resampling.  Record target validity,
   acceptance, sign transitions, distinct ancestors, and regional ancestry.
5. Require final ESS, maximum family weight where applicable, stable region mass
   across independent batches and a schedule/control arm, ancestry from both known
   regions, and no single-ancestor collapse.
6. Keep exhaustive-mode discovery as a nonclaim.  SMC can repair finite weighting
   over supported regions but cannot prove the proposal found every mode.
7. Only after stable SMC mass evidence should the physical replica-exchange travel
   lane be extended to repeated round trips and cold convergence.

## Failure classification

| Question | Answer |
|---|---|
| Harness invalid? | No. Known-law tests, XLA, target status, identities, and receipts passed. |
| Exact target invalid? | No evidence of target failure; all 1,000 paths were valid. |
| Mathematical AIS weight update invalid? | No. The helper uses the correct pre-move incremental weight and bridge-fresh HMC bootstrap. |
| Current candidate failed? | Yes. It failed movement and schedule-stability promotion gates. |
| Research direction rejected? | No. The failure triggers the planned resampling-based repair. |

## Resilient execution

All runs longer than about one minute must use detached transient user services with
unique unit names, explicit `WorkingDirectory`, CPU/GPU environment, service and
runner wall caps, append-only absolute logs, versioned output roots, atomic progress,
and overwrite refusal.  Monitor with short artifact reads.  This workflow survived
the session-stream risk without losing a batch or active computation.
