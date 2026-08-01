# PP-UKF True HMC Validation Plan

Date: 2026-07-22

Status: `PARTIAL_RESULT_EXECUTION_BOUNDARY_BLOCKED`

## Objective

Run the first claim-bearing PP-UKF frozen-kernel HMC validation for the
corrected ten-candidate next-round set:

```text
L=(5,9,12,13,14,17,18,19,24,25)
```

The test must use the repository sequential NeuTra HMC controller, separate
warmup from retained draws, and require health, modern R-hat, ESS, energy-error,
and declared PP-UKF target/reference diagnostics before admitting a candidate.

This plan separates implementation/preflight from the expensive GPU campaign.
The full run is not authorized by the prior tuning budget because that campaign
charged `13,750.560450 s` of a `14,400 s` ceiling. A new compute budget and a
fresh validation partition are continuation prerequisites.

The owner has now authorized a twenty-four-hour local compute budget.
This campaign binds that authorization to an `86,400 s` aggregate cap; it does not expand
the target, candidate set, hardware class, or privacy boundary.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Does any corrected PP-UKF frozen candidate support a healthy, converged retained HMC run under the exact target/transport scope? |
| Candidate mechanism | Fixed identity-metric PP-UKF NeuTra HMC with independently tuned primary epsilon or exact inherited one-hop coverage epsilon |
| Candidate set | `L=(5,9,12,13,14,17,18,19,24,25)`; no `L` ranking before validity gates |
| Primary promotion criterion | Warmup readiness, retained modern R-hat `<=1.01`, declared bulk/tail ESS, finite target/state/log acceptance, all-chain movement, native divergence when available, and declared PP-UKF reference/target checks |
| Hard veto | Any identity/scope drift, stale tuning artifact, validation-data overlap, nonfinite state/target/log acceptance, invalid status telemetry, no chain movement, positive native divergence when available, warmup failure, retained R-hat/ESS failure, or reference gate failure |
| Repair trigger | Candidate-local numerical failure, insufficient mixing, or target/reference disagreement; preserve the candidate as evidence and diagnose under a new bounded repair phase |
| Explanatory diagnostics | Acceptance, runtime, interval/tail summaries, epsilon, compile cost, and candidate ordering |
| Ranking | Forbidden in this phase; viable candidates remain an unranked set |
| Nonclaims | No sampler superiority, default readiness, broad PP-UKF correctness, or scientific claim beyond the declared validation scope |

## Evidence Contract

- Target: current PP-UKF principal-square-root UKF posterior with target
  signature `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`.
- Frozen transport SHA-256:
  `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`.
- Metric: fixed identity metric in transport coordinates; no runtime mass
  adaptation and no runtime epsilon retuning.
- Primary controls: each primary retains its own tuned epsilon from the
  statistically corrected tuning artifact.
- Coverage controls: each one-hop candidate retains the parent's epsilon
  bit-for-bit and its parent candidate identity.
- Fresh data: calibration/tuning and claim validation partitions must have
  distinct repository-issued signatures. Existing tuning screens cannot be
  reused as claim validation data. Because the PP-UKF posterior is defined on
  one immutable frozen observation sequence, the fresh claim partition is an
  execution partition: new initial latent states and new warm-up/retained
  chain seeds, hash-bound in the run manifest and disjoint from all tuning
  seeds. This is not a claim of a new observation-data split.
- Hardware/backend: TensorFlow/TFP, float64, GPU, XLA, TF32 setting recorded,
  memory growth verified before TensorFlow logical-device initialization.
- Controller: `bayesfilter.inference.neutra_hmc.run_sequential_neutra_hmc`.

## Sequential HMC Policy

Each candidate uses four chains and the shared sequential policy:

- warmup chunks retained privately but excluded from posterior summaries;
- warmup minimum 2,000 transitions per chain;
- warmup recent-window maximum rank-normalized split/folded R-hat `<=1.05`;
- retained sampling grows cumulatively;
- retained minimum 1,000 transitions per chain;
- retained modern R-hat `<=1.01`;
- retained bulk ESS and tail ESS thresholds declared before launch;
- retained maximum 10,000 transitions per chain;
- finite target/state/log-acceptance, movement, target-status, and
  native-divergence checks when exposed on every chunk.

Non-finite log acceptance is a hard health veto. A finite log acceptance below
the configured extreme-tail threshold is explanatory only: it records a poor
proposal, not a proven divergence, and does not stop warmup. The TFP HMC kernel
currently reports native divergence as not exposed. That is recorded as an
evidence limitation, not as zero divergences.

## Implementation Phases

1. **Plan and skeptical audit**
   - Record the exact target, transport, candidates, policy, stop conditions,
     budget requirement, and nonclaims.
   - Audit wrong baseline, proxy promotion, missing diagnostics, scope drift,
     stale artifacts, and resource mismatch.
2. **Generic validation integration**
   - Use the generic frozen-kernel candidate/scope/artifact contract.
   - Add a PP-UKF manifest/preflight driver that reconstructs the ten controls
     from the hash-bound corrected tuning artifact.
   - Bind each candidate to the sequential-controller configuration without
     executing retained sampling.
3. **CPU-safe mechanics tests**
   - Test candidate-set reconstruction, exact inherited epsilon, parent
     provenance, scope mismatch rejection, and configuration invariants.
   - Run only tiny fixture tests; no scientific interpretation.
4. **Trusted GPU preflight**
   - Verify GPU visibility, TensorFlow memory growth, XLA, target/transport
     signatures, candidate manifest, and output-root freshness.
   - Write a preflight artifact under a fresh versioned root.
5. **Full claim-bearing campaign, gated**
   - Requires a new explicit compute budget and a fresh validation partition.
   - Execute mechanics screen, sequential warmup, retained sampling, and
     final diagnostics for all ten candidates.
   - Stop on the first declared continuation veto; preserve completed evidence.
6. **Terminal interpretation**
   - Produce a decision table and inference-status table.
   - Distinguish hard vetoes, viable candidates, descriptive differences, and
     statistically supported rankings (none unless separately justified).

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Shared sequential controller | Owner NeuTra policy and existing implementation | A legacy driver may use fixed burn-in/terminal counts | Driver must instantiate `SequentialNeuTraHMCConfig` and record policy ID |
| Four chains | Shared controller minimum | Fewer chains invalidate modern R-hat | Configuration rejects fewer than four |
| R-hat and ESS gates | Repository convergence policy | Short runs can look acceptable | Cumulative retained ladder and thresholds are recorded |
| Native divergence unavailable | TFP kernel capability | Absence could be mistaken for zero | Manifest records `not_exposed_by_tfp_hamiltonian_monte_carlo` |
| Existing tuning artifact as control source | Corrected compatibility artifact | Tuning data could leak into claim validation | Fresh validation partition signature is mandatory |
| Fixed identity metric | PP-UKF current scope | Metric drift changes the target scope | Candidate and transport manifests bind metric signature |

## Skeptical Plan Audit

- **Wrong baseline:** this is not a new tuning run; it validates the corrected
  frozen candidates under one fixed target/transport scope.
- **Proxy promotion:** acceptance and runtime nominate/explain only; warmup,
  R-hat, ESS, health, energy-error, and reference gates control validity.
- **Missing stop conditions:** every chunk has health and energy checks; warmup
  and retained caps are explicit; output root must be fresh.
- **Unfair comparison:** all ten candidates use the same chain count, seeds
  policy, data scope, controller, and declared diagnostics.
- **Hidden assumptions:** native divergence is unavailable and recorded;
  retained ESS thresholds and reference estimands must be declared before the
  full run.
- **Stale context:** the prior artifact is hash-bound and used only for fixed
  controls; its tuning draws are not claim samples.
- **Environment mismatch:** GPU/XLA/memory growth are preflight gates, not
  post-hoc assumptions.
- **Resource mismatch:** the previous tuning ceiling has only about 649 s
  headroom, which is insufficient for ten candidates with 2,000+ retained
  transitions per chain. The owner-supplied 86,400 s campaign cap supersedes
  that blocker while preserving the old artifacts.

Audit decision: `PASS_FOR_CAMPAIGN_EXECUTION`; the owner-supplied budget and
fresh execution partition satisfy the continuation prerequisites. The run
still stops on any target, health, energy-error, warm-up, R-hat, ESS, or budget
veto.

## Planned Artifacts

```text
docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/
  attempt-02/
    preflight.json
    candidate_manifest.json
    run_manifest.json
    progress.json
    private/...
    public_result.json
```

No existing artifact may be overwritten.

## Execution Result Requirement

The implementation phase must leave a runnable preflight command and focused
tests. A preflight pass does not authorize retained sampling. A full campaign
may begin only when this plan is amended with a fresh validation partition,
compute budget, exact ESS/reference thresholds, and a user-authorized launch.

## Execution Status (2026-07-22)

- Phases 1--4 completed. The candidate manifest and preflight are preserved at
  `docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-02/`.
- The preflight binds all ten candidates to candidate-specific epsilon,
  leapfrog count, deterministic warm-up/retained seeds, and the shared
  `bayesfilter_neutra_sequential_hmc_v1` controller. It performs no sampling.
- Trusted GPU checks passed independently: one RTX 4080 SUPER was visible;
  TensorFlow 2.19.1 saw `/device:GPU:0`; and memory growth was verified before
  logical-device creation.
- A harness defect found during execution (positional call to a keyword-only
  `build_preflight` argument) was repaired and covered by the preflight test.
- The skeptical audit remains binding: the prior tuning campaign has only
  about 649 seconds of its 14,400-second ceiling left, and its calibration data
  cannot be reused as claim validation data. No retained HMC sampling was
  launched.
- Owner authorization supplies an 86,400-second aggregate cap. Exact prospective gates are
  now declared: retained rank-normalized split/folded R-hat `<=1.01`, bulk ESS
  `>=1000` per parameter, tail ESS `>=400` per parameter, finite state/target/
  log-acceptance, all-chain movement, valid target status, and zero declared
  energy-error vetoes. The native TFP divergence field remains unavailable and
  is recorded as unavailable, never as zero.
- The full campaign is executing under a fresh artifact root. No result is
  promoted until the terminal decision and inference-status tables are written.

## Partial Execution Result (2026-07-22)

- `attempt-04/progress.json` preserves three interrupted warmup prefixes:
  `L=5`, `L=9`, and `L=12`. All three passed finite-state, finite-target,
  target-status, and movement checks. Their finite extreme log-acceptance tails
  were incorrectly treated as energy-error vetoes by the prior controller.
  They are reclassified as explanatory only; the candidates remain unevaluated
  because only 1,000 of the required 2,000 warm-up transitions completed.
- Two later candidate-scoped retries (`attempt-05`, `attempt-06`) were
  terminated by the managed long-process boundary before their first durable
  checkpoint. They contain no scientific candidate result.
- The 86,400-second budget remains authorized, but the current driver is not
  yet a reliable resumable campaign harness. Further GPU execution is blocked
  until chunk-level checkpoint/resume is implemented and tested; the three
  preserved candidate rows remain valid evidence and are not rerun.
