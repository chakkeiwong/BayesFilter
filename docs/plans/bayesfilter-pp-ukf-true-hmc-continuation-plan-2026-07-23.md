# PP-UKF true-HMC retained continuation plan

Date: 2026-07-23

Status: `PLAN_REVIEWED_READY_FOR_EXECUTION`

Parent plan: `docs/plans/bayesfilter-pp-ukf-true-hmc-validation-plan-2026-07-22.md`

Reset memo: `docs/plans/bayesfilter-repository-hygiene-and-pp-ukf-reboot-reset-memo-2026-07-23.md`

## Research question

Do the three PP-UKF candidates censored at the driver’s incorrect 3,000-draw
retained cap (`L=9,12,17`) pass the declared sequential HMC retained gates when
their exact archived Markov-chain prefixes continue to the policy maximum of
10,000 retained transitions per chain?

This is a continuation of the same frozen target/transport scope, not a new
tuning campaign, target change, candidate ranking, or posterior claim.

## Evidence contract

| Field | Binding decision |
|---|---|
| Target | PP-UKF target signature `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5` |
| Frozen transport | SHA-256 `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221` |
| Candidates | Exactly `L=9,12,17`, replacing only their attempt-09 rows |
| Prefix source | Attempt-09 progress metadata and its hash-bound cumulative retained tensors; L=9/L=12 prefixes originate from the immutable attempt-07 archives referenced by the prior row |
| Primary criterion | Warmup already passed; cumulative retained modern R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS `>=400`, finite state/target/log acceptance, valid target telemetry, all-chain movement, and native divergence veto when exposed |
| Continuation method | Start each chain from the final archived latent prefix state and derive deterministic retained chunk seeds from the original retained seed at chunk index 6 onward |
| Retained maximum | `10,000` transitions per chain, repository policy maximum |
| Hard veto | Prefix hash/shape/identity mismatch, stale or missing archive, nonfinite continuation state/target/log acceptance, invalid target telemetry, no movement, native divergence when exposed, budget exhaustion, or output-root collision |
| Explanatory diagnostics | Acceptance, finite extreme log-acceptance counts, runtime, and prefix-versus-continuation changes |
| Ranking | Forbidden; the resulting viable rows remain unranked |
| Nonclaims | No sampler superiority, posterior recovery, exact likelihood claim, default readiness, or production claim |
| Artifact | Fresh `docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-10/` public result, manifest, ignored progress/private tensors, and this plan/result note |

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| 10,000 retained maximum | Repository NeuTra policy and parent plan | Repeating the 3,000-cap defect | Unit test and emitted config payload | Reviewed default |
| Archived-prefix continuation | Attempt-09 private archive and deterministic chunk-seed controller | Starting from the wrong state or duplicating draws | Verify every prefix SHA-256, shape, seed, and final-state hash before sampling | Required repair |
| Replacement rows | Attempt-09 terminal result has all ten rows | Silently dropping prior candidates or treating failed rows as complete | Merge-by-candidate-ID test and final ten-row count | Required repair |
| Same frozen controls | Parent campaign candidate manifest | Tuning or target drift during continuation | Target/transport/candidate identity checks before TensorFlow execution | Binding invariant |
| Fixed GPU/XLA/memory policy | Repository GPU policy | Device or allocator mismatch | Trusted preflight and run manifest | Binding invariant |
| Remaining budget | Attempt-09 aggregate `42,403.504540 s` of `86,400 s` | Continuation exceeds cap | Check before every candidate and every retained chunk; record aggregate charge | Hard stop |

## Skeptical plan audit

1. **Wrong baseline:** this does not retune or compare new controls; it tests only
   the three rows censored by a harness cap.
2. **Replay risk:** a fresh run from the original initial state would waste
   budget and would not demonstrate continuation. The plan requires archived
   prefix state and chunk index 6 seeds.
3. **Proxy promotion:** acceptance and extreme log-acceptance counts remain
   explanatory. Only cumulative R-hat/ESS and health gates determine viability.
4. **Hidden row corruption:** the final result must contain the seven preserved
   attempt-09 rows plus exactly three replacement rows, ordered by frozen `L`.
5. **Stale artifact risk:** the prefix hash, target signature, transport hash,
   candidate ID, step size, seed, shape, and retained count are checked before
   any continuation sampling.
6. **Budget risk:** attempt-10 carries in `42,403.504540 s`; it may consume at
   most `43,996.495460 s` and must stop before the aggregate cap.
7. **Scientific overclaim:** even a pass establishes only a valid retained HMC
   screen for this frozen scope; it does not establish posterior correctness,
   superiority, or default readiness.

Audit decision: `PASS_FOR_REPAIR_AND_CONTINUATION`. A replay-only implementation,
silent prefix omission, or candidate ranking would fail this audit and must not
be executed.

## Repair audit result (2026-07-23)

The implementation audit passed after the following bounded repair:

- `MAX_RESULTS` is now `10,000`, matching the sequential NeuTra policy.
- Continuation validates the hash-bound attempt-09 progress file, all ten
  candidate rows, each retained chunk seed/shape/hash, cumulative prefix
  tensor hashes, and finite prefix values before TensorFlow sampling.
- Continuation starts from each archived final latent state and resumes at
  retained chunk index `6`; it does not replay the original chain prefix.
- Replacement checkpoints and the terminal result merge exactly three fresh
  rows into the preserved ten-row attempt-09 payload.
- The aggregate budget is checked before each candidate and before every
  continuation chunk, including the `42,403.504540 s` carry-in charge.

Focused verification passed with `15 passed` and no sampling was performed by
the tests. The skeptical audit remains `PASS_FOR_REPAIR_AND_CONTINUATION`:
the repair changes only continuation machinery, leaves the frozen target,
transport, seeds, candidates, gates, and nonclaims unchanged, and introduces
no ranking or posterior interpretation.

## Implementation steps

1. Change the PP-UKF driver cap to the policy maximum `10,000`.
2. Add a generic shared-controller retained continuation helper that accepts a
   final latent state and archived retained prefix, runs only remaining chunks,
   and computes diagnostics over prefix plus continuation.
3. Add PP-UKF archive parsing and SHA-256/shape/seed/identity validation.
4. Add explicit `--replace-candidate-index` CLI controls and merge replacement
   rows without mutating attempt-09.
5. Add unit tests for cap, prefix validation, chunk-index seed continuation,
   replacement merge, output freshness, and full ten-row terminal payload.
6. Run focused CPU-only tests and a no-sampling continuation preflight.
7. Launch attempt 10 in trusted detached tmux with replacement indices `1,2,5`
   and carry-in `42,403.504540` seconds.
8. Monitor progress without editing the running source or prior artifacts.
9. Write a terminal continuation result note with decision/inference tables.

## Launch contract

Only after steps 1--6 pass:

```bash
tmux new-session -d -s pp_ukf_hmc_24h_20260723_reboot \
  "cd /home/chakwong/BayesFilter && \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
    docs/benchmarks/run_pp_ukf_true_hmc_validation_20260722.py \
    --output-root docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-10 \
    --replace-candidate-index 1 --replace-candidate-index 2 --replace-candidate-index 5 \
    --resume-progress docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/progress.json \
    --prior-elapsed-seconds 42403.504540 > \
    docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-10-launch.log 2>&1"
```

The output root is fresh and all prior artifacts are append-only. The launch
must be abandoned if any identity, prefix, budget, GPU, or memory-growth gate
fails.

## Stop and interpretation rules

- Stop before launching a candidate if the aggregate budget would be exceeded.
- Preserve a failed continuation as evidence; do not overwrite the prior row or
  relax the declared gates.
- If a candidate passes before 10,000, stop that candidate at the first valid
  cumulative gate and record its retained count.
- If all three candidates finish, produce a ten-row merged result and explicitly
  mark the three rows as continuation replacements.
- Do not rank any viable candidate from this phase.

## Planned result tables

The terminal note must include:

- a decision table with candidate, prefix count, final retained count, primary
  criterion, veto status, uncertainty, and next action; and
- an inference-status table covering hard vetoes, statistically supported
  ranking, descriptive differences, default readiness, and evidence still needed.
