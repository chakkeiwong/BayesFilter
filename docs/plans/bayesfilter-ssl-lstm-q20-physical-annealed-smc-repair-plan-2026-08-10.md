# SSL-LSTM q=20 physical annealed-SMC repair plan (2026-08-10)

Status: `CANARY_COMPLETE_AND_PASSED; MATERIAL_PLAN_REQUIRED`

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Can global resampling repair the schedule instability and proposal-dependent local isolation observed in physical AIS while preserving particles descended from both known sign regions? |
| Candidate | TensorFlow/TFP annealed SMC in the fixed physical chart: global conditional-ESS temperature placement, global systematic resampling at every nonterminal stage, one freshly bootstrapped fixed-HMC move after resampling, and explicit root/sign ancestry.  No NUTS. |
| Expected failure | Weight increments may force too many stages; resampling may collapse to a few roots or one sign; one HMC move may remain local; fresh-wave/XLA integration may be invalid. |
| Canary promotion criterion | Reach beta `1` within 24 stages; every state/value/status finite and valid; at least one nonterminal global resampling; terminal pre-resampling ESS finite; final ancestry contains at least one distinct initial root from each known sign; all 25 batch-4 workers use XLA and exact target identities; wall time within `3,600 s`. |
| Promotion veto | Non-finite/invalid target, beta stall, failure to reach beta 1, malformed resampling indices, loss of either known sign ancestry, receipt mismatch, or canary wall cap. |
| Continuation veto | Target/source identity drift, repeated XLA/fresh-wave harness invalidity after one localized repair, or campaign cap.  A scientifically valid canary failure triggers SMC schedule/mutation repair, not posterior issuance. |
| Repair trigger | Excessive stages triggers lower cESS target or a reviewed beta floor; root/sign collapse triggers a less aggressive resampling rule or more particles; low movement with ancestry retained triggers more HMC moves or a global mutation kernel. |
| Explanatory diagnostics | Beta path, incremental ESS, parent counts, unique root counts by sign, HMC acceptance, sign changes, runtime, and log-normalizer increments. |
| Must not be concluded | Posterior mass, exhaustive mode discovery, HMC convergence, NeuTra repair, posterior correctness, or predictive validity from the canary. |

## Evidence contract

The comparator is the failed sparse AIS material artifact
`docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r3/material.json`.
The exact target, proposal, and physical chart remain unchanged.  The mechanism
under test is global resampling with ancestry; local target/HMC evaluation remains
batch-4 CPU/XLA.

The SMC weight update is the Del Moral--Doucet SMC-sampler/AIS special case:
for bridge `pi_beta proportional q^(1-beta) p^beta`, the pre-mutation incremental
weight is `delta_beta * (log p - log q)`.  An unbiased global systematic resampler
selects particles from those normalized weights, resets weights to equal, and a
freshly bootstrapped HMC kernel invariant for the new bridge mutates the resampled
particles.  The terminal beta-1 weighted population is preserved before terminal
resampling.  This follows the inspected local paper at
`.localresources/papers/multimodal_hmc/del-moral-doucet-2002-smc-samplers-preprint.pdf`,
especially the sampling/weight/resampling construction and its AIS connection.

The canary output root is
`docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r1/`.

Exact canary command/environment:

```bash
systemd-run --user \
  --unit=bayesfilter-ssl-q20-physical-annealed-smc-canary-20260810-r1 \
  --collect \
  --property=WorkingDirectory=/home/ubuntu/python/BayesFilter \
  --property=CPUQuota=10000% \
  --property=RuntimeMaxSec=3600s \
  --property=TimeoutStopSec=60s \
  --property=StandardOutput=append:/home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r1/canary.log \
  --property=StandardError=append:/home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r1/canary.log \
  --setenv=CUDA_VISIBLE_DEVICES=-1 \
  --setenv=TF_CPP_MIN_LOG_LEVEL=3 \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 \
  --setenv=NUMEXPR_NUM_THREADS=4 \
  --setenv=TF_NUM_INTRAOP_THREADS=4 \
  --setenv=TF_NUM_INTEROP_THREADS=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_physical_annealed_smc_canary_2026_08_10.py
```

## Numeric/default audit

| Choice | Provenance/status | Justification | Failure mode / early diagnostic |
|---|---|---|---|
| 100 particles | Matches each prior independent AIS batch; canary hypothesis | Exercises one full global 25-worker population without claiming mass precision | Root survival may be noisy; ancestry telemetry and nonclaim |
| Conditional ESS target `0.80` | Conservative hypothesis; stricter than TFP experimental default `0.50` and prior local diagnostic `0.70` | Small increments reduce the 32-bridge weight collapse | Too many stages; 24-stage cap and beta path |
| 24 bisection iterations, beta tolerance `1e-6` | Derived numerical accuracy for `[0,1]` bisection (`2^-24 < 1e-6`) | Deterministic bounded selector | Stall from tiny increments; positive-increment and stage gates |
| Maximum 24 stages | Convenience compute cap, not a scientific default | Allows substantially more adaptive levels than the failed 64/8 sparse schedule's eight mutations | Candidate may need more; failure classified as schedule repair |
| Global systematic resampling at every nonterminal stage | Del Moral--Doucet mechanism and inspected TFP low-variance resampler | Directly tests the missing AIS mechanism and keeps terminal weights measurable | Ancestry collapse; parent/root/sign counts |
| One HMC move, step `0.03`, `L=4` | AIS canary-measured local kernel; warm-start hypothesis | Isolates resampling benefit within bounded cost | No sign movement; acceptance/sign telemetry, later mutation repair |
| Fresh 25-worker waves, batch 4, four cores each | Measured AIS topology and prelaunch affinity audit | `max_tasks_per_child=1` guarantees each task pins a fresh process before TensorFlow import; avoids thread-pool affinity drift | Repeated compile cost; stage timing and 3,600-second cap |
| Canary wall `3,600 s` | Derived from at most 24 one-move stages and AIS measured cost, with margin | Bounded mechanics test | Under-budget if adaptive stages are unexpectedly expensive |

## Baseline ladder

| Rung | Status |
|---|---|
| Direct importance sampling | Failed weight concentration and scale sensitivity; historical comparator. |
| Sparse AIS | 64-bridge weights passed local concentration gates but failed movement and 32/64 stability; direct comparator. |
| Plain adaptive SMC | Current canary: global systematic resampling plus one fixed HMC move. |
| Enhanced SMC | Only if triggered: diversity-aware resampling, more mutation, or reviewed global mutation. |

## Execution plan

1. Implement TensorFlow diagnostic primitives for normalized weights, conditional-ESS
   beta selection, systematic resampling, and ancestry diagnostics.  Add known-law,
   replay, threshold, and invalid-input tests.
2. Implement fresh 25-worker stage waves with `max_tasks_per_child=1`.  Spawn each
   wave before the coordinator imports TensorFlow.  Every task binds one disjoint
   four-core group before target construction, then runs the exact target/chart and
   XLA bridge-HMC step for its batch-4 shard.
3. Generate 100 initial proposal particles with stateless disjoint seeds.  Globally
   select beta, weight, and systematic-resample at every nonterminal stage.  Archive
   beta, weights, parent indices, root identities/signs, pre/post states, HMC
   telemetry, and target status after every stage.
4. At beta 1, preserve the weighted terminal population without resampling.  Apply
   only mechanics/timing gates; do not issue a mode-weight estimate.
5. If the canary passes, write a separate material SMC plan with independent batches,
   terminal mass uncertainty, schedule/cESS sensitivity, ancestry gates, and a new
   compute budget.  Do not infer those defaults from this canary.

## Skeptical plan audit

| Risk | Resolution |
|---|---|
| Wrong baseline | Direct IS and sparse AIS terminal artifacts are explicit comparators. |
| Proxy promotion | Acceptance, runtime, and beta count are explanatory; ancestry, target validity, and terminal reach are mechanics gates only. |
| Biased shard resampling | Resampling is global over 100 particles in the coordinator; workers cannot resample locally. |
| Terminal mass destroyed by resampling | Beta-1 weights and signs are recorded before terminal resampling, and no terminal resampling occurs. |
| Unreviewed TFP defaults | TFP experimental all-in-one tuning is not used; step, L, cESS, stage cap, and resampling are explicit hypotheses. |
| Hidden mode assumption | Both known signs are tracked, but exhaustive discovery remains a nonclaim. |
| Expected candidate failure stops direction | Failure triggers the declared schedule/mutation/particle repair unless target or harness validity fails. |
| Worker affinity drift | Fresh one-task processes pin before TensorFlow import; no framework thread pool is reused under a different shard affinity. |
| Stream loss | Detached service, runner cap, atomic per-stage progress, versioned receipts, and overwrite refusal. |

The audit passes for a mechanics/timing canary only.  It does not authorize a
posterior archive, a mode-weight claim, NeuTra training, or predictive validation.

## Pre-mortem

The canary could reach beta 1 with both sign labels merely because initial proposal
coverage was good; that would not prove global movement or mode completeness.  It
could fail by ancestry collapse even when the mathematical SMC idea is viable with
more particles or a different resampling cadence.  Worker shards could also produce
a clean-looking result while accidentally using shard-local weights; the
global parent-index shape and whole-population systematic-resampling tests are the
cheap discriminator for that implementation error.
