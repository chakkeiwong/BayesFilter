# Phase 16 Second Paired Seed for Coordinate/Arm Interaction

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_HARD_GATES_ROLE_LIMITED_INTERACTION_REPLICATED`  
Budget cap: `900 s` within the unchanged `64800 s` campaign cap  
Input: the same Phase 8 metadata-bound/audited N=300 bank used in Phases 14--15  
Outputs: fresh identity and affine attempt directories under `phase16`

## Question and scope

Does the Phase 15 arm-by-preconditioner interaction persist under a second
NeuTra initialization/optimizer seed? This is a replication of representation
optimization, not a particle-bank uncertainty study. The target, rows,
weights, mode axis, partitions, profile, architecture, and 300-update budget
remain fixed. A later phase is required before making claims across particle
authority seeds.

## Evidence contract

Primary evidence is the paired difference in full-bank weighted mean and
covariance residuals for each arm. All hard engineering and target/status gates
must pass in both coordinate routes. One additional seed still provides
descriptive evidence only; no p-value, superiority, default, posterior, or IID
claim is permitted.

Vetoes are stale/mismatched protocol or target hashes, wrong mode axis or
weights, non-finite values, failed target/status/parity, missing full-bank
metrics, failed memory-growth/XLA/batch receipts, or an overwritten output
directory. A large residual remains a candidate/evidence result, not a program
blocker.

## Assumption and numeric audit

The second seed tuple `20260825 8305` is a campaign hypothesis chosen only to
separate optimizer initialization from the Phase 15 tuple `20260825 7305`; it
is not a statistically sufficient seed count. The 300 updates, tuning profile,
N=300 bank, and three arm configurations are inherited comparison settings,
not promoted defaults. The earliest checks are artifact hash parity and the
full-bank metric presence.

## Pre-mortem

- A result may pass while the interaction is caused by one unusually favorable
  seed. The note will retain descriptive language and nominate more seeds only
  if the sign is replicated.
- A validation loss may select an arm that has worse full-bank moments. Selection
  is recorded, but full-bank metrics remain primary.
- An affine route may pass its round trip while its composition is wrong. The
  transformed target/status receipt and affine/flow parity gate are mandatory.
- A runtime failure could be infrastructure-only. Preserve the unique output,
  classify it, and repair the affected route without changing the comparator.

## Commands

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_neutra_screen_2026_08_25.py \
  --precondition identity --profile tuning \
  --plan docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase16-paired-replication-subplan-2026-08-25.md \
  --m0-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase6-attempt9-metadata-n300-seed2401 \
  --steps 300 --seed 20260825 8305 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase16-attempt1-identity-seed8305

TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_neutra_screen_2026_08_25.py \
  --precondition affine --profile tuning \
  --plan docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase16-paired-replication-subplan-2026-08-25.md \
  --m0-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase6-attempt9-metadata-n300-seed2401 \
  --steps 300 --seed 20260825 8305 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase16-attempt2-affine-seed8305
```

Run identity first, then affine only if identity passes its hard gates. No HMC
or source/default promotion is in scope.

## Exit and refresh

If both routes pass, write a paired two-seed result and refresh Phase 17 toward
the smallest discriminating artifact: a source-faithful modular transform
contract (ETPF/GenUT/LEDH) if conditioning remains interaction-limited, or a
bounded multi-seed representation ladder if the interaction is consistent. If
one route fails, repair that route and preserve the other receipt.
