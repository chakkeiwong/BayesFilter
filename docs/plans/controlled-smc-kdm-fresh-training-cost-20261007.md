# Fresh KDM initialization and author/TF training cost — 7 October 2026

Continue the authorized replication campaign. Source-stage mechanics and one TF
epoch from historical fitted weights passed, but that starting point cannot
establish training from scratch. This phase runs one fresh development epoch in
each source stage: weights, MDPF, A-MDPF. It does not qualify either replication.

## Question, criteria and interpretation

Use pinned author commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18 as comparator.
Execute its actual setup_training -> Trainer.train -> training/validation epoch
call chain, with constructor initialization and source best-checkpoint transfers.
Export source constructor parameters and generated data for the corresponding TF
run. Each TF stage starts from the same constructor values but consumes its own
preceding stage's best checkpoint. Independent native random streams implement
the same checked sampling law; equal seed integers do not imply identical draws.

The implementation screen requires complete train5000/validation1000 coverage,
79/4 batches including final 8/232, source frozen-parameter invariance, finite
losses/gradients/parameters, exact checkpoint coverage, verified source identity,
and stable TF traces for the four batch shapes. The source caps remain 400/800/800
in provenance; the explicit one-epoch override is a cost diagnostic. All three
stages must complete before estimating the full three-stage cost.

Missing data, invalid source identity, missing gradients, additional nonfinite
updates, broken checkpoint transfer or budget exhaustion stop the affected run.
Local harness repairs may retry within this allocation. Existing float32
discrepancies are replication promotion vetoes, not vetoes on collecting fresh
training/cost evidence. One-epoch losses and extrapolated costs are explanatory;
they cannot rank learned methods or establish no-worse performance. Heuristic
quality comparisons belong to subsequent evaluation, not this cost screen.

## Source and default audit

Source anchors: training_runner.py:90-129; trainer.py:407-610,613-699;
kde_particle_filter.py:280-355 (load_pretrained); individual_training/config.yaml;
configs/our_experiments.yaml and our_experiments_exp3_init.yaml; experiment0002_
importance/config.yaml and experiment0003_importance_init/config.yaml. Configuration
merging uses the author's main.py helpers. The previous source-stage result
documents the mathematical, sampling, gradient and optimizer comparisons.

Fresh development seeds: data 202610071, constructors 202610072..74. These are
convenience reproducibility choices, not selected performance seeds or untouched
qualification data. Dataset size, sparse labels, losses, architecture, clipping,
weight decay and batch sizes come from source. No new numerical protection,
optimizer tuning or tolerance is introduced. The CPU source uses one Torch
thread and one loader worker; CPU cost has not previously been measured and may
make full source training unaffordable. Source NumPy/PyTorch are independent
reference exceptions. TF numerical kernels use GPU/XLA, float32, TF32 off,
verified memory growth, and no NumPy. CPU row shuffling is orchestration only.
The existing development replay checks do not certify every fresh update.

## Execution and budget

At most six attempts, total <=1800 CPU seconds and <=1200 GPU seconds, deducted
once from the current ~90 CPU/~47.9 GPU hours. Fresh output roots are
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/
kdm-fresh-training-cost-20261007-XX. Keep complete logs, command/environment/source
hashes, seeds, wall time, checkpoints, per-epoch/batch progress and data hashes.
Launcher: scripts/run_kdm_fresh_training_cost.py --mode source|tf --output PATH;
TF additionally requires --reference PATH to the successful source result.

1. Verify source/configuration and run actual author stages on CPU with GPUs
   intentionally hidden, exporting fresh constructor states before transfers.
2. Run the checked TF consumers on the exported fresh data/initialization with
   the same source settings and its own stage checkpoints.
3. Record measured and extrapolated costs, uncertainty and the number of complete
   trained pairs the remaining allocation can plausibly support. A new bounded
   phase may launch full training after this evidence; do not silently expand
   this one-epoch diagnostic into long training.
4. Refresh master, checkpoint, result and once-only budget ledger.

## Skeptical audit before execution

Pass for initialization/cost collection only. The older 8.59-hour author trial
was GPU work and is not a CPU estimate. One epoch underestimates amortized cost
uncertainty and cannot predict early stopping. Export constructor states before
checkpoint loading, otherwise trained-source values would contaminate the TF
initialization. For more than one epoch, transfer the validation-selected best
checkpoint, not the final checkpoint. Preserve both pre-existing float32
failures; no interpretation of this phase closes them. Full training and fresh
qualification remain separate acceptance requirements.
