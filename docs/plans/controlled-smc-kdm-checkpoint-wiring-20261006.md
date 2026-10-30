# KDM checkpoint wiring audit — 6 October 2026

Parent: controlled-smc-kdm-replication-master-program-20260928.md.
This is a metadata/reference check under the existing replication-only authorization.

Question: which checkpoint does the released bearings evaluation actually load?
Comparator: unchanged author source at b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18,
under src/experiments/bearings_only_experiments_methodical_25_particles_multiple_runs.
Check both experiment0002_importance (MDPF) and experiment0003_importance_init
(A-MDPF), all eleven generated trials. Preserve the original source; do not
repair author configuration based on task labels.

Evidence contract: execute the author generator, common-parameter merger,
run_experiment and KDEParticleFilter.load_pretrained function bodies extracted
without modification. Substitute a recording torch.load and empty model state,
and stop at setup_evaluation. Record the path passed to torch.load and the
metrics passed to evaluation. Confirm the active-task paths agree with their
resolved configuration. A missing or unexpected call, changed source bytes,
unresolved placeholder or inconsistent trial path invalidates this audit.
Checkpoint existence is a training prerequisite, not a metadata pass criterion.
No tensor loading, numerical filter, training, paper-result or equivalence claim
follows from this check. NumPy, PyTorch and TensorFlow are not imported.

Default audit: eleven trials come from the author's generator; the MSE suffix
is a hypothesis about training provenance, not an authority. The source's
do_run flags select active evaluation. Device is explicitly CPU metadata-only,
with CUDA_VISIBLE_DEVICES=-1; the fake model is an observability boundary.
Mocked numerical state cannot establish checkpoint compatibility, so this is
only a wiring check. The exact untouched source hashes and extracted callable
line ranges must be saved, plus dependency paths and their local existence.

Skeptical review: the task name is not sufficient to identify the loaded model.
Following the actual consumer to torch.load resolves this ambiguity. Preserve
all training tasks in the dependency ledger even though only evaluation is
active. No stochastic ranking or heuristic-dominance claim is in scope.

Budget: at most two executions of 60 CPU seconds each, zero GPU, charged to
the existing allocation. Fresh kdm-checkpoint-wiring-20261006-NN directories
under artifacts/controlled-smc-kdm-replication-20260928. No packages, network,
training, GPU initialization or author-source writes. Repair localized harness
failures within this bound; source drift or changed scientific scope stops work.

Command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_kdm_checkpoint_wiring.py --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-checkpoint-wiring-20261006-01

Result: controlled-smc-kdm-checkpoint-wiring-result-20261006.md. Preserve summary,
per-task resolved dependency records, source hashes, extracted call-chain ranges,
execution log and a manifest with command, commit, environment and wall time.
