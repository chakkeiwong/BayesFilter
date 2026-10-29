# KDM checkpoint wiring result — 6 October 2026

All 22 active bearings evaluation tasks in the checked source load the
NLL-trained checkpoint. Their names end in “mse”, but none loads the separate
MSE-finetuned checkpoint. This is now an executable call-chain finding, not
an inference from filenames. It does not establish an author bug: the intended
meaning of that task suffix and the exact published checkpoint remain unresolved.

[Plan](controlled-smc-kdm-checkpoint-wiring-20261006.md).
Artifact: artifacts/controlled-smc-kdm-replication-20260928/kdm-checkpoint-wiring-20261006-01/.
The command, elapsed time, source hashes, task records and manifest are preserved.
Source commit: b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.

## What the executable check establishes

The diagnostic verified every inspected source file against the pinned Git blob.
It executed the unchanged author task generator (eleven trials), common-parameter
merger, src/main.py::run_experiment and
src/models/kde_particle_filter/kde_particle_filter.py::load_pretrained.
The model state, torch.load file I/O and final evaluation were recording stubs.
It inspected 110 generated tasks and followed all 22 active evaluations to the
actual torch.load call at kde_particle_filter.py:283. Complete callable source
ranges are in source.json. The active path is:

    full_dpf_evaluation_fixed_bands_mse_NNN
      → main.py::run_experiment (checkpoint substitutions)
      → KDEParticleFilter.load_pretrained
      → torch.load(.../run_NNN/full_dpf_fixed_bands/models/full_dpf_model_best.pt)

Both experiment0002_importance (MDPF) and experiment0003_importance_init
(A-MDPF) follow this path. Both evaluation metrics are NLL and RMSE.
The released MSE training task writes full_dpf_fixed_bands_mse instead.
None of the 22 active checkpoint paths exists in this local checkout.

The prerequisite sequence remains component pretraining → MDPF NLL training →
A-MDPF NLL training, with the source's separate optional MSE training tasks
recorded in tasks.json. The released active evaluations do not consume those
MSE outputs. We must retain the released checkpoint choice for a source-code
baseline. A different checkpoint may be a separately identified experiment;
it cannot silently replace that baseline.

This is CPU metadata execution with CUDA intentionally hidden, no framework
import, no tensor loading, no training and no numerical equivalence claim.
The numerical state conversion and weight compatibility are explicitly not checked.
The single execution completed successfully in 0.566610 CPU wall seconds,
including startup; zero GPU seconds. All prior evidence was preserved.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
|Checkpoint path established|22/22 consumer calls match resolved source configuration|No drift, unresolved placeholder or wrong trial index|Task-name intent and paper checkpoint identity|Freeze actual loaded NLL stage in source-baseline specification|Author bug or published MSE-result replication|
|Training prerequisite incomplete|0/22 active checkpoint paths locally present|Evaluation cannot run with these paths yet|Costs and trained-state provenance|Cost component and method training before a funded campaign|Training completion or filter quality|
|Both replications incomplete|No P1/K2 evidence added|Existing numerical and budget gaps remain|TF compatibility and downstream non-inferiority|Follow master closure sequence|No-worse, default or HMC qualification|

| Inference status | Finding |
|---|---|
|Hard veto screen|Metadata/source checks pass; missing checkpoints block actual evaluation|
|Statistically supported ranking|None; deterministic metadata only|
|Descriptive-only differences|None between numerical methods was measured|
|Default readiness|Not assessed|
|Next evidence|Actual training/checkpoint compatibility and matched downstream evaluation|

Post-run review: the strongest alternative explanation for the task suffix is
that it labels a reporting workflow rather than MSE training. The loader-path
finding is unaffected, but we cannot label this an author error. Executing
recorded stubs proves path selection only; a real checkpoint may still fail to
load or evaluate. The missing local checkpoints do not establish that the authors
never distributed trained models elsewhere.
