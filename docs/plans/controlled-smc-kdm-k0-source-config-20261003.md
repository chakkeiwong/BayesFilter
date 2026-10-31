# K0: source settings and generated bearings configurations

Date:3 October2026. Parent: controlled-smc-kdm-replication-first-execution-20261003.md.
This is metadata/reference preparation, not a training campaign.

Current follow-up,6 October2026: [checkpoint consumer audit](controlled-smc-kdm-checkpoint-wiring-result-20261006.md)
resolves the22 active bearings evaluations to NLL checkpoints, including tasks
labelled MSE. None of those paths exists locally. The old32.2-hour balance below
is historical; the active GPU remainder is11.71minutes. No full training launch
is authorized by this metadata note.

## Question and source anchors

Which parts of the KDM replication are concretely specified by the checked
author repository, and which remain unknown? Pin source
b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18; retain exact file hashes and copies.
The local paper is .localresources/papers/younis-sudderth-2023-long-range-tracking.pdf.
AppendixB equations20–22 (text lines933–980) specifies linear Gaussian dynamics,
bimodal observations, final-time density NLL,1000 sequences of length5, batch64,
resampling bandwidth0.05 and posterior bandwidth0.5. Its actual coefficient
values, perturbation rule and training schedule have not been recovered.
The local toy_problem_dataset.py:225–245 instead implements nonlinear Cappe
dynamics and quadratic observations. It is wrong for an AppendixB replication.

A bounded search of .py/.yaml/.md/.txt files found no occurrence of
mixture-Kalman, bimodal, synthetic-learning or perturbed-parameter terms.
This is a reproducible search limitation, not proof the driver does not exist.
The prior detailed source screen remains at
artifacts/iapf-mdpf-whole-repair-20260928-01/mdpf-source-and-screen-spec.md.

The README:41–64 maps MDPF to experiment0002_importance and A-MDPF to
experiment0003_importance_init, and requires individual component training
before the bearings experiment and MDPF before A-MDPF. In
src/experiments/bearings_only_experiments_methodical_25_particles_multiple_runs:
- experiment0002_importance/config.yaml:15–31 uses ImportanceSampling gradients,
  coupled resampling weights/bandwidths and true-initial-state noise.
- experiment0003_importance_init/config.yaml:15–31 decouples weights/bandwidths.
- configs/common_parameters_training.yaml specifies25 particles, batch64,
 800 epochs maximum, early-stopping start200/patience25, Adam, clip100 and the
 source scheduler. These are source comparators, not new universal defaults.
- configs/our_experiments.yaml separately declares KDE_NLL training and MSE
 refinement with checkpoint prerequisites. A saved NLL checkpoint does not
 establish the MSE-stage or11-trial result.
- configs/our_experiments_exp3_init.yaml:22 requires a matching MDPF checkpoint.
- scripts/generate_multiple_runs_config_files.py:72–85 fixes11 trials and writes
 five generated YAMLs. The importing configs point to generated files that are
 absent from the checkout. Running the inspected official generator in a fresh
 snapshot resolves this preparation gap without changing the author source.

## Action and evidence contract

Verify selected source files against the pinned Git blobs, copy them with
relative paths, and call the unchanged generator for11 trials in the snapshot.
Check task counts, unique trial-suffixed names, trial indices0..10 and output
hashes. Preserve source do_run flags; no training, dataset generation, GPU
initialization or numerical experiment occurs. Metadata PASS means only these
checks passed. It cannot complete K0 source specification, K1 or K2.

Default audit:11 trials and all training settings come from author source.
Source initialization has privileged true initial state plus stated noise;
later comparisons must match that information. Unknown AppendixB values are
unknown, not silently filled with convenient defaults. Heuristics and training
uncertainty belong to later numerical plans; they have no role in determining
whether a YAML generator reproduced its source instructions.

Skeptical review found a wrong-model substitution risk, missing generated
imports and hidden training prerequisites. This action fixes the generated
imports in a versioned snapshot and exposes the other two without claiming
they are solved. Stop for source drift or incorrect generated task identities.
A missing training specification blocks that experiment, not independent C work.

## Command and budget

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase k0_source_config

One attempt, at most20 CPU wall seconds from the existing remaining allocation;
no GPU charge or additional compute grant. This consumes the fourth authorized
attempt in the restructuring tranche (C0, two C1 pilots, K0 preparation).
Output: fresh attempt142_k0_source_config, author-source snapshot, five generated
configs, summary.json and supervisor manifest. CUDA is deliberately hidden.
Record self-review and focused syntax/path checks; no independent review claimed.

Next: recover AppendixB specifics or explicitly scope a reconstruction; for
bearings audit the full source dependency/initial-checkpoint chain, select the
exact published metric/stages and cost independent training trials before
starting K2a. The previous8.589-hour single-training estimate is historical,
not a reliable full-campaign quote; eleven trials plus prerequisites/comparators
cannot be promised inside the current32.2 GPU-hour balance.
