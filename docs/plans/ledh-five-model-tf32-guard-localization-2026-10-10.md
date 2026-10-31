# TF32 rejection localization

The five-model precision experiment shows range-bearing TF32 rejection despite covariance-restoration error below the owner-approved 1% threshold. Locate the first-step rejection using the shared map, mixture-density and reset primitives, with exactly the saved first-case input hashes for each model at N1008. This is a debugging extension to the current two-hour campaign budget, capped at 600 seconds and run serially after the precision measurements.

The question is which validity flag rejects. Record map, density and reset validity, finite state/tangent/increment/score, covariance residual, and branch covariance symmetry and factorization validity. The suspected mechanism is TF32 roundoff in B Q B-transpose exceeding the existing strict symmetry check. Its measured flag determines the finding. Do not relax any factorization, rank, finite, mean or correction-cap threshold.

Use scripts/five_model_precision_guard_diagnostic.py with --campaign docs/plans/artifacts/ledh-five-model-tf32-20261010-01 and --output docs/plans/artifacts/ledh-five-model-tf32-guard-20261010-01. Exact command, matched input hashes, source hashes, memory policy and timing are saved in guard-diagnostic.json. Use the same trusted GPU, TensorFlow/XLA, FP32/TF32 mode and scope-specific controls as the original worker.

Audit: this reconstructs only the first step and first analytical direction with shared primitives. It cannot establish full-horizon validity or precision. XLA may optimize this small graph differently; any repair must still be tested through the complete consumer. Missing input identity or an exception invalidates this localization. A confirmed numerical rejection triggers a reported precision weakness; it does not invalidate the completed FP64/ordinary-FP32 evidence.

Frozen FP64 controls, datasets and first case come from the parent experiment. They are reproduction inputs, not tuning choices. No statistical ranking, new default, posterior claim or algorithm comparison is made by this probe; heuristic comparisons are therefore inapplicable.
