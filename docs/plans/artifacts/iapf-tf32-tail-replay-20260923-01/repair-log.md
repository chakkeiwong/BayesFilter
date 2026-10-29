# Fresh-compilation replay investigation

Initial57-case replays complete in all three modes. FP64 and FP32 each match
all57 saved values exactly; TF32 matches36/57. The21 mismatches are exactly
the selected d2 cases, with maximum log-likelihood difference0.40255928. All
traces are neutral and all ancestor/mixture labels reconstruct independently.

The scientific/runtime source closure matches phase18 exactly. Inspecting the
saved optimized HLO finds10 changed backend configurations for the d2 fitted
kernel: several matrix operations switch between cuBLAS and Triton. Other
compiler bookkeeping also changes; raw HLO hash inequality alone is not the
evidence. `compiler-difference.json` preserves the matched operation/backend
comparison. This supports a compilation-dependent execution difference; it
does not yet establish why the compiler selected those backends, nor a wrong
mathematical operation.

Localized control, within phase19's existing600-second/attempt and2400-second
total budget: repeat TF32 only, first executing the original six phase18 old
fixture warm-up calls in their d1,d2,d5/fitted,constant order. Use the same
runtime, compiler settings, inputs and57 selected seeds. Preserve the first
decision/localization/completed/summary files under `initial-compilation/`
and all original attempt manifests; a fresh attempt04 stores the control.
No claim that warm-up is a numerical fix. The test distinguishes the omitted
initial compilation context from other compiler-selection variability.
Exact old-value replay remains required for attribution to phase18's original
selected paths. No tolerance relaxation, new default or statistical inference.

Skeptical review: PASS as a bounded localization control. A positive result
supports sensitivity to compiler context but does not prove a unique causal
autotuning mechanism. A negative result leaves the d2 original-case attribution
unresolved; preserve that limitation rather than looping until a favorable
compilation appears. At most this one original-warm-up retry is predeclared.
