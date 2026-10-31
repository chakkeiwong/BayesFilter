# Bounded invalidity localization

Question: which recorded guard rejects full d3 T2 calibration seed 195002,
IID, flow 8, N1020? Baseline is the unchanged failed GPU/XLA evaluation in
run-01/full_d3_T2__iid_dual_cap/tuning.json. A CPU/non-XLA trace of the same
value program is a debugging exception, with GPUs intentionally hidden.
Budget: one 180-second CPU process, no additional GPU charge; stop on timeout,
exception or finite-value disagreement. This is not a final score evaluation.

Expose reset row-mass error, column total variation, higher-moment validity,
state/tangent finiteness and covariance positivity from the shared trace.
A violated recorded guard localizes a sufficient reason for rejection; it
does not prove that no other check failed. Settings and acceptance tolerances
remain unchanged. One analytical direction suffices for this value-failure
localization because every direction in the saved result returned -Inf.

Skeptical audit: the same data seed, particle count, controls and shared
implementation are used. CPU/non-XLA is explanatory only and cannot replace
GPU evidence. All trace tensors and hashes are preserved under diagnostic-01.
No numerical source or active run setting is edited.
