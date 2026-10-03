# Actual-consumer fit observability completed

Twenty-four CPU tests pass. Four GPU FP64/XLA adaptive cases replay with zero
error in all original numerical fields: 14 recursive fit calls, 56 time-specific
fits, histories, coefficients, likelihoods and scores. Three cases still complete;
the same d5 case is rejected at the same fitting iteration. Every fit kernel has
one trace per configuration. The TensorFlow warning about multiple function
instances is not a repeat trace of one configuration.

Five explanatory columns are appended after the existing18: target-squared
effective count and maximum weight, initial/final log density energy, and initial
optimization loss. None enters fitting, convergence or acceptance. All16 final
fit targets match preserved independent R-checked calculations within
4.26e-14. Unit checks include constant-target ESS=N, dominant-target ESS=1,
amplitude invariance, scaled/native objective identities, and finite log energy
when native density energy underflows. The actual adaptive consumer writes the
same values returned by the recursive kernel; the call chain is checked.

| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | Limit |
|---|---|---|---|---|---|
| Keep appended reporting | Exact original replay, reference identities pass | None | Does not repair the fit | Test objective/design repair on fresh downstream controls | No filtering or paper promotion |

| Inference status | Evidence |
|---|---|
| Hard veto screen | No new failure; previous rejection preserved |
| Statistically supported ranking | Not applicable: deterministic replay |
| Descriptive differences | None in original outputs |
| Default readiness | Numerical defaults unchanged; reporting only |
| Next evidence needed | Fresh downstream likelihood/oracle and heuristic comparisons for a concrete repair |

The strongest alternative concern was changed XLA fusion due to extra outputs.
The exact replay resolves it for these four cases, not for all future workloads.
The fields report sampled concentration and energy; they do not certify global
fit quality or provide an empirical threshold for rejecting a guide. Prior
Eq15, shape-objective, floor, diagonal-family, initial-law and TF32 findings stay
open. No frozen R reference changed.

Plan: docs/plans/iapf-fit-observability-2026-09-22.md. Exact commands, source
snapshots, environments and timing are in the two launch manifests. GPU memory
growth was configured before initialization; CPU deliberately hid the GPU.
Verification: 330 checks. This phase used
55.859 CPU seconds and 31.424
GPU process seconds; 45.832715 CPU hours and
47.794664 GPU hours remain from the renewed48+48.
