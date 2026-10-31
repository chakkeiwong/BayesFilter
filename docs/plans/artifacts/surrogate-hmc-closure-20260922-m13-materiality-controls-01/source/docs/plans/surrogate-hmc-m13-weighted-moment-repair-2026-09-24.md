# M13 weighted-moment arithmetic repair — 2026-09-24

This is the next bounded repair in the surrogate-HMC master. The scientific
goal remains testing the canonical analytical DCTR proposal force with an
independent endpoint posterior. This stage changes neither target nor controls.

## Evidence contract and audit

Question: can FP32 compute finite weighted shape moments and their complete
analytical derivatives by applying weights before powers, removing the two
saved correction failures? Baselines: preserved controls-10 FP32/TF32 replay,
its exact-lift FP64 reference, and independent FP64 primitive formulas.

For a scalar coordinate z with weight w, write a0=w and a(k+1)=a(k)*z.
Then a(k)=w*z^k and da(k+1)=da(k)*z+a(k)*dz, with da0=dw. In exact arithmetic
this is the same polynomial and total directional derivative as
w*z^k and dw*z^k+k*w*z^(k-1)*dz. Summing over particles gives the same target
moments. Pair moments use a2_i*z_j and a2_i*z_j^2, with the product rule for
both factors. Weighting first avoids an unweighted z^4 intermediate; it does
not clip genuine overflow, omit weight derivatives, regularize covariance,
or change the intended finite algorithm. This is arithmetic regrouping rather
than a new moment target. Floating-point differences require measurement.

The shared higher_moment_contract_e helpers are the sole edit to the canonical
source closure. Every consumer still reaches the same general correction.
Preserved source closures bind the pre-change baseline; the replay permits
exactly this declared source change and saves both hashes. Any other source,
data, design or control difference vetoes the comparison. The realized FP32
inputs, seed1703/partition6 and nine nominal points remain exact.

| Role | Requirement |
| --- | --- |
| Engineering pass | Independent value/JVP parity for healthy and extreme representable moments; complete source binding and diagnostics |
| Candidate repair pass | Both previously failed rows become numerically valid; seven healthy rows remain valid |
| Non-harm screen | On formerly healthy rows, value/score/filter-moment normalized difference <=2e-4; diagnostic screen, not posterior materiality gate |
| Continuation veto | Unflagged nonfinite output, undeclared source/input drift, missing diagnostics or resource cap |
| Repair trigger | Remaining typed correction failure or failed non-harm screen |
| Explanatory | Precision differences, ESS, feasibility margins, speed |

The 2e-4 screen is an explicitly provisional arithmetic-regression tolerance,
chosen to allow accumulation roundoff while exposing consequential healthy-row
changes. Its failure requires inspecting the change, not adjusting the
scientific target or rejecting surrogate HMC. Saved-run differences remain
warnings for scientific materiality. Primitive FP64 parity uses scale-relative
floating point tolerances; genuinely unrepresentable results must stay flagged.

Cheap comparators are direct FP64 formulas, the preserved default, and the
preserved FP64 full consumer. Compare previously failed and healthy rows
separately. This is not a sampler efficiency or convergence experiment.

Default/assumption audit: FP32/TF32/XLA is the owner default; the source change
is a numerical hypothesis, not an assumed cure. Controls-07 settings are frozen
warm-start controls, not selected settings. The data/points are diagnostic
calibration data, never untouched claim data. No learned component, additional
prior bound, automatic precision fallback, or control retuning is introduced.

Pre-mortem: XLA may reassociate products; test the primitive on GPU/XLA as well
as the full consumer. Another operation may overflow after this one is fixed;
retain the typed result and inspect it rather than adding silent clipping.
Skeptical audit passes: the algebra and derivative are explicit, the comparator
preserves inputs, no likelihood/HMC correctness claim follows from the screen,
and each failure has a diagnostic interpretation.

## Budget and execution

Existing controls pool remaining197.444047 CPU seconds. One bounded repair
attempt: review80 CPU/90 wall seconds; GPU worker100 CPU/180 wall seconds,
total180 CPU seconds. This fits the pool and global budget. The master review
runs existing focused higher-moment value/JVP regressions plus new overflow and
wiring regressions with CUDA intentionally hidden. The worker uses verified
memory growth, trusted GPU access, FP32/TF32/XLA. No external review is needed.

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-moment-repair --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-controls-11
```

Preserve all commands, source snapshots, baseline hashes, primitive diagnostic,
consumer diagnostics, budget receipts and a result note under controls-11.
Successful arithmetic repair permits fresh scope-specific controls work; it
does not qualify the M13 posterior reference or endpoint and launches no HMC.
