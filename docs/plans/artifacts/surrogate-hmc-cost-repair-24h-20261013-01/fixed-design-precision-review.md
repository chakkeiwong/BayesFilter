# Corrected discrete-design precision result

The shared ancestor-index calculation now evaluates the stated inverse-CDF
floor in binary64. Saved FP32 uniforms near grid boundaries previously rounded
onto the next ancestor in FP32 multiplication, so the two precisions followed
different discrete designs. The correction changes those old FP32 trajectories;
it is not evidence of exact parity with that old finite program. The corrected
FP64 design remains the comparator. CPU boundary and actual draw checks passed.

All five runs used GPU ordinal0 (RTX4080SUPER), N4032/K2016, TF32 off,
XLA, analytical all-parameter scores, two parameter rows, T50 except range T20.
The saved controls are diagnostic baselines only. FP64 is an arithmetic reference,
not an exact model-likelihood oracle. The scale is max(1,abs(FP64 component)).

| Model | Decision | Maximum scaled value error | Maximum scaled score error |
|---|---|---:|---:|
| lgssm | PRECISION_PASS | 2.78472e-07 | 1.65325e-06 |
| ksc | PRECISION_PASS | 1.04723e-07 | 7.08279e-07 |
| predator_prey | PRECISION_PASS | 1.14906e-07 | 1.21124e-05 |
| sir_d18 | PRECISION_REPAIR_REQUIRED | 2.13425e-07 | 0.0450994 |
| range_bearing | PRECISION_PASS | 1.29758e-06 | 0.000261364 |

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Four arithmetic candidates pass | Final value and every score component within 1% | All rows healthy | Only two parameter positions, diagnostic controls | Fresh scope tuning and actual consumer checks | Model-score accuracy or posterior convergence |
| SIR FP32 needs repair | Maximum score gap 4.51% exceeds 1% | No invalid/nonfinite run | Residual reset/tangent arithmetic | Saved-input reset decomposition, then focused repair | Rejection of SIR or the surrogate-HMC research direction |

| Inference status | Evidence |
|---|---|
| Hard veto screen | No invalid/nonfinite result in these ten workers; SIR fails arithmetic promotion |
| Statistically supported ranking | None; this is deterministic same-input precision checking |
| Descriptive differences | Timings and magnitude of arithmetic errors at two positions |
| Default readiness | FP32 remains the user default; SIR is not cleared by this check |
| Next evidence | Localized repair, fresh per-scope tuning, actual consumer precision and measured cost |

The centered-residual moment rearrangement did not reduce the local SIR error
materially and was not added to runtime. Its CPU-only results are saved in
sir-centered-moment-cpu-diagnostic.json. The reset follow-up compares actual shared
weighted moments, transport, transported moments, recolouring, and restoration.
The code must preserve exact input checksums and compare staged with fused reset.

Sources, exact commands, environment, device/memory policy, seeds, wall times,
and original data paths are recorded in each worker result.json and source/.
Summary: fixed-design-full-precision-01/precision.json. Budget: budget.json.
Post-run red team: the largest remaining uncertainty is propagated sensitivity,
not likelihood agreement; a small likelihood error cannot clear the score gate.
The fixed-design correction also requires fresh tuning artifacts because the
source and old FP32 finite-design behavior changed.
