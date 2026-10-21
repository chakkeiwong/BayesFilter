# Final-source arithmetic screen

The retained score implementation was checked at the full declared horizons, N4032, on two fixed parameter rows per model. Every row completed and was finite. The reference is the same finite filter in FP64, not a model-likelihood oracle.

| Model | Scaled likelihood error | Maximum scaled score error | Decision |
|---|---:|---:|---|
| lgssm | 1.10787e-05% | 0.00201787% | PRECISION_PASS |
| ksc | 0.000425271% | 0.0236761% | PRECISION_PASS |
| predator_prey | 5.70431e-05% | 0.131492% | PRECISION_PASS |
| sir_d18 | 8.1011e-05% | 48.6936% | PRECISION_REPAIR_REQUIRED |
| range_bearing | 8.19126e-05% | 0.282204% | PRECISION_PASS |

Scale is max(1,abs(reference component)); these are not relative errors when the reference magnitude is below one. SIR likelihood agreement does not imply score agreement. All-FP32 SIR is vetoed for promotion at these controls. This triggers the already-planned explicit mixed-recursion scope and fresh numerical tuning, not a stop of the campaign. Range passes this local FP32 check; mixed precision is still an optional diagnostic scope because its earlier off-centre consumer health was not established by this check. FP32 remains the default.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue four finite precision-screen passes | <=1% all final values/scores | None on these rows | Region and filter approximation | Fresh scope tuning then consumer checks | Posterior validity |
| Repair SIR FP32 score | Fails at 48.7% | Arithmetic promotion veto | Recursion amplification | Explicit mixed recursion and scope tuning | Failure of surrogate HMC idea |

| Inference status | Finding |
|---|---|
| Hard veto screen | SIR all-FP32 score fails the numerical screen |
| Statistically supported ranking | None; deterministic paired arithmetic check |
| Descriptive-only differences | Timings and finite design accuracy elsewhere |
| Default-readiness | FP32 retained; optional mixed scopes are explicit, not default promotion |
| Next evidence needed | Fresh scope numerical/health checks, consumer cost, later valid adaptive posterior study |

Red-team: the same finite filter can agree across dtypes while disagreeing with the model likelihood/score. Independent Kalman/KSC diagnostics remain separate. Two rows do not cover posterior trajectories. Source and attempt logs are preserved in final-precision-01.
