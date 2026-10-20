# KDM unresolved-gradient follow-up

Pretraining and trained-checkpoint reconstructions completed on 13 October
2026. The angle-adjoint investigation and causal intervention are complete. This note does not
establish completed replication.

## Completed pretraining result

All 68 reconstructed unit01 FP32 updates exactly reproduce the earlier stored
inputs, losses, comparison summaries and flag lists (340 explicit checks).
At index 67, the evolving gradient difference is 2.609752119e-5, exceeding
the arithmetic localization tolerance. Restoring the complete author pre-state
reduces it to 1.117587090e-8. The maximum updated-parameter difference is
7.275957614e-12. All captured same-state comparisons pass. The earlier feature
flag at index 59 also clears, with maximum gradient difference 9.196810424e-9.

Thus the index-67 flag is attributable to different accumulated states on this
tested trajectory. No incorrect local gradient operation is demonstrated.
This says nothing by itself about final trained-quality noninferiority.

Evidence:
[pretraining result](artifacts/controlled-smc-kdm-replication-20260928/gradient-followup-20261013/pretraining01/diagnostic/result.json).
Complete source pre-state, inputs and evolving/restored TF tensors are retained
in unit01-float32-gradient67.json.gz in the same directory.

## Trained replay

First attempt failed while reporting the location of a minimum: TensorFlow
UnravelIndex received int64 indices and int32 dimensions. The source trajectory
control and numerical capture equality had passed before that call.
The repair requests int32 argmin output; a CPU-only focused regression passed.
The failed attempt is preserved and charged (150.343406 seconds per resource).
The second planned attempt completed, as recorded below.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Close the selected pretraining gradient flag | Same-source-state agreement passes | No local numerical veto | Other updates and trained quality are separate | Complete trained-case replay | Whole-training equivalence |
| KDM replication remains open | Final quality not established | Unresolved qualification evidence | Few independent units and adverse checkpoints | Localization then native-size selection study | KDM is no worse |

| Inference status | Finding |
|---|---|
| Hard veto screen | Pretraining finite, exact-repeat and input checks pass; trained reporting failure repaired |
| Statistically supported ranking | None |
| Descriptive-only differences | Local numerical differences and previous checkpoint quality gaps |
| Default readiness | Unchanged |
| Next evidence needed | Trained same-state replay, native-size validation selection and independent-unit precision design |

Pretraining charge: 129.394598 seconds per resource. Trained first failure:
150.343406 seconds per resource. All completed attempts are charged in their manifests and the campaign ledger.

## Completed trained replay

Trained02 passed 70/70 adapter controls, reproduced the original two-update
starting/best comparisons exactly, and verified capture-on/off numerical
identity. All traced functions have one trace. GPU allocator peak was
279209472 bytes. This supplies no evidence of uncontrolled retracing or an XLA
execution failure in this replay.

The best-checkpoint update-1 gradient discrepancy persists after full source
state restoration: maximum absolute 1.580119133e-4, scaled 4.999332228. The
loss difference is 4.768371582e-7, parameter difference 1.490116119e-8, and
there are zero actual ancestor disagreements. Thus different accumulated
optimizer state does NOT explain this trained case. The ordinary-start control
passes with gradient difference 2.384185791e-7. Minimum raw angle radii are
0.00351618789 (best) and 0.850253582 (starting); conditioning is a hypothesis,
not yet a causal verdict. This triggers the planned adjoint stage.

Trained02 charge: 265.251404 seconds each. Cumulative follow-up charge:
544.989408 seconds each, including the reporting failure. Source snapshots,
complete case arrays and exact commands are in the corresponding versioned
run directories. Both trained replay slots are exhausted.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Keep trained gradient case open | Same-state FP32 arithmetic flag persists | No invalidity or ancestry veto; diagnostic failure triggers localization | Forward conditioning versus backward-operation mismatch | Actual source adjoints and matched local TF/analytic derivatives | An algorithm bug or whole-paper failure |

The strongest alternative explanation is differing forward arithmetic amplified
by the normalization/angle geometry. Its weakest evidence is that a small
radius is only associated with the discrepancy. Exact-input local derivatives
and a signed perturbation panel can distinguish it from a backward formula
mismatch; neither can qualify the final trained filter alone.


## Angle-adjoint result

Angle01 exactly reproduces the saved source and TF native losses, parameter
gradients, proposals and samples. Each TF function traces once. The ordinary
minimum radius is 0.850254 and the trained minimum is 0.00351619. The source
and TF normalization pullbacks agree with matched-input local TF derivatives
within 1.49e-8; FP64 automatic and analytic derivatives agree within 5.56e-17.
The saved atan2 derivative also agrees with its analytic expression within
3.73e-9 in FP32 and 6.94e-18 in FP64 across both controls.

At the trained minimum, the TF raw first angle coordinate differs by
-7.152557373e-7. Its normalization-input adjoint differs by 3.266893327e-5,
whereas the incoming normalized-coordinate adjoint differs by 1.119769877e-7.
This is evidence of amplification by the small radius. The author's own
conditional full backward replay changes its parameter gradient by about
2.5e-6 under a representable 1e-8 perturbation and 2.58e-4 under 1e-6.
The ordinary control changes by at most 2.98e-8 at 1e-6.
These are internal coordinate perturbations on a saved detached draw tape;
they are not an RNG perturbation envelope or an acceptable final-error margin.

The angle formulas checked here are correct. Their matching does not yet
attribute the entire full-gradient discrepancy: the last diagnostic tests
the observed input difference directly. No ridge, clipping or damping is
added to the source replica. Peak allocator use is 140509184 bytes.
Angle01 charge is 73.789083 seconds per resource; cumulative follow-up charge
including the reporting failure is 618.778491 seconds each.
Evidence: gradient-followup-20261013/angle01/diagnostic/result.json and the
complete author, TF and local-derivative arrays beside it.


## Terminal causal result

Angle02 exactly reproduces all five saved TF controls in both checkpoint
regimes, and all four source controls plus a zero-perturbation repeat.
For the trained checkpoint, matching just the selected raw angle pair reduces
the gradient-vector discrepancy by 84.09%, but leaves a maximum difference
3.428012133e-5 (scaled 1.5382), so that single point is a partial explanation.
Matching all raw angle pairs reduces the discrepancy by 98.52%; the maximum
difference is 1.892447472e-6 (scaled 0.08953), below the localization tolerance.
Every intervention retains exactly the source ancestor choices. The ordinary
control remains below tolerance.

Conversely, injecting the actual -7.152557373e-7 raw coordinate difference into
the source changes its full parameter gradient by 1.846551895e-4. Its residual
difference from native TF is reduced by 80.11% in vector norm. Together with
the exact-input local derivative agreement, this identifies continuous angle
input differences and small-radius amplification as the cause of the selected
gradient flag. A single small-radius point is not the whole explanation.
The artificial hybrids are explanatory interventions, not proposed algorithms.
No numerical floor, damping, changed optimizer or source repair was introduced.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Close selected pretraining gradient localization | Restored source-state agreement passes | Exact input/repeat checks pass | Other training paths | Preserve regression evidence | Whole-training equivalence |
| Close selected trained gradient localization | All-angle matching clears arithmetic flag; single-pair and reverse-source tests explain most of it | Exact controls and unchanged ancestry pass | Frequency and final-quality effect | Native-size validation selection study | Every gradient difference is harmless |
| Keep KDM trained-quality open | 0.01 nat/1% noninferiority not established | Adverse development checkpoints remain unexplained downstream | Training and selection variability; few independent units | Cost and precision design before full qualification | Complete KDM replication |

| Inference status | Terminal finding |
|---|---|
| Hard veto screen | Exact replay, source/control identity, finite artifacts, memory growth and one-trace XLA checks pass; one reporting failure was repaired and retained |
| Statistically supported ranking | None |
| Descriptive-only differences | Saved-case gradient reductions, signed perturbations and earlier trained-quality gaps |
| Default readiness | Unchanged; no artificial intervention is an eligible implementation |
| Next evidence needed | Native-size checkpoint selection and independent-training-unit uncertainty sufficient for the accepted quality criteria |

Post-run skeptical review: forcing all angles could conceal an upstream
network problem and is not a full-program correctness proof. The actual
observed coordinate perturbation in the source, the single-point intervention,
previous matched-operation evidence and analytic Jacobian checks support the
specific sensitivity explanation. A matched-input network/gradient discrepancy
would overturn the broader no-defect inference and trigger a source-grounded
repair. The weakest remaining evidence is the link to final learned quality.

Angle02 charged 80.862400 seconds to each resource. All five launches,
including the reporting failure, charged 699.640891 seconds each (11.66
minutes). The terminal audit checks source snapshots, ledger entries, exact
controls, trace counts and phase bounds. Both angle slots and the follow-up
phase are closed. Further work uses a distinct bounded plan and fresh output.
