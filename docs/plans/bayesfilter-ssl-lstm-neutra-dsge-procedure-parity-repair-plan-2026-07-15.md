# SSL-LSTM NeuTra DSGE-Procedure Parity Repair Plan

Date: 2026-07-15

Status: `BOUNDED_EXECUTION_COMPLETE_MATERIAL_TRAINING_BUDGET_REQUIRED`

Closeout, 2026-07-15: R1-R4 passed after focused implementation repairs and
direct cross-repository parity tests. The authoritative final-source GPU/XLA
receipt is `timing-canary-r4.json`; earlier `r1`-`r3` timing receipts are
superseded because implementation or test source changed after they ran. R5 is
recorded in
`bayesfilter-ssl-lstm-neutra-dsge-procedure-parity-repair-result-2026-07-15.md`.
No material training, HMC, candidate promotion, or scientific comparison was
run under this repair plan.

## 1. Decision And Scope

Phase 4 did not test the established nonlinear NeuTra procedure used by the
Rotemberg and SGU second-order lanes in `dsge_hmc`. It tested a one-stage
`tanh` IAF with no coordinate mixing, no fixed reference translation, a fixed
`1e-3` learning rate, batch 64, 2,000 steps, and global gradient clipping.
That candidate remains useful negative ablation evidence, but its result is
reclassified as:

```text
ONE_STAGE_ABLATION_COMPLETE_NONLINEAR_PHASE_NOT_PASSED
```

The Phase 5 affine-control preflight is paused as the main program handoff.
Affine candidates remain classical controls, but cannot satisfy or replace the
nonlinear phase. This repair transfers the actual `dsge_hmc` procedure before
any new conclusion is drawn about learned NeuTra on the locked SSL-LSTM target.

This plan claims **DSGE-procedure parity**, not fidelity to every NeuTra paper,
and not proof that the procedure will work on SSL-LSTM. The local `dsge_hmc`
source at commit `d94566c9f70b3143e599a56eba7cb461ff2bda88` is the implementation
authority for this repair. No `dsge_hmc` file will be modified.

Gate-order audit: `dsge_hmc/AGENTS.md:96-102` requires its own Gate 1 paper
replication and Gate 2 same-test enhancement comparison before promoting
DSGE-like surrogate work in that repository. Its May 7 evidence reset records
Gate 1 closed and no default enhancement promoted. This repair is BayesFilter
work under BayesFilter's active July 13 academic-risk policy; it neither
modifies/promotes a `dsge_hmc` surrogate nor introduces a Gate 2 enhancement.
It transfers the plain, score-matching-off procedure into the user-directed
SSL-LSTM application lane. The user's explicit instruction to repair and
execute this lane is also the required gate-order override if the sibling rule
is construed to cover cross-repository application work. Paper-suite or
enhancement claims remain forbidden.

## 2. Research Intent Ledger And Evidence Contract

| Field | Prospective contract |
| --- | --- |
| Scientific/engineering question | Can BayesFilter reproduce the established Rotemberg/SGU trainable NeuTra topology and optimizer procedure on the locked four-coordinate SSL-LSTM target, then compile and time that exact path on trusted GPU/XLA? |
| Candidate mechanism | Three trainable dense autoregressive IAF stages, ELU hidden activations, reverse-coordinate mixing between stages, bounded log scale `s_max=1`, then a fixed identity-scale translation to the locked free-coordinate reference center; end-to-end reverse-KL training from standard-normal base draws. |
| Exact baseline/comparator | Source-level and numerical parity with the effective `dsge_hmc` Rotemberg/SGU procedure; the old BayesFilter one-stage `tanh` arm is a negative ablation, not the parity baseline. |
| Primary engineering pass | Source-to-spec trace is complete; parity and mutation tests pass; serialization/reload preserves the six-component order and numerical map; a trusted GPU/XLA canary compiles and executes the exact parity configuration. |
| Promotion criterion | None in this repair. Material transport promotion requires two complete prospectively independent 5,000-step runs under a separately recorded budget, followed by the existing support, exact transformed-target, HMC, replication, and predictive gates. |
| Hard/continuation vetoes | Wrong target chart or translation; inability to reproduce forward/logdet/gradient/update semantics; failed mutation test; serialization drift; nonfinite CPU parity result; GPU/XLA compile/device failure; or timing showing the approved future budget cannot complete both full seeds. |
| Repair triggers | A parity mismatch that localizes to masks, activation, component order, matrix orientation, logdet, score/VJP, schedule boundary, clipping, Adam semantics, or serialization triggers a focused implementation repair and rerun. |
| Explanatory only | CPU/GPU time, compile time, loss values, gradient norms, parameter norms, canary loss movement, and the historical Phase 4 candidate metrics. |
| What will not be concluded | Official-paper fidelity, transport quality, posterior correctness, HMC readiness, sampler convergence, predictive equivalence, superiority, default readiness, or failure/success of the NeuTra idea. |
| Result artifact | `docs/plans/bayesfilter-ssl-lstm-neutra-dsge-procedure-parity-repair-result-2026-07-15.md` and structured timing receipt under `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/procedure-parity-repair/`. |

Diagnostic roles are fixed prospectively: parity and mutation failures are
hard engineering vetoes; GPU/XLA failure is a continuation veto for material
training; timing is a resource veto only when two complete seeds cannot fit a
newly approved budget; loss and canary parameter movement are explanatory.

## 3. Source-To-Spec-To-Code Trace

Line anchors refer to the source commit above and must be refreshed if that
commit changes.

| Concern | `dsge_hmc` source authority | Frozen BayesFilter specification | Required BayesFilter proof |
| --- | --- | --- | --- |
| Dense IAF equations and masks | `src/dsge_hmc/estimation/_transports.py:1197-1355` | MADE masks; ELU hidden activation; `u_i=z_i exp(s_i)+t_i`; `s=tanh(s_raw)` because `s_max=1` | Explicit-tensor forward/logdet and gradient parity |
| Composition | `scripts/run_neutra_paper_style_at_baseline.py:5501-5526` | `IAF -> reverse mix -> IAF -> reverse mix -> IAF -> fixed affine` | Component-order and wrong-order mutations |
| Reverse mixing orientation | Same file, lines `5509-5522`; BayesFilter import tie-out in `docs/plans/bayesfilter-neutra-c603-followup-import-validation-result-2026-07-06.md:43-59` | Reversal matrix applied with the BayesFilter loader's row-vector convention; matrix must not be silently transposed except where the two APIs require it | Non-symmetric probe plus missing/transposed-mix mutations |
| Fixed output map | `scripts/prepare_neutra_rotemberg_second_order_svd_target.py:335-339`; `scripts/prepare_neutra_sgu_second_order_pruned_ukf_target.py:485-489` | Identity scale and fixed translation only; no learned affine, Hessian, MAP optimization, or sample-fit residual | Frozen affine component is last, non-trainable, scale all ones |
| SSL-LSTM chart and translation analogue | `bayesfilter/nonlinear/ssl_lstm_posterior_tf.py:105-111,223-224,441-447,507-520,666-679` | Train directly in the locked identity-oriented four-free-coordinate chart, in the exact order `latent_mean_weight.0.0`, `latent_mean_bias.0`, `observation_weight.0.0`, `observation_bias.0`; `PRIOR_CENTER_VALUES=(0.35,-0.08,0.65,0.05)` is the fixed translation and is already bound to the target manifest | Assert dimension, names, order, identity transform, target/adapter signatures, and translation in the trainer/frozen manifest; wrong-order, wrong-chart, and wrong-translation mutations |
| Objective and base distribution | `scripts/run_neutra_paper_style_at_baseline.py:6227-6241` | Standard-normal base; batch-mean `-log_pi(T(z))-logdet`; score matching off | Loss and every trainable-tensor gradient parity |
| Clipping and Adam | Same file, lines `6263-6283` | Replace nonfinite gradients by zero, then `tf.clip_by_norm` independently for every tensor; Adam defaults `beta1=.9`, `beta2=.999`, `epsilon=1e-7` | Tensor-specific clipping mutation and one-update parity with `tf.keras.optimizers.Adam` |
| Learning-rate schedule | `src/dsge_hmc/estimation/_flow_training.py:1071-1120` | `PiecewiseConstantDecay([999,3999],[.01,.001,.0001])` in zero-based optimizer-iteration semantics | Exact checks at iterations `0,998,999,1000,3998,3999,4000` |
| Rotemberg/SGU run settings | `scripts/launch_neutra_rotemberg_second_order_svd_fixed_grid_baseline.py:58-85,110-152`; `scripts/launch_neutra_sgu_second_order_pruned_ukf_fixed_grid_baseline.py:67-93,123-165` | 3 stages, init scale `.02`, `s_max=1`, 5,000 steps, batch 480, LR `.01`, paper schedule, clip 10, score matching off | Parity preset rejects incompatible overrides |
| Existing imported topology evidence | `docs/plans/bayesfilter-neutra-c603-followup-import-validation-result-2026-07-06.md:33-77` | Reuse existing frozen composed-artifact schema; do not invent a second inference representation | Trainable-to-frozen reload parity and inverse roundtrip |

The SSL-LSTM hidden widths are frozen to `(4,4)`. This is a dimension-relative
transfer of the actual launcher rule: Rotemberg explicitly supplies `(15,15)`
for its 15-dimensional training chart at
`launch_neutra_rotemberg_second_order_svd_fixed_grid_baseline.py:159-160`, and
SGU explicitly supplies `(7,7)` for its 7-dimensional chart at
`launch_neutra_sgu_second_order_pruned_ukf_fixed_grid_baseline.py:179-180`.
It also matches the generic `(dimension, dimension)` construction at
`run_neutra_paper_style_at_baseline.py:5507`.

The models do not share parameter meanings. Procedure parity means the same
transport/optimizer operations are applied in each model's declared training
chart. The SSL-LSTM chart adaptation is admissible only because its target
manifest declares a four-dimensional identity-oriented free chart with the
exact names and order above. Any constrained/unconstrained transform change,
coordinate reorder, or target/adapter signature drift is a hard veto.

The TensorFlow stateless initialization stream is a BayesFilter reproducibility
adaptation; parity tests assign identical explicit tensors before comparing
maps, objectives, gradients, and updates. Initial random numbers are therefore
not claimed bitwise identical across repositories.

## 4. Skeptical Pre-Execution Audit

Audit status: `PASS_AFTER_REPAIR_OF_PREVIOUS_BASELINE_ERROR`.

| Audit question | Finding and control |
| --- | --- |
| Wrong baseline | Material flaw found in the old program: a one-stage `tanh` arm was treated as the plain nonlinear method despite the established three-stage ELU/mixing procedure. This plan replaces that baseline prospectively and preserves the old run only as an ablation. |
| Proxy promoted | Training loss, support probes, timing, and a short canary cannot promote a transport. Only exact parity is a pass criterion here. |
| Missing stop | CPU parity/mutations must pass before GPU; GPU timing must precede a new material budget; both A/B seeds must be affordable before either is launched. |
| Unfair comparison | No material method ranking occurs. Future A/B use identical topology/procedure and independent seeds; affine controls cannot win the nonlinear phase. |
| Hidden assumptions | Translation, hidden widths, schedule indexing, clipping unit, Adam epsilon, matrix convention, and random-initialization adaptation are explicit above. |
| Stale context | Current BayesFilter commit is `3d353253dc93a102722e00cbca8803a1b3fce7fa`; the Phase 4 budget is closed and Phase 5 main handoff is paused. The sibling Gate 1/2 rule was audited above rather than silently imported or ignored. |
| Environment mismatch | CPU checks deliberately hide CUDA and are engineering references only. The canary and all serious training use TensorFlow/TFP, `float64`, trusted GPU, XLA JIT, and recorded TF32/device provenance. |
| Artifact answers question | The result must contain traceability, parity/mutation outcomes, component manifest, timing, command, environment, device, seeds, wall time, and the next budget calculation. |

Pre-mortem: a canary could show decreasing loss while the mix orientation or
translation is wrong; parity and mutations precede it. A forward comparison
could pass while the score or optimizer differs; all-tensor gradients and one
Adam update are compared. A tiny run could fit while full A/B cannot; measured
post-compile step time is extrapolated conservatively and no material run is
started without a complete-pair budget. A procedure-parity pass could still
mode-seek or miss posterior support; later candidate and sampler vetoes remain.

## 5. Execution Phases

### Phase R1: Status Correction And Specification Freeze

Update the master program, Phase 4 result, and Phase 5 plan so they consistently
record the one-stage ablation, paused nonlinear handoff, and non-substitutability
of affine controls. Preserve historical metrics and hashes unchanged.

Pass: no active document says the nonlinear phase passed or that affine alone
opens main transformed-HMC work. Stop on contradictory target or source anchors.

### Phase R2: Faithful Trainable Composition

Extend `bayesfilter/inference/neutra_training.py` with a named
`dsge_paper_dense_iaf` family/preset containing exactly three ELU IAF stages,
two fixed reverse-coordinate mixes, and the fixed prior-center translation.
Keep the existing affine and one-stage families as controls. Implement
zero-based paper schedule semantics, per-variable clipping, Keras-Adam-compatible
updates, exact state resume, and six-component serialization through the
existing frozen-artifact schema.

No new NumPy production path, learned output affine, Hessian whitening,
posterior-sample fitting, affine-residual loss, score matching, or hidden
fallback is allowed.

### Phase R3: Parity And Mutation Proof

Run CPU-hidden focused tests that cover:

1. locked target dimension, free-coordinate names/order, identity chart,
   target/adapter signatures, masks, ELU, three-stage order, reverse mixing,
   and fixed translation;
2. forward, logdet, inverse roundtrip, pullback score, and logdet score;
3. reverse-KL value and every trainable-tensor gradient from explicit equal
   tensors and equal base rows;
4. one Adam update and exact schedule boundary values;
5. per-variable clipping and exact state resume;
6. six-component serialization/reload parity; and
7. mutations for missing stage, missing/wrong mixing, `tanh`, global clipping,
   constant/wrong schedule, wrong translation, learned translation, and wrong
   component order.

The sibling checkout at `/home/ubuntu/python/dsge_hmc` and frozen commit is a
required input, not an optional convenience. Run a direct cross-repository
parity test by assigning identical explicit tensors and comparing the actual
`dsge_hmc` transport and training equations. If the checkout or commit is
unavailable, stale, or unimportable, R3 is blocked and the claim is not
downgraded silently to local-spec parity. Committed self-contained fixtures
still catch mutations, but they supplement rather than replace this direct
comparison.

Commands:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_neutra_reverse_kl_training.py \
  tests/test_neutra_dsge_procedure_parity.py
```

### Phase R4: Trusted GPU/XLA Timing Canary

After R3 passes, run one short exact-topology canary on the locked SSL-LSTM
target with batch 480. Include compile warmup and enough measured post-compile
steps to estimate step time without treating loss as evidence. Use a distinct
canary seed, write structured JSON, and record device/JIT/TF32 provenance and
the managed-session trust basis.

The canary cap is 10 trusted GPU-minutes. Stop immediately on target signature
drift, nonfinite value/score/update, CPU fallback, missing XLA evidence, or a
parity-config mismatch. It must not freeze or nominate the canary transport.

### Phase R5: Result And Material-Budget Gate

Write the result note with a decision table, inference-status table, manifest,
post-run red team, and conservative A/B cost estimate. A new material phase may
be proposed only if a separately recorded budget can finish both independent
5,000-step, batch-480 seeds plus validation and serialization. Sequential
stopping remains allowed for hard invalidity or a prospectively declared
candidate veto, but not merely to spend the remaining budget on a truncated
second seed.

## 6. Handoff And Stop Conditions

Successful bounded execution ends at:

```text
DSGE_PROCEDURE_PARITY_ENGINEERING_PASSED_MATERIAL_TRAINING_BUDGET_REQUIRED
```

It does not resume Phase 5. The next action is a newly budgeted two-seed
material training plan using the frozen parity preset. If parity or GPU/XLA
fails, write a localized blocker/repair result; do not run material training.

The research direction is rejected only by a later evidence-valid experiment,
not by the old one-stage ablation or by a fixable parity defect. At closeout,
state separately whether any failure invalidated the harness, implementation,
target, math, artifact, or only the tested candidate.
