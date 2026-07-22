# BayesFilter LGSSM NeuTra Knowledge-Transfer And Serious-Validation Plan

Date: 2026-07-13  
Status: `MATERIAL_PHASE4_AMENDMENT_IN_REVIEW`  
Owner: Codex supervisor/executor  
Read-only reviewer: Claude, when available  
Artifact root: `docs/benchmarks/artifacts/lgssm_neutra_serious_validation_2026_07_13/`

2026-07-14 amendment: the original 1,000-step Phase 4 recipe was found to be
an unvalidated convenience default after the source program's long-budget
training results and the current default/assumption policy were inspected.
Phases 0-3 remain valid. The Phase 4 dense-candidate definition and downstream
artifact root are superseded by
`docs/plans/bayesfilter-lgssm-neutra-target-specific-training-protocol-amendment-2026-07-14.md`.
The interrupted original Phase 4 artifacts remain historical and must not be
overwritten or promoted.

## Outcome Sought

Transfer the mature plain NeuTra implementation pattern from `~/python` into
BayesFilter, bind it to the already validated 18-parameter deterministic
`T=120` LGSSM target, train with exact-target reverse KL on GPU/XLA, freeze the
transport, tune HMC in transformed coordinates with modern rank-normalized
diagnostics, and run independent serious chains.

This is an engineering port and target integration, not new NeuTra research.
Historical trained weights are not reusable because they are not bound to this
exact target. Architecture, optimizer, checkpoint, tuning, and validation
lessons are reusable.

## Research Intent Ledger

| Field | Predeclared value |
| --- | --- |
| Main question | Can a plain learned dense-autoregressive NeuTra transport for the exact 18D LGSSM target be frozen and used by independently tuned HMC that passes the same predeclared convergence and integrity gates as the completed tuned plain-HMC campaign, with posterior-agreement and single-fixture recovery screens also passing? |
| Mechanism under test | Three dense autoregressive IAF stages separated by fixed reverse permutations, followed by a full affine geometry map, trained with exact-target reverse KL. |
| Exact comparator | Immutable completed tuned plain-HMC campaign in `docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/`; serious max modern R-hat `1.006167`, min bulk ESS `5561`, min tail ESS `3104`, acceptance after tuning `0.7078`, all 18 recovery rows passed. |
| Control candidate | Frozen full-affine geometry transport with no learned nonlinear component. |
| Proposed candidate | Plain dense-IAF residual transport plus the same declared affine geometry map. “Plain” means reverse KL only, with no force, Hessian, replay, sample-NLL, trust-region, or energy-error enhancement. |
| Expected failure mode | Target/transport gradient mismatch, GPU/XLA compile failure, unstable reverse-KL optimization, frozen reload drift, or transformed HMC that cannot satisfy modern convergence diagnostics. |
| Primary promotion criterion | A frozen candidate passes independent serious transformed-HMC sampling with all-parameter modern R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS `>=400`, and all finiteness/provenance/integrity checks. These are the exact threshold definitions used by the completed plain-HMC campaign, not its achieved metric values. |
| Supporting promotion screens | Raw-coordinate posterior summaries agree with the immutable plain-HMC retained archive under a predeclared MCSE-aware comparison, and all 18 single-fixture recovery distances are `<=3` posterior SD. Recovery is a fixture-specific end-to-end screen, not a posterior-correctness or convergence quantity. |
| Promotion vetoes | Target identity/parity failure; score or log-Jacobian mismatch; nonfinite artifact/training/HMC state; CPU fallback during NeuTra training; XLA fallback; missing checkpoint/reload parity; modern tuning R-hat failure; serious R-hat/ESS failure; provenance mismatch; material posterior disagreement with the same-target plain-HMC archive; or any failed recovery row. |
| Continuation vetoes | Invalid exact target or score, irreparable frozen-transport math mismatch, corrupted/no-overwrite artifact boundary, trusted GPU unavailable after an escalated probe, or repeated identical implementation failure after one bounded repair. A candidate’s poor loss or HMC diagnostics alone is not a continuation veto. |
| Repair triggers | Nonfinite training at the initial learning rate triggers one lower-rate retry from the last valid checkpoint; frozen parity failure triggers implementation repair; transformed-HMC tuning failure triggers one wider/lower step-size repair; dense candidate failure leaves the affine control and future architecture work viable. |
| Explanatory diagnostics | Reverse-KL trajectory, gradient norm, clipping rate, target-status telemetry, acceptance, runtime, compile time, per-parameter ESS margins, posterior contraction, and ESS per gradient. |
| What must not be concluded | Training loss is not transport quality; a smoke or tuning pass is not convergence; one fixture is not calibration or robustness; descriptive runtime/ESS differences do not establish superiority; no production/default readiness or broad NeuTra claim is in scope. |

## Baseline And Fairness

1. The completed plain-HMC artifact is read-only. It will not be overwritten or
   rerun.
2. The affine control and learned candidate use the same exact fixture,
   parameter order, target adapter, and declared geometry source.
3. No retained plain-HMC posterior draws are used to train the primary NeuTra
   candidate. Training evaluates the exact target through the reviewed
   value/score custom-gradient bridge.
4. The existing geometry center equals the fixture truth because the fixture
   prior is truth-centered. This favorable initialization is disclosed and
   prevents robustness or generality claims.
5. HMC tuning seeds, serious-chain seeds, and training seeds are disjoint.
6. Training candidates are not ranked by loss. Downstream fixed-transport HMC
   is the promotion computation.
7. Continuous efficiency differences against plain HMC are descriptive for
   this single fixture. No “better”, “faster”, or “superior” claim is allowed
   without replicated uncertainty evidence.
8. The affine control is conditional on a favorable, truth-centered geometry
   and is not a neutral or generally available no-learning baseline. Any
   dense-versus-affine result is conditional on this near-oracle centering and
   cannot establish how much nonlinear transport is generally needed for
   LGSSMs or other targets.

## Mathematical Convention

Let `z ~ N(0, I_18)`. The trainable residual flow `F_phi` is applied first and
the fixed full-affine geometry map `A(y) = c + L y` is applied last:

```text
T_phi(z) = A(F_phi(z)) = c + L F_phi(z).
```

The affine center `c` and nonsingular factor `L` are fixed throughout training;
only the dense-IAF parameters `phi` are trainable. The exact reverse-KL
objective, up to a constant independent of `phi`, is

```text
E_z[-log pi_x(T_phi(z)) - log|det J_T_phi(z)|],  z ~ N(0, I_18).
```

After training, `T_phi` is frozen. HMC runs in the same `z` coordinates against
the transformed density

```text
log pi_z(z) = log pi_x(T_phi(z)) + log|det J_T_phi(z)|.
```

All forward-map, log-determinant, score-pullback, checkpoint, frozen reload,
tuning, and serious-sampling checks use this convention. No inverse-density,
trainable-affine, affine-first, or posterior-sample objective is interchangeable
with it.

## Knowledge-Transfer Anchors

| Reusable lesson | Source anchor | BayesFilter destination |
| --- | --- | --- |
| Exact reverse-KL objective and deterministic base draws | `/home/chakwong/python/src/dsge_hmc/estimation/_flow_training.py` | New focused NeuTra training module |
| Dense MADE-style autoregressive IAF | `/home/chakwong/python/src/dsge_hmc/estimation/_transports.py` | New trainable transport implementation compatible with the existing frozen schema |
| Three IAF stages, reverse permutations, affine-last composition | `/home/chakwong/python/scripts/run_neutra_paper_style_at_baseline.py` | 18D campaign config and builder |
| Checkpoint/resume, heartbeat, clipping, deterministic training state | `/home/chakwong/python/scripts/run_neutra_paper_style_at_baseline.py` | Focused no-overwrite training controller, not a copy of the 10,000-line runner |
| Dimension-100 local dense-IAF evidence | `/home/chakwong/python/docs/plans/neutra-gate1-two-track-closure-result-2026-05-05.md` | Architecture prior only, not evidence for this target |
| Reviewed exact target score bridge | `bayesfilter/inference/batched_value_score.py` | Exact reverse-KL training target |
| Frozen artifact schema/loader | `bayesfilter/inference/neutra_artifacts.py` | Frozen candidate and transformed-HMC binding |
| Corrected modern diagnostics | `bayesfilter/inference/hmc_convergence.py` | Tuning and serious sampling gates |

## Phase Plan

### Phase 0: Exact-Target Extraction And Identity

Objective: expose the validated 18D adapter and exact fixture binding as a
reusable BayesFilter module without changing target mathematics.

Entry conditions: completed plain-HMC fixture and target artifacts exist and
remain immutable.

Required artifacts:

- reusable adapter/bundle loader;
- exact target signature containing fixture/config/source identity;
- parity tests against the benchmark-driver adapter at fixed and random points;
- Phase 0 result record.

Checks: artifact-hash validation, parameter-order equality, value/score parity,
target-status parity, batch/scalar shape checks, CPU/XLA compile smoke.

Evidence contract: exact equality within declared float64 tolerances and stable
identity under reload. This phase cannot claim NeuTra readiness.

Handoff: proceed only when the reusable adapter is indistinguishable from the
completed campaign adapter on the checked target surface.

Stop: target, score, fixture identity, or parameter-order mismatch.

### Phase 1: Trainable Transport And Checkpoint Controller

Objective: port the minimal full-affine and dense-IAF composition plus a
deterministic GPU/XLA reverse-KL trainer.

Entry conditions: Phase 0 parity passes.

Required artifacts:

- TensorFlow trainable dense-IAF components with fixed permutations and
  affine-last composition;
- manual Adam state, learning-rate schedule, global-norm clipping, stateless
  per-step base draws, checkpoint/resume, heartbeat, and no-overwrite behavior;
- frozen dense-IAF payload emitter using
  `bayesfilter.neutra.dense_iaf_frozen_transport.v1`;
- focused unit tests and Phase 1 result record.

Checks: historical-implementation forward/logdet parity for matched weights,
finite-difference gradient checks, uninterrupted-versus-resumed equality,
config/target mismatch rejection, and freeze/reload forward/logdet equality.

Evidence contract: establishes engineering equivalence and deterministic
restart only. Loss reduction is explanatory.

Handoff: all local math, serialization, and resume checks pass.

Stop: irreparable objective/composition mismatch or nondeterministic resume.

### Phase 2: Frozen Score Bridge And Modern Tuning Gate

Objective: close two pre-runtime gaps found by the skeptical audit.

Entry conditions: Phase 1 trainable/frozen parity passes.

Required artifacts:

- explicit frozen dense-IAF score pullback and log-Jacobian-score methods,
  including composed, mixing, and affine components;
- fixed-transport tuner mode that requires
  `max(rank-normalized split R-hat, folded rank-normalized split R-hat)` on a
  fresh verifier with exactly four chains and at least 1,000 retained draws per
  chain before handoff;
- regressions proving folded R-hat can veto when ordinary rank R-hat passes;
- Phase 2 result record.

Checks: explicit score versus autodiff and finite differences, scalar/batch
parity, XLA compile, modern diagnostic schema and threshold checks, and legacy
R-hat absence from campaign promotion paths. Configuration must reject fewer
than four chains or fewer than 1,000 retained verifier draws. A deterministic
four-chain, 1,000-draw synthetic integration fixture must exercise the same
modern diagnostic classifier used by Phase 5; mocked summary-only diagnostics
cannot close this check.

Evidence contract: score correctness and modern tuning-gate correctness. It is
not posterior convergence evidence.

Handoff: a synthetic frozen dense-IAF can be bound and tuned without fallback,
and the tuner fails closed on folded-R-hat, fewer than four chains, or fewer
than 1,000 retained draws per chain.

Stop: explicit transformed score cannot match the defined transformed density.

### Phase 3: Trusted Exact-Target GPU/XLA Canaries

Objective: prove the exact 18D target and optimizer execute on the trusted GPU
with XLA before spending a serious training budget.

Entry conditions: Phases 0-2 pass; escalated `nvidia-smi` and TensorFlow device
probe see a GPU.

Required artifacts:

- one-step exact-target training canary;
- short training canary with checkpoint/resume and frozen reload;
- device/XLA/TF32/dtype/seed/target-signature manifest;
- Phase 3 result record.

Checks: all objective tensors and trainable state on GPU, `jit_compile=true`,
no CPU or non-JIT fallback, finite losses/gradients/parameters, checkpoint and
reload parity, target status valid at all evaluated draws.

Evidence contract: engineering GPU/XLA viability only. Short-run loss and
short-chain behavior cannot promote NeuTra.

Handoff: exact-target short training and reload pass.

Stop: trusted GPU unavailable, XLA compile invalid, target invalid, or repeated
nonfinite result after one lower-rate retry.

### Phase 4: Candidate Training And Freezing

Objective: materialize the affine control and at least two deterministic-seed
plain dense-IAF training candidates.

Entry conditions: Phase 3 passes.

Planned candidate definition:

- affine control: the declared full affine geometry map, no learned weights;
- dense candidates: 3 IAF stages, hidden layers `[18, 18]`, ELU, `s_max=1`,
  fixed reverse permutations, affine-last composition, 1,000 reverse-KL steps,
  batch size 256, Adam with clipping, seeds `(20260713, 1201)` and
  `(20260713, 1202)`.

Required artifacts: per-seed checkpoints, progress/heartbeat logs, final
training state, frozen payload, exact reload parity, and Phase 4 result record.

Checks: finite state, exact target identity, deterministic checkpoint lineage,
no overwrite, no training after freeze, and independent held-out base draws for
explanatory loss/force summaries.

Evidence contract: candidates are nominated by engineering validity only.
Neither final loss nor relative loss selects a winner.

Handoff: every viable frozen candidate enters downstream tuning. A failed
dense seed does not invalidate the other seed or affine control.

Stop: all candidates fail due to the same target/math/artifact invalidity. If
only optimization fails, write the failure and preserve a future repair path.

### Phase 5: Transformed-HMC Tuning And Candidate Admission

Objective: tune each viable frozen candidate in identity-mass transformed
coordinates and admit candidates using fresh modern diagnostics.

Entry conditions: frozen payload reload and transformed score checks pass.

Execution boundary: tuning and candidate-admission HMC run on the same
CPU-hidden multicore TensorFlow/XLA route planned for Phase 6: two persistent
workers, two chains per worker, `CUDA_VISIBLE_DEVICES=-1`, float64, and
`jit_compile=true`. Before tuning, fixed probe points must show GPU-trained
frozen-artifact reload parity between the trusted GPU mechanics path and the
CPU-hidden objective path within the declared float64 tolerance. Training
remains GPU-only; HMC sample generation and tuning remain CPU-hidden.

Required artifacts: candidate tuning tables, fresh verifier archives with
chain axes preserved, GPU-versus-CPU fixed-objective probe parity, runtime
identity, rank/folded R-hat summaries, fixed-kernel handoffs, and Phase 5 result
record.

Primary admission screen:

- four chains and at least 1,000 retained verifier draws per chain;
- combined modern R-hat `<=1.01` for every parameter;
- finite target/sample/log-accept values and no divergence/target-status veto;
- acceptance within the declared repair band; acceptance alone cannot pass;
- tuning, verification, and future serious seeds are distinct.

Repair: one deterministic lower/wider step-size grid expansion is allowed when
the run is finite but no candidate passes. No transport retraining may occur
after looking at serious-chain results.

Evidence contract: admission freezes an HMC kernel for serious sampling. It is
not the final convergence or recovery claim.

Handoff: every admitted candidate may enter Phase 6. If only the affine control
passes, record dense candidate rejection and continue to affine validation; do
not reject the research direction.

Stop: target/score/artifact invalidity, or no admitted candidate after the one
declared tuning repair.

### Phase 6: Independent Serious Sampling And Recovery

Objective: test admitted frozen transport/kernel pairs using serious,
independent CPU-hidden multicore XLA chains.

Entry conditions: fixed transport and kernel are immutable; serious seeds were
not used in training or tuning.

Required artifacts:

- run manifest with commit, environment, device policy, worker count, seeds,
  wall time, commands, and artifact hashes;
- four independent chains with 4,000 retained draws each;
- retained sample archive in both transformed and raw coordinates;
- independent modern diagnostics and all-parameter recovery table;
- comparator binding to
  `docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/phase7_campaign/private/retained_samples.npz`,
  expected file SHA-256
  `1b0c05d4ea2981b1be179040d3a52039f05efe6c5b163f9bf7bba64ce2068920`,
  with its original chain axes and all per-chain statistics needed to recompute
  posterior means, SDs, and mean MCSE;
- Phase 6 result record.

Pass gates: modern R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS `>=400`, all
finite, and no provenance or source-drift veto. These are the same threshold
definitions used by the completed plain-HMC campaign; the comparator's achieved
values (`1.006167`, `5561`, and `3104`) are descriptive reference values, not
new thresholds. No manual chain exclusion or thinning is allowed.

Supporting/veto screens:

- compare raw-coordinate NeuTra and immutable plain-HMC posterior means using
  the combined mean MCSE, `abs(mean_N - mean_H) / sqrt(MCSE_N^2 + MCSE_H^2)`;
- require every parameter's standardized mean difference to be `<=4`, with a
  reported maximum and full table; this conservative check is a material
  same-target disagreement veto, not a method-ranking test;
- require all 18 truth distances to be `<=3` posterior SD as the inherited
  single-fixture recovery screen;
- label both checks as supporting end-to-end evidence. Neither proves posterior
  correctness, calibration, or robustness.

Evidence contract: a primary-gate pass plus both supporting screens supports
only “tuned NeuTra-HMC works on this fixed 18D LGSSM fixture under this
campaign.” A failure rejects that candidate for this campaign and identifies
the next repair; it does not by itself refute NeuTra.

Post-admission rule: Phase 6 is confirmatory for each frozen candidate/kernel
pair. A Phase 6 convergence, integrity, recovery, or posterior-agreement
failure cannot trigger retuning, retraining, reseeding, chain exclusion,
thinning, or a second serious run for that pair inside this campaign. Read-only
recomputation may localize an artifact/reporting defect; a bounded repair may
rerun only the failed reporting computation when the retained archive and
kernel are unchanged. Otherwise reject the pair, continue with other pairs
that were admitted before any serious result was observed, and write a future
repair plan with fresh serious seeds.

Handoff: proceed to comparison when at least one candidate completes; otherwise
write a blocker result with the smallest justified repair.

Stop for the affected pair: corrupted archive, target mismatch, nonfinite
transition, XLA fallback, material posterior disagreement, failed recovery, or
the fixed serious budget ending without convergence. Continue only with other
independently pre-admitted pairs; do not manufacture a post-result repair arm.

### Phase 7: Comparison, Red Team, And Closeout

Objective: compare viable NeuTra candidates with the immutable tuned plain-HMC
baseline without unsupported ranking.

Required artifacts: final result note, decision table, inference-status table,
engineering/numerical/scientific ledgers, post-run red team, and reset memo.

Interpretation rules:

- hard vetoes and candidate viability are categorical;
- runtime, acceptance, ESS, ESS/gradient, and recovery margins are descriptive;
- no statistically supported ranking is claimed from one fixture;
- default readiness remains false;
- agreement with the tuned plain-HMC comparator does not establish that either
  sampler is correct relative to the analytically unavailable exact 18D
  posterior. The claim remains comparator-relative plus target-math,
  convergence, integrity, and fixture-recovery evidence;
- every closeout summary must state that the fixture, prior, and affine
  geometry are favorably truth-centered. A pass is conditional on this
  near-oracle centering and does not establish calibration, robustness, or
  generalization;
- the historical 2D `LGSSM_REFERENCE_HMC_READY` label is not reused as evidence
  for the 18D campaign.

## Skeptical Pre-Execution Audit

Audit date: 2026-07-13. Initial audit found and repaired the following plan
defects before runtime:

| Audit risk | Finding | Plan repair |
| --- | --- | --- |
| Wrong baseline | Earlier 2D fixed-affine result was untuned and had no R-hat/ESS. | Use the completed 18D tuned plain-HMC campaign as the sole scientific comparator. |
| Baseline threshold ambiguity | The baseline's achieved R-hat/ESS values could be confused with its predeclared gates. | Reuse the exact `1.01/1000/400` gate definitions; report achieved baseline values as descriptive references only. |
| Proxy promotion | Reverse-KL loss could be mistaken for transport quality. | Loss is explanatory; downstream serious HMC is the promotion computation. |
| Fixture recovery promoted as convergence | Truth distance on one truth-centered fixture is not a convergence or posterior-correctness quantity. | Keep recovery as an inherited supporting/veto screen and add a separate MCSE-aware same-target posterior-agreement screen. |
| Ambiguous map convention | “Affine-last” did not fully define the trained density or fixed/trainable parameters. | Define `T_phi=A o F_phi`, exact reverse KL, and transformed HMC density explicitly; forbid inverse, affine-first, and trainable-affine substitutions. |
| Missing frozen score path | Existing dense-IAF loader exposes forward/logdet but not the explicit score pullback required by fixed-transport HMC. | Phase 2 must implement and verify the exact transformed score before runtime. |
| Wrong tuning diagnostic | Existing fixed-transport tuner selects from acceptance-only fresh verification. | Phase 2 adds a serious mode requiring max rank/folded rank-normalized split R-hat with sufficient draws. |
| Unfair comparator coupling | Training on retained baseline posterior draws would make the primary comparison dependent on the comparator. | Primary training uses exact-target reverse KL only. |
| Hidden target identity | Historical artifacts lacked canonical target binding; the current benchmark adapter signature omits exact observation content. | Phase 0 creates an exact fixture-bound target signature and rejects mismatch. |
| Environment mismatch | NeuTra training is a GPU workload while serious sample generation is CPU multicore. | Separate trusted GPU/XLA training artifacts from explicit CPU-hidden multicore HMC artifacts. |
| Admission/serious runtime drift | A candidate could be tuned under a different device/JIT route than serious sampling. | Run Phase 5 HMC on the exact Phase 6 CPU-hidden multicore XLA route and require fixed GPU/CPU objective parity first. |
| Post-admission optional stopping | A failed serious pair could trigger retuning or reseeding after its result is known. | Freeze all admitted pairs first; reject a failing pair with no same-campaign retune/retrain/rerun, while allowing reporting-only repair on an unchanged archive. |
| Comparator artifact ambiguity | A posterior-agreement veto is not reproducible without the exact baseline archive and chain axes. | Bind the comparator path and SHA-256 and preserve chainwise statistics in both archives. |
| Misleading pass | Truth-centered prior/geometry can make recovery unusually favorable. | Disclose this limitation and forbid calibration/robustness claims. |
| Oracle-like affine control | A truth-centered affine control could be mistaken for a generally available neutral baseline. | Make every affine-versus-dense interpretation conditional on favorable near-oracle centering. |
| Candidate failure treated as campaign failure | A dense arm may fail for optimization or tuning reasons. | Candidate rejection triggers the declared repair/control path unless a continuation veto invalidates the harness. |
| Shared comparator error | NeuTra and plain HMC could agree while both are wrong relative to the unavailable exact 18D posterior. | Treat agreement as a same-target disagreement veto only; require target-math/score checks and explicitly forbid a posterior-correctness conclusion. |

Audit verdict: `PASS_AFTER_REVISION_FOR_IMPLEMENTATION_AND_GATED_RUNTIME`.
Runtime may begin only after local plan checks and bounded read-only review are
recorded. Claude disagreement on a material correctness issue must be repaired
or explicitly documented; Claude unavailability alone is not a scientific or
engineering stop condition.

## Review Record

Claude health probe returned `CLAUDE_PROBE_OK`. The initial whole-file prompt
stalled, so the bounded review skill split the plan into research/baseline,
execution, and closeout slices. Codex remained supervisor/executor.

| Review slice | Initial verdict | Material revisions | Final verdict |
| --- | --- | --- | --- |
| Research and baseline | `REVISE` | Defined `T_phi=A(F_phi)` and the exact density convention; separated inherited gates from achieved baseline values; classified fixture recovery as supporting; disclosed the near-oracle affine control. | `AGREE` |
| Execution phases | `REVISE` | Fixed verifier size; bound Phase 5 and 6 to the same CPU/XLA route; prohibited post-admission optional stopping; bound the comparator archive and SHA-256. | `AGREE` |
| Audit and closeout | `BLOCK_NEUTRA_CLOSEOUT` | Added the shared-comparator-error nonclaim and made truth-centered limitations mandatory in the final result. | `PASS_NEUTRA_CLOSEOUT` |

Local plan checks passed: every cited path exists, the comparator archive hash
matches, and `git diff --check` reports no plan whitespace errors.

## Pre-Mortem

| Misleading outcome | Cheap discriminator |
| --- | --- |
| Loss falls while the transport maps mass into invalid target regions. | Held-out target-status telemetry and frozen transformed-score checks. |
| Forward/logdet reload matches but HMC score is wrong. | Explicit pullback versus autodiff and finite differences before tuning. |
| Short tuning acceptance looks ideal because the step is too small. | Require long fresh modern rank/folded R-hat verification; acceptance alone cannot pass. |
| Dense candidate appears efficient because it uses a different target or seed leakage. | Exact target signature, disjoint seed ledger, and raw-coordinate recovery. |
| Candidate passes this favorable truth-centered fixture but is not robust. | Restrict the conclusion and require a later offset-truth multi-fixture campaign. |
| Both samplers share a target or diagnostic error. | Independent target/score parity, transformed-density derivation, fixture recovery, and an explicit nonclaim that comparator agreement is not exact-posterior validation. |
| GPU canary passes through silent CPU placement or non-JIT fallback. | Device placement, physical/logical GPU manifest, compiler evidence, and soft-placement rejection. |
| Resume changes the stochastic trajectory. | Bitwise uninterrupted-versus-resumed test using stateless per-step draws and serialized Adam state. |

## Planned Commands And Environment

Implementation/local checks use deliberate CPU hiding:

```text
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -m pytest -q <focused tests>
```

All GPU detection, compile, and training commands run with trusted/escalated
permissions in conda env `tf-gpu`, with XLA JIT enabled and no CPU fallback.
Serious HMC sample generation runs with `CUDA_VISIBLE_DEVICES=-1`, two CPU
workers, two chains per worker, and XLA JIT enabled.

Exact executable commands will be recorded in the phase/final result rather
than guessed here before the campaign driver exists.

## Result Record Requirements

The final result must include:

- command actually run and exit status;
- git commit and dirty-worktree disclosure;
- Python/TensorFlow/TFP/CUDA/GPU or CPU-hidden provenance;
- target, training, tuning, and serious seed ledger;
- wall time and artifact hashes;
- candidate-by-candidate hard veto and viability status;
- whether any ranking is statistically supported;
- descriptive-only differences;
- default-readiness status;
- next evidence needed;
- explicit statement that comparator agreement is not exact-posterior
  correctness evidence because no analytic 18D posterior reference is
  available;
- explicit statement that the result is conditional on truth-centered
  prior/geometry initialization and does not establish calibration,
  robustness, or generalization;
- strongest alternative explanation, result that would overturn the
  conclusion, and weakest evidence component.
