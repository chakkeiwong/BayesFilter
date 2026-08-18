# GenUT Feasible Trust-Region Repair Result

Date: 2026-08-15

Plan: docs/plans/bayesfilter-genut-feasible-trust-region-repair-plan-2026-08-15.md

## Decision

Decision: keep the repaired route as an opt-in candidate; do not promote it as
the default or as an admitted NeuTra/HMC target yet.

The repair closes the demonstrated non-finite higher-moment value/JVP failure
for the current LGSSM T=10, N=1008 CPU replay; a prior pre-addendum RTX 5080
replay is retained as historical GPU/XLA evidence. It does not establish exact
moment matching, exact filtering, posterior correctness, score unbiasedness,
NeuTra convergence, HMC readiness, or broad model superiority.

## Selected Method

The final route is:

1. Retain the weighted empirical diagonal skewness/kurtosis as the declared
   least-squares target.
2. Record necessary Pearson and finite-particle moment diagnostics without
   clipping the target.
3. Solve each diagonal local moment problem with a column-scaled
   Levenberg--Marquardt system.
4. Apply a smooth row-RMS trust cap before rewhitening.
5. Differentiate every operation in the same scalar/batch value and JVP maps.

The route identifier is genut_column_scaled_lm_smooth_rms_trust_v1.

The route is a new finite composition. Ebeigbe et al.'s constrained GenUT
supports the principle that enforcing constraints can lose exact kurtosis
matching (local PDF, Sec. V, Algorithm 2). Easley--Berry's HOUT is a direct
higher-order competitor, but it constructs a variable-size signed-weight rule
from rank-one tensor decompositions and warns that its condition number can
grow as tolerance shrinks (local HOUT text, Sec. 4, Theorem 4.2, Remark 4.4,
Algorithm 4.1). Neither is a drop-in positive equal-weight differentiable
reset for the existing Contract E particle cloud.

Column scaling is directly grounded in Osborne's technical implementation
notes (local Osborne text, Sec. 3, steps (iii)--(viii), Notes (ii)--(iv));
Marquardt provides the classical nonlinear least-squares citation. Johnson--
Lowe is recorded for broader sample skewness/kurtosis bounds, but its closed
primary text was not technically inspected and is not used as proof support.

## Failure Repaired

The preserved prior failure was LGSSM T=10, N=1008, training step 169, with
source ESS about 1.0037, maximum weight about 0.9982, target maximum
skewness about 23.94, and target maximum kurtosis about 3005.62. The old
unscaled normal-equation solve became non-finite at diagonal iteration 3.
The controlled TF32-on/off replay showed the failure was structural, not a
TF32-specific defect.

The chapter now uses the directly proved necessary bound k <= N-1 for an
equal-weight standardized cloud. For N=1008, the bound is 1007; the preserved
target is therefore diagnostically infeasible. This is a necessary bound, not
a sufficient multivariate realizability theorem.

## Test Evidence

### CPU-hidden focused tests

Command:

    CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q tests/highdim/test_genut_shape_lm_tf.py tests/highdim/test_cubature_genut_batch.py tests/highdim/test_higher_moment_contract_e.py

Result: 36 passed.

Coverage includes finite scaled LM solve on the ill-scaled Jacobian, scaled
system condition bound, value/JVP parity for the LM solve, strict smooth
trust-radius bound and value/JVP parity, collapsed-weight scalar correction,
batch N=1008 finite value/JVP replay, batch/scalar value-score parity,
centered finite-difference score agreement, and existing higher-moment and
pairwise regressions.

### Target-adapter regression

Command:

    CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q tests/test_genut_neutra_targets.py tests/test_genut_three_model_adapters.py tests/highdim/test_cubature_genut_batch.py

Result: 13 passed.

### Prior RTX 5080 GPU/XLA replay

Command:

    TF_FORCE_GPU_ALLOW_GROWTH=true TF_DETERMINISTIC_OPS=1 CUDA_VISIBLE_DEVICES=GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_genut_feasible_trust_region_gpu_replay.py

Artifact:
docs/plans/artifacts/genut-feasible-trust-region-repair-20260815/gpu_replay_result.json

Result: `PASS_FINITE` for the pre-addendum checkout represented by the recorded
commit. It is retained as historical evidence and is not a current-code replay
after the public-schema edits below.

Final artifact provenance: git commit
18cfe60984252a9656d1d818c29a2fa86dbc8118; command
docs/benchmarks/run_genut_feasible_trust_region_gpu_replay.py; wall time
10.26 seconds; artifact SHA-256
650ab52af1d761d11c3664d6d2b0e343a2f727685f2555a8cf8b0e2c65ead913.

Recorded environment: TensorFlow 2.20.0-dev0+selfbuilt; RTX 5080 sm_120;
TF32 enabled; XLA compiled cluster observed; memory growth verified before
device initialization; N=1008; T=10; noise seed 140000; four batch rows all
program_valid=true; minimum covariance gap about 0.3031; maximum OT column
residual about 4.37e-7; maximum higher-moment kurtosis residual about 0.1033;
and maximum score component magnitude about 11.45.

The output artifact records input hashes, controls, device, dtype/backend,
memory policy, value, score, and diagnostics.

## Follow-up Evidence Addendum (2026-08-15)

The batch value and value/JVP routes now expose feasibility margins, maximum
scaled LM condition, and pre/post-cap RMS through their public diagnostics.
The NeuTra status adapter forwards these fields. The solver identity is
repository-owned and legacy control payloads remain signature-compatible.

Focused regression after this change:

    CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q tests/highdim/test_genut_shape_lm_tf.py tests/highdim/test_cubature_genut_batch.py tests/test_genut_neutra_targets.py tests/highdim/test_higher_moment_contract_e.py

Result: 41 passed.

Artifact: `docs/plans/artifacts/genut-feasible-trust-region-repair-20260815/cpu_screen_result.json`.

The diagnostic-only CPU-hidden screen used valid small counts: LGSSM `N=12`,
KSC-SV `N=12`, Austria-SIR `N=36`, and predator-prey `N=12`. LGSSM and
predator-prey were finite; KSC-SV and Austria-SIR failed closed at these small
counts. This is a low-particle scope diagnostic, not a model ranking or a
rejection of the production-count route.

Artifact: `docs/plans/artifacts/genut-feasible-trust-region-repair-20260815/lgssm_t10_n1008_cpu_result.json`.

The current repaired route at `N=1008,T=10`, noise seed `140000`, was finite
for two batch rows. Values were `-48.0904315710`; maximum score magnitude was
`8.2139396880`; minimum covariance-gap eigenvalue was `0.3365148604`; maximum
scaled-system condition was `1.5808112621`; maximum pre/post-cap RMS was
`0.0351738222/0.0350871086`; and the finite-particle upper margin was
`1003.7761840820`.

The preserved step-169 checkpoint is historical failure evidence, not an exact
current-route input: its target signature and source route differ from this
worktree. A fresh current-route checkpoint is required before that gate can
close. The RTX 5080 command was attempted twice but blocked before process
launch by command permission-review timeout; no current GPU result is inferred.

### Documentation build

Command:

    cd docs
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

Result: successful; docs/main.pdf produced. The repository still emits
pre-existing warnings including an unrelated multiply-defined label and
numerous box warnings.

## Review Status

### MathDevMCP

Doctor passed and confirmed the configured symbolic environment. Two bounded
substantive audit attempts timed out in the MathDevMCP permission-review layer.
They are therefore unavailable, not an agreement or proof certificate.

### Claude/Fable

No Fable-specific tool was available. Claude Opus max health probe passed, but
the substantive private-repository review was rejected by platform policy
because exporting the private document to the external Claude service was not
permitted. No workaround was attempted. The final derivation was reviewed by
Codex with explicit source-status and nonclaim boundaries.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Keep opt-in candidate | Preserved scalar/batch/GPU value and JVP finite | No hard numerical veto in tested scope | Broader model/horizon behavior | Fresh scope-specific calibration and support screen | Not a default |
| Preserve legacy route | Comparator remains available with zero repair controls | Historical route still fails preserved collapse | Whether a different established teacher is preferable | Compare against pairwise/dual-cap and HOUT where feasible | No superiority claim |
| Do not admit NeuTra/HMC | No training or posterior run was performed here | Admission gate intentionally not claimed | Downstream learned transport and sampler behavior | Separate reviewed NeuTra campaign | No convergence claim |

## Inference-Status Table

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for focused CPU mechanics and current `N=1008,T=10` CPU replay; current GPU replay unavailable |
| Statistically supported ranking | None; no multi-seed model comparison was run |
| Descriptive-only differences | Finite residuals, score magnitudes, and runtime diagnostics |
| Default readiness | Not established |
| Next evidence needed | Current-code GPU/XLA replay, exact current-route step-169 checkpoint, scope tuning, multiple seeds, broader horizons/models, and downstream NeuTra/HMC validation |

## Remaining Gaps

1. Add a repository-owned candidate/admission identity and scope-specific tuning
   artifact for the new solver controls.
2. Replay the exact stored training-step inputs from the old T=10 artifact,
   rather than the bounded centered/perturbed support screen used here.
3. Run fresh multi-seed value/score comparisons on LGSSM, KSC SV, Austria SIR,
   and predator-prey before any default decision.
4. Only after those gates, run a separate batch-native NeuTra training and
   sequential HMC campaign.

## Post-Run Red Team

The strongest alternative explanation is that the test points avoided the
original failure's exact downstream OT cloud, even though they used the same
model, N, horizon, and noise seed. The result would be overturned if the exact
stored training-step source cloud still produced a non-finite repaired
value/JVP, or if finite-difference parity failed on that replay. The weakest
current evidence is therefore exact step-169 reconstruction and broader
scope-specific statistical replication.
