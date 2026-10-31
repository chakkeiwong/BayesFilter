# SIR NeuTra machinery audit and proposed integration

Date: 2026-10-20. Checkout: ledh-graceful-failure-execution-20260917.
HEAD inspected: 10884528ca8b1a7463895256d30b9655d69c0955, with existing dirty
worktree changes preserved. This note answers whether the current SIR
surrogate-HMC campaign can train and use a nonlinear transport. No new GPU
training or sampling campaign has been launched or budgeted by this audit.

## Finding

The reusable machinery exists, but the current five-model master does not
connect it to SIR. SIR has three inferred parameters, an 18-dimensional latent
state, and T=50. NeuTra would transform the three parameter coordinates, not
the entire latent trajectory. The failed affine warm-up motivates nonlinear
preconditioning; it does not establish that geometry is the only failure cause.

Checked call chains:

- `bayesfilter/testing/surrogate_hmc_five_model.py::bindings` calls
  `covariance_proposal_score_batch_tf.make_batched_score_filter` with a genuine
  leading parameter batch. The recurrence returns value, all parameter scores,
  row validity and completed horizon. The campaign currently consumes scores
  as proposal forces and evaluates a separate endpoint likelihood.
- `NeuTraTransportTrainer.train_step` calls `reverse_kl_evaluate`, which invokes
  the target callback on stopped mapped points. The callback supplies matching
  log densities, first scores and row validity. TensorFlow differentiates the
  map, not the filter; no filter Hessian is requested. Batch size must exceed
  one, with a fixed input signature and XLA enabled by default.
- The declared canonical profile is
  `NeuTraTransportConfig.hoffman_author_iaf`; direct generic IAF construction
  retains legacy mask defaults and must not substitute for this profile.
  FP32 map/optimizer and independently configured target precision are supported.
- `NeuTraTransport` supplies batched forward maps, log determinants, score
  pullbacks and log-determinant gradients; frozen export and optimizer resume
  are implemented.
- The current `bind_neural_force_hmc_tuning_runner` explicitly requires raw
  coordinates. Existing generic fixed-transport adapters implement the Jacobian
  calculus, but this is not an executed SIR endpoint-corrected NeuTra call chain.
  A dedicated binding and integration tests are still needed.

## Training and sampling targets

Write q for the current three unconstrained coordinates and theta(q) for the
existing parameter chart. Keep the current scientific prior in q, including its
existing chart Jacobians. For a frozen LEDH design define

    ell_L(q) = log p_q(q) + log L_LEDH(theta(q)).

Here `L_LEDH` denotes the finite LEDH likelihood program, not the independent
endpoint reference. The callback must return this program's value and total
first score together. For z~N(0,I_3) and q=T_phi(z), train by minimizing

    E[-ell_L(T_phi(z)) - log|det D T_phi(z)|].

The omitted Gaussian-base term is constant in phi. Analytic filter scores enter
the map's parameter gradient through the chain rule. Combining the independent
endpoint value with the LEDH score would not be the gradient of this objective.
Derivative consistency at the actual SIR scope remains to be checked.

Freeze the trained map before HMC. To preserve the campaign target, use

    U_z(z) = U_endpoint(T(z)) - log|det D T(z)|,
    g_z(z) = D T(z)^T g_LEDH(T(z)) - grad_z log|det D T(z)|.

The first expression supplies the MH endpoint energy; the second supplies the
proposal potential-gradient field. All endpoint and force statuses must survive
the transformation. Preserve the current invalid-force fallback semantics.
The original endpoint for SIR is a fixed-design average of four bootstrap
particle-filter likelihood estimates, each with N=4032. It defines an
approximate posterior, not an exact SIR posterior. NeuTra changes coordinates
and does not repair endpoint approximation error or roughness from resampling.

## Proposed SIR repair phase

This is an additional master-program phase, not an already queued stage.

1. Implement the matched LEDH value/score/status callback using the existing
   native batched filter, current data, scientific prior, N=4032, frozen tuned
   controls and recorded SIR arithmetic policy. Check prior/chart terms and
   value-score directional consistency before training.
2. Check batch permutation/duplication parity, stable graph construction and
   GPU/XLA time/memory at small native batch sizes, initially 2 then 4 if feasible.
   These are capacity probes, not selected training defaults. Do not wrap a
   scalar complete filter in a row loop. Keep FP32 map weights and TF32 off;
   preserve SIR's documented FP64-recursion/FP32-pair exception.
3. Specify a bounded, target-specific training protocol: canonical IAF capacity
   choices, optimizer/learning-rate search, scale-cap rationale, initialization,
   multiple seeds, disjoint selection and assessment draws, checkpoint cadence,
   and a measured update/time ladder. Saved nonconverged warm-up states may
   nominate broad initial regions, but are not posterior training truth. Do not
   transfer the old T=20 legacy map or q20 training settings as defaults.
4. Freeze each nominated map and connect transformed LEDH forces plus the
   unchanged independent endpoint potential to the shared sequential sampling
   controller. Test Jacobian signs, transformed force, endpoint identity and
   reversible proposal mechanics before a long run.
5. Compare with affine SIR under the same endpoint target and a predeclared total
   cost budget, including training. Retune HMC toward acceptance 0.70; retain
   adaptive warm-up and cumulative sampling, no separate validation sampling
   phase, and finite energy errors as warnings. Preserve the established sample
   caps and criteria unless the owner changes them.

Primary question: does nonlinear preconditioning enable the current SIR
surrogate-HMC target to meet its sampling criteria within the budget? Training
loss and Gaussian-score residuals are explanatory/nomination diagnostics, not
convergence evidence. Compare independent starts and original parameter
diagnostics, not only latent coordinates. Reverse-KL mode omission, force
error, and endpoint roughness remain competing explanations for failure.

Skeptical audit: the generic trainer and derivative APIs do not prove this SIR
integration works. The crucial missing executable evidence is the complete
SIR -> training -> frozen map -> endpoint-corrected HMC chain. No target change
is proposed; the different training density is explicit. A full experiment
plan with measured cost, attempt budget and stopping rules is required before
launching the research campaign.

## Checks performed

CPU-only engineering check, with GPU deliberately hidden:

```sh
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q -p no:cacheprovider tests/test_neutra_single_authority.py -k 'hoffman_profile_masks or author_free_bias or exact_gaussian_map or xla_adam_resume or frozen_loader_roundtrip'
```

Result: 8 passed, 19 deselected, 2 deprecation warnings, 9.94 seconds. Log:
`docs/plans/surrogate-hmc-sir-neutra-machinery-tests-2026-10-20.log`. These check architecture masks,
the free scale bias, first-score-only training, XLA/Adam resume and frozen-map
round trips. They do not establish SIR GPU throughput or posterior performance.

The original NeuTra paper was downloaded to
`docs/papers/hoffman-2019-neutra-1903.03704-sir-audit-20261020.pdf`. Sections
2.2-2.3 give the variational objective and change-of-variables HMC method;
section 4.1.1 describes the IAF architecture and the paper's training settings.
Those settings are source evidence, not a SIR training prescription. The local
reference's archived author-source directory was absent in this checkout; a
fresh independent author-code architecture audit was not performed in this
initial inspection. The subsequent [training audit and integration plan](surrogate-hmc-sir-neutra-training-audit-plan-2026-10-20.md)
downloads pinned author sources, checks the core training gradient, identifies
the direct-versus-inverted stage convention, and supersedes this note's audit
coverage and next-action description. No SIR training has been performed by
either audit.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Machinery supports an integration attempt | Generic component checks pass; SIR integration not checked | No failure in the eight focused checks | Cost and effectiveness for the current SIR target | Implement the matched callback and nonlinear HMC binding, then measure a bounded batch probe | NeuTra fixes SIR, or the approximate endpoint equals the exact posterior |

Next action is implementation and a bounded integration/cost plan. No new
training allocation or automatic master continuation is implied by this note.
