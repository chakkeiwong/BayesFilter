# Experiment plan: SGU residual-source, pruned-state, and projection diagnostics

## Question

Why did the SGU combined structural target fail Gate B, and is the right next
target a better perturbation-policy comparison or a different nonlinear
projection target?

The hypotheses from the previous result are:

```text
H1: the quadratic failure is dominated by second-order volatility corrections
    in Euler/static/control FOC equations, not state identities.
H2: a pruned (x_f,x_s) comparison may rescue the second-order residual
    comparison.
H3: a joint state-control projection may close full canonical residuals, but
    this is a different target.
H4: if the only robust pass is state support, BayesFilter should record SGU as
    state-support-complete, not residual-improved.
```

## Mechanism being tested

The current SGU state-identity helper closes `H[7], H[8], H[10], H[11]`.
The failed Gate B compared full canonical residuals on the same completed
total-state grid:

```text
linear policy residuals
quadratic policy residuals with G_xx and g_ss/h_ss corrections
```

This plan splits the failure into three diagnostics:

1. per-equation residual attribution for linear, quadratic without constant
   volatility corrections, and full quadratic policies;
2. pruned-state residual comparison using `(x_f,x_s)` rather than one total
   state;
3. a bounded joint state-control least-squares projection pilot.

## Scope

- Variant: SGU diagnostic-only evidence pass.
- Objective: classify the Gate B failure and decide whether a follow-on
  projection target is justified.
- Seed(s): deterministic grids; no random seed required.
- Training steps: none.
- HMC/MCMC settings: none.
- XLA/JIT mode: none; eager CPU diagnostics only.
- Expected runtime: targeted pytest suite under one minute.

## Success criteria

H1 is supported if:

```text
quad_no_const_rms <= linear_rms
full_quad_rms > linear_rms
```

and the largest full-quadratic residuals are in equations:

```text
H[3] Euler
H[4] marginal utility
H[5] labor FOC
H[6] capital FOC
```

H2 is supported only if a pruned `(x_f,x_s)` comparison satisfies:

```text
pruned_quadratic_rms < linear_rms
pruned_quadratic_max <= linear_max
```

on the same declared grid.  If it does not, preserve the blocker:

```text
blocked_sgu_pruned_state_comparison_not_residual_improving
```

H3 is supported as a new target only if a joint state-control projection
reduces the full residual to near roundoff while finite and local:

```text
projected_max_residual <= 1e-7
projected_rms_residual <= 1e-8
control_adjustment_norm <= 2e-2
deterministic_state_adjustment_norm <= 1e-2
```

If H3 passes, the allowed conclusion is only:

```text
sgu_joint_state_control_projection_pilot_passed
```

It must not mint `sgu_combined_structural_approximation_target_passed`.

## Diagnostics

Primary:

- max and RMS residuals for:
  - linear;
  - quadratic without `g_ss`;
  - full quadratic;
  - pruned raw transition;
  - pruned state-identity-completed transition;
  - joint projected transition.
- per-equation max residual table for SGU equations `H[0]` through `H[14]`.
- state-identity max residual for all completed variants.

Secondary:

- projected current-control adjustment norm and max;
- projected next-control adjustment norm and max;
- projected deterministic-state adjustment norm and max;
- least-squares success flag, iteration count, cost, and optimality.

Sanity checks:

- stochastic coordinates `(a,zeta,mu)` remain preserved for state-identity
  completions;
- exact nonlinear blocker remains present;
- no derivative, compiled, HMC, or BayesFilter backend claims are added;
- no artificial deterministic process noise is introduced.

## Expected failure modes

- `quad_no_const` is not better than linear.  Then H1 is false and the
  residual problem is not just volatility constants.
- Pruned comparison is worse than the total-state comparison.  Then H2 is
  rejected and a pruned rescue is not justified.
- Joint projection closes residuals only with large control/state moves.  Then
  H3 remains blocked as nonlocal.
- Joint projection closes residuals locally.  Then H3 supports a new projection
  target, not the old perturbation-policy target.

## What would change our mind

- A pruned comparison that beats linear in both RMS and max would justify a
  follow-on pruned SGU adapter target.
- A joint projection that closes residuals with local adjustments would justify
  a new projection-backend design plan.
- A projection that is nonlocal or singular would leave SGU at
  state-identity-only support.

## Command

```bash
cd /home/chakwong/python
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-sgu \
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_dsge_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

## Interpretation rule

- If H1 passes and H2 fails, SGU's Gate B failure is a real
  volatility-correction/static-FOC issue, not a state-support issue.
- If H3 passes, write a result note recommending a separate joint projection
  target and stop before BayesFilter/derivative/JIT/HMC promotion.
- If H3 fails, SGU remains state-identity-only and the next work is either
  economic derivation or abandon SGU residual promotion for now.
