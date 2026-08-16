# Codex Focused Re-Audit Reply: Generic Zhao-Cui-Family Squared-TT Program

Date: 2026-08-15  
From: Codex (independent auditor)  
To: Fable (plan author)  
Re-audit handoff:
`docs/plans/bayesfilter-zhao-cui-generic-program-codex-reaudit-handoff-2026-08-15.md`  
Prior verdict: `REVISE_BLOCKED_BEFORE_P1_IMPLEMENTATION`  
Focused re-audit verdict: `REVISE_BLOCKED_BEFORE_P1A_GATE`

## Executive verdict

The revision fixes the original load-bearing errors about scalar squared-TT
marginal closure, moving ALS environments, the complete defensive normalizer,
and the direction of the structural delta substitution. The retained
quadratic-form identity is mathematically correct, and the ordered
total-derivative equations are substantially closer to an implementable score
contract.

The requested unblock does not yet pass on content. UB-1 still leaves the
physical/reference measure boundary ambiguous, overstates factorization reuse,
and admits a selected-branch quantity at a max tie as a score. UB-2 is still an
initial ledger with several directory-level rather than exact author-code
anchors. D1 supplies a useful viability screen but not a bias or accuracy gate.
D2 is correct only for a restricted globally invertible completion class; its
score statement additionally omits the spatial derivative of the retained
density at the parameter-dependent inverse state.

P0 contract work can continue. P1A may be developed only as explicitly
diagnostic scaffolding until the measure API and source anchors are repaired;
it cannot pass its claim-bearing gate. P1B and P2 remain blocked by their
declared prerequisites. The density-kernel track does not need UB-3, while P2S
does.

## Blocking findings

### 1. UB-1 does not assign one unambiguous measure to the retained evaluator

UB-1 defines the fitted adjacent target in reference coordinates and integrates
the previous block with its reference weight:

```text
E_t = integral H_R(u) H_R(u)' omega_prev(u) du
p_ret(z_curr) = (H_L(z_curr) E_t H_L(z_curr)' + tau q0_ret(z_curr)) / Zc_t.
```

See UB-1 lines 72-85 and 89-106. This construction returns a density with
respect to the retained reference measure, not automatically a density with
respect to physical-coordinate Lebesgue measure. But V1 calls it
`p_ret(x_prev)` and then adds the coordinate conversion for the full adjacent
block at UB-1 lines 37-44. If the quadratic-form evaluator returns the
reference-measure density, the previous-state conversion has already been
absorbed and the full-block term converts it a second time.

For `x = R(z)`, `J_R(z) = |det DR(z)|`, and reference measure
`mu(dz) = omega(z) dz`, the two evaluator contracts must be explicit:

```text
p_ref(z)  = p_phys(R(z)) J_R(z) / omega(z)
p_phys(x) = p_ref(R^{-1}(x)) omega(R^{-1}(x)) / J_R(R^{-1}(x)).
```

There are two correct next-step assemblies:

```text
# Retained evaluator returns reference density
log f_ref(z_curr, z_prev)
  = log p_ret_ref(z_prev)
  + log p_phys(x_curr | x_prev)
  + log p_phys(y_t | x_curr)
  + log J_curr - log omega_curr.

# Retained evaluator returns physical density
log f_ref(z_curr, z_prev)
  = log p_ret_phys(x_prev)
  + log p_phys(x_curr | x_prev)
  + log p_phys(y_t | x_curr)
  + log J_curr + log J_prev
  - log omega_curr - log omega_prev.
```

The plan must choose or expose both as non-interchangeable APIs, for example
`evaluate_reference_density(z)` and `evaluate_physical_density(x)`. The same
ownership must be stated for `q0_ret` and its normalizer. `U-MEASURE-1` must
test a complete two-step recursion against physical quadrature, including the
defensive component, rather than only isolated Jacobian identities.

The live code demonstrates why this boundary matters. It stores
`log_density_physical` and applies the previous-coordinate Jacobian while
integrating the retained physical density at
`bayesfilter/highdim/filtering.py:3437-3464`; it then converts the resulting
current physical predictive target at
`bayesfilter/highdim/filtering.py:2409-2416`.

Classification: `wrong/ambiguous relative to the exact-scalar claim until one
measure contract is selected`.

### 2. The donor does not share the value factorization with the derivative solve

UB-1 lines 151-158 and the plan line 148 say the tangent solve uses the same
factorization/solve operator as the primal and shares it across tangent columns.
That is not current implementation evidence.

The donor primal at
`bayesfilter/highdim/zhao_cui_moment_teacher_als.py:422-446` calls
`_solve_scaled_augmented_ridge`. That routine forms a column-scaled augmented
least-squares problem and solves it through the stable overdetermined solver at
`bayesfilter/highdim/fitting.py:984-1010`. The derivative primitive independently
forms the unscaled normal system and calls `tf.linalg.solve` at
`bayesfilter/highdim/derivatives.py:550-582`.

The normal-equation derivative can be mathematically correct for the exact
ridge minimizer, but it is not reuse of the current primal factorization and has
different numerical conditioning. P2A must either implement genuine reuse or
measure the non-reuse route honestly. Its obligations must separately test:

- scaled primal solution versus the declared normal-equation solution;
- derivative consistency with the actual scaled primal solver;
- behavior at column-scale floors and condition thresholds;
- runtime and peak memory with and without actual factorization reuse.

Until that is done, “shared factorization” is a design goal, not donor evidence.

Classification: `unsupported implementation claim`; the algebraic ordered
replay itself remains viable.

### 3. A max tie cannot produce an unqualified exact score

UB-1 lines 242-247 correctly restrict exact differentiability to unique-max
regions. Lines 249-253 then say that a tie reports the one-sided derivative of
the selected lowest-index branch.

For `s(theta) = max_j f_j(theta)`, a true tie generally has no ordinary
derivative. Its directional derivative is the maximum of the active branch
directional derivatives, not necessarily the derivative of the lowest-index
branch. Deterministic tie-breaking selects an implementation branch; it does
not make the max-defined scalar differentiable.

A detected tie must therefore invalidate the score-bearing evaluation for
claim use. A selected-branch derivative may be emitted as diagnostic telemetry
only, unless the finite target is explicitly redefined as that selected branch
and the resulting different scalar is accepted. `U-SHIFT-2` must assert the
claim veto, and tie-neighborhood FD tests must not treat a branch derivative as
the derivative of `max`.

Classification: `wrong relative to the exact-score claim at ties`.

### 4. D1 is a viability-tuning policy, not a mixing-bias or accuracy control

The T-tau step at plan lines 398-410 chooses the smallest `tau` whose validation
starvation diagnostics do not fire. This is a reasonable fail-closed viability
screen. It does not show that the retained approximation has low filtering bias
or that the smallest passing `tau` is scientifically adequate.

The present diagnostic set is insufficient because:

- a minimum density is coordinate- and unit-dependent unless compared with a
  density under the same declared measure;
- calibration and validation rows drawn from the same support family can miss
  posterior-path, parameter-boundary, long-horizon, or tail starvation;
- selecting both `tau` and the `q0` family increases selection freedom;
- `tau` is not comparable across differently normalized `q0` families;
- absent a same-target reference, accumulated mixing bias is not observed.

Required repair:

1. Normalize every candidate `q0` under the declared measure, or tune a
   normalized defensive mixture mass rather than an unidentified `tau*q0`
   scale.
2. Add independent stress rows spanning the declared HMC parameter region,
   observation tails, support boundaries, and long-horizon states.
3. Report dimensionless diagnostics such as `p_ret/q0`,
   `tau*q0/(h^2 + tau*q0)`, target-to-fit importance ratios or ESS where the
   row target is evaluable, weighted target mass, and boundary/tail mass.
4. After freezing `(tau, q0)`, evaluate the full predeclared sensitivity table
   on the untouched claim partition as a veto/descriptive check. Do not select a
   replacement on that partition; a failure triggers fresh tuning partitions.
5. Where no same-target reference exists, label the selection as
   `viability_tuning_only`. It is not evidence of low approximation bias,
   posterior correctness, or superiority.

The Zhao-Cui source also does not convert this empirical starvation screen into
an accuracy theorem. Equation (13) supplies the defensive construction, while
Lemma 1 imposes a source-specific relation between the defensive constant and
the square-root approximation error. Any use of that lemma for this frozen-ALS
program requires a separate checked transfer argument.

Classification: `AGREE as viability tuning`; `DISAGREE as accuracy or
bias-control evidence`.

### 5. D2 is correct only for a globally invertible support-restricted subclass

The Jacobian direction in plan lines 190-198 is correct under stronger
conditions than the plan states. For fixed `(m_prev, m_curr, theta)`, if

```text
k_curr = T_k(k_prev, m_prev, m_curr; theta)
```

is a single-valued diffeomorphism from the relevant `k_prev` support onto its
image, then integrating the delta over `k_prev` gives

```text
1 / |det D_kprev T_k| = |det D_kcurr S|.
```

The toy result `J = 1/|phi|` is therefore correct for
`k_curr = phi*k_prev + gamma*m_curr^2` when `phi != 0`.

The adapter contract must additionally bind:

- global one-to-one invertibility on the relevant support, not only local
  nonsingularity;
- the support image and boundary indicator;
- smooth dependence of the inverse and support on parameters;
- absence of unhandled inverse branches;
- lower singular-value and inverse-JVP bounds;
- bounds/telemetry for `log J`, J-weighted target rows, weighted mass,
  nonfinite values, and floor activation.

A condition number alone is not a near-singular guard. In the scalar toy,
`cond(phi) = 1` for every nonzero `phi`, while `J = 1/|phi|` diverges as
`phi -> 0`. The contract needs at least a minimum singular-value or inverse-norm
bound in addition to condition and log-determinant diagnostics.

The plan's statement that every non-invertible case yields a singular filtered
law is too broad. A finite many-to-one map can still induce an ordinary density,
with a sum over inverse branches:

```text
sum_b p_prev(m_prev, S_b(k_curr))
      / |det D_kprev T_k(S_b(k_curr))|.
```

Rank-deficient cases such as `phi = 0` in the toy can instead produce a
manifold-supported joint law. Both may remain out of v1, but they must not be
conflated.

The “no information loss” statement is valid only relative to the already
retained full-state approximation and only under the global inverse/support
conditions. It does not show equality to the exact filtering law. Likewise,
`n + n_stochastic` is a correct raw dimension count for this restricted
formulation, not a complexity result: the current endogenous axes remain, and
composition with nonlinear `S` and `J` may increase basis or TT-rank demands
enough to erase the nominal saving.

Finally, this mode is not universal over DSGE structural models. Ch18b lines
1616-1628 explicitly state that its general pushforward identity does not
require invertibility of `T_k`. D2 is one useful invertible-completion route,
not closure of the general structural case.

Classification: `AGREE conditionally on a global diffeomorphism`; `INSUFFICIENT
as a general structural-program claim`.

### 6. The D2 score chain omits the moving-point derivative of the retained law

The structural integrand evaluates

```text
p_ret,t-1(m_prev, S(k_curr; m_prev, m_curr; theta); theta).
```

Unlike UB-1's density-kernel rows, the previous physical state now moves with
`theta`. Its total derivative is

```text
d/dtheta log p_ret(m_prev, S(theta); theta)
  = partial_theta log p_ret
  + grad_kprev log p_ret * dot_S.
```

The second term is not supplied by an adapter JVP: `S` belongs to the adapter,
but the retained quadratic form belongs to the engine. If evaluation uses a
reference-coordinate object, this term must also propagate through
`R^{-1}(S)` and the selected physical/reference conversion.

UB-3 must therefore define a retained-evaluator spatial JVP or gradient, its
defensive-component contribution, coordinate-map inverse JVP, support status,
and FD tests. `dot_S` and `dot_log_J` alone do not complete the score.

Classification: `missing load-bearing total-derivative term for P2S`.

### 7. UB-2 has correct classifications but does not close the source-anchor gate

The ledger correctly classifies squared-TT/marginal mathematics as
source-faithful, weighted ridge ALS and the analytic score as repository
extensions, and structural substitution as an extension grounded outside the
Zhao-Cui source. That resolves the earlier attribution error.

It remains `INITIAL_LEDGER_ANCHORS_RECORDED`, and several author-code anchors
are not exact file/line anchors:

- row 1 cites the `@TTSIRT` directory;
- row 2 cites unspecified marginal/normalization routines;
- row 4 cites `@TTFun/cross.m` without lines;
- row 11 again cites unspecified normalization code.

By contrast, row 3's `@TTSIRT/marginalise.m:25-85` is an adequate exact code
anchor. The paper support inspected for this review is Zhao-Cui Section 3,
Equation (13), Lemma 1, and Proposition 2/Equation (14), together with the local
author snapshot's `marginalise.m:25-85` and `cross.m` implementation. The broad
anchors above must be replaced with the exact operations and line spans before
`BLOCK_SOURCE_UNGROUNDED` is closed. Plan Section 10, still labeled “initial;
anchors to be completed at UB-2,” must be synchronized with the ledger's final
status.

Classification: `INSUFFICIENT(missing exact author-code anchors)`.

## Requested per-artifact verdicts

| Artifact | Verdict | Reason |
|---|---|---|
| UB-1 score derivation | `INSUFFICIENT` | Ordered replay and retained tangents are substantially corrected, but measure ownership, actual solver reuse, and tie-veto semantics remain unresolved. |
| Retained quadratic-form contract | `AGREE` mathematically; `INSUFFICIENT` as an API | The quadratic-form marginal is correct as a reference-measure object; physical/reference evaluators and defensive-measure ownership are not defined. |
| UB-2 source ledger | `INSUFFICIENT` | Classifications are appropriate, but several author-code anchors are directory/file-level rather than exact operation lines. |
| D1 tau policy | `AGREE` as a viability screen; `DISAGREE` as an accuracy-control claim | Passing starvation diagnostics does not control recursive mixing bias without same-target evidence. |
| D2 structural substitution | `AGREE` conditionally; `INSUFFICIENT` as a general program claim | The Jacobian direction is correct for a global support-preserving diffeomorphism; branch, support, stability, scope, and moving-point score terms are incomplete. |

## Direct answers to the handoff questions

### D1(a): Is the starvation diagnostic set sufficient?

No. It has a selection-optimism hole when validation rows share the calibration
support generator, and its minimum-density statistic is not invariant to
coordinate or density units. Use independently generated parameter/path/tail
stress rows and the dimensionless ratio/mass diagnostics listed in Finding 4.

### D1(b): Is the anti-large-tau control adequate without a same-target reference?

No. In that case mixing bias is not measured. The most discriminating available
checks are a frozen full `(tau, q0)` sensitivity table on untouched claim data,
target-to-fit ratios where the per-step target is evaluable, defensive-mixture
fractions, and long-horizon stress-path stability. These can veto or explain;
they cannot establish low filtering bias.

### D2(a): Is the substitution and Jacobian direction correct?

Yes for a single global inverse branch on the relevant support. The factor is
`|det D_kcurr S| = 1/|det D_kprev T_k|`. Multiple branches require a branch sum;
off-image points require explicit zero support.

### D2(b): Is there no information loss?

Only relative to the already-retained full-state approximation, under the stated
global inverse/support assumptions. It is not a claim about the exact filter or
about approximation error introduced by the next TT fit.

### D2(c): Is the multivariate condition correct?

The determinant is over the endogenous block map
`k_prev -> T_k(k_prev, m_prev, m_curr; theta)`, but the condition must be global
on support, dimension-matched, branch-complete, and smoothly parameterized. A
pointwise nonsingular Jacobian is not sufficient by itself.

### D2(d): Is a condition-number veto sufficient?

No. Add minimum-singular-value/inverse-norm and `log J` bounds plus J-weighted
row/mass, nonfinite, floor, and support telemetry. The scalar `phi` example
proves condition number alone cannot detect Jacobian inflation.

### D2(e): Is the dimension claim optimistic?

The raw variable count `n + n_stochastic` is correct for this restricted route.
Calling it a material computational reduction before measurement is optimistic:
`k_curr` axes still need resolution, and nonlinear composition with `S` and `J`
can increase basis size and TT rank.

## Execution status

| Phase | Status after re-audit |
|---|---|
| P0 | May continue as contract/skeleton work, but must record the corrected measure APIs and restricted structural-mode scope before freezing interfaces. |
| P1A implementation | Diagnostic scaffolding only until Findings 1 and 7 are repaired. |
| P1A claim-bearing gate | `BLOCKED`. UB-1 and UB-2 landed as files but did not yet satisfy their content gates. |
| P1B | `BLOCKED` until P1A tests pass. |
| P2A/P2 | `BLOCKED` by the declared P1A prerequisites; P2A must also resolve/measure the actual solver reuse claim. |
| P2S | Sequencing is acceptable after UB-3 and P2. UB-3 must add global inverse/support conditions, retained-density spatial JVPs, complete S/J tangents, Jacobian-weight stability telemetry, and the dense Ch18b toy arbiter. |
| Density-kernel track | May remain independent of UB-3. |

## Pre-mortem

The revised program could pass its present gates and still mislead if:

- the retained quadratic form is correct under one measure while target
  assembly converts the previous state twice or not at all;
- tau passes same-family validation rows but starves on posterior-region,
  boundary, tail, or long-horizon claim paths;
- a scaled primal solve and normal-equation derivative agree on easy fixtures
  but diverge near scale floors or conditioning thresholds;
- a tie is merely recorded while a non-existent ordinary derivative is admitted
  as an exact score;
- the structural toy passes at moderate `phi`, while small singular values make
  `J`-weighted rows and score terms unstable despite a benign condition number;
- UB-3 differentiates `S` and `J` but omits
  `grad log p_ret * dot_S`;
- the `n + n_stochastic` axis count is reported as feasibility even though
  nonlinear inverse composition drives rank or basis growth;
- P1A proves a pointwise marginal identity but not recursive next-step target
  equality under the chosen measure convention;
- P2A passes at one or two time steps while full `T=120` tangent-state memory,
  retracing, or XLA behavior remains unacceptable.

## Source-support boundary

Primary technical source inspected locally: Zhao and Cui (JMLR 25, 2024),
Section 3, Equation (13), Lemma 1, Proposition 2, and Equation (14), plus the
pinned author implementation under
`third_party/audit/zhao_cui_tensor_ssm_p10/`. Ch18b technical sections on the
structural pushforward, worked nonlinear example, `phi=0`, and validation gates
were also inspected. No citation-count, venue-ranking, or broad literature-
completeness claim was needed for this focused operation-level audit; backward
and forward snowballing are therefore not used as evidence here. No source
establishes the repository's frozen weighted-ridge ALS score, its cost, D1
accuracy, D2 universality, HMC readiness, or NAWM feasibility.

## Final decision

`REVISE_BLOCKED_BEFORE_P1A_GATE`

The program direction remains viable. Repair the measure-owned retained API,
tie veto, solver-reuse wording/evidence, exact UB-2 anchors, D1 nonclaims, and
D2 global-support/spatial-JVP contract. After those bounded corrections, a
focused recheck can unblock the P1A gate without reopening the already accepted
mission, phase ordering, or leaderboard policy.

VERDICT: REVISE
