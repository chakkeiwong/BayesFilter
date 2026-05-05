# Plan: reviewer-response rewrite of Chapter 18b

## Scope

Rewrite `docs/chapters/ch18b_structural_deterministic_dynamics.tex` so that it
becomes a readable, literature-grounded, implementation-ready chapter on
structural deterministic dynamics in nonlinear DSGE filtering.

This plan responds directly to the reviewer concerns supplied on 2026-05-06.
It supersedes the chapter-level intent of
`docs/plans/ch18b-restructuring-and-literature-strengthening-plan-2026-05-05.md`
only for Chapter 18b.  It does not rewrite the implementation plans under
`docs/plans/bayesfilter-structural-svd-*`.

## Diagnosis

The current chapter has the right core thesis but poor shape.  It repeats the
same structural warning in several places, introduces the original UKF too
briefly, and then surrounds a useful example with too many proposition-like
blocks.  The formal material often proves a measure-theoretic pushforward
identity in words, then uses that identity to justify implementation policy
without clearly separating assumptions, definitions, exact claims, literature
claims, and BayesFilter policy.

The result is that a reader can miss the simple point:

1. the filtered state is still the full state `x_t`;
2. the sigma-point variable is the pre-transition random input that generates
   the predictive law;
3. deterministic-completion coordinates are outputs of the structural map, not
   independent new noise channels;
4. square-root/SVD methods can improve covariance factorization without fixing
   a wrong structural law.

## Source material to use

Use the local PDFs already present in `docs/`:

- `docs/A general method for approximating non-linear transformations of probability distributions Julier(96).pdf`
- `docs/Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models Merwe(03).pdf`

Use existing chapter and bibliography material:

- `docs/references.bib`
- `docs/chapters/ch02_state_space_contracts.tex`
- `docs/chapters/ch16_sigma_point_filters.tex`
- `docs/chapters/ch17_square_root_sigma_point.tex`
- `docs/chapters/ch18_svd_sigma_point.tex`
- `docs/chapters/ch28_nonlinear_ssm_validation.tex`
- `docs/chapters/ch32_production_checklist.tex`

Add a `vanDerMerweWan2003`-style bibliography entry only after checking the
publication metadata from the PDF or a reliable bibliographic source.

## Reviewer concern map

### R1. Algorithm is too abstract for implementation

Rewrite the algorithm sections into code-generation-grade pseudocode.

Required algorithm blocks:

1. `structural_filter_step`, backend-neutral.
2. `standard_additive_noise_ukf_step`, following the additive-noise UKF
   tradition.
3. `structural_ukf_step`, using the same moment/update notation as the standard
   UKF.

Each algorithm must specify:

- inputs: previous mean/covariance, transition maps, observation map, noise
  laws, sigma-point parameters, numerical factorization policy;
- augmented Gaussian variable;
- sigma-point construction and weights;
- propagation equations point by point;
- deterministic-completion call;
- observation point construction;
- predicted state mean/covariance;
- predicted observation mean/covariance;
- state-observation cross covariance;
- gain, posterior mean/covariance, log-likelihood contribution;
- returned metadata: integration space, deterministic policy, approximation
  label, factorization method, failure diagnostics.

### R2. Need review of original or naive UKF

Add a short literature subsection before structural UKF:

- define the unscented transform as a deterministic quadrature rule for
  approximating moments of `Y=g(X)`;
- reproduce the Julier (1996) sigma-point equations in BayesFilter notation;
- reproduce the scaled UKF weights from van der Merwe and Wan (2003) / standard
  SPKF notation;
- explain that the "naive" UKF in this chapter means applying the additive-noise
  full-state pattern to all coordinates of a mixed structural state.

Be precise: do not imply every UKF variant is identical.  Say which variant is
being used for exposition.

### R3. Compare and contrast standard UKF and structural UKF

After the two algorithm blocks, add a line-by-line comparison table:

| Object | Standard additive-noise UKF | Structural UKF |
| --- | --- | --- |
| filtered object | full state `x_t` | full state `x_t` |
| sigma-point variable | augmented state/noise vector declared by additive model | `(x_{t-1}, eps_t)` or equivalent pre-transition uncertainty |
| direct process noise | whatever the additive model declares | only declared stochastic block |
| deterministic coordinates | none unless the model declares constraints | completed by `T_k` pointwise |
| target law | additive-noise predictive law | structural pushforward law |
| failure if misused | approximation error | model-law error plus approximation error |
| SVD/square-root role | covariance factorization | covariance factorization, not structural repair |

Then add a small "exact difference" paragraph:

The update equations are the same.  The difference is entirely upstream: which
random variable receives sigma points and which map generates the predictive
state cloud.

### R4. Need proof of second-order approximation

Do not invent a new broad theorem.  State a narrow proposition:

Let `A_t=(x_{t-1}, eps_t)` have mean `mu_A` and covariance `P_A`.  Let
`F_theta(A_t)` be the structural transition map.  If the sigma-point rule
matches the first two central moments of `A_t` and `F_theta` is sufficiently
smooth/analytic near `mu_A`, then the structural UKF's predicted mean and
covariance inherit the usual unscented-transform accuracy for the transformation
`F_theta`.  For Gaussian inputs, use the Julier (1996) framing: the transformed
mean agrees through the lower-order terms captured by the sigma-point moment
matching, with errors beginning in the fourth-order terms for the standard
setup discussed there; the covariance is exact through the second-order terms,
with higher-order errors as described in the source.

Important boundaries:

- This accuracy applies to the transformation of the correct random variable
  `A_t`, not to a naively perturbed post-transition full state.
- Structural correctness says the target law is right; UT accuracy says the
  Gaussian moment approximation to that target has the usual order properties.
- Nonlinear observation moments need a separate application of the same UT
  argument to `h(F_theta(A_t))`.

The proof should be written as a reduction:

1. define `A_t` and `F_theta`;
2. show `x_t = F_theta(A_t)` almost surely;
3. invoke the UT moment-matching theorem/argument from Julier (1996) for
   `Y=F_theta(A_t)`;
4. conclude that structural UKF has the same local moment-approximation order as
   the UT applied to the correct input distribution;
5. state explicitly that no such conclusion follows for the naive full-state
   cloud unless its induced input law equals the structural pushforward law.

### R5. Define terms with notation

Add a definition block near the beginning.

Required terms:

- structural state: `x_t=(m_t,k_t)`;
- stochastic block: `m_t`;
- deterministic-completion block: `k_t`;
- structural transition map:
  `F_theta(x_{t-1}, eps_t)=(T_m(m_{t-1},eps_t), T_k(k_{t-1},m_{t-1},T_m(...)))`;
- predictive law:
  `pi_{t|t-1}^x(B)=Pr(x_t in B | y_{1:t-1})`;
- pushforward:
  if `A_t ~ pi_t^A`, then `F_{theta#} pi_t^A(B)=pi_t^A(F_theta^{-1}(B))`;
- "place sigma points":
  choose weighted points `{a_t^{(j)},w_j}` whose weighted mean/covariance match
  the approximation to `pi_t^A`, then propagate `x_t^{(j)}=F_theta(a_t^{(j)})`.

Avoid using these words later without the notation nearby.

### R6. Propositions 19.1 and 19.2--19.6 are too verbal

In the current file these correspond to one general proposition and five
example-specific proposition blocks.  Rewrite as follows:

- Keep exactly one main proposition in the general part:
  "Predictive pushforward and structural sigma-point variable."
- Add exactly one UT-accuracy proposition:
  "Structural UKF inherits unscented-transform moment accuracy on the correct
  input law."
- Downgrade example-specific propositions to:
  - "Identity",
  - "Worked derivation",
  - "Remark",
  - or "Diagnostic implication."

Proof style requirements:

- state assumptions before the proposition;
- define the probability measures and maps;
- prove equalities using test functions or measurable-set preimages, not only
  prose;
- separate exact mathematical statements from implementation policy;
- when a statement is not a theorem, label it as a diagnostic or policy rule.

### R7. Chapter is hard to read

Rewrite around this sequence:

1. Problem statement and what the chapter proves.
2. Definitions and notation.
3. Why exact linear Kalman can hide the problem.
4. Original unscented transform and standard UKF.
5. Structural UKF.
6. Direct comparison.
7. Two formal propositions.
8. Worked example A: simple nonlinear structural transition.
9. Reviewer edge case: `phi=0`.
10. Worked example B: degenerate linear transition with nonlinear observation.
11. SVD/square-root distinction.
12. DSGE adapter implications.
13. Validation checklist.

Delete repeated warnings once the comparison and propositions make the point.
Move numerical tables into the examples and avoid re-explaining the whole
doctrine inside each example.

### R8. Literature and SVD sigma-point issue

Add a literature map subsection or paragraph cluster:

- Julier and Uhlmann (1996, 1997): unscented transform and UKF accuracy
  intuition.
- van der Merwe and Wan (2003): SPKF family, scaled weights, augmented state
  conventions, square-root SPKF motivation.
- Durbin and Koopman (2012): state-space filtering and singular/degenerate
  Gaussian caution where appropriate.
- Herbst and Schorfheide (2015), An and Schorfheide (2007): DSGE likelihood and
  state-space estimation context.
- Kim et al. (2008), Andreasen et al. (2018): second-order/pruned DSGE
  state-space structure.
- Gordon et al. (1993), Doucet et al. (2001), Andrieu et al. (2010): particle
  propagation/weighting semantics.
- Chapters 17 and 18: square-root/SVD factorization and derivative-risk
  boundaries inside BayesFilter.

Answer the reviewer directly:

An SVD sigma-point filter can solve a covariance factorization problem for a
singular or ill-conditioned covariance matrix.  It does not automatically solve
the structural-law problem.  It solves the structural-law problem only if its
sigma points are built on the declared stochastic/pre-transition integration
space and deterministic coordinates are completed pointwise before observation
moments are computed.  If it instead takes a full post-transition covariance,
adds a nugget, and perturbs deterministic-completion coordinates directly, SVD
only gives a stable factorization of the wrong approximate law.

### R9. Reviewer's `phi=0` collapse question

Add a short edge-case box in the nonlinear toy example.

For

```tex
m_t = \rho m_{t-1}+\sigma\varepsilon_t,
\qquad
k_t = \phi k_{t-1}+\gamma m_t^2,
```

if `phi=0`, then

```tex
k_t = \gamma m_t^2.
```

The dependence on `k_{t-1}` disappears, but the structural state does not
generally collapse to a point unless `gamma=0` or `m_t` is degenerate.  The
support becomes a lower-dimensional curved manifold

```tex
\{(m,k): k=\gamma m^2\}
```

in the current state coordinates.  This actually strengthens the structural
message: direct independent perturbations of `k_t` move sigma points off that
manifold.  What collapses is the memory channel from `k_{t-1}` to `k_t`, not the
randomness induced by the current shock through `m_t`.

If `phi=0`, `gamma != 0`, and `sigma>0`, then `k_t` has nonzero predictive
variance in general because it is a nonlinear function of the random `m_t`.

## Rewrite inventory and pruning

Current blocks to retain, merge, or cut:

| Current block | Action |
| --- | --- |
| production warning | retain, shorten |
| structural split | retain, add definitions |
| linear Kalman hiding the problem | retain, shorten |
| nonlinear filters must respect structural map | merge into problem statement and backend-neutral algorithm |
| reusable structural filtering step | retain, rewrite as implementation-ready pseudocode |
| degenerate transitions are structural | merge into SVD/square-root distinction |
| original UT pattern | expand using Julier (1996) and van der Merwe/Wan notation |
| standard UKF algorithm | expand to code-generation detail |
| structural UKF algorithm | expand to parallel code-generation detail |
| why algorithms differ | replace with comparison table |
| predictive pushforward proposition | keep, rewrite with definitions |
| nonlinear structural toy example | keep as Example A; reduce repetition |
| numerical illustration | keep only if all numbers are rechecked |
| five formal propositions in exp-affine example | downgrade to identities/remarks/worked derivations |
| exp-affine numerical example | keep as Example B; focus on observation-side quadrature distinction |
| structural correctness boundary | retain, shorten |
| structural versus numerical degeneracy | retain, expand SVD answer |
| pruned DSGE and adapter implications | retain, tie to literature |
| source-project lesson | keep short or move to note; do not dominate chapter |
| validation gates | retain, make checklist concise |
| common misunderstandings | keep only non-duplicated items |

## MathDevMCP plan

MathDevMCP can help, but it should be used as a proof-audit and consistency
tool, not as a magical theorem prover for every measure-theoretic statement.

Use it for:

1. label lookup before editing:
   `search_latex`, `latex_label_lookup`, `extract_latex_context`;
2. proof hygiene diagnostics:
   `typed_obligation_label` and `audit_derivation_v2_label` for the two main
   propositions;
3. bounded algebra checks in examples:
   `check_equality` or `derive_label_step` for identities such as
   `k_t - phi k_{t-1} - gamma m_t^2 = 0` after substitution;
4. document-code consistency, if the chapter is tied to implementation:
   `implementation_brief`, `compare_label_code`, and `audit_implementation_label`;
5. release-quality reporting:
   `release_readiness` only after labels and proof obligations are stable.

Expected limitation:

- pushforward definitions, sigma-point placement policy, and literature
  accuracy claims will still require human mathematical judgment and careful
  citation.  MathDevMCP can identify ambiguous obligations and stale labels; it
  will not certify the whole chapter as a formal proof.

Initial diagnostic already run:

- `typed_obligation_label` on `prop:bf-structural-ukf-pushforward` returned a
  consistent diagnostic but routed the main obligation to human review.
- `audit_derivation_v2_label` on `prop:bf-structural-ukf-pushforward` and
  `lem:bf-exp-affine-latent-pushforward` returned `unverified`, mostly because
  the prose currently creates many ambiguous/manual obligations.  This supports
  the rewrite decision to reduce theorem-like blocks and split assumptions from
  conclusions.

## Research-assistant plan

The local `~/research-assistant` MCP is currently read-only and did not find
summary records for the two PDFs by query.  Use it for local source summaries if
records are later ingested, but for this rewrite use the PDFs directly via
`pdftotext` plus manual reading.

Required extraction from Julier (1996):

- sigma-point construction;
- transformed mean/covariance formulas;
- Taylor-series accuracy discussion;
- caveats about covariance PSD and parameter choices.

Required extraction from van der Merwe and Wan (2003):

- SPKF family framing;
- scaled UKF weights;
- statement that sigma points capture first and second moments;
- square-root SPKF numerical-stability role.

## Verification gates

Before accepting the rewrite:

1. `rg` for repeated heavy phrases:
   `predictive law`, `pre-transition uncertainty`, `wrong state-space model`,
   `deterministic-completion coordinate`.
2. `rg` theorem count:
   no more than two general proposition blocks in the main UKF part.
3. MathDevMCP:
   run `typed_obligation_label` and `audit_derivation_v2_label` on the two main
   propositions and record remaining manual obligations.
4. Numerical example check:
   recompute all shown numerical values from a small script or notebook and
   keep the script path or formula provenance.
5. Bibliography check:
   all citation keys compile and each source supports the exact claim attached
   to it.
6. LaTeX:
   `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

## Executable phases

Run the rewrite in these phases, with the reset memo updated after each phase.

### Phase A. Definitions, source setup, and notation

Deliverables:

- add missing bibliography support for van der Merwe's sigma-point source;
- introduce a definition block for structural state, stochastic block,
  deterministic-completion block, predictive law, pushforward, and sigma-point
  placement;
- tighten the opening so the chapter states exactly what it proves and what it
  does not prove.

Pass criteria:

- chapter builds;
- the terms `pushforward`, `predictive law`, and `place sigma points` are
  defined with notation before being used heavily.

### Phase B. Literature-grounded UKF algorithms

Deliverables:

- expand the original unscented-transform review using Julier (1996) and the
  van der Merwe sigma-point notation;
- write the standard additive-noise UKF algorithm in code-generation detail;
- write the structural UKF algorithm in parallel detail;
- add the exact comparison table.

Pass criteria:

- another coding agent can identify the inputs, sigma-point variable,
  propagation maps, moment formulas, update equations, and returned metadata for
  both UKF paths.

### Phase C. Formal statements and proof hygiene

Deliverables:

- keep one pushforward proposition;
- add one UT-accuracy inheritance proposition;
- downgrade example-specific propositions to identities, remarks, diagnostics,
  and worked derivations;
- run MathDevMCP on the two main propositions and record the boundary.

Pass criteria:

- the main UKF part has no more than two proposition environments;
- example-specific material no longer pretends to be a theorem ladder.

### Phase D. Worked examples and reviewer edge case

Deliverables:

- reorganize examples so the nonlinear structural example includes the
  `\phi=0` edge case explicitly;
- keep/recheck the numerical comparisons;
- present the degenerate linear-transition / nonlinear-measurement case as a
  second example focused on observation-side quadrature versus latent-law error.

Pass criteria:

- all displayed numerical values are recomputed;
- the `\phi=0` question is answered directly.

### Phase E. Literature/SVD/adapter/validation close

Deliverables:

- add a literature map tying each source class to the claims it supports;
- answer the SVD sigma-point objection directly;
- tighten DSGE adapter implications and validation gates.

Pass criteria:

- the chapter distinguishes structural-law correctness, quadrature accuracy,
  covariance factorization, and derivative/HMC risk.

### Phase F. Final audit, tidy, memo, and commit

Deliverables:

- run full LaTeX;
- run repeated-phrase/theorem-count/citation-key checks;
- update the reset memo with completion results and next hypotheses;
- commit only the files changed for this pass, plus the two local reference
  PDFs if they are still needed for provenance.

Pass criteria:

- build succeeds;
- git commit succeeds;
- final response includes results, residual risks, and next testable
  hypotheses.

## Done criteria

The rewrite is done when a reader can answer these questions from the chapter
without inference:

1. What is the structural transition map?
2. What is the predictive law and what is its pushforward representation?
3. What does it mean to place sigma points on a variable?
4. What exact algorithm does the naive additive-noise UKF run?
5. What exact algorithm does the structural UKF run?
6. Which equations are shared by both algorithms?
7. What changes when `phi=0` in the toy example?
8. What approximation order does structural UKF inherit from the UT, and under
   what assumptions?
9. Why does an SVD sigma-point filter not automatically fix the structural
   problem?
10. Which validation tests prevent the implementation from silently adding
    artificial noise to deterministic coordinates?
