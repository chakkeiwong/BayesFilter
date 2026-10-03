# Codex audit reply: UKF discontinuous-gradient HMC program

This second audit reads the requested survey sections, plan, governing program, G2.3 result, and hardbound source files. The repository hardbound lane contains no UKF marginal-likelihood implementation. The survey does describe a UKF route, but its own case-study text calls that route smooth; no execution artifact measures the asserted discontinuity. No verdict below relies on the survey's §7 prose as evidence for itself.

## Verdict table

| ID | verdict | anchor (file:line or eq label) | one-line basis |
|---|---|---|---|
| M1 | wrong relative to the stated target | `bayesfilter/hardbound/dns_curve_tf.py:45-49`; survey `eq:ukf1` | The coded softplus is C-infinity; the hard target has `tf.maximum`, but no UKF branch yields the asserted softplus gradient jump. |
| M2 | wrong relative to the stated target | survey `eq:56a`; §7.5 lines 1226-1235 | A true gradient jump makes the difference quotient diverge as delta tends to zero, unlike a finite smooth Lipschitz estimate. |
| M3 | wrong relative to the stated target | survey lines 999-1003, `eq:56a`; §7 lines 1130-1145 | The cited jump bound is vacuous for smooth softplus; stiffness selects stability and preconditioning, not event detection. |
| M4 | wrong relative to the stated target | survey lines 1073-1085 | Low softplus sensitivity gives zero Kalman gain and preserves covariance, so the displayed “thus” reverses the implication. |
| M5 | wrong relative to the stated target | survey lines 1095-1113; `model_tf.py:37` | `S >= R` bounds the inverse; no evidence shows P-collapse alone creates the claimed gradient amplification. |
| M6 | wrong relative to the stated target | survey §7.2 lines 1095-1105 | As P tends to zero, symmetric sigma points converge to the mean and the unscented mean tends to h(m), rather than flipping a whole cluster. |
| M7 | wrong relative to the stated target | survey `eq:94`-`eq:97`, lines 2817-2841 | Regime switching alone need not jump the value; the survey’s derivation locates jumps in multiplicity and selection. |
| M8 | wrong relative to the stated target | survey `eq:12b`-`eq:13`, lines 1729-1744, 1819-1843 | The displayed mixture is not the bootstrap increment, and ordinary resampling is discontinuous; smoothing needs explicit coupling or differentiable resampling and does not remove score bias. |
| M9 | wrong relative to the stated target | survey lines 193-238; §7.8 lines 1332-1335 | The taxonomy is Case 1 max/kink, Case 2 deterministic branch jump, Case 3 mixed support, Case 4 multiplicity; §7.8 calls the value jump Case 3. |
| M10 | wrong relative to the stated target | survey `eq:ukf1`, `eq:77`; `dns_curve_tf.py:45-49` | The equations agree only under alpha-temperature = 1/a-scale; the fixture and O(alpha) statements mix conventions and confuse curvature with a jump. |
| M11 | unsupported | survey lines 1270, 1343; `references.bib` search | The citation is dangling; no Tran-Kleppe paper or verified metadata/full text was available in the inspected corpus. |
| M12 | unsupported | survey lines 1040-1044; plan reference 4 at lines 765-770 | The exact-truncated-Gaussian method claim is explicitly based on an unread paper, so its source support is incomplete. |
| M13 | wrong relative to the stated target | survey §2 lines 193-224; §6 lines 979-1046; §13 lines 2103-2115, 2651-2681; DPF lines 1819-1843 | §7’s softplus-kink and DPF-certainty claims conflict with the survey’s smooth-model and finite-score qualifications. |
| M14 | unsupported | survey `sec:shadow` label at lines 2012-2013; lines 2103-2115; `bayesfilter/hardbound/` file list | The label resolves and the survey narrates a UKF route, but no repository implementation or measurement artifact substantiates it. |
| M15 | wrong relative to the stated target | master program lines 17-20, 29-43, 49-56; plan lines 6-23, 142-222, 613-638 | The plan changes Program A’s C1 filter-free scope into UKF/S1, event-aware, NeuTra, and production work; that requires a new program. |
| M16 | wrong relative to the stated target | plan lines 104-110; `model_tf.py:16-18`; G2.3 result lines 217-220 | G2.3 is an 8-state joint HMC fixture using hard max, not a UKF-filtered two-state test. |
| M17 | unsupported | G2.3 result lines 73-88, 179-193 | The executed evidence supports a theta8 noise-scale bottleneck and offers funnel/periodicity hypotheses; it does not test UKF kink crossing. |
| M18 | wrong relative to the stated target | plan lines 454-508; governance requirements; master lines 39-43, 251-253 | Phase 4 plans a PyTorch checkpoint and omits required GPU, manifest, inference-status, and pre-mortem controls. |
| M19 | unsupported | plan lines 22, 458, 548; G2.3 result lines 29-35, 92-145, 205-220 | The arithmetic 20,000 x 36 seconds = 200 hours is stated, but no UKF runtime supports it and the 800-hour allocation has no evidence-based cost model. |
| M20 | wrong relative to the stated target | survey lines 1265-1281; plan lines 511-531 | A surrogate proposal with true-target Metropolis correction is exact; HMC targeting the surrogate is a different, biased method. |
| M21 | wrong relative to the stated target | plan lines 22, 239, 525; G2.3 result lines 29-40 | The plan’s ESS/grad > 0.1 is about 222 times the measured 4.510e-4 ladder minimum and is not operationally defined or calibrated to the ladder. |
| M22 | unsupported | plan lines 6, 552-568; hardbound source file list | No NAWM II model or d=100/T=120 implementation is present; the stated target is aspirational. |
| M23 | correct | master lines 17-27; `model_tf.py:16-18`; G2.3 result lines 184-220 | The ledger separates executed Program A hard-max joint HMC from the unexecuted UKF plan phases. |
| M24 | wrong relative to the stated target | survey §2 lines 193-224; §13 lines 2103-2115 and 3222-3240 | §7 duplicates the taxonomy and contradicts the case study’s explicit smooth-UKF characterization; it should be narrowed or retracted pending measurement. |
| M25 | heuristic only | master lines 29-43, 64-100; survey lines 1181-1213 | A one-dimensional value/gradient/stability probe would discriminate smooth stiffness, a hard kink, a funnel, and tuning failure, but has not been run and is outside Program A. |

## M1

The coded map is

\[
s(u)=\ell+a\log(1+\exp z),\qquad z=(u-\ell)/a.
\]

Using (d\log(1+e^z)/dz=\sigma(z)) and (dz/du=1/a),

\[
s'(u)=\sigma(z),\qquad s''(u)=\frac{\sigma(z)(1-\sigma(z))}{a}.
\]

Since (0<\sigma(1-\sigma)\leq1/4),

\[
\sup_u s''(u)=\frac{1}{4a}.
\]

Thus curvature grows as (a^{-1}). In the survey temperature convention (α=1/a), it grows as (α/4). The inspected hardbound package has `tf.maximum` in the C1 hard map (`joint_target_tf.py:152,254`), a fixed Python `bound_map` branch (`dns_curve_tf.py:74-78`), and fixed Python horizon loops. Its Cholesky calls are in mass adaptation (`dense_mass_matrix_adaptation.py:220-232`, `windowed_dense_mass_adaptation.py:153-172`), not a UKF recursion; no `tf.where`, sort, pivot/fallback Cholesky, or parameter-dependent iteration was found in the target path. There is no UKF recursion in `bayesfilter/hardbound/` to inspect further.

For a finite composition of smooth maps, matrix operations, a positive-definite Cholesky, and log determinant, the parameter-to-likelihood map is smooth. Therefore the displayed unequal one-sided gradients in §7.3 are **wrong relative to the stated target** for the softplus case. The correct difficulty is finite but possibly severe curvature/stiffness, which belongs to the smooth geometry, not the continuous-kink geometry. A hard max or an actual fixed branch map is a separate case.

## M2

Anchor: survey section 7.3, lines 1124-1128, and section 7.5, lines 1226-1235.

For a differentiable gradient (g),

\[
\frac{\|g(\theta+\delta)-g(\theta)\|}{|\delta|}
\to \|g'(\theta)\|<\infty
\]

under a finite local Hessian. At a genuine jump, (g(\theta_b^+)\ne g(\theta_b^-)), the numerator tends to a positive constant while the denominator tends to zero, so the quotient tends to infinity. Hence §7.3 and a finite §7.5 estimate cannot both describe the same softplus object. The smooth finite-Lipschitz statement survives; §7.3 must state equal one-sided gradients and finite stiffness rather than a jump.

## M3

The proposition surrounding `eq:56a` assumes a fixed piecewise-regular partition and excludes implicit multi-root solvers, state-dependent iteration counts, and unfixed branch rules (survey lines 991-1003). A smooth softplus recursion can be treated as the degenerate (N=\varnothing) case if its covariance remains positive definite and its iteration schedule is fixed. It does not, however, supply a crossing with two one-sided gradients. Under M1,

\[
\|\nabla U^+-\nabla U^-\|=0,
\]

so `eq:56a` gives only (\Delta H=O(\epsilon\cdot0)=0) and is vacuous as an explanation of a kink. If the hypothetical UKF instead uses an implicit solver or parameter-dependent stopping/branching, it is outside the proposition until a separate theorem is supplied.

For a smooth potential, leapfrog has local truncation error (O(\epsilon^3)) (and fixed-time global error (O(\epsilon^2))); the usual smooth-potential local Hamiltonian error is (O(\epsilon^3)), not (O(\epsilon\Delta g)). For the harmonic mode with Hessian eigenvalue λ, the linear update is stable only when

\[
\epsilon\sqrt{\lambda}<2,
\qquad \epsilon<2/\sqrt{\lambda_{\max}}.
\]

Because (s''=O(a^{-1})), a dominant composed Hessian can scale as (\lambda_{\max}=O(a^{-1})=O(α)), giving a stability scale (\epsilon=O(\sqrt a)=O(α^{-1/2})). The mathematics therefore selects step-size reduction, mass scaling, or preconditioning. Event detection is a different program for an actual nonsmooth switching surface.

## M4

Let (m^-_t,P^-_t) be the predicted mean and covariance and let the scalar shadow input be (u_t=g(m^-_t,\theta)). For a scalar observed rate, (h_t=s(u_t)) and

\[
H_t=\frac{\partial h_t}{\partial x}=s'(u_t)\,\frac{\partial g}{\partial x},
\quad S_t=H_tP^-_tH_t^{\mathsf T}+R,
\]

\[
K_t=P^-_tH_t^{\mathsf T}S_t^{-1},
\quad P_{t|t}=(I-K_tH_t)P^-_t.
\]

For a vector observation, (H_t) is the Jacobian whose relevant rows carry the same factor (s'(u)), and the equations are unchanged. If (s'\to0), then (H_t\to0), (S_t\to R), (K_t\to0), and

\[
P_{t|t}\to P^-_t.
\]

The observation is uninformative about deep shadow depth; covariance is preserved by the update and can grow under prediction. If (s'\to1), (H_t\to\partial g/\partial x) and the ordinary informative Kalman contraction applies; in the scalar case (P_{t|t}=P^-_tR/(H_t^2P^-_t+R)), which is smaller than (P^-_t) when (H_t\ne0) and (R>0). Therefore the sentence at survey lines 1078-1085 is **wrong relative to the stated target**: the “thus” does not follow from the preceding uninformative regime. Statements about informative updates shrinking covariance can survive, but the claimed deep-below-bound mechanism and the downstream gradient-jump explanation do not.

## M5

Write (r=y-h(m,\theta)), (S=HPH^{\mathsf T}+R), and differentiate in a parameter direction (d\theta). For (q=r^{\mathsf T}S^{-1}r),

\[
dq=2(dr)^{\mathsf T}S^{-1}r-r^{\mathsf T}S^{-1}(dS)S^{-1}r.
\]

Since (S\succeq R\succ0), (\|S^{-1}\|\leq\|R^{-1}\|), hence

\[
|dq|\leq2\|dr\|\|R^{-1}\|\|r\|
 +\|r\|^2\|R^{-1}\|^2\|dS\|.
\]

For the log determinant,

\[
d\log|S|=\operatorname{tr}(S^{-1}dS),\qquad
|d\log|S||\leq n_y\|R^{-1}\|\|dS\|.
\]

With fixed measurement noise,

\[
dS=(dH)PH^{\mathsf T}+H(dP)H^{\mathsf T}+HP(dH)^{\mathsf T};
\]

add (dR) if noise is parameterized. The small inverse scale is controlled by measurement covariance (R), not by a collapsed (P). The fixture records a measurement noise scale of (5\times10^{-4}) (`model_tf.py:37`; the covariance convention must be stated before converting that scale to (R)). Collapse can affect (dP), (dH), and residuals, but no inspected UKF establishes such a bound-breaking derivative. Thus the §7.2 mechanism does not survive the (S\succeq R) argument as stated: it is **wrong relative to the stated target**, not an established consequence of variance collapse.

## M6

Anchor: survey section 7.2, lines 1095-1105; the sigma-point form below is the standard fixed-weight UKF construction.

For a standard (n)-dimensional UKF with symmetric sigma points, write (c_j=\sqrt{(n+\lambda)P}\,e_j). The unscented observation mean is

\[
\hat y=w_0h(m)+\sum_{j=1}^{n}w_j\{h(m+c_j)+h(m-c_j)\}.
\]

Taylor expansion gives

\[
h(m\!\pm\!c_j)=h(m)\pm J_h(m)c_j
 +\tfrac12 c_j^{\mathsf T}H_h(m)c_j+O(\|c_j\|^3).
\]

The odd terms cancel and the weights sum to one, so

\[
\hat y=h(m)+\tfrac12\sum_j w_jc_j^{\mathsf T}H_h(m)c_j+O(\|P\|^{3/2}),
\]

and therefore \(\hat y\to h(m)\) as \(P\to0\). Differentiating the second-order form gives the explicit parameter dependence:

\[
\partial_\theta\hat y
=h_\theta+J_hm_\theta
+\tfrac12\partial_\theta\!\left[\sum_jw_jc_j^{\mathsf T}H_hc_j\right]
+\partial_\theta O(\|P\|^{3/2}).
\]

The term involving \(P_\theta\) need not vanish just because \(P\) tends to zero; it is nevertheless a smooth covariance-sensitivity term when \(P(\theta)\) is smooth and the covariance stays positive definite. What vanishes is the point spread itself, and all sigma points converge to \(m\); there is no mathematical “simultaneous regime contribution.” A sharp response can still arise if \(m(\theta)\) traverses a high-curvature softplus layer, if \(P_\theta\) is large, or if an independent branch rule exists, but that is a curvature/covariance/branch explanation, not the stated collapse argument. Verdict: **wrong relative to the stated target**.

## M7

The blanket claim is false. At a guess-and-verify boundary, the binding inequality is normally attained with equality, and individually continuous branch maps can meet continuously. The survey’s own multiplicity derivation makes the source of a jump explicit. In `eq:94`-`eq:96`, both fundamental and binding solutions are admissible on the same open region, while each branch map is continuous. The text then states that the discontinuity is created “entirely by selection” (`eq:97`), where (T_f\ne T_b) throughout the multiplicity region, and that nonexistence occurs below the threshold (survey lines 2817-2836). It classifies the unresolved object as Case 4, a deterministic selection as Case 2, and a stochastic completion as Case 3 (lines 2837-2841). Thus regime switching per se does not imply a value jump; multiplicity, selection, or non-existence is required. §7.4 overclaims and is **wrong relative to the stated target**.

## M8

(a) In a bootstrap filter (q_t(x_t|x_{t-1})=p(x_t|x_{t-1},\theta)), so the incremental weight is (v_t^j=g(y_t|X_t^j,\theta)), after drawing an ancestor and propagating from the transition. More generally, the survey’s importance weight is `eq:12b`, with transition times observation density divided by the proposal. The predictive density used in the displayed mixture is instead

\[
p(y_t|X_{t-1}^i,\theta)=\int p(x_t|X_{t-1}^i,\theta)g(y_t|x_t,\theta)\,dx_t.
\]

For a nonlinear softplus observation this integral is not generally closed form. Conditional on (x_t), the Gaussian observation density is evaluable; that is not the same as a closed-form predictive integral. The survey itself gives the correct estimator product in `eq:13`.

(b) Multinomial resampling draws ancestor indices by thresholding uniforms against cumulative weights. With fixed random numbers, a small change in θ can change an index discontinuously, so the finite-particle likelihood computation is discontinuous even when the model is smooth (survey lines 1729-1734). Common random numbers provide coupling and variance reduction, not differentiability. Differentiable resampling, soft/transport resampling, or an explicitly constructed continuous extended random-variable map is required; it changes the finite forward law and needs a stated correction or approximation analysis.

(c) This is consistent with the repository/survey position on LEDH: the finite-particle score can be noisy and biased relative to the exact marginal force. The survey says stop-gradient genealogy scores are consistent only as particle count grows and are not covered by ordinary deterministic HMC exactness at finite count (lines 1819-1830); OT resampling adds regularization and changes the finite forward algorithm (lines 1832-1843). Therefore the unqualified “PFs smooth the gradient” sentence is **wrong relative to the stated target**; at most it is a heuristic about averaging under a specified differentiable construction.

## M9

The correct taxonomy in survey §2 is:

1. Censored `max` with ΔU = 0: Case 1, continuous kink (lines 193-199).
2. Deterministic branch map with ΔU given by `eq:5b` and possibly nonzero: Case 2 (lines 201-224).
3. Genuine mixed support with positive mass for both regimes: Case 3 (lines 226-230).
4. Multiplicity without a selection law: Case 4 (lines 232-238).

Section §7.8 correctly calls continuous-kink Case 1 but incorrectly calls the value-jump geometry Case 3. Its cross-reference is **wrong relative to the stated target**.

## M10

The two formulas are identical under a reciprocal change of parameter:

\[
\ell+\frac1\alpha\log(1+e^{\alpha(u-\ell)})
 =\ell+a\log(1+e^{(u-\ell)/a}),\qquad \alpha=1/a.
\]

For the fixture’s code scales (a_d=1.5\times10^{-3}), (a_f=1.0\times10^{-3}), the survey-temperature values would be α_d (=666.666\ldots) and α_f (=1000). Yet survey `eq:77` and lines 2074-2079 call (1.5\times10^{-3}) and (1.0\times10^{-3}) “alpha” scales, while §7 `eq:ukf1` uses alpha as temperature. This is a convention collision.

For temperature α, (s''\leq\alpha/4); for code scale (a), (s''\leq1/(4a)). A smooth softplus has no gradient jump at all. A hard-max kink can have an (O(1)) one-sided gradient difference, but that is not (O(\alpha)) merely because a smoothing temperature is named. Thus “jump (O(\alpha))” and “curvature (O(\alpha))” are different claims; mathematics supports the curvature scaling, not a softplus jump. The survey’s scaling statements are **wrong relative to the stated target** without a convention correction.

## M11

The search confirms “Tran and Kleppe (2025)” occurs at survey lines 1270 and 1343 and in the plan, but no corresponding entry occurs in the survey bibliography (`zlb_discontinuous_hmc_survey.tex` references) or `references.bib`. The repository and `/home/ubuntu/google-drive-papers` search found no Tran/Kleppe paper or local PDF. I therefore cannot responsibly identify the actual authors, year, venue, and title, nor determine from the primary text whether the method handles ΔU=0, ΔU≠0, or both. The line 1343 characterization is unsupported. The `CLAUDE.md` local-copy requirement is also unmet: `docs/.localresources/` does not exist and the planned PDF was not obtained. Verdict: **unsupported**.

## M12

The survey explicitly says it could not obtain Pakman and Paninski’s full text (lines 1040-1044). The claim that exact Hamiltonian flow for truncated multivariate Gaussians supplies the cited orientation therefore lacks the required primary-source inspection. The plan also cites “Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussians” as reference 4 and assigns it reflection/boundary-handling support (lines 765-770). The central softplus derivative result does not require that paper, but any method claim relying on its theorem or algorithm does. Verdict: **unsupported**.

## M13

The material contradictions are:

1. §2 says a `max` is continuous with a possible gradient kink and no energy jump (lines 193-199), whereas §7 says the smooth softplus/UKF composition has a gradient jump (lines 1050-1059, 1107-1128).
2. §6 limits `eq:56a` to a fixed partition and excludes implicit or state-dependent branch computation (lines 991-1003), while §7 applies that jump formula to an unspecified UKF composition without identifying such a partition (lines 1130-1145).
3. §13 says the studied UKF route is a smooth-model posterior because softplus is C-infinity and has no discontinuity, and says HMC is exact for that UKF target (lines 2103-2115); §7 calls the same route a continuous-kink primary obstacle (lines 1107-1145).
4. §13 route (i) describes UKF-HMC as exact only for its approximate UKF functional and requiring distance diagnostics against the model likelihood (lines 2651-2658), while §7 presents variance collapse and DPF smoothing as an established pathology and remedy (lines 1093-1113, 1237-1258).
5. §13 route (iv) states that the hard C1 target is a filter-free continuous piecewise-quadratic kink for which ordinary joint HMC is valid (lines 2674-2681); §7 instead treats a UKF replacement as the encountered two-country geometry (lines 1060-1062).
6. The DPF section says finite genealogy scores carry Monte Carlo noise and ordinary HMC exactness does not apply (lines 1819-1830), and OT changes the finite forward law and needs a full extended-target construction (lines 1832-1843). §7 says LEDH diversity “bounds” gradient variation and makes HMC geometrically tractable without stating those qualifications (lines 1255-1258).

## M14

The `sec:shadow` label is real at survey lines 2012-2013. The survey also narrates an “implementation we studied” using a square-root UKF at lines 2103-2115. However, the audited repository package contains only the hardbound DNS, joint-target, reference, and HMC files listed by `find`; no UKF module or UKF result artifact is present there. Consequently §7 is not backed by a measured repository phenomenon. It is a survey-level assertion about an unprovided route, not demonstrated evidence. Verdict: **unsupported**.

## M15

The governing program is explicitly Program A: filter-free joint HMC on `mf_c1_k40_hardmax` (master lines 17-20). Its binding non-goals prohibit particle likelihood machinery (lines 29-34), NK/OBC solution work, cell/root-aware layers (lines 35-38), NeuTra/neural transport and production/empirical claims (lines 39-40), and event-aware/reflection-refraction HMC (lines 41-43). The plan instead targets a UKF marginal likelihood and S1/softplus diagnosis (plan lines 6-23), proposes Tran-Kleppe event-aware HMC (lines 142-222), proposes production integration and a NeuTra change (lines 613-638), and even lists particle filtering as a fallback in its risk register (line 656). Its NAWM II target and production-candidate selection also exceed the master’s named fixture scope and non-claims. The plan is not executable under current governance; it requires a new, explicitly approved program.

## M16

The plan calls G2_3 a UKF-filtered case with “2 states near ZLB” (lines 104-110). `model_tf.py:16-18` defines targets `mf_s1_k40_softplus` and `mf_c1_k40_hardmax`; the hardbound package has no UKF. The G2.3 manifest identifies `target_id="mf_c1_k40_hardmax"`, horizon 40, and a joint HMC run (result lines 205-220). The 8-state fixture is two DNS triples plus two basis states, not a two-state UKF filter. The premise is **wrong relative to the stated target**.

## M17

The executed ladder’s worst coordinate is theta8, a log FX-noise scale; the failures are non-monotone in trajectory length and the result offers two competing explanations: a non-centred funnel between log noise scales and latent innovations, or trajectory-length periodicity at healthy acceptance (result lines 73-88). The result expressly says neither is established and records no UKF covariance, softplus-boundary, gradient-jump, or crossing telemetry (lines 179-193). The smallest discriminating diagnostic is paired: (1) evaluate one-sided finite-difference/autodiff gradients and value continuity across the suspected bound while recording crossing acceptance and energy error; (2) conditionally probe theta8 against its `eta_raw` innovation scale and repeat after a non-centred/re-scaled chart. A localized value-continuous gradient jump supports kink geometry; smooth derivatives with scale-dependent curvature and reparameterization improvement support a funnel; neither plus ordinary epsilon/mass improvement supports tuning failure. No such diagnostic has executed, so the plan’s problem premise is **unsupported**.

## M18

The plan has these governance defects and omissions:

- Phase 4 names `phase4_surrogate_trained_model.pt` (plan lines 502-505), a PyTorch artifact, while the repository backend rule requires TensorFlow/TFP and requires a reviewed exception for PyTorch. No exception is supplied.
- Phase 4 trains a network but records no TensorFlow/TFP batch-native target backend, device placement, XLA status, or sample-wise-loop status. Batch size 256 is greater than one, but that number alone does not satisfy the batch-native training rule.
- Serious GPU phases do not specify pre-import `TF_FORCE_GPU_ALLOW_GROWTH`, repository memory-policy verification, visible-device values, logical-device limits, or failure-closed behavior. The plan’s PNG/NPZ/Markdown artifact list is not a GPU run manifest.
- No per-run structured manifest records exact command, environment, seeds, hardware, wall time, XLA/TF32, memory policy, or artifact hashes for Phase 4 or Phase 5.
- A decision-criteria table is present inline at plan lines 589-598, but no executed decision artifact exists until the proposed phase runs. This is not evidence of a completed decision.
- The required inference-status table is absent from the plan and its Phase 4/5 artifact lists.
- The required pre-mortem is absent. The risk register at lines 646-674 is not an evidence-backed pre-mortem with falsifiers, stop conditions, and ownership.
- The proposed NeuTra integration (lines 613-616) conflicts with the master’s no-NeuTra scope and would also require route-ledger classification.

These are material violations or missing mandatory artifacts. Verdict: **wrong relative to the stated target**.

## M19

The plan’s arithmetic is internally (200\times3600/20{,}000=36) seconds per evaluation (plan lines 458 and 548), but it is not defensible from the available evidence. The ladder is a CPU-only joint hard-max HMC run: its five rungs take 70.9, 91.6, 127.2, 190.8, and 331.7 seconds (result lines 29-35), and the two gate runs take 379 and 778 seconds (lines 92-145). None is a UKF marginal-likelihood evaluation, and no UKF timing or GPU manifest exists. Phase 4 consumes 800 of the 1,000 planned GPU-hours (80 percent) despite being the least validated phase and despite the central diagnosis being untested. That allocation may be a future budget choice, but this audit has no measured cost model or scientific evidence that makes it defensible. Verdict: **unsupported**.

## M20

The survey’s §7.7 item 3 explicitly proposes a surrogate force for proposals while evaluating the true UKF target for Metropolis acceptance (lines 1276-1281). That construction can preserve the true target: the surrogate changes proposal efficiency only. Plan Phase 4.4 instead sets the sampling target to `log p_hat(y|theta)` and compares its posterior to a reference (lines 511-531). That is ordinary HMC on an approximate target; Metropolis acceptance against the same surrogate does not correct to the true posterior. The methods are different, and the plan silently substitutes a potentially biased method for the exact corrected-proposal method. Verdict: **wrong relative to the stated target**.

## M21

The ladder defines and reports minimum ESS per gradient evaluation; its selected (L=32) rung has (4.510\times10^{-4}) (result lines 29-40). The plan requires ESS/grad (>0.1) and later (>1.0) (plan lines 22, 239, 525). The first is approximately (0.1/(4.510\times10^{-4})=222) times the measured baseline, not a modest threshold adjustment. The plan never defines whether gradients include warmup, rejected proposals, surrogate evaluations, or true-target correction evaluations, so cross-method comparisons are not commensurate. On the ladder’s definition the criterion is far beyond the observed baseline; every promotion gate using it is mis-set unless a new unit definition and calibration experiment are predeclared. Verdict: **wrong relative to the stated target**.

## M22

No `NAWM II` implementation, data, or model module was found in the repository search. The hardbound source fixes an 8-state fixture and the executed evidence uses (T=40); the master also records (T=200) only for an identification diagnostic, not a d=100 production model (master lines 66-107). The plan’s (T=120,d=100) and “NAWM II proxy” entries (lines 6 and 552-568) are therefore aspirational and cannot be reached by the named code without a new model program. Verdict: **unsupported**.

## M23

The factual ledger is:

1. **Executed under the master program:** Program A built and ran the filter-free, non-centred joint HMC route on the C1 hard-max target. The master defines that scope and its deliverables (lines 17-27); the G2.3 result records the hard-max target, commands, environment, seeds, and wall times (lines 205-220), with the final gate decision at lines 184-193.
2. **Executed for the UKF program:** the survey §7 and the proposed research-plan document were written. No UKF implementation, UKF result, event-aware implementation, variance-inflation run, or surrogate training artifact exists in the inspected paths. The survey’s statement that a UKF implementation was “studied” (lines 2103-2115) is not accompanied by a repository artifact.
3. **Assumed by the plan but absent:** a UKF-filtered G2_3, measured gradient-discontinuity surfaces, Tran-Kleppe paper/extraction, `event_aware_hmc.py`, Phase 1/2/3 diagnostic artifacts, the Phase 4 training data/checkpoint/validation, and the NAWM II proxy/scale run.

Verdict: **correct** as a ledger of what exists and what does not.

## M24

§2 already provides the four-way support/geometry taxonomy (lines 186-238), and §6 states both the fixed-partition kink proposition and `eq:56a` (lines 979-1046). The case study later states that its softplus UKF route is C-infinity with no discontinuity and is exact only for the UKF approximate target (lines 2103-2115), while its route table reserves the hard-max kink and filter-free joint HMC for C1 (lines 2651-2681). §7 adds no measured UKF result; instead its central claim contradicts those statements. Recommendation: retract the asserted softplus gradient-discontinuity mechanism, split §7 into a narrowly labeled smooth-UKF approximation/stiffness note and a separately sourced hard-branch/multiplicity note, and require measurement before any HMC-method prescription. Verdict: **wrong relative to the stated target**.

## M25

If a UKF variant is first built under a new program, the smallest useful diagnostic is a scalar, fixed-positive-(R), fixed-schedule UKF with the coded softplus observation. Sweep one parameter θ through a candidate margin and, for shrinking δ, record:

\[
|U(\theta+\delta)-U(\theta-\delta)|,
\quad |U'(\theta+\delta)-U'(\theta-\delta)|,
\quad P,H,S,
\]

then run a short reversible leapfrog stability sweep over ε and (a). Predictions are:

- smooth-softplus stiffness: value and derivative differences tend to zero; finite curvature grows as (1/a); stability follows ε (<2/\sqrt{\lambda_{\max}});
- hard-kink geometry: values match but one-sided derivatives tend to different limits and the gradient difference quotient diverges;
- funnel geometry: no localized value/gradient jump, but curvature and mixing correlate with a noise-scale/latent scale and improve after reparameterization;
- ordinary tuning failure: no localized pathology and convergence responds to mass/step-size tuning.

The master program’s lines 29-43 and 64-100 exclude UKF marginal likelihoods and event-aware work, and state that a marginal-likelihood program would be new. Therefore this experiment is not authorized under current Program A; it requires a new program approval. Verdict: **heuristic only**.

## Findings I missed

1. The survey uses alpha as a scale in `eq:77` and as an inverse scale in `eq:ukf1`; this changes the stated asymptotics unless explicitly converted.
2. The survey itself already says the UKF route is smooth and exact only for its approximate target, contradicting the central premise added in §7.
3. The taxonomy’s multiplicity derivation says selection, not regime switching alone, creates the discontinuity; §7.4 omits that condition.
4. The G2.3 result’s strongest evidence points to a log-noise-scale bottleneck and explicitly withholds a funnel conclusion; the UKF plan treats those runs as if they diagnosed a different geometry.
5. The plan’s Phase 4 uses a PyTorch checkpoint and lacks the repository’s required evidence artifacts for a serious GPU training campaign.

## Completeness self-audit

Questions answered with anchor + verdict: 25 / 25
Verdicts by class: correct 1 | wrong 17 | unsupported 6 | not checked 0 | heuristic only 1
Derivations written out in full (M1, M3, M4, M5, M6 minimum): 5 / 5
Files opened: `docs/plans/ukf-discontinuous-gradient-hmc-codex-audit-request-2026-09-02.md`; `docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex`; `docs/surveys/zlb_discontinuous_hmc/references.bib`; `docs/surveys/zlb_discontinuous_hmc/source_support.md`; `docs/surveys/zlb_discontinuous_hmc/omitted_papers.md`; `docs/plans/ukf-discontinuous-gradient-hmc-research-plan-2026-08-26.md`; `docs/plans/hardbound-kink-hmc-master-program-2026-08-21.md`; `docs/plans/hardbound-g2-3-leapfrog-ladder-result-2026-09-01.md`; `bayesfilter/hardbound/dns_curve_tf.py`; `bayesfilter/hardbound/model_tf.py`; `bayesfilter/hardbound/joint_target_tf.py`; `bayesfilter/hardbound/hmc_runner.py`; `bayesfilter/hardbound/dense_mass_matrix_adaptation.py`; `bayesfilter/hardbound/windowed_dense_mass_adaptation.py`
Commands run: `sed`; `nl -ba`; `rg`; `find`; `wc`; `git diff --check`; `git status --short`; attempted `curl` Crossref query (DNS unavailable)
Papers read (not just located): none in this audit; no Tran-Kleppe or Pakman-Paninski full text was available, so their method-specific claims remain unsupported
Questions I could not answer and why: M11 (the actual Tran-Kleppe bibliographic record and method cannot be verified without a primary paper); M12 (Pakman-Paninski method details cannot be independently checked without the full text)
Claims in this reply I could not anchor: none
