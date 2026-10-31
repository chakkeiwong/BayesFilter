# Testing surrogate HMC across BayesFilter models

Current precision/materiality directive, 2026-09-24: FP32 is the force default
for speed and memory economy; explicit FP64 reference/derivative roles remain.
The M13 saved-run discrepancy is an explanatory warning, not a campaign
stopper. See the [implemented policy](surrogate-hmc-fp32-warning-policy-2026-09-24.md)
and [active closure program](surrogate-hmc-closure-master-program-2026-09-22.md).
Required numerical validity and downstream posterior criteria remain in force.

Master program, 2026-09-18. Source baseline: `8a4cffa2fd085f43ab5710c24d6e7b48770aea0d`
in `/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917`.

**Goal: test whether surrogate HMC driven by an imperfect DCTR force can
produce reliable parameter inference, and at what computational cost, across
the concrete models in this repository.** DCTR means the full LEDH-PFPF-OT
algorithm with GenUT, dual-cap trust-region correction, the UKF per-particle
covariance lifecycle, Contract E–Chol reset, and analytical recursive scores.
The force may disagree with the gradient of the density used for acceptance.
That disagreement is part of the experiment, not an automatic reason to stop.

This is an active research and implementation program, reviewed locally in the
[audit](surrogate-hmc-all-models-master-review-2026-09-18.md). It is not a claim
that the all-model runner, references, adapters, or results already exist.
The [Opus reply](surrogate-hmc-all-models-opus-reply-2026-09-18.md) withdraws
the incorrect force/endpoint matching requirement and confirms the design.
The owner instructed revision, thorough review and execution on 2026-09-18.
That instruction authorizes the bounded campaign below, including its two
additional LG1 inference attempts. Readiness checks remain scientific work;
they are not requests for renewed campaign permission.

Judgment revision, 2026-09-22: use the
[criteria revision](surrogate-hmc-judgement-criteria-2026-09-22.md) for new scope
design and reporting. Separate implementation validity, sampling health,
practical accuracy, formal confirmation and cost. Declare whether a run is a
bounded pilot or a confirmation, justify tolerable errors in the model's own
quantities, and budget for that claim. Practical acceptance can close a pilot
without converting an inconclusive formal comparison into equivalence.
Archived results and executable thresholds remain unchanged; the new reporting
distinction is a plan requirement, not a claim of an implemented runtime change.

LG1 pilot closeout, 2026-09-22: after reviewing the parameter differences,
acceptance rates and R-hat, the owner considers this case satisfactory. Close
LG1 as an accepted practical pilot and move the campaign to the remaining
model scopes. The surrogate-force/exact-endpoint arm has mean acceptance
73.33%, maximum R-hat 1.00148 and zero recorded divergences. The fresh
[likelihood diagnostic](surrogate-hmc-lg1-likelihood-result-2026-09-22.md)
finds a grid-MLE shift of 0.03340 (0.090 posterior SD) and only 0.2793%
exact-likelihood loss at the DCTR MLE. No further LG1-specific likelihood repair
or retained-chain extension is planned. Preserve the archived D/R discrepancy
under its original margin and the inconclusive sampler-equivalence verdicts;
practical pilot acceptance does not retrospectively change those tests or
establish an efficiency advantage. The shared uncertainty-diagnostic defect
still informs the design of future scopes; it does not reopen this LG1 pilot.

Current direction, 2026-09-22: the owner requests an explanation of the
algorithm, code and literature before further Claude review. The
[explanatory audit](surrogate-hmc-algorithm-code-literature-audit-2026-09-22.md)
separates proposal correctness, uncertainty calibration, finite-likelihood
accuracy and model representation. Before another retained inference scope,
prospectively qualify uncertainty for the actual functions/comparisons and
justify joint precision, trajectory lengths, cheap baselines and total cost.
The existing IID false-veto defect and precision calculation are not resolved
by a larger run budget alone. Closed results remain unchanged. This is the
current next action; dated execution-status paragraphs below preserve history.

The [remaining-gap inventory](surrogate-hmc-remaining-gaps-2026-09-22.md)
records the shared repairs, actual executable coverage, each family's remaining
work and the current budget. The six supported CLI scopes do not constitute
completed coverage of the fourteen model and three control families.

AR(2) direction clarified by the owner, 2026-09-22: use
[Chapter 18b's structural deterministic formulation](../chapters/ch18b_structural_deterministic_dynamics.tex),
with one genuine innovation and exact lag completion. Reuse the existing
TensorFlow structural interface; the canonical DCTR consumer still needs its
integration measure, flow/weights and analytical derivatives adapted. An SVD
swap alone is insufficient, and adding lag noise changes the target. This
adapter work can proceed independently of the uncertainty repair; new retained
inference still needs the qualification above. Terminology: B/R tests a DCTR
surrogate force against a reference endpoint. C/D is ordinary HMC for the finite
posterior if its field equals that endpoint's exact gradient; otherwise it also
uses a surrogate force. The explanation audit records the precise distinction.

KSC T2 has now completed an actual surrogate-HMC test: independent public
tuning followed by 2,000 retained draws per chain for both the exact-force and
full canonical DCTR-force arms. A/R and B/R remain INCONCLUSIVE; B's final
Monte Carlo uncertainty screen failed three functionals despite all retained
chunk health checks passing. Independent C tuning and 2,000 retained draws per
chain also completed: C/D and C/R are INCONCLUSIVE after the beta median-CDF
uncertainty screen failed. Independent finite-target quadrature gives D/R
EQUIVALENT under the declared margins and empirical numerical allowances.
See the [B result](surrogate-hmc-ksc-t2-result-2026-09-21.md),
[C and D result](surrogate-hmc-ksc-finite-result-2026-09-22.md), and corrected
[all-model report](artifacts/surrogate-hmc-all-models-20260922-ksc-t2-master-03/decision.json).
The reporting-only P0 stage now preserves completed LG1, C01, KSC, M10 and SV20 outcomes;
closed attempts never authorize new sampling. This is partial execution of the
master program, and all-model validation remains incomplete.

M10's full canonical DCTR consumer passed all15 declared GPU mechanics checks,
including direct shared-callable parity and proposal reversal. The master stage
`--scope M10 --stage P3-M10-qualify` reuses that checked evidence. Its independent
local reference and consumer required further endpoint qualification before
public HMC tuning; see
[the mechanics result](surrogate-hmc-nonlinear-consumer-result-2026-09-20.md).
The wider M10 uniform-grid reference screen then failed52/116 tail comparisons.
A composite quadrature of the same likelihood, retaining both observation
roots, passed all148 repair probes; see the
[repair result](surrogate-hmc-nonlinear-quadrature-repair-result-2026-09-21.md).
The repaired nominal posterior reference subsequently passed all six
qualification checks, including normalization, parameter tails, interpolation
and all thirteen function-precision allowances. The public
`--scope M10 --stage P3-M10-posterior` command reuses the source/data/grid-checked
result. All57 focused master regressions pass. See the
[posterior result](surrogate-hmc-nonlinear-posterior-result-2026-09-21.md).
The scoped protection ladder subsequently selected0.1 with healthy
nominal/stress rows. Its initial cost forecast exceeded the remaining family
allocation, so inference was deferred at that stage. The public
`--scope M10 --stage P3-M10-controls` command now verifies and reuses that
evidence;25 focused tests and actual checked reuse passed. See the
[control/cost result](surrogate-hmc-nonlinear-controls-result-2026-09-21.md).

On September22, the M10 log-domain endpoint passed all16 qualification checks,
including complete transitions, independent force/endpoint bindings and
reversal; see the [endpoint result](surrogate-hmc-m10-endpoint-result-2026-09-22.md).
The section12 redistribution funds the first B/R inference attempt under the
[frozen inference plan](surrogate-hmc-m10-inference-2026-09-22.md).
The public `--scope M10 --stage P3-M10-inference` route completed public tuning
and2,000 retained draws per chain on GPU1 in
`surrogate-hmc-all-models-20260922-m10-inference-02`.
The first master launch stopped before creating a worker; the discovery bug
was repaired, preserved and charged without consuming a statistical look.
There are59 passing focused checks after the repair. All eight retained
chunks passed health checks. B/R is INCONCLUSIVE: the first look had valid
but insufficient precision; the final log-Q median-CDF uncertainty screen
failed. C/D, D/R and C/R remain NOT_RUN. See the
[M10 inference result](surrogate-hmc-m10-inference-result-2026-09-22.md).
SV20 subsequently passed all17 claim-endpoint checks; see the
[endpoint result](surrogate-hmc-sv20-endpoint-result-2026-09-22.md).
The measured complete cost supports one L1-only B/R attempt under the
[SV20 inference plan](surrogate-hmc-sv20-inference-2026-09-22.md) and section12
redistribution. This was an imposed `max_leapfrog_steps=1` restriction, not a
selection over longer trajectories. It tests a corrected Gaussian drift
proposal and provides no evidence about sustained HMC trajectories or their
cost advantage. The public `--scope SV20 --stage P3-SV20-inference` route uses
the artifact-authority tuner, independently retained chunks, physical-parameter
functionals and the qualified numerical-reference allowances. All57 distinct
focused checks pass. A retained-diagnostics XLA shape failure was repaired
before inference; the fresh GPU check matches an independent batch exactly.
The run completed 2,000 retained draws per chain with eight healthy chunks.
B/R is INCONCLUSIVE: both frozen looks fail functional MCSE qualification.
The independent telemetry audit found zero original-force fallbacks. The
worker now enforces that scope condition before posterior analysis, with
49 passing focused regressions. C/D, D/R and C/R remain NOT_RUN. See the
[SV20 result](surrogate-hmc-sv20-inference-result-2026-09-22.md).

LG2 then passed all 16 claim-endpoint and complete-cost qualification checks
through `--stage P3-LG2-endpoint --scope LG2`. Its measured 2,000-draw B/R
pilot forecast is 15,327.99 CPU-worker / 16,120.32 GPU-wall seconds, exceeding
the remaining LG2 allocation of 1,024.00 / 6,665.04 seconds. Inference remains
NOT_RUN. The forecast also prices a pilot whose joint-equivalence precision
has not been justified. See the [LG2 result](surrogate-hmc-lg2-endpoint-result-2026-09-22.md)
and [reporting review](surrogate-hmc-master-reporting-review-2026-09-22.md).

Execution status (2026-09-20): LG1 attempt4 has completed independently tuned
exact-force A and imperfect-DCTR-force B chains against the exact endpoint.
Both are INCONCLUSIVE after the two planned looks; their health and mixing
screens pass, but not every equivalence interval fits its frozen margin.
See the [retained result](surrogate-hmc-lg1-inference-result-2026-09-20.md).
The public verified TensorFlow tuner and retained runner are implemented and
GPU qualified; old diagnostic configurations remain ineligible. The new named
finite DCTR endpoint and its independent quadrature D are qualified. D/R fails
the median-CDF accuracy margin. That finding does not stop the C/D sampler test:
separate C tuning passed, and retained C completed both planned looks under the
[finite comparisons plan](surrogate-hmc-lg1-finite-comparisons-2026-09-20.md).
C/D is INCONCLUSIVE with five of six functions equivalent and healthy chains;
C/R is also INCONCLUSIVE. See the [finite result](surrogate-hmc-lg1-finite-result-2026-09-20.md).
LG2 now has frozen independent data, a qualified model-posterior reference and
an actual full consumer that passed fourteen GPU checks. The master exposes
`P3-LG2-reference`, `P3-LG2-qualify` and `P3-LG2-controls` with independent
prerequisites and LG2 scope reporting. Its calibration selected multiplier 0.01,
but the measured optimistic B-only forecast exceeds the initial allocation
(4.035 GPU hours and 3.744 CPU-worker hours). LG2 inference is deferred; see
the [calibration result](surrogate-hmc-lg2-controls-result-2026-09-20.md).
The master now reports closed LG1 results without rerunning their sampling or
preparation. Historical reports cannot authorize new inference. Partial prior
looks also prohibit restarting the same attempt from scratch.
SV20 now has qualified independent likelihood and full-prior posterior
references, with all thirteen function-refinement checks passing. Its actual
full DCTR consumer passes fourteen GPU checks, and its own protection
calibration selects multiplier0.1. At that stage, even the optimistic one-arm
cost exceeded the initial family allocation and SV20 HMC was deferred; the
later redistribution and completed run are recorded above. See the
[SV20 control result](surrogate-hmc-sv20-controls-result-2026-09-20.md).
Other model consumers and heuristic comparisons remain incomplete.
C01 Gaussian mechanism controls have also completed: exact, half-strength and
rotational fields reached both retained looks but are INCONCLUSIVE; zero force
failed fresh tuning R-hat. Some MCSE checks failed even for exact force. All 36
GPU mechanics checks passed. This supplies no DCTR model inference evidence;
see the [control result](surrogate-hmc-gaussian-controls-result-2026-09-20.md).
The master preserves closed C01 results without drawing additional samples.
Independent calibration then found that the MCSE absolute batch-lag rule vetoed
454/512 exact IID controls. It is poorly calibrated for this schedule; see the
[diagnostic result](surrogate-hmc-mcse-diagnostic-result-2026-09-20.md).
Closed outcomes stay fixed. Model preparation can continue while prospective
uncertainty calibration remains required before interpreting new inference.
The first joint-randomization candidate also failed its full calibration:
52/512 IID replications were vetoed, and the rho 0.9 coverage check had only
26 admitted replications. See the [result](surrogate-hmc-mcse-randomization-result-2026-09-20.md).
A [prospective diagnostic repair](surrogate-hmc-mcse-binary-calibration-2026-09-20.md)
addresses the uninformative fold of balanced binary CDF summaries and provides
longer correlated simulations. It passed all eight declared screens on 2,048
fresh replications; see the [result](surrogate-hmc-mcse-binary-calibration-result-2026-09-20.md).
This supports only the tested Gaussian diagnostic scope. It changes neither
closed results nor runtime diagnostic defaults. Its 336.902 CPU seconds,
including tests, fit the existing shared repair reserve.
The SV20 branch now includes `P3-SV20-reference`, `P3-SV20-posterior`,
`P3-SV20-qualify` and `P3-SV20-controls`, with cumulative family-budget checks.
Reference, consumer and control qualification remain separate from retained
HMC inference. Their successful preparation cannot promote the model result.
See the [execution contract](surrogate-hmc-all-models-execution-2026-09-19.md),
[execution result](surrogate-hmc-all-models-execution-result-2026-09-19.md), and
[current checkpoint](artifacts/surrogate-hmc-all-models-20260919-review-tests/checkpoint.md).

For future execution, this program replaces the research ordering and scope of
the September 4/7 surrogate-HMC masters and the September 17 graceful-failure
program. Preserve their files and results. Keep useful numerical checks, but
do not reinstate their filtering-accuracy or exact-score requirements as
prerequisites for testing an imperfect proposal force. Existing expenditures,
failed attempts, owner exclusions, and repository policies remain in force.
The owner's execution instruction covers this campaign allocation; no approval
token or per-retry ceremony is required. Section 12 explicitly adds LG1 attempts rather
than pretending the old allowance has not been used.

## 1. The question and the evidence that can answer it

There are two distinct possible failures. An imperfect force may propose poor
trajectories even when acceptance uses a correct model likelihood. Separately,
a finite DCTR likelihood may define a posterior that differs materially from
the model posterior. A sampler can correctly sample that finite posterior and
still give inaccurate model inference. Conversely, inaccurate filtering moments
or a biased force can coexist with satisfactory endpoint-corrected inference.
The experiments below distinguish these cases directly.

| Research intent | Declaration |
|---|---|
| Main question | Can an imperfect full-DCTR force support reliable parameter inference through surrogate HMC on each declared model scope? |
| Mechanism | A frozen position-only DCTR field drives reversible, volume-preserving proposals; a separately bound endpoint potential and full kinetic energy determine acceptance. |
| Expected failures | Poor mixing, high cost, mode trapping, numerical invalidity, or a biased finite likelihood. These need different repairs. |
| Primary criterion | Accuracy of retained inference against an independently qualified reference for the **same declared posterior**, at the stage declared before launch. A practical pilot needs direct reference evidence, uncertainty and an explicit scope-specific judgment; formal confirmation requires MCSE-aware equivalence under section 7. Report those decisions separately. |
| Promotion veto | Invalid target/map or unresolved material mixing/accuracy concerns block posterior acceptance. Unresolved reference/MC precision blocks the affected precision claim; failed equivalence blocks formal confirmation. Demonstrated conditional heuristic dominance blocks practical usefulness. These veto the affected claim, not every later experiment. |
| Continuation veto | Broken shared kernel/target definition, unusable reference for the requested comparison, corrupted evidence, an exhausted allocation, or a changed boundary requiring owner direction. The veto follows the dependency that failed. |
| Repair trigger | Failed candidate, poor force geometry, endpoint approximation bias, failed numerical control, insufficient sampling, or excessive measured cost; use section 10. |
| Explanatory diagnostics | Force discrepancy, force angles/norms, filtering moments, acceptance, finite energy-error distributions, fallback frequency, loss/residual metrics, and unqualified timing differences. |
| Unsupported conclusions | Force accuracy is necessary; passing a toy proves model inference; sampling the DCTR posterior proves model-posterior accuracy; one dataset proves calibrated estimation; or one failed family rejects surrogate HMC everywhere. |

The program has three reportable outcomes per scope: **sampling accuracy for
the declared endpoint**, **accuracy relative to the model posterior**, and
**practical efficiency**. Keep these separate. Valid but slow inference is a
possible outcome. All-model success means all required model tests passed;
a report listing blocked or untested models is an incomplete validation,
even if its accounting is complete.

## 2. Checked starting point and limitations of earlier evidence

The [recovery report](ledh-surrogate-hmc-recovery-result-2026-09-18.md) preserves
the original plans, commands, source checks, and results. Its findings are the
starting evidence, not new posterior results:

* The recovered Gaussian toy coupled chains in the potential energy while
  accepting them independently. Its reported Wasserstein calculation was also
  wrong. Its PASS cannot establish surrogate-HMC inference.
* Five focused shared-kernel mechanics checks passed in a CPU-only debugging
  run. They do not test the complete DCTR consumer, GPU/XLA execution, or
  model-posterior agreement.
* The old five-coordinate LGSSM consumer supplied likelihood alone and left
  two coordinates inactive. Its target was not the claimed proper parameter
  posterior, and it used reduced corrections. It is not an eligible baseline.
* Three later LG1 attempts failed a filtering-calibration screen. They produced
  no retained surrogate-HMC inference. A matched before/after check did not
  find a material containment regression in that particular case.

No recovered result currently establishes successful full-DCTR surrogate-HMC
parameter inference for a repository model. Other historical HMC and filtering
studies are source/discovery leads, not automatic validation of this program.
Do not reuse pre-2026-08-21 LEDH results or prohibited transport-chunk results
as baselines, tuning evidence, or warm starts. A model definition can be
inspected as source without importing an invalidated algorithmic result.

## 3. Precisely separate posterior, endpoint, and proposal force

Let `theta` be the inferred parameters, `y` the fixed observations, and
`theta = b(z)` an explicitly declared one-to-one chart. Let `J_b` denote its
Jacobian. All unspecified model parameters are fixed in the scope card.

The model posterior in HMC coordinates has unnormalized log density

```text
log pi_M(z | y) = log p(b(z)) + log L_M(b(z); y) + log |det J_b(z)|.
```

With particle count `N`, frozen random inputs `omega`, and fixed value controls
`c_v`, the finite-DCTR posterior is

```text
log pi_D(z | y, omega, c_v)
  = log p(b(z)) + ell_DCTR(b(z); y, omega, c_v) + log |det J_b(z)|.
```

An exact or numerically qualified reference likelihood `L_R` gives endpoint
`U_R = -log pi_R`; `U_D = -log pi_D`. Use `pi_R = pi_M` only for an exact
likelihood identity. A numerically qualified approximation retains its own
`pi_R` notation and an explicit error allowance; empirical agreement does not
make the two densities equal. “Exact endpoint” elsewhere means the fixed density actually
evaluated, not an assertion that every finite likelihood is the model likelihood.
Record normalizability, prior support, observation timing, initialization law,
and every parameter dependence, including stationary initial covariance.

If the analytical DCTR likelihood score returned in physical coordinates is
`s_tilde_D(theta)`, the standard experimental proposal field is

```text
g_D(z) = - J_b(z)^T s_tilde_D(b(z))
         - grad_z log p(b(z)) - grad_z log |det J_b(z)|.
```

The HMC force interface consumes this **potential-gradient-sign** field.
It need not equal `grad U_R` or even `grad U_D`. If an adapter already returns
transformed posterior terms, do not add them twice. Verify sign, coordinates,
shape, dtype and parameter dependence at the actual consumer boundary.
Canonical analytical recursion remains the DCTR computation under test; autodiff
or finite differences may check it but cannot silently replace it.

The value and force may use independently frozen numerical controls, with both
identities recorded. A numerics-altering damping/capping candidate is a distinct
force arm. Hold `U_D` fixed when comparing forces; otherwise a changed endpoint
confounds the force comparison. Never select a convenient `omega` because its
posterior happens to agree with the reference. Each bank defines its own finite
target; concatenating draws across banks is not sampling any one of those targets.

Write the separate frozen identities as `U_v(z;omega_v,c_v)` and
`g_f(z;omega_f,c_f)`. Every current, cached, proposed and replayed endpoint
evaluation uses the same endpoint binding throughout retained sampling.
Every forward/reverse force call uses its own frozen force binding, which may
differ from the endpoint binding. Force evaluation must not mutate endpoint
inputs or its cache. Neither `omega_f=omega_v` nor `c_f=c_v` is required for
detailed balance. The proof below relies on the fixed proposal map and fixed
acceptance density independently.

### 3.1 Why an imperfect field can be used

For fixed positive-definite mass `M`, step `h`, leapfrog count `L`, and finite
deterministic position-only `g`, use the symmetric step

```text
p_half = p - (h/2) g(z)
z_new  = z + h M^{-1} p_half
p_new  = p_half - (h/2) g(z_new).
```

For each fixed position, a kick is a translation in momentum; for each fixed
momentum, a drift is a translation in position. These operations preserve
product volume, including for a measurable field with discontinuities. Their
inverses negate `h`. With momentum reversal `R`, the symmetric composition `S`
satisfies `R S R = S^{-1}`. Thus `R S^L` is an involution preserving volume.
For fresh `p ~ N(0,M)`, accept separately for chain `i` using

```text
K_i(p) = 0.5 p_i^T M^{-1} p_i
log alpha_i = min(0, U_i(z) + K_i(p) - U_i(z_new) - K_i(p_new)).
```

The target-weight ratio between two points paired by this involution gives
Metropolis detailed balance for the extended density proportional to
`exp(-U(z)-K(p))`. Integrating out momentum leaves the declared endpoint
posterior invariant. This argument does not use `g = grad U`. It establishes
ideal-arithmetic invariance, not irreducibility, rapid mixing, numerical
correctness, or finite-run convergence. Those are tested separately.

There must be no reduction over chains in `U_i` or `K_i`. Freeze the mass,
force inputs, step size and trajectory rule for retained sampling. Keep the
current endpoint cached without changing its random inputs or controls.

### 3.2 Numerical failure and support

The implementation must distinguish a true zero-density point, an endpoint
computation failure, a force failure, and proposal arithmetic overflow. A valid
reference endpoint does not become invalid because DCTR could not compute a
force there. A deterministic position-only finite extension, such as zero on
force-invalid positions, can retain the kick argument if reverse replay and
the executed status branches satisfy the stated map. Record its frequency and
location; high usage may explain why the experiment reduces to a weak proposal.

NaN masking, asymmetric early termination, or rejection of an arbitrary failed
trajectory is not itself a reversibility proof. Before accepting such a branch
as eligible, prove the failed-proposal set/rejection rule is reversible or show
that the branch does not fire in the claim-bearing run and qualification domain.
Any unqualified arithmetic branch firing vetoes that run's sampling claim.
Unsuccessful numerical attempts remain preserved evidence and consume budget.
The absence of observed failures supports empirical numerical qualification,
not proof that an unexamined branch is globally reversible. Preserve that
distinction in the result even when the empirical tests pass.

If a DCTR **endpoint** fails on otherwise valid model parameters, silently
mapping those points to `+infinity` in the potential changes support. Repair it,
or define a separately named restricted posterior and assess excluded model
mass against the section 7 error allowance using a qualified independent
reference. A restricted-target success alone cannot pass model inference.
If `A` is the retained support, write
`pi = (1-epsilon) pi(.|A) + epsilon pi(.|A^c)`, where `epsilon=pi(A^c)`.
The two conditional laws have disjoint support, so total variation between
`pi` and `pi(.|A)` is `epsilon`. This bounds event-probability error,
and bounds the expectation error of `f in [a,b]` by
`(b-a) pi(A^c)`. It does **not** control unbounded parameter means or second
moments: those need explicit excluded-tail moment bounds. Qualify these bounds
for the particular endpoint whose support was restricted; a model-reference
mass estimate does not automatically bound exclusions under `pi_D`.
True model clipping, censoring, atoms, and singular innovations must be retained
in the probability law; they cannot be replaced by a Gaussian density for ease
of implementation.

Do not put a freshly randomized particle likelihood into ordinary endpoint MH.
A particle-marginal reference is eligible only with a nonnegative unbiased
likelihood estimator and a correctly augmented state including its randomness.
Its extended density is proportional to `p(theta) L_hat(theta,u) m(u)`;
integrating `u` gives the desired likelihood when unbiasedness holds. The
current estimator/randomness must be retained on rejection. A frozen biased
particle approximation, log-likelihood averaging, or UKF/SGQF likelihood is not
an exact model-posterior reference.

## 4. The experiment that isolates the surrogate-HMC mechanism

Use the same data, priors, chart and physical summary definitions within a
comparison. Every arm receives its own valid tuning artifact. A fair comparison
does not mean blindly sharing one step size between different forces.

| Arm | Acceptance endpoint | Proposal | Question and requirement |
|---|---|---|---|
| A: conventional HMC | `U_R` | Verified matching `grad U_R` | Reference-target HMC comparator where a score is available. Must itself pass reference and mixing checks. |
| B: reference-endpoint surrogate HMC | `U_R` | Full imperfect `g_D` | Primary isolation of the force mechanism. Required when a qualified deterministic reference endpoint is feasible. |
| C: DCTR-endpoint surrogate HMC | `U_D` | Full imperfect `g_D` | Primary practical DCTR inference arm. Test both sampling of `pi_D` and agreement with `pi_M`. |
| D: independent finite-target reference | `U_D` | Quadrature, qualified random-walk/slice sampling, or verified matching-gradient HMC | Separates sampling error in C from approximation error of `pi_D`. It must be independently implemented at the sampler level and adequately converged. |
| E: bounded force repair | Same endpoint as B or C | Explicitly calibrated alternate full-DCTR controls or eligible finite field extension | Tests a declared repair after a failure. Preserve the unmodified arm; retune and use fresh evaluation randomness. |

If a deterministic reference likelihood is unavailable at full scale, C may
be assessed against an independently qualified model-posterior sample reference;
B is then reported unavailable at that scale. Run B on a smaller *same-law*
scope when possible, without claiming that it validates the larger one. If only
`pi_D` has a reference, the result answers finite-target sampling and leaves
model inference open. If D fails, C versus the model reference cannot identify
whether any discrepancy came from sampling or the finite likelihood.

A failed conventional-HMC comparator triggers investigation; it is a global
block only if evidence implicates the shared target or kernel. A difficult
target can defeat A while B or a non-HMC reference remains valid. Do not make
success of an optional exact-gradient computation a hidden requirement for an
imperfect-force method.

The baseline ladder is: cheap direct/gradient-free methods; a tuned conventional
method for the same endpoint where available; unmodified full-DCTR surrogate
HMC; then one explicitly identified repair. Neural-force fitting and NeuTra
transport training are not prerequisites or substitutes for this experiment.

Schedule B against the model reference first when it is ready. Qualification
of D and completion of C must not delay that independent mechanism test.
However, B alone leaves the practical DCTR-endpoint question open; the complete
scope still owes C/D and the model-posterior comparison. The stage budget may
produce an honest partial answer rather than a complete scope verdict.

Record four separate comparisons, with `R` denoting the qualified model-posterior
reference and D the qualified reference for the particular frozen `pi_D`:

| Comparison | Quantity tested | Interpretation of a supported discrepancy |
|---|---|---|
| B versus R | Reference-endpoint surrogate-HMC inference | A force, tuning, mixing or implementation problem remains after reference qualification. |
| C versus D | Sampling of the frozen DCTR posterior | The sampler has not reproduced its own declared endpoint posterior. |
| D versus R | Finite-likelihood effect on parameter inference | This frozen DCTR likelihood changes the tested posterior quantities. |
| C versus R | End-to-end parameter inference | The practical route fails the tested model-posterior criterion. |

Use the actual uncertainty of both sides in each comparison. Passing C versus
D and D versus R separately at margin `delta` permits a total difference of
`2*delta`; it does not establish C versus R at `delta`. Conversely, two errors
can cancel in C versus R. Perform all available comparisons directly and retain
their separate verdicts. When `U_R` is approximate, B first owes qualification
against its own endpoint law; its agreement with R also includes the declared
reference-approximation allowance.

## 5. Model coverage: fixed obligations, not a three-model convenience list

The [recovered inventory](ledh-surrogate-hmc-model-coverage-2026-09-18.md) is a
discovery aid. This audit also found range–bearing, structural AR1 quadratic
completion, banana and mixture targets outside that inventory. The table below
is the minimum coverage register. Each listed distinct law, inferred parameter
set, horizon and data regime becomes a separate executable scope card in P0.
State dimension is never a substitute for parameter dimension.

Use the following prior rule only where the repository supplies a filtering
fixture but no parameter posterior. **Proposed noise-scale inference scope:**
infer `lambda_Q, lambda_R` with independent `Normal(0, log(2)^2)` priors;
multiply existing innovation and observation scales by `exp(lambda_Q)` and
`exp(lambda_R)` respectively, or covariance matrices by their squares. Keep
other source coefficients and the actual initialization convention fixed.
This puts most prior mass within about a factor four of the source scales.
It is a new, declared inference exercise on that law, not an existing repository
posterior or a full-parameter test. Degenerate/non-Gaussian laws require the
equivalent explicitly derived scale parameterization; adding a ridge or noise
to make the law fit this rule is forbidden. Existing proper posterior priors
take precedence and must be copied with source anchors, not replaced.

| ID / family | Required concrete scopes and parameter task | Reference route and earliest missing prerequisite |
|---|---|---|
| M01 Linear Gaussian | LG1: `d=1,T=10`, infer `phi`; LG2: `d=2,T=20`; LG4: `d=4,T=50`; LG20: `d=20,T=120`, the latter three infer transition/process-scale coordinates from the [current fixtures](../../bayesfilter/highdim/ledh_program_fixtures_tf.py). Also diagonal and triangular/all-parameter targets, including the 18-parameter `d=4` [triangular target](../../bayesfilter/testing/multidim_triangular_lgssm_tf.py), and [score-study](../../bayesfilter/score_study/registry.py) Gaussian-all-parameters. Map constant-velocity fixtures explicitly. | Exact Kalman likelihood; low-dimensional posterior quadrature; separately qualified exact-target sampling in higher parameter dimension. Full-DCTR inference wiring/tuning is missing. Restricted LG1 cannot complete all linear scopes. |
| M02 Actual SV | Synthetic and available real-data instances; SV20/SV40 horizons; two-parameter [actual-SV target](../../bayesfilter/testing/zhao_cui_actual_sv_target_tf.py); C2 and TT instances mapped by their actual likelihood, initialization and parameterization. | Small scalar-state grid/domain refinement; independent particle-marginal or joint-state reference for larger scopes. Keep observed-return likelihood and data transformations explicit. Real-data version/window and DCTR posterior adapter needed. |
| M03 Transformed SV | [Exact transformed SV and KSC mixture-transformed SV](../../bayesfilter/highdim/sv_mixture_cut4.py) are separate observation laws. Preserve the existing two-parameter prior/chart if applicable; otherwise use the declared noise-scale task after derivation. | Same-law reference for each; separately quantify KSC approximation to actual SV. UKF and Gaussian-sum UKF are approximation comparators. No unexplained expansion of “KSC” and no transfer of their admission status to DCTR. |
| M04 Generalized SV | [Prior-mean projected model](../../bayesfilter/highdim/models.py) and [native two-state/five-parameter model](../../bayesfilter/highdim/native_generalized_sv.py) are separate scopes. Preserve native parameters/prior when defined; otherwise create and label a noise-scale scope. | Native dense calculations use TensorFlow; Legendre-node construction uses NumPy. Qualify the complete reference and its eligible runtime, including the full parameter call chain. A projected likelihood cannot certify the native posterior. |
| M05 Predator–prey | Six parameters `(r,K,a,s,u,v)` from [posterior source](../../bayesfilter/testing/predator_prey_ukf_neutra_target_tf.py); restricted PP20 `(r,v)` scope; [common-suite RK4](../../experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py) instances mapped separately where timing/law differs. | Small same-law quadrature/particle reference; larger qualified particle-marginal reference. UKF/SGQF are comparators. Keep initial-observation timing and positivity/clipping law. Full six-parameter DCTR scope and reference needed. |
| M06 SIR | [Spatial physical-clipped](../../bayesfilter/highdim/models.py), Austria 18-state/three-parameter `(kappa,nu,observation_sd_scale)` [posterior](../../bayesfilter/testing/sir_filter_neutra_target_design_tf.py), and [latent-preclip](../../bayesfilter/highdim/sir_latent_preclip_tf.py) are distinct laws. The six-state SIR3 filtering-only fixture needs a declared parameter task; it cannot count as completed HMC. | Same-law reduced reference, then qualified full-law reference. Clipping atoms, bounds and spatial coupling are obligations. Latent-preclip is an extension, not automatically canonical or source-faithful. SIR-UKF remains owner-excluded. |
| M07 Coupled nonlinear Gaussian | [Replicated coupled blocks](../../bayesfilter/highdim/zhao_cui_coupled_nonlinear.py); start with the declared noise-scale task, then existing named parameter targets if discovered. Record block number as a separate scale scope. | Conditional Gaussian proposals are not parameter-posterior oracles. Need a posterior adapter and independent marginal reference. |
| M08 Structural | [Affine/AR2 and nonlinear worked fixtures](../../bayesfilter/testing/structural_fixtures.py); five-parameter `(rho,sigma,phi,gamma,R)` [structural posterior](../../bayesfilter/testing/structural_ukf_neutra_target_design_tf.py); [AR1 quadratic-completion](../../experiments/dpf_implementation/tf_tfp/fixtures/structural_ar1_quadratic_tf.py) with the declared noise-scale task. Preserve deterministic state completion. | Kalman for genuinely affine laws; independent latent-innovation integration or particle reference for nonlinear laws. UKF is a comparator. Do not merge quadratic completion with a different nonlinear accumulation law merely because both have two states. |
| M09 Nonlinear accumulation and growth | [Model B and Model C](../../bayesfilter/testing/nonlinear_models_tf.py); use existing proper target parameters from the [generic adapter](../../bayesfilter/testing/simple_nonlinear_generic_target_adapter_tf.py), otherwise the declared noise-scale task. Model A maps explicitly to M01/M08. | Refined latent-innovation/state integration for tiny cases, then independent posterior sampling. One-step Gaussian moment projection is not a model-likelihood reference. |
| M10 Nonlinear score-study model | [Nonlinear scalar provider](../../bayesfilter/score_study/nonlinear_tf.py:1): sine transition and linear-plus-quadratic observation. Distinct from both M09 laws; see the [scope and reference plan](surrogate-hmc-nonlinear-reference-2026-09-20.md). Retain a separate noise-scale inference obligation. | Checked state-mesh/domain reference plus a proper parameter prior/chart and full DCTR posterior consumer. A score normalizer alone is not a parameter posterior. |
| M11 SSL-LSTM | [Fixed posterior](../../bayesfilter/nonlinear/ssl_lstm_posterior_tf.py) and [complexity-dependent posterior](../../bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py); preserve architecture, parameter count, priors and frozen inputs for each registered instance. | Independent model-posterior reference and full analytical DCTR callbacks required. A UKF or SGQF posterior is only that approximation's endpoint. Small architecture results do not validate the largest target. No new learning campaign is assumed. |
| M12 Hard-bound macrofinancial | [Yield-curve model](../../bayesfilter/hardbound/model_tf.py), [joint target and reduced gate](../../bayesfilter/hardbound/joint_target_tf.py); preserve priors and observation/boundary law, compare parameter marginals. | [Reduced grid reference](../../bayesfilter/hardbound/gate_grid_reference.py) for its own scope; qualified joint latent/parameter sampling may supply parameter marginals at larger scale. DCTR is marginalized parameter inference; changing its task to joint-state HMC would change the method under test. |
| M13 Range–bearing | [TF tracking fixture](../../experiments/dpf_implementation/tf_tfp/fixtures/range_bearing_tf.py), [common-suite range-bearing scopes](../../experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py), moderate/low-noise source fixtures. Start with the declared process/observation noise-scale task. | Small same-law reference then particle-marginal reference. Gaussian raw-angle and wrapped-angle likelihoods must be distinguished; circular residual code cannot silently redefine the density. DCTR parameter callback and posterior reference needed. |
| M14 BGS/adapter-defined applications | [BGS posterior adapters](../../bayesfilter/adapters/bgs.py) and concrete instantiated MacroFinance/DSGE applications found by P0. Retain the actual upstream model, prior and data. | An abstract protocol is not a concrete model. Each actual instantiation receives its own scope; missing upstream model/data stays `UNINSTANTIATED`, not PASS or silently excluded. |
| C01 Gaussian controls | Repaired anisotropic toy; [100-dimensional Gaussian](../../bayesfilter/inference/neutra_paper_d100_target.py); 16-dimensional correlated Gaussian in the [generic control source](../benchmarks/run_neutra_generic_five_stage_model_2026_08_15.py). | Exact moments and independent draws. Controlled deliberately imperfect fields test the kernel; these are not DCTR state-space evidence. |
| C02 Funnel / banana / mixture / ridge | [100-dimensional funnel](../../bayesfilter/inference/neutra_paper_d100_target.py), 16-dimensional banana and four-dimensional three-mode mixture in the [generic source](../benchmarks/run_neutra_generic_five_stage_model_2026_08_15.py), [varying-Hessian ridge](../../bayesfilter/inference/neutra_varying_hessian_target.py). | Exact transformed/component draws where supplied and checked; independently qualified integration/sampling otherwise. Explicit mode/tail checks. Use fixed synthetic fields, not an invented DCTR likelihood. |
| C03 German credit | [Logistic-regression posterior](../../bayesfilter/inference/neutra_german_credit_target.py), with pinned existing data/prior/preprocessing. | Exact regression log density plus independent qualified posterior reference. Surrogate-force HMC control only: no DCTR filtering likelihood exists here. Data access/preprocessing gaps remain visible. |

### 5.1 Coverage closure and scope cards

P0 must reconcile the register against `bayesfilter`, repository-owned
`experiments`, `docs/benchmarks`, model registries, fixture catalogs and concrete
test fixtures. Parse definitions/configurations without importing arbitrary
benchmark modules. The earlier 175-definition AST census is a starting index,
not proof of completeness. Vendored library examples are not automatically
BayesFilter-owned tasks; any vendor model actually instantiated by a repository
runner must be mapped. Explicitly identify historical-only implementations,
duplicate laws, interfaces, fixed-density controls, and concrete parameter models.
Deduplication needs source equations/parameter/data evidence. A tiny mechanics
fixture may be mapped to its family without being promoted to a new research
scope. A distinct statistical law may not be discarded for convenience.

Each executable scope card must give: model/source identity; equations or exact
source anchors; state/observation/parameter dimensions; inferred and fixed
parameters; normalized prior and chart; initialization and observation timing;
data version/horizon/regime; reference type and accuracy checks; DCTR value and
force settings/identity; tuning partitions; seed schedule; comparison arms;
summary vector/margins/events; heuristic set; hardware; measured cost/allocation;
exact command; artifact paths; and prerequisite status. Missing material fields
mean `SCOPE_UNRESOLVED`; a runner must refuse a research launch of that card.

For new noise-scale scopes, the stated prior is a hypothesis, justified by
testing inference near the existing source scales without dropping normalizers.
Check prior predictive behavior and identifiability before tuning. Do not alter
it after seeing posterior agreement. Existing source scales may themselves be
arbitrary. A proper, deliberately chosen synthetic benchmark prior still
defines a valid sampler test; it does not become empirically calibrated by
doing so. Record source-scale provenance and prior-predictive behavior. For
real-model inference, justify its physical interpretation and any data-derived
construction before examining retained answers; otherwise mark that claim
unresolved. Report sensitivity as a distinct scope where it affects the
interpretation. A family with only a restricted noise-scale test remains
incomplete for any existing full-parameter posterior listed above.

Closure produces `model-register.json`, including every discovery item and its
mapped scope/disposition, with source paths and reasons. It must detect an
unmapped new concrete model. Completing this register is coverage design;
it does not mean any of those models passed inference.

## 6. Data, references, tuning and freeze rules

### 6.1 Data and replication

Use disjoint DCTR calibration, validation and untouched evaluation datasets or
prepared-data partitions within each declared model/horizon regime. Keep the
generation seeds and identities for each role. For a real-data target, define
calibration/validation windows of the same stated regime and a fixed evaluation
window before tuning. If no honest separation is possible, record the scope as
unresolved rather than silently treating a same-data diagnostic as held-out
model validation. A temporal split changes the data target and must be explicit.

HMC mass/step/trajectory tuning is necessarily for the *actual endpoint and
dataset* being sampled. It may use that dataset with independent adaptation and
verification chains, all excluded from posterior estimates. This is distinct
from tuning DCTR controls against evaluation posterior answers. Never consume
an HMC artifact from a different dataset/endpoint binding. The DCTR scope
artifact must explicitly cover the named preparation regime and partition IDs;
it cannot be repurposed across horizons or model targets.

Start with one untouched dataset and one fixed DCTR bank for the LG1 mechanism
pilot. Subsequent model confirmation requires two independently seeded groups
of four chains per evaluated arm, at least two prespecified DCTR input banks,
and nominal plus one model-relevant stress dataset/regime. Report each bank,
dataset and group separately before any aggregate. These are modest robustness
checks, not proof of frequentist interval coverage or a basis for universal
claims. A new `N`, horizon, dimension, chart, or numerical execution mode requires
its own DCTR tuning scope and HMC binding.

Seed construction must be deterministic and process-independent. Use TF
stateless folding with a recorded master pair `(20260918, 918)`, stable integer
codes for model/scope/partition/attempt/dataset/bank/chain/purpose, and separate
purposes for simulation, DCTR inputs, tuning, momentum and acceptance. Do not
use Python's salted `hash`. Preserve the current fixture's established seed
mapping when reproducing it, with an explicit mapping into this ledger.
For paired arms, share the dataset and initial-state design; common random
numbers are optional and require a valid coupling record, not an assumption
that the same seed makes chains paired.

### 6.2 Reference qualification

For low-dimensional parameter models, integrate the proper posterior using
the exact likelihood or a separately refined likelihood. Vary both latent
integration domain/resolution and parameter quadrature support/resolution.
Check at least three refinements and an independent implementation or identity
where feasible. Two agreeing grids can share bias; call refinement differences
empirical numerical evidence unless an actual bound was derived. The numerical
error allowance is at most one tenth of each posterior-equivalence margin.
If empirical checks support it without a rigorous bound, the result must say
“agreement with the qualified numerical reference,” not exact correctness.

High-dimensional references must satisfy their own mixing and MCSE requirements
on every reported function, including modes and tails. A qualified joint-state
reference can be marginalized to the same parameter law. UKF, SGQF, TT, Laplace,
and frozen particle approximations cannot certify the model posterior merely
because their output is finite. Reference validation is charged to the scope;
its cost cannot be omitted to make the proposed sampler appear economical.

Reference center/scale/event thresholds used to define summaries are fixed
before opening surrogate retained results. Use independent construction and
assessment reference samples, or propagate their estimation uncertainty in the
comparison. Source comparisons remain the same even when the surrogate fails.

### 6.3 Two distinct tuning authorities

DCTR numerical controls need a repository-issued scope-matching offline
selection artifact. Preserve the canonical route identity and all capabilities;
do not self-stamp exact-score admission. If the existing issuer requires exact
score agreement and cannot express an **imperfect proposal-field experiment**,
implement a separate honest use classification with full route/source identity,
checked value/force validity and scope matching. Test that it cannot grant
canonical exact-score status or bypass existing admission rules. This is an
explicit implementation prerequisite, not permission to turn every old failure
into an admitted default.

Calibrate Class C controls, including leaving damping/ridge/protection off,
using derivation or measured non-harm curves: agreement on healthy trajectories
within predeclared numerical tolerance and bounded, flagged behavior on failing
ones. Control candidates that lack that justification cannot become defaults.
Evaluate a bounded hypothesis set on calibration/validation only; numerical
moment and force errors diagnose it, while downstream validation trajectories
may nominate an eligible force. Do not tune to the final heuristic results.
Retain value controls unchanged during a force-only repair; if value controls
change, create a new finite target and requalify its reference.

HMC tuning must follow [the public interface](../reference/hmc-tuning-interface.md)
and the active `HMC_TUNING_INTERFACE_CAPABILITIES` in
[tuning_contract.py](../../bayesfilter/inference/tuning_contract.py).
For B/C/E use `FrozenPositionOnlyForce`, `FrozenTargetPotential`,
`bind_neural_force_hmc_tuning_runner`, and public `tune_hmc_kernel`.
Set the force semantics explicitly to
`DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS`: the current wrapper's
default claims a scalar-potential gradient and is inappropriate for this
experiment. Return `PositionOnlyForceEvaluation` with original validity and
`TargetPotentialEvaluation` with explicit support/numerical status; do not let
a bare `+infinity` from a failed likelihood inherit the kernel's default
outside-support interpretation. These are consumer wiring requirements.
The binding/runner does not issue artifacts. Ordinary matching-score HMC uses
the ordinary public tuner. `tune_fixed_transport_hmc_kernel` is for a genuine
frozen transport and its Jacobian-corrected target, not an arbitrary force.
No NeuTra transport is introduced here.

Record the tuner's actual bounded mass, step-size and leapfrog search, health
checks, source closure and fresh fixed-kernel verification. Import thresholds
and schemas rather than duplicating their values. Freeze after tuning; every
arm receives equal stated access to tuning resources, with actual usage
reported. If one arm is under-tuned or under-budgeted, its timing cannot support
a superiority claim. Fixed-mass sensitivity comparisons may supplement each
arm's own tuning but cannot replace it.

## 7. Posterior criteria, uncertainty and sequential sampling

The numerical margins and interval rule below define formal confirmation.
They are not a single verdict on the value of a bounded pilot. Apply the
[judgment criteria](surrogate-hmc-judgement-criteria-2026-09-22.md) to state
practical accuracy and the next research action separately. All target,
numerical and exploration checks remain in force. An accepted pilot does not
count as a formal equivalence result or evidence of computational superiority.

All analysis uses physical parameters and specified predictive quantities,
with transformed coordinates additionally checked for mixing. Let `f_j` be
a prespecified function, `d_j = E_candidate[f_j] - E_reference[f_j]`, and
`delta_j` its equivalence margin. Use the following provisional scientific
tolerances; P0 must translate them into each scope's quantities before results
are inspected. They are research tolerances, not previously validated defaults.
When a model reference is available, use its fixed centers, scales, events and
the same function vector for all four comparisons in section 4. If only D is
qualified, define summaries against D and report an endpoint-sampling test;
do not present differently standardized comparisons as an error decomposition.

| Function class | Initial margin hypothesis and purpose |
|---|---|
| Mean of each parameter, centered/scaled by the fixed reference mean/SD | `0.10` reference SD; limits location distortion. |
| Centered second moment of each standardized parameter | `0.10`; limits variance distortion to roughly ten percent when reference variance is one. |
| Prespecified coupled cross-moments | `0.10` in reference-standardized units; include every structurally important coupling, not only easy coordinates. |
| Marginal CDF at reference 0.1, 0.5 and 0.9 quantiles; mode/boundary-event probabilities | Absolute probability error `0.02`. |
| One or more scientifically meaningful predictive event probabilities per model | Absolute probability error `0.03`, with the simulator/law and event fixed in the scope card. |

Choose scope-specific margins from the error that would materially change
inference, including rare-event and decision-threshold consequences. The
numbers above are starting hypotheses, not universal defaults. A materially
different margin is a documented design change made before that scope's claim
run, never a way to convert an observed failure to PASS.
Mean/CDF agreement alone does not determine the full joint posterior; report
the actual functions checked and supplement with joint plots, mode membership,
pairwise dependencies and predictive checks. A conspicuous unresolved joint
or mode discrepancy vetoes an assertion of satisfactory posterior inference.

For a predictive event, evaluate the same independently qualified model
conditional probability `f(theta)=Pr_model(event | theta,y)` under each
parameter posterior. Include its numerical/Monte Carlo uncertainty. Substituting
a DCTR filtering prediction only in the candidate arm would mix parameter
inference error with an additional filtering approximation. That combined
prediction is a separate, explicitly named comparison if desired.

For each comparison, predeclare `m` functions and at most `K=2` inferential
looks. Construct simultaneous uncertainty intervals for their differences,
using total family error `alpha=0.05` across those `m*K` tests. For example,
under a checked stationary-chain CLT and independent reference sampling,

```text
I_j = d_hat_j +/- [t_(nu, 1-alpha/(2*m*K))
                  sqrt(MCSE_candidate_j^2 + MCSE_reference_j^2) + b_reference_j].
```

Use a valid covariance term for deliberately paired estimates; otherwise treat
them as independent and do not advertise pairing. `b_reference_j` is an actual
numerical bound if available; an empirical refinement allowance must be labeled
as such. Estimate MCSE with autocorrelation-aware batch/spectral methods and
cross-check batch-length sensitivity. Require enough effective batches for the
chosen approximation; IID standard errors or ESS inserted into an IID binomial
interval are not acceptable substitutes. Rare-event or nonmixing cases with
unreliable uncertainty remain inconclusive.

Pass equivalence only when **every** interval lies inside `[-delta_j,delta_j]`.
An interval entirely outside that range supports discrepancy. Overlap with the
boundary is inconclusive, not evidence of equivalence or of failure of the idea.
The familywise statement is for this declared scope/comparison, not simultaneous
statistical certification of every repository model. Any stronger cross-model
ranking or multiplicity claim requires its own analysis.

### 7.1 Sampling schedule

Use the public tuner's adaptation/fresh verification, archive all tuning and
warm-up draws, and exclude them from inference. After freezing the kernel,
run groups of four overdispersed chains in fixed 250-transition chunks, recording
each chunk. Initial states cover the posterior's relevant regions, including
distinct modes when known; a reference-only initialization that hides trapping
is insufficient. Use valid prior states and independently constructed coarse
reference regions, with the chosen design frozen.

Choose retained length `n` per chain from calibration precision/cost estimates
before sampling, with `1000 <= n <= 5000`; evaluate equivalence at `n` and,
if needed and budgeted in advance, `2n <= 10000`. Reserve both looks at launch
or declare one look. Intermediate chunks may monitor numerical health and
mixing, but cannot be additional unaccounted opportunities to declare agreement.
Optional-stopping correction is the predeclared two-look allocation above;
its uncertainty remains conditional on the MCSE/CLT assumptions being adequate.

The [September22 precision audit](surrogate-hmc-inference-precision-result-2026-09-22.md)
exposes a weakness in the first 13-function/four-comparison pilots: at 2,000
draws per chain, an ideal IID median-CDF interval has nominal radius 0.02146,
already larger than the 0.02 margin before numerical-reference error. This is
not a bound on every realized MCSE estimate, but it means an affordable
2,000-draw schedule is not by itself a justified equivalence design. Existing
looks and outcomes remain frozen. Before any new inference scope, assess
prospective precision using its actual function count, comparison allocation,
reference allowances and plausible dependence, and justify the diagnostic
policy for those functions. Fund the resulting complete schedule. A smaller
explicit pilot may study implementation or nomination, but cannot silently
replace an underfunded equivalence experiment. The diagnostic-only binary
repair's Gaussian calibration does not establish this wider scope.

At an inferential look require maximum rank-normalized split and folded R-hat
`<=1.01`, bulk and tail ESS `>=400` for all nonconstant inferred parameters,
all-chain movement, and qualified MCSE for every function. These are minimum
screens; the equivalence intervals may require far more draws. A parameter is
not “constant” because a buggy adapter never uses it. Reference-supported
degeneracy must be declared before sampling. If precision cannot be afforded,
classify `INCONCLUSIVE_BUDGET`, not a successful short-chain test.

For confirmation, both independently seeded four-chain groups and each required
DCTR bank must pass the relevant comparisons; pooled statistics cannot hide a
failed group or mode. Dataset replication is not additional MCMC replication.
Do not infer repeated-data credible-interval coverage from these few datasets.
Such a coverage claim would require a separately powered simulation design.

### 7.2 Diagnostic roles and decisions

| Diagnostic | Role and consequence |
|---|---|
| Proper posterior, parameter dependence, chart/prior, same-law reference, complete canonical call chain, per-chain full energy, replay/map validity | Engineering/mathematical continuation veto on affected dependencies until repaired. |
| Reference uncertainty or numerical error too large | Veto on that comparison; repair/reference-budget trigger. Finite-target work may proceed if independently valid. |
| Posterior equivalence | Primary formal-confirmation criterion. Preserve pass/discrepancy/inconclusive separately from practical pilot judgment. Candidate failure is not a global continuation veto. |
| R-hat/ESS/MCSE, mode traversal, unqualified numerical endpoint exclusions | Mixing, exploration or validity problems block posterior acceptance. Insufficient MC precision limits the corresponding claim; extend only for a material question under a qualified diagnostic and the prespecified budget. |
| Force error, filtering moment error, finite energy-error size, acceptance | Explanatory/repair diagnostics. No fixed accuracy or acceptance threshold substitutes for posterior inference. |
| Force fallback | Explanatory if the map remains qualified; unqualified stateful/asymmetric behavior is a validity veto. |
| C samples `pi_D` adequately, but `pi_D` differs from `pi_M` | Model-inference promotion veto and finite-likelihood repair trigger; it does not invalidate the surrogate-HMC mechanism. |
| Reliable conditional loss to a cheap heuristic | Promotion veto for practical usefulness in that situation; preserve any independently established sampling-accuracy result. |
| Unsupported cost forecast, exhausted hours or attempt allowance | Stop the dependent launch/continuation; retain incomplete coverage explicitly. |

## 8. Cheap adversaries and conditional evaluation

Construct and run the following three-to-five simple alternatives on the same
target/data, with their own frozen calibration settings. Each supplies a
different falsification opportunity. An approximation can be inexpensive yet
inaccurate; it counts as a successful adversary only on the quantities and
situations where its uncertainty/error is adequate. Reference oracles are
additional to this adversary set.

| Model situation | Constructed adversaries and rationale |
|---|---|
| Small bounded/near-Gaussian parameter problems, especially M01 | Tuned random-walk MH (only endpoint values); prior-proposal importance sampling (simple global coverage); Laplace approximation (cheap local geometry); direct posterior quadrature where feasible (also an oracle). |
| SV, generalized SV, predator–prey, coupled/structural/accumulation/growth models | Tuned random-walk MH (no fragile force); coordinate slice sampler with valid support handling (local scale adaptation); defensive prior-plus-Laplace importance mixture (global and local coverage); qualified UKF/SGQF-posterior approximation where applicable (cheap practitioner alternative with its model bias explicitly measured). |
| SIR, range–bearing, hard-bound macrofinancial models | Prior-coordinate random-walk MH (bounded geometry); support-aware coordinate slice sampling (boundary exploration); defensive prior/Laplace importance mixture (retains mass away from a single mode). SIR-UKF is not reintroduced. |
| SSL-LSTM and high-parameter targets | Blocked random-walk MH (simple separable moves); valid coordinate/block slice sampling (heterogeneous scales); diagonal Laplace (cheap uncertainty approximation). Any discovered target structure can justify a fourth simple exact conditional update before evaluation. |
| Gaussian/funnel/banana/mixture/ridge controls and German credit | Random-walk MH; coordinate slice sampling; Laplace or simple known-component mixture approximation. Exact independent control draws provide the reference rather than an efficiency trick attributed to DCTR. |

Evaluate at least: nominal versus stress data regimes; central versus lower
and upper posterior tails; distinct modes when present; and initialization
near versus away from the principal posterior region. For parameter functions,
use **reference-defined** events and compare their probabilities and conditional
moments. For metastability, record chain occupancy and transitions by initial
region. Do not estimate correctness from only the draws that the candidate
managed to put in a region. Insufficient reference/event counts mean that
conditional claim is inconclusive.

Stress regimes are predeclared from each model's structure: near-stationary
boundaries/low observation noise for linear models; low/high volatility and
high persistence for SV; low populations/strong nonlinear observations for
PP/SIR; strong completion/nonlinearity for structural models; small range and
angular wrap regions for tracking; active bounds for macrofinance; larger
registered parameter dimension for SSL-LSTM; and narrow necks/separated modes
for controls. Preserve the statistical law when changing a regime. Calibrate
the actual numerical settings per resulting scope.

Do not use heuristic results as a tuning objective. If a heuristic reliably
satisfies the inference criterion where surrogate HMC does not, make that the
headline. If both meet accuracy, practical dominance requires predeclared
accuracy-constrained cost comparisons with uncertainty. One-run runtime/ESS
differences remain descriptive. Include compilation, DCTR preparation, offline
tuning, endpoint evaluations, force evaluations, reference construction and
retained sampling in cost reports; additionally report amortized costs only
for an explicit number of future datasets. ESS per gradient alone is not the
primary efficiency measure when different forces have different costs.

## 9. Implementation contract and executable acceptance checks

Build one registry-driven consumer of the shared
[neural-force kernel](../../bayesfilter/inference/neural_force_hmc.py).
Despite that module name, a neural network is not required. Share proposal,
acceptance, tuning binding, archive and decision logic across models. Model
adapters supply their laws and analytical callbacks; do not copy a reduced
integrator or omit full-DCTR features in a difficult lane.

This program makes no new Zhao-Cui source-faithfulness claim. Any proposed
Zhao-Cui reference or consumer change must meet the repository's paper-and-author-
code source-anchor rule before implementation; an existing route name does not
establish that qualification.

Every DCTR consumer must reach the repository canonical route with
`contract_e_chol_v1` and total composition
`contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1`.
Verify the direct source-cloud moment/weight and transport contributions and
the UKF covariance lifecycle at the **consumer call chain**. A generic routine
existing elsewhere is insufficient. The imperfect-field label does not excuse
an implementation falsely claiming to execute the canonical algorithm.

Use TF/TFP, GPU FP32 with TF32 enabled and stable-signature XLA numerical
kernels. Honor `dpf_transport_exact_divisor_cap3000_v1`: `N=1024 -> K=1024`,
`N=10000 -> K=2500`, `N=10240 -> K=2560`; no tiny historical chunk fallback.
The initial particle-count hypothesis is `N=1024`, not a universal tuned
default. Use FP64/FP32-no-TF32 only as explicit reference/parity arms. Run GPU
commands with trusted/escalated access; set and verify memory growth before
initialization and record allocator peaks. No package/environment mutation is
assumed. CPU-only checks hide GPUs before framework import and say so.

Pure TF kernels must have explicit stable signatures and bounded retracing;
file I/O, provenance and strings stay outside XLA. No implicit/direct pfor.
Use analytical recursion and TF loops where needed. Batch chains natively
when feasible; if the full route only supports a verified sequential lane,
record actual sequential cost and preserve capabilities. This is not NeuTra
training, and a fictitious parallel speedup is not an admissible forecast.
The first LG1 reference comparison uses FP64 HMC state, momentum and Kalman
endpoint arithmetic as an explicit numerical-reference exception, while the
full DCTR field/value still executes in GPU FP32/TF32/XLA. Define the field as
`cast64(g_D_FP32(cast32(z)))` with its precise physical-chart boundary checked;
do not label it an exact FP64 gradient. The qualified LG1 C candidate is
`lg1_dctr_physical_fp32_chart_fp64_v1`: apply the chart in FP64, explicitly round
the physical coordinate to FP32 (using the adjacent interior value if rounding
hits the support boundary), evaluate the unchanged full physical FP32 DCTR
density, and add the FP64 chart Jacobian once. Archive every boundary-rounding
flag. This is a named finite-arithmetic target distinct from the original
unconstrained FP32 scalar; promoting that old scalar alone did not repair its
numerical support failures. D integrates this same named target over the full
physical support. Its qualification checks include physical-value non-harm,
replay, boundary cases and row independence.
This mixed-precision scope needs its own typed binding and parity tests. A later
all-FP32 sampler is a distinct scope; it is not validated by casting its force
inside an FP64 integrator.
Classify this scope as `FP64_INTEGRATOR_FP32_DCTR`. Also record separate
integrator, momentum, force, endpoint and reference dtypes, TF32, device and
XLA settings per arm. B uses an FP64 Kalman endpoint; C combines the physical
FP32 DCTR density with the specified FP64 chart. The common precision label
must not hide that target difference.

NumPy is confined to explicit independent-reference/post-run diagnostic code
and cannot be imported by runtime/admission/manifest construction.

Before any model inference launch, executable checks must establish:

1. The consumer's `U` includes the correct prior, chart, initial law and every
   declared parameter; perturbing each inferred coordinate exercises its real
   mathematical dependence. Separate genuine weak identification from an
   inactive code path.
2. Actual B and C force/endpoint calls reach the declared full DCTR/reference
   routines with correct coordinates, signs, shapes, dtype and source identity.
   Test scalar and batched lanes, unequal chain positions, and batch permutation.
   Also test row independence: appending/replacing other chains cannot change a
   chain's field or endpoint. Permutation equivariance alone does not exclude
   cross-chain coupling, which would invalidate independent accept/reject steps.
   Replay identical positions before and after unrelated evaluations to expose
   changing random inputs, controls, counters or caches; record GPU repeatability.
   Exercise distinct frozen force and endpoint identities and verify each
   independently. Detect endpoint/cache mutation by force calls; do not require
   equality of the two identities as a validity condition.
3. Per-chain potential plus kinetic acceptance passes exact counterexamples;
   the proposal with momentum flip reverses within independently calibrated
   floating-point tolerances. Include finite non-gradient fields and force
   fallback branches; compare CPU reference and GPU/XLA behavior.
4. The ordinary public tuner consumes the typed imperfect-field binding and
   produces an artifact accepted only for that precise endpoint/force/scope.
   Assert the explicit proposal-field semantics and the original force-validity
   and endpoint-status propagation, including finite fallback and failed values.
   Wrong data, bank, horizon, prior/chart, mass coordinates, dtype or source
   closure fails closed. The finite-field route cannot issue exact-score status.
5. Warm-up exclusion, fixed retained kernel, deterministic seeds, continuation
   archives, per-chain telemetry, two-look accounting and budget charging are
   actually wired. Decision assembly uses uncertainty and reference quality;
   a smoke PASS cannot become an inference PASS.
6. Failure injection distinguishes candidate rejection, reference failure,
   force fallback and target invalidity. A failure in one model does not stop
   unrelated ready scopes; a shared energy/kernel bug does stop dependents.

Existing supporting tests include `test_proposal_with_momentum_flip_is_an_involution`,
`test_full_energy_trace_and_cached_endpoint_call_count`,
`test_omitted_kinetic_rule_fails_known_swap_counterexample`, and typed-binding
tests in [test_neural_force_hmc.py](../../tests/test_neural_force_hmc.py).
The new [consumer checks](../../tests/test_surrogate_hmc_program.py) verify LG1's
law/chart/gradient against an independent dense observation law and check wiring
to the full callable. The actual GPU probe supplies executable four-chain
mechanics evidence. Neither covers all model consumers or admits inference.

### 9.1 Runner deliverables and commands

Implemented entry point: [run_surrogate_hmc_program.py](../../scripts/run_surrogate_hmc_program.py),
with stdlib orchestration in [surrogate_hmc_program.py](../../bayesfilter/inference/surrogate_hmc_program.py)
and the [full LG1 probe](../../scripts/surrogate_hmc_lg1_probe.py). Stages are
`P0`, `P1`, `P2-reference`, `P2-controls`, `P2-tune`, `P2-sample`,
`P2-endpoint`, `P2-finite-reference`, `P2-C-tune`, `P2-C-sample`, and `all`.
`all` schedules the implemented LG1 stages in dependency order and preserves
the unimplemented model register; it does **not** mean all-model inference.
Use `--help` for stage input paths and bounded wall/CPU options. Existing
evidence is resumed through explicit reference, selection, endpoint, finite
reference and tuning paths; never rerun a completed claim merely to populate
one directory.

Each fresh root contains a source index, the 17-family minimum register, LG1
scope card, run manifests, per-job logs, accounting, diagnostics and decisions.
Source discovery is not concrete-instance coverage closure. Previous campaign
cost records are carried forward automatically from standard artifact roots;
external checks enter through `--previous-accounting`. Failed jobs and timeout
cleanup remain charged. Output directories cannot be reused.

Dry-run inspection uses no TensorFlow or CUDA initialization:

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage all --scope all --output /tmp/surrogate-hmc-next-preview --dry-run
```

The preview prints exact commands, input paths, precision and remaining budget.
Standalone retained stages require a measured forecast from their own verified
tuning. Workers then check actual frozen identities, source/data checksums,
scoped controls, tuning authority and retained continuation before sampling.
`all` cannot forecast an unexecuted tuning stage; it checks that forecast when
the stage becomes ready. A preview alone never admits a sampler.

The public `VerifiedTensorFlowHMCConfig` dispatches through `tune_hmc_kernel`
and supplies fresh dispersed-chain verification, modern R-hat, health checks,
TensorFlow/XLA execution, source-bound artifacts and checked retained handoff.
Its Gaussian exact/imperfect-force qualification and LG1 call-chain evidence
are recorded in the [tuner repair result](surrogate-hmc-tf-tuner-repair-result-2026-09-19.md).
Legacy diagnostic configurations still cannot issue this authority. LG1
independent references, scoped DCTR selection, A/B/C tuning, archived retained
chunks and two-look comparisons are implemented. Remaining model consumers,
heuristic comparisons and broad replication are still required. Master exit2
reports incomplete all-model implementation even when the requested worker
completed successfully; inspect `execution_status` and comparison statuses.

The checked Python environment is
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; recovery recorded TensorFlow
`2.20.0-dev0+selfbuilt`. Recheck the actual version/device/XLA/memory policy in
the new run manifest. Existing mechanics can be checked, when needed, with:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/test_surrogate_hmc_program.py tests/test_hmc_tuning_dispatch.py tests/test_neural_force_hmc.py -k 'not small_dimensional_proposal_jacobian and not biased_position_only_force_preserves_gaussian_moments'
```

That is an explicit CPU diagnostics command, not a model-inference launch.
The Jacobian test is excluded because its current implementation implicitly uses
pfor; the long Gaussian moments test is outside this focused mechanics check.
The original authoring pass ran no numerical jobs. The owner has now authorized
execution; new checks and experiments belong in fresh versioned run directories.

## 10. Stages, dependencies and bounded repair

| Stage | Work and terminal artifact | Advancement rule |
|---|---|---|
| P0 Coverage and target cards | Reconcile every model definition/instance; complete runnable scope cards in queue order; pin priors/data/references/summary functions; record all missing prerequisites. | A complete register is required for an all-model coverage claim. Ready scopes can move while other cards remain unresolved. |
| P1 Shared correctness and controlled-force tests | Repair the toy in a new driver; test exact, multiplicatively biased, rotational non-gradient and zero fields on exact densities; qualify shared energy/map/tuner/archives and the actual LG1 DCTR consumer. | Shared mathematical/consumer invariants must pass before dependent sampling. Zero or poor fields may fail mixing without failing kernel validity. |
| P2 First DCTR inference experiment | LG1 with A/B/C/D, exact Kalman posterior quadrature, one data/bank pilot, full DCTR. Preserve the old failed filtering screen as explanation. | Decide sampling versus likelihood bias directly. Its candidate failure triggers one bounded repair and does not veto M02–M14. |
| P3 Reference-feasible model tests | Low-parameter linear scopes, actual/transformed SV, M09/M10, small structural/range–bearing scopes and exact-density controls, following reference readiness. | Run at least one complete target/reference comparison before spending on a larger version of that family. A missing large reference is not repaired by claiming the small result transfers. |
| P4 Broad model coverage | PP, SIR, generalized SV, coupled nonlinear, remaining structural/range–bearing scopes, and concrete BGS/macrofinancial/SSL-LSTM tasks as their full adapters and references qualify. | Advance each dependency independently. Record exact blockers and unused coverage obligations. No global “LG1 must have a perfect score” gate. |
| P5 Confirmation and scale | Independent chain groups, banks and stress datasets; declared full parameter/horizon/dimension scopes, including LG20, full PP/Austria, and registered SSL-LSTM complexity. | Apply section 7 separately to each scope; fresh tuning for changed scope. Limited budget leaves explicit incomplete validation. |
| P6 Synthesis | Complete coverage/decision tables, accounting, uncertainty, conditional heuristics, and a terminal result review. | Success, failure and unresolved evidence are reported separately. No averaging away a failed family. |

In P1 the artificial fields are mechanism controls, not selected approximations
to replace DCTR: on a Gaussian with precision `Q`, include `g=Qz`,
`g=0.5 Qz`, `g=Qz+0.2 Az` for a fixed antisymmetric `A`, and `g=0`.
The first is a comparator; the next two deliberately violate the matching-field
assumption; zero is a weak-proposal control. Coefficients are frozen diagnostic
choices with no scientific ranking attached. Test a repaired correct sampler
against independent draws. Historical broken acceptance serves only as a
deterministic counterexample, never a new baseline for successful inference.

| Observed failure | What it says | Next bounded action |
|---|---|---|
| Wrong endpoint/prior/Jacobian, chain energy, or call-chain fork | Implementation does not test the claimed target/method. | Repair and rerun focused correctness checks; invalidate dependent results only. |
| Full finite position-only force differs from reference gradient | The surrogate premise is present. | Continue B/C; use discrepancy to explain energy and movement, not to require exactness. |
| A/B poorly mixes, numerical map remains valid | Present tuning/geometry or initialization is inadequate. | Use remaining public-tuner repair allowance; check modes/mass/trajectory, then fresh retained samples. Compare cheap methods. |
| C fails D but D is qualified | Current surrogate sampler has not demonstrated sampling of its finite endpoint. | Repair force/tuning under unchanged value controls; do not blame likelihood bias first. |
| C agrees with D; D differs from the model reference | Finite likelihood, not that sampling comparison, is inaccurate. | Evaluate a fresh scoped value-control/N repair if budgeted; retain B's mechanism finding. |
| Invalid force with valid endpoint | DCTR cannot provide a usable field there. | Qualify a deterministic position-only extension or repair numerical controls; do not alter endpoint support. |
| Invalid reference or mismatched probability law | The current comparison cannot identify inference accuracy. | Repair reference/target mapping or mark it blocked; other qualified comparisons may continue. |
| Too little uncertainty resolution or cost beyond cap | Current evidence is under-budgeted. | Use a prespecified second look or record incomplete; never relax margins or report short-chain convergence. |

A change to frozen controls or selected model/data/target creates a fresh
candidate/attempt with fresh validation/evaluation partitions as applicable.
An infrastructure repair with unchanged target, settings and kernel can resume
the same attempt into a new versioned output segment. Charge all failed and
resumed work. Never discard the failed holdout and present its tuned rerun as
untouched evidence.

### 10.1 Exact first LG1 inference task

Use `x_t = phi*x_(t-1) + epsilon_t`, `epsilon_t ~ N(0,1)`,
`y_t = x_t + eta_t`, `eta_t ~ N(0,0.5)`, `T=10`, and
`x_0 ~ N(0,1/(1-phi^2))`. Infer only `phi ~ Uniform(-0.95,0.95)`;
set data-generating `phi=0.8`. Use
`phi=-0.95+1.90*sigmoid(z)` with its full log Jacobian exactly once.
The endpoint and matching Kalman score must include the phi-dependent
stationary initial law. These choices are from the current LG1 fixture,
not the incompatible historical five-coordinate consumer.

Use fresh attempt index 4 and distinct calibration/validation/evaluation
partition seeds in the fixture's existing mapping. Attempt 5 is available
only for the one planned repair and uses fresh partitions as required. Never
relabel the earlier failed calibration data as a new holdout. Start full DCTR
at N=1024/K=1024, with controls nominated by the bounded scoped procedure;
no controls are declared tuned by this document.

First compute the FP64 Kalman posterior reference and qualify the typed B
consumer. Run A and B with their own tuning and the section 7 physical mean,
centered second moment, three CDF thresholds and one-step predictive event
`Pr(y_(T+1)>0 | y)`. The predictive probability integrates the **model's**
Kalman conditional predictive law over each parameter posterior; using a DCTR
filter prediction here would mix parameter and filtering errors. Evaluate C/D
next if their separately measured forecast fits the reserved P2 allocation.
The finite-DCTR reference uses the exact same frozen value program as C.

For this pilot, one dataset/bank and one four-chain group gives a conditional
mechanism result only. Its existing filtering moment/likelihood-ratio screen
may explain a failure but is not a prerequisite for B. A completed all-LG1
confirmation still needs the independent groups/banks/stress regime in P5 and
the separately stated model-posterior assessment for C.

## 11. Default and assumption audit; pre-mortem

| Choice | Provenance and justification | Failure mode / earliest discriminating check | Promotion status |
|---|---|---|---|
| Full DCTR, Contract E–Chol, analytical recursive path | Owner/repository definition of method under test | A lane bypasses features or direct moment terms: endpoint-to-route wiring/parity tests | Required identity, not proof of score accuracy |
| Imperfect position-only force with corrected endpoint | Derived in section 3 | Stateful fallback or numerical map violates reversibility: force/reverse replay and branch injection | Mechanism hypothesis with explicit assumptions |
| Proper source priors/charts; new noise-scale priors only for missing tasks | Existing posterior definitions; explicit new inference design for filtering-only fixtures | Omitted normalizers/inactive parameters or strong prior domination: coordinate perturbation and prior predictive/reference checks | Source choice or declared hypothesis, never silently transferred |
| LG1 first, stationary prior on the latent initial state | Existing low-cost exact-reference fixture | It is unusually easy or its stationarity derivative is omitted: test initialized likelihood and later stress/full-parameter scopes | Mechanism pilot |
| N=1024, GPU FP32/TF32/XLA, exact-divisor chunk policy | Initial production-direction hypothesis and required backend/chunk policy | Approximation or precision inadequate: same-law reference and FP64 diagnostic, actual memory/cost probe | Hardware policy; N not a tuned universal default |
| No extra damping/protection until calibrated | Current baseline must retain documented control rationale | Unexamined zero setting fails at a boundary: mandatory non-harm calibration, not primary-metric tuning | Baseline only if justified; otherwise unresolved |
| Public tuner, diagonal/affine mass supported by its actual binding | Inspected registry, avoids arbitrary-force use of a transport tuner | Incompatible coordinates or inadequate geometry: binding tests and fresh tuning/mode checks | Supported API, model tuning still required |
| Four chains, R-hat 1.01, ESS 400, two inferential looks | Conservative minimum diagnostic design, consistent with local inference practice | False confidence from mode trapping or imprecise MCSE: overdispersed starts, mode probabilities, simultaneous equivalence intervals | Proposed screens, not proof of convergence |
| 0.10-scale / 0.02-probability / 0.03-predictive margins | Initial error-tolerance hypotheses, not literature-established universal limits | Margins too loose for meaningful bias or too tight for the intended use: justify physical/decision consequences and precision before launch | Scope-specific design required; no retrospective threshold changes |
| Two banks, two chain groups, nominal/stress confirmation | Minimum checks against seed and local-regime accidents | Too few replications to rank methods or establish coverage: preserve per-unit results and uncertainty | Modest robustness evidence only |
| Reference refinement and qualified sampling | Exact methods where available; independent numerical checks otherwise | Shared bias or reference nonmixing: third refinement/independent identity and tail/mode checks | Exact, bounded-numerical or empirical-reference status must be explicit |
| Stage budgets below the remaining envelope | Administrative allocation from recovered accounting | Full models cannot fit: measured 1.5-times forecast before launch, fair coverage queue | Caps, not performance predictions |

The main pre-mortem is a plan that “passes” because it samples the wrong
posterior, pools chains, omits priors, uses the same approximate likelihood as
both candidate and oracle, ignores modes, or only validates an easy restricted
model. Sections 3–9 directly test these possibilities. Another misleading pass
would optimize controls until a reused holdout agrees; partitions, frozen
identities and fresh attempts prevent that interpretation. The likely innocent
failure is compilation/cost or poor tuning before a useful inference comparison
is obtained; the early measured probe and explicit incomplete classification
distinguish it from evidence against surrogate HMC.

## 12. Budget, attempt accounting and launch readiness

Recovered containment allowance: 64 GPU device-hours and 16 CPU worker-hours.
For conservative accounting, charge its recorded work, the closed regression
audit, and the recovery checks together: approximately **0.4313063 GPU-hours**
and **0.8984004 CPU-worker-hours**. The remaining envelope is approximately
**63.5686937 GPU-hours / 15.1015996 CPU-worker-hours**. Reconcile manifests before
any future launch, debit intervening work, and preserve the original detailed
seconds in the [recovery accounting](artifacts/surrogate-hmc-recovery-20260918-01/run-manifest.json).
The older surrogate campaign's balance is unknown and supplies no extra hours.

This program is authorized for **at most 63 additional GPU-hours and 15 additional CPU
worker-hours**, within that conservative remainder; it does not authorize paid,
different-hardware, or unbounded compute. The allocations below sum to those
caps. They include compilation, reference work, failures, tuning, heuristics,
sampling and analysis. CPU time concurrent with GPU work is still charged.

| Allocation | GPU hours | CPU worker-hours |
|---|---:|---:|
| P0 scope closure / manifest preparation | 0 | 0.5 |
| P1 shared consumer / controlled-force checks | 1 | 0.75 |
| P2 LG1 inference and one repair | 3 | 3.25 |
| P3 reference-feasible models and controls | 12 | 8.194444… (29,500 seconds) |
| P4 broader families and references | 18 | 0.888889… (3,200 seconds) |
| P5 replication and declared scale scopes | 20 | 0.916667… (3,300 seconds) |
| Shared repair reserve and P6 analysis | 9 | 0.5 |
| Total authorized additional maximum | **63** | **15** |

September19 redistribution: move1.5 CPU-worker hours from the shared reserve
to LG1 for public-tuner repair, independent references and actual inference.
The total campaign limits and3-GPU-hour LG1 allocation remain unchanged.

September21 redistribution: move2,200 CPU-worker seconds from unused P5
replication to P3 for the first complete KSC T2 surrogate-HMC test. P3 increases
from10,800 to13,000 seconds; P5 decreases from10,800 to8,600 seconds. KSC's
cumulative family ceiling increases from1,800 to4,000 CPU seconds; its7,200
GPU-second ceiling and the54,000/226,800 CPU/GPU campaign caps stay fixed.
The KSC execution plan recorded this redistribution before launch. Its P3
subtotal of12,800 was an arithmetic typo;13,000 is the correct sum. The
launched reservation was checked against the correct4,000-second family and
54,000-second campaign ceilings. This correction changes no scientific look,
attempt count, or running job limit.

September20 redistribution after C tuning: move0.5 CPU-worker hours from P0
and0.25 from P1 to LG1. Their remaining preparation allowances cover the work
observed so far; this funds the actual C/D sampler test after its own tuning
qualified. The unchanged conservative forecast is4738.34 GPU seconds and
5686.01 CPU-worker seconds. Before the next regression check, cumulative
campaign use is4995.05 GPU seconds and5250.90 CPU-worker seconds. A4800-second
GPU /5800-second CPU job fits the unchanged10800-second LG1 GPU ceiling and
revised11700-second CPU ceiling. The whole campaign remains63 GPU hours and
15 CPU-worker hours. This is a budget redistribution, not another inference
attempt or a change to the two looks, six functions, margins or vetoes.

September21 independent C/D redistribution: after all 27 KSC finite-endpoint
checks passed, the measured reference and complete C inference forecasts were
911.68 and 2,983.73 CPU seconds. Transfer another 2,500 CPU seconds from unused
P5 replication to P3: P3 becomes 15,500, P5 becomes 6,100, and KSC's cumulative
ceiling becomes 6,500. Its 7,200 GPU-second ceiling and both campaign totals stay
fixed. LG2, SV20 and M10 already received their initial preparation/cost turns.
This completes an independently required sampler comparison, not another B
attempt or a new posterior look. The transfer reduces future replication funds;
the allocations never promised full coverage or confirmation. See the
[prospective finite-target plan](surrogate-hmc-ksc-finite-target-2026-09-21.md).

September22 M10 inference redistribution: transfer2,800 CPU-worker seconds
from unused P5 to P3 after endpoint qualification passes16 checks. P3 becomes
18,300, P5 becomes3,300, M10's cumulative CPU ceiling becomes4,600; global
CPU/GPU and the M10 7,200 GPU-second ceiling stay fixed. Reserve3,200 CPU /
3,600 GPU seconds for the first M10 B/R attempt, plus60 CPU tests and30 CPU
orchestration. The corrected forecast includes both nomination candidates and
8,000 retained predictive evaluations, using the real claim endpoint. See
[surrogate-hmc-m10-inference-2026-09-22.md](surrogate-hmc-m10-inference-2026-09-22.md).

September22 SV20 inference redistribution: after all17 claim-endpoint checks
pass, transfer11,200 CPU seconds from unused P4 to P3. P3 becomes29,500 and
P4 becomes3,200; P5 remains3,300. SV's cumulative ceiling becomes13,000 CPU
and14,400 GPU seconds within existing P3 capacity. Reserve12,000 CPU/12,600
wall seconds for one public-tuned B/R attempt, plus60 CPU checks and30 CPU
orchestration. The measured complete L1-only forecast is11,655.45 CPU and
11,661.13 wall seconds. Global caps remain unchanged. See
[SV20 inference plan](surrogate-hmc-sv20-inference-2026-09-22.md).

These are allocation ceilings, **not a claim that full all-model validation
fits**. Existing timings do not establish such feasibility. Measure the actual
complete endpoint+force transition, including the real chain batching strategy,
before estimating a scope. Forecast `1.5 * (compilation + preparation + tuning
+ reference + planned inference + analysis)` from measured components. Reserve
the full chosen look/replication schedule before launch. Stop before a launch
whose forecast exceeds its allocation; a short pilot cannot be relabeled as
confirmation because a longer run is unaffordable.

Within P3/P4, initially cap each ready family at 2 GPU-hours and 0.5 CPU-worker
hours for its first complete feasibility/inference attempt; give other ready
families a turn before any receives a second exploratory allocation. These caps
may yield `INCONCLUSIVE_BUDGET`. For P5 allocate at most 4 GPU-hours and 1 CPU
worker-hour to one full scope before returning to the remaining ready queue.
Redistribute unused stage/reserve hours within the total and record why; do not
consume the entire campaign on LG1 repairs or one expensive model. No more than
three substantive candidate attempts per new scope. The existing LG1 three
filtering attempts remain used; the owner has authorized **two
additional inference attempts** (initial plus one repair), at most five lifetime
substantive attempts under this combined accounting. Execution still requires
the numerical readiness checks above. No earlier count is silently reset.

Retain the inherited four-hour limit for a single GPU job and set an explicit
wall-time timeout for every job in its scope card. Wall-clock limits are measured
alongside device/worker hours. At most one serious GPU scope is active at once
on the current hardware; CPU reference/data
work uses at most four workers by default and counts all worker time. Do not
launch unaffordable references in the background. Routine document work is not
a numerical attempt; actual computational diagnostics remain charged.

For each scope, launch readiness requires a complete card, qualified consumer,
eligible reference for the intended claim, DCTR/HMC tuning schedule and
authority, exact implemented command, frozen evaluation rules, fresh output
root, and a measured affordable forecast. This is ordinary scientific readiness,
not another mandatory review chain. At present those implementation and
model-specific requirements remain open. Reviewer unavailability alone is not
a scientific veto; a substantive target, numerical, cost or source-law finding is.

## 13. Required results and completion criteria

Each serious run records Git commit and dirty source closure, command and cwd,
environment, GPU/CPU/XLA/TF32 and verified memory policy, data/partition identities,
seeds/banks, scope and prior/chart, endpoint/force identity, route/tuning artifacts,
all costs and attempts, output paths, this plan and its result note. Preserve
`precision_classification` plus per-arm integrator, momentum, force, endpoint
and reference dtype/device/TF32/XLA fields in scope and decision artifacts.
Preserve warm-up and retained arrays separately and retain every failed attempt. Ordinary
SHA-256 plus Git and versioned directories suffice; no authority-token machinery.

`decision.json` must contain separate fields for engineering validity,
the four comparisons in section 4 (force mechanism, endpoint sampling,
finite-likelihood bias and end-to-end model inference), reference quality,
mixing/MCSE, conditional heuristic dominance, efficiency inference, default
readiness, remaining budget, and next action. Allowed evidence states include
`PASS`, `DISCREPANCY`, `INCONCLUSIVE_PRECISION`, `INCONCLUSIVE_BUDGET`,
`INVALID_IMPLEMENTATION`, `REFERENCE_BLOCKED`, `SCOPE_UNRESOLVED`, and
`NOT_RUN`. `NOT_APPLICABLE` requires a concrete reason; it cannot remove a
difficult model or missing parameter adapter from the coverage count.

New reports must additionally identify the declared stage, practical decision
and its rationale separately from the formal comparison states. This is a
prospective reporting requirement; existing executable reports do not yet
implement those additional fields. Until implemented and checked, record them
in the scope/result note without changing archived machine verdicts.

The result note must include these two tables, filled with actual evidence:

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Per arm/target/scope decision | Intervals versus frozen margins | Target/map/reference/mixing/heuristic findings | Reference, Monte Carlo or model approximation | Continue / repair / complete / budget stop | State the remaining stronger claim explicitly |

| Inference status | Required statement |
|---|---|
| Hard veto screen | Which mathematical/numerical failures are actually supported? |
| Statistically supported ranking | State none unless a predeclared uncertainty analysis supports it. |
| Descriptive-only differences | Identify unqualified timing, acceptance, ESS, force error and tail observations. |
| Default-readiness | This research program alone does not change production defaults or exact-score admission. |
| Next evidence needed | Name the missing replication, reference, scope, precision or repair. |

Add a post-run red-team paragraph: strongest alternative explanation, weakest
evidence, and the result that would overturn the interpretation. Distinguish
implementation failure, tuning failure, reference/diagnostic failure and evidence
against a particular DCTR force/likelihood candidate. The final cross-model
table reports planned, actually attempted, accuracy-passing, efficiency-passing,
failed and unresolved scopes separately; do not average model verdicts.

The first useful scientific deliverable is an actual correct LG1 surrogate-HMC
posterior comparison. The final intended deliverable is the full per-model
surrogate-HMC assessment. Neither a numerical-containment report nor a force
accuracy leaderboard completes this goal. Successful all-model validation
requires the required scopes and confirmation comparisons to pass; a bounded
campaign that stops earlier must explicitly say which of those tests remain
undone and why.
An assessment of accepted practical pilots is a separate deliverable and must
preserve any unresolved formal comparisons. Closing LG1 as a practical pilot
does not require another LG1 run merely to obtain a stronger label.
