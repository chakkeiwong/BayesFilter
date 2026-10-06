# Attributing the NAF fitting result to scalar nonlinearity

Status: owner authorized implementation and execution after review. The design
below is preserved; the executable allocation at the end makes its proposed
controls concrete. No default architecture change or HMC claim is authorized
by a successful attribution test.

Completion: all 87 executable workers finished on 2026-10-06 at 04:38
Asia/Shanghai. The terminal artifact review passed; scientific conclusions and
remaining limits are recorded in
[the completed result and recovery note](bayesfilter-neutra-nonlinearity-attribution-results-2026-10-06.md).

## Question and limits

Does allowing a scalar autoregressive transformation to depend nonlinearly on
its own input improve fitting under a fixed training procedure, independently
of the conditioner, its initialization rule and its nominal parameter count?
Does that result persist with separately calibrated optimizers and on unseen
target geometries? These are distinct from proving that every IAF fails, every
NAF succeeds, or a better density fit necessarily gives better HMC.

The completed study changed scalar transformation, conditioner, initialization
and parameter count together. Its five viable NAF fits do not identify a unique
cause. The primary contrast below changes only the scalar readout within an
otherwise identical parameterized network. Functional shape capacity is the
intervention itself; it cannot also be held constant.

Sources inspected: Huang et al. (2018), section 3.1 equation (8), section 4,
section 6.1 and appendices C, E and G in
`.localresources/q20-flow-training-literature-20260923/papers/huang-2018-naf.txt`;
author `code/naf-flows.py:221-317` and `code/naf-iaf_modules.py:135-177` under
the same resource root. Current local operations are in
`bayesfilter/inference/neutra_transport_core.py` and the configuration and source
differences are in `docs/reference/neutra-implementation.md`. The affine control
below is a new diagnostic intervention, not a reproduction of the author's IAF.

## A mathematical control with an optimizer-independent lower bound

Let

\[
p(x)=\tfrac12\mathcal N(x;-m,s^2)+\tfrac12\mathcal N(x;m,s^2),
\quad m>0,\ s>0.
\]

For a one-dimensional Gaussian base, with no extra latent dimensions, every
composition of affine autoregressive layers is affine. Its density is Gaussian
regardless of conditioner width, depth, optimizer or training duration. The
best Gaussian in forward KL has mean zero and variance \(v=m^2+s^2\): minimize

\[
-E_p\log\mathcal N(X;\mu,\tau^2)
=\tfrac12\log(2\pi\tau^2)
 +\frac{v+\mu^2}{2\tau^2}
\]

over \(\mu\) and \(\tau^2\). Thus the minimum KL is
\(\tfrac12\log(2\pi e v)-h(p)\). With the mixture label \(C\),
\(h(X)=h(X\mid C)+I(X;C)\leq\tfrac12\log(2\pi e s^2)+\log2\).
Consequently every such affine flow satisfies

\[
D_{\rm KL}(p\Vert q_{\rm affine})
\geq\tfrac12\log(1+m^2/s^2)-\log2.
\]

At the illustrative separation \(m/s=3\), this is
\(\tfrac12\log10-\log2=0.4581453659\) nats. The separation is an explicitly
chosen explanatory fixture, not a fitted benchmark or universal default.

The monotone nonlinear map \(T=F_p^{-1}\circ\Phi\) attains the target exactly:
\(F_p(T(z))=\Phi(z)\), hence \(p(T(z))T'(z)=\varphi(z)>0\).
This proves the need for scalar nonlinearity in this restricted control.
It does not prove that the finite four-unit DSF will learn that map. Huang's
distributional approximation theorem does not itself imply convergence in KL,
uniform score accuracy or HMC readiness. A fitted DSF beating the Gaussian
lower bound requires its own checked KL evaluation and uncertainty bounds.
The analytic optimal Gaussian, not a poorly optimized affine network, is the
baseline for this control. A Gaussian target is the positive implementation
control: both scalar families can represent it.

The proof does not extend to the current multivariate IAF by assertion.
Permuted affine autoregressive layers can create nonlinear multivariate maps.

## Primary intervention: identical network, two scalar readouts

Keep the current cMADE conditioner, its masks, all weights and biases, its
initialization and all twelve output values per coordinate. Write its positive
slopes, offsets and normalized weights as \(a_k(h),b_k(h),w_k(h)\), where
\(h=u_{<j}\), \(a_k>0\), \(w_k>0\), and \(\sum_kw_k=1\).
Use these same outputs to define

\[
A_\theta(u;h)=\sum_kw_k(a_ku+b_k),\qquad
N_\theta(u;h)=\operatorname{logit}\sum_kw_k\sigma(a_ku+b_k).
\]

Compare \(A_\theta\) with \(N_\theta\), or express the intervention as

\[
T_{\theta,\lambda}=(1-\lambda)A_\theta+\lambda N_\theta,
\quad\lambda\in\{0,1\}.
\]

Zero is the affine NAF control; one is the existing DSF equation. This changes
neither the head size nor the nominal number of trainable parameters. In the
affine arm all three output groups can affect the effective slope or intercept;
the comparison does not pad a small IAF with unused dummy weights. Nevertheless
those outputs enter through only two aggregate functions in the affine arm.
Equal nominal counts do not mean equal identifiable functional degrees of
freedom; that loss of shape freedom is what is being tested.

For \(0\leq\lambda\leq1\), writing \(S=\sum_kw_k\sigma(a_ku+b_k)\),

\[
\partial_uT=(1-\lambda)\sum_kw_ka_k+
\lambda\frac{\sum_kw_ka_k\sigma(a_ku+b_k)[1-\sigma(a_ku+b_k)]}{S(1-S)}>0.
\]

Both component maps have limits minus and plus infinity. They therefore give
invertible scalar maps with triangular Jacobians and analytic diagonal log
derivatives. The existing crossings \((y-b_k)/a_k\) bracket the inverse of
both maps and their convex interpolation: at the lower crossing all
\(a_ku+b_k\leq y\), and at the upper crossing all are at least \(y\).
The affine inverse also has the analytic form
\((y-\sum_kw_kb_k)/(\sum_kw_ka_k)\), giving an independent numerical check.

The interpolation is a local diagnostic, not an author method or a new
production default. Only the endpoints are required. Intermediate lambda
values would be exploratory; performance need not be monotone in lambda.
Implementation must extend the shared numerical authority and explicitly
identify diagnostic artifacts, rather than copying a second flow kernel.

Paired starts copy identical parameter arrays and optimizer states. This holds
the parameter initialization rule fixed; it does not give identical initial
functions because the readout changes. Record initial density, derivatives and
fit metrics before training. Initial function changes are a consequence of the
intervention. If they dominate the result, use a separately checked common-map
initialization experiment before attributing the gain specifically to learning
new curvature. Do not force all sigmoid units identical merely to obtain an
identity map: that can impose a symmetry that obstructs learning shape.

## Conditioner, capacity and optimization controls

First establish the primary paired contrast with the current cMADE. Then use
this crossed design to check whether the effect depends on that conditioner:

| Conditioner | Affine readout A | Nonlinear readout N |
|---|---|---|
| Current author-derived cMADE | Primary control | Existing NAF scalar map |
| Hoffman-style masked ELU network with the common twelve-output head | Conditioner control | Nonlinearity with the alternative conditioner |

The second row is an explicitly diagnostic adaptation of the conditioner, not
canonical author IAF. Within each row, masks, arrays, initialization and output
dimensions match exactly. A benefit in one row only establishes an interaction,
not a conditioner-independent effect. The difference between the two paired
effects estimates that interaction; additive effects must not be assumed.

Keep canonical IAF as an external baseline and include a wider canonical IAF
as a capacity control. For the current two-dimensional, three-stage IAF with
two hidden layers of width W, the nominal count is
\(3(W^2+8W+4)\). W=134 is the nearest permitted even width to the NAF's 56,868
nominal parameters: 57,096, a 0.40% excess. This is a derived starting point,
not adequate evidence of matched effective capacity. Record masked versus
unmasked parameters and verify actual connectivity. The exact common-network
contrast remains the primary attribution experiment.

The new affine control has positive uncapped slopes. This prevents an apparent
gain caused solely by the canonical IAF's conditional tanh cap from being
mistaken for a need for scalar curvature. If attribution relative to that cap
remains material, compare canonical IAF's existing capped and identity-scale
configurations separately, with matched parameter arrays and calibration.

Use exact samples initially so teacher quality and mode discovery are held
fixed. The primary fixed-recipe comparison inherits the already executed three
stages, width 64, batch 64 and 8,192 updates at .001 followed by 8,192 at .0003
with Adam reset. These are comparison settings, not universally tuned defaults.
Use identical sample sequences, endpoint rules, precision, clipping and
optimizer settings within each pair. Measure wall time; equal updates are not
equal compute. A subsequent equal-compute affine extension and equal-budget,
family-specific optimizer calibration distinguish a finite-training advantage
from an inadequately optimized comparator. Use development validation for that
calibration, then freeze choices before independent confirmation.

## Evidence contract and interpretation

For this attribution question, the primary numerical outcome is the paired
independent-reference log-density difference

\[
\Delta=E_p[\log q_N(X)-\log q_A(X)]
=D_{\rm KL}(p\Vert q_A)-D_{\rm KL}(p\Vert q_N).
\]

Positive delta favors the nonlinear readout for density fitting. Normalizing
constants cancel. Each pair shares reference samples; training seeds, not
individual reference rows, are the units of training replication. Preserve
both training variation and reference Monte Carlo error in paired uncertainty
estimates. Use a separate pilot to price the study and estimate variance, then
freeze the number of fresh confirmation pairs and inference procedure before
seeing their outcomes. A conventional familywise error level of .05 is a
proposed statistical design choice; simultaneously cover the planned target
and conditioner contrasts. Report effect intervals, not only a p-value. The
scientifically material effect and power target must be declared with that
allocation; the original three seeds do not become new confirmation evidence.

Retain the existing coverage, conditional-moment and tail checks. The ordinary
1,000-point score probe remains explanatory unless it exposes invalid
computation. Density accuracy is explicitly the outcome for this mechanism
question; it cannot promote a map to HMC readiness. Any downstream claim needs
fresh target-specific tuning and the shared sequential HMC assessment, with
identity latent mass and all existing physical-space checks.

| Finding | Interpretation and next action |
|---|---|
| Nonlinear arm passes while the matched affine arm fails, with positive paired effect intervals | Evidence that scalar nonlinearity matters under the specified training protocol |
| Effect persists across both conditioners, capacity control and calibrated optimizers | Stronger attribution; report interactions and the finite scope |
| Wider or better-calibrated affine arm closes the gap | Do not claim nonlinear scalar maps are necessary for these multivariate targets |
| Only cMADE/nonlinear combination works | Conditioner–transform interaction remains part of the explanation |
| Affine arm fails the Gaussian control or gradients/inverses fail | Harness or numerical defect; repair before interpreting architecture |
| Both pass density checks but HMC differs | Investigate score/geometry; density contrast alone does not explain sampling |

As a secondary counterfactual, replace a fitted NAF's readout by A and measure
the immediate loss, then refit the affine arm with declared effort. Immediate
damage alone is not evidence: arbitrary removal of a trained component is
distribution shift. Preserve the original checkpoint so restoring N can verify
that the intervention, rather than corruption, caused the change.

Freeze the protocol before using untouched randomized two- and three-center
targets. Keep the already reserved final target seeds out of pilot selection.
Generalization must be assessed over new target geometries as well as fitting
seeds; many seeds on one geometry do not establish generalization.

## Execution boundary, audit and records

The existing campaign allocation remains the ceiling. Price the complete next
stage, including preparation, numerical checks, all comparison arms and
assessment, before assigning a launch reservation. No additional compute is
assumed. Use a fresh versioned attribution output root, a source snapshot/run
manifest and explicit costs; preserve all prior fitting evidence.
External sample generation uses the CPU lane. Serious fitting uses trusted
GPU/XLA with memory growth before initialization. Retain the current FP64
diagnostic exception for comparability; no TF32 readiness claim follows.

First implementation checks must cover autoregressive dependence, monotonicity,
analytic affine-inverse parity, DSF endpoint parity with existing saved maps,
parameter gradients, full log determinants, initialization and resume state.
Invalid shared math, data, derivatives or artifacts stop dependent comparisons.
A valid failed fit triggers the next declared control; it is not a campaign
continuation veto. Stop at the priced allocation or if the stated question
cannot be answered without a materially different scientific contract.

Skeptical review: the earlier comparison conflated several properties. Merely
setting K=1 changes the head size and capacity, and adding dummy parameters does
not fix that. The common-output readouts avoid those defects, while admitting
that effective shape capacity changes intentionally. Same numeric seeds across
different constructors do not guarantee identical arrays; copy and check them.
Symmetric exact-identity initialization can defeat the nonlinear arm. A
destructive readout swap without refitting is insufficient. A fixed learning
rate can favor one family, and one-dimensional impossibility does not transfer
to permuted multivariate IAF. Each issue has an explicit control above. The
design is suitable for implementation and pricing; it is not yet a priced,
powered, executable confirmation plan or evidence of an observed effect.

## Executable allocation and second skeptical review

Owner instruction: create, thoroughly review and execute the test. Use the
existing remaining allocation (81,927.76 GPU-process and 84,631.89 CPU-core
seconds at preparation), retaining all prior charges. The same shared campaign
lock and accounting controller must charge each worker once in both ledgers.
Command: `bash scripts/run_neutra_scientific_campaign.sh attribution`.
Outputs use fresh `attribution-*-rN` directories in the existing campaign root;
summary, frozen design and result paths contain `attribution` explicitly.

1. Verify the shared affine control, full Jacobians/inverses, parameter
   derivatives and saved-NAF compatibility on tiny CPU reference fixtures.
   First GPU pricing uses 2,048 updates per arm (two 512-update blocks under
   each optimizer stage to separate tracing from warmed execution), an engineering timing check,
   not scientific fit evidence. Reserve 600 GPU-process / 1,200 CPU-core seconds.
2. Full Gaussian and symmetric one-dimensional mixture controls use one fresh
   fitting seed, 401, and the full 16,384-update recipe. Their claims are
   control-specific, with independent-reference Monte Carlo uncertainty;
   one seed is not a method-success-rate estimate. The analytic best Gaussian
   remains the comparator for the mixture theorem.
3. Two pilot seeds, 409 and 419, on each development target and each conditioner
   run paired affine/nonlinear fits. They measure costs and variability and do
   not contribute to confirmation. Within each pair the order alternates by
   seed, but parameters, sample tensors and optimizer settings match.
4. Freeze eight new paired seeds (431, 433, 439, 443, 449, 457, 461, 463) per
   target/conditioner before confirmation, subject to full-stage affordability.
   This is a bounded replication choice, not a powered guarantee. The primary
   test is an exact one-sided sign test of a positive median paired KL benefit,
   Bonferroni over the four contrasts at familywise .05. Eight positives give
   p=1/256 per contrast, so consistent directional effects are detectable.
   Report paired means and Monte Carlo standard errors descriptively; do not
   equate reference rows to independent fitting replications. An effect of
   .02 nats is a declared practical reference (roughly 2% geometric mean
   likelihood), not a validity threshold. Effects indistinguishable from
   reference noise count conservatively as nonpositive in the sign screen.
5. On pilot data only, compare canonical width-64 IAF, width-134 IAF, and the
   width-134 uncapped-scale variant, with the two already investigated final
   rates (.001 and .0003). Use the same 8,192 first updates at .001 and reset
   Adam for each 8,192-update continuation. Select rates by mean validation
   cross entropy across pilot seeds, separately by target/family. These are
   bounded calibration candidates, not global optimality claims. Give the
   matched affine cMADE control a separate longer continuation up to eight
   times the original update allocation, stopping once its measured training
   time equals its paired nonlinear fit or the cap is reached. Report achieved
   time ratios; an update cap before time equality is an unresolved control.
   Rate and width controls can explain attribution without requiring HMC.
6. If the primary effect is supported, freeze the selected recipe and test
   four independent randomized geometries (two and three modes, generator
   seeds 3101, 3102, 3201, 3202) with two fitting seeds, 479 and 487. These
   support only a bounded transfer check, not a population-wide ranking.
   They do not consume the final campaign target seeds 1103/1104/2103/2104.
   No target is modified or dropped after its result is seen.

Use 131,072 independent target reference rows per fitted pair, an inherited
power-of-two batching convenience, and preserve per-row log-density differences
for later independent inspection. Reference error is estimated from iid rows;
99% normal-approximation intervals are conditional Monte Carlo diagnostics,
not distribution-free bounds. An ambiguous sign needs more independent
reference rows before interpretation, or remains nonpositive. Keep the
32,768-row validation and 2,048 raw-map feature checks from the existing study,
and require the full standard 1,000-point probe at every final endpoint.
Calibration, confirmation and transfer use separate seeds and streams.

The alternate conditioner uses Hoffman block masks and variance-scaled weights
with the same twelve-output head for both readouts. Add deterministic independent
Normal(0,.001) component offset biases in both arms to break the exact symmetry
of the first coordinate. The .001 scale is inherited from the NAF source small
initialization and is an explicit diagnostic hypothesis. Check the resulting
asymmetry and record initial density differences. This is a diagnostic
conditioner adaptation, not an author-default modification.

The measured pricing worker determines later worker limits and full-stage
reservations. Use twice measured setup plus extrapolated steady training time,
with existing two-core CPU ceilings. If conservative reservation exceeds the
remaining budget, revise the allocation before launching that stage; do not
truncate runs or relabel pilot evidence as confirmation. Save phase results
and refresh next actions after every worker. Candidate-screen failure does not
stop later controls. Numerical invalidity stops only dependent contrasts until
repaired and rechecked; preserve and charge failed attempts.

Second skeptical review passed for implementation and pricing: the Gaussian
baseline is analytic, identical arrays are checked rather than assumed from
seeds, no symmetry trap is imposed, sampling data is common, confirmation is
separate, nominal/effective capacity are distinguished, failed candidates are
preserved, controls have explicit limits, and density attribution cannot promote
HMC readiness. Statistical power may be insufficient; the honest terminal
answer may be inconclusive. This campaign need not prove the proposed cause.

Placement check: GPU 1 now has two unrelated scientific workers; GPU 0 also
has unrelated work. GPU 2 has only the remote desktop process. The installed
readiness helper supports indices 0/1 only, so its no-idle result does not
cover GPU 2. Use ordinary trusted execution on GPU 2 with the repository
memory-growth verifier and recorded device provenance. This changes the device
assignment, not hardware class or scientific contract. Desktop sharing means
timing is descriptive and any equal-compute control must report observed times.
The first 40 CPU reference/derivative/compatibility checks passed before launch.

Implementation review before confirmation: the short pricing run measures
compilation and a new reference-evaluation path as well as training. Its 2,072
second pair ceiling is a timeout, not a mean-cost estimate. After all full pilots
finish, reserve confirmation using 1.25 times the largest observed full pilot
wall and CPU cost for each conditioner separately. The 25% margin is an explicit
engineering allowance for run variability, not a probability bound. Enforce the
resulting separate CPU and wall ceilings, and preserve any timeout as an
infrastructure outcome requiring repair, never a negative architectural result.
This replaces the overly conservative aggregate assumption that every worker
consumes its wall timeout times two CPU cores. Actual costs remain fully charged.

The lower-rate and original-rate continuations are also tested on both paired
conditioner families using pilot seeds; they are optimizer sensitivity evidence,
not added confirmation replications. The equal-training-time control restores
the affine map and Adam state from a pilot endpoint, uses fresh independent
training data, and retains the same validation/reference tensors. It stops at
the nonlinear arm's measured cumulative training time, checked every 2,048
updates, or at 131,072 lifetime updates. Record time overshoot and the ratio.
Reference uncertainty and shape checks are recomputed without changing their
criteria. Parameter/source/checkpoint hashes are checked on resume.

Generalization follows a supported cMADE contrast on both development targets,
even if the alternate conditioner reveals an interaction. Requiring success
of the alternate conditioner before that transfer check would silently turn
an explanatory control into a continuation veto. The final interpretation must
then retain any observed conditioner dependence.

The transfer test retains the common .001-to-.0003 training recipe of the
primary contrast; it does not adapt rates to an unseen target's component
count. Pilot family-specific rate selections are saved as optimizer-sensitivity
controls. They are not silently substituted into the primary confirmation or
used as independent evidence of success on unseen targets.

Additional derivative audit: the alternate nonlinear conditioner initially
showed an absolute directional finite-difference discrepancy of 5.41e-7 when
the inverse tolerance was 1e-11 and the difference step 1e-5. A coarser step
sweep then exposed O(h^2) error for cMADE's small normalized weight directions.
Using reference-only inverse tolerance 1e-13 and steps 1e-4/3e-5/1e-5 resolves
both effects with the original comparison tolerance. All 18 expanded reference
and controller tests pass. The production diagnostic inverse and scientific
training tolerances were not relaxed or changed. Failed test attempts are
preserved and charged; this is numerical-check design repair, not a fitted-map
repair. The initial full suite passed 136 checks.

Queue review and supervision: 13 focused controller checks cover complete
positive/negative queues, retained repairs after a failed primary contrast,
terminal resumes without relaunch, and cumulative accounting. A bounded process
supervisor waits for the initial pilot controller to exit, then resumes this
reviewed queue using the same lock and fixed wrapper. It does not interrupt
active workers and is not an AI repair service. A numerical worker failure
stops for inspection; an expected pricing/allocation boundary can proceed with
the reviewed full-pilot reservation. Independent proof/control and pilot
artifacts are checked against source and output checksums before reuse.
