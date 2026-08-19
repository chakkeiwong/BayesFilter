# Austria GenUT NeuTra And Value-Surrogate Strategy

Date: 2026-08-03

> **Partially superseded 2026-08-19** by
> `docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-result-2026-08-18.md`
> for the score-instability premise: the "unstable GenUT score" motivating
> this strategy was the manual recursive score, later found wrong relative to
> the complete finite value program and replaced by forward autodiff of a
> shared primal (exact CPU authority at `T=1,2,20`). The score-free
> proposal-force mathematics below is unaffected as mathematics; whether the
> strategy is still needed on the repaired route is not checked.

Status: `RECORDED_STRATEGY_WITH_SCORE_FREE_RETAINED_HMC_CANARY`

## Direct conclusion

A smooth approximation to the Austria GenUT posterior value is a viable
**proposal-force strategy**.  The retained chain need not use the unstable
GenUT score.  A frozen position-only force can construct a reversible,
volume-preserving leapfrog proposal, while the Metropolis step uses the
original deterministic GenUT posterior value and both endpoint kinetic
energies.  The resulting kernel targets the finite GenUT posterior even when
the proposal force is not its score.

NeuTra makes the simplest version especially attractive.  In frozen NeuTra
coordinates, first try only the Gaussian potential

\[
  \widetilde U_0(z)=\frac12 z^\mathsf{T}z,
  \qquad \nabla \widetilde U_0(z)=z.
\]

If this already yields well-moving, well-accepted proposals after exact GenUT
endpoint correction, neither the GenUT score nor a learned residual force is
needed during retained HMC.

There is one important limitation.  The current repository NeuTra trainer is a
reverse-KL trainer with an explicit target-score boundary.  Its transport
gradient uses the target score even though TensorFlow does not differentiate
through the filter.  Therefore the statement "we will train NeuTra, so the
GenUT score does not matter" is **wrong for the current training procedure**.
The score may be unnecessary during retained HMC, but it is still required to
train a new target-specific chart unless training is changed to a value- or
sample-based procedure.

## Mathematical target and proposal

Let \(T_\psi:z\mapsto\theta\) be a frozen NeuTra transport.  The endpoint
potential in NeuTra coordinates is

\[
 U_{\mathrm G}^{z}(z)
 =-\left\{
     \widehat\ell_{\mathrm G}(T_\psi(z);\omega_0)
     +\log p(T_\psi(z))
     +\log\left|\det J_{T_\psi}(z)\right|
   \right\},
\]

where all GenUT innovation and design tensors \(\omega_0\), numerical
controls, data, and branches are frozen.  Leapfrog may use any frozen scalar
proposal potential \(\widetilde U(z)\), initially \(\widetilde U_0\) above.
The proposal is accepted with

\[
 \alpha=\min\left[
  1,
  \exp\{-U_{\mathrm G}^{z}(z')-K(p')
            +U_{\mathrm G}^{z}(z)+K(p)\}
 \right].
\]

Thus:

- the claimed target is the posterior defined by the frozen finite GenUT
  scalar;
- the proposal computes \(\nabla\widetilde U\), not the GenUT score;
- these quantities are generally different;
- target invariance follows from the symmetric position-only proposal and the
  complete endpoint correction, not from force accuracy; and
- force accuracy affects acceptance, movement, and cost, not the invariant
  finite target.

The shared implementation in
`bayesfilter/inference/neural_force_hmc.py` already implements this corrected
kernel.  `docs/chapters/ch26c_hnn_surrogate_hmc.tex` contains the local
derivation and evaluation ladder.

## Why a value surrogate remains useful

If the Gaussian NeuTra force is insufficient, fit a smooth scalar residual

\[
 \widetilde U(z)=\frac12z^\mathsf{T}z+r_\phi(z)
\]

to cached evaluations of \(U_{\mathrm G}^{z}\).  Differentiating the scalar
residual gives a conservative position-only force.  Because Austria has only
three inferred parameters, an RBF or Matern kernel fit should precede a neural
network: it is easier to inspect, cheap to fit, and naturally smooth.

Dense value accuracy alone does not imply derivative accuracy.  A claim that
\(\nabla\widetilde U\) approximates the GenUT score would require derivative,
Sobolev, or curvature evidence.  This strategy does not need that claim.  Its
promotion evidence is downstream corrected-chain validity and efficiency.

Strathmann et al., *Gradient-free Hamiltonian Monte Carlo with Efficient
Kernel Exponential Families*, Sections 3--5 and Algorithm 1, provide the main
literature precedent: a learned smooth force constructs the proposal and the
target value corrects the endpoint.  Their force is learned by score matching;
the paper does not prove that dense value interpolation automatically yields
correct derivatives.  A local source copy is preserved at
`.localresources/papers/arxiv-1506.02564.pdf`, SHA-256
`5fa6abac306b46273d9bc51f9a9c24becc7e599e18159321e304ff6af250423a`.

## Research intent ledger

| Field | Frozen intent |
|---|---|
| Main question | Can Austria GenUT retained HMC obtain useful corrected proposals without evaluating the GenUT score? |
| Candidate mechanism | Frozen NeuTra coordinates plus a cheap frozen scalar proposal potential |
| Expected failure mode | GenUT endpoint non-replay, invalid endpoint values, chart mismatch, or poor Gaussian-force acceptance/movement |
| Primary viability criterion | At least one predeclared zero-residual configuration passes all endpoint/kernel vetoes, has pooled acceptance at least `0.5`, acceptance at least `0.25` in every chain, at least four nonzero accepted moves per chain, and pooled `mean(||z[t+1]-z[t]||^2)/3 >= 0.01` in the short canary |
| Promotion veto | Nondeterministic same-process endpoint, endpoint mismatch, nonfinite program-valid endpoint at an occupied state, reversibility/energy reconstruction failure, or a stuck chain |
| Continuation veto | No true value-only GenUT endpoint can be made deterministic and batch-native without changing the frozen finite scalar |
| Repair trigger | Viable endpoint but weak zero-residual movement triggers the smooth value-residual arm |
| Explanatory diagnostics | Acceptance above the screen floor, normalized ESJD, energy-error distribution, endpoint time, cross-process drift, short-chain R-hat/ESS, and value-fit residuals |
| Must not be concluded | SGQF chart transferability, exact latent-model inference, GenUT score correctness, score-surrogate accuracy, sampler superiority, convergence, or default readiness |

The acceptance threshold is a cheap viability screen, not an optimal HMC
target and not ranking evidence.  Movement is required so that a small step
size cannot pass by producing nearly identity proposals.

## Smallest honest test

### Behavioral smoke: no new GenUT implementation

The smallest test of the user's immediate hypothesis can precede the true
value-only implementation:

1. Load the frozen SIR-SGQF NeuTra chart as a diagnostic warm start.
2. Call the existing `finite_value_score` route once at each current/proposed
   endpoint, use only its value in the full Metropolis energy, and discard its
   computed score.
3. Use only the frozen Gaussian force
   \(\nabla\widetilde U_0(z)=z\) for every leapfrog kick.
4. In one GPU/XLA process, first require eight exactly repeated finite values
   at the initial state.  Then run four chains for 32 transitions at each of
   `L=10`, step sizes `{0.1, 0.2, 0.4}`: at most `384` new endpoint calls.
5. Check the primary viability criterion above, full energy reconstruction,
   endpoint finiteness/status, and that no returned GenUT score reaches the
   proposal-force callable.

This smoke answers only: *can a Gaussian force in the transferred NeuTra chart
produce useful proposals for the finite GenUT endpoint target?*  It does not
show a computational saving because the existing endpoint still allocates and
computes tangents.  The diagnostic harness may sequentially row-map that
existing scalar endpoint across the four chains; this is an explicit smoke-only
exception and is a hard veto for Stage 1, NeuTra training, or any performance
claim.  The smoke also does not validate the transferred chart, establish
convergence, or make short-chain R-hat/ESS interpretable as confirmation.

If a ladder cell passes, proceed to Stage 0 so the score/tangent computation is
actually removed and the speed premise can be tested.  If all cells fail while
the endpoint remains valid, the smoke rejects this transferred-chart
zero-residual configuration only; it cannot distinguish inadequate SGQF-to-
GenUT transfer from a need for a target-specific chart or residual force.

Execution repair note, 2026-08-04: attempt 1 failed before target evaluation
because TensorFlow-bearing imports preceded GPU memory-growth verification.
That import order was repaired without changing the experiment.  Attempt 2
then found at least one nonfinite endpoint among the four full-scale initial
offsets and stopped before HMC.  The historical seed-`140000` center value at
exactly \(\theta=0\) is finite.  To distinguish an overly dispersed
different-filter initialization from center-target failure, the only approved
repair is the fixed scale ladder `{1, 1/2, 1/4, 1/8, 0}` applied to all four
offsets.  The retry selects the largest scale with four finite endpoints and
records every row.  This posthoc repair changes neither target, force, HMC
ladder, viability criteria, nor the nonclaim that the SGQF chart is not a
GenUT-trained chart.

Attempt 3 completed all `384` endpoint evaluations.  The `0.1` cell passed the
predeclared acceptance and movement screens descriptively: pooled acceptance
`0.625`, per-chain acceptance `0.5625--0.6875`, `18--22` nonzero accepted moves
per chain, normalized accepted ESJD `0.4915`, and finite occupied potentials.
The harness nevertheless marked every cell failed because its full-energy
diagnostic evaluated `abs(inf-inf)` on deliberately rejected nonfinite proposed
endpoints.  This is a reporting defect: the shared kernel permits a
`+infinity` proposed potential as an ordinary support rejection, while energy
reconstruction is defined on finite proposed endpoints.  One additional
`128`-endpoint confirmation of the nominated `0.1` cell is authorized after
masking the reconstruction diagnostic to finite proposals.  The total executed
behavioral budget becomes at most `512` endpoint evaluations.  No force,
target, initialization, HMC seed, acceptance threshold, or movement threshold
changes in this confirmation.

### Stage 0: true value-only endpoint preflight

This is a required implementation prerequisite, not a score repair.

1. Split a genuine value-only route from
   `bayesfilter/highdim/cubature_genut_filter.py::finite_value_score`.  It must
   omit particle and weight tangents, tangent transport, and score
   accumulation rather than call `finite_value_score` and discard its score.
2. Freeze the current-source Austria data, `N=1008` particle/design tensors,
   tuned controls, parameter chart, prior, XLA mode, dtype, and matrix-precision
   setting.  This preserves the scalar studied by the 2026-08-03 antithetic
   campaign; changing these defines a different target.
3. Expose a TensorFlow batch-native rank-two endpoint over parameter rows.  A
   Python row loop, `tf.map_fn`, `tf.vectorized_map`, or scalar fallback is not
   an eligible NeuTra/HMC endpoint.
4. At the center and a small fixed set of posterior-scale perturbations, check
   value-only parity against the value returned by `finite_value_score`,
   same-process repeated evaluation, four-chain batch consistency, and
   separate-process replay on trusted GPU/XLA.
5. Record TF32 and matrix-precision policy explicitly.  Same-process exact
   repeatability is a kernel veto.  Cross-process disagreement is a
   reproducibility veto for pooling independently launched chains, although it
   does not by itself disprove invariance of one fixed in-process target.

Stop before HMC if same-process values are not deterministic or the independent
value-only scalar does not match the old scalar at valid points.  The existing
route exposes only `finite_value_score`; calling it and discarding the score
would not test the proposed computational saving.

### Stage 1: zero-residual NeuTra canary

Use the already frozen SIR-SGQF NeuTra chart only as a **diagnostic warm start**.
It has the same three physical parameters and previously gave zero-residual
acceptance `0.9770` for the SIR-SGQF posterior, but it was trained for SGQF,
not GenUT.  It is not target-matched GenUT evidence.

Run in one GPU process so four chains share one endpoint program:

- arm A: identity/raw coordinates with their fixed Gaussian baseline force;
- arm B: frozen SIR-SGQF NeuTra coordinates with
  \(\nabla\widetilde U_0(z)=z\);
- endpoint: the same value-only GenUT posterior in both arms, including prior
  and the appropriate chart Jacobian;
- tuning screen: `L=10` and step sizes `{0.1, 0.2, 0.4}`, inherited from the
  successful SGQF `0.2/10` setting only as a centered warm-start ladder;
- budget: `4` chains, `64` transitions for each of two arms at each ladder
  cell, then `256` transitions per chain for the selected viable arm; at most
  `2,560` GenUT endpoint evaluations plus preflight;
- hardware: trusted GPU, XLA on, memory growth verified, and a fresh versioned
  output directory.

Required kernel checks are endpoint parity, exact force identity, proposal
reversibility, full potential-plus-kinetic energy reconstruction, finite target
status, all-chain movement, acceptance, normalized accepted squared jumping
distance, endpoint calls, force calls, and wall time.  Short-chain modern
R-hat and bulk/tail ESS are explanatory only and cannot establish convergence.

Decision:

- If arm B passes the viability criterion with nontrivial movement, retained
  GenUT-score evaluation is unnecessary for the next longer validation.  Do
  not fit a residual merely because one could be fitted.
- If arm B fails but the endpoint is valid, this rejects only transfer of the
  SGQF chart or the zero-residual mechanics.  It does not reject a
  target-specific GenUT NeuTra chart.
- If both arms fail through endpoint invalidity, repair the GenUT value route;
  changing the proposal force cannot repair an invalid MH target.

### Stage 2: value-surrogate residual, only if Stage 1 needs it

Cache exact GenUT transformed values at posterior-relevant states and fit an
RBF/Matern scalar residual.  Freeze the fit before sampling.  Compare it with
the Stage 1 zero-residual arm under matched endpoint, step-size tuning budget,
chains, seeds, and hardware.

Held-out value error, finite-difference force checks, and curvature are
explanatory or learner-veto diagnostics.  The primary downstream comparison is
corrected-chain movement and ESS per wall-second after charging data generation
and fitting.  With a short run and one seed, differences remain descriptive.

## NeuTra training paths after the canary

If a target-specific GenUT chart is required, there are three distinct paths:

1. Repair and validate the GenUT score, then use the existing reverse-KL
   NeuTra trainer.  This does not remove the score problem during training.
2. Generate GenUT-posterior draws using the value-only corrected kernel or a
   value-only random-walk/independence kernel, then fit the transport by
   forward-KL/maximum likelihood on those draws.  This removes target-score
   use but requires a new reviewed, batch-native GPU training procedure and
   adequate effective training samples.
3. Optimize the chart with a derivative-free or value-surrogate objective.
   This is a new method and needs its own downstream validation; held-out
   value loss alone cannot promote it.

Path 2 is the most direct score-free training strategy if the warm-start chart
is inadequate, but it is deliberately outside the small canary.

## Skeptical plan audit

| Audit risk | Finding and response |
|---|---|
| Wrong baseline | Avoided: raw/identity Gaussian force is included; SGQF success is motivation only, not the GenUT comparator or promotion evidence. |
| Proxy promoted to criterion | Avoided: the behavioral smoke has explicit acceptance-and-movement viability screens; interpolation RMSE, acceptance above the floor, and short-chain ESS are not superiority or convergence evidence. |
| Missing stop condition | Fixed: endpoint parity/determinism, invalid occupied endpoints, kernel invariants, and all-chain movement can stop the canary. |
| Unfair comparison | Both arms use the same finite GenUT target, endpoint budget, ladder, chains, and hardware; chart-Jacobian terms are included where required. |
| Hidden training assumption | Exposed: current reverse-KL NeuTra training consumes the target score; the reused SGQF chart is only a diagnostic warm start. |
| Stale target controls | Stage 0 binds current-source data, design, controls, chart, dtype, XLA, and matrix precision and checks the new scalar against the existing callable. |
| Environment mismatch | All GPU/CUDA work uses trusted permissions, GPU memory growth, TensorFlow/TFP, batch-native execution, and XLA. |
| Artifact cannot answer question | The canary records exact endpoint calls and uses no GenUT score calls, so a pass directly answers whether retained HMC can move without the score. |

Verdict: the staged strategy survives audit.  Run only the behavioral smoke
initially.  A pass justifies Stage 0 followed by the Stage 1 canary;
target-specific NeuTra retraining and a value surrogate are repair phases, not
prerequisites silently folded into the first test.

## Default and assumption audit

| Choice | Provenance and status | Failure mode | Early diagnostic |
|---|---|---|---|
| `N=1008` and current GenUT controls | Existing Austria antithetic campaign; frozen baseline, not a universal default | Scalar differs from prior evidence or is numerically invalid | Value parity and repeatability preflight |
| SGQF NeuTra chart | Existing same-model, different-filter artifact; warm-start hypothesis | Cross-filter posterior geometry differs | Arm A/B canary and endpoint validity |
| Gaussian force | Existing corrected-HMC zero-residual baseline | High rejection or negligible movement | Acceptance plus normalized ESJD |
| `L=10`, step sizes `{0.1,0.2,0.4}` | Centered on prior SIR-SGQF `0.2/10`; convenience screen | Misses a viable GenUT scale | Treat failure as tuning/transfer evidence, not method rejection |
| Smoke movement floors | Convenience viability screens in standard-normal NeuTra coordinates, not scientific defaults | A weak force passes through high acceptance alone, or finite-sample noise rejects a viable force | Require both per-chain moves and normalized ESJD; treat failure as diagnostic only |
| Four chains and short draws | Bounded diagnostic budget | R-hat/ESS too uncertain for convergence claims | Explicit short-canary nonclaim |
| RBF/Matern before neural residual | Three-dimensional smooth-fit hypothesis | Tail extrapolation or oversmoothing | Held-out trajectory split and corrected-chain alarms |

## Evidence and decision tables

| Evidence contract field | Required evidence |
|---|---|
| Exact comparator | Same frozen value-only GenUT posterior at both endpoints |
| Primary canary criterion | Endpoint/kernel gates pass and at least one zero-residual configuration meets the pooled/per-chain acceptance, move-count, and normalized-ESJD floors in the research intent ledger |
| Hard vetoes | Scalar mismatch, same-process nondeterminism, invalid occupied value, nonfinite energy, reversibility/energy failure, hidden score use by the proposal, or a stuck chain; after the behavioral smoke, a scalar/row-mapped endpoint is also a Stage 0/1 veto |
| Explanatory only | Cross-process drift when chains are not pooled, acceptance above the floor, ESJD magnitude, short-chain R-hat/ESS, surrogate loss, and runtime differences |
| Preserved result | Fresh versioned result, run manifest, exact command, source identity, environment, GPU/memory policy, XLA/TF32 policy, seeds, timings, endpoint counts, and plan path |

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Current strategy record | Not run | Existing GenUT value/score route has invalid and replay warnings | No true value-only endpoint and no GenUT-trained score-free chart | Run the `384`-endpoint behavioral smoke; on pass implement Stage 0 and run Stage 1 | No GenUT HMC viability, convergence, speedup, or score irrelevance result yet |

| Inference-status item | Current status |
|---|---|
| Hard veto screen | Not run; recent antithetic campaign warns of invalid evaluations and cross-process drift |
| Statistically supported ranking | None |
| Descriptive-only evidence | Prior SIR-SGQF zero-residual acceptance `0.9770` and all anticipated canary performance metrics |
| Default readiness | Not established |
| Next evidence needed | The bounded existing-endpoint behavioral smoke; on pass, a deterministic batch-native GenUT value-only endpoint and the fuller zero-residual canary |

## Post-audit red team

The strongest alternative explanation for a successful warm-start canary is
that the GenUT posterior happens to be locally close to the SGQF posterior near
the short-run initialization while differing in tails or another mode.  A
longer run from dispersed initial states and posterior agreement against a
separate value-only reference would be needed to overturn that explanation.
Conversely, failure of the warm-start canary would most weakly implicate the
research strategy: it could simply mean that a chart trained on SGQF does not
transfer to GenUT.
