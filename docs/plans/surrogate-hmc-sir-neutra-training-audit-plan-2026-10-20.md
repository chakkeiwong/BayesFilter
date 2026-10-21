# SIR NeuTra training: source audit and integration plan

Date: 2026-10-20. Branch: `ledh-graceful-failure-execution-20260917`.
Inspected HEAD: `10884528ca8b1a7463895256d30b9655d69c0955`, plus the existing
working tree. This is an audit and implementation plan, not authorization or
evidence for a new long training run. Existing unrelated changes are preserved.

## Active checkpoint

2026-10-21: the owner authorized committing the baseline, implementing the
batching repair, reviewing the plan and executing it. The implementation and
bounded capacity diagnostics are tracked in
[the active execution plan](sir-native-batching-execution-2026-10-21.md).
Native model/proposal/reset operations and all-parameter propagation now retain
B without tracing a model body per row. Shared Gaussian and transport contractions
avoid pair-by-state/sensitivity intermediates; optional native batched sharding
is implemented. The isolated source-tree suite passed 119 checks, including
row/permutation, sharding and public-entry regressions.

Full-size old/new FP64 scores agree within 4.35e-11 relative vector norm, but
FP32 row 3 differs by 5.769% (old) and 8.057% (native) from this common
reference. The owner has retired the 1% scientific acceptance criterion:
[compare against the adapted Zhao–Cui reference with uncertainty](sir-batching-materiality-policy-2026-10-21.md).
The nominal batching change is accepted as immaterial for continuation; the
other rows lack matched reference/replication evidence. Bounded capacity testing
is complete: GPU B32 uses 4.04 GiB, and logical B128/native S4 completes in
0.761 GiB with exact B4 output parity. CPU B16/B32 reach their 600-second caps
without a completed evaluation. See the
[batching result](sir-native-batching-results-2026-10-21.md).
These are engineering diagnostics, not numerical or training admission.
The original capacity benchmark remains comparison evidence in
[sir-fp32-batch-capacity-results-2026-10-20.md](sir-fp32-batch-capacity-results-2026-10-20.md).
No optimizer update or HMC campaign is part of this batching execution. The
matched target-to-trainer adapter, complete optimizer memory check and trained
SIR map remain missing. This is a Phase 1 prerequisite, not completion of the
whole integration plan. The next implementation step is the Phase 1
target-to-trainer adapter and complete-update checks. FP32/FP64 discrepancies
remain reported diagnostics, not a percentage-based prerequisite for that work.

The prior paper, author-source and consumer-call-chain audit is complete: 57
CPU tests passed, one optional historical-artifact check skipped, and the
independent standard-gradient check passed. See
`docs/plans/artifacts/sir-neutra-training-audit-20261020/`. No SIR NeuTra map
has been trained; the master still lacks a NeuTra phase. The matched SIR
training callback and frozen-map HMC integration remain Phase 1 onward below.
The optional exact fixed-index preparation used by the FP32 benchmark passed
eight focused checks, including unchanged small-SIR outputs and no FP64 types
in optimized XLA code; this is not large-SIR numerical admission.

## Audit evidence contract

The comparator for the standard training gradient is direct differentiation of
the same finite-sample reverse-KL loss on a smooth coupled analytic target.
The comparator for source operations is the pinned author's
`Invert(MaskedAutoregressiveFlow(...))`, evaluated with the same conditioner,
and its explicit algebraic expression. The diagnostic uses CPU/XLA with GPU
deliberately hidden, at most two minutes, and no SIR filtering or optimization
campaign. Finite gradient disagreement beyond FP64 rounding, a mismatched
bijector formula, or a missing parameter gradient invalidates the relevant
implementation claim. The source convention comparison is explanatory about
source equivalence, not a measure of posterior or training quality. This test
cannot establish SIR score accuracy, GPU capacity, trained-map quality, or HMC
convergence. Complete results and environment are saved alongside the diagnostic.

Skeptical pre-run audit: a Gaussian-only zero-gradient test could hide a wrong
chain rule, so use a coupled quartic target and nonzero parameter gradients.
Do not import the old author program into the runtime or claim bitwise 2019
reproduction from a modern TFP primitive. A same-conditioner difference tests
source parameterization; it does not invalidate the paper's direct affine
autoregressive equation. This bounded plan passes that audit.

## Findings and completion status

The reusable learning algorithm exists. The current SIR campaign does not call
it. Completing this work requires target adapters, a frozen-map HMC binding,
and a target-specific training protocol, rather than another implementation of
IAF, Adam, or reverse KL.

| Component | Status on 2026-10-20 | Evidence or missing work |
|---|---|---|
| Canonical configured IAF, inverse, log determinant and score pullback | EXISTS; generic tests pass | `neutra_transport.py` delegates to `neutra_transport_core.py`; use `hoffman_author_iaf`, not the generic legacy-mask default |
| Standard reverse-KL training using supplied first scores | EXISTS; chain rule checked independently | `reverse_kl_evaluate`, plus the coupled-quartic diagnostic below |
| Batched, compiled Adam updates and checkpoint/restore | EXISTS; generic tests pass | `NeuTraTransportTrainer`; rollback covers weights and Adam state |
| Original paper and maintained author-source comparison | DONE, with an explicit difference | Local stages use the paper's direct affine formula; author code uses an inverted MAF with a different same-weight parameterization |
| Current SIR LEDH value and analytical score kernel | EXISTS | `make_batched_score_filter`; the campaign force wrapper discards the returned value |
| Matched SIR training callback, including scientific prior and chart | MISSING | Must expose value, corresponding score, and per-row validity together; never combine the independent endpoint value with an LEDH score |
| SIR training regularity and capacity evidence | NOT CHECKED | Validity over training draws, gradient noise, memory, compilation and update cost at N=4032, T=50 |
| SIR training search, heldout assessment, reproducible resume | MISSING | Core checkpointing exists; SIR scheduler, random-stream state, budget and selection records do not |
| Frozen nonlinear map around surrogate-force HMC | PARTIAL; not connected | Pullbacks and transformed-target containers exist; the ordinary tuning binding explicitly rejects transformed coordinates |
| Shared sequential inference for this transformed force/endpoint pair | MISSING WIRING | Must preserve the endpoint, campaign precision, 70% tuning target and 20,000 retained-draw cap |
| SIR NeuTra phase in the master | MISSING | Current master has endpoint audit, filter tuning, mechanics and inference, with no transport-training phase |
| SIR trained-map quality and posterior convergence | NOT CHECKED | No such result is established by this audit |

The inspected SIR case (`sir_d18` in the master) has three inferred parameters, 18 state coordinates,
nine observation coordinates and T=50. NeuTra acts on the **three parameter
coordinates**, not the 18-dimensional latent state or the particle cloud.

## Paper and author-code correspondence

Primary paper: Hoffman et al., *NeuTra-lizing Bad Geometry in Hamiltonian
Monte Carlo Using Neural Transport*, arXiv:1903.03704v1. The technical sections
inspected were §§2.2–2.3, 3.1 and 4.1.1. The locally stored PDF and pinned source
files are listed in the source manifest beside the downloaded author code.
The arXiv submission history inspected for this audit lists v1 only. The code
snapshot is maintained Google Research source at revision
`222560a989700a4e247e4aa9faadcbdd70188a7b`, not a reconstructed 2019 environment.

| Topic | Paper / author source | Current implementation and verdict |
|---|---|---|
| Variational objective | Paper Eq. (3); author `neutra.py:929` forms mean(log q − log p) | Standard estimator differentiates the same reverse-KL objective up to parameter-independent terms. Correct when supplied values and scores describe the same target |
| Map architecture | Paper Eq. (4), §4.1.1: affine autoregressive stages, three stages, two ELU hidden layers, reversals | Canonical local profile supplies those mechanisms. Masks and free scale bias have focused tests. Width is a target-specific choice |
| Exact stage convention | Author `neutra.py:132` wraps MAF in `Invert` | Local direct-affine stage is not the same finite parameterization with identical conditioner weights; see derivation below |
| Scale head | Author `utils.py:734–763` has an optional pre-bias tanh cap and a free bias outside it | Local `b+c*tanh(h/c)` preserves that placement. c=2 is a local choice, not a universally required paper setting |
| Initialization | Author `neutra.py:122`, `utils.py:612`: small nonzero variance-scaled kernels | Local author profile uses variance scale 0.02 and stateless random draws. This is not identical RNG state or a bitwise port |
| Training schedule | Paper §4.1.1: Adam, 5,000 updates, batch 4,096, initial LR 0.01, reductions at 1,000 and 4,000; source `neutra.py:943–949` | Core supplies Adam but no SIR schedule/search controller. Copying that batch size into an N=4032 filter is unjustified |
| Optional path gradient | Not the standard estimator used for the paper's Eq. (3) implementation | A separately documented estimator exists. It is not necessary for the first SIR integration; no universal variance advantage is claimed |
| HMC target | Paper Eq. (5); author `MakeNeuTra`, `neutra.py:687` | The paper transports one target. Here LEDH trains the map and supplies proposal forces, while a separate fixed-filter endpoint supplies MH values. This is an explicit local adaptation |
| Export and diagnostics | No paper requirement for 1,000 successful local probe points | Local `finalize()` has that rule. It must not silently become an additional campaign stopping condition |

For one local stage, write the conditioner outputs as mu(x) and s(x). The
code computes

\[
 y_i=x_i\exp(s_i(x_{<i}))+\mu_i(x_{<i}),
 \qquad \log|J|=\sum_i s_i(x_{<i}).
\]

With the same conditioner outputs, the author's inverted TFP MAF computes

\[
 y_i=(x_i-\mu_i(x_{<i}))\exp(-s_i(x_{<i})),
 \qquad \log|J|=-\sum_i s_i(x_{<i}).
\]

Both are invertible affine autoregressive constructions. The first matches the
paper's displayed Eq. (4). Replacing the author outputs by
`s_author=-s_local` and `mu_author=-mu_local*exp(-s_local)` gives the same
mathematical map. That nonlinear output transformation is not the identical
finite neural-network head or optimizer trajectory. Stage application order
also follows the respective composition conventions. Consequently, an exact
same-weight author-code reproduction claim is wrong; correctness of the
change-of-variables formula is a different question and passes the checked
tests. Preserve the owner-designated canonical architecture and document this
difference. This audit supplies no evidence that changing that convention
would improve SIR.

## The two targets and the required chain rules

Let u be the current standardized, unconstrained parameter vector, and
theta=C(u) the campaign's model-parameter chart. `case.log_prior(u)` already
includes the scientific prior's required natural-parameter and chart
Jacobians. Define the auxiliary log density used for training as

\[
 \ell_L(u)=\log p_u(u)+\widehat\ell_{\rm LEDH}(C(u)),\qquad
 s_L(u)=\nabla_u\log p_u(u)+D C(u)^T
                   \widehat s_{\rm LEDH}(C(u)).
\]

Both LEDH outputs must come from the same frozen finite filter program and
particle designs. In the current diagonal chart this pullback is componentwise
multiplication by the chart derivative. A score missing the prior or chart
term is wrong for this training objective. Counting the prior Jacobian again
would also be wrong.

For z distributed as standard normal and u=T_phi(z), train with

\[
 L(\phi)=\mathbb E_z[-\ell_L(T_\phi(z))-\log|\det D_zT_\phi(z)|],
\]

which is reverse KL up to constants whenever the auxiliary density and
expectation are well defined. For one batch the required gradient is

\[
 \nabla_\phi L_B=-\frac1B\sum_{b=1}^B
 \left[(D_\phi T_\phi(z_b))^T s_L(T_\phi(z_b))
       +\nabla_\phi\log|\det D_zT_\phi(z_b)|\right].
\]

The existing core implements this chain rule by attaching the supplied first
score to a stopped target value, then differentiating the map and log
determinant. It does not need a target Hessian or autodifferentiation through
the particle filter. Its independent diagnostic agrees with direct
differentiation of the identical finite-batch loss to FP64 rounding.

This conclusion is conditional: a mathematically valid trainer cannot repair
an incorrect SIR callback. The actual SIR value/total-score pairing, validity
over training draws and integrability are not established by the toy check.
Persistent invalid batches are a repair trigger. Repeatedly discarding them
and training only on successful draws would change the sampling distribution
and would not establish the stated reverse-KL objective.

For the current SIR MH endpoint, retain

\[
 \ell_E(u)=\log p_u(u)+
  \log\left\{\frac14\sum_{r=1}^4
           \widehat L_{{\rm BPF},r}(C(u))\right\},
\]

using the four existing fixed particle-filter designs, each at N=4032. This is
the log of an average of likelihoods, not an average of log likelihoods. With
the trained map frozen, use

\[
 U_E^z(z)=-\ell_E(T(z))-\log|\det DT(z)|,
 \qquad
 g_L^z(z)=-DT(z)^Ts_L(T(z))-\nabla_z\log|\det DT(z)|.
\]

The force is the gradient of the auxiliary transformed potential; MH uses the
endpoint potential. A deterministic position-only force still gives
volume-preserving momentum kicks and a reversible leapfrog proposal after
momentum reversal. Thus the MH ratio can use U_E even when the force differs
from its gradient, provided those implementation conditions hold and invalid
proposals are handled consistently. The nonlinear adapter must be tested for
these conditions; its existence as a mathematical formula is not wiring
evidence.

This samples the fixed numerical endpoint density if the chain converges. It
does not make a finite fixed-design particle-filter likelihood equal to the
exact SIR likelihood. Nor does training under ell_L prove that the resulting
map straightens ell_E. Endpoint roughness and training/endpoint disagreement
remain possible explanations for poor downstream performance.

## Call-chain audit and specific adapters

The current chain is `stage_worker -> runtime.inference -> bindings ->
bind_neural_force_hmc_tuning_runner -> shared tuning/sequential sampling`.
No canonical transport trainer is called along that chain.

| Connection | Existing code | Required change / check |
|---|---|---|
| SIR target to trainer | `surrogate_hmc_five_model.py::bindings`; `covariance_proposal_score_batch_tf.py::make_batched_score_filter` | Add a reusable callback `(u[B,3]) -> (ell_L[B], s_L[B,3], valid[B])`, preserving analytical total scores and row diagnostics. Reuse the existing filter and prior; do not reuse the force wrapper's discarded-value interface |
| Target batching | Fixed-signature filter with native batched recursion and heavy batched transport kernels | Start at B=2, the recorded force scope. Model construction/observation pieces include static Python unrolling at trace time; graph size can grow with B. Measure larger batches rather than claiming all operations are free of per-row expansion |
| Training state | `NeuTraTransportTrainer` owns weights, Adam slots and iteration | Add the latent stateless-seed counter, LR schedule, budget, target identity, and phase state to the SIR campaign checkpoint. Do not rebuild a legacy trainer |
| Post-training bridge | `PostTrainingProbe` accepts `value_score_status(points,beta)` and `parameter_dim` | Provide this protocol using the same matched callback. Use explicit native batches; the generic default of 20 is not a SIR capacity decision. Probe row count and cost must be declared |
| Freezing | `frozen_payload`, `load_configured_frozen`, `as_dtype` | Reuse these APIs, bind both auxiliary and endpoint identities, and test serialization round trips. An FP64 frozen evaluation copy is a precision conversion of FP32 weights, not FP64 retraining |
| Transformed density and score | `FixedTransportValueScoreAdapter` | Reuse for a matched auxiliary density only. Do not advertise a value from ell_E and a score from ell_L as one exact value/score pair |
| Surrogate-force inference | `bind_transformed_neural_force_target` exists; ordinary binder rejects non-raw coordinates | Extend the shared fixed-transport transition/tuning connection to accept a distinct frozen force and endpoint. Do not relabel latent coordinates as raw to bypass that rejection |
| Endpoint parity | `validate_value_only_endpoint_parity` compares an endpoint with its adapter's scalar | Here compare transformed ell_E with the SAME raw ell_E plus map Jacobian. Comparing ell_E to an auxiliary LEDH adapter is the wrong invariant |
| Sequential sampling | Existing shared `neutra_hmc` controller | Pass the campaign's precision, sequential limits, diagnostics and transition binding explicitly. Add route-ledger/wiring coverage; do not create another sampling controller |
| Master integration | `scripts/run_surrogate_hmc_five_model_master.py` and `restore_selected_scope` | Add training and frozen-map phases and refresh the next-phase record at each boundary. Load the selected scientific prior, arithmetic, filter controls and source identity before constructing callbacks |

Two reuse hazards are concrete. `neutra_shared_procedure.py` and its current
sequential wrapper explicitly enable TF32, while this SIR campaign disables
it. The end-to-end helper also constructs a 10,000 retained-draw maximum,
whereas the campaign permits 20,000. These wrapper settings need explicit
configuration; inheriting them would change the experiment. The current
selected SIR precision exception is FP64 recursion with FP32 pair operations.
Keep that scope, FP32 trainable transport/Adam, and TF32 off until an explicit
scope-specific comparison supports a change. FP32 remains the campaign
preference; silently replacing the selected mixed SIR route is not reuse.

The generic `finalize()` currently requires all 1,000 probe rows to be valid.
Use the public diagnostic and freezing primitives separately for this campaign
and record probe failures. Auxiliary-target failure at a diagnostic point is
not, by itself, authority to stop endpoint HMC. An invalid current state,
noninvertible/nonfinite map, corrupted checkpoint or invalid transition
implementation does require repair. This is consistent with retaining MH
rejection of bad proposals and reporting finite energy errors without adding
a new energy-based stopping rule. These probe evaluations are not a second
posterior validation chain.

Historical `neutra_campaign.py`, the old training path in
`neutra_end_to_end.py`, and the earlier SIR SGQF demonstration are not eligible
shortcuts to a current SIR-trained canonical map. The shared procedure is a
frozen-map tuning orchestrator, not a target-specific transport trainer. FAB,
weighted and tempered trainers are optional different objectives; their full
paper/source audits are outside this bounded audit and none is needed for the
first standard reverse-KL implementation.

## Implementation and experiment sequence

These are phases of the proposed SIR master repair, not separate changes of
research direction. Only Phase 0 has been executed. Implementation and SIR
training have not been started as part of this audit request.

### Phase 0 — source and core audit: DONE

Preserve the paper and author sources, audit the current consumer call chain,
run the focused generic suite and independent chain-rule/source diagnostic,
and qualify the implementation reference. The results are recorded below.

### Phase 1 — matched training adapter: MISSING

The capacity benchmark exposed incomplete native batching and large temporary
tensors in the target it would consume. Apply the
[shared batching repair design](sir-native-batching-repair-design-2026-10-20.md)
within this phase. The native implementation now exists. The former full-size
percentage-only FP32 veto is retired by the owner materiality correction;
retain implementation invariants and reference/uncertainty reporting while
proceeding with the adapter and complete-update checks below.

Add a small campaign adapter module, proposed path
`bayesfilter/testing/surrogate_hmc_sir_neutra.py`, using the existing SIR case,
selected-scope loader, batched filter, prior and canonical trainer. Keep the
filter's numerical implementation in its current authority. The adapter must:

1. Load the actual literature-prior selection and exact filter scope. Explicitly
   restore N=4032, T=50, chunk=2016, selected controls, mixed SIR precision and
   TF32 off. A bare `load_case()` can select benchmark priors/uniform precision
   and is not an acceptable substitute.
2. Return the matched auxiliary value, total parameter score and per-row status
   with a stable `[B,3]` signature. Preserve `steps_completed` and failure
   diagnostics; an incomplete likelihood sum is not a valid training value.
3. Provide the post-training bridge and a frozen-map identity containing both
   the auxiliary training density and the distinct inference endpoint.
4. Add focused tests, proposed `tests/test_surrogate_hmc_sir_neutra.py`: chart
   and prior chain rules, values/scores from the same filter call, negative
   tests for a swapped endpoint or wrong prior, invalid-batch rollback, and
   frozen-map round trips. Use analytic toy callbacks for fast wiring tests;
   mark real SIR GPU checks separately.

Completion evidence: the trainer consumes this callback, not merely a function
with the right name; fixed-noise updates before and after checkpoint restore
agree. A same-program SIR directional finite-difference diagnostic is still
needed in Phase 3. It checks the supplied analytical score and never replaces
that score with autodiff in the claim-bearing lane.

### Phase 2 — transformed HMC binding and master wiring: MISSING

Extend the existing fixed-transport/shared tuning path to carry separate
position-only force and endpoint callbacks. Reuse map pullbacks, transition
mechanics, tuner and sequential controller. Supply explicit precision and
sequential configuration rather than the current TF32/10,000-draw wrapper
constants. Record original u/theta draws as well as latent z; convergence in
latent marginal coordinates alone is insufficient.

Before training an expensive map, use an identity map and a known nonlinear
map on an analytic toy problem to test endpoint/Jacobian parity, force
pullback, reversibility, status handling, serialization, and the exact
consumer-to-shared-controller connection. An identity map must reproduce the
existing raw force/endpoint arithmetic. Nonlinear tests must include the log
determinant gradient; an identity-only test would miss its omission.

Add master phases `neutra-capacity`, `neutra-train`, `neutra-freeze`, and
`neutra-inference`, with a saved next-phase decision after each. The new route
must be included in the existing NeuTra HMC route ledger and enforcement test.
Do not use the old SGQF target, an old trained map, or a fresh residual-force
network as a substitute for this transport integration.

The intended command interface extends the existing master, for example:

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_five_model_master.py stage-worker --phase neutra-capacity --model sir_d18 --priors literature-model-priors-v1 --selection <selected-sir-tuning.json> --output <fresh-versioned-output> --gpu <assigned-gpu> --budget-seconds 3600
```

This is an implementation specification, **not an executable command today**:
the new phase and its deadline enforcement are missing, and the placeholders
must be resolved from the current campaign records. The repaired master must
enforce each worker's budget, write its fully resolved argv/environment to the
manifest, and use trusted GPU execution. Training, freezing and inference use
the same entry point with their declared phase configurations. A successful
phase refreshes and advances the saved plan automatically within the remaining
budget. No allow-list broadening or fresh approval per local retry is part of
this design; ordinary tool permission requirements still apply.

### Phase 3 — actual SIR GPU capacity and training pilot: NOT RUN

Start with a bounded **four-GPU-hour pilot allocation proposal**, of which at
most one hour is capacity/compile diagnosis and three hours are training
calibration. This is a cap, not a runtime forecast or a claim of enough
training. Before launch, reconcile it with the campaign's recorded remaining
budget; this audit has allocated or consumed none. Do not assume that an old
48-hour extension remains unspent.

Use trusted GPU access, memory growth configured before initialization,
TensorFlow/TFP, stable signatures and XLA. For B=2, measure compilation
separately, then repeated filter calls and actual optimizer updates. Record
host/device peak memory, graph size, validity, gradient norms, update time and
target-evaluation time. Try B=4 and then B=8 only if the previous size leaves
measured headroom and the one-hour capacity cap permits it. A changed batch
scope needs the repository's matching tuning identity and parity checks, not a
caller-stamped copy of B=2 evidence. Stop escalation after a resource failure;
keep the last eligible native batch. Never replace the batch with scalar
row-mapping or introduce pfor.

Check the real SIR value/score chain rule using a small declared bank covering
the center, dispersed starting region and parameter-tail directions. Compare
directional finite differences over several step sizes, with the selected
arithmetic's rounding floor recorded. A derivative disagreement is an adapter
or filter repair trigger; a single tiny finite-difference step is not an
oracle. Probe both ell_L and ell_E at those points to expose target mismatch,
without selecting parameters to make that discrepancy disappear.

Use the remaining pilot for short updates, rollback/resume, batch-variance and
scale-cap diagnostics. Observe pre-cap activations, free scale biases,
Jacobian conditioning, invalid batches and optimizer state. Post-update finite
weights alone are insufficient: the core does not re-evaluate the target
after each update, so add declared independent monitoring batches. Their loss
and geometry are diagnostics, not posterior convergence evidence.

The capacity result supplies the cost estimate:

`training seconds = compilation + sum(candidate updates * measured update seconds) + probe seconds`.

For HMC use measured force/endpoint/transition times and the actual tuning and
adaptive sampling limits. Publish those estimates and the remaining budget
before advancing; an arbitrary promise that a fixed update count is enough
would not answer the research question.

### Phase 4 — target-specific training and freeze: MISSING / NOT RUN

Use the canonical three-stage IAF and standard reverse-KL estimator first.
The initial search is deliberately small: widths `(3,3)`, `(6,6)` and `(9,9)`;
learning rates 0.001 and 0.01; one declared search seed per configuration.
These are six hypotheses, not six certified defaults. Evaluate staged budgets
of 128, 512, 1,024 and at most 4,096 updates, extending only viable candidates
within the measured budget. The paper's 5,000 updates with batch 4,096 do not
justify 4,096 updates with a small SIR batch; lack of stabilization means
under-training, not automatic map acceptance.

Use heldout latent draws, disjoint from optimizer draws, for paired loss and
geometry assessment at the rungs. This separation concerns parameter draws
for the same observed SIR posterior; it does not change the observation data.
Nominate up to two configurations, then train each with three fresh seeds.
Use paired uncertainty on common heldout draws and across training seeds. When
loss differences are unresolved, label configurations indistinguishable and
use a predeclared lower-cost tie break for nomination only. Do not rank
posterior performance from ELBO alone.

The search and three-seed fits have a proposed **12-GPU-hour ceiling beyond
the pilot**, subject to the Phase 3 cost calculation and available campaign
budget. Save partial results if the cap prevents adequate fitting; do not
promote a short run because it exhausted its allocation. The maximum new
pre-inference request is therefore 16 GPU hours, not a claim that all 16 are
needed. Sequential HMC has a separate measured-cost allocation at Phase 5;
its cost is currently missing. No long run is launched by this document.

Use deterministic seed families recorded per configuration and purpose:
search, fresh fit, monitoring, heldout assessment and HMC must have disjoint
streams. Persist the next latent seed and LR-schedule position with Adam so a
restart does not reuse training noise or reset the learning rate. A decayed
schedule is available in the author source; add it to the SIR orchestrator
with explicit breakpoints after the pilot, rather than pretending the current
constant-LR configuration implements the paper's schedule.

Fit initialization and any outer affine transform from declared calibration
draws only. Nonconverged old warmup can inform a labeled initialization
hypothesis, not posterior moments or coverage. Compare with an identity outer
map early. Never initialize from a historical noncanonical NeuTra artifact.

Freeze weights and numerical controls before HMC. Save the ordinary checkpoint,
frozen payload, objective/target identities, precision-conversion provenance,
probe results and selected configuration. Use the public probe with a measured
batch size and a predeclared row budget; do not invoke the generic mandatory
1,000-valid-point export gate unchanged. Persistent map or target failures
trigger the relevant repair, with no silent density clipping or invented
surrogate objective.

### Phase 5 — downstream inference and comparison: NOT RUN

Run four dispersed chains through the shared sequential controller using the
same scientific prior, fixed numerical endpoint and N=4032 as the existing
campaign. The frozen map changes coordinates, not the endpoint. Tune acceptance
toward 0.70. Preserve the current adaptive warmup convention: minimum 2,000,
latest 1,000-transition assessment window, modern R-hat <=1.05, maximum 10,000
per chain. Retained sampling grows cumulatively toward modern R-hat <=1.01,
with the owner's 20,000-per-chain ceiling and the campaign's declared ESS
criteria. Do not require every sliding retained window to pass. There is no
additional posterior validation chain, and all eligible retained draws count
toward inference. Finite energy errors remain explanatory; invalid states and
broken transition arithmetic require their proper rejection/repair handling.

Construct three simple comparators using the same endpoint and force budget:
raw identity coordinates, tuned diagonal affine coordinates and tuned dense
affine coordinates. These address no deformation, marginal scaling and linear
correlation respectively. Compare with canonical nonlinear NeuTra separately
for central, dispersed and tail-start regimes. Use identical data, fixed
particle designs, clearly allocated tuning cost and paired seed policies.
Archive all warmup but exclude it from posterior estimates.

Primary acceptance is valid downstream sampling under the declared cumulative
diagnostics and compatible posterior summaries for the same numerical target,
with MCSE reported. Report bulk/tail ESS, chain movement, rejection reasons,
acceptance, uncertainty and total time including training. A speed ranking
requires replicated, uncertainty-supported cost per effective draw after the
validity checks. Underperformance against a simple comparator prevents
performance promotion; an inconclusive short comparison supports no ranking.
Neither outcome by itself rejects the NeuTra research direction.

Before this phase, the master must state the remaining wall/GPU budget and
predicted cost of the comparison and maximum sampling, then continue within
that allocation. If there is not enough authorized compute, report that
specific budget gap; do not invent a new fixed sample count or call an
unfinished posterior converged.

## Default audit, diagnostic roles and skeptical review

| Choice | Provenance / status | Failure mode | Earliest discriminating check |
|---|---|---|---|
| Three stages, author masks, ELU, nonzero initializer | Owner canonical architecture; paper/source grounded | Generic constructor silently uses old masks | Configuration and mask tests before tracing |
| Widths 3, 6, 9 | Paper dimension-wide baseline plus explicit capacity hypotheses | Underfit or unnecessary compute | Heldout loss/geometry and cost across widths; downstream comparison still required |
| Standard estimator first | Paper Eq. (3), checked finite-sample chain rule | Noisy gradients with small batches | Independent gradient variance estimates in pilot; path estimator is a later declared repair candidate |
| c=2 initial conditional cap | Existing explicit local configuration, a SIR hypothesis | Saturation or excessive conditional scales; free bias remains unbounded | Compare c=1,2,4 at common diagnostic weights/inputs and monitor activation/scale/conditioning curves. Calibrate for numerical non-harm, not HMC performance; record the selected rationale before serious fitting |
| No new clipping threshold | Preserve the core's explicit optimizer configuration until calibrated | Large finite updates despite rollback of nonfinite ones | Dedicated clipped/unclipped pilot comparison, recording norm scale, clipping incidence and unchanged healthy updates. A cutoff needs measured calibration, not an inherited q20 number |
| LR 0.01 and 0.001 | Paper anchor and one order-smaller SIR hypothesis | Overshoot, slow fitting | Pilot update norms, losses, invalidity and seed sensitivity; freeze a declared schedule |
| B=2 initially; larger B only measured | Existing SIR kernel scope, not paper batch size | High gradient noise or graph/VRAM explosion | Timed actual updates, variance and memory, with matching scope identity |
| FP32 transport; selected mixed SIR target; TF32 off | User campaign choice and recorded SIR exception | Silent casts or global TF32 change alter the scope | End-to-end dtype/device manifest and paired value/score checks |
| FP64 frozen-map evaluation if retained by HMC | Existing HMC coordinate convention, not a training requirement | Rounding changes between training and inference | Frozen-copy parity and endpoint/Jacobian check |
| Affine initialization | Optional calibration hypothesis | Warmup misses modes; initial flow maps into invalid regions | Identity-vs-calibrated initialization, dispersed starts and heldout validity |
| Fixed independent BPF endpoint | Existing SIR inference target | Roughness or bias persists despite a smooth LEDH-trained map | Same-point target differences and actual HMC; no claim of exact SIR posterior |
| Training and sample ceilings | Pilot/campaign resource bounds | Declaring success at an arbitrary ceiling | Report unmet criteria as under-budgeted or unconverged |

The conditional cap bounds only the nonconstant log-scale contribution, not
the free bias, off-diagonal Jacobian entries or global condition number. It is
not an overflow guarantee. A scale-cap or clipping change is a numerical
intervention requiring its dedicated calibration; it must not be selected
silently because one sampler run looks faster.

| Diagnostic / event | Role in this plan |
|---|---|
| Wrong target, Jacobian, chart, gradient pairing, resume state or transformed transition | Implementation invalidity; pause dependent execution, repair and rerun focused checks |
| Scalar training fallback, batch size one, missing GPU provenance, unrecorded precision/scope | Training eligibility veto; repair the route before training claims |
| Invalid training batch | Reject/rollback that update; repeated failures trigger repair, not conditional-on-valid training |
| ELBO, transformed residuals, probe-point validity, covariance, acceptance, finite energy error | Explanatory or candidate-nomination evidence only; never posterior certification or an automatic campaign stop |
| Warmup maximum reached without readiness | Current candidate not ready; proceed to the next planned geometry/initialization repair if budget remains |
| Cumulative posterior R-hat/ESS and same-target posterior comparison | Downstream inference criteria; a failure prevents promotion but does not cancel a planned repair |
| Loss to a simple baseline with adequate comparison evidence | Performance-promotion veto; no declaration of useful acceleration |
| Exhausted compute, corrupted artifacts, unresolved target/transition invalidity | Continuation limit until budget or correctness is restored |

Pre-mortem: the run could appear successful by learning ell_L well while ell_E
remains badly shaped; by copying a historical wrapper that changes TF32,
priors, or sample limits; by applying the prior Jacobian twice; by dropping
invalid rows; or by testing only an identity map. The phase ordering tests
those possibilities before expensive fitting. Model construction still has
trace-time per-row expansion, so larger batches are a capacity experiment,
not an assumed speedup. Reverse KL can miss modes; heldout ELBO alone cannot
establish coverage. The original-parameter chains and simple baselines remain
necessary downstream evidence.

Skeptical plan verdict: suitable as an implementation and bounded pilot plan.
It is not yet a launch-ready full training/HMC budget: measured SIR update and
transition costs, the frozen LR/cap/clipping calibration, and the remaining
campaign allocation are explicitly missing. Phases 1–3 produce those inputs.
No new architecture, endpoint, scientific prior or approximate-score claim is
authorized by convenience or a passing toy test.

## Executed checks and interpretation

The audit used deliberate CPU-only engineering checks with GPU hidden:

```sh
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q -p no:cacheprovider tests/test_neutra_single_authority.py tests/test_neutra_batching.py tests/test_neutra_precision.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py
```

Result: **57 passed, 1 skipped**, 3 warnings, 81.95 seconds. Log:
`docs/plans/surrogate-hmc-sir-neutra-training-audit-tests-2026-10-20.log`.
The optional historical-artifact skip is not a completed SIR check. The suite
includes generic mechanism, precision, training and handoff checks; its name
does not establish that the current SIR HMC consumer is wired.

The additional diagnostic was executed as:

```sh
PYTHONPATH=. /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/plans/artifacts/sir-neutra-training-audit-20261020/diagnostic.py
```

The script hides GPU before TensorFlow import. Its log and JSON are saved
beside it. The JSON's `wall_seconds` measures the diagnostic after framework
import, not complete process startup. On a coupled quadratic-plus-quartic
target with a nonzero parameter gradient, the standard-gradient maximum
absolute discrepancy is **5.55e-17** and the loss difference is zero.
The same-conditioner source comparison finds a **1.4130** output difference
and **0.3021** log-determinant difference, while the explicitly reparameterized
formula agrees to **2.22e-16**. These are deterministic engineering checks,
not stochastic evidence of a better map or sampler. The modern TFP primitive
checks the source's displayed operation; the entire old author training
program was not executed in its historical environment.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reuse canonical trainer; implement the SIR adapters | Generic chain rule and relevant tests pass | No failure in the executed engineering checks; exact same-weight source reproduction is disproved | SIR target regularity, capacity and downstream benefit | Phase 1 callback and Phase 2 transformed-consumer wiring, then the bounded pilot | SIR is trained, mixing improves, or the numerical endpoint is the exact posterior |

| Inference status | Audit conclusion |
|---|---|
| Hard veto screen | No SIR training/inference screen has run; source-identity overclaim needs correction |
| Statistically supported ranking | None |
| Descriptive-only differences | No SIR candidate metrics were generated in this audit |
| Default-readiness | No new SIR default or trained map is ready |
| Next evidence needed | Actual adapter parity, GPU capacity, target-specific fits and sequential same-endpoint HMC |

Post-audit red team: the strongest alternative explanation for eventual poor
mixing is that ell_E remains rough or differs substantially from ell_L, even
with a correct and well-trained map. Better downstream evidence could overturn
the expectation that nonlinear transport helps. The weakest current evidence
is the absence of any canonical SIR training/update or transformed HMC run;
the generic tests cannot fill that gap.

Source-audit coverage is intentionally bounded to the canonical 2019 NeuTra
construction, the maintained author files, the local trainer/core and the
current SIR consumers. It is not an exhaustive literature survey, an errata
search across every publication venue, a FAB/NAF/path-gradient paper audit, or
a reproduction of all author experiments. No broad literature claim depends
on those unperformed tasks.
