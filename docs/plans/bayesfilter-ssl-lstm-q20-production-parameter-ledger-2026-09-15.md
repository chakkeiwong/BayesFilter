# q20 NeuTra production pipeline: numerical choices and selection rules

Date: 2026-09-15. Status: `PROPOSED_PARAMETER_PROTOCOL_NOT_A_QUALIFIED_RUN_CONFIG`.
Companion to the [pipeline design](bayesfilter-ssl-lstm-q20-production-pipeline-design-2026-09-15.md).
The [mathematical parameter audit](bayesfilter-ssl-lstm-q20-parameter-mathematical-audit-2026-09-15.md)
covers every table entry below: derivation or rationale, assumptions, a concrete
check, failure interpretation and present evidence status. It also audits the
exact new constants proposed here; a documented origin does not establish adequacy.
For observed failures, follow the
[owner-directed investigation order](bayesfilter-ssl-lstm-q20-parameter-mathematical-audit-2026-09-15.md#first-checks-when-a-run-has-problems).
The implicated uncalibrated values are the first quantities to examine and test;
record the result against their parameter IDs before selecting a repair or
attributing the problem to the method.
This document answers which numbers to use, where they come from, and how to
determine values that cannot be chosen responsibly before calibration. No
training, target evaluation, GPU probe or HMC was launched to prepare it.

The configuration below is a concrete calibration proposal. It is not a claim
that its training caps, accuracy goals or complete comparison fit the remaining
budget. An executable production configuration is obtained by completing the
selection rules and the cost calculation here, preserving the resulting values,
and exercising the actual training-to-posterior consumer.

## Reading the tables

- **Fixed:** defines the existing scientific target or an applicable owner policy.
- **Measured:** an observed value in a specified earlier scope; not automatically
  valid for the repaired implementation.
- **Inherited hypothesis:** a current code or earlier plan value whose suitability
  must be checked for q20.
- **Proposed hypothesis:** a concrete new calibration choice, with its reason and
  failure response stated here; not a universal or already qualified default.
- **Derived:** calculated from a stated equation or selected configuration.

Selection diagnostics nominate or reject a configuration for development.
Posterior validity, requested precision and a statistically supported method
comparison remain separate decisions. Training-bank uncertainty is not
training-seed uncertainty, and neither is posterior Monte Carlo uncertainty.

The [source inventory](artifacts/ssl-lstm-q20-production-parameter-ledger-2026-09-15/r1/source-default-inventory.json)
preserves inspected paths, SHA-256 hashes, class fields and line numbers. Target
and q20 training sources come from audit checkout `3ec9affb`; shared HMC sources
come from the concurrent main checkout. The latter's repair is now recorded as
complete in its [terminal result](/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-hmc-overall-repair-result-2026-09-15.md).
That engineering result does not qualify q20 map export or q20 posterior sampling.
The inventory includes legacy class fields; the tables below identify active
dispatcher behavior so those fields are not accidentally revived.

## 1. Target and fixed data

These values are kept to answer the existing question. They are not knobs for
making the sampler pass. Changing them changes the target or numerical scope.

| Choice | Concrete value | Provenance, rationale and required check |
| --- | --- | --- |
| Free parameter dimension | 4 | Fixed: `latent_mean_weight.0.0`, `latent_mean_bias.0`, `observation_weight.0.0`, `observation_bias.0`, in that order. |
| Latent / hidden / augmented dimensions | 20 / 20 / 60 | Fixed; augmented dimension is `20+2*20`. The q20 name is not a 20-dimensional parameter posterior. |
| Horizon / observations per time | 30 / 1 | Fixed observed-data posterior; use the same 30 observations in every phase. |
| Prior | Independent Gaussian, center `(0.35,-0.08,0.65,0.05)`, SD 4, variance 16 | Fixed current target. The omitted Gaussian normalizer is parameter-independent. |
| Other model coefficients | `0.025*sin(0.371*i)` before the documented overrides | Fixed deterministic fixture; `i` is the zero-based full parameter index. Hash the whole resulting vector. Do not estimate only its convenient subset differently in a comparator. |
| Initial raw SD coordinates | `-0.85+0.011*i`, `i=0,...,59` | Fixed fixture. Actual SD is `softplus(raw)+1e-4`, approximately 0.355965 to 0.597789. |
| Process raw SD coordinates | `0.55+0.017*i`, `i=0,...,19` | Fixed fixture; actual SD approximately 1.005592 to 1.222133. |
| Observation raw SD | -0.2 | Fixed fixture; actual SD `softplus(-0.2)+1e-4 = 0.5982388694`; variance approximately 0.3578897448. |
| SD floor | `1e-4` | Existing model transformation, not an inference safety knob. Check its derivatives and preserve it. |
| Synthetic data seeds | Process `(20260719,1020)`; observation `(20260719,2020)` | Fixed q20 dataset; fold in time index. Construct on the declared CPU device, then preserve observation bytes/hash. |
| Parameter constraints | Four free coordinates in `R^4` | Fixed. The sign condition used for starts and diagnostics does not truncate the posterior. |
| Region quantity | Posterior probability that `observation_weight.0.0 > 0`, coordinate index 2 | Fixed existing research question. The two strict sign regions are not a proof of exactly two posterior modes. |
| Bridge | `ell_beta(theta)=log prior(theta)+beta*log L_UKF(theta)`, `0<=beta<=1` | Fixed proper Gaussian-prior likelihood bridge. Do not temper the prior or divide the likelihood by 30. |

These are synthetic fixture choices, not empirically calibrated finance
parameters. The free-coordinate simulated truth equals the prior center, and
data generation starts at the initial mean without an initial covariance draw.
Fixing those choices preserves the sampler question but does not establish
model adequacy or repeated-dataset calibration under the full inference model.

Source: `bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py:32,85,94,117`,
`ssl_lstm_sgqf_ukf_adapters.py:170`, and `tempered_target_tf.py`.

## 2. Numerical implementation, tolerances and hardware

| Choice | Concrete value / determination | Status and failure response |
| --- | --- | --- |
| Runtime arrays and derivatives | TensorFlow/TFP; float64 for this q20 UKF target | Inherited target-specific protocol. The LEDH TF32 direction does not make this UKF target float32. |
| Training and sampling device | One GPU visible per worker; memory growth before import/initialization; XLA on | Fixed owner policy. Record GPU UUID, actual device placement, versions and allocator peak. No silent CPU, non-XLA or scalar-training fallback. |
| TF32 flag | `True`, recorded explicitly, as in the audited launcher | Inherited launcher setting; it is not a claim that float64 operations use TF32. Any float32 subpath needs separate scrutiny. |
| UKF rule | Unscented: alpha=1, beta=2, kappa=0 | Fixed current target. At dimension 60, lambda=0, 121 points, center mean weight 0, center covariance weight 2, other weights `1/120`. |
| Square-root backend | `tensorflow_eigh_strict` as the initial target-specific route | Earlier feasibility/parity evidence exists. The factor-cached route is an optional separately checked engineering arm, using identical maps, starts and scopes for a causal comparison. |
| Explicit placement floor / added jitter | 0 / 0 at the public UKF call | Fixed current inputs. **This does not mean the implementation applies no shift:** the strict internal rule below remains active. |
| Innovation floor | `1e-12` | Inherited finite-program choice. Monitor actual innovation eigenvalues and floor applications; changing it changes the scope. |
| Rank / spectral-gap / fixed-null / reconstruction arguments | `1e-12` / `1e-8` / `1e-10` / `1e-10` | Current public wrapper defaults. Their use is branch-specific: the strict-SPD route does not use the reduced-support spectral-gap rule as its derivative engine. Record active branches rather than claiming all tolerances screened every row. |
| Strict roundoff scale rho | `rho=max(requested_floor,1e-14)` | Current strict classifier. Valid covariance gets `C+rho*I`; a permitted near-SPD repair gets `C+2*rho*I`. This belongs to the finite target/value-score audit. |
| Permitted near-SPD repair | Minimum eigenvalue at least `-1e-14`, maximum absolute covariance entry at most `1e8` | Inherited numerical rule, not statistically calibrated. Exercise both sides of each boundary; no loosening to avoid a failed proposal. |
| Invalid target sentinel | `-1e100` plus invalid status | Existing implementation sentinel. It is not a legitimate finite posterior value; consumers must reject invalid status even though the sentinel is finite. |
| Eigensolver refinement | 8 sweeps | Existing measured repair: four failed a recorded residual check, eight passed. For the 60-dimensional block this is `8*(60-1)=472` rotation rounds. Recheck under the new batch/device scope. |
| Eigensystem residual and orthogonality limit | `64*n*epsilon_machine`; at n=60, approximately `8.5265e-13` | Existing engineering bound; residual is relative to matrix norm. Failed checks produce invalid numerical evidence. It is not proof that eight sweeps work on every covariance. |
| Value replay parity | `abs(diff)<=1e-10+1e-12*abs(reference)` | Existing q20 saved-endpoint regression tolerance. Retain for same-scope replay; extend the bank, not the tolerance, when coverage is missing. |
| Score replay parity | `abs(diff)<=1e-9+1e-10*abs(reference)` | Same provenance and limitation. Cross-backend parity alone cannot prove both scores correct. |
| Finite-difference initial step | `h_i=epsilon_machine^(1/3)*max(4,abs(theta_i))`; multiplier approximately `6.05545e-6` | Derived central-difference starting scale under smoothness, not a universal optimum. Compare `h/4,h/2,h,2h,4h` on the same numerical branch. |
| Finite-difference error assessment | Richardson truncation estimate `abs(D(h)-D(h/2))/3`, plus independently measured value-error contribution divided by h | Determine a resolved comparison interval from the observed refinement curve. No stable interval or a branch crossing means derivative equality is unresolved; do not choose the step that happens to agree. Small-q fixture tolerances are not q20 authority. |
| Same-map checkpoint tensors | Exact equality after restoring stored float64 values | Engineering check. Cross-device computations use the replay tolerances above instead of assuming bit equality. |
| Map inverse / logdet screen | Existing q20 absolute tolerance `1e-8`; max condition diagnostic `1e8` | Inherited screening hypotheses. Also report scale-normalized errors. A condition estimate below `1e8` does not establish accurate scores or posterior coverage. The helper's alternative `max(1e-12,64*d*epsilon)` is not silently interchangeable. |
| Stress points | Signed coordinate axes at latent radii 2, 4, 6, plus self/cross-map and independently obtained posterior points | Inherited `sqrt(d),2*sqrt(d),3*sqrt(d)` at d=4. Axis checks do not establish global tail coverage. |
| GPU memory budget | 4 GiB pilot allocation as an inherited resource hypothesis; confirm machine availability before execution | This is a post-call stop budget under memory growth, not a hard reservation or proof of current free memory. Measure peak for each batch and complete graph. |
| CPU sample-generation workers | Probe `1,2,4,8,...` up to available affinity/physical cores and the measured memory bound | Select the smallest count whose throughput is indistinguishable from the largest feasible rate in the bounded pilot. Record explicit W. Per-worker intra/inter/OMP threads start at 1 as an oversubscription-avoidance hypothesis. |
| HMC chain parallelism | The inspected shared candidate binding builds independent chains with `chain_mode='serial'` by default | Record this actual execution topology. Batch-native training does not imply batched HMC. Any batched q20 extension needs a supported binding and parity/cost checks before assuming a speedup. |

Fixed numeric controls remain hypotheses about numerical adequacy even when
they define the current finite program. Boundary tests, same-value derivative
checks and reference agreement are their evidence; frequency of repetition in
old plans is not evidence. Newton-Schulz's 24 iterations, particle counts,
Sinkhorn settings and LEDH chunk sizes are **inactive** for this UKF campaign.

## 3. Training: concrete starting search and bounded repair

The objective is the batch mean of `-ell_beta(T(z))-log|det J_T(z)|`, for fresh
IID `z~N(0,I4)`. The omitted `log q0(z)` is constant with respect to trainable
weights. The full 30-observation likelihood has coefficient beta and the
Jacobian has coefficient one. Averaging the likelihood over time, clipping its
value, or dropping invalid rows would change the requested objective.

| Choice | Initial value / search | How determined and what failure means |
| --- | --- | --- |
| Prior map | `theta=prior_center+4*z` | Analytic beta-zero endpoint. Zero output-layer initialization makes the initial trainable stages identity up to their fixed permutations. Check the actual composition. |
| Beta-zero optimizer updates | 0 | Analytic prior endpoint needs no fitting. A failed density/score check is an implementation problem. |
| Plain NeuTra target | beta=1 | Required plain-method comparator. It must receive actual final-target training. |
| Initial architecture/LR search | Hidden `(16,16)` and `(32,32)`, each with LR `5e-4` and `1e-3`; two IAF stages | Four inherited q20 hypotheses. Earlier 32-update loss improvements did not establish adequate capacity or training duration. |
| Activation / ordering | tanh / full reverse between stages | Inherited warm starts. Examine saturation and downstream geometry. Do not quietly transfer the general class defaults of ELU and three stages. |
| Scale-log cap | `s_max=2` per stage; extra scale-linear paths disabled | Inherited bounded map family. Each diagonal scale lies between `exp(-2)` and `exp(2)` before composition; shear can still cause large condition numbers. |
| Hidden weight initialization | Normal SD 0.02; output-layer weights and all biases 0 | Actual implementation. Distinct seeds initially change hidden weights but may represent the same affine law until optimization progresses. |
| Adam | beta1=0.9, beta2=0.999, epsilon=`1e-7` | Inherited optimizer hypotheses. Record moment/denominator scales; epsilon dominance or slowly adapting moments can trigger a focused optimizer repair. |
| Gradient clipping | Global norm cap 10 | Inherited. Record raw/clipped norms and clipping on every update. More than 10% clipped updates is a **proposed explanatory alert**, not an automatic numerical veto or proof of undertraining. |
| Extra penalties / dropout | None; extra loss weights 0 | Preserve current reverse-KL method. Adding penalties is a separately named objective comparison. |
| Batch-size candidates | Start at 32; cost-check 8, 32, 128 | 8/32 inherited, 128 a proposed fourfold expansion. All are genuinely batch native. The eight-update cost pilot cannot qualify training quality or declare a best batch. See the paired batch-selection rule below. |
| Cost measurement | One compilation, then 8 timed valid updates per batch hypothesis at beta=1; price each additional temperature before reserving it | Proposed bounded engineering pilot; eight estimates only descriptive timing. These pilot maps are not training candidates. Include checkpoint and validation timing. Persistent timing variation calls for a revised forecast. |
| Initial training seeds | 3 independent roots per architecture/LR hypothesis at the selected batch size | Proposed minimum seed replication, inherited from the earlier plan's intended protocol. Twelve histories for the first plain beta-one grid. This is too few to assert broad seed robustness. Each added batch contrast or ensemble chart/temperature is additional funded work. |
| Cumulative positive-beta training rungs | 128, 512, 2,048, 8,192 updates per chart, seed and beta | Proposed logarithmic exploration, factor four between rungs. 128 is the old pilot cap used as a **first observation**, not a production endpoint. Every numerically viable initial history is funded through 512 before starting the grid. Later work follows the evidence rules below. |
| Final training count | Selected checkpoint from the funded rungs after learning, capacity and map checks | It is not known in advance. The initial grid reaches at least 512; a plateau-based nomination needs two comparisons, ordinarily through at least 2,048. Reaching 8,192 with unresolved progress/coverage means `inconclusive_at_training_cap`; it does not mean adequately trained. |
| LR schedule | Constant within the first measured segment; a paired half-LR continuation on a plateau/deterioration | Proposed controlled repair, not an unrecorded decay. Continue full optimizer state on both same-beta branches; compare same starts and validation rows. |
| Conditional capacity repairs | `(64,64)` with two stages; `(32,32)` with four stages, at the retained viable LR | Proposed one-factor capacity contrasts. Compare matched cost and seeds; fund one bounded repair cohort, not the entire cross-product with every other option. |
| Conditional optimizer/cap repairs | LR halved; if justified by telemetry, clipping 10 versus 30, cap 2 versus 3, or Adam beta2 0.999 versus 0.99 | Proposed single-factor hypotheses. Saturation/clipping/moment evidence selects which contrast is run. These are not automatic changes; preserve the parent and diagnose whether the modification solves its stated problem. |
| Adam across temperatures | Carry weights; compare carry versus reset of Adam once if continuation stalls | Carry is the proposed initial continuity choice. Reset was the old inherited protocol, not an established improvement. Resume within the same beta must always restore moments and iteration. |
| Save cadence | Every 128 updates, each validation decision, each temperature boundary, and clean exit | Proposed recovery-cost bound tied to the smallest rung. Preserve weights, Adam slots/iteration, beta, RNG counter and batch/schema identity. Archive all per-update telemetry. |

For Adam's exponential moments, `1/(1-beta)` gives memory scales of roughly
10 and 1,000 updates. These are descriptive averaging scales, not required
convergence counts. They help explain why inspecting only six updates cannot
establish whether this optimizer has learned the posterior geometry.

Batch selection: B=32 is the initial measured-throughput hypothesis. For each
feasible B in {8,32,128}, train the same compact-high architecture from the same
three initialization roots for a common measured update budget equal to
`512*c_update(B=32)` seconds. Freeze the corresponding update counts before
starting this contrast; include each arm's compilation and validation in its
total cost. Compare paired held-out loss improvement, its uncertainty and
numerical health. If the alternatives are indistinguishable, retain B=32 as an
explicit operational choice, without calling it optimal. If B=32 is infeasible,
use B=8 as the declared initial comparator. This is up to nine histories; at
B=32, three can be reused as the exact compact-high grid histories when source,
seeds, optimizer, endpoint and validation scope coincide. Otherwise every
history is additional charged work. A trained map selected at another batch
requires a fresh training selection and frozen-map tuning scope. This contrast
matches priced update work, not total cost: compilation and validation may
differ. If B=32 is infeasible, reprice the common allowance using B=8 before
freezing counts. Do not use an unavailable B=32 cost as its denominator.

### Training validation and the decision to continue

| Choice | Concrete value | Selection rule |
| --- | --- | --- |
| Initial objective-validation rows | Three independent banks of 256; total M=768 | Inherited q20 design; fixed shared base rows pair checkpoint comparisons. These are selection data. |
| Bank expansion | M=768, 3,072, 12,288 | Proposed factor-four ladder; approximately halves IID Monte Carlo standard error each extension. Chunk target calls using the qualified batch size. |
| Validation cadence | At start, every training rung, and before/after a proposed repair | Proposed bounded cost. Cheap per-update telemetry remains continuous. |
| Material objective change | delta=`0.01*d=0.04` nats per draw at d=4 | Proposed optimization-resolution choice. One hundredth of a nat per dimension is a resolution target, not a bound on posterior bias or a universal loss. |
| Paired objective uncertainty | Let `I_m=loss_old(z_m)-loss_new(z_m)`; interval `mean(I) +/- 1.96*s(I)/sqrt(M)` | Approximate IID normal interval for a fixed comparison, inherited 95% calibration level. Heavy tails or unstable variance make this unavailable. Once a reused bank influences selection or repair, these are development summaries, including later comparisons on that bank. A nominal final confidence claim needs a frozen comparison, fresh bank and declared stopping/multiplicity treatment. |
| Continue learning | Lower interval endpoint greater than delta | Material progress motivates another funded rung. If the interval overlaps delta, increase M or training evidence within its cap; do not call that a plateau. |
| Plateau candidate | Interval lies within `[-delta,+delta]` for both 128-to-512 and 512-to-2,048, or a later pair of successive rung comparisons | Proposed persistence rule. It triggers capacity/optimizer/coverage assessment; it does not qualify the map by itself. |
| Deterioration | Upper interval endpoint below `-delta` | Preserve the earlier checkpoint; run the specified repair if funded. One noisy high training loss is not enough. |
| Seed viability | At least two of three roots improve against their own beta-start checkpoint and pass numerical checks | Inherited nomination concept; two of three is a small-sample screen, not a success-probability estimate. Carry uncertainty/failure of the third root into reporting. |
| Cheap generated-map occupancy bank | 4,096 Gaussian rows per map | Inherited; worst-case IID proportion SE `1/(2*sqrt(4096))=0.0078125`. It estimates the map's own proposal occupancy, not posterior region mass. |
| Actual whitening | Inverse-map independently validated posterior draws; report mean, covariance/eigenvalues, radial tails, scores and coverage with uncertainty | No universal pass number. Unit covariance does not imply Gaussianity. Generated `z` or `T^{-1}(T(z))` cannot establish posterior whitening. |

For a desired half-width `h`, the validation-row estimate is
`M_required = ceil((1.96*s(I)/h)^2)`. For resolving the proposed delta closely,
take `h=delta/2=0.02`; if the estimate exceeds 12,288 or its target-call budget,
the comparison is inconclusive. A clearly separated progress interval can
support extension without first reaching that fine half-width. Compare losses
only at the same beta and objective; unknown normalizers differ across beta.

The practical training decision is: fund the initial grid, observe its learning
curves, extend improving viable candidates, diagnose poor plateaus, and test
qualified maps downstream. Do not run a large fixed number merely to acquire
the appearance of production training.

## 4. Temperatures, charts and ensemble transitions

| Choice | Concrete initial values | How selected |
| --- | --- | --- |
| Single-map comparator | K=1, beta=1 | Fully trained plain NeuTra. |
| Initial ensemble | K=2 | Smallest actual mixture of charts; historical C5 nomination, reopened as a training hypothesis after the audit. |
| Chart-count repair | K=4 | Inherited bounded capacity/diversity contrast. K=4 needs enough training for four maps; copying or briefly initializing maps does not test it. |
| Initial bridge ladder | `(0,0.5,1)` | Historical small ladder, not established overlap. |
| Ladder repair | `(0,0.25,0.5,0.75,1)` | Inherited interval-halving contrast. Compare valid cold sampling and total cost; swap rate alone cannot choose a winner. |
| Branching contrast | Pure continuation versus restart of half the charts at beta=0.5 | Existing method contrast. Independent roots and positive-temperature training must actually occur. |
| Additional temperatures | Bisect the diagnosed poorly connected interval only in a funded development revision | No unbounded adaptive ladder inside retained sampling. The new scope must be trained, tuned and frozen. If L5 is all that was funded and still inadequate, report that limit. |
| Chart selection probabilities gamma | `1/K`: 0.5 for K=2, 0.25 for K=4 | Fixed state-independent weights in the initial method. Freeze before sampling. No state-dependent reweighting without a different invariant-kernel derivation. |
| Density-mixture alpha | Uniform if a diagnostic mixture density is evaluated; train-alpha disabled | The optional joint-mixture objective is not part of the initial production proposal. Alpha and kernel-selection gamma are different quantities. |
| Within/swap cadence | One within-temperature HMC update at every slot, then adjacent swaps; alternate even/odd edge parity each outer transition | Actual existing exact ensemble program. Initial parity 0; no tuned acceptance target for swaps. |
| Replica systems | 4 independent systems; 3 or 5 temperature slots each | Derived: 12 or 20 physical replica states. K is chart count, not an extra independent-chain multiplier. Only the four beta-one streams enter posterior assessment. |
| Region movement screen | Both strict sign regions visited and at least 4 retained crossings per cold chain | Inherited minimal observability screen for the current region question. It does not estimate mixing time or require 50/50 region mass. Reference evidence that a region is negligible requires an explicit scientific-contract revision. |
| Replica travel screen | At least 16 aggregate round trips in each orientation, and at least one per replica identity | Inherited operational hypothesis. Round trips are dependent: the old `1/sqrt(16)=25%` Poisson argument is not a calibrated uncertainty statement here. Preserve actual travel times and dependence. |

A bottleneck may require more training, another chart, another temperature or a
different kernel. The observed failure selects the smallest contrast; it does
not justify changing all four controls simultaneously. The physical replica-
exchange comparator uses the same temperature ladder and swap cadence.

## 5. HMC preparation, candidate search and starts

Use the [current public interface](/home/ubuntu/python/BayesFilter/docs/reference/hmc-tuning-interface.md)
and registry. The frozen-map implementation currently uses identity latent mass.
The q20 `reference_affine_weighted_dense_iaf` checkpoint needs verified export,
restore and value/logdet/score parity before this consumer is usable.

| Choice | Concrete value / procedure | Provenance and limits |
| --- | --- | --- |
| Kernel | Fixed-trajectory Metropolis HMC with exact transformed target/score; fresh Gaussian momentum every transition | Existing method. Map weights, metric, epsilon and L stay fixed throughout retained sampling. |
| NeuTra mass | `I4` in z | Public-interface constraint. Residual latent mass adaptation is not supported by silently passing an ordinary mass tuner. |
| Classical naive comparator | Identity mass in declared physical coordinates | Explicit baseline, not the sole comparison. |
| Tuned classical comparator | Public ordinary metric/affine preparation, `preset='serious'`, `metric_update_requirement='require_operational_update'` | Proposed use of the existing supported preparation. A successful unchanged incumbent is not described as newly learned mass. |
| Initial classical geometry | Prior center with scale vector `(4,4,4,4)` or equivalent prior covariance `16*I4` | Defined warm start. No undocumented Hessian flooring or favorable fitted center. If an independently estimated covariance is tested, name its data and include its cost. |
| Initial ordinary metric budget | 1,000 adaptation transitions at d=4 under current serious policy | Derived below. This is preparation, followed by separate discarded posterior equilibration. Inspect the actual windows and metric updates; 1,000 alone proves nothing. |
| Tuning chain count | Exactly 4 | Active candidate-set acceptance implementation requirement. |
| Physical starting-bank proposal | Two valid starts in each strict sign region from the same frozen Gaussian prior, with fresh stage-specific streams | Proposed dispersed-start policy. Inspect prior proposals in fixed order; at most 128 proposals (four batches of 32) before reporting unavailable starts. Preserve rejected start proposals. This is initialization only; never apply this conditioning to training batches or the posterior target. |
| Matching starts across methods | Same physical bank before each method's declared preparation | Transform through each map's inverse. Preparation may change endpoints; record and charge it. Avoid comparing one method's local favorable start with another's broad prior starts. |
| Confirmation starts | Fresh bank under the frozen start policy, with its own checked binding and fresh verification for the selected pairs | Integration requirement. Existing retained runners continue verified endpoints; they cannot silently reset to arbitrary new states. Verify the public path supports this confirmation binding, or implement/test that connection before claiming fresh starts. |
| Initial L grid | `(3,5,9,13,18,25)` | Current public broad-grid hypothesis. Every L has its own measured epsilon; no ESS-based short-chain winner. |
| Initial epsilon | Per map/beta/mass scope, starting from `min(0.1,0.8*2/sqrt(lambda_max))` when resolved positive-curvature diagnostics exist | Proposed local seed using public 0.1 and existing 0.8 stability hypothesis. `lambda_max` is the largest observed positive eigenvalue of the mass-coordinate potential Hessian over the declared development bank. This is not a global stability bound. Otherwise 0.1 is explicitly an unqualified pilot seed. Ordinary automatic preparation supplies its own measured epsilon instead. |
| Epsilon repair | Factor 2 up/down; at most 5 directional repairs per family in the initial search | Existing bounded search hypothesis. Public dispatcher domain is `[min(proposals)/64,64*max(proposals)]` because it uses `2^(5+1)`; a supplied geometry upper bound replaces the upper endpoint. Preserve this actual domain in the scope. A failed parent never grants its child verification. |
| Epsilon refinement | Multipliers 0.8 and 1.25; one refinement round around each viable family | Existing reciprocal local-search hypotheses. Retain all independently verified members. |
| L refinement | Integer midpoints `(4,7,11,15,21)` | Derived from initial grid. Optional declared stage, subject to measured budget. |
| Longer travel | If development shows inadequate travel, propose geometrically larger L values 50 and 100 | Proposed conditional repair. The ordinary convenience config currently enforces a maximum of 25; broader scope must use a supported explicit binding/path after verification. Do not pretend this extension is a working public override. A cap-limited classical arm cannot establish the best possible classical baseline. |
| Acceptance target | 0.70; practical interval `[0.65,0.75]`, repair interval `[0.55,0.85]` | Current shared operational policy. These screen mechanics, not stationarity or optimality. No ranking by closeness to 0.70. |
| Acceptance evidence | 4 chains, 4 temporal blocks/chain, at least 16 decisions/block; 90% compatibility interval | Active implementation: at least 64 decisions per chain; Student-t critical value `2.3533634348` for four chain means. Operational repeated-look evidence is not anytime-valid confidence. |
| Pilot / measurement / fresh verification | Start with 64 / 64 / 64 decisions per chain; 32 preliminary discarded transitions per stage | Proposed explicit q20 execution config, at the active evidence minimum. The 32 discarded transitions are startup handling, not a claim of equilibration. Pilot never qualifies a pair. |
| Inconclusive evidence rungs | `(1,2,4)`: 64,128,256 measurement or verification decisions/chain | Current bounded extension hypothesis. At the cap, report inconclusive. Do not retain a pair merely because its point acceptance is in-band. |
| Minimum movement / max repeats | 0.05 / 0.95 per chain | Current heuristics; all-chain movement remains mandatory. These are not exploration or precision criteria. |
| Normalized return displacement floor | `1e-4` | Current recurrence/movement screen, scope-sensitive. Inspect normalization and its sensitivity on the actual chart. |
| Path recurrence | Lags 2 through 16; fraction limit 0.95; matching atol=`1e-12`, rtol=`1e-10` | Current path screen. Record exact values and do not describe them as calibrated q20 thresholds. |
| Finite energy magnitude | Report only; existing magnitude/proxy field 1,000 and legacy log-accept alert -1,000 are not new vetoes | Preserve the current campaign contract. Native divergences, nonfinite states/scores or invalid target status retain their actual veto roles. |
| Status cadence | `per_chain_step`, accepted and proposed state health; all integration telemetry promised by the target contract | Explicit requirement. A finite invalid sentinel must not pass. |
| Tuning chunk size | 64 initially, increase to 256 only if measured native-call cost and memory permit | Proposed pause/recovery granularity within the existing API. This does not reduce the total evidence requirement. |
| Candidate/attempt/gradient caps | Derived from the explicit funded cohorts and work formula below | Do not inherit 100 candidates, 100 budget units or a 20-unit repair reserve without a cost calculation. Each candidate reserves its mandatory remaining stages. |

For a quadratic potential in identity-mass coordinates, leapfrog is stable only
when `epsilon*sqrt(lambda)<2` for each positive-curvature mode. That gives the
local starting scale above, not its final tuned epsilon. At the **old failed**
point, `lambda=33319.68301` gives `2/sqrt(lambda)=0.0109567`; the old
`epsilon=0.0275` exceeds it. A newly trained map changes the Hessian, so neither
number transfers as its tuning result. Also, shrinking epsilon at fixed L
reduces trajectory duration `tau=L*epsilon`; test travel as well as acceptance.

### Ordinary metric preparation: numbers hidden behind `serious`

These are current implementation choices, not newly established q20 defaults.
The preparation must serialize its realized settings. The convenient name
`serious` cannot substitute for checking this table and the resulting metric.

| Control | Current values / derived q20 setting | Audit/selection rule |
| --- | --- | --- |
| Geometry budget | `N0=clip(ceil(20*d*m),1000,5000)`; `m` clipped to `[1,4]`; later legacy ladder capped at 10,000 | At d=4, `20*d*m<=320`, so N0=1,000 regardless of m. The current preparation dispatch uses its first budget; larger retries require an explicit funded preparation path. |
| Geometry-pressure coefficients | log10 condition weight .25, square-root anisotropy weight .50, clipped-eigenvalue weight .05, nonpositive weight .25, diagonal fallback multiplier 1.50 | Inherited heuristic forecast. At d=4 the 1,000 floor dominates; these coefficients do not demonstrate sufficient mass learning. |
| Geometry initialization | scale coefficient .5, stability guard .8, covariance jitter `1e-9`, eigenvalue floor `1e-9` | Inherited controls. Preserve source of the covariance/Hessian and check the issued transform and score. |
| Ordinary bootstrap screen | `clip(ceil(4*sqrt(d)),32,1024)` decisions, burnin one quarter | At d=4 initial screen 32 and burnin 8; mechanics preparation only, not the 64-decision candidate qualification. |
| Windows for 1,000 steps | Initial 100; slow 200,400,200; final 100 | Derived current window constructor: buffers 10%, first slow window one quarter of remaining span, doubling and truncation. This is unrelated to sufficient posterior burn-in. |
| Empirical metric shrinkage | .25 toward diagonal, preserving variances | Actual active covariance decision. Investigate inadequate information or poor downstream mixing before asserting a tuned metric. |
| Dense/diagonal minimum states | `max(64,4*d)=64` / `max(32,2*ceil(log2(d+1)))=32` | Inherited information screens. |
| Dense/diagonal ESS floor | `max(8,d+1)=8` / `max(4,ceil(log2(d+1)))=4` | Preparation-only information screens, far below posterior precision requirements. |
| Dense/diagonal location check | Split R-hat at most 1.10 / 1.25 when independent chains are available | Heuristic covariance qualification; not the retained rank/folded criterion. |
| Dense condition / discrepancy | Raw standardized condition at most `1e8`; dense shrinkage discrepancy at most .50; diagonal discrepancy at most .75 | Existing heuristic checks. Do not relax them in response to a failed q20 result without evaluating the numerical consequence. |
| Low-level window metadata | jitter `1e-6`, eigenvalue floor `1e-9`, simple step-rate .03, step range `1e-6` to 10 | The active empirical metric path issues no absolute floor; active step adaptation is TFP dual averaging. These config fields are not evidence that those legacy algorithms execute. |
| Actual dual averaging | exploration shrinkage .05, pseudo-count 10, decay exponent .75, target acceptance .70; shrinkage target `min(10*epsilon,scope_upper_bound)` | Installed TFP defaults with repository overrides; inspected source, not an imported runtime. Reset adaptation when the actual metric changes; preserve exact state across an interruption. |

## 6. Posterior quantities, precision and reference comparison

I propose a concrete accuracy specification for this synthetic study: posterior
mean Monte Carlo error no greater than **2% of posterior SD**, sign-probability
MCSE no greater than **0.01**, and the three reported quantiles' MCSE no greater
than **5% of posterior SD**. These are new practical accuracy choices, not
properties of the posterior and not guarantees of sufficient inference for a
different financial decision. If the application needs a different margin,
derive the tolerances from that decision and rerun the cost forecast before
execution. The existing 5%/0.025 settings remain identifiable historical bounds.

| Choice | Proposed concrete value | Provenance / interpretation |
| --- | --- | --- |
| Means | All four; `MCSE(mean)/SD<=0.02` | Proposed relative accuracy target. Equivalent asymptotically to original-scale mean ESS at least 2,500. |
| Quantiles | .025, .50, .975 for each of four parameters; `MCSE(quantile)/SD<=0.05` | Proposed median and central 95% interval reporting. Quantile-specific MCSE required. |
| Sign probability | Absolute MCSE<=.01 | Proposed one-percentage-point standard error. For a Bernoulli indicator, worst-case ESS requirement is .25/.01^2=2,500. |
| Additional predictive quantities | None silently added | If included in the claim, specify names, horizons, conditional/path estimator and absolute error tolerance in the same ledger. The current fixed inference task does not require inventing a predictive horizon. |
| Independent posterior chains | 4; ensemble means four independent replica systems | Fixed minimum and tuning interface. Dispersed initialization and reference checks are still necessary. |
| Warmup | Minimum 2,000/chain; window latest 1,000; maximum 10,000; rank/folded R-hat<=1.05 | Owner policy, operational screen. No estimate of guaranteed sufficient burn-in. |
| Warmup check cadence | Every 1,000 transitions; one passing window initially | Inherited shared policy. Consecutive checks may be an explicit sensitivity study; overlapping windows are not independent confirmation. |
| Retained draws | Start at 1,000/chain; grow by 1,000 to 10,000 maximum | Owner/shared controller settings. All retained prefixes stay included; tuning/warmup excluded. |
| Retained R-hat | Maximum rank-normalized split and folded statistic <=1.01 | Owner policy with repaired arithmetic. Apply to physical parameters and declared functionals. |
| Bulk / tail ESS screen | Each >=400 per monitored quantity, pooled over four chains | Inherited diagnostic floor. Tail implementation uses 5%/95% indicators, which do not certify the requested 2.5%/97.5% quantiles. Direct MCSE targets are additional. |
| Extra warmup ESS floors | 0 initially | Current policy; report ESS without inventing a new burn-in criterion. Required retained precision remains explicit and nonzero. |
| Primary mean-MCSE estimator | Current named TFP positive-pairs autocorrelation estimator | Preserve explicit method identity; it is not Stan's initial-monotone estimator. Validate on saved draws and target-specific uncertainty comparisons. |
| MCSE sensitivity | Ordinary/lugsail batch means; `b=floor(sqrt(n))`, r=3, c=.5, at least 20 complete batches per chain | Existing optional literature-baseline implementation. At n=1,000, b=31 and 32 batches; at n=10,000, b=100 and 100 batches. Negative/nonfinite/underbatched estimates are unavailable. No selection of whichever estimator reports the smallest error. |
| Repeated-look interpretation | Operational precision stopping only | No nominal anytime-valid confidence guarantee. Quantiles, rare events and weak exploration need their own checks. |
| Constant/unvisited event | Precision unavailable unless independently established as a deterministic quantity | Zero observed variance cannot establish negligible posterior probability. |
| Mean / event equivalence margins | .10 pooled posterior SD / .05 absolute probability | Inherited q20 research-resolution hypotheses. Before confirmation, freeze the scale from independent reference/development evidence or propagate its estimation uncertainty; a random plug-in scale does not give a fixed-margin confidence guarantee. Compare start groups and independent reference. |
| Quantile equivalence margin | .20 pooled posterior SD | Proposed research-resolution hypothesis, distinct from .05-SD MCSE. Larger quantile uncertainty does not itself justify tolerating more bias. Use the same explicit scale-uncertainty treatment as for means. |
| Reference accuracy allocation | Reference MCSE, or certified numerical error, <=one third of each requested sampler-MCSE tolerance | Proposed error allocation: reference variance at most 1/9 of the sampler's **permitted** MCSE squared. At the requested limit it increases combined SE by about 5.4%; actual contribution uses measured errors. Deterministic bias is an error bound, not variance. |
| Reference comparison confidence | Simultaneous approximate 95% comparisons across 17 quantities for one method | Four means + twelve quantiles + one event. Bonferroni normal critical value `Phi^-1(1-.05/(2*17))=2.9738199`; meaningful only when underlying MCSE/normal approximations are adequate. A joint claim over A methods uses 17*A comparisons, or another explicitly specified multiplicity rule. |
| Start-group comparison | Five quantities: four means and sign probability; 95% family level | Corresponding normal critical value 2.5758293. Two starts per sign group is limited replication. Meeting a pooled MCSE goal does not automatically meet start-equivalence margins. |

For a stationary mean estimator with long-run variance v and total effective
sample size E, `MCSE^2=Var(theta)/E` asymptotically. Thus a relative target r
requires `E>=1/r^2`: r=.05 gives 400, .02 gives 2,500, .01 gives 10,000.
**Rank bulk ESS cannot be substituted for this original-scale E.**

For an event probability p, `MCSE^2=p*(1-p)/E_indicator`; use the direct
indicator uncertainty estimate and the worst case p=.5 for forecasting.
For a continuous p-quantile q, the local approximation is
`MCSE(q)^2=p*(1-p)/(E_indicator*f(q)^2)`. A small tail density increases its
cost. Even for a standard normal, a .05-SD quantile MCSE requires roughly 628
effective draws for the median and 2,855 for the .025/.975 quantiles. These
are illustrative calculations, not evidence that q20 is Gaussian.

For independent sampler/reference estimates and reference bias bound b, the
declared comparison is

```text
abs(estimate_sampler - estimate_reference)
  + critical * sqrt(MCSE_sampler^2 + MCSE_reference^2)
  + b <= equivalence_margin.
```

The reference's own target implementation and region coverage must be checked.
A common missed mode defeats R-hat, estimated MCSE and a reference using the
same missed region. Comparing different starts is evidence, not an exhaustive
mode-discovery proof. No likelihood observations are held out to create these
random-stream partitions.

### Determining reference-method numbers

The reference has not been constructed, so its final nodes/draws cannot be
reported as already determined. Use the same error specification above to
choose them, with the following explicit feasibility procedure.

1. Assess integration in the four prior-standardized coordinates
   `u=(theta-prior_center)/4`. Initial box radius R=4; expansion candidates R=6
   and 8. These are proposed domain probes, not proven posterior truncation.
2. A tensor-grid feasibility count is 9,17,33 nodes per axis: 6,561, 83,521,
   1,185,921 evaluations. Measure target-call cost first; a sparse/adaptive
   rule may be preferable, but must name its actual rule, refinement and error
   estimate. Do not launch the larger grid simply because d=4 sounds small.
3. Require the combined domain, quadrature, arithmetic and other reference error
   for every reported quantity below its one-third allowance, with a justified
   tail bound. Do not give each error source the full allowance. A difference
   between two grid estimates is not automatically a certified error bound;
   justify the error estimator and include normalization error.
   The prior probability outside R=4 is at most about .0002534 by a union
   bound; that is **not** a posterior tail bound. If `L<=Lmax` and a validated
   lower bound `Zlower` exists, then
   `P_posterior(outside)<=Lmax*P_prior(outside)/Zlower`. For moments, bound the
   corresponding tail integral too. A loose bound cannot certify truncation.
4. If integration cannot meet those bounds affordably, use a separately
   specified independently implemented sampler/reference protocol with the
   same quantity-specific uncertainty budget. Its own kernel, starts, seeds,
   precision and coverage must be resolved before accepting it as a reference.
   The tuned classical comparator alone is not independent target verification.
   This remains an explicit unresolved reference-method decision if neither
   approach can be funded and validated; the pipeline cannot claim completion.

## 7. Seeds, comparisons and cost allocation

| Choice | Concrete proposal / rule | Reason and limit |
| --- | --- | --- |
| New root namespace | Two-int root `(20260915,150001)` with explicit role/arm/replicate/beta/chart/stage/block folds | Proposed reproducibility identifier, not a scientific constant. Check the complete derived int32 schedule against prior and current attempts before launch; the printed root is not proof of no collision. |
| Independent roles | Separate initialization, training, selection validation, stress, reference, tuning pilot, tuning measurement, tuning verification, development, confirmation warmup, confirmation retained | Preserve independence boundaries. Common random rows across development candidate comparisons must be declared; confirmation shares no consumed draws with selection. |
| Training replications | 3 in the initial grid | Limited candidate evidence; all outcomes preserved. |
| Final method replications | Begin with 3 independent four-chain systems per frozen method, only if complete funding exists | Proposed minimum research comparison. It establishes conditional evidence for the frozen maps, not population-wide performance over all possible trained maps. |
| Efficiency statistic | Total cost to meet all posterior/error criteria, including apportioned training/search/tuning and failures | Compare at matched accuracy. Sampler ESS/gradient on invalid or nonstationary runs cannot choose the method. |
| Efficiency uncertainty | Paired log-cost differences over independent replications; 95% paired interval if assumptions adequate | With 3 pairs, normality of log-cost differences is largely untested; report no robust superiority conclusion from three descriptive means. A two-sided exact sign test cannot reject at .05 with fewer than 6 nonzero unanimous pairs (`2/2^6=.03125`). Six is not a power guarantee. |
| Further replication | Estimate paired log-cost SD s and scientifically worthwhile effect Delta; fixed-size planning approximation `n≈((1.96+0.8416)*s/Delta)^2` for .05 two-sided, 80% power | These confidence/power levels are proposed design conventions. Freeze n before independent confirmation; heavy tails, censoring and selection require another model or no ranking. Timed-out methods cannot be omitted as missing fast runs. |
| Available campaign | 135,275.83289109988 seconds = 37.5766 aggregate worker hours | Existing amended ledger. Parallel workers' elapsed time is charged separately. |
| Included diagnostic allocation | 54,299.37991617 seconds = 15.0832 hours | Included in campaign, not added to it. |
| Per-arm cap | 28,800 seconds = 8 hours | Existing boundary; splitting one experiment into several processes does not silently enlarge it. |
| Training/rung max | 8,192 per positive beta, additionally bounded by affordable remaining work | Proposed search cap, not a promise that every chart reaches it. If needed work is unaffordable, report under-budgeted. |
| Repair reserve | Price one complete declared localized repair plus its revalidation | Proposed bounded reserve derived from tasks, not an unexplained percentage. More repair requires a new allocation within the same overall budget. |
| Checkpoint/polling overhead | Checkpoint every 128 optimizer updates and every sampler chunk; progress heartbeat 60 seconds | Proposed operational constants. Include I/O, diagnostics and compilation in measured costs. Native calls cannot always obey a cooperative deadline. |

### Converting observations into an affordable frozen configuration

Measure compile time, update cost `c_train(B,architecture,beta)`, validation
target cost, four-chain transition cost `c_HMC(L,beta,chart)`, reference cost,
checkpoint overhead and peak memory on the actual repaired source. Use an
observed upper timing range for initial reservation; a few timings do not
estimate a reliable extreme quantile. Reforecast when those measurements change.

For training, use cumulative increments, not the sum of checkpoint labels:

```text
C_train = sum_histories [compile + final_update_count*c_update
                        + validation + checkpoint + failed_attempt_cost].
```

For one initial HMC pair, the proposed pilot, measurement and verification cost
is four chains times `3*(64+32)=288` transitions per chain, before evidence
extension. A conservative gradient-work count is

```text
G_pair = 4 * sum_stages(draws_stage + discarded_stage) * (L+1).
```

The exact binding's target calls may exceed this count; use measured costs for
wall-time reservation. K=2 with three beta levels has six chart/beta scopes.
Six initial L values then mean 36 pairs and approximately **546,048** gradient
work units at those base counts, if all reach verification, excluding refinements,
repairs and compilation. This arithmetic makes the scope visible before launch.

For a posterior mean requiring E effective draws over four chains, with
development integrated autocorrelation time tau, estimate
`n_per_chain≈ceil(E*tau/4)`. For E=2,500 and the 10,000 retained cap, tau around
16 already consumes the entire cap. Use estimates only from a viable,
reference-consistent development run; failed canary ESS does not forecast tau.
Quantile, reference and start-equivalence checks may demand more.

Reserve reference, all required tuning, development, final confirmation, reporting
and the specified repair before admitting the training cohort. The remaining
training grant divided by measured cost gives the affordable histories/rungs.
If twelve histories cannot reach the promised 512-update screen while preserving
that reserve, do not launch that grid and later pretend partial selection was
complete. Revise the funded comparison explicitly; no automatic promotion from
an incomplete grid or shortened posterior run.

For scale only, an older beta=.5 B=32 strict-backend pilot measured
**2.04114 seconds/update**. At that obsolete cost, 128/512/2,048/8,192 updates
would cost approximately **.073/.290/1.161/4.645 hours per chart per beta**;
12 histories through 512 would cost **3.484 hours**, excluding other stages.
The eigensolver and consumer changed afterward. These are arithmetic examples,
not a current affordability forecast. The full calculations are preserved in
[derived examples](artifacts/ssl-lstm-q20-production-parameter-ledger-2026-09-15/r1/derived-examples.json).

## 8. What must be frozen before a serious launch

The target constants, owner-policy limits and proposed search are concrete.
The selected batch/architecture/optimizer, actual trained update counts,
temperature/chart choices, each measured epsilon/L pair, reference method,
replication count and complete cost allocation remain **measured outputs of
this procedure**, not numbers known from the previous canary.

Write them into one resolved run configuration, with no missing precision
requirements or implicit legacy defaults. Have the actual setup/resume path
load the selected trained map and optimizer state, checked HMC binding and
quantity definitions. Validate wiring for changed map/target, unqualified smoke
maps, missing precision and attempted callback bypass. Run records preserve
source, environment, data, seeds, wall time, budget, diagnostics and terminal
failure as well as success. Ordinary versioned files are sufficient; no new
launch-token or approval ceremony is introduced.

Skeptical audit: the main unresolved risks are unaffordable complete comparison,
undercoverage despite low reverse-KL, noisy training stopping, reference error,
incomplete q20 binding and an L-limited classical baseline. The early cost and
numerical checks address these before expensive training. Selection screens do
not promote the posterior; failure of a candidate triggers the specified repair
without rejecting the whole method. The current document is a design and source
inspection, not an executed experiment or a statistically supported ranking.
