# Surrogate-HMC recovery: what was tested and what remains open

**Status: diagnostic recovery complete; no model-inference candidate promoted.**
The research question remains whether an imperfect DCTR force can support
reliable parameter inference across the repository's models. Disagreement
between that force and a reference gradient is not, by itself, a reason to
stop this investigation.

The documents and results are recoverable from merged `main`. They do not yet
establish a successful DCTR surrogate-HMC model-inference experiment. The
Gaussian toy's recorded PASS has a reproducible acceptance-rule defect. The
older LGSSM successes concern different computations, and the recent LG1
failure is a filtering-calibration failure. Neither rejects the surrogate-HMC
research direction. A corrected inference experiment is the next useful step.

## Recovered programs and execution state

| Item | Recovered evidence | Finding |
|---|---|---|
| Initial recipe | [August 29 implementation plan](implementation_plan_surrogate_force_hmc_2026-08-29.md) | Frozen DCTR inputs; executed value for acceptance; damped analytical force for trajectories. Approximate log likelihood does not by itself justify a claim about the model posterior. |
| Original master | [September 4 master](ledh-surrogate-force-hmc-master-program-2026-09-04.md) | Infrastructure, route identity, toys, LGSSM, then a wider model phase. Criteria included posterior agreement, ESS/gradient at least half the comparator, acceptance at least 0.2. Planned about 61 GPU-hours. |
| Intermediate replacement | [September 6 unified program](ledh-surrogate-hmc-unified-program-2026-09-06.md) | Preserve as program history, not a separate source of additional compute allowance. |
| Later executable master | [September 7 master](ledh-surrogate-hmc-executable-master-program-2026-09-07.md) | Explicitly targets the frozen finite-particle posterior; excludes a model-posterior claim. Changes phases and narrows model coverage. Forecasts 32.5–44.5 GPU-hours for the whole program, without a reconciled actual-use ledger. |
| Earlier recognition of drift | [September 12 reconciliation](ledh-surrogate-hmc-program-reconciliation-2026-09-12.md) | Identified the lost damping-calibration phase and its replacement by seed checks. Did not supply missing posterior evidence. |
| Review requested by the owner | [Codex review memo](artifacts/ledh-graceful-failure-codex-review-memo.md) | Exact named document recovered. It concerns the subsequent numerical-containment program. |
| Original revised program | [September 17 revised program](ledh-graceful-failure-revised-program-2026-09-17.md) and [original snapshot](artifacts/ledh-graceful-failure-revised-program-20260917-01/codex-original-program-snapshot.md) | Original content was already restored; it is not currently lost. |
| Claude's review | [separate review](ledh-graceful-failure-revised-program-claude-review.md) | Approves the revised program and recommends LG1 first. Its success probabilities are reviewer estimates without supporting calibration; they are not evidence of feasibility or inference success. |
| Current containment results | [execution result](ledh-graceful-failure-execution-result-2026-09-18.md) | Engineering checks succeeded; three LG1 calibration attempts failed the prescribed filtering screen. No retained posterior comparison. |

History resolves the overwritten-program question. Commit `153f9667` contained
the intervening 723-line review/replacement. Commit `1c885c5e` restored the
584-line original; that blob matches the protected original snapshot byte for
byte (SHA-256 `67358680a052102f3b7d52ce216bc7933ef45b3d0fb995c2615c782f996c2183`).
At `6fbcf314`, the substantive subsequent change explicitly requires
`rho(F) < 1` before solving for stationary covariance. The
[restoration record](artifacts/ledh-graceful-failure-revised-program-20260917-01/checkpoint.md)
and [pre-restoration replacement](artifacts/ledh-graceful-failure-revised-program-20260917-01/claude-replacement-before-restoration.md)
preserve both versions. No document was overwritten during this recovery.

The shared `/home/chakwong/BayesFilter` checkout is dirty and was inspected
read-only. Its `main` and `surrogate-hmc` refs both resolve to
`6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef`. This isolated execution checkout is
on `ledh-graceful-failure-execution-20260917` at
`8a4cffa2fd085f43ab5710c24d6e7b48770aea0d`, one containment commit above that
baseline. Pre-existing edits to the execution result and LG1 regression-audit
files were preserved. No merge, commit, push, environment mutation, or new GPU
run was performed by this recovery.

## What the actual acceptance rule must target

For parameters theta and data y, distinguish

1. the model posterior, proportional to `p(theta) L(theta; y)`;
2. the frozen finite-DCTR posterior, proportional to
   `p(theta) exp(ell_DCTR(theta; y, omega, c_value))`; and
3. the deterministic proposal field from the analytical DCTR computation with
   frozen inputs/settings `omega, c_force`, which may disagree with either
   posterior gradient.

With unconstrained coordinates z and theta = b(z), the endpoint log density
must also contain `log |det Db(z)|`. Its negative is U(z). The original dual
adapter's word “exact” means the chosen executed finite value, not the exact
model likelihood. The
[adapter](../../bayesfilter/inference/ledh_dual_parameter_target.py) returns
log likelihood alone; a caller must supply the prior and coordinate Jacobian.

The validity argument does not require the field to be the gradient of U.
For fixed mass M, step h and deterministic finite position-only field g, a kick
`(z,p) -> (z,p-h*g(z))` preserves volume: for each fixed z it is a translation
of p. A drift `(z,p) -> (z+h*M^-1*p,p)` does likewise. Their inverses use -h.
A symmetric half-kick/drift/half-kick composition S obeys
`R S R = S^-1`, where R flips momentum. Thus `R S^L` is a volume-preserving
involution for fixed L. Metropolis acceptance with
`min(1, exp[-U(z')-K(p')+U(z)+K(p)])` preserves the extended target, with
`K(p)=p^T M^-1 p/2` and fresh Gaussian momentum. Each independent chain needs
its own U and K. This derivation is local to the algorithm, not an unchecked
appeal to a paper's corollary.

It assumes a frozen field, fixed retained-run kernel, and the stated proposal
map. Numerical overflow, asymmetric early termination, stateful randomness,
or trajectory-dependent repairs require separate treatment; merely rejecting
an arbitrary failed computation is not a proof of reversibility. A finite
position-only zero extension of a failed force can retain the kick argument,
but must be deterministic and tested in both directions. Excluding points
where the *endpoint* value fails can instead change the sampled support and
requires an exclusion-mass assessment.

The supported implementation is
[neural_force_hmc.py](../../bayesfilter/inference/neural_force_hmc.py):
`neural_force_proposal` executes the symmetric map and terminal flip;
`neural_force_hmc_transition` evaluates per-chain endpoint and kinetic energies.
Despite its name, it accepts an honestly labeled deterministic position field;
a neural network or NeuTra transport is not required. The
[capability registry](../../bayesfilter/inference/tuning_contract.py) and
[public interface](../reference/hmc-tuning-interface.md) support
`FrozenPositionOnlyForce` + `FrozenTargetPotential` through
`bind_neural_force_hmc_tuning_runner` and the public `tune_hmc_kernel`.
The binding and chain runner cannot issue tuning artifacts themselves.

Five existing focused tests passed in this recovery: typed binding with an
imperfect field, proposal involution, full-energy endpoint accounting, a
counterexample to dropping kinetic energy, and batch permutation/shape/dtype.
[Recorded command and status](artifacts/surrogate-hmc-recovery-20260918-01/shared-hmc-mechanics.json),
[complete log](artifacts/surrogate-hmc-recovery-20260918-01/shared-hmc-mechanics.log).
These were CPU-only, non-JIT debugging checks, with GPUs hidden. They do not
verify the complete DCTR-to-tuner consumer, GPU/XLA execution, convergence, or
posterior agreement. That remaining endpoint wiring must be checked before a
new inference run.

## The earlier simple-function evidence

The [September 4 result](ledh-surrogate-hmc-phase2-result-2026-09-04.md)
already corrects its own original COMPLETE label: the prescribed acceptance
ladder was never measured. A force-norm check had replaced it. Determinism and
small trajectory diagnostics remain useful mechanics evidence, not a sampled
posterior result.

The September 7 experiment did run a manual leapfrog toy. Its target was a
three-dimensional centered Gaussian with precision `diag(1,4,9)`; the
alternative log-density force was scaled by `1/1.01`. It used two chains,
500 warm-up and 1,000 retained transitions per chain, step 0.1, ten leapfrog
steps, seeds 12345/12346, float64 CPU eager execution. The
[tracked output](../../results/phase2-output.txt) and
[preserved ignored summary](artifacts/surrogate-hmc-recovery-20260918-01/preserved-phase2-summary.json)
report acceptances 0.775/0.758 and an alleged W2 of 0.069214 below 0.1.
The summary's GPU accounting conflicts with the CPU script and is not a
reliable remaining-budget ledger. No complete retained-draw artifact or
source-bound run manifest was located.

The actual [script](../../scripts/phase2_toy_potential_test.py) has two
material defects, reproduced without running another stochastic chain:

| Checked quantity | Actual computation | Consequence |
|---|---|---|
| Independent-chain Metropolis potential | `reduce_sum(theta**2 * Q)` sums both coordinate and chain axes, returning one scalar. Kinetic energies and accept/update decisions remain per-chain. | Wrong relative to the independent-chain MH target. With per-chain current energies `[0.5,2]`, proposed energies `[2,0]`, and zero kinetic difference, correct log ratios are `[-1.5,2]`; the script gives `[0.5,0.5]`. At log-uniforms `[-1,-1]`, the first chain is wrongly accepted. |
| Gaussian Wasserstein statistic | `sqrt(||mean_a-mean_b||^2 + ||cov_a-cov_b||_F^2)` | Neither Gaussian W2 nor its asserted upper bound. For one-dimensional standard deviations 0.1 and 0.2, it gives 0.03; actual W2 is 0.1. |

[Actual-source counterexamples](artifacts/surrogate-hmc-recovery-20260918-01/toy-arithmetic-audit.json)
and [log](artifacts/surrogate-hmc-recovery-20260918-01/toy-arithmetic-audit.log)
preserve the checks. The source checksum matches the recovered completion
commit `36f3d3b0`, later `c361e377`, merged main, and this checkout. The test
script was preserved unchanged in the evidence directory. Its acceptance bug
therefore cannot be explained as a new containment change.

The historical PASS is **invalid as a posterior demonstration**. Plausible
sample moments do not rescue an incorrect transition rule. The tolerance was
also adjusted after an observed failure, without measured autocorrelation-based
uncertainty. None of this disproves surrogate HMC; the derivation above and
the separate shared-kernel mechanics checks address a different question.

## Reconciling LGSSM success with subsequent failure

| Evidence | Actual scope and result | Relationship to the current question |
|---|---|---|
| [August 30 correction](../benchmarks/surrogate_force_correction_and_graph_diagnosis_20260830.md) | Retracts earlier LGSSM claims based on short runs, mismatched hashed seeds, and incomplete diagnostics. | The earlier final-results/progress prose is not independent posterior evidence. |
| [September 3 trust-region result](../benchmarks/artifacts/ledh_trust_region_lgssm_t50_20260903/result.json) and [runner](../benchmarks/run_ledh_trust_region_phase3_lgssm_t50.py) | d=3, T=50, N=1008 at a fixed five-entry theta; GenUT cloud. Calls `cubature_genut_filter.finite_value_score`; checks finite output and correction diagnostics, without Kalman agreement or retained HMC. | A real SUCCESS for its stated computation, not full canonical DCTR or parameter-posterior validation. Post-August-21 date alone does not make a reduced route eligible. |
| [Phase 3.5 result](../benchmarks/ledh_phase_3_5_result.md) | Austria SIR d=18, observation dimension 9, N=252, T=50; graph/timing consistency work. | Does not substantiate a successful LGSSM posterior experiment, despite the executable master's broader phase wording. |
| [Phase 4a diagnostic](ledh-surrogate-hmc-phase4a-diagnostic-result-2026-09-07.md), [runner](../benchmarks/ledh_surrogate_hmc_phase4a_diagnostic.py) | Two approximately 55–56 minute attempts ended in Cholesky warnings/nonfinite propagation during the first arm; no exact-versus-damped posterior comparison. | Numerical implementation failure. The logs do not identify the failed matrix or prove the cause was insufficient damping. Force mismatch was not the observed rejection criterion. |
| [Recent LG1 calibration](ledh-graceful-failure-execution-result-2026-09-18.md) | d=1, T=10, varying phi, scope-specific finite designs and controls; none passed all prescribed calibration checks after three attempts. | Rejection of those filtering candidates under that screen; no HMC inference was attempted. |
| [Before/after LG1 audit](ledh-lg1-regression-audit-result-2026-09-18.md) | Identical inputs at `6fbcf314` and `8a4cffa2`, phi=-0.9, N=1024: covariance errors 16.000620% and 16.000531% against a 5% screen; scaled snapshot difference at most 9.08e-7 against 1e-4. | No material containment regression in the checked case. It does not prove parity at all settings, explain all error, or admit N=4096. |

The last substantive calibration command was:

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  docs/plans/artifacts/ledh-graceful-failure-20260917T105313Z-01/lg1_cap_repair_curves.py \
  docs/plans/artifacts/ledh-graceful-failure-20260917T105313Z-01
```

Its [manifest](artifacts/ledh-graceful-failure-20260917T105313Z-01/lg1-cap-repair-curves-01/manifest.json)
records `6fbcf314` plus dirty-source hashes, the conda environment, GPU identity,
and 229.425996 device-seconds. It exited zero: the evaluation completed, while
the candidates failed the scientific screen. The calibration screen required
mean/covariance error <=0.05 and centered log-likelihood-ratio error <=0.02
across its specified calibration datasets and parameter points. This was not
a dense parameter-posterior comparison. The manifest's top-level seed list is
empty; frozen input/design records and generation scripts must be followed for
seed provenance rather than treating that field as a complete manifest.

The old Phase 4a target has an additional problem before any timing question:
`_diagonal_lgssm_fused_model` uses only the first three theta entries, with q=0.35
and r=0.45 hard-coded. The runner passes `DualParameterLEDHTarget` directly to
TFP HMC without a prior or parameter chart. Its alleged five-parameter target
is consequently flat in two unrestricted directions. Wherever its likelihood
is positive, integration over those directions is infinite: this is not a
proper five-parameter posterior. This conclusion follows from the inspected
consumer-to-model source path; a full endpoint numerical invariance run was
**not checked** in this recovery. The runner also disables correction and
pairwise steps and sets the coordinate cap to zero, so it does not test the
owner's full DCTR configuration.

The two original failure logs now have local preserved copies:
[initial attempt](artifacts/surrogate-hmc-recovery-20260918-01/phase4a_sequential.log)
and [repair attempt](artifacts/surrogate-hmc-recovery-20260918-01/phase4a_repair.log).
Their terminal “direction changed the primal canonical value” assertion
compares NaNs after Cholesky failure. It does not demonstrate that two finite
primal values differ. A parameter proposal and failed factor were not saved,
so the precise numerical root cause remains unresolved.

The current [bounded LG fixtures](../../bayesfilter/highdim/ledh_program_fixtures_tf.py)
repair several old target-definition omissions: proper priors, a stationary
initial law, complete chart Jacobian, fixed random inputs, and the shared
canonical batch route with reset and correction requirements. Existing LG1
replay evidence checks that actual consumer. These improvements should be
carried forward; they do not turn a failed filtering screen into a posterior
result.

## Verified drift and the corrected next question

The September 17 revision explicitly distinguishes model/finite/guarded
posteriors and permits imperfect proposal forces. It is incorrect to say that
it flatly prohibits surrogate HMC. The departure is in emphasis, dependency
order, and coverage: numerical containment and filtering accuracy became the
work that actually ran; LG1 completion became a prerequisite for wider models;
and “all models” became eight fixtures, one filtering-only. The September 7
master had already narrowed the endpoint claim to the finite posterior and
shrunk the model list. The
[repository model inventory](ledh-surrogate-hmc-model-coverage-2026-09-18.md)
records the missing families and exact adapter/reference gaps.

A filtering-moment error can veto a particular filtering claim and nominate a
repair. It need not veto using that deterministic DCTR force with an exact
Kalman endpoint. Conversely, exact Metropolis correction to a finite DCTR
likelihood cannot remove that likelihood's model-posterior error. Test both
questions explicitly, in that order, with a proper scalar LG1 target and the
shared public tuner. The
[proposed inference continuation](ledh-surrogate-hmc-inference-continuation-2026-09-18.md)
defines this bounded next experiment. It preserves full DCTR and does not
replace the all-model objective with a permanent scalar-only program.

## Accounting and decision

| Ledger | Recorded charge | Remaining authority |
|---|---|---|
| September 17 containment campaign | GPU 1309.310489 s / 230400 s; aggregate CPU 3187.843224 s / 57600 s | Nominal time remainder 63.6363 GPU-hours / 15.1145 CPU-worker-hours. **All three substantive LG1 attempts used.** |
| Separate regression audit | GPU 243.392077 s; CPU 37.766647 s, including stated reserves | Audit budget explicitly closed; unused audit allowance is not a new campaign. |
| This recovery's two numerical checks | No GPU; CPU 8.631416 s; combined wall 8.708359 s | Two-check numerical allowance closed. Read-only inspection/report preparation are separate from these process measurements. |
| Older surrogate-HMC program | Forecasts and partial logs recovered; no consistent total actual-use ledger | Remainder unknown. Do not add its nominal budget to the current allowance. |

Conservatively debiting the separate audit and these checks to the containment
envelope as well leaves about **63.5687 GPU-hours and 15.1016 CPU-worker-hours**.
This conservative calculation is not a rewritten accounting record and does
not replenish the exhausted LG1 attempt count. The next proposed attempt needs
an explicit one-attempt extension under §12.3–12.4 of the active revised
program. This is a real recorded budget boundary, not a new approval-token
requirement. No fourth attempt was launched under a different label.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject historical toy PASS as inference evidence | Independent-chain MH arithmetic fails | Implementation/artifact validity veto | Original retained draws absent | Use tested shared mechanics and a proper target | Surrogate HMC is mathematically invalid |
| Preserve containment and LG1 replay work | Engineering checks and matched-input replay completed | No checked material regression | Other scopes and failure root causes | Carry fixes into new endpoint wiring tests | All old/future paths are correct |
| Continue surrogate-HMC research, subject to attempt budget | Model-posterior criterion not yet measured | No evidence rejecting the research direction | Cost, usable DCTR force, finite-likelihood bias | One proposed bounded LG1 inference attempt, then staged model expansion | Any model is already inference-ready |

| Inference status | Finding |
|---|---|
| Hard veto screen | Toy transition wrong; old advertised five-parameter target improper; old HMC attempts fail numerically; recent LG1 candidates fail their filtering screen. These are different vetoes. |
| Statistically supported ranking | None recovered or produced. |
| Descriptive-only differences | Historical acceptance, moment distances and timings; recent candidate error magnitudes. |
| Default-readiness | Not established. GPU/TF32/XLA DCTR remains the owner's implementation direction; scientific admission remains separate. |
| Next evidence needed | Supported tuner binding, scope tuning, frozen retained kernels, converged replicated chains, oracle agreement with uncertainty, exclusion accounting, conditional heuristic comparisons, and separately quantified finite-likelihood bias. |

Post-run red team: the strongest alternative explanation for the old apparent
toy success is that its weak damping and favorable Gaussian geometry hid a
wrong acceptance rule. Correct retained samples under the shared kernel would
overturn the *empirical failure of the demonstration*, not the identified
arithmetic defect in the old script. The weakest parts of this recovery are
missing original retained draws/manifests and unverified full DCTR-to-tuner
wiring; both remain explicit. No source-faithfulness, posterior correctness,
statistical superiority, or all-model completion claim follows from recovery.

Evidence and reproducibility: [recovery plan](ledh-surrogate-hmc-recovery-plan-2026-09-18.md),
[provenance](artifacts/surrogate-hmc-recovery-20260918-01/recovery-provenance.json),
[run manifest](artifacts/surrogate-hmc-recovery-20260918-01/run-manifest.json),
[checkpoint](artifacts/surrogate-hmc-recovery-20260918-01/checkpoint.md).
