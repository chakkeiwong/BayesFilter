# GenUT Four-Model NeuTra Readiness Result

Date: 2026-08-04

> **Superseded 2026-08-19** by
> `docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-result-2026-08-18.md`
> and the execution checkpoint for the Austria-blocked-reason claims: the
> tangent-free/tangent-carrying mismatch described here was root-caused
> (redundant JVP iteration-start standardization) and repaired via a shared
> primal with forward autodiff; CPU value/score now match independent forward
> autodiff exactly at `T=1,2,20`. Austria remains blocked, but by the GPU
> graph-mode identity failure and the XLA `T=20` nonfinite hard veto, not by
> the mismatch recorded here. The three-model admission and HMC nonclaims are
> unaffected.

Plan: `docs/plans/bayesfilter-genut-four-model-neutra-readiness-plan-2026-08-04.md`

Aggregate: `docs/benchmarks/artifacts/genut_four_model_neutra_readiness_20260804/aggregate_attempt04/result.json`

Verdict: `THREE_MODELS_TRAINING_ELIGIBLE_ONE_MODEL_BLOCKED_NONE_HMC_READY`

## Outcome

The campaign implemented and tested a genuine leading-batch TensorFlow/XLA
GenUT posterior value/score route.  It is eligible to enter a separate serious,
target-specific NeuTra training campaign for LGSSM, KSC-SV, and predator-prey.
Austria SIR is not eligible because its tangent-carrying value/score route and
tangent-free endpoint do not compute the same finite scalar.

This is not completion of NeuTra training and not readiness to launch a
claim-bearing HMC run.  No learned transport has been selected, no training
protocol or hyperparameter ladder has been completed, and no HMC chain has been
run.  The next justified action is serious target-specific NeuTra training for
the three admitted targets only.  Austria needs a finite-program repair before
training.

## Readiness Matrix

| Model | Final arithmetic scope | Exact-scope tuning | Real-scope scalar parity | Two-step same-value FD | Cross-process replay | `B=4` GPU/XLA | One optimizer update | Training eligible | HMC ready |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LGSSM | FP32, TF32 disabled | Pass | Pass | Pass | Pass | Pass | Pass | Yes | No |
| KSC-SV | FP32+TF32 | Pass | Pass | Pass | Pass | Pass | Pass | Yes | No |
| Predator-prey | FP32+TF32 | Pass | Pass | Pass | Pass | Pass | Pass | Yes | No |
| Austria SIR | FP32+TF32 diagnostic | Historical scope only | Not reached | Not reached | Not promoted | `B=2` only | Not run | No | No |

LGSSM is a model-specific arithmetic exception.  Its TF32 route failed the
predeclared real-scope score-parity tolerance and is rejected.  This does not
change the repository-wide TF32 default for other models.

## Numerical Evidence

| Model | Target signature | Endpoint value relative error | Scalar parity value / max score relative error | FD steps and max relative errors | Replay value / score relative error |
|---|---|---:|---:|---|---:|
| LGSSM | `f33c1f18...aeaa` | `1.08e-7` | `0 / 1.72e-6` | `0.008: 0.00220`, `0.016: 0.00983` | `0 / 0` |
| KSC-SV | `53c41570...3627` | `0` | `0 / 1.19e-5` | `0.016: 0.01810`, `0.032: 0.00796` | `0 / 0` |
| Predator-prey | `41b9b4f2...0dbe` | `6.41e-6` | `1.88e-6 / 0.001684` | `0.016: 0.04910`, `0.032: 0.02049` | `0 / 0` |
| Austria SIR | `ddd4f9a7...bcd3` | `0.001568` versus `0.0002` | Not admissible | Not admissible | Not admissible |

All finite-difference stencils retained the same validity branch.  KSC-SV's
additional `h=0.064` explanatory arm failed at `0.0717`, bracketing the accepted
low-error interval at `h=0.016,0.032`; it does not invalidate those predeclared
passing steps.  Predator-prey also passed an additional `h=0.064` arm at
`0.0446`.  These are numerical derivative checks of each fixed finite target,
not filter-accuracy or posterior-correctness evidence.

The target-aligned LGSSM Kalman gross-error diagnostic passed in NeuTra
coordinates.  Observed posterior-value error was `0.796`, maximum absolute
posterior-score error was `3.084`, and score-direction cosine was `0.699`, under
the deliberately loose predeclared screens `10`, `10`, and `>0`.  This rejects
a gross target mismatch; it does not establish Kalman equivalence.

## Capacity And Training Interface

| Model | `B=4` allocator peak | One-step allocator peak | Condition estimate available | Minimum eigenvalue available | Scalar/row fallback |
|---|---:|---:|---:|---:|---:|
| LGSSM | `269,325,312` bytes | `135,105,280` bytes | No | No | None |
| KSC-SV | `71,500,288` bytes | `37,942,528` bytes | No | No | None |
| Predator-prey | `268,643,328` bytes | `134,425,856` bytes | No | No | None |

Each one-step run used a real `PlainDenseIAFTransport` optimizer update with
batch size two, GPU, XLA, deterministic operations, memory growth, and the
repository batch-native binding.  No Python loop over training rows,
`tf.map_fn`, `tf.vectorized_map`, scalar callback, or scalar fallback was used.
The scalar parity runner is an independent diagnostic and cannot update NeuTra
parameters.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Admit LGSSM target to serious NeuTra training | All final no-TF32 target/interface gates pass | TF32 scope rejected; no-TF32 scope passes | Training geometry and downstream HMC behavior unknown | Target-specific no-TF32 NeuTra training protocol | No HMC or posterior claim |
| Admit KSC-SV target to serious NeuTra training | All final TF32 target/interface gates pass | No target/interface veto | `T=1000` training cost and geometry unknown | Target-specific TF32 NeuTra training protocol | No HMC or posterior claim |
| Admit predator-prey target to serious NeuTra training | All final TF32 target/interface gates pass | No target/interface veto | Training stability and downstream geometry unknown | Target-specific TF32 NeuTra training protocol | No HMC or posterior claim |
| Block Austria SIR | Endpoint/value identity fails | Hard same-finite-program veto | Root cause within tangent-free versus tangent-carrying recurrence remains | Localize and repair finite value semantics before retuning | Not evidence against GenUT on all models |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Three targets pass; Austria fails the endpoint/value identity veto |
| Statistically supported ranking | None; no method or model ranking was tested |
| Descriptive-only differences | Runtime, allocator peaks, FD error magnitudes, and the TF32/no-TF32 localization |
| Default readiness | Not established for any model |
| HMC readiness | Not established for any model |
| Next evidence needed | Per-target NeuTra architecture/optimizer/budget search, heldout transport criteria, then sequential NeuTra-HMC tuning and confirmation |

## Evidence Ledgers

### Engineering Correctness

- True `theta[B,p] -> value[B], score[B,p], status[B,...]` tensor path exists.
- Forward JVP has bounded time memory and the endpoint does not allocate tangents.
- Repository admission reads the fail-closed aggregate, binds controls and
  signature, enforces TF32 mode, and rejects Austria.
- Focused CPU-hidden suites passed, ending with `39/39` after telemetry and
  admission regressions.  Successful admitted construction was also checked in
  trusted GPU processes for all three eligible targets.

### Numerical Validity

- Same-process replay is exact and paired fresh-process replay is exact for the
  final three target identities.
- Real-scope scalar value/score parity passes for the final arithmetic scopes.
- Two adjacent valid finite-difference steps pass for every admitted target.
- LGSSM TF32 scalar score parity failed and is preserved as a rejected scope;
  the retuned no-TF32 scope passes.
- Austria value-only and value/score routes disagree materially and are not
  admitted.

### Scientific Interpretation

- Passing means the finite GenUT posterior can be consumed by the current
  batch-native NeuTra trainer at the tested points and batch sizes.
- It does not show that GenUT approximates the scientific posterior well away
  from those points or that a learned transport will be useful.
- It does not show HMC convergence, posterior agreement, robustness across
  particle designs, or superiority to another filter.

## Attempt And Repair Ledger

| Attempt or issue | Classification | Repair or decision |
|---|---|---|
| Condition estimate recorded as available merely because a placeholder field existed | Telemetry implementation defect | Respect explicit availability tensor; regression added |
| KSC-SV and predator-prey tuning identities predated deterministic operations | Stale tuning evidence | Fresh deterministic exact-scope tuning artifacts |
| LGSSM initially bound historical scalar-route tuning | Scope-policy gap | Fresh batch-route exact-scope tuning and full downstream rerun |
| Suggested LGSSM Kalman helper used transition-before-first-observation | Wrong comparator event order | Target-aligned initial-observation Kalman recurrence |
| Kalman source replay missed an auxiliary `1e-9` threshold by `7.4e-8` | Reference tolerance defect at float32 observation boundary | Repaired auxiliary tolerance to `1e-6`; scientific gross-error screens unchanged |
| LGSSM TF32 real-scope score parity failed | Numerical arithmetic-scope veto | Rejected TF32 scope; retuned and reran FP32-no-TF32 scope |
| KSC-SV additional `h=0.064` FD failed | Explanatory truncation diagnostic | Preserve failure; use passing `0.016/0.032` interval |
| Austria endpoint mismatch | Scientific/numerical candidate veto | Stop Austria before training; shared core and other models continue |

## Post-Run Red Team

The strongest alternative explanation for the three passing targets is that
the checks cover a small proposal neighborhood around one center and one fixed
particle design.  A target can pass every interface and derivative gate yet be
too rough, biased, or poorly scaled over the region visited during real
training.  One optimizer update can also pass even when optimization later
diverges or overfits a narrow region.

Evidence that would overturn the readiness decision includes any target-status
failure during a predeclared training region, loss of replay or scalar parity
at representative heldout points, nonfinite multi-step training, or stale
target identity at endpoint evaluation.  Evidence that would unblock Austria
is equality of the tangent-free and tangent-carrying values for the same frozen
program, followed by fresh exact-scope tuning and the complete gate ladder.

The weakest current evidence is downstream relevance: no learned transport or
HMC chain has yet tested whether the batch-native score is useful across the
posterior.  Therefore the correct next milestone is serious NeuTra training,
not an HMC-readiness declaration.
