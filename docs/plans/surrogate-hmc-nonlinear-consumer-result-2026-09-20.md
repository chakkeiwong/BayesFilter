# M10 full-DCTR consumer mechanics result

The actual M10 sine-transition/quadratic-observation consumer passed all15
declared GPU mechanics checks. This advances the test of surrogate HMC: the
proposal force is the complete canonical DCTR analytical score, while
acceptance uses an independently computed finite-grid potential. No retained
posterior comparison has run for M10.

The claimed proposal field and the quantity computed agree at the tested
points: direct execution of `canonical_batch_fused_value_score` gave identical
values, scores and validity/failure codes. The call includes the UKF particle
covariance lifecycle, Contract E--Chol and full dual-cap/GenUT correction at
N1024/K1024. Bitwise replay, row permutation, row independence, force sign and
independent endpoint bindings passed. The original field was healthy at all
four probes. Two-step proposal reversal error was2.7755575615628914e-17,
below the predeclared3.814697265625e-6 tolerance. Per-chain energy accounting,
endpoint cache consistency and both transition health checks passed.

The endpoint actually computed is the FP64 513-node, radius9 state-grid
likelihood plus the proper Normal(0,log(2)^2) prior on each log-noise scale.
Its equality to the nonlinear model likelihood throughout the full prior is
not established. The local reference supports only its declared probes.
Outside the diagnostic coordinate box the endpoint correctly reports a
numerical qualification failure, without changing prior support. This
endpoint cannot yet support retained model-posterior inference.

Four focused CPU tests passed before the GPU run: independent model/tangent
checks, shared-callable consumer wiring, prior-overflow failure reporting and
unqualified endpoint behavior. The plan is
[M10 consumer mechanics](surrogate-hmc-nonlinear-consumer-2026-09-20.md).
The frozen control multiplier1, identity mass, step size.02 and two leapfrogs
were mechanics hypotheses; none is a tuned sampling choice.

Evidence is in
[the report](artifacts/surrogate-hmc-all-models-20260920-m10-consumer-01/report.json),
[manifest](artifacts/surrogate-hmc-all-models-20260920-m10-consumer-01/run-manifest.json)
and that directory's `worker.log`, source snapshots and accounting.
The command was `python scripts/qualify_surrogate_hmc_nonlinear_consumer.py
--output docs/plans/artifacts/surrogate-hmc-all-models-20260920-m10-consumer-01`
using the tftwogpu environment, trusted GPU1, TF32/XLA, verified memory growth
and two/one TensorFlow intra/inter-op threads. All data/fixed-input seeds and
source hashes are in the manifest. Peak live allocator memory was33,623,552
bytes. The complete attempt charged51.067084383 GPU wall seconds and
53.708821203 CPU-worker seconds; the tests charged8.528390294 CPU seconds.
M10 cumulative use is51.067084383 GPU /104.854559497 CPU seconds of7200/1800.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Mechanics pass | All15 checks pass | No observed wiring, status, energy or reversal failure | Four local probes and two transitions do not cover parameter tails | Qualify full-prior reference, then scoped controls and public HMC tuning within budget | M10 posterior recovery, all-model validation or force accuracy |

| Inference status | Finding |
|---|---|
| Hard veto screen | No failure in the declared local mechanics checks |
| Statistically supported ranking | None; no retained-chain comparison |
| Descriptive-only differences | Steady two-leapfrog transition0.2996 seconds; timing is diagnostic |
| Default-readiness | Not established |
| Next evidence needed | Full-prior reference, M10 control calibration/validation, separate public tuning and retained equivalence comparisons |

Post-run red team: the strongest remaining alternative explanation for later
apparent posterior agreement is shared quadrature bias outside the local
reference probes. A larger state/parameter domain and independently refined
quadrature must address it before sampling. The weakest current evidence is
global endpoint accuracy, not the checked canonical call chain. A future
field failure triggers scope-specific repair; it does not by itself reject
the surrogate-HMC idea.
