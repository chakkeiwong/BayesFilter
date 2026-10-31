# Structural integration result within the surrogate-HMC master

The additive structural transition now executes through the actual canonical
batched analytical DCTR route. The AR(2) lag is exact before and after flow;
the full finite-program score agrees with central differences. This closes
the earlier singular-full-state-Q implementation failure for additive affine
innovation charts. Nonlinear current-innovation completion manifolds remain
outside this implementation.

Evidence: `artifacts/surrogate-hmc-closure-20260922-structural-02/`.
The command was `scripts/run_surrogate_hmc_program.py --stage closure
--closure-stages structural-integration --output
docs/plans/artifacts/surrogate-hmc-closure-20260922-structural-02`, using
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`. The archived manifest binds
the commit, source snapshot, seeds, device policy, commands and resources.

The focused suite passed 95 tests with one skip. GPU/XLA FP64 checks had
zero lag residual, maximum relative analytical-score error 4.54e-11 (linear)
and 3.65e-11 (nonlinear observation control), batch/single error <=8.89e-16,
and full-rank B=I parity error 4.45e-16. All tested parameters include flow,
weighting, full-state reset and correction dependence; no autodiff force is
substituted. Singular/invalid chart tests fail closed.

Attempt 01 had 92 passes, three regression-harness failures and one skip.
One misplaced assertion and two tests reading live stale M10 source-bound
history caused those failures. The tests now isolate their historical inputs;
the runtime source guard is unchanged. All numerical structural tests already
passed in attempt 01. Attempt 02 passed both tests and numerical stage, and
source hashes were unchanged during both attempts.

| Decision | Primary criterion | Vetoes | Main uncertainty | Next justified action | Unsupported conclusion |
|---|---|---|---|---|---|
| Accept additive structural mechanics | Support, score and identity parity pass | No numerical veto in tested cases | FP32 transfer and inferential scope controls untested | Continue independent AR2 model reference, then scoped controls and public HMC tuning | Posterior correctness or all-structural-model support |

| Inference status | Finding |
|---|---|
| Hard veto screen | No supported veto in checked mechanics; bad charts rejected |
| Statistically supported ranking | None; no sampler comparison ran |
| Descriptive-only differences | Numerical residuals above concern deterministic implementation checks |
| Default-readiness | Not established by these mechanics checks |
| Next evidence needed | Scoped FP32 validation, model/finite posterior references and retained HMC comparisons |

Strongest alternative explanation: these small, well-conditioned fixtures may
miss numerical problems at posterior-tail scales or poor chart conditioning.
Such failures in the forthcoming scoped checks would overturn broader
admissibility, without invalidating the tested finite-program identities.

Charges for both structural attempts: 524.722861 CPU-worker and 82.625520 GPU
seconds. Campaign cumulative: 28,926.155274 CPU / 26,705.052376 GPU seconds;
remaining: 25,073.844726 CPU / 200,094.947624 GPU. The current structural pass
retains 1,875.277139 CPU / 3,517.374480 GPU seconds. The AR2 reference stage is
inside this reservation. This is an existing-master continuation, not a new
direction, and there is no scientific or budget reason to stop here.

## AR2 reference and force-consumer continuation

Structural-03 qualified all three independently simulated T10 reference
partitions under the declared initial law and normal log-scale priors. The
Kalman/direct-observation Gaussian likelihood discrepancy was at most 3.56e-15;
reference-gradient central-difference relative error was at most 2.34e-10.
Nested grid/domain integration passed the predeclared allowance checks for all
13 posterior functions. These are empirical integration allowances, not proven
tail-error bounds.

Structural-04 qualified N64/T10 canonical FP32 proposal controls on calibration
and validation only. Every tested nominal and three-prior-SD corner was valid.
The first stable protection pair selected multiplier 1.0. The largest
normalized FP32/FP64 discrepancy was 0.00112305, below 0.002. Twenty focused
checks passed, including actual consumer force/endpoint wiring and derivatives
of both log-noise scales through the finite canonical program and prior.

| Decision | Primary criterion | Vetoes | Main uncertainty | Next justified action | Unsupported conclusion |
|---|---|---|---|---|---|
| Admit this AR2 force candidate to HMC tuning preparation | Scoped controls and precision passed | No observed force invalidity in declared probes | Proposal mixing and finite-target approximation remain untested | Existing master: cost/precision design, scoped MCSE calibration, public tuning, B/R chains | Posterior correctness, speed advantage or C/D and D/R closure |

| Inference status | Finding |
|---|---|
| Hard veto screen | No supported numerical veto in reference and control probes |
| Statistically supported ranking | None; no AR2 chains yet |
| Descriptive-only differences | Numerical control sensitivity and precision differences |
| Default-readiness | Not established |
| Next evidence needed | Retained B/R posterior agreement with qualified uncertainty |

The strongest alternative explanation is that the tested force probes miss
posterior trajectories that fail or mix slowly. Endpoint-corrected HMC with
archived status and retained diagnostics will test that limitation directly.
Reference/control charges: 295.987003 CPU-worker and 247.368330 GPU seconds.
Campaign cumulative after structural-04: 29222.142277 CPU / 26952.420707 GPU.
This completes prerequisites, not the master; execution continues.
