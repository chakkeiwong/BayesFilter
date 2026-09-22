# SSL-LSTM q=20 performance and whitening continuation result

Date: 2026-09-02  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-plan-2026-09-02.md`  
Authority: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Status: `CLOSED_N0_N2_NO_NOMINATION_N3_BLOCKED`

## Outcome

The bounded M3-C diagnostic completed on GPU 0 in `417.3732072210405` seconds.
The repaired run passed the source, target, device, memory-growth, finiteness,
and update-health checks. It found no candidate satisfying the predeclared
nomination rule: none of the three arms reduced held-out aggregate pullback
score RMS by at least 10 percent on two of three seeds. The result is a
diagnostic rejection of these short ladder candidates, not evidence against
the tempered reverse-KL research direction as a whole.

The exact grouped HMC path remains unadmitted. Its outputs were finite, but its
state, target, gradient, and log-acceptance values differed from the scalar
receipt. The explicit `tf.while_loop` control that calls the scalar graph was
exactly equivalent, so the discrepancy is in the proposed grouped TFP
semantics rather than in the scalar reference or the fixture's row-loop
plumbing.

## Evidence and provenance

The authoritative continuation receipt is
`docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n0-n2-02-gpu/run_manifest.json`
with manifest hash
`a73661f50e2b559e54549c950ccaadaa0b027c1e9dbbdecd74971f692e68f4e4`.
The fresh CPU N1 control receipt is
`docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n1-cpu-05/run_manifest.json`
with manifest hash
`649b10aaa9a00f4d5df8645c2f3abca89b24001861afc2425a0a52c51193d78c`.
Both receipts were produced at Git commit
`54201f5cd925ed15036bad8156606b812d53b045`; source hashes, command lines,
TensorFlow version, device state, and memory-policy fields are in the
manifests. The GPU run used TensorFlow 2.20.0, one RTX 4080 SUPER, strict
`tensorflow_eigh` diagnostics, TF32 enabled, and memory growth verified before
logical-device initialization.

The post-run hash and regression reconciliation is recorded in
`docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n4-r4-closeout.json`.

The earlier GPU attempt
`n0-n2-01-gpu/run_manifest.json` is preserved as a harness-repair artifact.
It used an arm-dependent validation seed, so its cross-arm comparison is
invalid. Its within-arm values are not used for a research decision. The
second run changed the validation bank to a common seed per seed index before
being treated as evidence.

## Regression audit

The route-specific regression command

```text
pytest -q tests/test_tempered_lineage_transitions.py \
  tests/test_tempered_transport_ensemble.py \
  tests/test_fixed_transport_hmc_step_cap.py
```

passed with `38 passed` in `15.44` seconds. The broader focused command
(`tests/test_fixed_transport_hmc_step_cap.py`,
`tests/test_fixed_transport_hmc_tuning.py`,
`tests/test_hmc_tuning_posterior_oracle.py`,
`tests/test_lgssm_neutra_serious_validation.py`, and
`tests/test_experimental_batched_svd_sigma_point_tf.py`) returned `113 passed,
2 failed`. The failures are outside this continuation: the ordinary-tuner
oracle expects a legacy calibration pass that is false under the current
measured-policy migration, and the LGSSM static-input test requires the absent
private file `docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026-07-13/phase7_campaign/private/retained_samples.npz`.
Neither failure touched the M3-C target, harness, or artifact contract, so
they are recorded as repository migration/data-fixture debt rather than a
continuation veto. They must not be reported as a green repository-wide suite.

GPU command actually run:

```text
BAYESFILTER_NEXT_MAX_SECONDS=500 BAYESFILTER_NEXT_ATTEMPT_ID=n0-n2-02 BAYESFILTER_GPU_ID=0 \
bash scripts/run_ssl_lstm_q20_performance_whitening_next_gpu.sh \
  --output-dir docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n0-n2-02-gpu
```

CPU control actually run:

```text
BAYESFILTER_NEXT_ATTEMPT_ID=n1-cpu-05 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_performance_whitening_next_2026_09_02.py \
  --cpu-n1-only \
  --output-dir docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n1-cpu-05 \
  --max-seconds 300
```

## Phase closeout

### N0/R0: source and contract audit

The route scan found no `tf.map_fn`, `tf.vectorized_map`, Jacobian-pfor, or
direct `pfor` token in the four inspected runtime files. The target signature
and strict square-root backend matched the plan. The GPU launcher set
`TF_FORCE_GPU_ALLOW_GROWTH=true` before import and the manifest reports growth
for every visible physical GPU. The CPU N1 control also passed after the
source-level repair. N0/R0 therefore closes without a continuation veto.

### N1/R1: grouped-transition fixture

The fixture used four rows, step size `0.20`, and three leapfrog steps. The
scalar and explicit row-loop paths were finite and each compiled to one trace.
The row-loop control matched the scalar path exactly at the declared
`1e-12` tolerance:

| Quantity | Row-loop error versus scalar |
|---|---:|
| State maximum absolute error | `0` |
| Gradient maximum absolute error | `0` |
| Target maximum absolute error | `0` |
| Log-acceptance maximum absolute error | `0` |
| Accept/reject equality | `true` |

The fast grouped TFP call was finite and also had one trace, but it was not
equivalent:

| Quantity | Fast grouped maximum error versus scalar |
|---|---:|
| State | `1.9390212244446556` |
| Gradient | `1.9390212244446556` |
| Target | `0.86641743413635` |
| Log-acceptance | `0.008664174341362962` |
| Accept/reject equality | `true` |

The receipt records expected target-call counts of scalar total `4`, fast
grouped `1`, and row-loop `4`. These are accounting expectations from the
fixture, not an instrumented proof of internal TFP calls. R1 therefore fires
the integration veto for the fast grouped path while confirming the scalar
reference and row-loop control.

### N2/R2: common-bank training ladder

Each arm used three seeds, batch size 32, beta 0.5, 12 finite updates, and
gradient clipping 10. Calibration streams were arm-specific. The held-out
latent bank was shared across arms for each seed index (`98000`, `98100`,
`98200`), which repairs the comparability defect in the first attempt.

| Arm | Seed changes in aggregate held-out score RMS | Seeds meeting `<= -10%` | Nomination |
|---|---|---:|---|
| A `(16,16)`, 2 stages, `1e-3` | `-5.3908%`, `-1.2066%`, `-5.4949%` | 0/3 | No |
| B `(16,16)`, 2 stages, `3e-4` | `-1.4538%`, `-1.2755%`, `+0.3022%` | 0/3 | No |
| C `(32,32)`, 3 stages, `3e-4` | `-2.2830%`, `-2.0148%`, `-0.4583%` | 0/3 | No |

All 9/9 candidates were finite and all 9/9 had 12/12 valid updates. The
centered log-density changes were descriptive only (A `-3.83%,-5.61%,-6.47%`;
B `-1.16%,-1.64%,-0.82%`; C `-2.85%,-2.30%,-3.05%`). No ranking is supported:
there are three seeds per arm, one short run, and no uncertainty interval.
The run's `default_change_allowed` field is `false`.

The large residual scale remains. For example, the final aggregate score RMS
was approximately `669.45`, `894.00`, and `1617.61` for A's three seeds; the
corresponding C values were approximately `1187.17`, `1243.72`, and `1778.54`.
These values are diagnostic and do not define an IID-Gaussian tolerance.

### N3/R3 and N4/R4

N3 was not entered because N1 did not establish fast grouped-transition
equivalence and N2 nominated no candidate. No HMC wrapper, optimizer default,
seed policy, target, or whitening claim was changed. N4 closes the bounded
continuation with this result, the reset memo, and the master refresh below.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Use common-bank ladder as evidence | Shared held-out bank, disjoint calibration, complete manifests | Pass on repaired run; first attempt quarantined | Short optimization horizon | Preserve and use only `n0-n2-02` for this decision | Generalization beyond this bank |
| Integrate fast grouped HMC | Exact state/gradient/target/log-acceptance equivalence | Failed | TFP stateless-seed semantics and kernel construction | Keep scalar/row-loop route; require a separately designed deterministic grouped kernel | Any speedup or sampler validity |
| Promote an N2 chart/training setting | 2/3 seeds with at least 10% score-RMS reduction | Failed for A, B, C | Three seeds and 12 updates are insufficient to distinguish under-training from capacity/objective limits | Write a new reviewed hypothesis only if more diagnostic budget is approved | Whitening, posterior correctness, convergence |
| Open M3 replay or Phase 9B | Master entry gate and valid grouped/performance evidence | Blocked | Original replay resource failure remains terminal | Keep M3 terminal and Phase 9B closed | Replay feasibility or HMC readiness |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | No target/signature, finiteness, memory-growth, route-scan, or artifact veto fired on the repaired run. The grouped-equivalence veto fired; the first run's cross-arm comparability veto is quarantined. |
| Statistically supported ranking | None. No interval, paired test, or MCSE-aware comparison was run. |
| Descriptive-only differences | Per-seed score/log changes, losses, gradient norms, trace counts, and wall time. |
| Default readiness | Not established; no active default changed. |
| Next evidence needed | A new, reviewed hypothesis with longer or better-targeted training evidence, and a separately designed semantically equivalent grouped kernel if batching remains a goal. |

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| Three seeds and 12 updates | Provisional plan hypothesis | Misses slow optimization or seed variability | Per-seed update trace and held-out residual | Diagnostic only |
| 10% and 2-of-3 nomination rule | Provisional separation threshold | Can reject useful smaller repairs or accept noisy ones | Replication and uncertainty analysis | Not a default |
| Shared validation bank per seed | Repair derived from post-run audit | A differing bank confounds arms | Seed-map check in manifest | Required for valid cross-arm reading |
| Batch size 32 and beta 0.5 | Target-specific ladder hypothesis | Changes optimization geometry | Per-update validity and residual diagnostics | Not promoted |
| GPU 0, XLA, TF32, memory growth | Repository execution policy | Device/backend mismatch or allocator failure | Manifest and trusted device receipt | Passed for this run |

## Post-run red-team

The strongest alternative explanation for the failed nomination is
under-training: twelve updates may be too short for the transport to respond.
The competing explanation is that the current chart/objective is poorly
conditioned for this target. The common-bank repair removes one comparison
confound but does not distinguish those explanations. A future longer run
could still fail to whiten, and a lower residual on one bank would not prove
global Gaussianization or mode discovery. The weakest evidence here is the
small stochastic ladder; its role is screening, not ranking.

## Authority transition

The active next plan is closed by this result. The master now records M3 as
terminal, M3-C as complete without nomination, N3 as blocked by its declared
entry conditions, and Phase 9B/M4 as still blocked. Any further work requires a
new dated subordinate plan with its own evidence contract, budget, seeds, and
skeptical audit. No prior partial replay calls may be reused.
