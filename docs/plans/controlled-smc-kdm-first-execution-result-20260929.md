# Controlled-SMC / KDM execution result, 29 September 2026

**The first validation tranche is complete. Both empirical replications remain incomplete.**
The author controlled-SMC implementation runs, and the independently derived scalar
TensorFlow implementation passes fixed-input parity at the full 3,000-observation
horizon. The KDM Gaussian-mixture/IWSG primitive also matches the actual author
resampling endpoint. This establishes working numerical components; it does not
establish replicated learning curves or comparative performance.

Master: [active program](controlled-smc-kdm-replication-master-program-20260928.md).
Contracts: [first tranche](controlled-smc-kdm-execution-20260929.md),
[KDM primitive](kdm-iwsg-primitive-validation-20260929.md),
[Kalman control](controlled-smc-kalman-validation-20260929.md).
[Source-to-experiment map](../reference/controlled-smc-replication-source-map-20260929.md).

## Review findings and repairs

1. The original master was a scientific proposal without a successor dispatcher.
   Implemented `scripts/run_controlled_smc_kdm.py`, reusing existing accounting,
   process identity, timeout, recovery, source snapshots and manifest helpers.
   Future phase names in the master remain specifications, not implemented jobs.
2. The source demo and paper experiments were conflated by loose completion wording.
   The executed demo has N=1024 and I=2; Figure 4 right uses N=128, I=3 and
   100 repetitions at each of 20 variance values, with BPF N=5529. The published
   PDF now has a local copy and a scoped version/source mapping. Smoothing and
   PMMH are separate experiments. Source CPU cost matching cannot be assumed on GPU.
3. Native MATLAB was unavailable. An explicitly labeled Octave 6.4.0 reference
   with isolated, pinned statistics 1.5.0 now runs the unchanged author code.
   Missing RNG/JSON wrappers were harness compatibility failures, not cSMC failures.
4. A reference-index conversion and an expected-value FP64 ingestion issue were
   repaired without changing scientific tolerances or the numerical target.
   Attempt08 supersedes attempt06 as the short-fixture parity authority.
5. KDM's reference failed before sampling because the system C++ library lacked
   its environment's required ABI. A subprocess-only library-path correction
   repaired it, without installing packages. The two source 1e-8 additions in
   KDE weight normalization and IWSG weights remain explicit comparison settings.
6. General controlled-SMC adapters, full KDM sampling/training/optimizer consumers,
   normalized bearings baselines, Appendix B settings and empirical repetitions
   are still missing. These are real scope gaps; neither XLA nor a working
   primitive closes them.

## Executed evidence

All paths below are under `artifacts/controlled-smc-kdm-replication-20260928/`.
Every attempt preserves a command log, source snapshots and a manifest. Successful
GPU summaries record TensorFlow 2.20.0-dev0+selfbuilt, FP64, TF32 off, XLA on,
GPU:0 (RTX 4080 SUPER) and verified memory growth on both visible GPUs.

| Evidence | Result | Scope |
|---|---|---|
| [Author pilot](artifacts/controlled-smc-kdm-replication-20260928/attempt03_c1_pilot/summary.json) | PASS | N=256, H=128, I=2; saved fixed-input source fixture |
| [Full author demo](artifacts/controlled-smc-kdm-replication-20260928/attempt04_c1_demo/summary.json) | PASS, 26.644346 s charged | N=1024, H=3000, I=2; all policies, precisions and reported values finite |
| [Short FP64 parity](artifacts/controlled-smc-kdm-replication-20260928/attempt08_c2_parity_fp64/summary.json) | 49/49 checks pass | Actual forward/cloud-weight/backward-fit calls, source ancestry, integral/normalization and rejection controls |
| [Full GPU execution](artifacts/controlled-smc-kdm-replication-20260928/attempt07_c2_full/summary.json) | PASS, 31.019409 s charged | Fresh streams at each refinement; valid full-horizon execution, not source-RNG parity |
| [Full source parity](artifacts/controlled-smc-kdm-replication-20260928/attempt12_c2_full_parity/summary.json) | 34/34 checks pass, 35.558359 s charged | H=3000, N=1024, three source particle runs and two backward fits; exact replay ancestry |
| [Exact Kalman control](artifacts/controlled-smc-kdm-replication-20260928/attempt11_c2_kalman/summary.json) | 12/12 checks pass | One fit recovers optimal quadratic policy, constant weights and exact likelihood on two fresh streams |
| [KDM/IWSG source endpoint](artifacts/controlled-smc-kdm-replication-20260928/attempt10_k1_iwsg/summary.json) | 15/15 checks pass | Batched Gaussian densities, raw and self-normalized expectation gradients, stopped sample gradients and density normalization |

Full-horizon parity's largest absolute discrepancy was 1.8941e-8; its largest
error divided by the frozen absolute-plus-relative tolerance was 0.003504.
The Gaussian control's exact log likelihood was -10.501153258557636, with
maximum checked discrepancy 3.20e-14. The KDM checks' maximum discrepancy was
2.08e-13. These are deterministic numerical comparisons, not stochastic rankings.

The actual diagnostic endpoints load the snapshotted TensorFlow modules and
call `ScalarGaussianControlledSMC.forward`, `cloud_weights`, `fit` and
`GaussianMixtureIWSG.score_check`. The KDM reference extracts and executes the
unchanged author's `resample_particles` method with the real source KDE sampler
and density implementation. The TF primitive consumes fixed samples; a complete
TF sampling-to-optimizer call chain has not been implemented or checked.

No XLA compilation failure or retracing appeared in the completed numerical
checks: each checked signature traced once. The controlled-SMC time recursion
uses native TensorFlow loops; Python loops orchestrate refinements and reference
diagnostics outside the kernels. No runtime NumPy or pfor path was added.
The full fresh-stream run spent about 25.06 s in its two backward fits. Its
allocator peak was 183,593,472 bytes; full source replay peaked at 490,066,432
bytes. These observations identify work to profile; they establish no speedup.
They do not diagnose every historical iAPF/XLA failure.

## Failures and provenance corrections

Attempt01: missing Octave RNG wrapper, 0.214463 CPU s. Attempt02: missing JSON
export and a supervisor parse failure; conservatively recovered/charged 45 CPU s.
Attempt05: source integer-valued ancestry arrived as doubles, 4.381349 GPU s.
Attempt06 completed, but its comparison ingestion could round expected Python
lists to FP32; attempt08 reruns corrected FP64 comparisons. Attempt09 failed
at the external C++ ABI import, 2.880029 GPU s. It also has a wrong author_commit
metadata value; its KDM source hashes remain available, but it is not numerical
success evidence. Subsequent KDM manifests correct that field. Earlier GPU
manifests link summaries containing device/memory policy; new manifests also
copy those fields directly. Historical records are preserved, not rewritten.

## Decisions and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept scalar cSMC numerical parity for tested scopes | Source and analytic identities pass | No invalid accepted values, failed ancestry or deficient healthy fits | Multivariate/nonlinear adapters and untested source precision boundaries | Implement general adapters and the Figure 4 right comparator/cost contract | Full controlled-SMC paper replication |
| Accept Gaussian KDM/IWSG primitive parity | Actual-source values and gradients pass | Invalid-bandwidth rejection and stopped-gradient checks pass | Full sampling, sequential learning and optimizer wiring | Implement/validate the exact short-horizon mixture oracle and full batched consumer | A trained KDM filter or replicated learning curve |
| Keep empirical promotion closed | Required comparisons/repetitions absent | Missing evidence vetoes promotion | Runtime and source settings for remaining cells | Resolve each scoped contract and measure cost before claims | Superiority, defaults or HMC readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Numerical checks pass for the declared scalar/Gaussian scopes; earlier harness failures were repaired. General-route and empirical evidence remains absent. |
| Statistically supported ranking | None tested or supported. |
| Descriptive-only differences | Individual logZ, ESS, fit condition, runtime and allocator usage. |
| Default-readiness | Not established; source margins and epsilon conventions remain explicit comparison settings. |
| Next evidence needed | Frozen source experiment matrix, conditional normalized baselines, independent repetitions and declared uncertainty; full optimizer-to-filter validation for KDM. |

Engineering correctness, numerical validity and scientific interpretation remain
separate: executed call-chain checks support the first two only within their
recorded scopes. Missing full training is an implementation/evidence gap, not
evidence against IWSG. Deferred iAPF remains incomplete; controlled-SMC success
neither completes it nor demonstrates a flaw in its paper.

## Budget and continuation

Twelve attempts consumed 72.828086 numerical CPU s and 105.044156 GPU-charged s,
including failed attempts, imports, reference export and compilation. Inherited
remaining balance: 680.338850 numerical CPU s and 130846.256195 GPU s
(36.346182 h). No numerical job is running. The initial 300 CPU s / 7200 GPU s
tranche is complete at its reviewed twelve-attempt limit; this is a stage
boundary, not exhaustion of the owner-authorized campaign.

Next: specify/implement the Figure 4 right comparator/cost driver and general
model adapters; independently finish KDM source-setting discovery, mixture-Kalman
controls, normalized bearings comparisons and batch-native training. Ordinary
local implementation and bounded repair remain authorized. New empirical jobs
need the master's evidence contract and a budget allocation before launch.
At the recorded old training rate, ten additional MDPF trials alone need about
85.89 GPU h, exceeding the balance. A measured equivalent faster port or a
clearly partial empirical scope is necessary; the full campaign is not yet
shown to fit.

Post-run red-team: the strongest alternative explanation for apparent progress
is that tractable scalar/Gaussian cases conceal failures in multivariate or
learned sequential consumers. A failing full-consumer/source comparison would
overturn extrapolation to those routes. The weak point remains the unexecuted
learning/empirical scope, not the passed arithmetic identities. This was a
Codex source/code/result review; no independent reviewer was used.
