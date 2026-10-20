# Controlled SMC/KDM gap-closure localization result — 10 October 2026

## Decision

**Localization stage complete; both replications remain incomplete.** The bounded
run was valid and fully accounted, but it was intentionally diagnostic and did
not qualify either method.

| Decision | Primary criterion | Vetoes | Uncertainty | Next action | Limit |
|---|---|---|---|---|---|
| cSMC fit/forward localization | Source controls and first-divergence trace | No tape/control/artifact veto | Not a Q2 qualification sample | Treat regression as the repair target; inspect prospective precision | No new cSMC qualification claim |
| KDM coupled replay | Source native/replay identity and FP64/FP32 localization | No harness/data/device veto | Four development cases, not population evidence | Assess whether FP32 drift is materially harmless downstream; design K1 only if budget permits | No training ranking |
| Allow-list execution | Exact command forms and negative arbitrary-Python check | None | Platform reload is not assumed | Reuse literal supervisor form in a fresh session if needed | Rules do not override managed permissions |

## Execution and provenance

Command executed:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter/docs/benchmarks/run_replication_gap_localization.py --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/gap-closure-20261010/localization-01
```

The supervisor ran at commit `89065bc6354801cb368d5e163ba49fd9c4372d10`
on branch `surrogate-hmc`. GPU preflight succeeded on
`GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`; TensorFlow memory growth was
verified, TF32 was disabled, and XLA was used. Charged wall time was 12.7435
CPU seconds and 126.6963 GPU seconds, below the phase caps of 1,200 and 3,600
seconds. The complete manifest and child logs are in
[localization-01](artifacts/controlled-smc-kdm-replication-20260928/gap-closure-20261010/localization-01/manifest.json). The exact pre-execution plan text whose hash is recorded in that manifest is preserved as [controlled-smc-kdm-gap-closure-execution-20261010-as-executed.md](controlled-smc-kdm-gap-closure-execution-20261010-as-executed.md); the active plan only corrects its allow-list wording after the run.

The project rule source and installed rule have four literal diagnostic
prefixes. The installer passed exact-prefix checks and rejected an arbitrary
`python -c` command. The installed rule is recorded in
[allowlist-install-20261010T042718](artifacts/controlled-smc-kdm-replication-20260928/gap-closure-20261010/allowlist-install-20261010T042718/result.json).
A future Codex session may need to reload rules; this session does not assume
hot reload.

## cSMC result

The frozen source fixture contains six cases and 22 stages. All source policy
and random-tape controls passed. The source forward controls in the GPU run
passed for all 16 fit stages, so the RK4 transition, Gaussian draws, source
cloud inputs, weights, ESS and ancestry path are valid before the fitted policy
is applied.

Both explicit TF solver arms produced finite fits at all 16 stages, but none of
the fitted policies passed the strict diagnostic comparison with the source
policy, and none of the resulting forward comparisons passed:

- source-normal-equation arm: 16/16 finite fits, 0/16 policy screens, 0/16
  downstream forward screens;
- centered-QR arm: 16/16 finite fits, 0/16 policy screens, 0/16 downstream
  forward screens;
- first divergence: fitted coefficients at the first fit stage (stage 1 in
  the first case), before sampling, resampling, or log-weight comparison.

This does **not** mean the whole cSMC port is invalid. The source trace already
showed 14 of 101 regression systems taking Octave's singular-matrix branch.
Octave's `A\\b` is conditional; the diagnostic TF
`source_normal_equations` arm is not a complete reproduction of that
branching behavior. The centered-QR arm has much smaller regression residuals
on the source trace, but ill-conditioned polynomial coefficients can differ
greatly while the fitted values remain the relevant downstream quantity. The
first-divergence artifact is
[summary.json](artifacts/controlled-smc-kdm-replication-20260928/gap-closure-20261010/localization-01/csmc/summary.json);
the original source conditioning trace is in
[c0-source-fit-20261010](artifacts/controlled-smc-kdm-replication-20260928/gap-closure-20261010/c0-source-fit-20261010/summary.json).

The cSMC log also emitted TensorFlow retracing warnings for the copied
`_forward` bound method. Per-engine trace counts in the summary were one, so
the warning does not by itself establish an unstable claim-bearing graph.
It is an engineering repair trigger: the next diagnostic must instrument the
actual callable wrappers and either reuse stable wrappers or explain the
multiple-wrapper warning before a new qualification run.

## KDM result

The source worker replayed the actual TF-trained development checkpoint for
MDPF and A-MDPF over four consecutive batches, retaining parameters and Adam
moments after every update. All 64 native/replayed source controls passed.

The TF consumer evaluated 3,616 checks:

- MDPF FP64: 820/820 passed;
- A-MDPF FP64: 988/988 passed;
- MDPF FP32: 47 failures, first at update 0 in
  `proposal_model.proposal_model.4.weight` raw and clipped gradients
  (maximum absolute difference (3.67\times10^{-5}), diagnostic bound
  (1.53\times10^{-5})); update 3 then shows broad gradient, norm and first
  moment drift;
- A-MDPF FP32: two update-1 gradient checks failed in the same proposal layer.

The exact source replay, data/mask/shape controls, shared realized samples and
proposal noise, FP64 path, GPU placement, and XLA execution all pass. The
evidence therefore supports float32 arithmetic/reduction sensitivity that
accumulates across coupled updates. It does not identify a wrong batch,
random tape, optimizer state, or derivative translation. The artifact is
[kdm/comparison.json](artifacts/controlled-smc-kdm-replication-20260928/gap-closure-20261010/localization-01/kdm/comparison.json);
the source reference is
[kdm/author.json](artifacts/controlled-smc-kdm-replication-20260928/gap-closure-20261010/localization-01/author.json).

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | No source-tape, finite-value, GPU-policy, or artifact veto |
| Statistically supported ranking | None; no population comparison was run |
| Descriptive differences | cSMC fit coefficients differ; KDM FP32 gradients drift; both are diagnostic |
| Default readiness | Not established |
| Evidence needed next | cSMC stable-wrapper/solver localization and prospective uncertainty design; KDM downstream non-harm or a predeclared three-unit matched training study |

The strongest alternative explanation is numerical ill-conditioning in cSMC and
ordinary float32 reduction/operation-order sensitivity in KDM. A result that
would overturn this interpretation is a mismatch in the FP64 source replay or
a source-valid/TF-invalid cSMC pair under the controlled forward path. Neither
occurred here. The weakest evidence is the strict coefficient/gradient screen:
it is useful for localization but is not the final replication criterion.

## What remains open

1. cSMC Q2 is still not qualified because the frozen run has inadequate
   variance/tail precision in most cells, even though it has zero extra TF
   invalids and likelihood intervals inside the margin in 39/40 grid cells.
2. cSMC’s source conditional solver behavior and the copied graph’s retracing
   warning need a focused repair diagnostic before another qualification run.
3. KDM FP32 coupled drift is localized but its downstream NLL/RMSE impact has
   not been measured under a predeclared multi-unit design. The present run is
   not a training qualification.
4. No author code has been repaired, and no Task 3/E0 improvement has been
   attempted.
