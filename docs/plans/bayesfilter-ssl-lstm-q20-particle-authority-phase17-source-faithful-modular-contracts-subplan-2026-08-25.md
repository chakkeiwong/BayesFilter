# Phase 17 Source-Faithful Modular Contract Audit

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_METHOD_IDENTITY_AUDIT_REPAIR_REQUIRED`  
Budget cap: `7200 s` within the unchanged `64800 s` campaign cap  
Input: Phase 8 audited N=300 bank, Phase 3 runner, and project-local paper copies  
Output root: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase17`

## Question

Do the current q=20 M1--M4 implementations compute the methods named in the
reviewed program, or only role-limited scaffolds? This phase audits method
identity before further performance experiments. It cannot promote an arm by
passing moment, target-status, or runtime checks.

## Source contracts

1. **Second-order ETPF.** Acevedo--de Wiljes--Reich equations (16), (20),
   (26), (42)--(44), and (48)--(57) require an LETF matrix `D`, first-order
   marginal constraints, a Sinkhorn/direct optimal-transport solution, and a
   second-order Riccati correction. Local source:
   `.localresources/papers/ledh_replay_solution_20260824/acevedo-dewiljes-reich-2017-second-order-etpf.txt:198-320,355-471`.
2. **GenUT.** Ebeigbe et al. define `2d+1` sigma points that match mean,
   covariance, and selected diagonal skewness and kurtosis, subject to the
   paper's feasibility/constraint qualifications. Local source:
   `.localresources/papers/ebeigbe-et-al-genut-2104.01958.txt:105-165,186-205`.
3. **Invertible LEDH-PFPF.** Li--Coates require the actual pre-flow proposal,
   repeated affine step maps, determinant product, post-flow target terms, and
   covariance lifecycle in the corrected weight. Local source:
   `.localresources/papers/ledh_replay_solution_20260824/li-coates-2017-particle-filtering-invertible-flow.txt:139-188,210-340`.
4. **Full second-order ET-PF.** This is not established by reusing an affine
   moment scaffold; it requires the declared filter transition/analysis route
   and its own reference comparison.

## Evidence contract

The audit binds exact source hashes and code hashes. It records, for each arm,
the claimed target, quantity actually computed, equality classification, and
missing operations. It also runs an actual-bank skewness diagnostic for M2:
the weighted cloud's marginal third moments are compared with the sigma rule's
third moments. This diagnostic proves or disproves selected-moment equality for
the measured cloud; it does not establish density fidelity.

Hard vetoes are missing/mismatched source files, unreadable authority tensors,
stale protocol/target hash, non-finite moment diagnostics, or an artifact that
labels a scaffold as source-faithful. Explanatory diagnostics include moment
residuals and support-range excursions. Performance ranking is out of scope.

## Skeptical audit and pre-mortem

- A code path can have the right output moments while being the wrong method.
  Method identity is checked from required operations and source anchors, not
  output covariance alone.
- A symmetric M2 cloud could appear adequate if the source cloud happens to
  have zero skewness. The receipt records the measured source third moments;
  an inconclusive near-zero case would not certify GenUT.
- Static pattern checks cannot prove implementation correctness. They can prove
  that named required operations are absent from the bounded runner and can
  prevent an overclaim; dynamic fixtures remain necessary after implementation.
- Existing repository LEDH/GenUT modules belong to other model/tuning scopes.
  Their existence cannot be silently transferred to SSL-LSTM q=20.

## Work and exact execution

1. Add a TensorFlow-only, CPU-hidden identity-audit runner and focused tests.
2. Run it on `phase6-attempt9-metadata-n300-seed2401`.
3. Repair the Phase 3 labels/status if the source identities are absent.
4. Refresh Phase 18 to the smallest source-faithful implementation slice; do
   not combine arms and do not launch HMC.

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_method_identity_audit_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase6-attempt9-metadata-n300-seed2401 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase17-attempt1
```

## Exit and refresh

If a named method is absent, record `wrong relative to the named method` while
preserving any explicitly defined scaffold result. This is an implementation
repair trigger, not a rejection of the scientific idea. Phase 18 may implement
only a bounded one-factor method with exact source equations and fixtures. A
real blocker occurs only under the master program's global definition.
