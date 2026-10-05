# Transitive GenUT execution and cost result

September 26 follow-up through 04140: the cap-report mismatch is now traced
to XLA's power/division rewrite on identical operands. Exact instrumentation,
80-digit Decimal and a diagnostic optimization barrier reproduce the mechanism
on CPU/GPU; the strict reporting gate remains open with no waiver. See
`filter_gradient_genut_cap_diagnostic_result_20260926.md`. Reduced-primal capacity
through 04134, FP64 attribution 04135--04137 and matched CPU timing 04139 are
reported in `filter_gradient_genut_capacity_result_20260926.md`. The original
result below retains its evidence scope and costs.

The reduced primal GenUT correction now uses TensorFlow control flow for its
diagonal and pairwise iterations. The Austria observation callback builds its
fixed infectious-coordinate matrix with one batched `tf.one_hot`. Fifteen
previously uncovered modules are now guarded: 268 sources, 1,422 exact
allowances, with all 129 policy tests passing in 04035. The 63 added allowances
cover 64 reviewed metadata/schema sites; no numerical recurrence was waived.
See the [construct audit](filter_gradient_transitive_import_audit_20260926.md).

The first loop conversion exposed an FP32 GPU compiler defect in the pairwise
projection: the GEMM autotuner aborts when it combines the two layouts in a
symmetric factor. Both transpose-only candidates failed. An explicit tensor
contraction compiles; the frozen original FP32 GPU XLA program independently
hits the same compiler abort. Its graph remains a labeled comparator, with no
original-XLA timing or equivalence claim for that device/dtype.

Fresh GPU diagnostics then exposed TF32 moment drift even with zero correction
iterations: mean error 4.26e-5 and covariance error 2.81e-3. A diagnostic
TF32-disabled arm reduces both to ordinary FP32 rounding. The final candidate
uses tensor products and reductions for moment, pairwise and final affine
contractions while keeping FP32 tensors and global TF32 enabled. It preserves
the mathematical algorithm, caps, ridges, controls and comparison bounds.
The final full-correction GPU mean/covariance errors are recorded in
`genut-transitive-summary-04037.json`; the unchanged 2e-5 moment gates pass.

Final numerical-source checks 04022--04027 pass CPU FP32/FP64, GPU FP32/FP64,
complete same-mode records where the original compiler is available, changed
operands, independent moment restoration, radial-cap bounds, scalar no-op,
zero-step cases, exact replay, one trace and enclosing HLO. The real streaming
reset consumer uses K=N=72 and both dtypes. Callback value/Jacobian/residual
checks pass. The reduced correction explicitly remains noncanonical; these
tests do not qualify the full LEDH rebuild, analytical score admission or HMC.

One strict complete-record comparison remains open. 04036/04037 fail on the
FP32 `fraction_coordinatewise_cap_active` report: graph gives 0.8935185075 and
XLA gives 0.8888888955, one coordinate out of 216. The predicate compares a
displacement with 1e-7. The frozen original CPU XLA has the same discrepancy,
and the final GPU numerical fields meet the original bounds; the report
discontinuity is retained as an unresolved gate, with no threshold/tolerance
waiver. The GPU original-XLA comparator remains unavailable separately.

Fresh-process FP32 costs, N=72,d=3, 20 exact replays per arm:

| Device | Arm | Cold s | Warm median ms | RSS after compile / replay, MiB |
| --- | --- | ---: | ---: | ---: |
| CPU | original graph | 0.626 | 4.142 | 619.9 / 619.9 |
| CPU | original XLA | 1.557 | 3.061 | 856.7 / 856.9 |
| CPU | repaired graph | 0.360 | 5.238 | 605.1 / 605.2 |
| CPU | repaired XLA | 0.720 | 3.135 | 802.5 / 802.5 |
| GPU 3 | original graph | 2.140 | 10.616 | 1124.6 / 1124.6 |
| GPU 3 | repaired graph | 1.886 | 10.586 | 1104.0 / 1104.0 |
| GPU 3 | repaired XLA | 1.844 | 3.917 | 1028.5 / 1028.5 |

These are single-cohort descriptive costs, including synchronized host result
materialization. Another process was present on GPU 3 in every GPU cost arm;
no uncontended GPU ranking is justified. CPU repaired XLA uses about 54 MiB
less host RSS than original XLA, while warm time is similar. It still uses
about 197 MiB more RSS than repaired graph. Twenty replays add less than
0.2 MiB RSS per arm; this does not test repeated constructor retention.
Native HLO is 292,727 bytes versus original CPU XLA's 573,799 bytes.

GPU allocator peaks are 18,944 bytes for original graph and repaired XLA,
and 25,600 bytes for repaired graph. The graph increase is consistent with
the new logical N*d*d product temporaries; this observation does not prove
which buffers cause the peak. Target-size scaling remains a required capacity
check. Tensor allocator bytes do not measure the CUDA context or all native
compiler allocations. Every GPU worker verified memory growth before device
initialization and recorded device, TF32 and managed-session trust provenance.

All attempts 04005--04037, including compiler aborts, the harness construction
error and failed numerical comparisons, are preserved with exact source
snapshots r1--r11. New tests/runner pass Ruff; the two touched runtime files
retain exactly nine pre-existing lint findings, checked against Git 072959c00.
Archive and byte-verification receipt are `genut-transitive-evidence-04037.tar.gz`
and `genut-transitive-verification-04037.json` in the campaign artifact root.

Review disposition: retain the execution/precision repair on the repair branch
with the explicit comparison and capacity limits above. It repairs the tested
numerical path; it does not close all F01--F20 findings or authorize main merge.
The next bounded work is the latent SIR simulation and mixed KR transport
closure, then the master program's remaining public-consumer and terminal gates.
The unit charged 214.734434 CPU / 231.462572 GPU seconds, leaving
32.488836 CPU / 30.624976 GPU hours under the existing 56/52-hour caps.
