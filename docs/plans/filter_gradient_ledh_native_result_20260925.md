# LEDH native execution repair result

The shared flow and seeded-input dependencies pass CPU/GPU qualification. The
new supplied-input value program is **not integrated** into the public wrapper:
the dual-cap/trust-region FP32 case still fails the unchanged1e-6 comparison.
The new explicit analytical-score owner passes independent derivative checks,
but existing public/consumer migration and endpoint costs remain incomplete.

The complete result/charge/archive receipt is
`artifacts/filter-gradient-repair-20260917/ledh-native-verification-03862.json`.
Raw03819--03862 attempts, failures, logs, HLO, source hashes and current changed
Python files are preserved in its checksum-bound tar archive. Earlier failed
uncommitted revisions have hashes, not a complete byte snapshot per revision;
do not claim their source bytes are all reconstructible. Reference code comes
from Git9d8202b77, after the LEDH historical-results invalidation cutoff.

SeedSequence/PCG64 integer state/raw words and uniforms are exact over the
predeclared seeds; native Philox Box--Muller matches old TensorFlow normal
realizations within1e-12FP64 /1e-5FP32. No seeded stream was migrated. The shared
flow uses native time control and pivoted-LU determinant arithmetic. Explicit
binary32 fraction rounding preserves Python tf.cast coefficients that GPU XLA
would otherwise evaluate with excess precision. CPU/GPU component groups each
pass38 checks (03855/03856); duplicate03857 is preserved and charged.

For the value program, one/composed/annealed stages, supplied-input changes,
replay, finite-state branches and native resampling pass their frozen-source
checks. Dual-trust CPU final ESS differs by2.53058928e-5, GPU value by1.68284015e-5.
03826 reproduces this discrepancy by compiling only the old FP32 reset;03827
finds FP64 eager/graph/XLA agreement below9.8e-15. The first old reset is already
invalid while its weaker program_valid is true. Its LM condition proxy is201;
this is not established severe ill-conditioning. Added reset-validity/condition
outputs expose the problem without changing the algorithm, precision or gate.

The initial score-owner prototype could freeze a mutable direction under lazy
tracing. It is replaced by a pure model-builder interface using explicit theta,
direction, initial states/covariances and their tangents, noises and observations.
03860CPU and03861GPU each pass8 checks against the frozen score executor and
independent five-point value differences at steps1e-3 and5e-4. Maximum absolute
errors are4.14e-12CPU and5.33e-12GPU. Directions including zero and a linear
combination, changed operands, exact replay, one trace and enclosing HLO pass.
The no-resetT>1 slice is only a derivative diagnostic. Contract-E checks here
also do not establish canonical admission. The builder must be pure; arbitrary
external mutable configuration is not automatically made safe.

The 18 accepted flow-cost workers03837--03854 use three fresh-process repeats,
N=32,d=o=2,24substeps,FP64,seeds13/37. GPU2 UUID
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba` is pinned with verified memory growth.
Initial03832--03836 are preserved;03836 exposes the uint64 GPU AddN failure,
repaired with signed intermediates without changing rounding. The independent
cost analyzer exactly reproduces the saved JSON and its provenance checks.

| Arm | Cold seconds | Warm milliseconds | Incremental sampled RSS MiB | GPU allocator peak bytes |
|---|---:|---:|---:|---:|
| CPU prior graph |1.02447|5.24361|52.44|unavailable|
| CPU native graph |0.23199|6.46707|25.68|unavailable|
| CPU native XLA |0.44496|0.70067|177.45|unavailable|
| GPU prior graph |1.20472|9.33138|135.94|92416|
| GPU native graph |2.57146|48.65483|153.95|35072|
| GPU native XLA |0.77304|2.95000|126.61|30976|

Graph nodes fall2628to359. Warm XLA/prior-graph ratios are0.134CPU/0.316GPU.
CPU XLA sampled RSS rises about125MiB relative to prior graph; cause, retention
and capacity remain unresolved. GPU graph cold/warm and CPU graph warm regressions
remain reported. These are dependency measurements, not endpoint speed rankings.

03862 renews all129 policy checks. The guard covers244 sources with1346 existing
allowances; only the new owner factory is covered in the mixed score module.
No numerical-loop, NumPy or non-XLA waiver was added. Ruff and whitespace checks
pass. This unit used21/24endpoint workers and23/24cost workers, within7200seconds
per unit. Supplemental120CPU seconds conservatively charge four unmetered
diagnostics; they are not admission evidence.

| Decision | Criterion/veto | Remaining uncertainty | Next action |
|---|---|---|---|
| Accept bounded flow/seed/score-owner repair | Focused CPU/GPU references, derivative and execution checks pass | Complete consumer wiring/costs not qualified | Continue consumer integration |
| Hold public value integration | Dual-trust raw-equivalence veto remains | FP32 reset backend sensitivity and invalid reset | Explicit valid/rejected reset qualification; no gate relaxation |
| Keep memory/cost findings open | CPU XLA RSS and non-default graph regressions observed | No compiler-memory cause or executable eviction proved | Isolated capacity/attribution checks |
| Keep main unmerged | F01--F20 terminal audit and actual initializer/DZ5 score evidence incomplete | Whole-repository completion | Fresh DZ5 oracle, initializer/supervisor and terminal repairs |

Review: the strongest misleading interpretation would transfer dependency passes
to complete filter or scientific admission. The frozen full-value failure prevents
that. The score test's independent finite differences reduce shared-derivative
risk but cover a small deterministic scope. A fresh complete endpoint comparison
or a validity failure at larger scale could overturn the bounded acceptance.
No canonical LEDH rebuild, NeuTra training, posterior or HMC claim is made.
