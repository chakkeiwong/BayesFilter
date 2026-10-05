# KDM auxiliary native repair result

September 26 continuation: 03917 identifies the exact baseline rejection.
The first reset's column TV residual is 0.0005588870472403515, exceeding the
unchanged 0.0001 gate. Both reset factorizations and all four atom/KDM kernel
validity flags pass. Canonical validity is false from that first step onward;
its `-inf`/zero result makes the atom-reconstruction comparison fail correctly.
This is insufficient Sinkhorn convergence, not established ill-conditioning.
03916 completed numerics but failed JSON serialization of a NumPy scalar bool;
03917 fixes only the diagnostic recorder. Both workers remain charged.

03918 preserves the 2/2 rejected fixture and qualifies the separately declared
8/8 control on identical seed/inputs/gates using frozen auxiliary source
`ab431169d`. Its canonical value is -11.01702701955782 and score
-4.788503556702084; atom reconstruction errors are exactly zero. KDM value is
-10.84526439568048; analytical score -4.770346601456209 matches finite difference
-4.770346599825004 at the unchanged `rtol=4e-4, atol=4e-5` gate.

03919 passes the new native auxiliary in CPU graph mode: every original public
field, changed observations, exact replay, finite differences, invalid-model
tensor rejection, one trace and no host callbacks. CPU XLA 03920 passes.
The new candidate's complete patch and test bytes were archived before execution
as `kdm-native-candidate-r1.patch` and `kdm-native-candidate-r1-test.py` under
the raw root. The shared reset loop is converted in the next candidate; 03923 passes the
complete FP64 auxiliary again on CPU XLA, and 03925 passes on GPU 2 with
verified memory growth. Ownership and both updated public controls pass in
03924, alongside the reset finite-difference/autodiff checks. The validity helper squeezes singleton batch axes: B=1 returns a scalar,
B=3 returns a vector. Old tests indexed the singleton as a vector; the first
repair overgeneralized the scalar case (03926). Corrected assertions preserve
both shapes and all four reset tests pass in 03927. The other 23 regressions
pass in 03926, and shared FP32/FP64 graph/XLA checks pass on GPU in 03928.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Limit |
| --- | --- | --- | --- | --- | --- |
| Preserve original rejection | Exact failed column-mass gate localized | Canonical validity false | No conditioning diagnosis established | Keep rejected-input regression | No valid-target equivalence claim for this fixture |
| Continue native candidate | CPU graph complete records and score pass | XLA/GPU/cost gates pending | Shared reset and ownership coverage | Complete bounded qualification | No LEDH/HMC/training admission |

03921/03922 preserve a separate FP32 cross-mode failure. Original and native
XLA reset outputs are bitwise equal; native versus original graph differs by
at most 0.124 tolerance units. Both XLA implementations miss the eager
comparison by up to 1.347 units of the unchanged `atol=rtol=1e-6` gate. The
loop conversion did not create this difference. The failing cross-mode gate
remains registered as mandatory, separate from same-mode implementation parity;
it cannot disappear through an explanatory-test exemption. FP64 checks pass.

Source guard coverage is now 246 files / 1354 exact allowances. The eight new
allowances cover fixed schema/configuration, completed trace presentation and
the tested enclosing-XLA step function. The numerical field-validity loop was
removed instead of exempted. No numerical recurrence, NumPy path or pfor waiver
was added. The historical `kdm_auxiliary_legacy_cpu` group is now mandatory and
checks both the valid and rejected public controls.

The 6.03-MiB archive `dz5-kdm-evidence-03924.tar.gz` and receipt
`dz5-kdm-verification-03924.json` preserve and verify all 140 listed raw files
from 03904--03924 plus candidate patches/test bytes. Original DZ5 snapshots
and environments are referenced through prior receipts, not duplicated. The
removed 03911 prototype was not archived; its hashes alone do not reconstruct
it. Failures remain evidence and charged work.

Earlier checkpoint follows; its proposed candidate was removed and is distinct
from the archived September 26 candidate above.

The proposed KDM native repair was not installed. Source audit confirms that
`bayesfilter/highdim/ledh_younis_kdm_tf.py::canonical_linear_gaussian_kdm_auxiliary`
still owns a Python time loop, performs eager `.numpy()` model-validity checks,
and creates its atom and KDM TensorFlow kernels inside that loop. Its reusable
mixture kernels already default to XLA, but that does not compile the enclosing
auxiliary.

The first candidate was removed before commit after its compatibility worker
failed the existing auxiliary validity gate. The restored baseline worker 03912
fails the same assertion (`result["valid"]` is false), so the failure cannot be
attributed to the candidate. No runtime bytes, numerical method, tolerances,
validity semantics, or external source were changed. The preserved baseline
failure required rejection localization before candidate comparison; 03917--03919
now provide that localization and the separately declared valid control.

The next valid repair must add a private owner/factory around the existing
analytical score's TensorArray trace, prebuild the two KDM kernels once per
owner, and use a `tf.while_loop` for accumulation. It must compare the complete
public step records, canonical no-feedback fields, changed operands, invalid
model behavior, finite differences, graph/XLA host-callback absence and costs.
The public wrapper may materialize records and raise completed model mismatch
errors at the host boundary; the numerical body may not use `.numpy()`. This
gap remains open and diagnostic-only; it does not authorize a canonical LEDH
rebuild, claim-bearing KDM route, HMC, or scientific promotion.

The 03911 candidate and 03912 restored-baseline logs are preserved under the
campaign raw root. The campaign classifies the baseline worker as explanatory,
so its expected failure cannot be mistaken for a passing repair.


The complete matched cohort 03929--03946 passes all numerical comparisons
(three independent processes per arm/device). Descriptive medians are:

| Device | Arm | Public cold s | Public repeat s | Retained call ms | Peak host MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | original | 4.947 | 4.635 | n/a | 1096 |
| CPU | graph | 4.344 | 4.052 | 16.389 | 903 |
| CPU | XLA | 7.948 | 7.565 | 1.769 | 2068 |
| GPU 2 | original | 9.589 | 6.738 | n/a | 1466 |
| GPU 2 | graph | 6.994 | 4.250 | 63.981 | 1435 |
| GPU 2 | XLA | 10.371 | 9.892 | 9.394 | 2108 |

The original is eager outer orchestration with XLA mixture components. Native
public calls deliberately rebuild an owner to preserve mutable callback closure
semantics; retained calls use one explicit owner. Peak host values for native
arms include three public owners followed by a retained owner; the original has
only its three public calls. Direct public-only RSS medians are CPU
1096/902/1849 MiB and GPU 1456/1426/1943 MiB (original/graph/XLA). Do not compare
those phases as if they used the same owner lifecycle. GPU allocator peaks are
1,158,144 / 2,640,640 / 94,464 bytes: the XLA overhead in this fixture is host
memory, not whole-device reservation. No statistical ranking or target-capacity
claim follows from these three-repeat descriptive costs.

Single-owner CPU diagnostics 03947/03948 keep exactly one trace over 20 exact
replays. The XLA worker is approximately 1203 MiB after compilation and stays
there through all replays; a public-call cohort reaches approximately 2068 MiB.
After the owner is deleted its FuncGraph remains reachable. Trace-only
retention attribution 03949 is the next discriminating diagnostic. Object
collection alone would not imply native executable eviction. The memory finding
remains open and is not accepted by the numerical checks.


Final component verification through 03956: the 30-test combined CPU group
passes, including independent analysis and six corruption-rejection cases.
The public enclosing GPU/XLA and empty-horizon checks pass (03955); policy passes
129 checks (03956). The zero-horizon failure in 03954 is preserved: XLA attempted
to compile an unreachable out-of-bounds slice; the static empty finite sum now
returns the frozen original's neutral record inside XLA. Positive-horizon
numerics and the measured cost cohort are unchanged.

The four single-owner workers show stable RSS/functions over 20 calls. GPU XLA
RSS is about 1353 MiB and allocator peak 76,288 bytes, versus graph 1301 MiB and
2,645,248 bytes. In the isolated workers the graph survives a first collection;
combined trace-only check 03953 releases it, and the registry inspector finds no
retaining gradient entry. This does not prove a permanent leak. The next
[bounded plan](filter_gradient_kdm_remaining_gaps_20260926.md) specifies both the
rounded-input FP64 precision reference and repeated-construction attribution.

The second verified archive `kdm-native-evidence-03956.tar.gz` is 6,531,431 bytes
and contains all 144 listed raw files/qualified-source supplements from the
03925--03956 cohort. Its receipt is `kdm-native-verification-03956.json`.
The unit consumed 736.573433 CPU / 492.431629 GPU process-seconds, including
failures, in 28 CPU / 15 GPU workers. No numerical worker remains active.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Limit |
| --- | --- | --- | --- | --- | --- |
| Retain native execution repair on branch | FP64 complete-record, score, replay, API and CPU/GPU XLA gates pass | Master merge still blocked | FP32 cross-mode and construction memory | Execute remaining-gaps plan | No master completion or canonical LEDH admission |
| Report measured costs | Three fresh processes per arm/device, numerical/provenance checks pass | XLA public-call memory/cold-cost findings remain open | Target-scale capacity and repeated ownership | Attribute before accepting performance | Descriptive tiny-fixture medians only |

Red-team review: component passing tests do not close the enclosing public LEDH
or target-capacity gaps. The original rejected fixture must stay rejected; a
valid control is a distinct finite program with explicitly changed iteration
counts. A single scalar ownership check cannot represent every batch shape,
and source hashes cannot reconstruct the discarded 03911 prototype. Those
limitations are preserved in the tests, result and evidence receipts. No user
reporting decision, tolerance, seeded numerical stream or LEDH admission rule
was inferred or relaxed.
