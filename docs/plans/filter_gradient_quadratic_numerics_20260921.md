# Quadratic fit and trust numerical qualification

Question: can the existing dense score fit, paired score fit and SPD trust
solver execute together under a stable XLA signature while preserving every
numerical field and rejection decision of original `3582b4ac`?

The dense fit remains unrestricted least squares followed by symmetrization.
The paired fit retains both axial widths, selects the smaller-width raw fit,
and uses the same antisymmetry, condition, generalized-eigenvalue, response and
covariance-error gates. Trust solving retains its scale normalization, KKT
equation, 100 bisections and exact boundary comparisons. No projection, new
regularizer, retuning, tolerance waiver, random-stream change or score
substitution is authorized. The original isolated Git dependency closure is
the numerical authority; `51bc753a` is the immediate execution baseline.

First compare raw current graph/XLA with the original graph on frozen float64
inputs at D=3/5. Cover non-diagonal quadratic and nonquadratic responses,
rank-deficient dense design, paired inconsistency and nonfinite rejection,
and interior/boundary trust solutions. All fields use atol=rtol=1e-10 and
all discrete decisions agree exactly, including NaN masks on rejection.
Independent least-squares and KKT residual checks prevent parity with a shared
defect from counting as mathematical evidence. Dense nonfinite inputs remain
outside the kernel contract: the enclosing controller checks them before the
call, because XLA cannot reproduce throwing graph assertions.

Repair only demonstrated execution/numerical blockers using established local
TensorFlow machinery. Verify changed runtime inputs retain one trace, all HLO
operands, and no PyFunc/pfor or Python numerical loops. Check affected full
public records and consumers on CPU/GPU before default integration. A compiled
dependency cannot close the outer fit-round/design-partition work, the separate
D5 lifecycle mismatch, actual DZ5 evidence or any terminal F01--F20 gate.

After numerical checks pass, compare original graph, current graph and current
XLA in fresh CPU/GPU processes at both dimensions. Use identical frozen inputs,
output lifetimes and 20 warm calls. Record build, trace, cold, warm and copy
time, RSS/VmHWM observations, TF allocator current/peak, graph/HLO size and
trace count. Investigate >20% warm regression, >2x device peak, >256 MiB host
increase or continuing allocation growth. These dependency costs are
descriptive, not complete initializer costs or terminal repeated evidence.

Use the existing absolute campaign driver, registered test groups, sequential
workers, GPU2 idle preflight and verified growth. CPU reference workers hide
GPUs. Each focused group is bounded by 300 seconds; each cost worker by 120
seconds. All runs consume the existing 52 GPU / 32 CPU process-hour caps and
write fresh numbered directories under the shared campaign artifact root.
Source, tests and driver stay frozen during each active worker. Stop on broken
source isolation, missing observations, uncontrolled growth or budget exhaustion.
Numerical failure rejects the candidate and triggers localized repair.

Skeptical pre-execution review: simply enabling JIT can silently weaken raw
SVD/eigensolver precision and erase assertion failures. The checks compare
all diagnostics and invalid paired decisions, with the dense finite-input
precondition explicit. A symmetry-constrained fit would change the algorithm
and is excluded. Source-pinned graph references avoid the intermediate faulty
eigensolver. Frozen clouds distinguish arithmetic changes from RNG changes.
Small-kernel speed cannot establish outer-controller speed; complete costs
remain a separate gate. Primary-agent review accepts this bounded investigation;
independent terminal review is still pending. No package/environment, external
source/pin, sampler/tuner or scientific admission change is included.

01880 finds two genuine issues: D5 trust multipliers/steps miss the fixed
tolerance, and singular dense-design condition numbers differ by factors of
12--107 between original graph and XLA. Rank and raw precision agree in these
fixtures, but the complete-field gate fails and remains mandatory. Two harness
errors also occurred: the independent KKT check mixed a TF matrix with a NumPy
vector, and HLO extraction did not bind the paired function's keyword defaults.
Use a NumPy matrix at the independent diagnostic boundary and a positional-only
test wrapper to fix those errors; no comparison tolerance changes.

Next use the already-qualified isolated refined eigenpair program inside the
XLA trust context. Ordinary graph references retain tf.linalg.eigh. Replace
the paired numerical finite-input Python loop by one tensor stack/reduction.
The dense singular parity tests remain in mandatory quadratic_numerics while
the independently qualified trust/paired subset can proceed to consumer/cost
checks. Do not enable the uniform-cloud XLA fitter while its all-field gate
fails. This is an implementation qualification failure, not evidence against
the quadratic method.

01881 CPU and 01882 GPU pass all 17 other checks after the trust repair and
harness corrections. Both singular-design condition fields remain failed;
their original and XLA records are retained. Expand the independently passing
paired checks to each nonfinite input, non-SPD/zero/excess-condition/repeated
spectra and extreme scales, plus the raw-precision pullback. These are direct
guards for enabling the paired fitted program; ordinary uniform-cloud fitting
stays unrepaired until its complete-field gate passes.

01883 CPU / 01884 GPU and the final factory check 01885 GPU pass all 35
paired/trust checks. Fresh-process costs 01886--01909 preserve every original
and changed-input field at unchanged tolerances. No declared ratio trigger
fires. GPU paired medians change from 3.200/3.256 ms to 1.237/1.431 ms at D3/5;
CPU from .765/.773 to .502/.477 ms. CPU XLA observed RSS increases 162/171 MiB
for paired fitting and 192/196 MiB for trust solving. GPU observed RSS and
allocator peaks decrease. Warm GPU live allocations are constant. Small
sampled host changes (0--12 KiB for XLA over twenty calls) do not prove general
allocation stability. Exact analysis is `/tmp/analyze_quadratic_numerics_costs_20260921.py`
and `quadratic-numerics-cost-comparison-01909.json` in the campaign root.

The trust comparison uses original graph, because original default trust XLA
has demonstrated precision errors; it does not measure pre-repair XLA speed.
All measurements are single-process dependency costs. After these gates, bind
the paired public fitter to its XLA factory and report `jit_compile_fit`.
Uniform-cloud fitting remains explicitly non-XLA debt. Compare complete
paired public histories at D3/5 against the original graph trust authority,
normalizing only execution metadata after asserting its values. Preserve all
numerical fields, strict decisions, target order, seeds and tolerances.

01910's six complete paired public cases add a previously missed spectrum:
nonquadratic paired widths produce a generalized matrix near the identity with
small off-diagonals. The raw XLA eigensolver misses 4--5e-9 in the smallest
generalized eigenvalue. All four quadratic public cases pass, but this vetoes
integration qualification and supersedes the paired cost candidate. Use the
same isolated refined eigenpair helper for that generalized matrix and rerun
complete records, focused checks and fresh matched costs; no control or gate
changes. This failure demonstrates why simple quadratic fixtures alone were
insufficient. Earlier measurements remain evidence of their exact source only.

01911 CPU / 01912 GPU pass all 41 focused and complete-public cases after
generalized eigenpair refinement. Before renewing costs, add the analogous
near-identity raw precision spectrum: the paired raw batched eigenvalue call
has the same backend stopping rule and must be checked as well. This directly
tests the unresolved risk exposed by 01910, not a broader test-budget expansion.

01913 confirms raw eigenvalue errors of 6--7e-9 on near-identity precision.
Refine both fixed-width raw spectra with the existing isolated helper; this
uses two named tensor calls, no Python numerical loop or sample mapping.
Graph references, raw matrices, scale normalization and every gate stay fixed.

01914 CPU / 01915 GPU pass all 43 expanded focused/public cases. 01916--01921
pass 51 quadratic-center, 38 paired-pilot and 25 batch-history checks on each
device; 01922 passes all 68 policy/controller checks. Guard coverage is partial,
201 sources / 1281 exact exceptions. The four added exceptions validate fixed
steps and input schemas, never numerical array recurrence.

Renewed cost processes 01923--01946 match every initial and changed-input field
and preserve one trace. XLA paired graph size is 634 nodes at both D3/5. The
final D5 GPU baseline process is faster than its earlier measurement: 2.222 ms
versus candidate 2.961 ms, triggering the declared >20% investigation. Preserve
that result; two additional fresh processes per before/graph/XLA D5 GPU arm
will provide three matched process observations including this first run.
Use `test --group quadratic_numerics_memory_<arm>_paired_5_gpu --repeat 1/2
--device GPU --test-gpu-index 2 --test-timeout-seconds 120` with the unchanged
source, inputs, limits and budget. These are descriptive repeats, not a
statistically supported ranking. Do not discard the faster baseline.

The final comparison is `quadratic-numerics-cost-comparison-01946.json`,
produced by `/tmp/analyze_quadratic_numerics_costs_final_20260921.py`.
The repeat analysis is `quadratic-numerics-repeat-comparison-01953.json`,
produced by `/tmp/analyze_quadratic_numerics_repeats_20260921.py`. All numerical
source hashes, fixture hashes and output records agree across their matched
arms; initial and changed inputs pass unchanged tolerances. The original graph
is the trust precision authority, not a pre-repair default-XLA speed baseline.

| Device / kernel / D | Original warm ms | Current graph warm ms | XLA warm ms | XLA minus original observed RSS MiB | Original / XLA device peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU / paired / 3 | .744 | .729 | .541 | 223.6 | n/a |
| CPU / paired / 5 | .695 | .811 | .552 | 231.3 | n/a |
| CPU / trust / 3 | 4.112 | 4.267 | .408 | 193.2 | n/a |
| CPU / trust / 5 | 3.555 | 4.211 | .432 | 194.5 | n/a |
| GPU2 / paired / 3 | 3.240 | 3.122 | 1.760 | 18.5 | 1097728 / 19712 |
| GPU2 / paired / 5, first process | 2.222 | 3.094 | 2.961 | 19.4 | 1125632 / 22272 |
| GPU2 / trust / 3 | 24.790 | 25.786 | 2.687 | -16.2 | 549376 / 10240 |
| GPU2 / trust / 5 | 26.933 | 24.916 | 3.081 | -17.0 | 562176 / 11008 |

D5 GPU paired process medians over three repeats are original 3.123 ms
(range 2.222--3.390), graph 2.962 (2.170--3.094), XLA 2.961
(2.918--3.217). No repeat-median >20% trigger persists; ranges overlap.
This is a descriptive disposition, not statistical superiority. The faster
first baseline stays in the comparison. Final XLA paired graph size is 634
nodes at both dimensions, trust 487. Each has one trace and unchanged HLO
across changed runtime inputs. Cold XLA calls and tracing are materially
costlier: paired first calls are 1.22/1.54 seconds on CPU and 2.49/2.83 on GPU,
versus original graph .055/.056 and 1.22/1.20 seconds respectively.

XLA warm GPU allocator current is constant in these twenty calls. Observed
host changes are 4--12 KiB; CPU paired XLA costs 224--231 MiB extra RSS.
This fits the predeclared investigation boundary but is still a real compile
overhead, not free memory or a proof of allocation stability. The measurements
cannot establish exact RSS peaks, executable-cache eviction or complete
initializer costs. No cleanup, allocator policy or system limit was changed.

| Decision | Criterion | Veto | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain paired XLA and refined trust on repair branch | 382 focused/public/consumer/policy checks pass; final cost records match | No remaining demonstrated paired/trust numerical failure; initial timing trigger resolved descriptively | Larger dimensions, complete enclosing controller costs | Compile remaining probe/controller work and retain terminal repeats | Whole-repository repair, posterior or HMC readiness |
| Keep uniform dense fitter unrepaired | Full-field original parity | Singular design_condition fails at D3/5 on both devices | Backend roundoff in a singular-value ratio | Localize under original-field contract; preserve mandatory tests | Permission to omit a field or waive tolerance |
| Keep main unmerged | All master terminal gates required | D5 lifecycle, outer loops, external/DZ5 and terminal evidence still open | Complete source-frozen qualification | Continue within unchanged budget | Campaign completion |

Post-run primary-agent review: the complete nonquadratic record and near-identity
tests exposed real defects after simpler checks passed. Their repairs preserve
the same eigenproblems and analytical callback scores, using the existing
isolated spectral helper. The strongest alternative explanation for kernel
speed differences is process scheduling/host-launch variation, supported by
overlapping D5 ranges; no statistical speed claim follows. Low-dimensional
fixtures and independent-kernel costs remain the weakest evidence for a full
initializer. Independent terminal review is pending. Focused Ruff/whitespace
pass; the driver retains prior lint debt. No package, OS, external pin or
canonical LEDH rebuild change occurred. Through 01953, CPU charge is
42695.82916168675/115200 seconds and GPU 38888.51101641379/187200 seconds.
