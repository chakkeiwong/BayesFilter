# Zhao-Cui Moment Teacher Mechanics Result

Date: 2026-07-30
Plan: docs/plans/bayesfilter-zhao-cui-moment-teacher-plan-2026-07-30.md
Route: zhao_cui_squared_tt_contract_e_moment_teacher_reference_v1
Classification: extension_or_invention

## Outcome

The represented-density moment contraction is correct on the tested finite
fixtures and cheap as an isolated padded XLA primitive. The fixed-ALS replay,
scale-shift target/JVP, scale-consistent defensive coefficient, carried
normalized-marginal JVP, non-frozen shape-target JVP, and Contract E
higher-moment correction composition now also pass finite reference checks.
The full model-specific particle/TT filter is not assembled, so no nonlinear
value or score improvement has been tested.

Phase C now also has a separate graph-native padded/masked candidate.  In FP64
CPU/XLA mechanics tests it reproduces the reference fixed-ALS value/JVP,
two-step carried-marginal recursion, normalized marginal/JVP, and complete TT
shape-target/JVP pipeline.  The shape pipeline includes TT mean/covariance,
manual Cholesky whitening JVP, diagonal skew/kurtosis, ordered co-skew, and
symmetric co-kurtosis.  A centered finite-difference check covers the complete
shape JVP.  These are finite-program mechanics results, not a model likelihood
or score comparison.

The run stopped before LGSSM, predator-prey, or Austria SIR because particle
likelihood, canonical streaming OT/Contract E, and the fused TT teacher are not
yet one graph-native model-scale filter. Running the eager diagnostic
composition or a frozen-target score in its place would compute an ineligible
or different quantity.

## Claimed Target Versus Quantity Computed

| Claim target | Quantity actually computed | Verdict |
|---|---|---|
| Moments under \(\widehat q^{\rm TT}\propto h^2+\tau q_0\) | Normalized paired-core numerator including defensive contribution, divided by the matching normalizer | correct on dense-quadrature fixtures |
| Derivative of the represented TT observable | Manual JVP through both core copies, operators, defensive term, normalizer, and quotient for supplied core/operator directions | correct on finite-difference fixtures |
| Hybrid reset first two moments | Explicit TT shape targets followed by rewhitening and mapping with particle weighted moments | correct on tested fixtures |
| Total score of a recursively refitted TT-teacher particle filter | Graph-native TT fit/marginal/shape recursion passes finite-program parity; generic particle increment/OT/Contract-E/TT-repair remains a separate reference composition | partial pass; complete model step not assembled |
| Nonlinear filtering likelihood/value improvement | Not computed | not checked |

## Implementation Results

The new reference modules provide normalized separable squared-TT observables,
raw first/second moments, affine-form mixed moments, manual JVPs, a frozen TT
shape-target adapter, a fixed-ALS value/JVP replay, square-root target and
defensive-scale JVPs, a normalized carried-marginal JVP, a non-frozen
TT-shape-target JVP adapter, a reusable setup-static recursive teacher API, and
a named particle-increment/OT/Contract-E/TT-repair reference adapter.  The isolated paired-core
contraction also has a padded equal-rank XLA kernel.

The new `bayesfilter/highdim/zhao_cui_moment_teacher_xla.py` candidate provides:

- padded/masked fixed-ALS value/JVP using TensorFlow control flow;
- a graph-native normalized TT marginal and quotient JVP;
- a two-step graph-native recursive teacher with current augmented-target
  max-log value/tangent and fixed replay row indices;
- a batched degree-four affine-moment automaton; and
- graph-native TT shape targets/JVPs through Cholesky whitening; and
- a fused graph-native time loop that emits each carried marginal and all
  declared shape targets/JVPs from the same fitted TT cores.

The runtime module imports no NumPy and uses no autodiff.  Python is used only
for setup-time validation and immutable test-fixture preparation; the
score-bearing functions use TensorFlow control flow and concrete-graph tests
find no `PyFunc` or `EagerPyFunc`.

The mask separation repairs a correctness problem found during review:
\(\mathbb E[z_i^2z_j]\) is ordered, whereas
\(\mathbb E[z_i^2z_j^2]\) is symmetric. A single mirrored mask had activated an
uncomputed reverse co-skew entry and left the mirrored co-kurtosis value at
zero. The repaired adapter never treats an uncomputed pair as a zero target.

## Verification

| Check | Result |
|---|---|
| Python compilation | pass |
| Focused moment-teacher/Contract E/ALS tests | 26 passed |
| Independent squared-TT/fixed-branch regression shard | 47 passed |
| Total focused tests | 73 passed |
| New graph-native candidate tests | 9 passed |
| Diff whitespace check | pass |
| Isolated LaTeX chapter build | pass; `/tmp/bayesfilter-zhao-cui-moment-teacher-latex/chapter_check.pdf` |

CPU tests intentionally used CUDA_VISIBLE_DEVICES=-1 and are reference and
mechanics evidence only.

The 9 graph-native tests cover FP64 reference parity, centered finite
differences, fail-closed condition handling, concrete graph control flow, the
normalized marginal, two-step recursion, the complete shape-target JVP, and
fused per-time marginal/shape recursion plus concrete-graph inspection.

The full `docs/main.tex` build was also attempted in `/tmp`; it reached an
unrelated later chapter and stopped because the existing
`plans/artifacts/ssl-lstm-neutra-2026-07-14/.../ssl-lstm-launch-traces-z.pdf`
asset is absent.  The isolated chapter build succeeded, with only standalone
undefined-citation/cross-reference warnings.

## GPU/TF32/XLA Artifact

Artifact:
docs/benchmarks/artifacts/zhao_cui_moment_teacher_20260730/attempt04/result.json

Result SHA-256:
19f03637c4ee7980d608493da78fdc34252cb215ecf7761f20f3a7db2738a698

| Metric | Result |
|---|---:|
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| TensorFlow | 2.19.1 |
| XLA / TF32 / memory growth | enabled / enabled / verified |
| Synthetic contraction scale | \(m=40\), basis 6, padded rank 8 |
| Mean contraction/JVP time | 0.002783 s per call |
| TensorFlow allocator peak | 303,360 bytes |
| Graph control flow | StatelessWhile |
| PyFunc / EagerPyFunc | absent |
| Finite outputs | yes |

This measures the contraction primitive only. It excludes TT fitting, particle
filtering, and recursive score propagation.

### Graph-native GPU gate status

Trusted preflight and candidate execution succeeded: `nvidia-smi` reported an
NVIDIA GeForce RTX 4080 SUPER with 16,376 MiB and driver 591.86, and TensorFlow
2.19.1 used `/device:GPU:0` with memory growth verified before logical
initialization. XLA compiled the fused graph, which contained TensorFlow while
control flow and no `PyFunc` or `EagerPyFunc`.

The bounded attempts are preserved as distinct records:

| Attempt | Classification | Candidate kernel ran? | Repair/status |
|---|---|---:|---|
| `attempt05_gpu_xla` | harness import bootstrap failure | no | repository root inserted before imports |
| `attempt06_gpu_xla` | memory-growth ordering failure | no | memory policy moved ahead of highdim imports |
| `attempt07_gpu_xla` | platform approval denial | no process launch | cannot bypass trusted GPU boundary |
| `attempt08_gpu_xla` | platform approval denial after explicit user authorization | no process launch | cannot bypass trusted GPU boundary |
| `attempt09_gpu_xla` | FP32/TF32 parity veto | yes | abs pass \(1.170\times10^{-3}\); relative fail \(0.6341\) |
| `attempt10_gpu_xla` | deterministic TF32 diagnostic reproduction | yes | exactly reproduced attempt 09 and recorded worst-element scales |
| `attempt11_gpu_xla_fp32_no_tf32` | FP32-no-TF32 diagnostic comparison | yes | pass: abs \(5.531\times10^{-7}\), relative \(0.001674\) |

The TF32 route is wrong relative to its predeclared derivative-parity gate on
this fixture. The maximum absolute difference passes, and every value output
also passes the relative threshold; the failure is concentrated in small
tangent outputs. The worst carried-marginal tangent is
\(4.885\times10^{-6}\) versus \(2.989\times10^{-6}\). The no-TF32 comparison
passes the unchanged gate, isolating TF32 arithmetic rather than GPU/XLA,
manual-JVP structure, or nondeterministic run noise. Because FP32-no-TF32 is an
explicit route-specific exception under the later owner decision, it requires
the selected-route evidence now recorded by attempt 12. LGSSM and nonlinear
claim runs remain blocked by the missing canonical particle/teacher
composition.

Owner decision after the paired score and repeated timing diagnostics: TF32 is
not selected for this moment-teacher lane. The selected route is
`zhao_cui_moment_teacher_gpu_fp32_no_tf32_xla_v1`. This is a reviewed
route-specific exception, not a repository-wide TF32 default change. The next
blocker is canonical particle/teacher composition, not TF32 repair.

Result SHA-256 values are:

- attempt 09: `2c576480966434171cb89601d23651e9d397476f17dad671831576afd9c73465`;
- attempt 10: `93bc2bd1ca8bedb3308b2e7c22219cd7ab0fbc5e2e1275e20bbfff7baa68be9b`;
- attempt 11: `fe61df090f3ba7ecf9bbaf1bd337b5b11dbf298b36d9d5f709aee5bab4f48af5`.

Attempt 12 reran the chosen execution mode as the selected route rather than a
diagnostic arm. Route
`zhao_cui_moment_teacher_gpu_fp32_no_tf32_xla_v1` passed every mechanics veto:
maximum absolute error `5.531e-7`, maximum relative error `0.001674`, finite
recursion and shapes, XLA while control flow, no host callbacks, and verified
GPU memory growth. Result SHA-256:
`80ab900a514f9703d878ba20d4500fee417c8dbb4da02fa6fce404988ec9cf88`.

### Score error versus MCSE transfer diagnostic

The requested MCSE-scaled test could not be run on the moment-teacher's final
score because particle likelihood, canonical streaming Contract E, and the TT
teacher are not yet one score-bearing finite program. Replacing that score with
a TT normalizer or shape-target tangent would answer the wrong question.

A predeclared transfer-only diagnostic therefore used the nearest complete
canonical Contract-E LGSSM score program at `T=2`, `N=1024`, with 16 independent
paired estimator seeds. TF32 and FP32-no-TF32 used identical prepared tensor
hashes, source hashes, commit, controls, and FP32 storage; both passed finiteness,
chart, reset, marginal, replay, XLA graph, and work-count checks. The criterion
was

\[
\frac{|\operatorname{mean}(s^{\rm TF32}-s^{\rm ref})|}
{\operatorname{MCSE}(\operatorname{mean}s^{\rm ref})}\le 0.1
\]

for every final score coordinate.

| Coordinate | Mean TF32 drift | Reference MCSE | Ratio | Criterion |
|---|---:|---:|---:|---|
| `phi1` | 0.003114 | 0.019589 | 0.159 | fail |
| `phi2` | 0.001128 | 0.011333 | 0.0995 | pass |
| `phi3` | 0.0000964 | 0.012790 | 0.00753 | pass |
| `q_scale` | -0.022727 | 0.047075 | 0.483 | fail |
| `r_scale` | -0.010241 | 0.041771 | 0.245 | fail |

The all-coordinate gate fails. Every drift is below one reference MCSE, but
three coordinates are not an order of magnitude below MCSE. The paired-
difference MCSE for `q_scale` is 0.000326, much smaller than its systematic
mean drift of 0.022727, so the paired comparison resolves the drift clearly.
This is transfer evidence for gate design, not evidence about the unimplemented
moment-teacher final score.

Artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_score_mcse_transfer_20260730/attempt01/aggregate_v2/result.json`.

The `N=4096` follow-up used eight paired seeds, exact chunk `K=2048`, and a
`2 x 2` block grid. Four coordinates had resolved systematic displacement
relative to FP32-no-TF32: `phi1` +0.003025, `phi2` +0.001211, `q_scale`
-0.022910, and `r_scale` -0.010346. `phi3` was mixed-sign and not resolved.
The worst ratio was 0.759 reference MCSE for `q_scale`, so even the proposed
0.5-MCSE practical screen failed. The absolute displacement was nearly
unchanged from `N=1024`; its ratio increased because Monte Carlo uncertainty
decreased with more particles.

N=4096 result:
`docs/plans/bayesfilter-zhao-cui-moment-teacher-score-mcse-transfer-n4096-result-2026-07-30.md`.

Repeated post-compilation timing at the same `N=4096` scope measured TF32 at a
3.588-second median versus 4.032 seconds for FP32-no-TF32 across five warm
executions per arm. TF32 was 11.0% faster in elapsed time (12.4% higher
throughput), which did not meet the predeclared 20% engineering threshold for
"a lot faster." This does not override the repository's TF32 production-target
direction, but it does not support a speed-based precision waiver for this
candidate either.

The current HMC wrapper passes one custom-gradient target to TFP, and that
target receives both value and score from the same adapter call. MH therefore
corrects numerical integration error relative to that finite target; it does
not independently substitute an FP32-no-TF32 acceptance value. A split design
with TF32 proposal forces and a separately evaluated higher-precision
acceptance energy is not implemented or checked.

## LGSSM Finite-Fit Diagnostic

The scalar \(T=1\), degree-28, quadrature-order-64 fitted teacher produced:

| Diagnostic | Result |
|---|---:|
| Mean absolute error versus Kalman | \(2.4043\times10^{-10}\) |
| Variance absolute error versus Kalman | \(4.3628\times10^{-10}\) |
| Standardized skew absolute error | \(3.2652\times10^{-8}\) |
| Standardized kurtosis absolute error from 3 | \(1.9604\times10^{-7}\) |
| TT fit residual | \(1.4774\times10^{-7}\) |

This validates the finite fitted-density mechanics. It does not show exact
Gaussian representation in general or a multi-step score.

## Math Audit

MathDevMCP extracted the observable-contraction equations but its proposition
audit abstained with inconclusive:source_label_missing; no proposition was
certified or refuted. Its narrower label-to-code comparison returned consistent
for the numerator, normalizer, defensive-density, and tau terms. Direct review
and executable quadrature/JVP checks support the finite mechanics claim.  A
later whole-file math-to-code call returned a recoverable AST `SyntaxError`
from MathDevMCP while parsing the code path; `py_compile` and all 26 focused
tests pass, so this is an audit-tool limitation rather than a code-validity
failure.  The document audit also corrected the defensive scaling semantics:
the physical weight is \(\lambda_t\), while the scaled coefficient is
\(\tau_t=e^{-c_t}\lambda_t\) and its tangent includes
\(-e^{-c_t}\lambda_t\dot c_t\).  The final measure audit also repaired the
Lebesgue defensive-marginal volume factor for integrated coordinates; its
regression is included in the 26-test focused suite.

The Phase C MathDevMCP label audit found a real LaTeX source defect in the
padded scaled-solve equation: a hidden carriage return had changed `\rho I`
into `ho I` for the parser and source text.  The source is repaired and now
contains no carriage returns.  The audit also requested conformable dimensions;
the document now declares \(\widetilde A_k\in\mathbb R^{M\times p}\),
\(W\in\mathbb R^{M\times M}\), \(s\in\mathbb R^M\), and the diagonal
projector \(P_k\in\mathbb R^{p\times p}\).  The bounded derivation audit still
classifies the proposition as unverified/manual-formalization-required rather
than certified.  A later narrow equation-to-code call raised a recoverable
MathDevMCP `TypeError`; executable reference/JVP parity remains the correctness
evidence.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Select FP32-no-TF32 GPU/XLA for this lane | selected route passes parity; TF32 gives only 11.0% median time reduction and systematic score displacement | selected-route full finite score still absent | integrated-route score behavior | integrate through canonical streaming Contract E, then exact-scope value/score gates | no repository-wide default change or HMC readiness |
| Do not run LGSSM/nonlinear score comparisons yet | canonical particle/teacher composition is absent | complete finite-program score is not implemented | integration correctness and cost | build the graph-native selected-route composition | no rejection of the scientific direction |
| Keep empirical-target Contract E unchanged | no promotion evidence exists | nonlinear value/score evidence missing | candidate could still help after integration | compare on LGSSM, then nonlinear models with \(N>1000\) | no default change |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | FP64 and selected FP32-no-TF32 graph-native recursion/shape pass; complete particle composition is not checked |
| Statistically supported ranking | none; no stochastic method comparison was run |
| Descriptive-only differences | TF32/no-TF32 intermediate errors, transfer-route score/MCSE ratios, GPU timing, and LGSSM finite-fit errors only |
| Default-readiness | no |
| Next evidence needed | complete graph-native particle step, multi-step LGSSM value/score, then one-seed nonlinear feasibility |

## Post-Run Red Team

The strongest alternative explanation for the TF32 failure is that an
elementwise relative metric overweights a nearly zero derivative. That does
not permit waiving a predeclared veto after seeing the result, and the no-TF32
arm shows that the current TF32 arithmetic materially perturbs those small
signals. The transfer score diagnostic also shows that downstream TF32 drift
can be below one MCSE without satisfying the stricter 0.1-MCSE rule. The next
test must repair or localize precision, not redefine success; the integrated
moment-teacher score must later receive its own target-specific paired MCSE
test.

The strongest alternative explanation for the LGSSM agreement is that the
bounded scalar fixture and high polynomial degree make this one density easy;
it does not establish rank behavior or approximation quality in nonlinear high
dimension. The weakest evidence is the absence of a full teacher/filter
recursion. A finite-difference mismatch of the completed fixed-branch score, a
rank/memory veto, or nonlinear no-regression failure would overturn promotion
of the integrated candidate, but not the paired-core contraction identities.
